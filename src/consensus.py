"""
Merge + state expansion + performance metrics.

This module:

  1. Merges primary and secondary entries (secondary wins on same-day conflict).
  2. Expands entry events into a persistent active state per day: BUY, SELL,
     or None. Positions persist until PSAR flips against the trade.
  3. Computes per-ticker performance metrics over the last 52 trading weeks:
     - trade_count
     - avg_hold_days     (calendar days entry-to-exit, mean)
     - avg_gain_pct      (realized PnL at PSAR exit, signed)
     - avg_peak_gain_pct (Maximum Favorable Excursion, floored at 0%)
     - success_rate      (kept for internal use; not displayed in app)
  4. Classifies current position strength based on close-to-PSAR distance.

All of this is pure data transformation.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ===== Merge primary + secondary entries ====================================

def merge_entries(primary: dict[str, str],
                  secondary: dict[str, str]) -> dict[str, tuple[str, str]]:
    """
    Merge two signal dicts into one, keyed by date.

    Values are (side, source). On same-day conflict, secondary wins.
    """
    merged: dict[str, tuple[str, str]] = {}
    for date, side in primary.items():
        merged[date] = (side, "primary")
    for date, side in secondary.items():
        merged[date] = (side, "secondary")
    return merged


# ===== Expand entries to persistent BUY/SELL/None state =====================

def expand_signals_to_active_status(
    df_asc: pd.DataFrame,
    entry_signals: dict[str, tuple[str, str]],
) -> list[dict]:
    """
    Walk ASC-ordered bars. Each bar is in one of three states:
      - In a BUY position
      - In a SELL position
      - In cash (None)

    Transitions:
      - Entry day: state := entry signal's side
      - In BUY and close < PSAR: exit to cash
      - In SELL and close > PSAR: exit to cash
      - Otherwise: hold

    Returns one dict per bar with all the data trade reconstruction needs:
      {date, close, high, low, psar, signal, signal_source, signal_started_date}

    high/low are included so peak-gain (MFE) can be computed without re-walking
    the source frame.
    """
    from .strategies import _date_to_str  # local import to avoid cycles
    out: list[dict] = []

    state: Optional[str] = None
    source: Optional[str] = None
    started: Optional[str] = None

    for _, row in df_asc.iterrows():
        date = _date_to_str(row.get("date"))
        if date is None:
            continue

        close = row.get("close")
        high = row.get("high")
        low = row.get("low")
        psar = row.get("psar")
        close_v = float(close) if pd.notna(close) else None
        high_v = float(high) if pd.notna(high) else None
        low_v = float(low) if pd.notna(low) else None
        psar_v = float(psar) if pd.notna(psar) else None

        if date in entry_signals:
            side, src = entry_signals[date]
            if state != side:
                state = side
                source = src
                started = date
        elif state == "BUY" and close_v is not None and psar_v is not None and close_v < psar_v:
            state = None
            source = None
            started = None
        elif state == "SELL" and close_v is not None and psar_v is not None and close_v > psar_v:
            state = None
            source = None
            started = None

        out.append({
            "date": date,
            "close": close_v,
            "high": high_v,
            "low": low_v,
            "psar": psar_v,
            "signal": state,
            "signal_source": source,
            "signal_started_date": started,
        })

    return out


# ===== Trade reconstruction =================================================

def _extract_trades(active_status: list[dict]) -> list[dict]:
    """
    Walk the per-bar active status and extract trades.

    A trade runs from the day of entry to the day of exit (when signal flips
    or exits to cash). Per-trade metrics computed:
      - pnl_pct        : realized PnL at exit, signed (snapshot for active)
      - peak_gain_pct  : Maximum Favorable Excursion using intraday high (BUY)
                         or intraday low (SELL). Floored at 0%.
      - hold_days      : calendar days between entry and exit dates
      - profitable     : pnl_pct > 0
      - is_active      : True if this is an in-progress trade, False if closed

    The CURRENTLY-OPEN position (if any) is included as the final trade with
    is_active=True. Its exit_date and exit_close are set to today's bar (the
    last in active_status), so peak_gain_pct reflects the best moment so far
    and hold_days reflects days-in-trade so far. The pnl_pct field is a
    snapshot of unrealized PnL — useful for debugging but mixes apples and
    oranges with closed trades' realized PnL.
    """
    trades: list[dict] = []
    prev_side: Optional[str] = None
    prev_source: Optional[str] = None
    entry_date: Optional[str] = None
    entry_close: Optional[float] = None
    peak_price: Optional[float] = None  # max high for BUY, min low for SELL

    for bar in active_status:
        side = bar["signal"]
        close = bar["close"]
        high = bar.get("high")
        low = bar.get("low")

        # Update running peak tracker BEFORE processing transitions, so the
        # exit bar's intraday extreme is included in the trade's MFE.
        if prev_side == "BUY" and high is not None:
            if peak_price is None or high > peak_price:
                peak_price = high
        elif prev_side == "SELL" and low is not None:
            if peak_price is None or low < peak_price:
                peak_price = low

        if side != prev_side:
            # Close out previous position.
            if (prev_side in ("BUY", "SELL")
                    and entry_date is not None
                    and entry_close is not None
                    and close is not None):

                # Realized PnL.
                if prev_side == "BUY":
                    pnl_pct = (close - entry_close) / entry_close * 100.0
                    profitable = close > entry_close
                else:
                    pnl_pct = (entry_close - close) / entry_close * 100.0
                    profitable = close < entry_close

                # Peak gain (MFE).
                if peak_price is not None:
                    if prev_side == "BUY":
                        peak_gain_pct = (peak_price - entry_close) / entry_close * 100.0
                    else:
                        peak_gain_pct = (entry_close - peak_price) / entry_close * 100.0
                    peak_gain_pct = max(0.0, peak_gain_pct)
                else:
                    peak_gain_pct = 0.0

                # Hold time in calendar days.
                try:
                    hold_days = (pd.Timestamp(bar["date"]) - pd.Timestamp(entry_date)).days
                    if hold_days < 0:
                        hold_days = 0
                except Exception:
                    hold_days = 0

                trades.append({
                    "entry_date": entry_date,
                    "entry_close": entry_close,
                    "exit_date": bar["date"],
                    "exit_close": close,
                    "side": prev_side,
                    "source": prev_source or "primary",
                    "profitable": bool(profitable),
                    "pnl_pct": round(pnl_pct, 4),
                    "peak_gain_pct": round(peak_gain_pct, 4),
                    "hold_days": int(hold_days),
                    "is_active": False,
                })

            # Open new position (or transition to cash).
            if side in ("BUY", "SELL") and close is not None:
                entry_date = bar["date"]
                entry_close = close
                prev_source = bar.get("signal_source")
                # Initialize peak with today's intraday extreme.
                if side == "BUY":
                    peak_price = high if high is not None else close
                else:
                    peak_price = low if low is not None else close
            else:
                entry_date = None
                entry_close = None
                prev_source = None
                peak_price = None

            prev_side = side

    # If we exit the loop while still holding a position, synthesize a final
    # "active" trade snapshot using the last bar's data. This includes the
    # current open position in the metrics (peak gain so far, days-in-trade
    # so far) rather than excluding it as if it didn't exist.
    if (prev_side in ("BUY", "SELL")
            and entry_date is not None
            and entry_close is not None
            and active_status):

        last_bar = active_status[-1]
        last_close = last_bar.get("close")
        last_high = last_bar.get("high")
        last_low = last_bar.get("low")

        # Final peak update for the last bar's intraday extreme.
        if prev_side == "BUY" and last_high is not None:
            if peak_price is None or last_high > peak_price:
                peak_price = last_high
        elif prev_side == "SELL" and last_low is not None:
            if peak_price is None or last_low < peak_price:
                peak_price = last_low

        if last_close is not None:
            # Snapshot unrealized PnL.
            if prev_side == "BUY":
                pnl_pct = (last_close - entry_close) / entry_close * 100.0
                profitable = last_close > entry_close
            else:
                pnl_pct = (entry_close - last_close) / entry_close * 100.0
                profitable = last_close < entry_close

            # Peak gain so far.
            if peak_price is not None:
                if prev_side == "BUY":
                    peak_gain_pct = (peak_price - entry_close) / entry_close * 100.0
                else:
                    peak_gain_pct = (entry_close - peak_price) / entry_close * 100.0
                peak_gain_pct = max(0.0, peak_gain_pct)
            else:
                peak_gain_pct = 0.0

            # Days in trade so far.
            try:
                hold_days = (pd.Timestamp(last_bar["date"]) - pd.Timestamp(entry_date)).days
                if hold_days < 0:
                    hold_days = 0
            except Exception:
                hold_days = 0

            trades.append({
                "entry_date": entry_date,
                "entry_close": entry_close,
                "exit_date": last_bar["date"],
                "exit_close": last_close,
                "side": prev_side,
                "source": prev_source or "primary",
                "profitable": bool(profitable),
                "pnl_pct": round(pnl_pct, 4),
                "peak_gain_pct": round(peak_gain_pct, 4),
                "hold_days": int(hold_days),
                "is_active": True,
            })

    return trades


# ===== Per-ticker metrics ===================================================

def compute_ticker_metrics(active_status: list[dict],
                            min_trades: int = 5) -> dict:
    """
    Compute all per-ticker performance metrics over the active_status window.

    Returns:
      {
        'trade_count':        int,
        'avg_hold_days':      float | None,    # mean calendar days
        'avg_gain_pct':       float | None,    # mean realized PnL, signed
        'avg_peak_gain_pct':  float | None,    # mean MFE, >= 0
        'success_rate':       float | None,    # kept for internal use
      }

    When trade_count < min_trades, all aggregate values are None (insufficient
    data). trade_count itself is always populated.
    """
    trades = _extract_trades(active_status)
    count = len(trades)

    null_metrics = {
        "trade_count": count,
        "avg_hold_days": None,
        "avg_gain_pct": None,
        "avg_peak_gain_pct": None,
        "success_rate": None,
    }
    if count < min_trades:
        return null_metrics

    avg_hold = sum(t["hold_days"] for t in trades) / count
    avg_gain = sum(t["pnl_pct"] for t in trades) / count
    avg_peak = sum(t["peak_gain_pct"] for t in trades) / count
    wins = sum(1 for t in trades if t["profitable"])

    return {
        "trade_count": count,
        "avg_hold_days": round(avg_hold, 1),
        "avg_gain_pct": round(avg_gain, 4),
        "avg_peak_gain_pct": round(avg_peak, 4),
        "success_rate": round(wins / count, 4),
    }


# Backwards-compat shim for any caller that still uses the old name.
def compute_ticker_success_rate(active_status: list[dict],
                                 min_trades: int = 5) -> dict:
    metrics = compute_ticker_metrics(active_status, min_trades=min_trades)
    return {
        "success_rate": metrics["success_rate"],
        "trade_count": metrics["trade_count"],
    }


# ===== Strategy-wide aggregation ============================================

def aggregate_strategy_metrics(all_trades: list[dict]) -> dict:
    """
    Compute strategy-wide aggregate metrics across trades from all tickers.
    Kept in JSON for debugging/internal use; not surfaced in the app UI.

    Returns same shape as compute_ticker_metrics. min_trades is not enforced
    at the strategy level — even one trade gives a meaningful number when
    aggregated across the universe.
    """
    count = len(all_trades)
    if count == 0:
        return {
            "trade_count": 0,
            "avg_hold_days": None,
            "avg_gain_pct": None,
            "avg_peak_gain_pct": None,
            "success_rate": None,
        }
    avg_hold = sum(t["hold_days"] for t in all_trades) / count
    avg_gain = sum(t["pnl_pct"] for t in all_trades) / count
    avg_peak = sum(t["peak_gain_pct"] for t in all_trades) / count
    wins = sum(1 for t in all_trades if t["profitable"])
    return {
        "trade_count": count,
        "avg_hold_days": round(avg_hold, 1),
        "avg_gain_pct": round(avg_gain, 4),
        "avg_peak_gain_pct": round(avg_peak, 4),
        "success_rate": round(wins / count, 4),
    }


# Old name kept as alias.
def aggregate_strategy_success_rate(all_trades: list[dict]) -> dict:
    metrics = aggregate_strategy_metrics(all_trades)
    return {
        "success_rate": metrics["success_rate"],
        "trade_count": metrics["trade_count"],
    }


# ===== Position strength classification =====================================

def classify_position_strength(close: Optional[float],
                               psar: Optional[float],
                               signal: Optional[str]) -> Optional[str]:
    """
    Classify how "safe" the current position is based on distance from PSAR.

    Buckets:
      - 'Strong'            : close is > 5% away from PSAR
      - 'Moderate'          : close is 2% to 5% away
      - 'Close to flipping' : close is < 2% away

    Returns None when signal is None (in cash).
    """
    if signal not in ("BUY", "SELL"):
        return None
    if close is None or psar is None or psar == 0:
        return None

    dist_pct = abs(close - psar) / abs(psar) * 100.0
    if dist_pct >= 5.0:
        return "Strong"
    if dist_pct >= 2.0:
        return "Moderate"
    return "Close to flipping"


def days_since(start_date_str: Optional[str],
               current_date_str: Optional[str]) -> Optional[int]:
    """Approx trading days between two dates (calendar * 5/7)."""
    if start_date_str is None or current_date_str is None:
        return None
    try:
        a = pd.Timestamp(start_date_str)
        b = pd.Timestamp(current_date_str)
    except Exception:
        return None
    delta_days = (b - a).days
    if delta_days < 0:
        return None
    return int(round(delta_days * 5 / 7))


# ===== Convenience: process one ticker end-to-end ===========================

def process_ticker_signals(df_asc: pd.DataFrame,
                            primary_entries: dict[str, str],
                            secondary_entries: dict[str, str]) -> dict:
    """
    Full per-ticker pipeline: merge entries, expand to active state, compute
    trades + metrics, classify current position.
    """
    merged = merge_entries(primary_entries, secondary_entries)
    active = expand_signals_to_active_status(df_asc, merged)

    if not active:
        return {
            "active_status": [],
            "trades": [],
            "current": {
                "signal": None,
                "signal_source": None,
                "signal_started_date": None,
                "signal_trading_days": None,
                "position_strength": None,
            },
            "metrics": {
                "trade_count": 0,
                "avg_hold_days": None,
                "avg_gain_pct": None,
                "avg_peak_gain_pct": None,
                "success_rate": None,
            },
        }

    latest = active[-1]
    trades = _extract_trades(active)
    metrics = compute_ticker_metrics(active)

    current = {
        "signal": latest["signal"],
        "signal_source": latest["signal_source"],
        "signal_started_date": latest["signal_started_date"],
        "signal_trading_days": days_since(latest["signal_started_date"], latest["date"]),
        "position_strength": classify_position_strength(
            latest["close"], latest["psar"], latest["signal"]
        ),
    }

    return {
        "active_status": active,
        "trades": trades,
        "current": current,
        "metrics": metrics,
    }