"""
Export full trade history from cached OHLC for chart inspection.

Uses the existing strategy unchanged; enriches trades with peak_date and
exit_reason for visual diagnosis (no API calls).

  python -m src.export_trades
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from . import consensus, fetch, strategies as strat
from .pipeline import WARMUP_BARS, compute_indicators


CSV_COLUMNS = [
    "display_symbol",
    "direction",
    "signal_source",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "hold_days",
    "gain_pct",
    "peak_gain_pct",
    "peak_date",
    "exit_reason",
    "status",
]


def _run_strategy_on_df(df: pd.DataFrame) -> list[dict] | None:
    """Run primary + secondary + consensus on full history; return active_status."""
    if df.empty or len(df) < WARMUP_BARS + 20:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    df = compute_indicators(df)

    primary = strat.compute_primary_signals(df)
    primary = strat.apply_psar_alignment_filter(df, primary)
    secondary = strat.compute_trending_stocks_signals(df)

    merged = consensus.merge_entries(primary, secondary)
    return consensus.expand_signals_to_active_status(df, merged)


def _peak_gain_pct(side: str, entry_close: float, peak_price: float) -> float:
    if side == "BUY":
        pct = (peak_price - entry_close) / entry_close * 100.0
    else:
        pct = (entry_close - peak_price) / entry_close * 100.0
    return round(max(0.0, pct), 4)


def _pnl_pct(side: str, entry_close: float, exit_close: float) -> float:
    if side == "BUY":
        return round((exit_close - entry_close) / entry_close * 100.0, 4)
    return round((entry_close - exit_close) / entry_close * 100.0, 4)


def _hold_days(entry_date: str, exit_date: str) -> int:
    try:
        days = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
        return max(0, int(days))
    except Exception:
        return 0


def _update_peak(
    side: str,
    bar: dict,
    entry_close: float,
    peak_price: float | None,
    peak_date: str | None,
) -> tuple[float | None, str | None]:
    """Track MFE extreme and the date it last improved (mirrors consensus._extract_trades)."""
    high = bar.get("high")
    low = bar.get("low")
    date = bar["date"]

    if side == "BUY" and high is not None:
        if peak_price is None or high > peak_price:
            return high, date
    elif side == "SELL" and low is not None:
        if peak_price is None or low < peak_price:
            return low, date
    return peak_price, peak_date


def _exit_reason(prev_side: str, new_side: Optional[str], is_active: bool) -> str:
    if is_active:
        return "still open"
    if new_side is None:
        return "PSAR flip"
    if new_side in ("BUY", "SELL") and new_side != prev_side:
        return "signal reversal"
    return "PSAR flip"


def extract_trades_detailed(active_status: list[dict]) -> list[dict]:
    """
  Walk active_status and build trade rows with peak_date and exit_reason.
  Logic mirrors consensus._extract_trades; does not modify strategy code.
    """
    trades: list[dict] = []
    prev_side: Optional[str] = None
    prev_source: Optional[str] = None
    entry_date: Optional[str] = None
    entry_close: Optional[float] = None
    peak_price: Optional[float] = None
    peak_date: Optional[str] = None

    for bar in active_status:
        side = bar["signal"]
        close = bar["close"]

        if prev_side in ("BUY", "SELL"):
            peak_price, peak_date = _update_peak(
                prev_side, bar, entry_close or 0.0, peak_price, peak_date
            )

        if side != prev_side:
            if (
                prev_side in ("BUY", "SELL")
                and entry_date is not None
                and entry_close is not None
                and close is not None
            ):
                is_active = False
                exit_date = bar["date"]
                exit_close = close
                reason = _exit_reason(prev_side, side, is_active=False)

                trades.append({
                    "direction": prev_side,
                    "signal_source": prev_source or "primary",
                    "entry_date": entry_date,
                    "entry_price": entry_close,
                    "exit_date": exit_date,
                    "exit_price": exit_close,
                    "hold_days": _hold_days(entry_date, exit_date),
                    "gain_pct": _pnl_pct(prev_side, entry_close, exit_close),
                    "peak_gain_pct": _peak_gain_pct(prev_side, entry_close, peak_price)
                    if peak_price is not None
                    else 0.0,
                    "peak_date": peak_date or entry_date,
                    "exit_reason": reason,
                    "status": "closed",
                })

            if side in ("BUY", "SELL") and close is not None:
                entry_date = bar["date"]
                entry_close = close
                prev_source = bar.get("signal_source")
                high = bar.get("high")
                low = bar.get("low")
                if side == "BUY":
                    peak_price = high if high is not None else close
                else:
                    peak_price = low if low is not None else close
                peak_date = entry_date
            else:
                entry_date = None
                entry_close = None
                prev_source = None
                peak_price = None
                peak_date = None

            prev_side = side

    if (
        prev_side in ("BUY", "SELL")
        and entry_date is not None
        and entry_close is not None
        and active_status
    ):
        last_bar = active_status[-1]
        last_close = last_bar.get("close")
        if last_close is not None:
            peak_price, peak_date = _update_peak(
                prev_side, last_bar, entry_close, peak_price, peak_date
            )
            trades.append({
                "direction": prev_side,
                "signal_source": prev_source or "primary",
                "entry_date": entry_date,
                "entry_price": entry_close,
                "exit_date": "",
                "exit_price": "",
                "hold_days": _hold_days(entry_date, last_bar["date"]),
                "gain_pct": _pnl_pct(prev_side, entry_close, last_close),
                "peak_gain_pct": _peak_gain_pct(prev_side, entry_close, peak_price)
                if peak_price is not None
                else 0.0,
                "peak_date": peak_date or entry_date,
                "exit_reason": "still open",
                "status": "open",
            })

    return trades


def _giveback_lag_days(peak_date: str, exit_date: str) -> int | None:
    if not peak_date or not exit_date:
        return None
    try:
        return (pd.Timestamp(exit_date) - pd.Timestamp(peak_date)).days
    except Exception:
        return None


def _print_pair_summary(display_symbol: str, trades: list[dict]) -> None:
    closed = [t for t in trades if t["status"] == "closed"]
    n = len(trades)
    print(f"\n{display_symbol}")
    print(f"  trades: {n} ({len(closed)} closed, {n - len(closed)} open)")

    lags = [
        d
        for t in closed
        if (d := _giveback_lag_days(t["peak_date"], t["exit_date"])) is not None
    ]
    if lags:
        avg_lag = sum(lags) / len(lags)
        print(f"  give-back lag (peak_date -> exit_date): {avg_lag:.1f} days avg "
              f"({min(lags)}-{max(lags)} over {len(lags)} closed trades)")
    else:
        print("  give-back lag: n/a (no closed trades)")

    if closed:
        avg_peak = sum(t["peak_gain_pct"] for t in closed) / len(closed)
        avg_real = sum(t["gain_pct"] for t in closed) / len(closed)
        print(f"  avg peak_gain_pct: {avg_peak:+.2f}%  |  avg realized gain_pct: {avg_real:+.2f}%  "
              f"|  give-back cost: {avg_peak - avg_real:.2f} pp")
    elif trades:
        avg_peak = sum(t["peak_gain_pct"] for t in trades) / len(trades)
        avg_real = sum(t["gain_pct"] for t in trades) / len(trades)
        print(f"  avg peak_gain_pct: {avg_peak:+.2f}%  |  avg gain_pct (incl. open): {avg_real:+.2f}%")


def export_trades(
    pairs: list[dict],
    cache_dir: Path,
    out_path: Path,
) -> int:
    rows: list[dict] = []

    for pair in pairs:
        symbol = pair["symbol"]
        display = pair["display_symbol"]
        df = fetch.load_cache(symbol, cache_dir)
        active = _run_strategy_on_df(df)
        if not active:
            print(f"  skip {display}: insufficient cached history")
            continue

        trades = extract_trades_detailed(active)
        for t in trades:
            rows.append({"display_symbol": display, **t})

        _print_pair_summary(display, trades)

    rows.sort(key=lambda r: (r["display_symbol"], r["entry_date"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export trade log from cached forex history (no API calls)",
    )
    ap.add_argument("--pairs", default="forex_pairs.json")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "cache"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "out"))
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    data = json.loads(pairs_path.read_text(encoding="utf-8"))
    out_path = Path(args.out) / "forex_trades.csv"

    print(f"Exporting trades from cache: {Path(args.cache_dir).resolve()}")
    total = export_trades(data, Path(args.cache_dir), out_path)
    print(f"\nWrote {total} trades to {out_path.resolve()}")


if __name__ == "__main__":
    main()
