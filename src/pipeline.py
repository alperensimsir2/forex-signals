"""
End-to-end forex signals pipeline.

Strategy logic is identical to the S&P 500 pipeline (Primary 15/4 Pivot
Confluence + Secondary PSAR Trend Continuation). Outputs signals-forex.json
with forex-specific fields (display_symbol, category, change_pips, pip_size).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import consensus, fetch, indicators as ind, pips, strategies as strat


SCHEMA_VERSION = "1.0.0"
SUCCESS_WINDOW_BARS = 260
WARMUP_BARS = 60


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    adj = df["adj_close"]
    macd_line, _, _ = ind.macd(adj, 12, 26, 9)
    df["macd"] = macd_line
    df["stoch_k"] = ind.stoch_slow_k(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_d"] = ind.stoch_d(df["stoch_k"], 3)
    df["psar"] = ind.parabolic_sar(df["high"], df["low"])
    return df


def _price_num(x) -> float | None:
    """Preserve full API precision for OHLC in output (no rounding)."""
    if x is None:
        return None
    try:
        if pd.isna(x) or not np.isfinite(x):
            return None
    except TypeError:
        return x
    return float(x)


def _metric_num(x, decimals: int = 4) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x) or not np.isfinite(x):
            return None
    except TypeError:
        return x
    return round(float(x), decimals)


def _latest_change_pct(df: pd.DataFrame) -> float | None:
    if len(df) < 2:
        return None
    latest_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    if pd.isna(latest_close) or pd.isna(prev_close) or prev_close == 0:
        return None
    return round(float((latest_close / prev_close - 1.0) * 100.0), 4)


def process_pair(
    pair: dict,
    df: pd.DataFrame,
) -> tuple[dict, dict, list[dict]] | None:
    symbol = pair["symbol"]
    if df.empty or len(df) < WARMUP_BARS + 20:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    df = compute_indicators(df)

    df_full = df.copy()
    primary = strat.compute_primary_signals(df_full)
    primary = strat.apply_psar_alignment_filter(df_full, primary)
    secondary = strat.compute_trending_stocks_signals(df_full)

    window = df_full.tail(SUCCESS_WINDOW_BARS).reset_index(drop=True)
    window_dates = set(window["date"].astype(str).str[:10])
    primary_win = {d: v for d, v in primary.items() if d in window_dates}
    secondary_win = {d: v for d, v in secondary.items() if d in window_dates}

    result = consensus.process_ticker_signals(window, primary_win, secondary_win)
    if not result["active_status"]:
        return None

    current = result["current"]
    metrics = result["metrics"]

    active_peak_gain_pct = None
    if result["trades"] and result["trades"][-1].get("is_active"):
        active_peak_gain_pct = result["trades"][-1]["peak_gain_pct"]

    sparkline_closes = [
        _price_num(row["close"]) for row in result["active_status"][-30:]
    ]
    sparkline_closes = [v for v in sparkline_closes if v is not None]

    latest_close = _price_num(result["active_status"][-1]["close"])
    prev_close = _price_num(window["close"].iloc[-2]) if len(window) >= 2 else None
    pip_sz = pips.pip_size(symbol)

    summary = {
        "symbol": symbol,
        "display_symbol": pair["display_symbol"],
        "category": pair["category"],
        "close": latest_close,
        "change_pct": _latest_change_pct(window),
        "change_pips": pips.change_pips(latest_close, prev_close, symbol)
        if latest_close is not None and prev_close is not None
        else None,
        "pip_size": pip_sz,
        "signal": current["signal"],
        "signal_source": current["signal_source"],
        "signal_started_date": current["signal_started_date"],
        "signal_trading_days": current["signal_trading_days"],
        "position_strength": current["position_strength"],
        "active_peak_gain_pct": active_peak_gain_pct,
        "sparkline": sparkline_closes,
        "trade_count_52w": metrics["trade_count"],
        "avg_hold_days": metrics["avg_hold_days"],
        "avg_gain_pct": metrics["avg_gain_pct"],
        "avg_peak_gain_pct": metrics["avg_peak_gain_pct"],
        "success_rate_52w": metrics["success_rate"],
    }

    hist_rows = []
    for row in result["active_status"]:
        hist_rows.append({
            "date": row["date"],
            "close": _price_num(row["close"]),
            "signal": row["signal"],
            "psar": _metric_num(row["psar"]),
        })

    history_payload = {
        "symbol": symbol,
        "display_symbol": pair["display_symbol"],
        "schema_version": SCHEMA_VERSION,
        "current": {
            "signal": current["signal"],
            "signal_source": current["signal_source"],
            "signal_started_date": current["signal_started_date"],
            "signal_trading_days": current["signal_trading_days"],
            "position_strength": current["position_strength"],
            "peak_gain_pct": active_peak_gain_pct,
        },
        "metrics": {
            "trade_count": metrics["trade_count"],
            "avg_hold_days": metrics["avg_hold_days"],
            "avg_gain_pct": metrics["avg_gain_pct"],
            "avg_peak_gain_pct": metrics["avg_peak_gain_pct"],
            "success_rate": metrics["success_rate"],
        },
        "history": hist_rows,
    }

    return summary, history_payload, result["trades"]


def run(
    pairs: list[dict],
    cache_dir: Path,
    out_dir: Path,
    do_backfill: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)

    if do_backfill:
        print(f"Backfilling {len(pairs)} pairs (target {fetch.BACKFILL_YEARS}y history)...")
        failures, _reports = fetch.backfill_universe(pairs, cache_dir)
        if failures:
            print(f"  {len(failures)} failures: {failures}")
    else:
        print(f"Refreshing {len(pairs)} pairs from EODHD...")
        failures = fetch.refresh_universe(pairs, cache_dir)
        if failures:
            print(f"  {len(failures)} failures: {failures}")

    summaries = []
    all_trades: list[dict] = []
    for pair in pairs:
        symbol = pair["symbol"]
        df = fetch.load_cache(symbol, cache_dir)
        result = process_pair(pair, df)
        if result is None:
            continue
        summary, history, trades = result
        summaries.append(summary)
        all_trades.extend(trades)
        with open(out_dir / "pairs" / f"{symbol}.json", "w") as f:
            json.dump(history, f, separators=(",", ":"))

    agg = consensus.aggregate_strategy_metrics(all_trades)

    trade_date = None
    if summaries:
        any_symbol = summaries[0]["symbol"]
        df = fetch.load_cache(any_symbol, cache_dir)
        if not df.empty:
            trade_date = df["date"].max().strftime("%Y-%m-%d")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trade_date": trade_date,
        "universe_size": len(summaries),
        "strategy_trade_count_52w": agg["trade_count"],
        "strategy_avg_hold_days": agg["avg_hold_days"],
        "strategy_avg_gain_pct": agg["avg_gain_pct"],
        "strategy_avg_peak_gain_pct": agg["avg_peak_gain_pct"],
        "strategy_success_rate_52w": agg["success_rate"],
        "signals": summaries,
    }
    with open(out_dir / "signals-forex.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    buy_count = sum(1 for s in summaries if s["signal"] == "BUY")
    sell_count = sum(1 for s in summaries if s["signal"] == "SELL")
    none_count = sum(1 for s in summaries if s["signal"] is None)
    print(f"Done. {len(summaries)} pairs processed. "
          f"BUY={buy_count} SELL={sell_count} CASH={none_count}.")
    def _fmt_pct(v):
        return f"{v:+.2f}%" if v is not None else "n/a"

    print(f"Strategy 52w: {agg['trade_count']} trades, "
          f"avg hold {agg['avg_hold_days']}d, "
          f"avg gain {_fmt_pct(agg['avg_gain_pct'])}, "
          f"avg peak gain {_fmt_pct(agg['avg_peak_gain_pct'])}, "
          f"success {agg['success_rate']}.")
