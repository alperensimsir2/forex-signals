"""
Per-pair detail JSON export (scanner observational product).

Writes out/pairs/{SYMBOL}.json — OHLC + indicator history plus a latest-state
snapshot, alongside the current scan/composite matches. Independent schema
(2.0.0) from forex_scans.json.

Schema 2.0.0 is strictly additive over 1.0.0: every history[] entry still
carries `date` and `close` with the exact same values as before, so the simple
chart / detail screen keep working unchanged. New per-bar fields (open/high/low
+ indicators) and the latest_state block are layered on top.

Indicator values are pulled from the scanner's own indicator frame
(scanner.build_indicator_frame) so they always agree with the published scans —
nothing is recomputed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fetch, pips

PAIR_DETAIL_SCHEMA_VERSION = "2.0.0"
HISTORY_CALENDAR_DAYS = 365
HISTORY_MAX_BARS = 260

# ~126 trading days ≈ 6 months; used for the BB-width squeeze percentile.
BB_WIDTH_PERCENTILE_LOOKBACK = 126

# Display precision for the v2.0.0 indicator fields. Price-scale fields (in the
# pair's quote currency) round like the rate: JPY pairs 3dp, others 5dp.
# Non-price fields (oscillators, MACD, ATR) round to 6dp. `close` is never
# rounded — it is preserved byte-identical from v1.0.0.
PRICE_DECIMALS_JPY = 3
PRICE_DECIMALS_DEFAULT = 5
NON_PRICE_DECIMALS = 6

# Post-`close` per-bar history fields: output key -> indicator-frame column.
# (open/high/low are emitted before `close` and handled explicitly.)
INDICATOR_FIELD_MAP: dict[str, str] = {
    "sma_9": "sma9",
    "sma_20": "sma20",
    "sma_50": "sma50",
    "sma_200": "sma200",
    "rsi_14": "rsi",
    "macd": "macd",
    "macd_signal": "macd_signal",
    "macd_hist": "macd_hist",
    "bb_upper": "bb_upper",
    "bb_middle": "bb_mid",
    "bb_lower": "bb_lower",
}

# Which history fields are price-scale (rounded to the pair's price precision).
HISTORY_PRICE_FIELDS = {
    "open", "high", "low",
    "sma_9", "sma_20", "sma_50", "sma_200",
    "bb_upper", "bb_middle", "bb_lower",
}


def _num(x) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x) or not np.isfinite(x):
            return None
    except TypeError:
        return x
    return float(x)


# Backwards-compatible alias for the original close serializer.
_close_num = _num


def _round(x, decimals: int) -> float | None:
    """Round to `decimals`, preserving null for None/NaN warmup values."""
    v = _num(x)
    if v is None:
        return None
    return round(v, decimals)


def _price_decimals(symbol: str) -> int:
    """JPY pairs price to 3 decimals, everything else to 5 (mirrors rate)."""
    return PRICE_DECIMALS_JPY if pips.pip_size(symbol) == 0.01 else PRICE_DECIMALS_DEFAULT


def _history_window(df: pd.DataFrame) -> pd.DataFrame:
    """Last 365 calendar days or last 260 bars, whichever yields fewer rows."""
    df = df.sort_values("date").reset_index(drop=True)
    by_bars = df.tail(HISTORY_MAX_BARS)
    latest = pd.Timestamp(df["date"].iloc[-1])
    cutoff = latest - pd.Timedelta(days=HISTORY_CALENDAR_DAYS)
    by_calendar = df[pd.to_datetime(df["date"]) >= cutoff]
    if len(by_calendar) < len(by_bars):
        return by_calendar.reset_index(drop=True)
    return by_bars.reset_index(drop=True)


def _history_rows(indf: pd.DataFrame, price_decimals: int) -> list[dict]:
    """
    Build history[] from the scanner's indicator frame.

    `date` and `close` are emitted exactly as in schema 1.0.0 (close unrounded).
    OHL + indicator fields are added per bar at display precision; indicators
    undefined for a bar (warmup) emit null.
    """
    window = _history_window(indf)
    rows: list[dict] = []
    for _, row in window.iterrows():
        close = _num(row["close"])
        if close is None:
            continue
        entry: dict = {
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "open": _round(row["open"], price_decimals),
            "high": _round(row["high"], price_decimals),
            "low": _round(row["low"], price_decimals),
            "close": close,
        }
        for out_key, col in INDICATOR_FIELD_MAP.items():
            if out_key in HISTORY_PRICE_FIELDS:
                entry[out_key] = _round(row[col], price_decimals)
            else:
                entry[out_key] = _round(row[col], NON_PRICE_DECIMALS)
        rows.append(entry)
    return rows


def _bb_width_percentile(indf: pd.DataFrame) -> float | None:
    """
    Today's BB width as a percentile rank vs the last ~126 trading days of BB
    widths (low percentile == squeeze). Rounded to 1 decimal.
    """
    widths = indf["bb_width"].dropna()
    if widths.empty:
        return None
    today = widths.iloc[-1]
    if not np.isfinite(today):
        return None
    window = widths.tail(BB_WIDTH_PERCENTILE_LOOKBACK)
    pct = float((window <= today).mean() * 100.0)
    return round(pct, 1)


def _latest_state(indf: pd.DataFrame, price_decimals: int) -> dict:
    """Most-recent-bar snapshot of indicators surfaced by the advanced chart."""
    last = indf.iloc[-1]
    return {
        "rsi_14": _round(last["rsi"], NON_PRICE_DECIMALS),
        "stoch_k": _round(last["stoch_k"], NON_PRICE_DECIMALS),
        "stoch_d": _round(last["stoch_d"], NON_PRICE_DECIMALS),
        "macd": _round(last["macd"], NON_PRICE_DECIMALS),
        "macd_signal": _round(last["macd_signal"], NON_PRICE_DECIMALS),
        "macd_hist": _round(last["macd_hist"], NON_PRICE_DECIMALS),
        "atr_14": _round(last["atr"], NON_PRICE_DECIMALS),
        "bb_width_vs_6mo_low_pct": _bb_width_percentile(indf),
        "donchian_high_20": _round(last["donchian_high"], price_decimals),
        "donchian_low_20": _round(last["donchian_low"], price_decimals),
        "high_52w": _round(last["high_52w"], price_decimals),
        "low_52w": _round(last["low_52w"], price_decimals),
    }


def _composites_matched(payload: dict, display_symbol: str) -> list[str]:
    matched: list[str] = []
    for comp in payload.get("composites", []):
        for entry in comp.get("matches", []):
            if entry.get("display_symbol") == display_symbol:
                matched.append(comp["scan_id"])
                break
    return sorted(matched)


def export_per_pair_files(
    payload: dict,
    pairs: list[dict],
    cache_dir: Path,
    out_dir: Path,
) -> int:
    """
    Write one detail JSON per pair in payload['pairs'] to out/pairs/{SYMBOL}.json.

    Pairs skipped by the scanner (not in payload) get no file. Per-pair write errors
    are logged and do not abort the export.
    """
    # Local import avoids a circular import (scanner imports this module).
    from .scanner import build_indicator_frame

    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    exported_symbols = {p["symbol"] for p in payload.get("pairs", [])}
    skipped = [
        pair["display_symbol"]
        for pair in pairs
        if pair["symbol"] not in exported_symbols
    ]
    if skipped:
        print(f"Per-pair export skipped ({len(skipped)}): {', '.join(skipped)}")

    generated_at = payload["generated_at"]
    trade_date = payload["trade_date"]
    written = 0
    errors: list[str] = []

    for pair_row in payload.get("pairs", []):
        symbol = pair_row["symbol"]
        display = pair_row["display_symbol"]
        try:
            df = fetch.load_cache(symbol, cache_dir)
            if df.empty:
                raise ValueError("empty cache")

            indf = build_indicator_frame(df)
            price_decimals = _price_decimals(symbol)

            detail = {
                "schema_version": PAIR_DETAIL_SCHEMA_VERSION,
                "generated_at": generated_at,
                "trade_date": trade_date,
                "display_symbol": display,
                "symbol": symbol,
                "category": pair_row["category"],
                "history": _history_rows(indf, price_decimals),
                "latest_state": _latest_state(indf, price_decimals),
                "scan_ids": pair_row["scan_ids"],
                "scan_count": pair_row["scan_count"],
                "composites_matched": _composites_matched(payload, display),
            }
            out_path = pairs_dir / f"{symbol}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(detail, f, indent=2)
            written += 1
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    if errors:
        print(f"Per-pair export errors ({len(errors)}):")
        for msg in errors:
            print(f"  {msg}")

    print(f"Wrote {written} per-pair files to {pairs_dir.resolve()}")
    return written
