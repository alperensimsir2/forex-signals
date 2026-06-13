"""
Per-pair detail JSON export (scanner observational product).

Writes out/pairs/{SYMBOL}.json — close history + current scan/composite matches.
Independent schema (1.0.0) from forex_scans.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fetch

PAIR_DETAIL_SCHEMA_VERSION = "1.0.0"
HISTORY_CALENDAR_DAYS = 365
HISTORY_MAX_BARS = 260


def _close_num(x) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x) or not np.isfinite(x):
            return None
    except TypeError:
        return x
    return float(x)


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


def _history_rows(df: pd.DataFrame) -> list[dict]:
    window = _history_window(df)
    rows: list[dict] = []
    for _, row in window.iterrows():
        close = _close_num(row["close"])
        if close is None:
            continue
        rows.append({
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "close": close,
        })
    return rows


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

            detail = {
                "schema_version": PAIR_DETAIL_SCHEMA_VERSION,
                "generated_at": generated_at,
                "trade_date": trade_date,
                "display_symbol": display,
                "symbol": symbol,
                "category": pair_row["category"],
                "history": _history_rows(df),
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
