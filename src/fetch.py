"""
EODHD client for forex daily OHLC.

Endpoint:
  https://eodhd.com/api/eod/{SYMBOL}.FOREX?api_token={TOKEN}&fmt=json&period=d
  Optional: from=YYYY-MM-DD&to=YYYY-MM-DD for historical range (backfill).

Forex responses include adjusted_close (equals close) and volume (always 0).
We use close for all calculations and do not store or use volume.

Caching strategy:
  - Backfill: fetch up to BACKFILL_YEARS via from/to, store full range in Parquet.
  - Daily runs: merge latest bars; cache retains up to CACHE_MAX_BARS (~6 years).
  - If cache is missing or stale (>7 days), refetch the full backfill range.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

EODHD_KEY = os.environ.get("EODHD_API_KEY")
BASE = "https://eodhd.com/api"
DEFAULT_TIMEOUT = 60
STALE_DAYS = 7

# Research backfill: aim for 5 years; flag pairs with less than 3 years.
BACKFILL_YEARS = 5
MIN_RESEARCH_YEARS = 3
CACHE_MAX_BARS = 1600  # ~6+ years of trading days; do not truncate research cache


def _require_key():
    if not EODHD_KEY:
        raise RuntimeError("EODHD_API_KEY environment variable is not set")


def backfill_date_range(years: int = BACKFILL_YEARS) -> tuple[str, str]:
    """Return (from_date, to_date) strings for EODHD API (UTC calendar dates)."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=int(years * 365.25))
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _normalize_eod_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "adjusted_close" in df.columns:
        df["adj_close"] = df["adjusted_close"]
    else:
        df["adj_close"] = df["close"]
    keep = ["date", "open", "high", "low", "close", "adj_close"]
    return df[[c for c in keep if c in df.columns]]


def fetch_pair_eod(
    eodhd_symbol: str,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
    days: int | None = None,
) -> pd.DataFrame:
    """
    Fetch EOD history for one forex pair.

    Use from_date/to_date for extended backfill (EODHD returns all bars in range).
    Use days= only to keep the last N rows after fetch (short refresh windows).
    """
    _require_key()
    url = f"{BASE}/eod/{eodhd_symbol}"
    params: dict = {"api_token": EODHD_KEY, "fmt": "json", "period": "d"}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    df = _normalize_eod_df(r.json())
    if days is not None and not df.empty:
        df = df.tail(days).reset_index(drop=True)
    return df


def load_cache(symbol: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def save_cache(symbol: str, df: pd.DataFrame, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{symbol}.parquet", index=False)


def _years_span(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).days / 365.25


def _summarize_cache(df: pd.DataFrame, requested_from: str) -> dict:
    if df.empty:
        return {
            "bars": 0,
            "start_date": None,
            "end_date": None,
            "years": 0.0,
            "short_history": True,
        }
    start = pd.Timestamp(df["date"].min())
    end = pd.Timestamp(df["date"].max())
    years = _years_span(start, end)
    return {
        "bars": len(df),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "years": round(years, 2),
        "short_history": years < MIN_RESEARCH_YEARS,
        "requested_from": requested_from,
    }


def print_backfill_report(reports: list[dict], failures: list[str]) -> None:
    if not reports:
        print("No pairs backfilled.")
        return

    print("\nBackfill summary (per pair)")
    print(
        f"{'Pair':<10} {'Bars':>6} {'Start':>12} {'End':>12} {'Years':>6}  Note"
    )
    print("-" * 62)

    short_flags: list[str] = []
    for r in sorted(reports, key=lambda x: x["display_symbol"]):
        note = ""
        if r["short_history"]:
            note = f"WARNING: < {MIN_RESEARCH_YEARS}y history"
            short_flags.append(r["display_symbol"])
        print(
            f"{r['display_symbol']:<10} {r['bars']:>6} "
            f"{r['start_date'] or 'n/a':>12} {r['end_date'] or 'n/a':>12} "
            f"{r['years']:>6.2f}  {note}"
        )

    total_bars = sum(r["bars"] for r in reports)
    avg_years = sum(r["years"] for r in reports) / len(reports)
    print("-" * 62)
    print(
        f"{'TOTAL':<10} {total_bars:>6}  "
        f"{len(reports)} pairs OK, {len(failures)} failed, "
        f"avg span {avg_years:.2f}y"
    )
    if short_flags:
        print(
            f"\nFlagged (< {MIN_RESEARCH_YEARS} years): {', '.join(short_flags)}"
        )
        print("  Check EODHD plan limits or pair listing date for these symbols.")
    if failures:
        print(f"\nFailed symbols: {', '.join(failures)}")

    print("\nCache verification:")
    for r in sorted(reports, key=lambda x: x["display_symbol"]):
        print(f"  {r['symbol']}: {r['bars']} bars  ({r['start_date']} .. {r['end_date']})")


def backfill_universe(
    pairs: Iterable[dict],
    cache_dir: Path,
    years: int = BACKFILL_YEARS,
    sleep_s: float = 0.15,
) -> tuple[list[str], list[dict]]:
    """
    Fetch maximum available daily history (target ``years``) per pair via from/to.
    Overwrites cache. Returns (failed_symbols, per_pair_reports).
    """
    from_date, to_date = backfill_date_range(years)
    print(
        f"Backfill range: from={from_date} to={to_date} "
        f"(target {years} years, min research {MIN_RESEARCH_YEARS}y)"
    )

    failures: list[str] = []
    reports: list[dict] = []

    for pair in pairs:
        symbol = pair["symbol"]
        eodhd = pair["eodhd_symbol"]
        display = pair["display_symbol"]
        try:
            df = fetch_pair_eod(eodhd, from_date=from_date, to_date=to_date)
            if df.empty:
                print(f"[backfill] {display}: empty response")
                failures.append(symbol)
                continue
            if len(df) > CACHE_MAX_BARS:
                df = df.tail(CACHE_MAX_BARS).reset_index(drop=True)
            save_cache(symbol, df, cache_dir)
            summary = _summarize_cache(df, from_date)
            reports.append({
                "symbol": symbol,
                "display_symbol": display,
                **summary,
            })
        except Exception as e:  # noqa: BLE001
            print(f"[backfill] {display}: {e}")
            failures.append(symbol)
        time.sleep(sleep_s)

    print_backfill_report(reports, failures)
    return failures, reports


def _cache_is_stale(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    last = df["date"].max()
    if pd.isna(last):
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
    return pd.Timestamp(last).tz_localize(None) < cutoff.replace(tzinfo=None)


def merge_eod_into_cache(symbol: str, fresh: pd.DataFrame, cache_dir: Path) -> None:
    """Append/merge fresh bars; keep rolling window up to CACHE_MAX_BARS."""
    if fresh.empty:
        return
    df = load_cache(symbol, cache_dir)
    merged = pd.concat([df, fresh], ignore_index=True) if not df.empty else fresh.copy()
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.tail(CACHE_MAX_BARS).reset_index(drop=True)
    save_cache(symbol, merged, cache_dir)


def refresh_universe(
    pairs: Iterable[dict],
    cache_dir: Path,
    sleep_s: float = 0.1,
) -> list[str]:
    """
    Daily refresh: merge recent bars, or full-range refetch if cache stale/missing.
    """
    from_date, to_date = backfill_date_range(BACKFILL_YEARS)
    failures: list[str] = []

    for pair in pairs:
        symbol = pair["symbol"]
        eodhd = pair["eodhd_symbol"]
        try:
            cached = load_cache(symbol, cache_dir)
            if _cache_is_stale(cached):
                df = fetch_pair_eod(eodhd, from_date=from_date, to_date=to_date)
                if df.empty:
                    failures.append(symbol)
                    continue
                if len(df) > CACHE_MAX_BARS:
                    df = df.tail(CACHE_MAX_BARS).reset_index(drop=True)
                save_cache(symbol, df, cache_dir)
            else:
                recent_from = (
                    datetime.now(timezone.utc).date() - timedelta(days=45)
                ).strftime("%Y-%m-%d")
                df = fetch_pair_eod(
                    eodhd, from_date=recent_from, to_date=to_date, days=45
                )
                if df.empty:
                    failures.append(symbol)
                    continue
                merge_eod_into_cache(symbol, df, cache_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[refresh] {symbol}: {e}")
            failures.append(symbol)
        time.sleep(sleep_s)
    return failures
