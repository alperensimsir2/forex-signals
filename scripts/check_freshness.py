#!/usr/bin/env python3
"""Gate GitHub Pages publish: deploy only when forex_scans.json is complete and recent."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCANS_PATH = Path("out/forex_scans.json")
EXPECTED_UNIVERSE = 28
ET = ZoneInfo("America/New_York")


def freshness_window() -> tuple[date, date]:
    utc_today = datetime.now(timezone.utc).date()
    et_today = datetime.now(ET).date()
    low = min(et_today, utc_today) - timedelta(days=4)
    high = max(et_today, utc_today) + timedelta(days=1)
    return low, high


def main() -> int:
    if not SCANS_PATH.is_file():
        print(f"STALE: missing {SCANS_PATH}, skipping publish", file=sys.stderr)
        return 1

    try:
        with SCANS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"STALE: cannot read {SCANS_PATH} ({exc}), skipping publish", file=sys.stderr)
        return 1

    universe_size = data.get("universe_size")
    if universe_size != EXPECTED_UNIVERSE:
        print(
            f"STALE: universe_size={universe_size} expected={EXPECTED_UNIVERSE}, skipping publish",
            file=sys.stderr,
        )
        return 1

    pairs = data.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_UNIVERSE:
        pair_count = len(pairs) if isinstance(pairs, list) else "invalid"
        print(
            f"STALE: pairs count={pair_count} expected={EXPECTED_UNIVERSE}, skipping publish",
            file=sys.stderr,
        )
        return 1

    trade_date_raw = data.get("trade_date")
    try:
        if not isinstance(trade_date_raw, str):
            raise ValueError("trade_date must be YYYY-MM-DD string")
        trade_date = date.fromisoformat(trade_date_raw)
    except (TypeError, ValueError):
        print(
            f"STALE: trade_date={trade_date_raw!r} invalid, skipping publish",
            file=sys.stderr,
        )
        return 1

    window_low, window_high = freshness_window()
    if not (window_low <= trade_date <= window_high):
        print(
            f"STALE: trade_date={trade_date} outside freshness window "
            f"[{window_low} .. {window_high}]",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: trade_date={trade_date} universe={universe_size} pairs={len(pairs)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
