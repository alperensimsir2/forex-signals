#!/usr/bin/env python3
"""Gate GitHub Pages publish: only deploy when today's ET scan file is complete."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SCANS_PATH = Path("out/forex_scans.json")
EXPECTED_UNIVERSE = 28
ET = ZoneInfo("America/New_York")


def main() -> int:
    expected_date = datetime.now(ET).date().isoformat()

    if not SCANS_PATH.is_file():
        print(f"STALE: missing {SCANS_PATH}, skipping publish", file=sys.stderr)
        return 1

    try:
        with SCANS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"STALE: cannot read {SCANS_PATH} ({exc}), skipping publish", file=sys.stderr)
        return 1

    trade_date = data.get("trade_date")
    if trade_date != expected_date:
        print(
            f"STALE: trade_date={trade_date} expected={expected_date}, skipping publish",
            file=sys.stderr,
        )
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
