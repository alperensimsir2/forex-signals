"""
CLI: `python -m src` (daily) or `python -m src --backfill` (one-time history build).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import fetch, pipeline


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Pair list not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {path}")
    required = {"symbol", "eodhd_symbol", "display_symbol", "category"}
    for i, row in enumerate(data):
        missing = required - set(row)
        if missing:
            raise SystemExit(f"Pair entry {i} missing fields: {sorted(missing)}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Forex daily signals pipeline")
    ap.add_argument("--backfill", action="store_true",
                    help="Fetch extended history per pair before running.")
    ap.add_argument("--backfill-years", type=int, default=None,
                    help="Years of history for --backfill (default: fetch.BACKFILL_YEARS).")
    ap.add_argument("--pairs", default="forex_pairs.json",
                    help="Path to forex pair list JSON.")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "cache"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "out"))
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs))
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    if args.backfill:
        years = args.backfill_years or fetch.BACKFILL_YEARS
        print(f"Extended backfill for {len(pairs)} pairs...")
        failures, _ = fetch.backfill_universe(pairs, cache_dir, years=years)
        if failures:
            print(f"Done with {len(failures)} failures.")
        return

    pipeline.run(
        pairs=pairs,
        cache_dir=cache_dir,
        out_dir=out_dir,
        do_backfill=False,
    )


if __name__ == "__main__":
    main()
