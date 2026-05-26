"""
Historical validation of composite scans over cached 5-year OHLC.

Replays scanner condition logic bar-by-bar (no API, no definition changes).

  python -m src.composite_validation
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import fetch
from .scanner import (
    COMPOSITE_CATALOG,
    MIN_BARS,
    _any,
    _weekend_gap_fade_match,
    build_indicator_frame,
    evaluate_conditions,
)


def _weekend_gap_fade_second_leg(c: dict) -> bool:
    """Second AND leg only (fill / reversal), for bottleneck stats."""
    fill_or = _any(c, ("gap_fill_up", "gap_fill_down"))
    if c.get("weekend_gap_up"):
        fill_or = fill_or or _any(
            c, ("gap_fill_down", "bearish_engulfing", "shooting_star", "tweezer_top")
        )
    if c.get("weekend_gap_down"):
        fill_or = fill_or or _any(
            c, ("gap_fill_up", "bullish_engulfing", "hammer", "tweezer_bottom")
        )
    return fill_or


# Named legs for bottleneck diagnosis (mirrors COMPOSITE_CATALOG AND structure).
COMPOSITE_LEGS: dict[str, list[tuple[str, Callable[[dict], bool]]]] = {
    "composite_volatility_squeeze": [
        ("compression (NR4|NR7|inside)", lambda c: _any(c, ("nr4", "nr7", "inside_day"))),
        ("ATR expansion", lambda c: c.get("atr_expansion", False)),
        (
            "breakout up (Donchian|BB|engulfing)",
            lambda c: _any(c, ("donchian_high_20", "bb_breakout_up", "bullish_engulfing")),
        ),
    ],
    "composite_uptrend_squeeze_breakout": [
        (
            "extended above SMA50|200",
            lambda c: _any(c, ("sma50_extended_above", "sma200_extended_above")),
        ),
        ("SMA50 rising", lambda c: c.get("sma50_rising", False)),
        ("BB squeeze in last 5 sessions", lambda c: c.get("bb_squeeze_recent", False)),
        ("BB breakout up today", lambda c: c.get("bb_breakout_up", False)),
    ],
    "composite_oversold_reversal": [
        (
            "extended below SMA20|50",
            lambda c: _any(c, ("sma20_extended_below", "sma50_extended_below")),
        ),
        ("RSI<30 | Stoch<20", lambda c: _any(c, ("rsi_lt_30", "stoch_lt_20"))),
        (
            "reversal candle",
            lambda c: _any(
                c,
                (
                    "stoch_oversold_turning_up",
                    "bullish_engulfing",
                    "hammer",
                    "outside_day",
                ),
            ),
        ),
    ],
    "composite_trend_pullback_long": [
        ("SMA50 rising (today)", lambda c: c.get("sma50_rising", False)),
        (
            "3-bar pullback up in last 2 sessions",
            lambda c: c.get("pullback_3bar_up_recent", False),
        ),
        (
            "turn-up today (stoch | RSI cross above 50)",
            lambda c: _any(c, ("stoch_oversold_turning_up", "rsi_cross_above_50")),
        ),
    ],
    "composite_trend_pullback_short": [
        ("SMA50 falling (today)", lambda c: c.get("sma50_falling", False)),
        (
            "3-bar pullback down in last 2 sessions",
            lambda c: c.get("pullback_3bar_down_recent", False),
        ),
        (
            "turn-down today (stoch | RSI cross below 50)",
            lambda c: _any(c, ("stoch_overbought_turning_down", "rsi_cross_below_50")),
        ),
    ],
    "composite_breakdown_expansion": [
        (
            "below SMA200",
            lambda c: _any(c, ("sma200_below", "sma200_extended_below")),
        ),
        (
            "breakdown (BB down|Donchian low)",
            lambda c: _any(c, ("bb_breakout_down", "donchian_low_20")),
        ),
        ("ATR expansion", lambda c: c.get("atr_expansion", False)),
        (
            "bearish candle",
            lambda c: _any(c, ("bearish_engulfing", "marubozu_bearish", "shooting_star")),
        ),
    ],
    "composite_failed_breakout": [
        (
            "recent highs (Donchian|ext SMA200|BB up)",
            lambda c: _any(c, ("donchian_high_20", "sma200_extended_above", "bb_breakout_up")),
        ),
        (
            "overbought (RSI>70|stoch turn down)",
            lambda c: _any(c, ("rsi_gt_70", "stoch_overbought_turning_down")),
        ),
        (
            "reversal candle",
            lambda c: _any(
                c,
                ("bearish_engulfing", "shooting_star", "outside_day", "tweezer_top"),
            ),
        ),
    ],
    "composite_golden_cross": [
        ("golden cross", lambda c: c.get("golden_cross", False)),
        ("SMA50 rising", lambda c: c.get("sma50_rising", False)),
        (
            "MACD confirm",
            lambda c: _any(c, ("macd_bull_cross", "macd_zero_cross_up")),
        ),
    ],
    "composite_weekend_gap_fade": [
        ("weekend gap", lambda c: _any(c, ("weekend_gap_up", "weekend_gap_down"))),
        ("fill or reversal vs gap", _weekend_gap_fade_second_leg),
    ],
}


@dataclass
class CompositeStats:
    scan_id: str
    name: str
    total_fires: int = 0
    distinct_pairs: set = field(default_factory=set)
    last_fire_date: str | None = None
    clean_fires: int = 0
    in_downtrend_fires: int = 0
    fires_by_date: dict = field(default_factory=lambda: defaultdict(list))


def _date_str(v) -> str:
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def diagnose_bottleneck(
    comp_id: str,
    leg_hits: dict[str, int],
    leg_and_hits: int,
    total_pair_days: int,
) -> None:
    legs = COMPOSITE_LEGS.get(comp_id, [])
    print(f"\n  BOTTLENECK DIAGNOSIS — {comp_id} ({leg_and_hits:,} joint hits / {total_pair_days:,} pair-days)")
    print(f"  Evaluated pair-days: {total_pair_days:,}")
    print(f"  All legs true together: {leg_and_hits:,} ({100*leg_and_hits/max(1,total_pair_days):.3f}%)")
    ranked = sorted(leg_hits.items(), key=lambda x: x[1])
    print("  Per-leg hit rate (rarest first):")
    for leg_name, hits in ranked:
        pct = 100.0 * hits / max(1, total_pair_days)
        print(f"    {leg_name:<42} {hits:>6,}  ({pct:5.2f}%)")


CHANGED_COMPOSITE_IDS = (
    "composite_uptrend_squeeze_breakout",
    "composite_trend_pullback_long",
    "composite_trend_pullback_short",
)

PULLBACK_COMPOSITE_IDS = (
    "composite_trend_pullback_long",
    "composite_trend_pullback_short",
)


def _catalog_subset(only_ids: tuple[str, ...] | None):
    if not only_ids:
        return COMPOSITE_CATALOG
    allowed = set(only_ids)
    return [c for c in COMPOSITE_CATALOG if c.scan_id in allowed]


def run_validation(
    pairs: list[dict],
    cache_dir: Path,
    only_ids: tuple[str, ...] | None = None,
) -> tuple[dict[str, CompositeStats], float, int, dict, dict]:
    catalog = _catalog_subset(only_ids)
    stats = {c.scan_id: CompositeStats(scan_id=c.scan_id, name=c.name) for c in catalog}
    leg_hit_counts: dict[str, dict[str, int]] = {
        c.scan_id: {name: 0 for name, _ in COMPOSITE_LEGS[c.scan_id]} for c in catalog
    }
    leg_and_counts: dict[str, int] = {c.scan_id: 0 for c in catalog}

    min_date = None
    max_date = None
    total_pair_days = 0

    for pair in pairs:
        sym = pair["symbol"]
        display = pair["display_symbol"]
        df = fetch.load_cache(sym, cache_dir)
        if len(df) < MIN_BARS + 1:
            continue
        indf = build_indicator_frame(df)

        for i in range(MIN_BARS, len(indf)):
            window = indf.iloc[: i + 1]
            cond = evaluate_conditions(window)
            if not cond:
                continue
            d = _date_str(window["date"].iloc[-1])
            total_pair_days += 1
            if min_date is None or d < min_date:
                min_date = d
            if max_date is None or d > max_date:
                max_date = d

            for comp in catalog:
                legs = COMPOSITE_LEGS.get(comp.scan_id, [])
                all_legs = all(fn(cond) for _, fn in legs)
                if all_legs:
                    leg_and_counts[comp.scan_id] += 1
                for leg_name, fn in legs:
                    if fn(cond):
                        leg_hit_counts[comp.scan_id][leg_name] += 1
                if comp.match(cond):
                    st = stats[comp.scan_id]
                    st.total_fires += 1
                    st.distinct_pairs.add(display)
                    if st.last_fire_date is None or d > st.last_fire_date:
                        st.last_fire_date = d
                    st.fires_by_date[d].append(display)
                    if comp.quality:
                        q = comp.quality(cond)
                        if q == "clean":
                            st.clean_fires += 1
                        elif q == "in_downtrend":
                            st.in_downtrend_fires += 1

    if min_date and max_date:
        years = max(
            (pd.Timestamp(max_date) - pd.Timestamp(min_date)).days / 365.25,
            1.0,
        )
    else:
        years = 5.0

    return stats, years, total_pair_days, leg_hit_counts, leg_and_counts


def _health_flag(total: int, years: float, comp_id: str) -> str:
    per_year = total / years if years else 0
    if total == 0:
        return "ZERO - structurally blocked?"
    if comp_id in ("composite_trend_pullback_long", "composite_trend_pullback_short"):
        if total < 10:
            return "TOO STRICT - under 10 in 5y"
        if total > 1000:
            return "LOOSE - over 1000 in 5y"
        if total >= 200:
            return "WIDE WINDOW - hundreds+ fires; try 2-session pullback lookback"
        if 20 <= total <= 150:
            return "OK - selective but real (20-150 target)"
        if total < 20:
            return "LOW - below 20-150 target but present"
        return "HIGH - above 150 target but not extreme"
    if comp_id == "composite_uptrend_squeeze_breakout":
        if per_year > 200:
            return "LOOSE - very high for squeeze+breakout"
        if total < 5:
            return "LOW - fires but very rare"
        return "OK - selective (squeeze-then-breakout)"
    if per_year > 500:
        return "LOOSE - very high frequency"
    if total < 12:
        return "rare but present"
    return "ok — selective"


def print_table(
    stats: dict[str, CompositeStats],
    years: float,
    catalog: list,
    title: str = "Composite historical validation",
) -> None:
    print(f"\n{title}")
    print(f"History span used for fires/year: ~{years:.1f} years\n")
    hdr = f"{'Composite':<32} {'Total':>7} {'Pairs':>6} {'/yr':>7} {'Last fire':>12}  Health"
    print(hdr)
    print("-" * len(hdr))

    for comp in catalog:
        st = stats[comp.scan_id]
        per_year = st.total_fires / years if years else 0
        extra = ""
        if comp.scan_id == "composite_oversold_reversal":
            extra = f"  clean={st.clean_fires} downtrend={st.in_downtrend_fires}"
        health = _health_flag(st.total_fires, years, comp.scan_id)
        print(
            f"{st.name:<32} {st.total_fires:>7,} {len(st.distinct_pairs):>6} "
            f"{per_year:>7.1f} {st.last_fire_date or 'n/a':>12}  {health}{extra}"
        )


def print_pullback_long_example(stats: dict[str, CompositeStats]) -> None:
    st = stats.get("composite_trend_pullback_long")
    if not st or not st.fires_by_date:
        print("\nNo Trend Pullback - Long fires in history - cannot show example.")
        return
    best_date = max(
        st.fires_by_date.keys(),
        key=lambda d: len(st.fires_by_date[d]),
    )
    pairs = sorted(set(st.fires_by_date[best_date]))
    print(
        f"\nTrend Pullback - Long example ({best_date}, "
        f"{len(pairs)} pairs):"
    )
    print(f"  {', '.join(pairs)}")


def print_uptrend_example(stats: dict[str, CompositeStats]) -> None:
    st = stats.get("composite_uptrend_squeeze_breakout")
    if not st or not st.fires_by_date:
        print("\nNo Uptrend Squeeze Breakout fires in history — cannot show example.")
        return
    best_date = max(
        st.fires_by_date.keys(),
        key=lambda d: len(st.fires_by_date[d]),
    )
    pairs = sorted(set(st.fires_by_date[best_date]))
    print(
        f"\nUptrend Squeeze Breakout example ({best_date}, "
        f"{len(pairs)} pairs):"
    )
    print(f"  {', '.join(pairs)}")


def print_example_day(stats: dict[str, CompositeStats], catalog: list) -> None:
    """Most recent calendar date with at least one composite fire."""
    date_to_composites: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for comp in catalog:
        st = stats[comp.scan_id]
        for d, pairs in st.fires_by_date.items():
            for p in pairs:
                date_to_composites[d][comp.name].append(p)

    if not date_to_composites:
        print("\nNo historical composite fires found — cannot show example day.")
        return

    # Busiest day by number of (composite, pair) hits — easier to eyeball than a lone gap fade.
    example_date = max(
        date_to_composites.keys(),
        key=lambda d: sum(len(v) for v in date_to_composites[d].values()),
    )
    n_hits = sum(len(v) for v in date_to_composites[example_date].values())
    print(
        f"\nExample day (busiest composite activity in history): {example_date} "
        f"({n_hits} pair-composite hits)"
    )
    day = date_to_composites[example_date]
    for comp_name in sorted(day.keys()):
        pairs = sorted(set(day[comp_name]))
        print(f"  {comp_name}: {', '.join(pairs)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate composites over cached history")
    ap.add_argument("--pairs", default="forex_pairs.json")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "cache"))
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated composite scan_ids (default: all). "
        "Use 'changed' for S3c trio, 'pullbacks' for Trend Pullback long/short.",
    )
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))

    only_ids: tuple[str, ...] | None = None
    if args.only == "changed":
        only_ids = CHANGED_COMPOSITE_IDS
    elif args.only == "pullbacks":
        only_ids = PULLBACK_COMPOSITE_IDS
    elif args.only.strip():
        only_ids = tuple(s.strip() for s in args.only.split(",") if s.strip())

    catalog = _catalog_subset(only_ids)
    print(f"Validating composites: {cache_dir.resolve()}")
    print(f"Pairs: {len(pairs)}  |  MIN_BARS: {MIN_BARS}")
    if only_ids:
        print(f"Scope: {', '.join(only_ids)}")

    stats, years, pair_days, leg_hits, leg_and = run_validation(
        pairs, cache_dir, only_ids=only_ids
    )
    print(f"Total pair-days evaluated: {pair_days:,}")

    if only_ids == PULLBACK_COMPOSITE_IDS:
        title = "S3e re-validation (2 Trend Pullback composites, 2-session window)"
    elif only_ids == CHANGED_COMPOSITE_IDS:
        title = "S3c re-validation (3 changed composites)"
    else:
        title = "Composite historical validation"
    print_table(stats, years, catalog, title=title)

    for comp in catalog:
        st = stats[comp.scan_id]
        health = _health_flag(st.total_fires, years, comp.scan_id)
        if st.total_fires == 0 or "TOO STRICT" in health or "ZERO" in health:
            diagnose_bottleneck(
                comp.scan_id,
                leg_hits[comp.scan_id],
                leg_and[comp.scan_id],
                pair_days,
            )

    if only_ids == PULLBACK_COMPOSITE_IDS:
        print_pullback_long_example(stats)
    elif only_ids == CHANGED_COMPOSITE_IDS:
        print_uptrend_example(stats)
    else:
        print_example_day(stats, catalog)

    print("\nHealth guide (S3c pullbacks target ~20-150 fires over 5y):")
    print("  Uptrend Squeeze: must fire >0; relatively rare is expected")
    print("  ZERO / TOO STRICT = needs further adjustment")


if __name__ == "__main__":
    main()
