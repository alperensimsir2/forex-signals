"""
Daily-timeframe character diagnostic: trend vs mean-reversion vs random walk.

Uses cached OHLC only (no API calls, no strategy code).

  python -m src.character_diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import fetch

# Classification bands (conservative — avoids over-calling weak signals)
ACF_TREND = 0.05
ACF_MR = -0.05
VR_TREND = 1.05
VR_MR = 0.95
HURST_TREND = 0.55
HURST_MR = 0.45
CONT_TREND = 0.52
CONT_MR = 0.48
MIN_BARS = 80


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def lag1_autocorr(returns: pd.Series) -> float | None:
    if len(returns) < 30:
        return None
    return float(returns.autocorr(lag=1))


def variance_ratio(returns: pd.Series, k: int) -> float | None:
    """
    Lo-MacKinlay style (simplified): VR(k) = Var(k-day sum) / (k * Var(1-day)).
    VR > 1 → trending; VR < 1 → mean-reverting; VR ≈ 1 → random walk.
    Uses overlapping k-day sums of daily returns.
    """
    r = returns.dropna()
    if len(r) < k + 30:
        return None
    var_1 = float(r.var())
    if var_1 < 1e-16:
        return None
    r_k = r.rolling(k).sum().dropna()
    return float(r_k.var() / (k * var_1))


def hurst_rs(log_prices: np.ndarray) -> float | None:
    """
    Rescaled-range (R/S) estimate of Hurst exponent on log prices.
    H > 0.5 persistent/trending; H < 0.5 anti-persistent/MR; H ≈ 0.5 random.
    """
    n = len(log_prices)
    if n < 60:
        return None

    max_lag = max(20, n // 4)
    lags = np.unique(
        np.logspace(np.log10(10), np.log10(max_lag), num=12).astype(int)
    )
    points: list[tuple[float, float]] = []

    for lag in lags:
        if lag < 10:
            continue
        n_seg = n // lag
        if n_seg < 4:
            continue
        rs_vals: list[float] = []
        for i in range(n_seg):
            seg = log_prices[i * lag : (i + 1) * lag]
            if len(seg) < 2:
                continue
            dev = seg - seg.mean()
            z = np.cumsum(dev)
            r_span = float(z.max() - z.min())
            s = float(seg.std(ddof=1))
            if s > 1e-12:
                rs_vals.append(r_span / s)
        if len(rs_vals) >= 2:
            points.append((np.log(lag), np.log(np.mean(rs_vals))))

    if len(points) < 3:
        return None
    x, y = zip(*points)
    return float(np.polyfit(x, y, 1)[0])


def continuation_rate(returns: pd.Series) -> float | None:
    """% of days where sign(t) == sign(t-1). 50% = no momentum edge."""
    r = returns.dropna()
    if len(r) < 30:
        return None
    s = np.sign(r)
    same = (s.iloc[1:].values == s.iloc[:-1].values).sum()
    return float(same / (len(r) - 1))


def _score_metric(
    acf: float | None,
    vr5: float | None,
    vr10: float | None,
    hurst: float | None,
    cont: float | None,
) -> tuple[int, list[str]]:
    """+1 trend, -1 mean-revert, 0 neutral per test. Returns (total, labels)."""
    votes: list[int] = []
    notes: list[str] = []

    if acf is not None:
        if acf > ACF_TREND:
            votes.append(1)
            notes.append("acf+")
        elif acf < ACF_MR:
            votes.append(-1)
            notes.append("acf-")
        else:
            votes.append(0)
            notes.append("acf0")

    vr_avg = None
    vrs = [v for v in (vr5, vr10) if v is not None]
    if vrs:
        vr_avg = sum(vrs) / len(vrs)
        if vr_avg > VR_TREND:
            votes.append(1)
            notes.append("vr+")
        elif vr_avg < VR_MR:
            votes.append(-1)
            notes.append("vr-")
        else:
            votes.append(0)
            notes.append("vr0")

    if hurst is not None:
        if hurst > HURST_TREND:
            votes.append(1)
            notes.append("H+")
        elif hurst < HURST_MR:
            votes.append(-1)
            notes.append("H-")
        else:
            votes.append(0)
            notes.append("H0")

    if cont is not None:
        if cont > CONT_TREND:
            votes.append(1)
            notes.append("cont+")
        elif cont < CONT_MR:
            votes.append(-1)
            notes.append("cont-")
        else:
            votes.append(0)
            notes.append("cont0")

    return sum(votes), notes


def classify_pair(score: int, n_tests: int) -> str:
    """
    Majority of directional votes required; weak/mixed → RANDOM.
    With 4 tests: |score| >= 2 and same sign → directional verdict.
    """
    if n_tests < 3:
        return "RANDOM"
    if score >= 2:
        return "TREND"
    if score <= -2:
        return "MEAN-REVERT"
    return "RANDOM"


def analyze_pair(df: pd.DataFrame) -> dict | None:
    if len(df) < MIN_BARS:
        return None

    close = df.sort_values("date")["close"].astype(float)
    rets = daily_returns(close)
    log_p = np.log(close.values)

    acf = lag1_autocorr(rets)
    vr5 = variance_ratio(rets, 5)
    vr10 = variance_ratio(rets, 10)
    hurst = hurst_rs(log_p)
    cont = continuation_rate(rets)

    score, _ = _score_metric(acf, vr5, vr10, hurst, cont)
    n_tests = sum(
        1 for x in (acf, vr5, vr10, hurst, cont) if x is not None
    )
    verdict = classify_pair(score, min(n_tests, 4))

    return {
        "bars": len(close),
        "acf_lag1": acf,
        "vr5": vr5,
        "vr10": vr10,
        "hurst": hurst,
        "continuation_pct": cont,
        "vote_score": score,
        "verdict": verdict,
    }


def _fmt_acf(v: float | None) -> str:
    if v is None:
        return "     n/a"
    return f"{v:+.4f}".rjust(8)


def _fmt_vr(v: float | None) -> str:
    if v is None:
        return "    n/a"
    return f"{v:.3f}".rjust(7)


def _fmt_hurst(v: float | None) -> str:
    if v is None:
        return "    n/a"
    return f"{v:.3f}".rjust(7)


def overall_verdict(rows: list[dict]) -> tuple[str, str]:
    n = len(rows)
    trend = sum(1 for r in rows if r["verdict"] == "TREND")
    mr = sum(1 for r in rows if r["verdict"] == "MEAN-REVERT")
    rnd = sum(1 for r in rows if r["verdict"] == "RANDOM")

    acfs = [r["acf_lag1"] for r in rows if r["acf_lag1"] is not None]
    vr5s = [r["vr5"] for r in rows if r["vr5"] is not None]
    vr10s = [r["vr10"] for r in rows if r["vr10"] is not None]
    hs = [r["hurst"] for r in rows if r["hurst"] is not None]
    cs = [r["continuation_pct"] for r in rows if r["continuation_pct"] is not None]

    agg_note = (
        f"Avg lag-1 ACF={np.mean(acfs):+.4f}, "
        f"VR5={np.mean(vr5s):.3f}, VR10={np.mean(vr10s):.3f}, "
        f"Hurst={np.mean(hs):.3f}, continuation={np.mean(cs)*100:.1f}%"
    )

    if trend > mr and trend > rnd:
        label = "TREND-LEANING (weak majority)"
        prose = (
            "More pairs lean trending than mean-reverting, but many are still "
            "classified RANDOM. A trend-following system is directionally plausible "
            "but do not expect strong daily edges — calibration matters."
        )
    elif mr > trend and mr > rnd:
        label = "MEAN-REVERT-LEANING"
        prose = (
            "More pairs lean mean-reverting than trending on daily bars. A trend "
            "strategy (including your PSAR/pivot stack) fights the typical daily "
            "dynamics. Mean-reversion entries deserve priority over more trend tuning."
        )
    elif rnd >= trend and rnd >= mr:
        label = "RANDOM / NO CLEAR EDGE"
        prose = (
            "Most pairs look like random walks on the daily timeframe under these "
            "tests. Neither trend nor mean-reversion shows a robust structural "
            "advantage. Be skeptical of backtest improvements — they may be noise."
        )
    else:
        label = "MIXED / NO CONSENSUS"
        prose = (
            "Pairs split across trend, mean-revert, and random classifications. "
            "There is no single forex 'character' — pair selection or separate "
            "models per bucket would be needed."
        )

    if abs(np.mean(acfs)) < 0.03 and 0.95 < np.mean(vr5s) < 1.05 and 0.45 < np.mean(hs) < 0.55:
        label = "RANDOM / NO CLEAR EDGE"
        prose = (
            "Aggregate statistics sit near random-walk benchmarks (ACF≈0, VR≈1, "
            "Hurst≈0.5, continuation≈50%). Daily forex in this sample does not "
            "show persistent momentum or reversal strong enough to rely on."
        )

    bucket = (
        f"Buckets: {trend} TREND, {mr} MEAN-REVERT, {rnd} RANDOM (of {n} pairs)."
    )
    return label, f"{bucket}\n{agg_note}\n\n{prose}"


def run(pairs: list[dict], cache_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for pair in pairs:
        sym = pair["symbol"]
        df = fetch.load_cache(sym, cache_dir)
        stats = analyze_pair(df)
        if stats is None:
            print(f"  skip {pair['display_symbol']}: insufficient cache")
            continue
        rows.append({
            "display_symbol": pair["display_symbol"],
            "symbol": sym,
            **stats,
        })
    return rows


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("No pair data in cache. Run: python -m src --backfill")
        return

    print("\nForex daily character diagnostic (cached history)")
    print(f"Pairs analyzed: {len(rows)}  |  min bars: {MIN_BARS}+\n")

    hdr = (
        f"{'Pair':<10} {'Bars':>5} {'ACF(1)':>8} {'VR(5)':>7} {'VR(10)':>7} "
        f"{'Hurst':>7} {'Cont%':>7} {'Vote':>5} {'Verdict':<12}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in sorted(rows, key=lambda x: x["display_symbol"]):
        cont = r["continuation_pct"]
        cont_s = f"{cont * 100:5.1f}%" if cont is not None else "   n/a"
        print(
            f"{r['display_symbol']:<10} {r['bars']:>5} "
            f"{_fmt_acf(r['acf_lag1'])} {_fmt_vr(r['vr5'])} {_fmt_vr(r['vr10'])} "
            f"{_fmt_hurst(r['hurst'])} {cont_s:>7} {r['vote_score']:>+5} "
            f"{r['verdict']:<12}"
        )

    # Aggregate row
    def _mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    print("-" * len(hdr))
    m_cont = _mean("continuation_pct")
    cont_agg = f"{m_cont * 100:5.1f}%" if m_cont is not None else "   n/a"
    print(
        f"{'AVERAGE':<10} {'':>5} "
        f"{_fmt_acf(_mean('acf_lag1'))} {_fmt_vr(_mean('vr5'))} {_fmt_vr(_mean('vr10'))} "
        f"{_fmt_hurst(_mean('hurst'))} {cont_agg:>7} {'':>5} {'':<12}"
    )

    print("\n" + "=" * 72)
    label, prose = overall_verdict(rows)
    print(f"OVERALL VERDICT: {label}")
    print("=" * 72)
    print(prose)

    print("\nHow to read metrics:")
    print("  ACF(1) > 0 -> trending; < 0 -> mean-reverting; ~0 -> random")
    print("  VR(k)  > 1 -> trending; < 1 -> mean-reverting; ~1 -> random")
    print("  Hurst  > 0.5 -> persistent; < 0.5 -> anti-persistent; ~0.5 -> random")
    print("  Cont%  > 50% -> momentum; < 50% -> reversal; ~50% -> random")
    print(
        f"\nPer-pair verdict: sum of 4 tests (+1 trend / -1 MR / 0 neutral); "
        f"|score|>={2} required for TREND or MEAN-REVERT."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Trend vs mean-reversion diagnostic")
    ap.add_argument("--pairs", default="forex_pairs.json")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "cache"))
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    cache_dir = Path(args.cache_dir)
    print(f"Loading cache: {cache_dir.resolve()}")
    rows = run(pairs, cache_dir)
    print_report(rows)


if __name__ == "__main__":
    main()
