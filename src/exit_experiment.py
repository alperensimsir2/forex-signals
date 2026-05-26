"""
Exit-improvement experiment: same entries as live strategy, alternate exit rules.

Uses cached OHLC only (no API calls). Does not modify pipeline/strategy modules.

  python -m src.exit_experiment
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from . import consensus, fetch, indicators as ind, strategies as strat
from .export_trades import CSV_COLUMNS
from .pipeline import WARMUP_BARS, compute_indicators
from .pips import pip_size


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _date_str(v) -> str:
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v)[:10]


def prepare_pair(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty or len(df) < WARMUP_BARS + 20:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    df = compute_indicators(df)

    primary = strat.compute_primary_signals(df)
    primary = strat.apply_psar_alignment_filter(df, primary)
    secondary = strat.compute_trending_stocks_signals(df)
    merged = consensus.merge_entries(primary, secondary)

    high = df["high"]
    low = df["low"]
    psar_03 = ind.parabolic_sar(high, low, af_step=0.03, af_max=0.2)
    psar_04 = ind.parabolic_sar(high, low, af_step=0.04, af_max=0.2)

    bars: list[dict] = []
    for i, row in df.iterrows():
        bars.append({
            "idx": int(i),
            "date": _date_str(row["date"]),
            "close": float(row["close"]) if pd.notna(row["close"]) else None,
            "high": float(row["high"]) if pd.notna(row["high"]) else None,
            "low": float(row["low"]) if pd.notna(row["low"]) else None,
            "psar": float(row["psar"]) if pd.notna(row["psar"]) else None,
            "psar_03": float(psar_03.iloc[i]) if pd.notna(psar_03.iloc[i]) else None,
            "psar_04": float(psar_04.iloc[i]) if pd.notna(psar_04.iloc[i]) else None,
        })

    active = consensus.expand_signals_to_active_status(df, merged)
    entries = _extract_entries(active)

    return {
        "bars": bars,
        "merged": merged,
        "entries": entries,
        "date_to_idx": {b["date"]: b["idx"] for b in bars},
    }


def _extract_entries(active_status: list[dict]) -> list[dict]:
    """Entry events exactly when the live strategy opens a position."""
    out: list[dict] = []
    for bar in active_status:
        side = bar.get("signal")
        started = bar.get("signal_started_date")
        if side in ("BUY", "SELL") and started == bar["date"]:
            out.append({
                "entry_date": bar["date"],
                "entry_price": bar["close"],
                "direction": side,
                "signal_source": bar.get("signal_source") or "primary",
                "entry_idx": None,  # filled by caller
            })
    return out


def _gain_pct(side: str, entry: float, price: float) -> float:
    if side == "BUY":
        return (price - entry) / entry * 100.0
    return (entry - price) / entry * 100.0


def _update_mfe(side: str, entry: float, bar: dict,
                peak_price: float | None) -> tuple[float | None, float]:
    high, low = bar.get("high"), bar.get("low")
    if side == "BUY" and high is not None:
        peak_price = high if peak_price is None else max(peak_price, high)
    elif side == "SELL" and low is not None:
        peak_price = low if peak_price is None else min(peak_price, low)
    if peak_price is None:
        return None, 0.0
    return peak_price, max(0.0, _gain_pct(side, entry, peak_price))


def _spread_cost_pct(symbol: str, category: str, entry_price: float) -> float:
    pips = 2.0 if category == "major" else 3.0
    return pips * pip_size(symbol) / entry_price * 100.0


# ---------------------------------------------------------------------------
# Exit simulation (entries fixed; only exit logic varies)
# ---------------------------------------------------------------------------

def _check_baseline_exit(
    side: str,
    bar: dict,
    merged: dict[str, tuple[str, str]],
) -> Optional[str]:
    date = bar["date"]
    close, psar = bar["close"], bar["psar"]
    if close is None:
        return None
    if date in merged:
        new_side, _ = merged[date]
        if new_side != side:
            return "signal reversal"
    if psar is not None:
        if side == "BUY" and close < psar:
            return "PSAR flip"
        if side == "SELL" and close > psar:
            return "PSAR flip"
    return None


def _check_psar_exit(side: str, bar: dict, psar_key: str) -> bool:
    close = bar["close"]
    psar = bar.get(psar_key)
    if close is None or psar is None:
        return False
    if side == "BUY" and close < psar:
        return True
    if side == "SELL" and close > psar:
        return True
    return False


def simulate_trade(
    ctx: dict,
    entry: dict,
    exit_fn: Callable[..., Optional[tuple[str, str]]],
    exit_kw: dict,
) -> dict | None:
    bars = ctx["bars"]
    merged = ctx["merged"]
    idx = ctx["date_to_idx"].get(entry["entry_date"])
    if idx is None:
        return None

    side = entry["direction"]
    entry_price = entry["entry_price"]
    if entry_price is None:
        return None

    peak_price: float | None = None
    peak_gain_pct = 0.0
    peak_date = entry["entry_date"]
    last_peak_advance = entry["entry_date"]
    state = {"peak_gain_pct": 0.0, "last_peak_advance": last_peak_advance}

    entry_bar = bars[idx]
    if side == "BUY":
        peak_price = entry_bar.get("high") or entry_price
    else:
        peak_price = entry_bar.get("low") or entry_price
    peak_gain_pct = max(0.0, _gain_pct(side, entry_price, peak_price))
    state["peak_gain_pct"] = peak_gain_pct

    for j in range(idx + 1, len(bars)):
        bar = bars[j]
        peak_price, peak_gain_pct = _update_mfe(side, entry_price, bar, peak_price)
        if peak_gain_pct > state["peak_gain_pct"] + 1e-9:
            state["peak_gain_pct"] = peak_gain_pct
            state["last_peak_advance"] = bar["date"]
            peak_date = bar["date"]

        close = bar["close"]
        if close is None:
            continue

        hit = exit_fn(
            side=side,
            entry_price=entry_price,
            entry_date=entry["entry_date"],
            entry_idx=idx,
            bar=bar,
            bar_idx=j,
            merged=merged,
            peak_price=peak_price,
            peak_gain_pct=peak_gain_pct,
            peak_date=peak_date,
            state=state,
            **exit_kw,
        )
        if hit is not None:
            reason, _ = hit if isinstance(hit, tuple) else (hit, None)
            gain = round(_gain_pct(side, entry_price, close), 4)
            return _trade_row(
                entry, bar["date"], close, gain, peak_gain_pct, peak_date,
                reason, "closed", j - idx,
            )

    last = bars[-1]
    if last["close"] is None:
        return None
    gain = round(_gain_pct(side, entry_price, last["close"]), 4)
    return _trade_row(
        entry, "", last["close"], gain, peak_gain_pct, peak_date,
        "still open", "open", len(bars) - 1 - idx,
    )


def _trade_row(
    entry: dict,
    exit_date: str,
    exit_price: float | str,
    gain_pct: float,
    peak_gain_pct: float,
    peak_date: str,
    exit_reason: str,
    status: str,
    hold_days: int,
) -> dict:
    ed = entry["entry_date"]
    if exit_date:
        hold = max(0, (pd.Timestamp(exit_date) - pd.Timestamp(ed)).days)
    else:
        hold = hold_days
    return {
        "direction": entry["direction"],
        "signal_source": entry["signal_source"],
        "entry_date": ed,
        "entry_price": entry_price if (entry_price := entry["entry_price"]) else None,
        "exit_date": exit_date,
        "exit_price": exit_price if exit_date else "",
        "hold_days": hold,
        "gain_pct": gain_pct,
        "peak_gain_pct": round(peak_gain_pct, 4),
        "peak_date": peak_date,
        "exit_reason": exit_reason,
        "status": status,
    }


# --- Exit rule callbacks (return reason string or None) --------------------

def exit_baseline(**kw) -> Optional[str]:
    reason = _check_baseline_exit(kw["side"], kw["bar"], kw["merged"])
    return (reason, "") if reason else None


def exit_trail_giveback(giveback_pct: float, **kw) -> Optional[str]:
    if kw["peak_gain_pct"] <= 0:
        return None
    close = kw["bar"]["close"]
    cur = _gain_pct(kw["side"], kw["entry_price"], close)
    if cur <= kw["peak_gain_pct"] - giveback_pct:
        return (f"trail giveback {giveback_pct}%", "")
    return None


def exit_take_profit_then_baseline(target_pct: float, **kw) -> Optional[str]:
    side, entry_price, bar = kw["side"], kw["entry_price"], kw["bar"]
    high, low, close = bar.get("high"), bar.get("low"), bar.get("close")
    if side == "BUY" and high is not None:
        if _gain_pct(side, entry_price, high) >= target_pct:
            return (f"take profit {target_pct}%", "")
    elif side == "SELL" and low is not None:
        if _gain_pct(side, entry_price, low) >= target_pct:
            return (f"take profit {target_pct}%", "")
    reason = _check_baseline_exit(side, bar, kw["merged"])
    return (reason, "") if reason else None


def exit_faster_psar(psar_key: str, **kw) -> Optional[str]:
    if kw["bar"]["date"] in kw["merged"]:
        new_side, _ = kw["merged"][kw["bar"]["date"]]
        if new_side != kw["side"]:
            return ("signal reversal", "")
    if _check_psar_exit(kw["side"], kw["bar"], psar_key):
        return (f"faster PSAR ({psar_key})", "")
    return None


def exit_peak_stall(stall_days: int, **kw) -> Optional[str]:
    stalled = (
        pd.Timestamp(kw["bar"]["date"]) - pd.Timestamp(kw["state"]["last_peak_advance"])
    ).days
    if stalled >= stall_days and kw["peak_gain_pct"] > 0:
        return (f"peak stall {stall_days}d", "")
    return None


def exit_combo_trail_time(
    giveback_pct: float,
    max_hold_days: int,
    **kw,
) -> Optional[str]:
    held = (pd.Timestamp(kw["bar"]["date"]) - pd.Timestamp(kw["entry_date"])).days
    if held >= max_hold_days:
        return (f"time stop {max_hold_days}d", "")
    hit = exit_trail_giveback(giveback_pct=giveback_pct, **kw)
    return hit


# ---------------------------------------------------------------------------
# Rule catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    label: str
    exit_fn: Callable
    exit_kw: dict


def build_rule_catalog(best_trail_giveback: float = 0.5) -> list[RuleSpec]:
    rules = [
        RuleSpec("R0", "BASELINE: PSAR flip / signal reversal", exit_baseline, {}),
    ]
    for gb in (0.3, 0.5, 0.7):
        rules.append(RuleSpec(
            f"R1_{gb}",
            f"R1 trail giveback {gb}% from peak",
            exit_trail_giveback,
            {"giveback_pct": gb},
        ))
    for tp in (0.8, 1.2, 1.5):
        rules.append(RuleSpec(
            f"R2_{tp}",
            f"R2 TP {tp}% else baseline",
            exit_take_profit_then_baseline,
            {"target_pct": tp},
        ))
    for key, step in (("psar_03", "0.03"), ("psar_04", "0.04")):
        rules.append(RuleSpec(
            f"R3_{step}",
            f"R3 faster PSAR step {step}/max 0.20",
            exit_faster_psar,
            {"psar_key": key},
        ))
    for n in (2, 3):
        rules.append(RuleSpec(
            f"R4_{n}",
            f"R4 peak stall {n} days",
            exit_peak_stall,
            {"stall_days": n},
        ))
    rules.append(RuleSpec(
        "R5",
        f"R5 combo trail {best_trail_giveback}% + 8d max hold",
        exit_combo_trail_time,
        {"giveback_pct": best_trail_giveback, "max_hold_days": 8},
    ))
    return rules


def _wrap_exit_fn(spec: RuleSpec) -> Callable:
    fn = spec.exit_fn
    kw = spec.exit_kw

    def wrapped(**kwargs) -> Optional[tuple[str, str]]:
        return fn(**kwargs, **kw)

    return wrapped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _half_label(entry_date: str, split_date: pd.Timestamp) -> str:
    return "h1" if pd.Timestamp(entry_date) < split_date else "h2"


def compute_metrics(trades: list[dict], split_date: pd.Timestamp) -> dict[str, dict]:
    """Return metrics for 'all', 'h1', 'h2'."""
    buckets: dict[str, list[dict]] = {"all": [], "h1": [], "h2": []}
    for t in trades:
        buckets["all"].append(t)
        h = _half_label(t["entry_date"], split_date)
        buckets[h].append(t)

    out = {}
    for key, rows in buckets.items():
        out[key] = _metrics_for_rows(rows)
    return out


def _metrics_for_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "win_rate": None,
            "avg_gain": None,
            "avg_loss": None,
            "avg_giveback": None,
            "avg_hold": None,
            "net_after_cost": None,
        }

    closed = [r for r in rows if r["status"] == "closed"]
    gains = [r["gain_pct"] for r in rows]
    givebacks = [
        r["peak_gain_pct"] - r["gain_pct"]
        for r in rows
        if r.get("peak_gain_pct") is not None
    ]
    losses = [g for g in gains if g < 0]
    nets = [r["net_after_cost"] for r in rows]

    return {
        "n": len(rows),
        "win_rate": round(100.0 * sum(1 for g in gains if g > 0) / len(gains), 1),
        "avg_gain": round(sum(gains) / len(gains), 3),
        "avg_loss": round(sum(losses) / len(losses), 3) if losses else 0.0,
        "avg_giveback": round(sum(givebacks) / len(givebacks), 3) if givebacks else 0.0,
        "avg_hold": round(sum(r["hold_days"] for r in rows) / len(rows), 1),
        "net_after_cost": round(sum(nets) / len(nets), 3),
    }


def enrich_trade(trade: dict, pair: dict) -> dict:
    cost = _spread_cost_pct(pair["symbol"], pair["category"], trade["entry_price"])
    trade["net_after_cost"] = round(trade["gain_pct"] - cost, 4)
    trade["display_symbol"] = pair["display_symbol"]
    return trade


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    pairs: list[dict],
    cache_dir: Path,
) -> tuple[pd.Timestamp, dict[str, list[dict]], dict[str, Any]]:
    contexts: dict[str, dict] = {}
    all_entry_dates: list[str] = []

    for pair in pairs:
        sym = pair["symbol"]
        df = fetch.load_cache(sym, cache_dir)
        ctx = prepare_pair(df)
        if ctx is None:
            continue
        contexts[sym] = {**ctx, "pair": pair}
        all_entry_dates.extend(e["entry_date"] for e in ctx["entries"])

    if not all_entry_dates:
        raise SystemExit("No cached pair data. Run: python -m src --backfill")

    split_date = pd.Timestamp(sorted(all_entry_dates)[len(all_entry_dates) // 2])

    results: dict[str, list[dict]] = {}
    rule_trades: dict[str, list[dict]] = {}

    # Pass 1: all rules except R5 (needs best trail from R1)
    base_catalog = build_rule_catalog()
    r1_specs = [r for r in base_catalog if r.rule_id.startswith("R1_")]
    non_r5 = [r for r in base_catalog if r.rule_id != "R5"]

    for spec in non_r5:
        trades = _simulate_all(contexts, spec)
        rule_trades[spec.rule_id] = trades
        results[spec.rule_id] = _row_from_trades(spec, trades, split_date)

    # Best R1 by both-halves net vs baseline
    baseline_net = results["R0"]
    best_trail = 0.5
    best_score = -1e9
    for spec in r1_specs:
        m = results[spec.rule_id]
        score = _both_halves_score(m, baseline_net)
        if score > best_score:
            best_score = score
            best_trail = spec.exit_kw["giveback_pct"]

    r5_spec = build_rule_catalog(best_trail_giveback=best_trail)[-1]
    r5_trades = _simulate_all(contexts, r5_spec)
    rule_trades["R5"] = r5_trades
    results["R5"] = _row_from_trades(r5_spec, r5_trades, split_date)
    results["R5"]["label"] = (
        f"R5 combo trail {best_trail}% + 8d max hold (best R1 trail)"
    )

    return split_date, rule_trades, results


def _simulate_all(contexts: dict[str, dict], spec: RuleSpec) -> list[dict]:
    wrapped = _wrap_exit_fn(spec)
    trades: list[dict] = []
    for sym, ctx in contexts.items():
        pair = ctx["pair"]
        for entry in ctx["entries"]:
            raw = simulate_trade(ctx, entry, wrapped, {})
            if raw is None:
                continue
            trades.append(enrich_trade(raw, pair))
    return trades


def _both_halves_score(metrics_row: dict, baseline: dict) -> float:
    """Prefer higher net on both halves vs baseline."""
    h1_ok = metrics_row["h1_net"] > baseline["h1_net"]
    h2_ok = metrics_row["h2_net"] > baseline["h2_net"]
    if h1_ok and h2_ok:
        return metrics_row["h1_net"] + metrics_row["h2_net"]
    return metrics_row["all_net"] - 1.0  # weak


def _row_from_trades(spec: RuleSpec, trades: list[dict], split_date: pd.Timestamp) -> dict:
    m = compute_metrics(trades, split_date)
    flags = []
    if m["all"]["win_rate"] and m["all"]["win_rate"] > 55:
        flags.append("win>55%")
    if m["all"]["net_after_cost"] and m["all"]["net_after_cost"] > 0.15:
        flags.append("net suspiciously high")

    return {
        "rule_id": spec.rule_id,
        "label": spec.label,
        "flags": flags,
        "all_n": m["all"]["n"],
        "all_win": m["all"]["win_rate"],
        "all_gain": m["all"]["avg_gain"],
        "all_loss": m["all"]["avg_loss"],
        "all_giveback": m["all"]["avg_giveback"],
        "all_hold": m["all"]["avg_hold"],
        "all_net": m["all"]["net_after_cost"],
        "h1_net": m["h1"]["net_after_cost"],
        "h2_net": m["h2"]["net_after_cost"],
        "h1_giveback": m["h1"]["avg_giveback"],
        "h2_giveback": m["h2"]["avg_giveback"],
        "h1_n": m["h1"]["n"],
        "h2_n": m["h2"]["n"],
        "metrics": m,
        "trades": trades,
    }


def _beats_baseline_on_both_halves(row: dict, baseline: dict) -> bool:
    if row["h1_net"] is None or row["h2_net"] is None:
        return False
    if baseline["h1_net"] is None or baseline["h2_net"] is None:
        return False
    net_ok = row["h1_net"] > baseline["h1_net"] and row["h2_net"] > baseline["h2_net"]
    gb_ok = (
        row["h1_giveback"] is not None
        and row["h2_giveback"] is not None
        and baseline["h1_giveback"] is not None
        and row["h1_giveback"] < baseline["h1_giveback"]
        and row["h2_giveback"] < baseline["h2_giveback"]
    )
    return net_ok and gb_ok


def pick_winner(results: dict[str, dict]) -> tuple[str, dict]:
    baseline = results["R0"]
    candidates = [
        r for rid, r in results.items()
        if rid != "R0" and _beats_baseline_on_both_halves(r, baseline)
    ]
    if not candidates:
        # Relax: both halves net improvement only
        candidates = [
            r for rid, r in results.items()
            if rid != "R0"
            and r["h1_net"] is not None
            and r["h2_net"] is not None
            and r["h1_net"] > baseline["h1_net"]
            and r["h2_net"] > baseline["h2_net"]
        ]
    candidates.sort(key=lambda r: (r["all_net"] or -999), reverse=True)
    if candidates:
        w = candidates[0]
        return w["rule_id"], w
    # Fallback: best all_net
    ranked = sorted(
        results.values(),
        key=lambda r: r["all_net"] if r["all_net"] is not None else -999,
        reverse=True,
    )
    return ranked[0]["rule_id"], ranked[0]


def print_comparison_table(results: dict[str, dict], split_date: pd.Timestamp) -> None:
    baseline = results["R0"]
    rows = list(results.values())
    rows.sort(key=lambda r: r["all_net"] if r["all_net"] is not None else -999, reverse=True)

    print(f"\nData split date (entry_date): {split_date.date()}  (older = H1, newer = H2)")
    print(
        "Sorted by net avg gain after spread (all trades). "
        "BASELINE = Rule 0.\n"
    )
    hdr = (
        f"{'Rule':<42} {'Trd':>4} {'Win%':>6} {'AvgGn%':>7} {'AvgLs%':>7} "
        f"{'GiveBk%':>8} {'Hold':>5} {'NetAll%':>8} {'NetH1%':>8} {'NetH2%':>8} "
        f"{'GbH1%':>7} {'GbH2%':>7} {'Flags':<12}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        flag = ",".join(r["flags"])
        if r["rule_id"] == "R0":
            flag = "CONTROL," + flag
        both = _beats_baseline_on_both_halves(r, baseline)
        marker = " *" if both and r["rule_id"] != "R0" else ""
        print(
            f"{r['label'][:42]:<42} {r['all_n']:>4} "
            f"{r['all_win'] or 0:>6.1f} {r['all_gain'] or 0:>+7.3f} {r['all_loss'] or 0:>+7.3f} "
            f"{r['all_giveback'] or 0:>+8.3f} {r['all_hold'] or 0:>5.1f} "
            f"{r['all_net'] or 0:>+8.3f} {r['h1_net'] or 0:>+8.3f} {r['h2_net'] or 0:>+8.3f} "
            f"{r['h1_giveback'] or 0:>+7.3f} {r['h2_giveback'] or 0:>+7.3f} {flag:<12}{marker}"
        )

    print(
        "\n* = beats BASELINE on BOTH halves: higher net after costs AND lower give-back.\n"
        "Flags: win>55% or net>0.15% may be curve-fit / too good to trust.\n"
    )


def export_improved_csv(trades: list[dict], out_path: Path) -> None:
    rows = sorted(trades, key=lambda r: (r["display_symbol"], r["entry_date"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exit rule comparison experiment")
    ap.add_argument("--pairs", default="forex_pairs.json")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "cache"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "out"))
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out)

    print(f"Exit experiment (cached data): {cache_dir.resolve()}")
    split_date, rule_trades, results = run_experiment(pairs, cache_dir)

    print_comparison_table(results, split_date)

    winner_id, winner = pick_winner(results)
    baseline = results["R0"]
    print(f"BASELINE (R0): net all={baseline['all_net']:+.3f}%  "
          f"H1={baseline['h1_net']:+.3f}%  H2={baseline['h2_net']:+.3f}%  "
          f"give-back all={baseline['all_giveback']:+.3f}%")

    print(f"\nWinner (both-halves test): {winner['label']}")
    print(f"  Rule id: {winner_id}")
    print(f"  Net after costs: all={winner['all_net']:+.3f}%  "
          f"H1={winner['h1_net']:+.3f}%  H2={winner['h2_net']:+.3f}%")
    print(f"  Give-back: all={winner['all_giveback']:+.3f}%  "
          f"(baseline {baseline['all_giveback']:+.3f}%)")

    if not _beats_baseline_on_both_halves(winner, baseline):
        print(
            "  WARNING: No rule strictly beats baseline on BOTH halves "
            "(net + give-back). Winner is best overall net; treat with caution."
        )

    out_csv = out_dir / "forex_trades_improved.csv"
    export_improved_csv(rule_trades[winner_id], out_csv)
    print(f"\nExported {len(rule_trades[winner_id])} trades -> {out_csv.resolve()}")


if __name__ == "__main__":
    main()
