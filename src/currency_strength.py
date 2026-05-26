"""
Technical currency strength from per-pair scan leans (snapshot, observational).

Aggregates existing scan/composite lean fields — no new indicator math.
Bullish pair lean -> base +1, quote -1; bearish -> base -1, quote +1.
"""

from __future__ import annotations

from typing import Any


def parse_pair_currencies(display_symbol: str) -> tuple[str, str]:
    """Base (first) and quote (second) from display symbol e.g. EUR/USD."""
    parts = display_symbol.strip().split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid display_symbol for currency parse: {display_symbol!r}")
    return parts[0], parts[1]


def derive_currency_set(pair_entries: list[dict]) -> list[str]:
    """All currencies appearing in the universe, sorted."""
    codes: set[str] = set()
    for p in pair_entries:
        base, quote = parse_pair_currencies(p["display_symbol"])
        codes.add(base)
        codes.add(quote)
    return sorted(codes)


def build_scan_lean_index(payload: dict) -> dict[str, str]:
    index: dict[str, str] = {}
    for s in payload.get("scans", []):
        index[s["scan_id"]] = s.get("lean", "neutral")
    for c in payload.get("composites", []):
        index[c["scan_id"]] = c.get("lean", "neutral")
    return index


def pair_lean_from_scans(
    scan_ids: list[str],
    lean_index: dict[str, str],
) -> tuple[str, int, int, int]:
    """
    Net bullish vs bearish matching scans.
    Returns (pair_lean, bullish_count, bearish_count, neutral_count).
    """
    bullish = bearish = neutral = 0
    for sid in scan_ids:
        lean = lean_index.get(sid, "neutral")
        if lean == "bullish":
            bullish += 1
        elif lean == "bearish":
            bearish += 1
        else:
            neutral += 1
    net = bullish - bearish
    if net > 0:
        return "bullish", bullish, bearish, neutral
    if net < 0:
        return "bearish", bullish, bearish, neutral
    return "neutral", bullish, bearish, neutral


def pair_contributions(pair_lean: str) -> tuple[int, int]:
    """(base_contribution, quote_contribution) for base/quote currencies."""
    if pair_lean == "bullish":
        return 1, -1
    if pair_lean == "bearish":
        return -1, 1
    return 0, 0


def technical_strength_label(normalized: float) -> str:
    if normalized >= 0.15:
        return "strong"
    if normalized <= -0.15:
        return "weak"
    return "neutral"


def build_currency_strength(payload: dict) -> dict[str, Any]:
    """
    Build currency_strength section from scanner payload (pairs + scans + composites).
    """
    pair_entries = payload.get("pairs", [])
    lean_index = build_scan_lean_index(payload)
    trade_date = payload.get("trade_date")

    currencies = derive_currency_set(pair_entries)
    pairs_per_currency: dict[str, set[str]] = {c: set() for c in currencies}
    for p in pair_entries:
        base, quote = parse_pair_currencies(p["display_symbol"])
        pairs_per_currency[base].add(p["display_symbol"])
        pairs_per_currency[quote].add(p["display_symbol"])

    # Per-pair lean and contributions
    pair_rows: list[dict[str, Any]] = []
    for p in pair_entries:
        display = p["display_symbol"]
        base, quote = parse_pair_currencies(display)
        scan_ids = p.get("scan_ids", [])
        pair_lean, b_cnt, s_cnt, n_cnt = pair_lean_from_scans(scan_ids, lean_index)
        base_c, quote_c = pair_contributions(pair_lean)
        pair_rows.append(
            {
                "display_symbol": display,
                "base_currency": base,
                "quote_currency": quote,
                "pair_lean": pair_lean,
                "bullish_scan_count": b_cnt,
                "bearish_scan_count": s_cnt,
                "neutral_scan_count": n_cnt,
                "matching_scan_count": len(scan_ids),
                "base_contribution": base_c,
                "quote_contribution": quote_c,
            }
        )

    currency_data: dict[str, dict[str, Any]] = {}
    for code in currencies:
        raw = 0
        contributions: list[dict[str, Any]] = []
        for row in pair_rows:
            display = row["display_symbol"]
            if display not in pairs_per_currency[code]:
                continue
            if row["base_currency"] == code:
                contrib = row["base_contribution"]
            else:
                contrib = row["quote_contribution"]
            raw += contrib
            contributions.append(
                {
                    "display_symbol": display,
                    "pair_lean": row["pair_lean"],
                    "contribution": contrib,
                    "bullish_scan_count": row["bullish_scan_count"],
                    "bearish_scan_count": row["bearish_scan_count"],
                }
            )
        involved = len(pairs_per_currency[code])
        normalized = raw / involved if involved else 0.0
        currency_data[code] = {
            "technical_strength_score": raw,
            "technical_strength_normalized": round(normalized, 4),
            "pairs_involved": involved,
            "technical_strength_label": technical_strength_label(normalized),
            "contributions": sorted(contributions, key=lambda x: -abs(x["contribution"])),
        }

    ranking = sorted(
        currencies,
        key=lambda c: currency_data[c]["technical_strength_normalized"],
        reverse=True,
    )

    verification = _sign_logic_verification(pair_rows)

    return {
        "methodology": (
            "Technical strength from current scan/composite leans per pair: "
            "net bullish scans -> pair leans bullish (base +1, quote -1); "
            "net bearish -> base -1, quote +1; tied -> neutral (0). "
            "Normalized = technical_strength_score / pairs_involved."
        ),
        "snapshot_note": (
            "Snapshot of technical scan state on trade_date only; "
            "not fundamental strength, not predictive."
        ),
        "trade_date": trade_date,
        "currency_count": len(currencies),
        "sign_logic_verification": verification,
        "by_currency": currency_data,
        "ranking_strongest_to_weakest": ranking,
    }


def _sign_logic_verification(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a non-neutral pair (prefer EUR/JPY) and document base/quote contributions."""
    preferred = "EUR/JPY"
    row = next((r for r in pair_rows if r["display_symbol"] == preferred), None)
    if row is None or row["pair_lean"] == "neutral":
        row = next((r for r in pair_rows if r["pair_lean"] != "neutral"), None)
    if row is None:
        row = pair_rows[0] if pair_rows else None
    if row is None:
        return {"status": "no_pairs", "message": "No pairs to verify sign logic."}

    base, quote = row["base_currency"], row["quote_currency"]
    lean = row["pair_lean"]
    base_c, quote_c = row["base_contribution"], row["quote_contribution"]
    if lean == "bullish":
        rule = "bullish pair: base +1, quote -1"
        expected = {base: 1, quote: -1}
    elif lean == "bearish":
        rule = "bearish pair: base -1, quote +1"
        expected = {base: -1, quote: 1}
    else:
        rule = "neutral pair: base 0, quote 0"
        expected = {base: 0, quote: 0}

    ok = base_c == expected[base] and quote_c == expected[quote]
    return {
        "status": "ok" if ok else "MISMATCH",
        "example_pair": row["display_symbol"],
        "pair_lean": lean,
        "bullish_scan_count": row["bullish_scan_count"],
        "bearish_scan_count": row["bearish_scan_count"],
        "base_currency": base,
        "quote_currency": quote,
        "base_contribution": base_c,
        "quote_contribution": quote_c,
        "rule_applied": rule,
        "check": (
            f"{base} got {base_c:+d} (expected {expected[base]:+d}), "
            f"{quote} got {quote_c:+d} (expected {expected[quote]:+d})"
        ),
    }


def print_currency_strength_summary(currency_strength: dict[str, Any]) -> None:
    """Console ranked table, top/bottom drivers, sign verification trace."""
    by = currency_strength.get("by_currency", {})
    ranking = currency_strength.get("ranking_strongest_to_weakest", [])
    if not ranking:
        print("\n[CURRENCY STRENGTH] No currencies in universe.")
        return

    print("\n[CURRENCY STRENGTH] Technical scan-based (snapshot, not fundamental)")
    print(currency_strength.get("snapshot_note", ""))
    print()
    hdr = f"{'Currency':<8} {'Norm':>7} {'Raw':>5} {'Pairs':>5}  Label"
    print(hdr)
    print("-" * len(hdr))
    for code in ranking:
        c = by[code]
        print(
            f"{code:<8} {c['technical_strength_normalized']:>+7.2f} "
            f"{c['technical_strength_score']:>+5} {c['pairs_involved']:>5}  "
            f"{c['technical_strength_label']}"
        )

    strongest, weakest = ranking[0], ranking[-1]
    for label, code in (("Strongest", strongest), ("Weakest", weakest)):
        c = by[code]
        drivers = [x for x in c["contributions"] if x["contribution"] != 0]
        drivers.sort(key=lambda x: -abs(x["contribution"]))
        print(f"\n{label}: {code} (norm {c['technical_strength_normalized']:+.2f}, raw {c['technical_strength_score']:+d})")
        if not drivers:
            print("  (no directional pair leans — all neutral)")
            continue
        for d in drivers[:8]:
            print(
                f"  {d['display_symbol']}: pair lean {d['pair_lean']}, "
                f"contribution to {code} {d['contribution']:+d} "
                f"(bullish scans {d['bullish_scan_count']}, bearish {d['bearish_scan_count']})"
            )
        if len(drivers) > 8:
            print(f"  ... +{len(drivers) - 8} more pairs")

    ver = currency_strength.get("sign_logic_verification", {})
    print("\nSign logic verification trace:")
    if ver.get("status") == "no_pairs":
        print(f"  {ver.get('message')}")
        return
    print(f"  Example pair: {ver.get('example_pair')} -> pair lean {ver.get('pair_lean')}")
    print(f"  Scans: {ver.get('bullish_scan_count')} bullish, {ver.get('bearish_scan_count')} bearish")
    print(f"  Rule: {ver.get('rule_applied')}")
    print(f"  {ver.get('base_currency')}: contribution {ver.get('base_contribution'):+d}")
    print(f"  {ver.get('quote_currency')}: contribution {ver.get('quote_contribution'):+d}")
    print(f"  Check: {ver.get('check')} [{ver.get('status')}]")
