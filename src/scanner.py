"""
Forex technical scanner — observational conditions on the latest daily bar.

Describes what IS true now (e.g. RSI below 30). Not predictions or trade signals.

  python -m src.scanner
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import currency_strength, fetch, indicators as ind

SCHEMA_VERSION = "1.3.0"
MIN_BARS = 220
EMA_FAST = 12
EMA_SLOW = 26
DONCHIAN = 20
WEEK_HIGH = 252
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_LOOKBACK = 120
BB_SQUEEZE_RECENT_DAYS = 5
PULLBACK_RECENT_DAYS = 2
ATR_PERIOD = 14
HH_LL_BARS = 3

# SMA distance bands (% vs SMA), calibrated on ~39k daily observations across 28
# pairs / 5-year cache: inner = p40 |pct|; outer = p88 positive/negative tail.
# SMA200: no near band; modest = p50 positive distance; extended = p88 (~5.3%).
SMA_BANDS: dict[int, dict[str, float]] = {
    9: {"inner": 0.33, "outer": 1.05},
    20: {"inner": 0.51, "outer": 1.70},
    50: {"inner": 0.81, "outer": 2.70},
    200: {"modest": 2.25, "extended": 5.30},
}
SMA_BAND_RATIONALE = (
    "SMA bands from pooled (close-SMA)/SMA % over all cached bars: "
    "SMA9/20/50 inner=p40 |distance| (hugging ends), outer=p88 tail (stretched); "
    "SMA200 modest=p50 / extended=p88 on positive side (mirrored below); no near band."
)

# Candle pattern detection thresholds (documented in JSON output).
CANDLE_DETECTION: dict[str, float | str] = {
    "wick_to_body_min": 2.0,
    "small_wick_max_body_mult": 1.2,
    "doji_body_max_range_frac": 0.10,
    "marubozu_body_min_range_frac": 0.70,
    "marubozu_wick_max_range_frac": 0.10,
    "hammer_body_max_range_frac": 0.35,
    "hammer_body_position": "upper third (close/open in top 35% of range)",
    "tweezer_match_max_diff_frac": 0.0002,
    "harami_body_inside_prior": True,
    "star_middle_max_body_frac": 0.35,
    "soldiers_min_body_range_frac": 0.50,
    "soldiers_close_progression": True,
}


@dataclass(frozen=True)
class ScanSpec:
    scan_id: str
    name: str
    category: str
    lean: str  # bullish | bearish | neutral
    test: Callable[[dict], bool]


def _pct_vs_sma(close: float, sma_val: float) -> float | None:
    if sma_val is None or pd.isna(sma_val) or sma_val == 0:
        return None
    return (close - sma_val) / sma_val * 100.0


def _cross_above(a_prev, a_now, b_prev, b_now) -> bool:
    if any(pd.isna(x) for x in (a_prev, a_now, b_prev, b_now)):
        return False
    return a_prev <= b_prev and a_now > b_now


def _cross_below(a_prev, a_now, b_prev, b_now) -> bool:
    if any(pd.isna(x) for x in (a_prev, a_now, b_prev, b_now)):
        return False
    return a_prev >= b_prev and a_now < b_now


def _body_ohlc(o, h, l, c) -> tuple[float, float, float, float]:
    body = abs(c - o)
    rng = h - l
    upper = h - max(o, c)
    lower = min(o, c) - l
    return body, rng, upper, lower


def _candle_parts(o: float, h: float, l: float, c: float) -> dict:
    body, rng, upper, lower = _body_ohlc(o, h, l, c)
    top = max(o, c)
    bot = min(o, c)
    return {
        "body": body,
        "range": rng,
        "upper": upper,
        "lower": lower,
        "top": top,
        "bottom": bot,
        "bullish": c > o,
        "bearish": c < o,
    }


def _apply_sma_distance_bands(cond: dict, close: float, sma_val: float, period: int) -> None:
    pct = _pct_vs_sma(close, sma_val)
    if pct is None:
        return

    if period == 200:
        outer = SMA_BANDS[200]["extended"]
        cond["sma200_extended_above"] = pct > outer
        cond["sma200_above"] = 0 < pct <= outer
        cond["sma200_extended_below"] = pct < -outer
        cond["sma200_below"] = -outer <= pct < 0
        return

    inner = SMA_BANDS[period]["inner"]
    outer = SMA_BANDS[period]["outer"]
    p = period
    cond[f"sma{p}_near"] = -inner <= pct <= inner
    cond[f"sma{p}_above"] = inner < pct <= outer
    cond[f"sma{p}_extended_above"] = pct > outer
    cond[f"sma{p}_below"] = -outer <= pct < -inner
    cond[f"sma{p}_extended_below"] = pct < -outer


def build_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)

    out = pd.DataFrame({"date": df["date"], "open": o, "high": h, "low": l, "close": c})

    out["rsi"] = ind.rsi(c, 14)
    sk = ind.stoch_slow_k(h, l, c, 14, 3)
    out["stoch_k"] = sk
    out["stoch_d"] = ind.stoch_d(sk, 3)

    macd_line, sig, hist = ind.macd(c, 12, 26, 9)
    out["macd"] = macd_line
    out["macd_signal"] = sig
    out["macd_hist"] = hist

    for p in (9, 20, 50, 200):
        out[f"sma{p}"] = ind.sma(c, p)

    out["ema_fast"] = ind.ema_spec(c, EMA_FAST)
    out["ema_slow"] = ind.ema_spec(c, EMA_SLOW)

    bb_u, bb_m, bb_l = ind.bollinger(c, BB_PERIOD, BB_STD)
    out["bb_upper"] = bb_u
    out["bb_mid"] = bb_m
    out["bb_lower"] = bb_l
    out["bb_width"] = (bb_u - bb_l) / bb_m.replace(0, np.nan)

    out["atr"] = ind.atr(h, l, c, ATR_PERIOD)
    out["atr_ma20"] = out["atr"].rolling(20, min_periods=20).mean()

    out["range"] = h - l
    out["donchian_high"] = h.rolling(DONCHIAN, min_periods=DONCHIAN).max()
    out["donchian_low"] = l.rolling(DONCHIAN, min_periods=DONCHIAN).min()
    out["high_52w"] = h.rolling(WEEK_HIGH, min_periods=WEEK_HIGH).max()
    out["low_52w"] = l.rolling(WEEK_HIGH, min_periods=WEEK_HIGH).min()

    return out


def _pullback_3bar_up_at(x: pd.DataFrame, end: int) -> bool:
    """3 declining closes ending at `end`, last close still above SMA50 at `end`."""
    if end < 2 or end >= len(x):
        return False
    c0 = float(x.iloc[end]["close"])
    c1 = float(x.iloc[end - 1]["close"])
    c2 = float(x.iloc[end - 2]["close"])
    sma = x.iloc[end]["sma50"]
    if pd.isna(sma) or pd.isna(c0) or pd.isna(c1) or pd.isna(c2):
        return False
    sma = float(sma)
    return c0 < c1 < c2 and c0 > sma


def _pullback_3bar_down_at(x: pd.DataFrame, end: int) -> bool:
    """3 rising closes ending at `end`, last close still below SMA50 at `end`."""
    if end < 2 or end >= len(x):
        return False
    c0 = float(x.iloc[end]["close"])
    c1 = float(x.iloc[end - 1]["close"])
    c2 = float(x.iloc[end - 2]["close"])
    sma = x.iloc[end]["sma50"]
    if pd.isna(sma) or pd.isna(c0) or pd.isna(c1) or pd.isna(c2):
        return False
    sma = float(sma)
    return c0 > c1 > c2 and c0 < sma


def evaluate_conditions(x: pd.DataFrame) -> dict[str, bool]:
    """Boolean condition flags for the latest bar (index -1) vs prior (-2)."""
    if len(x) < 3:
        return {}

    i, p = -1, -2
    row = x.iloc[i]
    prev = x.iloc[p]
    close = float(row["close"])
    prev_close = float(prev["close"])

    def g(col):
        return float(row[col]) if pd.notna(row[col]) else np.nan

    def gp(col):
        return float(prev[col]) if pd.notna(prev[col]) else np.nan

    o, h, l, c = g("open"), g("high"), g("low"), g("close")
    po, ph, pl, pc = gp("open"), gp("high"), gp("low"), gp("close")
    body, rng, upper_wick, lower_wick = _body_ohlc(o, h, l, c)

    rsi = g("rsi")
    rsi_p = gp("rsi")
    sk, sd = g("stoch_k"), g("stoch_d")
    sk_p, sd_p = gp("stoch_k"), gp("stoch_d")
    macd, msig, mhist = g("macd"), g("macd_signal"), g("macd_hist")
    macd_p, msig_p = gp("macd"), gp("macd_signal")

    cond: dict[str, bool] = {}

    # --- Momentum / oscillators ---
    cond["rsi_lt_30"] = rsi < 30
    cond["rsi_lt_15"] = rsi < 15
    cond["rsi_gt_70"] = rsi > 70
    cond["rsi_gt_85"] = rsi > 85
    cond["rsi_cross_above_50"] = _cross_above(rsi_p, rsi, 50, 50)
    cond["rsi_cross_below_50"] = _cross_below(rsi_p, rsi, 50, 50)
    cond["stoch_lt_20"] = sk < 20
    cond["stoch_gt_80"] = sk > 80
    cond["stoch_oversold_turning_up"] = sk < 20 and _cross_above(sk_p, sk, sd_p, sd)
    cond["stoch_overbought_turning_down"] = sk > 80 and _cross_below(sk_p, sk, sd_p, sd)

    cond["macd_bull_cross"] = _cross_above(macd_p, macd, msig_p, msig)
    cond["macd_bear_cross"] = _cross_below(macd_p, macd, msig_p, msig)
    cond["macd_zero_cross_up"] = _cross_above(macd_p, macd, 0, 0)
    cond["macd_zero_cross_down"] = _cross_below(macd_p, macd, 0, 0)

    # --- SMA distance bands (word labels; % only in JSON thresholds) ---
    for period in (9, 20, 50, 200):
        _apply_sma_distance_bands(cond, close, g(f"sma{period}"), period)

    # --- SMA / EMA crosses ---
    pairs_cross = [
        ("sma9", "sma20", "sma9_cross_above_sma20", "sma9_cross_below_sma20"),
        ("sma20", "sma50", "sma20_cross_above_sma50", "sma20_cross_below_sma50"),
        ("sma50", "sma200", "golden_cross", "death_cross"),
    ]
    for a, b, up_id, dn_id in pairs_cross:
        cond[up_id] = _cross_above(gp(a), g(a), gp(b), g(b))
        cond[dn_id] = _cross_below(gp(a), g(a), gp(b), g(b))

    for period in (20, 50, 200):
        cond[f"price_cross_above_sma{period}"] = _cross_above(
            prev_close, close, gp(f"sma{period}"), g(f"sma{period}")
        )
        cond[f"price_cross_below_sma{period}"] = _cross_below(
            prev_close, close, gp(f"sma{period}"), g(f"sma{period}")
        )

    sma50_now, sma50_5ago = g("sma50"), float(x["sma50"].iloc[-6]) if len(x) >= 6 else np.nan
    if not pd.isna(sma50_now) and not pd.isna(sma50_5ago):
        cond["sma50_rising"] = sma50_now > sma50_5ago
        cond["sma50_falling"] = sma50_now < sma50_5ago
    else:
        cond["sma50_rising"] = False
        cond["sma50_falling"] = False

    cond["ema_fast_cross_above_slow"] = _cross_above(
        gp("ema_fast"), g("ema_fast"), gp("ema_slow"), g("ema_slow")
    )
    cond["ema_fast_cross_below_slow"] = _cross_below(
        gp("ema_fast"), g("ema_fast"), gp("ema_slow"), g("ema_slow")
    )

    # --- Volatility / range ---
    bb_w = x["bb_width"].dropna()
    if len(bb_w) >= BB_SQUEEZE_LOOKBACK:
        width_now = g("bb_width")
        cond["bb_squeeze_now"] = width_now <= bb_w.tail(BB_SQUEEZE_LOOKBACK).min() * 1.02
        recent = x["bb_width"].iloc[-BB_SQUEEZE_RECENT_DAYS:]
        cond["bb_squeeze_recent"] = (recent <= bb_w.tail(BB_SQUEEZE_LOOKBACK).min() * 1.02).any()
    else:
        cond["bb_squeeze_now"] = False
        cond["bb_squeeze_recent"] = False

    cond["bb_breakout_up"] = close > g("bb_upper")
    cond["bb_breakout_down"] = close < g("bb_lower")

    atr_n, atr_ma = g("atr"), g("atr_ma20")
    if not pd.isna(atr_n) and not pd.isna(atr_ma) and atr_ma > 0:
        cond["atr_expansion"] = atr_n > atr_ma * 1.1
        cond["atr_contraction"] = atr_n < atr_ma * 0.9
    else:
        cond["atr_expansion"] = False
        cond["atr_contraction"] = False

    cond["donchian_high_20"] = close >= g("donchian_high")
    cond["donchian_low_20"] = close <= g("donchian_low")
    cond["high_52w"] = close >= g("high_52w")
    cond["low_52w"] = close <= g("low_52w")

    cond["inside_day"] = h < ph and l > pl
    cond["outside_day"] = h > ph and l < pl

    r = x["range"].iloc[-7:]
    if len(r) >= 4:
        cond["nr4"] = row["range"] <= r.tail(4).min() * 1.001
    else:
        cond["nr4"] = False
    if len(r) >= 7:
        cond["nr7"] = row["range"] <= r.tail(7).min() * 1.001
    else:
        cond["nr7"] = False

    # --- Candles / price action ---
    n = HH_LL_BARS
    highs = x["high"].iloc[-n:].values
    lows = x["low"].iloc[-n:].values
    cond["higher_highs"] = len(highs) >= 2 and all(
        highs[j] > highs[j - 1] for j in range(1, len(highs))
    )
    cond["lower_lows"] = len(lows) >= 2 and all(
        lows[j] < lows[j - 1] for j in range(1, len(lows))
    )
    if len(x) >= 2:
        cond["hh_hl"] = row["high"] > prev["high"] and row["low"] > prev["low"]
        cond["lh_ll"] = row["high"] < prev["high"] and row["low"] < prev["low"]
    else:
        cond["hh_hl"] = cond["lh_ll"] = False

    prev_bear = pc < po
    today_bull = c > o
    prev_bull = pc > po
    today_bear = c < o
    wick_mult = float(CANDLE_DETECTION["wick_to_body_min"])
    small_wick = float(CANDLE_DETECTION["small_wick_max_body_mult"])
    doji_frac = float(CANDLE_DETECTION["doji_body_max_range_frac"])
    maru_body = float(CANDLE_DETECTION["marubozu_body_min_range_frac"])
    maru_wick = float(CANDLE_DETECTION["marubozu_wick_max_range_frac"])
    hammer_top_frac = float(CANDLE_DETECTION["hammer_body_max_range_frac"])
    tweezer_frac = float(CANDLE_DETECTION["tweezer_match_max_diff_frac"])
    star_mid = float(CANDLE_DETECTION["star_middle_max_body_frac"])
    soldier_body = float(CANDLE_DETECTION["soldiers_min_body_range_frac"])

    cond["bullish_engulfing"] = (
        prev_bear and today_bull and o <= pc and c >= po and body > abs(pc - po) * 0.9
    )
    cond["bearish_engulfing"] = (
        prev_bull and today_bear and o >= pc and c <= po and body > abs(pc - po) * 0.9
    )
    cond["doji"] = rng > 0 and body <= rng * doji_frac

    # Hammer: long lower wick, body in upper part of range
    body_top = max(o, c)
    cond["hammer"] = (
        rng > 0
        and lower_wick >= wick_mult * max(body, 1e-12)
        and upper_wick <= body * small_wick
        and (h - body_top) / rng <= hammer_top_frac
    )

    # Shooting star: long upper wick, body low in range, bearish bias
    body_bot = min(o, c)
    cond["shooting_star"] = (
        rng > 0
        and upper_wick >= wick_mult * max(body, 1e-12)
        and lower_wick <= body * small_wick
        and (body_bot - l) / rng <= hammer_top_frac
        and c <= o + rng * 0.35
    )

    # Inverted hammer: long upper wick, small body at bottom (bullish shape, not star)
    cond["inverted_hammer"] = (
        rng > 0
        and upper_wick >= wick_mult * max(body, 1e-12)
        and lower_wick <= body * small_wick
        and (body_bot - l) / rng <= hammer_top_frac
        and c >= o - rng * 0.05
    )

    cond["marubozu_bullish"] = (
        rng > 0 and c > o and body >= rng * maru_body
        and upper_wick <= rng * maru_wick and lower_wick <= rng * maru_wick
    )
    cond["marubozu_bearish"] = (
        rng > 0 and c < o and body >= rng * maru_body
        and upper_wick <= rng * maru_wick and lower_wick <= rng * maru_wick
    )

    if len(x) >= 2:
        h1, h2 = float(row["high"]), float(prev["high"])
        l1, l2 = float(row["low"]), float(prev["low"])
        ref_h = max(h1, h2, 1e-12)
        ref_l = max(l1, l2, 1e-12)
        cond["tweezer_top"] = abs(h1 - h2) / ref_h <= tweezer_frac
        cond["tweezer_bottom"] = abs(l1 - l2) / ref_l <= tweezer_frac
    else:
        cond["tweezer_top"] = cond["tweezer_bottom"] = False

    prev_top, prev_bot = max(po, pc), min(po, pc)
    today_top, today_bot = max(o, c), min(o, c)
    cond["bullish_harami"] = (
        pc < po and c > o
        and today_top <= prev_top and today_bot >= prev_bot
        and body < abs(pc - po) * 0.95
    )
    cond["bearish_harami"] = (
        pc > po and c < o
        and today_top <= prev_top and today_bot >= prev_bot
        and body < abs(pc - po) * 0.95
    )

    if len(x) >= 3:
        b0 = x.iloc[i - 2]
        b1 = x.iloc[i - 1]
        b2 = row
        c0 = _candle_parts(float(b0["open"]), float(b0["high"]), float(b0["low"]), float(b0["close"]))
        c1 = _candle_parts(float(b1["open"]), float(b1["high"]), float(b1["low"]), float(b1["close"]))
        c2 = _candle_parts(float(b2["open"]), float(b2["high"]), float(b2["low"]), float(b2["close"]))
        mid0 = (float(b0["open"]) + float(b0["close"])) / 2

        cond["morning_star"] = (
            c0["bearish"]
            and c1["range"] > 0
            and c1["body"] / c1["range"] <= star_mid
            and c2["bullish"]
            and float(b2["close"]) > mid0
        )
        cond["evening_star"] = (
            c0["bullish"]
            and c1["range"] > 0
            and c1["body"] / c1["range"] <= star_mid
            and c2["bearish"]
            and float(b2["close"]) < mid0
        )

        soldiers = [x.iloc[i - 2], x.iloc[i - 1], row]
        bull_ok = True
        bear_ok = True
        for j, bar in enumerate(soldiers):
            oo, hh, ll, cc = (
                float(bar["open"]),
                float(bar["high"]),
                float(bar["low"]),
                float(bar["close"]),
            )
            br, rng_j = abs(cc - oo), hh - ll
            if rng_j <= 0 or br < rng_j * soldier_body:
                bull_ok = bear_ok = False
                break
            if cc <= oo:
                bull_ok = False
            if cc >= oo:
                bear_ok = False
            if j > 0:
                pcc = float(soldiers[j - 1]["close"])
                if cc <= pcc:
                    bull_ok = False
                if cc >= pcc:
                    bear_ok = False
        cond["three_white_soldiers"] = bull_ok
        cond["three_black_crows"] = bear_ok
    else:
        cond["morning_star"] = cond["evening_star"] = False
        cond["three_white_soldiers"] = cond["three_black_crows"] = False

    n = len(x)
    last = n - 1
    cond["pullback_3bar_up"] = _pullback_3bar_up_at(x, last)
    cond["pullback_3bar_down"] = _pullback_3bar_down_at(x, last)
    cond["pullback_3bar_up_recent"] = any(
        _pullback_3bar_up_at(x, last - k)
        for k in range(min(PULLBACK_RECENT_DAYS, n))
        if last - k >= 2
    )
    cond["pullback_3bar_down_recent"] = any(
        _pullback_3bar_down_at(x, last - k)
        for k in range(min(PULLBACK_RECENT_DAYS, n))
        if last - k >= 2
    )

    # --- Weekend gaps (forex) ---
    dates = pd.to_datetime(x["date"])
    gap_up = gap_down = gap_fill_up = gap_fill_down = False
    if len(dates) >= 2:
        gap_days = (dates.iloc[i] - dates.iloc[i - 1]).days
        if gap_days >= 2:
            gap_pct = (o - pc) / pc * 100 if pc else 0
            if gap_pct > 0.15:
                gap_up = True
            elif gap_pct < -0.15:
                gap_down = True
        # Gap fill: prior bar had weekend gap, today revisits pre-gap close
        if len(dates) >= 3:
            gap_days2 = (dates.iloc[i - 1] - dates.iloc[i - 2]).days
            if gap_days2 >= 2:
                g_open = float(x["open"].iloc[i - 1])
                g_pc = float(x["close"].iloc[i - 2])
                if g_pc and (g_open - g_pc) / g_pc * 100 > 0.15 and close <= g_pc:
                    gap_fill_down = True
                if g_pc and (g_open - g_pc) / g_pc * 100 < -0.15 and close >= g_pc:
                    gap_fill_up = True

    cond["weekend_gap_up"] = gap_up
    cond["weekend_gap_down"] = gap_down
    cond["gap_fill_up"] = gap_fill_up
    cond["gap_fill_down"] = gap_fill_down

    return cond


def base_scan_catalog() -> list[ScanSpec]:
    """All single-condition scans."""
    specs: list[tuple[str, str, str, str, str]] = [
        # id, name, category, lean, condition_key
        ("rsi_oversold", "RSI below 30 (oversold)", "momentum", "bullish", "rsi_lt_30"),
        ("rsi_deep_oversold", "RSI below 15 (deeply oversold)", "momentum", "bullish", "rsi_lt_15"),
        ("rsi_overbought", "RSI above 70 (overbought)", "momentum", "bearish", "rsi_gt_70"),
        ("rsi_extreme_overbought", "RSI above 85 (extremely overbought)", "momentum", "bearish", "rsi_gt_85"),
        ("rsi_cross_above_50", "RSI crossed above 50", "momentum", "bullish", "rsi_cross_above_50"),
        ("rsi_cross_below_50", "RSI crossed below 50", "momentum", "bearish", "rsi_cross_below_50"),
        ("stoch_oversold", "Slow Stochastic below 20", "momentum", "bullish", "stoch_lt_20"),
        ("stoch_overbought", "Slow Stochastic above 80", "momentum", "bearish", "stoch_gt_80"),
        ("stoch_oversold_turning_up", "Oversold Stochastic turning up (%K above %D)", "momentum", "bullish", "stoch_oversold_turning_up"),
        ("stoch_overbought_turning_down", "Overbought Stochastic turning down (%K below %D)", "momentum", "bearish", "stoch_overbought_turning_down"),
        ("macd_bull_cross", "MACD bullish crossover", "macd", "bullish", "macd_bull_cross"),
        ("macd_bear_cross", "MACD bearish crossover", "macd", "bearish", "macd_bear_cross"),
        ("macd_zero_cross_up", "MACD zero-line cross up", "macd", "bullish", "macd_zero_cross_up"),
        ("macd_zero_cross_down", "MACD zero-line cross down", "macd", "bearish", "macd_zero_cross_down"),
    ]

    for period in (9, 20, 50):
        specs.extend([
            (f"sma{period}_near", f"Near SMA {period}", "trend", "neutral", f"sma{period}_near"),
            (f"sma{period}_above", f"Above SMA {period}", "trend", "bullish", f"sma{period}_above"),
            (
                f"sma{period}_extended_above",
                f"Extended above SMA {period}",
                "trend",
                "bullish",
                f"sma{period}_extended_above",
            ),
            (f"sma{period}_below", f"Below SMA {period}", "trend", "bearish", f"sma{period}_below"),
            (
                f"sma{period}_extended_below",
                f"Extended below SMA {period}",
                "trend",
                "bearish",
                f"sma{period}_extended_below",
            ),
        ])
    specs.extend([
        ("sma200_above", "Above SMA 200", "trend", "bullish", "sma200_above"),
        ("sma200_extended_above", "Extended above SMA 200", "trend", "bullish", "sma200_extended_above"),
        ("sma200_below", "Below SMA 200", "trend", "bearish", "sma200_below"),
        ("sma200_extended_below", "Extended below SMA 200", "trend", "bearish", "sma200_extended_below"),
    ])

    candle_scans = [
        ("doji", "Doji (indecision)", "doji_body_max_range_frac"),
        ("hammer", "Hammer (long lower wick rejection)", "wick_to_body_min"),
        ("inverted_hammer", "Inverted hammer (long upper wick, body low)", "wick_to_body_min"),
        ("shooting_star", "Shooting star (long upper wick rejection)", "wick_to_body_min"),
        ("bullish_engulfing", "Bullish engulfing", None),
        ("bearish_engulfing", "Bearish engulfing", None),
        ("marubozu_bullish", "Marubozu bullish (full body, minimal wicks)", "marubozu_body_min_range_frac"),
        ("marubozu_bearish", "Marubozu bearish (full body, minimal wicks)", "marubozu_body_min_range_frac"),
        ("tweezer_top", "Tweezer top (matched highs)", "tweezer_match_max_diff_frac"),
        ("tweezer_bottom", "Tweezer bottom (matched lows)", "tweezer_match_max_diff_frac"),
        ("bullish_harami", "Bullish harami", None),
        ("bearish_harami", "Bearish harami", None),
        ("morning_star", "Morning star (3-candle)", "star_middle_max_body_frac"),
        ("evening_star", "Evening star (3-candle)", "star_middle_max_body_frac"),
        ("three_white_soldiers", "Three white soldiers", "soldiers_min_body_range_frac"),
        ("three_black_crows", "Three black crows", "soldiers_min_body_range_frac"),
    ]

    specs.extend([
        ("sma9_cross_above_sma20", "SMA 9 crossed above SMA 20", "trend", "bullish", "sma9_cross_above_sma20"),
        ("sma9_cross_below_sma20", "SMA 9 crossed below SMA 20", "trend", "bearish", "sma9_cross_below_sma20"),
        ("sma20_cross_above_sma50", "SMA 20 crossed above SMA 50", "trend", "bullish", "sma20_cross_above_sma50"),
        ("sma20_cross_below_sma50", "SMA 20 crossed below SMA 50", "trend", "bearish", "sma20_cross_below_sma50"),
        ("golden_cross", "SMA 50 crossed above SMA 200 (golden cross)", "trend", "bullish", "golden_cross"),
        ("death_cross", "SMA 50 crossed below SMA 200 (death cross)", "trend", "bearish", "death_cross"),
        ("price_cross_above_sma20", "Price crossed above SMA 20", "trend", "bullish", "price_cross_above_sma20"),
        ("price_cross_below_sma20", "Price crossed below SMA 20", "trend", "bearish", "price_cross_below_sma20"),
        ("price_cross_above_sma50", "Price crossed above SMA 50", "trend", "bullish", "price_cross_above_sma50"),
        ("price_cross_below_sma50", "Price crossed below SMA 50", "trend", "bearish", "price_cross_below_sma50"),
        ("price_cross_above_sma200", "Price crossed above SMA 200", "trend", "bullish", "price_cross_above_sma200"),
        ("price_cross_below_sma200", "Price crossed below SMA 200", "trend", "bearish", "price_cross_below_sma200"),
        ("sma50_rising", "SMA 50 rising (5-day slope up)", "trend", "bullish", "sma50_rising"),
        ("sma50_falling", "SMA 50 falling (5-day slope down)", "trend", "bearish", "sma50_falling"),
        ("ema_fast_cross_above_slow", "EMA fast crossed above EMA slow", "trend", "bullish", "ema_fast_cross_above_slow"),
        ("ema_fast_cross_below_slow", "EMA fast crossed below EMA slow", "trend", "bearish", "ema_fast_cross_below_slow"),
        ("bb_squeeze", "Bollinger Band squeeze (width at ~6-month low)", "volatility", "neutral", "bb_squeeze_now"),
        ("bb_breakout_up", "Close above upper Bollinger Band", "volatility", "bullish", "bb_breakout_up"),
        ("bb_breakout_down", "Close below lower Bollinger Band", "volatility", "bearish", "bb_breakout_down"),
        ("atr_expansion", "ATR above recent average (expansion)", "volatility", "neutral", "atr_expansion"),
        ("atr_contraction", "ATR below recent average (contraction)", "volatility", "neutral", "atr_contraction"),
        ("donchian_high_20", "New 20-day high (Donchian)", "volatility", "bullish", "donchian_high_20"),
        ("donchian_low_20", "New 20-day low (Donchian)", "volatility", "bearish", "donchian_low_20"),
        ("high_52w", "At or above 52-week high", "volatility", "bullish", "high_52w"),
        ("low_52w", "At or below 52-week low", "volatility", "bearish", "low_52w"),
        ("inside_day", "Inside day", "volatility", "neutral", "inside_day"),
        ("outside_day", "Outside day", "volatility", "neutral", "outside_day"),
        ("nr4", "NR4 (narrowest range in 4 days)", "volatility", "neutral", "nr4"),
        ("nr7", "NR7 (narrowest range in 7 days)", "volatility", "neutral", "nr7"),
        ("higher_highs_3", "Higher highs (3 bars)", "price_action", "bullish", "higher_highs"),
        ("lower_lows_3", "Lower lows (3 bars)", "price_action", "bearish", "lower_lows"),
        ("hh_hl", "Higher high and higher low", "price_action", "bullish", "hh_hl"),
        ("lh_ll", "Lower high and lower low", "price_action", "bearish", "lh_ll"),
        ("pullback_3bar_uptrend", "3-bar pullback in uptrend (above SMA 50)", "price_action", "bullish", "pullback_3bar_up"),
        ("pullback_3bar_downtrend", "3-bar pullback in downtrend (below SMA 50)", "price_action", "bearish", "pullback_3bar_down"),
        ("weekend_gap_up", "Weekend gap up (Monday open vs Friday close)", "gaps", "bullish", "weekend_gap_up"),
        ("weekend_gap_down", "Weekend gap down", "gaps", "bearish", "weekend_gap_down"),
        ("gap_fill_up", "Gap fill up (price back through pre-gap close)", "gaps", "bullish", "gap_fill_up"),
        ("gap_fill_down", "Gap fill down", "gaps", "bearish", "gap_fill_down"),
    ])

    for scan_id, name, thresh_key in candle_scans:
        lean = "bullish" if scan_id in (
            "hammer", "inverted_hammer", "bullish_engulfing", "marubozu_bullish",
            "tweezer_bottom", "bullish_harami", "morning_star", "three_white_soldiers",
        ) else "bearish" if scan_id in (
            "shooting_star", "bearish_engulfing", "marubozu_bearish", "tweezer_top",
            "bearish_harami", "evening_star", "three_black_crows",
        ) else "neutral"
        specs.append((scan_id, name, "price_action", lean, scan_id))

    out: list[ScanSpec] = []
    for scan_id, name, cat, lean, key in specs:
        out.append(ScanSpec(
            scan_id=scan_id,
            name=name,
            category=cat,
            lean=lean,
            test=lambda c, k=key: c.get(k, False),
        ))
    return out


def _any(c: dict, keys: tuple[str, ...]) -> bool:
    return any(c.get(k, False) for k in keys)


def _all(c: dict, keys: tuple[str, ...]) -> bool:
    return all(c.get(k, False) for k in keys)


@dataclass(frozen=True)
class CompositeDef:
    scan_id: str
    name: str
    description: str
    lean: str
    match: Callable[[dict], bool]
    quality: Callable[[dict], str | None] | None = None


def _oversold_reversal_quality(c: dict) -> str:
    return "in_downtrend" if c.get("sma50_falling") else "clean"


def _weekend_gap_fade_quality(c: dict) -> str | None:
    if c.get("weekend_gap_up"):
        return "fade_gap_up"
    if c.get("weekend_gap_down"):
        return "fade_gap_down"
    return None


def _weekend_gap_fade_match(c: dict) -> bool:
    if not _any(c, ("weekend_gap_up", "weekend_gap_down")):
        return False
    fill_or_reversal = _any(c, ("gap_fill_up", "gap_fill_down"))
    if c.get("weekend_gap_up"):
        fill_or_reversal = fill_or_reversal or _any(
            c,
            ("gap_fill_down", "bearish_engulfing", "shooting_star", "tweezer_top"),
        )
    if c.get("weekend_gap_down"):
        fill_or_reversal = fill_or_reversal or _any(
            c,
            ("gap_fill_up", "bullish_engulfing", "hammer", "tweezer_bottom"),
        )
    return fill_or_reversal


COMPOSITE_CATALOG: list[CompositeDef] = [
    CompositeDef(
        "composite_volatility_squeeze",
        "Volatility Squeeze",
        "Compression resolving upward: narrow range, volatility expanding, price pushing higher.",
        "bullish",
        lambda c: (
            _any(c, ("nr4", "nr7", "inside_day"))
            and c.get("atr_expansion")
            and _any(c, ("donchian_high_20", "bb_breakout_up", "bullish_engulfing"))
        ),
    ),
    CompositeDef(
        "composite_uptrend_squeeze_breakout",
        "Uptrend Squeeze Breakout",
        "Strong uptrend: Bollinger squeeze within the last 5 sessions, then close above the upper band today.",
        "bullish",
        lambda c: (
            _any(c, ("sma50_extended_above", "sma200_extended_above"))
            and c.get("sma50_rising")
            and c.get("bb_squeeze_recent")
            and c.get("bb_breakout_up")
        ),
    ),
    CompositeDef(
        "composite_oversold_reversal",
        "Oversold Reversal",
        "Stretched below moving averages and oversold, with a reversal candle forming.",
        "bullish",
        lambda c: (
            _any(c, ("sma20_extended_below", "sma50_extended_below"))
            and _any(c, ("rsi_lt_30", "stoch_lt_20"))
            and _any(
                c,
                (
                    "stoch_oversold_turning_up",
                    "bullish_engulfing",
                    "hammer",
                    "outside_day",
                ),
            )
        ),
        quality=_oversold_reversal_quality,
    ),
    CompositeDef(
        "composite_trend_pullback_long",
        "Trend Pullback — Long",
        "Uptrend: 3-bar pullback in the last 2 sessions, momentum turn up today (stochastic or RSI).",
        "bullish",
        lambda c: (
            c.get("sma50_rising")
            and c.get("pullback_3bar_up_recent")
            and _any(c, ("stoch_oversold_turning_up", "rsi_cross_above_50"))
        ),
    ),
    CompositeDef(
        "composite_trend_pullback_short",
        "Trend Pullback — Short",
        "Downtrend: 3-bar pullback in the last 2 sessions, momentum turn down today (stochastic or RSI).",
        "bearish",
        lambda c: (
            c.get("sma50_falling")
            and c.get("pullback_3bar_down_recent")
            and _any(c, ("stoch_overbought_turning_down", "rsi_cross_below_50"))
        ),
    ),
    CompositeDef(
        "composite_breakdown_expansion",
        "Breakdown Expansion",
        "Breaking down out of a weak regime with volatility expanding.",
        "bearish",
        lambda c: (
            _any(c, ("sma200_below", "sma200_extended_below"))
            and _any(c, ("bb_breakout_down", "donchian_low_20"))
            and c.get("atr_expansion")
            and _any(c, ("bearish_engulfing", "marubozu_bearish", "shooting_star"))
        ),
    ),
    CompositeDef(
        "composite_failed_breakout",
        "Failed Breakout / Bull Trap",
        "Pushed to new highs, overbought, then showing reversal candles.",
        "bearish",
        lambda c: (
            _any(c, ("donchian_high_20", "sma200_extended_above", "bb_breakout_up"))
            and _any(c, ("rsi_gt_70", "stoch_overbought_turning_down"))
            and _any(
                c,
                ("bearish_engulfing", "shooting_star", "outside_day", "tweezer_top"),
            )
        ),
    ),
    CompositeDef(
        "composite_golden_cross",
        "Golden Cross Regime Shift",
        "Long-term trend flips up with momentum confirming.",
        "bullish",
        lambda c: (
            c.get("golden_cross")
            and c.get("sma50_rising")
            and _any(c, ("macd_bull_cross", "macd_zero_cross_up"))
        ),
    ),
    CompositeDef(
        "composite_weekend_gap_fade",
        "Weekend Gap Fade",
        "Weekend gap with early signs of fill or reversal against the gap direction.",
        "neutral",
        _weekend_gap_fade_match,
        quality=_weekend_gap_fade_quality,
    ),
]

# Base-scan mapping notes (no silent inventions — printed after each run).
COMPOSITE_SUBSTITUTIONS = [
    "Trend Pullback — Long: pullback uses pullback_3bar_up_recent (last 2 sessions); turn-up today only.",
    "Trend Pullback — Short: pullback uses pullback_3bar_down_recent (last 2 sessions); turn-down today only.",
    "Uptrend Squeeze Breakout: squeeze uses bb_squeeze_recent (last 5 sessions), breakout up on today only.",
    "Breakdown Expansion: 'close in lower part of range' -> bearish_engulfing OR marubozu_bearish OR shooting_star.",
    "Failed Breakout: 'Donchian breakout up' -> donchian_high_20 and/or bb_breakout_up.",
    "Weekend Gap Fade: reversal vs gap uses gap-fill scans plus bearish/bullish candle scans per gap direction.",
]


CANDLE_SCAN_THRESHOLDS: dict[str, str | float | None] = {
    "doji": CANDLE_DETECTION["doji_body_max_range_frac"],
    "hammer": CANDLE_DETECTION["wick_to_body_min"],
    "inverted_hammer": CANDLE_DETECTION["wick_to_body_min"],
    "shooting_star": CANDLE_DETECTION["wick_to_body_min"],
    "bullish_engulfing": "prior bearish body engulfed by bullish candle",
    "bearish_engulfing": "prior bullish body engulfed by bearish candle",
    "marubozu_bullish": CANDLE_DETECTION["marubozu_body_min_range_frac"],
    "marubozu_bearish": CANDLE_DETECTION["marubozu_body_min_range_frac"],
    "tweezer_top": CANDLE_DETECTION["tweezer_match_max_diff_frac"],
    "tweezer_bottom": CANDLE_DETECTION["tweezer_match_max_diff_frac"],
    "bullish_harami": "today body inside prior bearish body",
    "bearish_harami": "today body inside prior bullish body",
    "morning_star": CANDLE_DETECTION["star_middle_max_body_frac"],
    "evening_star": CANDLE_DETECTION["star_middle_max_body_frac"],
    "three_white_soldiers": CANDLE_DETECTION["soldiers_min_body_range_frac"],
    "three_black_crows": CANDLE_DETECTION["soldiers_min_body_range_frac"],
}


def run_scanner(
    pairs: list[dict],
    cache_dir: Path,
) -> tuple[dict, list[str]]:
    """Returns (payload dict, skipped display symbols)."""
    pair_conditions: dict[str, dict] = {}
    pair_meta: dict[str, dict] = {}
    skipped: list[str] = []
    trade_date = None

    for pair in pairs:
        sym = pair["symbol"]
        display = pair["display_symbol"]
        df = fetch.load_cache(sym, cache_dir)
        if len(df) < MIN_BARS:
            skipped.append(display)
            continue
        indf = build_indicator_frame(df)
        cond = evaluate_conditions(indf)
        if not cond:
            skipped.append(display)
            continue
        pair_conditions[display] = cond
        pair_meta[display] = {"symbol": sym, "category": pair["category"]}
        td = pd.Timestamp(indf["date"].iloc[-1]).strftime("%Y-%m-%d")
        if trade_date is None or td > trade_date:
            trade_date = td

    universe = len(pair_conditions)
    base_scans = base_scan_catalog()

    scan_results: list[dict] = []
    pair_to_scans: dict[str, list[str]] = {d: [] for d in pair_conditions}

    for spec in base_scans:
        matches = [d for d, c in pair_conditions.items() if spec.test(c)]
        for d in matches:
            pair_to_scans[d].append(spec.scan_id)
        low_info = len(matches) == 0 or len(matches) == universe
        entry = {
            "scan_id": spec.scan_id,
            "name": spec.name,
            "category": spec.category,
            "lean": spec.lean,
            "pairs": sorted(matches),
            "match_count": len(matches),
            "low_information": low_info,
        }
        if spec.scan_id in CANDLE_SCAN_THRESHOLDS:
            entry["detection_threshold"] = CANDLE_SCAN_THRESHOLDS[spec.scan_id]
        scan_results.append(entry)

    composite_results: list[dict] = []
    for comp in COMPOSITE_CATALOG:
        match_entries: list[dict] = []
        for display, c in sorted(pair_conditions.items()):
            if not comp.match(c):
                continue
            row: dict = {"display_symbol": display}
            if comp.quality is not None:
                q = comp.quality(c)
                if q is not None:
                    row["quality"] = q
            match_entries.append(row)
            pair_to_scans[display].append(comp.scan_id)

        pairs_flat = [m["display_symbol"] for m in match_entries]
        low_info = len(match_entries) == 0 or len(match_entries) == universe
        composite_results.append({
            "scan_id": comp.scan_id,
            "name": comp.name,
            "category": "composite",
            "lean": comp.lean,
            "description": comp.description,
            "pairs": pairs_flat,
            "matches": match_entries,
            "match_count": len(match_entries),
            "low_information": low_info,
        })

    pairs_out = [
        {
            "display_symbol": d,
            "symbol": pair_meta[d]["symbol"],
            "category": pair_meta[d]["category"],
            "scan_ids": sorted(pair_to_scans[d]),
            "scan_count": len(pair_to_scans[d]),
        }
        for d in sorted(pair_conditions)
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trade_date": trade_date,
        "universe_size": universe,
        "threshold_notes": SMA_BAND_RATIONALE,
        "sma_band_thresholds_pct": SMA_BANDS,
        "candle_detection_thresholds": CANDLE_DETECTION,
        "scans": scan_results,
        "composites": composite_results,
        "pairs": pairs_out,
    }
    return payload, skipped


def print_summary(payload: dict, skipped: list[str]) -> None:
    universe = payload["universe_size"]
    print(f"\nForex scanner summary - trade date {payload['trade_date']}")
    print(f"Universe: {universe} pairs")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")
    print(f"\n{SMA_BAND_RATIONALE}\n")
    print("SMA band thresholds (% vs SMA, in JSON sma_band_thresholds_pct):\n")
    for period, bands in sorted(payload.get("sma_band_thresholds_pct", {}).items(), key=lambda x: int(x[0])):
        print(f"  SMA {period}: {bands}")

    by_cat: dict[str, list[dict]] = {}
    for s in payload["scans"]:
        by_cat.setdefault(s["category"], []).append(s)

    cat_order = [
        "momentum", "macd", "trend", "volatility", "price_action", "gaps",
    ]
    for cat in cat_order:
        items = by_cat.get(cat, [])
        if not items:
            continue
        print(f"\n[{cat.upper()}]")
        for s in sorted(items, key=lambda x: -x["match_count"]):
            flag = "  ** low information" if s["low_information"] else ""
            print(f"  {s['match_count']:>2}/{universe}  {s['name']}{flag}")

    low = [s for s in payload["scans"] if s["low_information"]]
    print(f"\nLow-information base scans (0 or all {universe} pairs): {len(low)}")

    tweezer_top = next((s for s in payload["scans"] if s["scan_id"] == "tweezer_top"), None)
    tweezer_bot = next((s for s in payload["scans"] if s["scan_id"] == "tweezer_bottom"), None)
    if tweezer_top and tweezer_bot:
        print(
            f"\nTweezer tolerance {CANDLE_DETECTION['tweezer_match_max_diff_frac']} "
            f"(0.02% of price): top {tweezer_top['match_count']}/{universe}, "
            f"bottom {tweezer_bot['match_count']}/{universe}"
        )

    composites = payload.get("composites", [])
    if composites:
        print(f"\n[COMPOSITE] ({len(composites)} setups)")
        for comp in composites:
            flag = "  ** low information" if comp["low_information"] else ""
            line = f"  {comp['match_count']:>2}/{universe}  {comp['name']} ({comp['lean']}){flag}"
            if comp["scan_id"] == "composite_oversold_reversal":
                clean = sum(1 for m in comp.get("matches", []) if m.get("quality") == "clean")
                downtrend = sum(
                    1 for m in comp.get("matches", []) if m.get("quality") == "in_downtrend"
                )
                line += f"  [clean={clean}, in_downtrend={downtrend}]"
            print(line)

    low_c = [c for c in composites if c["low_information"]]
    print(f"\nLow-information composites: {len(low_c)}")

    print("\nComposite base-scan substitutions (documented):")
    for note in COMPOSITE_SUBSTITUTIONS:
        print(f"  - {note}")


def export_forex_scans(
    payload: dict,
    out_dir: Path,
    skipped: list[str] | None = None,
) -> Path:
    """Write forex_scans.json and print the same summaries as the scanner CLI."""
    skipped = skipped or []
    out_path = out_dir / "forex_scans.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_summary(payload, skipped)
    currency_strength.print_currency_strength_summary(payload["currency_strength"])
    print(f"\nWrote {out_path.resolve()}")
    n_comp = len(payload.get("composites", []))
    print(
        f"Base scans: {len(payload['scans'])}  |  Composites: {n_comp}  |  "
        f"Pairs indexed: {len(payload['pairs'])}"
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Forex technical condition scanner")
    ap.add_argument("--pairs", default="forex_pairs.json")
    ap.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "cache"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "out"))
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))

    print(f"Scanner: cache {Path(args.cache_dir).resolve()}")
    payload, skipped = run_scanner(pairs, Path(args.cache_dir))

    if payload["universe_size"] == 0:
        raise SystemExit("No pairs with sufficient cache. Run backfill first.")

    payload["currency_strength"] = currency_strength.build_currency_strength(payload)
    export_forex_scans(payload, Path(args.out_dir), skipped)


if __name__ == "__main__":
    main()
