"""
Technical indicators.

Conventions:
- SMA / EMA / RSI / MACD / Bollinger / Z-score: compute on adjusted close
- ADX / Stochastic / ATR / PSAR: compute on raw H/L/C

All smoothing is explicit. Do NOT replace with pandas-ta without verifying that
pandas-ta's defaults match these (for ATR/ADX they do not — those default to
EMA, not Wilder).

Strategy module uses:
- parabolic_sar (for 15/4 pivot strategy's PSAR alignment filter + snap-back strategy)
- stoch_slow_k + stoch_d (14,3,3 slow stochastic — pre-smoothed %K, then %D is SMA of that)
- macd (standard 12/26/9)

Legacy helpers (bollinger, zscore_dev, adx) are retained; they are not used by
the current pivot strategy, but we leave them so any ad-hoc script that still
imports them keeps working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----- Moving averages -------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema_spec(series: pd.Series, period: int) -> pd.Series:
    """
    EMA with alpha = 2/(N+1), seeded with SMA(N) at index N-1. Matches the
    Excel convention exactly; pandas' native ewm(adjust=False) uses a different
    seed and diverges subtly.
    """
    alpha = 2.0 / (period + 1)
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(valid) < period:
        return out
    vals = valid.to_numpy()
    smoothed = np.full(len(vals), np.nan)
    smoothed[period - 1] = vals[:period].mean()
    for i in range(period, len(vals)):
        smoothed[i] = alpha * vals[i] + (1.0 - alpha) * smoothed[i - 1]
    out.loc[valid.index] = smoothed
    return out


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: seed with SMA(N), then avg_t = avg_{t-1}*(N-1)/N + x_t/N."""
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(valid) < period:
        return out
    vals = valid.to_numpy()
    smoothed = np.full(len(vals), np.nan)
    smoothed[period - 1] = vals[:period].mean()
    for i in range(period, len(vals)):
        smoothed[i] = smoothed[i - 1] * (period - 1) / period + vals[i] / period
    out.loc[valid.index] = smoothed
    return out


# ----- Bollinger / RSI / MACD ------------------------------------------------

def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=1)
    return mid + num_std * std, mid, mid - num_std * std


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema_spec(close, fast)
    ema_slow = ema_spec(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_spec(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ----- Stochastic (fast + slow) ---------------------------------------------

def stoch_k_fast(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
    """
    Fast %K: 100 * (close - lowestLow_N) / (highestHigh_N - lowestLow_N).
    "Fast" = no pre-smoothing.
    """
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    rng = (highest - lowest).replace(0, np.nan)
    return 100 * (close - lowest) / rng


def stoch_slow_k(high: pd.Series, low: pd.Series, close: pd.Series,
                 k_period: int = 14, k_smoothing: int = 3) -> pd.Series:
    """
    Slow %K for the 14,3,3 Slow Stochastic convention: compute fast %K, then
    smooth it with SMA(k_smoothing). The output is called "slow %K" and is
    the series the PHP strategy compares to its %D line.
    """
    fast_k = stoch_k_fast(high, low, close, k_period)
    return fast_k.rolling(k_smoothing, min_periods=k_smoothing).mean()


def stoch_d(k: pd.Series, period: int = 3) -> pd.Series:
    """%D is SMA(period) of the %K series (fast or slow)."""
    return k.rolling(period, min_periods=period).mean()


# Backwards-compat alias.
stoch_k = stoch_k_fast


# ----- True range / ATR / ADX -----------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    tr = true_range(high, low, close)
    up = high.diff()
    down = -low.diff()
    plus_dm_raw = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=high.index
    )
    minus_dm_raw = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index
    )
    atr_val = wilder_smooth(tr, period)
    plus_dm_s = wilder_smooth(plus_dm_raw, period)
    minus_dm_s = wilder_smooth(minus_dm_raw, period)
    plus_di = 100 * plus_dm_s / atr_val
    minus_di = 100 * minus_dm_s / atr_val
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_val = wilder_smooth(dx, period)
    return adx_val, plus_di, minus_di


# ----- Parabolic SAR (Wilder) -----------------------------------------------

def parabolic_sar(high: pd.Series, low: pd.Series,
                  af_start: float = 0.02,
                  af_step: float = 0.02,
                  af_max: float = 0.2) -> pd.Series:
    """
    Wilder's Parabolic Stop-and-Reverse.

    Returns a Series of SAR values aligned with the input index. The first
    element is always NaN (SAR needs at least two bars to initialize).

    Interpretation:
      - If close > SAR, trend is UP. The SAR sits below price and rises
        as the trend continues.
      - If close < SAR, trend is DOWN. The SAR sits above price.
      - When price crosses SAR, the trend flips and SAR jumps to the prior EP.

    Implementation follows the standard rules:
      1. Initialize on bar 1. If high[1] > high[0], start in uptrend with
         SAR = low[0], EP = high[1]. Else start in downtrend with
         SAR = high[0], EP = low[1].
      2. Each subsequent bar: tentative SAR = prev_SAR + AF * (EP - prev_SAR).
      3. Cap the SAR so it doesn't penetrate the previous 2 bars' extreme on
         the "wrong" side (for uptrends, SAR can't exceed min(prior_low_1,
         prior_low_2); for downtrends, SAR can't go below max(prior_high_1,
         prior_high_2)).
      4. Check reversal: if uptrend and today's low < SAR, flip. If downtrend
         and today's high > SAR, flip.
      5. On flip: SAR becomes the prior EP, AF resets to af_start, EP is
         today's extreme in the new direction.
      6. On continuation: if today's extreme beats EP, bump AF (capped at
         af_max) and update EP.

    Defaults (0.02 / 0.02 / 0.2) are Wilder's originals and match every
    PSAR reference (Stockcharts, TradingView, etc.) unless explicitly tuned.
    """
    n = len(high)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=high.index, dtype=float)

    h = high.to_numpy()
    l = low.to_numpy()

    if (np.isnan(h[0]) or np.isnan(l[0]) or
        np.isnan(h[1]) or np.isnan(l[1])):
        return pd.Series(sar, index=high.index, dtype=float)

    uptrend = h[1] > h[0]
    ep = h[1] if uptrend else l[1]
    sar_prev = l[0] if uptrend else h[0]
    af = af_start
    sar[1] = sar_prev

    for i in range(2, n):
        hi = h[i]
        lo = l[i]
        if np.isnan(hi) or np.isnan(lo):
            sar[i] = np.nan
            continue

        sar_today = sar_prev + af * (ep - sar_prev)

        if uptrend:
            cap_lo = min(l[i - 1], l[i - 2])
            if sar_today > cap_lo:
                sar_today = cap_lo
        else:
            cap_hi = max(h[i - 1], h[i - 2])
            if sar_today < cap_hi:
                sar_today = cap_hi

        if uptrend and lo < sar_today:
            uptrend = False
            sar_today = ep
            ep = lo
            af = af_start
        elif (not uptrend) and hi > sar_today:
            uptrend = True
            sar_today = ep
            ep = hi
            af = af_start
        else:
            if uptrend:
                if hi > ep:
                    ep = hi
                    af = min(af + af_step, af_max)
            else:
                if lo < ep:
                    ep = lo
                    af = min(af + af_step, af_max)

        sar[i] = sar_today
        sar_prev = sar_today

    return pd.Series(sar, index=high.index, dtype=float)


# ----- Z-score (legacy, unused by new strategy) -----------------------------

def zscore_dev(close: pd.Series, sma50: pd.Series, dev_window: int = 60) -> pd.Series:
    dev = (close - sma50) / sma50
    dev_mean = dev.rolling(dev_window, min_periods=dev_window).mean()
    dev_std = dev.rolling(dev_window, min_periods=dev_window).std(ddof=1)
    return (dev - dev_mean) / dev_std.replace(0, np.nan)