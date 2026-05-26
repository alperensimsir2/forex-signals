"""
Trading strategy ported from the reference PHP implementation.

Two components:

1) PRIMARY (15/4 pivot confluence)
   A BUY/SELL entry fires on the "confirmation day" 4 bars after a price pivot,
   IFF all three align:
     - Price pivot: bar p's high (for SELL) or low (for BUY) is the extreme
       over the 20-bar window [p-15, p+4]
     - MACD pivot of the same direction anywhere in the symmetric window
       [p-3, p+3]
     - Stochastic cross (14,3,3 slow) in the overbought (SELL) or oversold
       (BUY) zone anywhere in [p-3, p+3]
   Then PSAR alignment is applied as a separate filter:
     - BUY requires close > PSAR (rising PSAR). If the raw signal fires on
       a day when PSAR is opposite, wait up to 2 more bars for PSAR to flip.
     - SELL mirror. If PSAR never aligns within 2 bars, the raw signal is
       discarded.

2) SECONDARY (trend continuation / PSAR snap-back)
   Detects a long trend that briefly reverses, then snaps back. Pattern:
     - Run 1 ("anchor trend"): >=15 bars in one PSAR direction
     - Run 2 ("counter-trend"): <=10 bars in the opposite direction
     - Ratio len(Run 1) / len(Run 2) >= 3.0
     - Trigger fires on the FIRST bar of Run 3 (the return to Run 1's
       direction): BUY if returning to Rising, SELL if returning to Falling.
   Nulls in PSAR don't break a run — they're treated as gaps.

Both entry streams are later merged (secondary wins on same-day conflict)
and expanded into persistent BUY/SELL/None state per day. The merging and
expansion logic lives in consensus.py, not here.

Bars in this module are always ASC-ordered (oldest first, index 0).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ===== Pivot and cross detectors =============================================
# Each expects ASC-ordered series (index 0 = oldest).
# Index i is the bar being tested. Returns True/False.

def _val(series: pd.Series, idx: int) -> Optional[float]:
    if idx < 0 or idx >= len(series):
        return None
    v = series.iloc[idx]
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
        return None
    return float(v)


def is_price_pivot_top(high: pd.Series, i: int) -> bool:
    """High[i] is the MAX of high over [i-15, i+4]."""
    n = len(high)
    if i < 15 or i + 4 >= n:
        return False
    hi = _val(high, i)
    if hi is None:
        return False
    max_v = None
    for j in range(i - 15, i + 5):
        v = _val(high, j)
        if v is None:
            continue
        if max_v is None or v > max_v:
            max_v = v
    return max_v is not None and hi >= max_v


def is_price_pivot_bottom(low: pd.Series, i: int) -> bool:
    """Low[i] is the MIN of low over [i-15, i+4]."""
    n = len(low)
    if i < 15 or i + 4 >= n:
        return False
    lo = _val(low, i)
    if lo is None:
        return False
    min_v = None
    for j in range(i - 15, i + 5):
        v = _val(low, j)
        if v is None:
            continue
        if min_v is None or v < min_v:
            min_v = v
    return min_v is not None and lo <= min_v


def is_macd_pivot_top(macd_line: pd.Series, i: int) -> bool:
    """MACD[i] is the MAX over [i-15, i+4]."""
    n = len(macd_line)
    if i < 15 or i + 4 >= n:
        return False
    m = _val(macd_line, i)
    if m is None:
        return False
    max_v = None
    for j in range(i - 15, i + 5):
        v = _val(macd_line, j)
        if v is None:
            continue
        if max_v is None or v > max_v:
            max_v = v
    return max_v is not None and m >= max_v


def is_macd_pivot_bottom(macd_line: pd.Series, i: int) -> bool:
    """MACD[i] is the MIN over [i-15, i+4]."""
    n = len(macd_line)
    if i < 15 or i + 4 >= n:
        return False
    m = _val(macd_line, i)
    if m is None:
        return False
    min_v = None
    for j in range(i - 15, i + 5):
        v = _val(macd_line, j)
        if v is None:
            continue
        if min_v is None or v < min_v:
            min_v = v
    return min_v is not None and m <= min_v


def is_bullish_stoch_cross_at(k: pd.Series, d: pd.Series, j: int) -> bool:
    """%K crosses above %D at bar j, AND today's %K < 20 (oversold zone)."""
    if j < 1:
        return False
    k_prev = _val(k, j - 1)
    d_prev = _val(d, j - 1)
    k_curr = _val(k, j)
    d_curr = _val(d, j)
    if None in (k_prev, d_prev, k_curr, d_curr):
        return False
    return k_prev < d_prev and k_curr > d_curr and k_curr < 20


def is_bearish_stoch_cross_at(k: pd.Series, d: pd.Series, j: int) -> bool:
    """%K crosses below %D at bar j, AND today's %K > 80 (overbought zone)."""
    if j < 1:
        return False
    k_prev = _val(k, j - 1)
    d_prev = _val(d, j - 1)
    k_curr = _val(k, j)
    d_curr = _val(d, j)
    if None in (k_prev, d_prev, k_curr, d_curr):
        return False
    return k_prev > d_prev and k_curr < d_curr and k_curr > 80


# ----- ±3-day window around a pivot -----------------------------------------

def _has_in_window(detector, pivot_idx: int, series_args, n: int,
                    min_j: int = 0) -> bool:
    """Generic: was detector(...) true for any j in [pivot-3, pivot+3]?"""
    start = max(min_j, pivot_idx - 3)
    end = min(n - 1, pivot_idx + 3)
    for j in range(start, end + 1):
        if detector(*series_args, j):
            return True
    return False


def has_macd_pivot_top_in_window(macd_line: pd.Series, pivot_idx: int) -> bool:
    return _has_in_window(is_macd_pivot_top, pivot_idx, (macd_line,), len(macd_line))


def has_macd_pivot_bottom_in_window(macd_line: pd.Series, pivot_idx: int) -> bool:
    return _has_in_window(is_macd_pivot_bottom, pivot_idx, (macd_line,), len(macd_line))


def has_bearish_stoch_cross_in_window(k: pd.Series, d: pd.Series, pivot_idx: int) -> bool:
    return _has_in_window(is_bearish_stoch_cross_at, pivot_idx, (k, d), len(k), min_j=1)


def has_bullish_stoch_cross_in_window(k: pd.Series, d: pd.Series, pivot_idx: int) -> bool:
    return _has_in_window(is_bullish_stoch_cross_at, pivot_idx, (k, d), len(k), min_j=1)


# ===== Primary strategy: 15/4 pivot + MACD pivot + Stoch cross ==============

def compute_primary_signals(df_asc: pd.DataFrame) -> dict[str, str]:
    """
    Scan all valid confirmation days and emit BUY/SELL signals where the
    three confluence conditions align. The PSAR filter is applied separately
    via apply_psar_alignment_filter.

    df_asc is ASC-ordered with columns:
      - 'date' (datetime or string YYYY-MM-DD)
      - 'high', 'low', 'close'
      - 'macd' (MACD line)
      - 'stoch_k' (14,3,3 slow), 'stoch_d'

    Returns: {date_str: 'BUY'|'SELL'}.
    """
    n = len(df_asc)
    out: dict[str, str] = {}
    if n < 20:
        return out

    high = df_asc["high"].reset_index(drop=True)
    low = df_asc["low"].reset_index(drop=True)
    macd = df_asc["macd"].reset_index(drop=True)
    k = df_asc["stoch_k"].reset_index(drop=True)
    d = df_asc["stoch_d"].reset_index(drop=True)
    dates = df_asc["date"].reset_index(drop=True)

    # Confirmation day i = pivot p + 4. Need i >= 19 (since p must be >= 15).
    for i in range(19, n):
        p = i - 4
        date = dates.iloc[i]
        date_str = _date_to_str(date)
        if date_str is None:
            continue

        if (is_price_pivot_top(high, p)
                and has_macd_pivot_top_in_window(macd, p)
                and has_bearish_stoch_cross_in_window(k, d, p)):
            out[date_str] = "SELL"
            continue

        if (is_price_pivot_bottom(low, p)
                and has_macd_pivot_bottom_in_window(macd, p)
                and has_bullish_stoch_cross_in_window(k, d, p)):
            out[date_str] = "BUY"

    return out


# ===== PSAR alignment filter ================================================

def apply_psar_alignment_filter(df_asc: pd.DataFrame,
                                 raw_signals: dict[str, str]) -> dict[str, str]:
    """
    For each raw signal, the trade may only trigger on a day when PSAR aligns:
      - BUY requires close > PSAR
      - SELL requires close < PSAR

    If the raw day doesn't align, look at the next 1-2 bars. If alignment
    happens within that window, trigger on that day. Otherwise, discard.

    Returns: {date_str: 'BUY'|'SELL'} with the potentially-shifted trigger date.
    """
    if not raw_signals:
        return {}

    n = len(df_asc)
    if n == 0:
        return {}

    dates = df_asc["date"].reset_index(drop=True)
    close = df_asc["close"].reset_index(drop=True)
    psar = df_asc["psar"].reset_index(drop=True)

    date_to_idx: dict[str, int] = {}
    for idx in range(n):
        ds = _date_to_str(dates.iloc[idx])
        if ds is not None:
            date_to_idx[ds] = idx

    out: dict[str, str] = {}
    for raw_date in sorted(raw_signals.keys()):
        side = raw_signals[raw_date]
        if raw_date not in date_to_idx:
            continue

        i = date_to_idx[raw_date]
        trigger_date = None

        for j in range(i, min(i + 3, n)):  # today + up to 2 more bars
            c = _val(close, j)
            s = _val(psar, j)
            if c is None or s is None:
                continue
            if side == "BUY" and c > s:
                trigger_date = _date_to_str(dates.iloc[j])
                break
            if side == "SELL" and c < s:
                trigger_date = _date_to_str(dates.iloc[j])
                break

        if trigger_date is not None and trigger_date not in out:
            out[trigger_date] = side

    return out


# ===== Secondary strategy: trend-continuation snap-back =====================

def compute_trending_stocks_signals(df_asc: pd.DataFrame) -> dict[str, str]:
    """
    Pattern:
      Run1 (anchor trend, dir=Rising or Falling) >= 15 bars,
      Run2 (counter-trend) <= 10 bars,
      ratio len(Run1)/len(Run2) >= 3.0,
      and Run3 returns to Run1's direction.

    Trigger: first bar of Run3. BUY if Run1 dir == Rising, SELL if Falling.

    Nulls in PSAR (or close) are treated as gaps — they do NOT break a run.
    Only a flip to the opposite direction starts a new run.

    Returns: {date_str: 'BUY'|'SELL'}.
    """
    n = len(df_asc)
    out: dict[str, str] = {}
    if n < 15 + 1 + 1:
        return out

    close = df_asc["close"].reset_index(drop=True)
    psar = df_asc["psar"].reset_index(drop=True)
    dates = df_asc["date"].reset_index(drop=True)

    # Per-bar direction: "Rising" / "Falling" / None.
    directions: list[Optional[str]] = []
    for i in range(n):
        c = _val(close, i)
        s = _val(psar, i)
        if c is None or s is None:
            directions.append(None)
        else:
            directions.append("Rising" if c > s else "Falling")

    # Build runs: null bars are skipped (don't break run); only opposite
    # direction starts a new one. Length counts only non-null bars.
    runs: list[dict] = []
    i = 0
    while i < n:
        if directions[i] is None:
            i += 1
            continue
        start = i
        dir_ = directions[i]
        length = 1
        i += 1
        while i < n:
            if directions[i] is None:
                i += 1
                continue
            if directions[i] == dir_:
                length += 1
                i += 1
            else:
                break
        runs.append({"start": start, "end": i - 1, "dir": dir_, "length": length})

    min_phase_1 = 15
    max_phase_2 = 10
    min_ratio = 3.0

    for r in range(len(runs) - 2):
        run1 = runs[r]
        run2 = runs[r + 1]
        run3 = runs[r + 2]

        if run1["dir"] != run3["dir"]:
            continue
        if run2["dir"] == run1["dir"]:
            continue

        len1 = run1["length"]
        len2 = run2["length"]

        if len1 < min_phase_1 or len2 > max_phase_2 or len2 < 1:
            continue
        if len1 / len2 < min_ratio:
            continue

        trigger_idx = run3["start"]
        trigger_date = _date_to_str(dates.iloc[trigger_idx])
        if trigger_date is None:
            continue
        if trigger_date in out:
            continue  # first pattern wins

        out[trigger_date] = "BUY" if run1["dir"] == "Rising" else "SELL"

    return out


# ===== Helpers ==============================================================

def _date_to_str(v) -> Optional[str]:
    """Coerce a date value (datetime or string) to 'YYYY-MM-DD'."""
    if v is None:
        return None
    if isinstance(v, str):
        return v[:10] if len(v) >= 10 else v
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except Exception:
        return None