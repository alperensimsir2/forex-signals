"""
Pip helpers for forex pairs.

JPY-quote or JPY-base pairs use 1 pip = 0.01; all others use 1 pip = 0.0001.
"""

from __future__ import annotations


def pip_size(symbol: str) -> float:
    """Return pip size for a pair symbol (e.g. EURUSD, USDJPY)."""
    if "JPY" in symbol.upper():
        return 0.01
    return 0.0001


def change_pips(close: float, previous_close: float, symbol: str) -> float | None:
    """Daily change in pips, rounded to 1 decimal."""
    if previous_close is None or close is None or previous_close == 0:
        return None
    size = pip_size(symbol)
    return round((close - previous_close) / size, 1)
