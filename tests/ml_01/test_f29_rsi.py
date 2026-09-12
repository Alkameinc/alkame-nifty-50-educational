"""F29: Wilder RSI must hit 100 on a pure rise, 0 on a pure fall, and 50 when flat.

Warm-up bars (first RSI_PERIOD rows) are not treated as valid RSI predictions.
"""

import pandas as pd
import pytest

from config import RSI_PERIOD
from feature_engineer import FeatureEngineer


def _final_valid_rsi(close: pd.Series) -> float:
    """Return the last RSI after the Wilder warm-up window."""
    rsi = FeatureEngineer.compute_rsi(close, period=RSI_PERIOD)
    post_warmup = rsi.iloc[RSI_PERIOD:]
    assert len(post_warmup) > 0, "Need more closes than RSI_PERIOD to inspect a valid RSI"
    return float(post_warmup.iloc[-1])


def test_f29_rsi_is_100_on_strictly_rising_closes():
    close = pd.Series([float(price) for price in range(100, 120)])  # 100 .. 119
    assert len(close) == 20
    assert _final_valid_rsi(close) == pytest.approx(100.0)


def test_f29_rsi_is_0_on_strictly_falling_closes():
    close = pd.Series([float(price) for price in range(119, 99, -1)])  # 119 .. 100
    assert len(close) == 20
    assert _final_valid_rsi(close) == pytest.approx(0.0)


def test_f29_rsi_is_50_on_flat_closes():
    close = pd.Series([100.0] * 20)
    assert len(close) == 20
    assert _final_valid_rsi(close) == pytest.approx(50.0)
