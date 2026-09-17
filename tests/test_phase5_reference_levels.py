# 1. Standard library imports
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from config import HORIZON_INTRADAY
from reference_level_engine import (
    InsufficientDataError,
    ReferenceLevelDeltas,
    ReferenceLevelEngine,
    ReferenceLevelError,
    ReferenceLevels,
)


def _build_test_price_df(n_bars: int = 150, base_price: float = 1000.0, trend: float = 0.5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01 09:15", periods=n_bars, freq="5min", tz="UTC")
    closes = [base_price]
    for _ in range(n_bars - 1):
        closes.append(max(10.0, closes[-1] + trend + rng.normal(0, 1.5)))
    highs = [c + abs(rng.normal(2.0, 0.5)) for c in closes]
    lows = [c - abs(rng.normal(2.0, 0.5)) for c in closes]
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes}, index=dates)


# ============================================================================
# REF-001: No Plausible Fallback Values
# ============================================================================
def test_ref_001_empty_dataframe_returns_unavailable_without_fallbacks():
    """REF-001: Empty dataframe must return UNAVAILABLE with None bands (no fake numbers)."""
    engine = ReferenceLevelEngine()
    empty_df = pd.DataFrame()

    levels = engine.get_reference_levels("RELIANCE", empty_df, horizon="30D")
    assert isinstance(levels, ReferenceLevels)
    assert levels.is_available is False
    assert levels.status == "UNAVAILABLE"
    assert levels.support_band_low is None
    assert levels.support_band_high is None
    assert levels.resistance_band_low is None
    assert levels.resistance_band_high is None
    assert levels.moving_average_value is None

    deltas = engine.compute_deltas(levels)
    assert isinstance(deltas, ReferenceLevelDeltas)
    assert deltas.status == "UNAVAILABLE"
    assert deltas.pct_from_moving_average == 0.0
    assert deltas.pct_from_support_band == 0.0
    assert deltas.pct_from_resistance_band == 0.0


def test_ref_001_insufficient_bars_never_manufactures_current_price():
    """
    REF-001: When data is insufficient to compute reliable bands,
    never synthesize support = current_price or resistance = current_price.
    """
    engine = ReferenceLevelEngine()
    # Only 2 bars: insufficient for swing quantiles or moving averages
    short_df = _build_test_price_df(n_bars=2, base_price=2500.0)

    levels = engine.get_reference_levels("INFY", short_df, horizon="30D")
    assert levels.status == "UNAVAILABLE"
    assert levels.support_band_low is None
    assert levels.resistance_band_high is None
    # Must never equal current price
    assert levels.support_band_low != 2500.0
    assert levels.resistance_band_high != 2500.0


def test_ref_001_swing_levels_failure_returns_none():
    """REF-001: compute_swing_levels failure returns (None, None, None, None)."""
    engine = ReferenceLevelEngine()
    empty_df = pd.DataFrame({"High": [], "Low": [], "Close": []})

    sl, sh, rl, rh = engine.compute_swing_levels(empty_df, lookback_window=10)
    assert sl is None
    assert sh is None
    assert rl is None
    assert rh is None


def test_ref_001_bollinger_failure_returns_none():
    """REF-001: compute_bollinger_bands failure returns (None, None, None, None)."""
    engine = ReferenceLevelEngine()
    nan_df = pd.DataFrame({"Close": [np.nan] * 5})

    sl, sh, rl, rh = engine.compute_bollinger_bands(nan_df, period=10)
    assert sl is None
    assert sh is None
    assert rl is None
    assert rh is None


# ============================================================================
# REF-002: Structured Error Handling
# ============================================================================
def test_ref_002_structured_logging_on_insufficient_data(caplog):
    """REF-002: Structured logging includes symbol, timestamp, operation, exception_type."""
    engine = ReferenceLevelEngine()

    with caplog.at_level(logging.WARNING):
        levels = engine.get_reference_levels("TCS", pd.DataFrame(), horizon="30D")
        assert levels.status == "UNAVAILABLE"

    # Verify structured logging occurred
    assert any("REF-002" in record.message for record in caplog.records)
    # Check that exception_type or InsufficientDataError is mentioned
    assert any("InsufficientDataError" in record.message or hasattr(record, "exception_type") for record in caplog.records)


def test_ref_002_corrupted_data_types_handled_gracefully():
    """REF-002: Corrupted or non-numeric columns trigger structured validation failure without crash."""
    engine = ReferenceLevelEngine()
    dates = pd.date_range("2026-01-01", periods=5, tz="UTC")
    corrupt_df = pd.DataFrame({"Close": ["not_a_number"] * 5, "High": [100.0] * 5, "Low": [90.0] * 5}, index=dates)

    levels = engine.get_reference_levels("CORRUPT", corrupt_df, horizon="30D")
    assert levels.status == "UNAVAILABLE"
    assert "error" in levels.error.lower() or "validation" in levels.error.lower()


# ============================================================================
# REF-003: Method Validation & Sensitivity Analysis
# ============================================================================
def test_ref_003_band_ordering_invariant():
    """
    REF-003: For valid market data, assert the fundamental invariant:
    support_low <= support_high <= resistance_low <= resistance_high.
    """
    engine = ReferenceLevelEngine()
    df = _build_test_price_df(n_bars=300, base_price=1500.0, seed=777)

    levels = engine.get_reference_levels("HDFC", df, horizon="30D")
    assert levels.status == "AVAILABLE"
    assert levels.is_available is True

    assert levels.support_band_low <= levels.support_band_high
    assert levels.support_band_high <= levels.resistance_band_low
    assert levels.resistance_band_low <= levels.resistance_band_high


def test_ref_003_sensitivity_across_lookbacks_and_quantiles():
    """
    REF-003: Sensitivity analysis across multiple lookback windows (30, 60, 120, 252)
    and quantile thresholds (5%, 10%, 20%).
    """
    engine = ReferenceLevelEngine()
    df = _build_test_price_df(n_bars=400, base_price=2000.0, seed=888)

    lookbacks = [30, 60, 120, 252]
    quantiles = [0.05, 0.10, 0.20]

    sens = engine.validate_band_sensitivity(df, lookbacks=lookbacks, quantiles=quantiles)
    assert len(sens) == len(lookbacks) * len(quantiles)

    for key, metrics in sens.items():
        assert metrics["valid_ordering"] is True, f"Ordering violated for {key}: {metrics}"
        assert metrics["support_low"] <= metrics["support_high"]
        assert metrics["support_high"] <= metrics["resistance_low"]
        assert metrics["resistance_low"] <= metrics["resistance_high"]
        assert metrics["spread_pct"] >= 0.0


def test_ref_003_bollinger_method_sensitivity():
    """REF-003: Validate Bollinger method when configured for horizon."""
    engine = ReferenceLevelEngine()
    df = _build_test_price_df(n_bars=100, base_price=500.0)

    levels = engine.get_reference_levels("SBIN", df, horizon=HORIZON_INTRADAY)
    if levels.is_available:
        assert levels.support_band_low <= levels.support_band_high
        assert levels.resistance_band_low <= levels.resistance_band_high
