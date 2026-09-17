"""
test_f42_magic_numbers.py

Comprehensive test suite for F42: Magic numbers (2 pts, P2)

Tests cover:
1. Magic numbers extraction to named constants
2. Constants have proper documentation
3. Constants are used correctly in production code
4. Utility functions work correctly
5. No hardcoded magic numbers remain in key files

Part of PY-01 Assignment

Author: Engineering Team
Date: 2026-09-14
"""

import pytest
from pathlib import Path
import sys
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from magic_numbers_config import (
    # Time constants
    MINUTES_PER_HOUR,
    TRADING_DAYS_PER_MONTH_APPROX,
    TRADING_DAYS_PER_QUARTER_APPROX,
    TRADING_DAYS_PER_YEAR_APPROX,
    # Technical indicators
    RSI_NEUTRAL_VALUE,
    DEFAULT_MA_PERIOD,
    # Lookback periods
    LOOKBACK_PERIOD_30D,
    LOOKBACK_PERIOD_3M,
    LOOKBACK_PERIOD_6M,
    LOOKBACK_PERIOD_1Y,
    DEFAULT_LOOKBACK_PERIOD,
    HORIZON_LOOKBACK_MAP,
    # Support/resistance
    RESISTANCE_HIGH_QUANTILE,
    RESISTANCE_LOW_QUANTILE,
    SUPPORT_HIGH_QUANTILE,
    SUPPORT_LOW_QUANTILE,
    RESISTANCE_BAND_LOWER_THRESHOLD,
    RESISTANCE_BAND_UPPER_THRESHOLD,
    # Test constants
    MUTATION_TEST_PRICE_DELTA,
    MUTATION_TEST_VOLUME_MULTIPLIER,
    DEFAULT_TEST_DAYS,
    DEFAULT_TEST_BARS_PER_DAY,
    LONG_TEST_DAYS,
    # Functions
    get_horizon_lookback,
    get_trading_days_for_period,
    validate_quantile,
)


# ============================================================================
# Test Suite 1: Time Constants
# ============================================================================

def test_f42_time_conversion_constants():
    """F42-1: Time conversion constants are correct."""
    assert MINUTES_PER_HOUR == 60
    print("✓ F42-1: MINUTES_PER_HOUR = 60")


def test_f42_trading_days_constants():
    """F42-2: Trading days constants are reasonable."""
    assert TRADING_DAYS_PER_MONTH_APPROX == 21
    assert TRADING_DAYS_PER_QUARTER_APPROX == 63
    assert TRADING_DAYS_PER_YEAR_APPROX == 252
    
    # Sanity checks
    assert TRADING_DAYS_PER_QUARTER_APPROX == TRADING_DAYS_PER_MONTH_APPROX * 3
    assert TRADING_DAYS_PER_YEAR_APPROX == TRADING_DAYS_PER_MONTH_APPROX * 12
    
    print("✓ F42-2: All trading days constants correct and consistent")


# ============================================================================
# Test Suite 2: Technical Indicator Constants
# ============================================================================

def test_f42_rsi_neutral_value():
    """F42-3: RSI neutral value is 50 (middle of 0-100 range)."""
    assert RSI_NEUTRAL_VALUE == 50.0
    assert 0 <= RSI_NEUTRAL_VALUE <= 100
    print("✓ F42-3: RSI_NEUTRAL_VALUE = 50.0")


def test_f42_default_ma_period():
    """F42-4: Default MA period is 50 (common medium-term MA)."""
    assert DEFAULT_MA_PERIOD == 50
    assert DEFAULT_MA_PERIOD > 0
    print("✓ F42-4: DEFAULT_MA_PERIOD = 50")


# ============================================================================
# Test Suite 3: Lookback Period Constants
# ============================================================================

def test_f42_lookback_periods_defined():
    """F42-5: All horizon lookback periods are defined."""
    assert LOOKBACK_PERIOD_30D == 60
    assert LOOKBACK_PERIOD_3M == 120
    assert LOOKBACK_PERIOD_6M == 252
    assert LOOKBACK_PERIOD_1Y == 500
    assert DEFAULT_LOOKBACK_PERIOD == 120
    
    print("✓ F42-5: All lookback periods defined")


def test_f42_lookback_periods_ordered():
    """F42-6: Lookback periods increase with horizon length."""
    assert LOOKBACK_PERIOD_30D < LOOKBACK_PERIOD_3M
    assert LOOKBACK_PERIOD_3M < LOOKBACK_PERIOD_6M
    assert LOOKBACK_PERIOD_6M < LOOKBACK_PERIOD_1Y
    
    print("✓ F42-6: Lookback periods correctly ordered")


def test_f42_horizon_lookback_map():
    """F42-7: Horizon lookback map contains all expected horizons."""
    assert "30D" in HORIZON_LOOKBACK_MAP
    assert "3M" in HORIZON_LOOKBACK_MAP
    assert "6M" in HORIZON_LOOKBACK_MAP
    assert "1Y" in HORIZON_LOOKBACK_MAP
    
    assert HORIZON_LOOKBACK_MAP["30D"] == LOOKBACK_PERIOD_30D
    assert HORIZON_LOOKBACK_MAP["3M"] == LOOKBACK_PERIOD_3M
    assert HORIZON_LOOKBACK_MAP["6M"] == LOOKBACK_PERIOD_6M
    assert HORIZON_LOOKBACK_MAP["1Y"] == LOOKBACK_PERIOD_1Y
    
    print("✓ F42-7: Horizon lookback map correct")


# ============================================================================
# Test Suite 4: Support/Resistance Constants
# ============================================================================

def test_f42_quantiles_valid_range():
    """F42-8: All quantiles are in valid [0, 1] range."""
    quantiles = [
        RESISTANCE_HIGH_QUANTILE,
        RESISTANCE_LOW_QUANTILE,
        SUPPORT_HIGH_QUANTILE,
        SUPPORT_LOW_QUANTILE,
    ]
    
    for q in quantiles:
        assert 0.0 <= q <= 1.0, f"Quantile {q} out of range"
    
    print("✓ F42-8: All quantiles in valid range")


def test_f42_quantiles_ordered():
    """F42-9: Quantiles are properly ordered."""
    assert RESISTANCE_HIGH_QUANTILE > RESISTANCE_LOW_QUANTILE
    assert RESISTANCE_LOW_QUANTILE > SUPPORT_HIGH_QUANTILE
    assert SUPPORT_HIGH_QUANTILE >= SUPPORT_LOW_QUANTILE
    
    print("✓ F42-9: Quantiles properly ordered")


def test_f42_resistance_band_thresholds():
    """F42-10: Resistance band thresholds are reasonable."""
    assert RESISTANCE_BAND_LOWER_THRESHOLD < 1.0
    assert RESISTANCE_BAND_UPPER_THRESHOLD > 1.0
    assert RESISTANCE_BAND_UPPER_THRESHOLD > RESISTANCE_BAND_LOWER_THRESHOLD
    
    # Should be close to 1.0 (small percentages)
    assert abs(RESISTANCE_BAND_LOWER_THRESHOLD - 1.0) < 0.01
    assert abs(RESISTANCE_BAND_UPPER_THRESHOLD - 1.0) < 0.01
    
    print("✓ F42-10: Resistance band thresholds valid")


# ============================================================================
# Test Suite 5: Test/Validation Constants
# ============================================================================

def test_f42_mutation_test_constants():
    """F42-11: Mutation test constants are defined."""
    assert MUTATION_TEST_PRICE_DELTA == 500.0
    assert MUTATION_TEST_VOLUME_MULTIPLIER == 20.0
    
    # Should be large enough to be significant
    assert MUTATION_TEST_PRICE_DELTA > 100
    assert MUTATION_TEST_VOLUME_MULTIPLIER > 10
    
    print("✓ F42-11: Mutation test constants defined")


def test_f42_synthetic_data_constants():
    """F42-12: Synthetic data generation constants are reasonable."""
    assert DEFAULT_TEST_DAYS == 5
    assert DEFAULT_TEST_BARS_PER_DAY == 75
    assert LONG_TEST_DAYS == 70
    
    assert DEFAULT_TEST_DAYS > 0
    assert DEFAULT_TEST_BARS_PER_DAY > 0
    assert LONG_TEST_DAYS > DEFAULT_TEST_DAYS
    
    print("✓ F42-12: Synthetic data constants reasonable")


# ============================================================================
# Test Suite 6: Utility Functions
# ============================================================================

def test_f42_get_horizon_lookback_function():
    """F42-13: get_horizon_lookback() returns correct values."""
    assert get_horizon_lookback("30D") == 60
    assert get_horizon_lookback("3M") == 120
    assert get_horizon_lookback("6M") == 252
    assert get_horizon_lookback("1Y") == 500
    
    # Unknown horizon returns default
    assert get_horizon_lookback("UNKNOWN") == DEFAULT_LOOKBACK_PERIOD
    
    print("✓ F42-13: get_horizon_lookback() works correctly")


def test_f42_get_trading_days_function():
    """F42-14: get_trading_days_for_period() returns correct values."""
    assert get_trading_days_for_period("1M") == 21
    assert get_trading_days_for_period("1Q") == 63
    assert get_trading_days_for_period("3M") == 63
    assert get_trading_days_for_period("1Y") == 252
    
    print("✓ F42-14: get_trading_days_for_period() works correctly")


def test_f42_get_trading_days_error_handling():
    """F42-15: get_trading_days_for_period() raises error for invalid period."""
    with pytest.raises(ValueError) as exc_info:
        get_trading_days_for_period("INVALID")
    
    assert "Unknown period" in str(exc_info.value)
    
    print("✓ F42-15: Error handling works correctly")


def test_f42_validate_quantile_function():
    """F42-16: validate_quantile() accepts valid quantiles."""
    # Should not raise error
    validate_quantile(0.0)
    validate_quantile(0.5)
    validate_quantile(1.0)
    validate_quantile(0.25)
    validate_quantile(0.75)
    
    print("✓ F42-16: validate_quantile() accepts valid values")


def test_f42_validate_quantile_error_handling():
    """F42-17: validate_quantile() rejects invalid quantiles."""
    with pytest.raises(ValueError):
        validate_quantile(-0.1)
    
    with pytest.raises(ValueError):
        validate_quantile(1.5)
    
    print("✓ F42-17: validate_quantile() rejects invalid values")


# ============================================================================
# Test Suite 7: Integration with feature_engineer.py
# ============================================================================

def test_f42_feature_engineer_imports_constants():
    """F42-18: feature_engineer.py imports magic_numbers_config."""
    feature_engineer_path = Path(__file__).parent.parent.parent / "feature_engineer.py"
    
    if not feature_engineer_path.exists():
        pytest.skip("feature_engineer.py not found")
    
    with open(feature_engineer_path, 'r') as f:
        content = f.read()
    
    assert "from magic_numbers_config import" in content
    assert "RSI_NEUTRAL_VALUE" in content
    assert "DEFAULT_MA_PERIOD" in content
    assert "HORIZON_LOOKBACK_MAP" in content
    
    print("✓ F42-18: feature_engineer.py imports constants")


def test_f42_no_hardcoded_rsi_neutral():
    """F42-19: No hardcoded 50.0 for RSI neutral value."""
    feature_engineer_path = Path(__file__).parent.parent.parent / "feature_engineer.py"
    
    if not feature_engineer_path.exists():
        pytest.skip("feature_engineer.py not found")
    
    with open(feature_engineer_path, 'r') as f:
        lines = f.readlines()
    
    # Check for hardcoded 50.0 in RSI context (excluding imports and comments)
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('#'):
            continue
        if 'import' in line.lower():
            continue
        
        # Look for patterns like ", 50.0)" in RSI context
        if '50.0' in line and 'rsi' in line.lower():
            # Should use RSI_NEUTRAL_VALUE instead
            assert 'RSI_NEUTRAL_VALUE' in line, \
                f"Line {i} has hardcoded 50.0 in RSI context: {line.strip()}"
    
    print("✓ F42-19: No hardcoded RSI neutral value")


def test_f42_no_hardcoded_lookback_map():
    """F42-20: No hardcoded lookback map dictionary."""
    feature_engineer_path = Path(__file__).parent.parent.parent / "feature_engineer.py"
    
    if not feature_engineer_path.exists():
        pytest.skip("feature_engineer.py not found")
    
    with open(feature_engineer_path, 'r') as f:
        content = f.read()
    
    # Should not have hardcoded lookback_map = {"30D": 60, ...}
    # Look for the pattern but allow it in comments
    lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
    code_content = '\n'.join(lines)
    
    # Pattern: lookback_map = {
    if 'lookback_map = {' in code_content:
        # If it exists, should use HORIZON_LOOKBACK_MAP
        assert 'HORIZON_LOOKBACK_MAP' in content, \
            "Should use HORIZON_LOOKBACK_MAP from magic_numbers_config"
    
    print("✓ F42-20: No hardcoded lookback map")


# ============================================================================
# Test Suite 8: Documentation Quality
# ============================================================================

def test_f42_constants_have_docstrings():
    """F42-21: Constants have documentation."""
    import magic_numbers_config as mnc
    
    # Check module docstring
    assert mnc.__doc__ is not None
    assert len(mnc.__doc__) > 100
    assert "Magic numbers" in mnc.__doc__ or "magic numbers" in mnc.__doc__
    
    print("✓ F42-21: Module has comprehensive docstring")


def test_f42_functions_have_docstrings():
    """F42-22: Utility functions have documentation."""
    assert get_horizon_lookback.__doc__ is not None
    assert get_trading_days_for_period.__doc__ is not None
    assert validate_quantile.__doc__ is not None
    
    # Check they contain examples
    assert "Example" in get_horizon_lookback.__doc__
    assert "Example" in get_trading_days_for_period.__doc__
    assert "Example" in validate_quantile.__doc__
    
    print("✓ F42-22: All functions have documented examples")


# ============================================================================
# Test Suite 9: Consistency Checks
# ============================================================================

def test_f42_trading_days_consistency():
    """F42-23: Trading days constants are internally consistent."""
    # Quarter should be ~3 months
    assert abs(TRADING_DAYS_PER_QUARTER_APPROX - (TRADING_DAYS_PER_MONTH_APPROX * 3)) <= 1
    
    # Year should be ~12 months
    assert abs(TRADING_DAYS_PER_YEAR_APPROX - (TRADING_DAYS_PER_MONTH_APPROX * 12)) <= 1
    
    print("✓ F42-23: Trading days constants internally consistent")


def test_f42_horizon_lookback_consistency():
    """F42-24: Horizon lookback periods scale reasonably."""
    # 3M should have ~2x the lookback of 30D
    assert 1.8 <= LOOKBACK_PERIOD_3M / LOOKBACK_PERIOD_30D <= 2.2
    
    # 6M should have ~4x the lookback of 30D
    assert 3.5 <= LOOKBACK_PERIOD_6M / LOOKBACK_PERIOD_30D <= 4.5
    
    # 1Y should have ~8x the lookback of 30D
    assert 7.0 <= LOOKBACK_PERIOD_1Y / LOOKBACK_PERIOD_30D <= 9.0
    
    print("✓ F42-24: Lookback periods scale reasonably")


def test_f42_quantile_sum_logic():
    """F42-25: Resistance and support quantiles make sense."""
    # High resistance should be higher quantile than low resistance
    assert RESISTANCE_HIGH_QUANTILE > RESISTANCE_LOW_QUANTILE
    
    # Support quantiles should be in lower range
    assert SUPPORT_HIGH_QUANTILE < 0.5
    assert SUPPORT_LOW_QUANTILE <= SUPPORT_HIGH_QUANTILE
    
    # Resistance quantiles should be in upper range
    assert RESISTANCE_LOW_QUANTILE > 0.5
    
    print("✓ F42-25: Quantile relationships logical")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
