"""
magic_numbers_config.py

Centralized configuration for numeric constants that were previously hardcoded
as "magic numbers" throughout the codebase.

Part of PY-01 F42: Magic numbers (2 pts, P2)

Magic numbers are numeric literals hardcoded in code without explanation.
They make code harder to understand, maintain, and modify. This module
extracts all such numbers into named constants with clear documentation.

Author: Engineering Team
Date: 2026-09-14
"""

# ===========================================================================
# TIME PERIOD CONSTANTS
# ===========================================================================

# Conversion factors
MINUTES_PER_HOUR = 60
"""Minutes in one hour - used for time interval conversions."""

# Approxim trading days in various periods
TRADING_DAYS_PER_MONTH_APPROX = 21
"""
Approximate trading days in one month.
Actual: ~20-22 depending on month and holidays.
Used for monthly return calculations.
"""

TRADING_DAYS_PER_QUARTER_APPROX = 63
"""
Approximate trading days in one quarter (3 months).
Actual: ~60-65 depending on holidays.
Used for quarterly return calculations.
"""

TRADING_DAYS_PER_YEAR_APPROX = 252
"""
Approximate trading days in one year.
Standard assumption in quantitative finance.
Actual: ~250-253 depending on year and holidays.
"""

# ===========================================================================
# TECHNICAL INDICATOR DEFAULTS
# ===========================================================================

# RSI (Relative Strength Index)
RSI_NEUTRAL_VALUE = 50.0
"""
Neutral RSI value when both gains and losses are zero.
RSI ranges from 0-100, with 50 being perfectly neutral.
"""

# Moving Averages
DEFAULT_MA_PERIOD = 50
"""
Default moving average period when horizon is not specified.
50-period is a common medium-term moving average.
"""

# ===========================================================================
# HORIZON-SPECIFIC LOOKBACK PERIODS
# ===========================================================================

LOOKBACK_PERIOD_30D = 60
"""
Lookback period (in bars) for 30-day horizon support/resistance calculation.
60 bars = ~2 months of daily data for context.
"""

LOOKBACK_PERIOD_3M = 120
"""
Lookback period (in bars) for 3-month horizon support/resistance calculation.
120 bars = ~6 months of daily data for context.
"""

LOOKBACK_PERIOD_6M = 252
"""
Lookback period (in bars) for 6-month horizon support/resistance calculation.
252 bars = ~1 year of daily data for context.
"""

LOOKBACK_PERIOD_1Y = 500
"""
Lookback period (in bars) for 1-year horizon support/resistance calculation.
500 bars = ~2 years of daily data for context.
"""

DEFAULT_LOOKBACK_PERIOD = LOOKBACK_PERIOD_3M
"""
Default lookback period when horizon is not specified or not in lookup table.
120 bars provides reasonable balance between history and responsiveness.
"""

# Convenience dictionary for horizon-based lookback
HORIZON_LOOKBACK_MAP = {
    "30D": LOOKBACK_PERIOD_30D,
    "3M": LOOKBACK_PERIOD_3M,
    "6M": LOOKBACK_PERIOD_6M,
    "1Y": LOOKBACK_PERIOD_1Y,
}

# ===========================================================================
# SUPPORT/RESISTANCE CALCULATION PARAMETERS
# ===========================================================================

# Quantile thresholds for support/resistance bands
RESISTANCE_HIGH_QUANTILE = 0.90
"""
90th percentile of High prices defines upper resistance level.
Strong resistance: only 10% of historical highs are above this.
"""

RESISTANCE_LOW_QUANTILE = 0.75
"""
75th percentile of High prices defines lower resistance level.
Moderate resistance: 25% of historical highs are above this.
"""

SUPPORT_HIGH_QUANTILE = 0.25
"""
25th percentile of Low prices defines upper support level.
Moderate support: 75% of historical lows are below this.
"""

SUPPORT_LOW_QUANTILE = 0.00
"""
Minimum (0th percentile) of Low prices defines lower support level.
This is the absolute historical low in the window.
Note: Using min() is equivalent to 0.0 quantile.
"""

# Threshold multipliers for resistance band breakout detection
RESISTANCE_BAND_LOWER_THRESHOLD = 0.995
"""
Lower threshold for resistance band (99.5% of upper resistance).
Price below this but near resistance is not flagged as breakout.
Used to create a band rather than a single line.
"""

RESISTANCE_BAND_UPPER_THRESHOLD = 1.005
"""
Upper threshold for resistance band (100.5% of upper resistance).
Price above this is flagged as significant resistance breakout.
0.5% above resistance is considered a confirmed breakout.
"""

# ===========================================================================
# TEST & VALIDATION CONSTANTS
# ===========================================================================

# Future-mutation invariance test parameters
MUTATION_TEST_PRICE_DELTA = 500.0
"""
Price change applied to last bar in future-mutation invariance test.
Large enough to be significant, arbitrary value for testing lookahead.
"""

MUTATION_TEST_VOLUME_MULTIPLIER = 20.0
"""
Volume multiplier applied to last bar in future-mutation invariance test.
20x volume is clearly extreme, used to verify no lookahead.
"""

# Synthetic data generation for tests
DEFAULT_TEST_DAYS = 5
"""
Default number of trading days to generate in synthetic test data.
5 days provides enough data for most feature calculations.
"""

DEFAULT_TEST_BARS_PER_DAY = 75
"""
Default number of 5-minute bars per trading day (6.25 hours * 12 bars/hour).
Standard NSE trading session: 9:15 AM - 3:30 PM = 6.25 hours.
"""

LONG_TEST_DAYS = 70
"""
Extended test period for features requiring longer history.
70 days provides ~65 trading days, enough for quarterly returns.
"""

# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def get_horizon_lookback(horizon: str) -> int:
    """
    Get appropriate lookback period for a given horizon.
    
    Args:
        horizon: Horizon identifier ("30D", "3M", "6M", "1Y", etc.)
    
    Returns:
        Lookback period in bars
    
    Example:
        >>> get_horizon_lookback("3M")
        120
        >>> get_horizon_lookback("UNKNOWN")
        120  # Returns default
    """
    return HORIZON_LOOKBACK_MAP.get(horizon, DEFAULT_LOOKBACK_PERIOD)


def get_trading_days_for_period(period: str) -> int:
    """
    Get approximate trading days for common period labels.
    
    Args:
        period: Period identifier ("1M", "1Q", "1Y", etc.)
    
    Returns:
        Approximate trading days
    
    Raises:
        ValueError: If period is not recognized
    
    Example:
        >>> get_trading_days_for_period("1M")
        21
        >>> get_trading_days_for_period("1Q")
        63
    """
    period_map = {
        "1M": TRADING_DAYS_PER_MONTH_APPROX,
        "1Q": TRADING_DAYS_PER_QUARTER_APPROX,
        "3M": TRADING_DAYS_PER_QUARTER_APPROX,
        "1Y": TRADING_DAYS_PER_YEAR_APPROX,
    }
    
    if period not in period_map:
        raise ValueError(
            f"Unknown period: {period}. "
            f"Supported: {list(period_map.keys())}"
        )
    
    return period_map[period]


def validate_quantile(value: float, name: str = "quantile") -> None:
    """
    Validate that a value is a valid quantile (0.0 to 1.0).
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
    
    Raises:
        ValueError: If value is not in [0.0, 1.0]
    
    Example:
        >>> validate_quantile(0.75, "resistance_quantile")
        # No error
        >>> validate_quantile(1.5, "bad_quantile")
        ValueError: bad_quantile must be between 0.0 and 1.0, got 1.5
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{name} must be between 0.0 and 1.0, got {value}"
        )


# ===========================================================================
# MODULE SELF-TEST
# ===========================================================================

def _self_test():
    """Self-test to verify all constants and functions work correctly."""
    print("\n" + "=" * 80)
    print("MAGIC NUMBERS CONFIG SELF-TEST")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Time conversion
    print("\nTest 1: Time conversion constants")
    assert MINUTES_PER_HOUR == 60
    assert TRADING_DAYS_PER_MONTH_APPROX == 21
    assert TRADING_DAYS_PER_QUARTER_APPROX == 63
    assert TRADING_DAYS_PER_YEAR_APPROX == 252
    print("  ✓ All time conversion constants correct")
    tests_passed += 1
    
    # Test 2: Technical indicator defaults
    print("\nTest 2: Technical indicator defaults")
    assert RSI_NEUTRAL_VALUE == 50.0
    assert DEFAULT_MA_PERIOD == 50
    print("  ✓ Technical indicator defaults correct")
    tests_passed += 1
    
    # Test 3: Lookback periods
    print("\nTest 3: Lookback periods")
    assert LOOKBACK_PERIOD_30D == 60
    assert LOOKBACK_PERIOD_3M == 120
    assert LOOKBACK_PERIOD_6M == 252
    assert LOOKBACK_PERIOD_1Y == 500
    assert DEFAULT_LOOKBACK_PERIOD == 120
    print("  ✓ All lookback periods correct")
    tests_passed += 1
    
    # Test 4: Horizon lookback function
    print("\nTest 4: get_horizon_lookback() function")
    assert get_horizon_lookback("30D") == 60
    assert get_horizon_lookback("3M") == 120
    assert get_horizon_lookback("6M") == 252
    assert get_horizon_lookback("1Y") == 500
    assert get_horizon_lookback("UNKNOWN") == 120  # Default
    print("  ✓ get_horizon_lookback() works correctly")
    tests_passed += 1
    
    # Test 5: Trading days function
    print("\nTest 5: get_trading_days_for_period() function")
    assert get_trading_days_for_period("1M") == 21
    assert get_trading_days_for_period("1Q") == 63
    assert get_trading_days_for_period("3M") == 63
    assert get_trading_days_for_period("1Y") == 252
    print("  ✓ get_trading_days_for_period() works correctly")
    tests_passed += 1
    
    # Test 6: Trading days function error handling
    print("\nTest 6: get_trading_days_for_period() error handling")
    try:
        get_trading_days_for_period("INVALID")
        print("  ✗ Should have raised ValueError")
        tests_failed += 1
    except ValueError as e:
        assert "Unknown period" in str(e)
        print("  ✓ Correctly raises ValueError for invalid period")
        tests_passed += 1
    
    # Test 7: Quantile validation
    print("\nTest 7: validate_quantile() function")
    try:
        validate_quantile(0.5)
        validate_quantile(0.0)
        validate_quantile(1.0)
        print("  ✓ Valid quantiles pass validation")
        tests_passed += 1
    except ValueError:
        print("  ✗ Valid quantiles should not raise error")
        tests_failed += 1
    
    # Test 8: Quantile validation error handling
    print("\nTest 8: validate_quantile() error handling")
    try:
        validate_quantile(1.5)
        print("  ✗ Should have raised ValueError for quantile > 1.0")
        tests_failed += 1
    except ValueError:
        print("  ✓ Correctly raises ValueError for invalid quantile")
        tests_passed += 1
    
    # Test 9: Support/resistance quantiles
    print("\nTest 9: Support/resistance quantiles")
    assert 0.0 <= RESISTANCE_HIGH_QUANTILE <= 1.0
    assert 0.0 <= RESISTANCE_LOW_QUANTILE <= 1.0
    assert 0.0 <= SUPPORT_HIGH_QUANTILE <= 1.0
    assert 0.0 <= SUPPORT_LOW_QUANTILE <= 1.0
    assert RESISTANCE_HIGH_QUANTILE > RESISTANCE_LOW_QUANTILE
    assert RESISTANCE_LOW_QUANTILE > SUPPORT_HIGH_QUANTILE
    print("  ✓ All support/resistance quantiles valid and ordered")
    tests_passed += 1
    
    # Test 10: Resistance band thresholds
    print("\nTest 10: Resistance band thresholds")
    assert RESISTANCE_BAND_LOWER_THRESHOLD < 1.0
    assert RESISTANCE_BAND_UPPER_THRESHOLD > 1.0
    assert RESISTANCE_BAND_UPPER_THRESHOLD > RESISTANCE_BAND_LOWER_THRESHOLD
    print("  ✓ Resistance band thresholds valid")
    tests_passed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"\n✗ {tests_failed} TEST(S) FAILED")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
