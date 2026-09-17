"""
Horizon Configuration Validator

Validates that horizon configurations are consistent with market calendar reality:
- Trading days vs calendar days
- Bar counts match expected business days
- Interval/horizon combinations are valid

Part of F27 fix: Inconsistent calendar and bar validation
"""

import logging
from datetime import date, datetime, timedelta
from typing import TypedDict

from config import (
    ALL_HORIZONS,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    MARKET_TIMEZONE,
)
from market_calendar import MarketCalendar

logger = logging.getLogger(__name__)


class HorizonValidationResult(TypedDict):
    """Result of horizon configuration validation."""

    horizon: str
    is_valid: bool
    expected_bars: int | None
    actual_bars: int
    expected_calendar_days: int | None
    discrepancy: str | None
    recommendation: str | None


class HorizonValidator:
    """
    Validates horizon configurations against market calendar reality.

    Key validation rules:
    1. Intraday horizons: bar_count × bar_interval should make sense
    2. Daily horizons: horizon_bars should account for weekends/holidays
    3. Calendar day labels should match trading day reality
    """

    # Expected average trading days per calendar period (NSE historical average)
    TRADING_DAYS_PER_WEEK = 5.0
    TRADING_DAYS_PER_MONTH = 21.0  # ~21-22 trading days typical
    TRADING_DAYS_PER_QUARTER = 63.0  # ~63 trading days
    TRADING_DAYS_PER_HALF_YEAR = 126.0  # ~126 trading days
    TRADING_DAYS_PER_YEAR = 252.0  # Standard equity market year

    # Tolerance for validation (bars can vary ±tolerance from expected)
    BAR_COUNT_TOLERANCE = 2  # Allow ±2 bars difference

    def __init__(self, market_calendar: MarketCalendar | None = None):
        """Initialize validator with market calendar."""
        self.calendar = market_calendar or MarketCalendar()

    def count_trading_days(self, start_date: date, calendar_days: int) -> int:
        """
        Count actual trading days within a calendar period.

        Args:
            start_date: Starting date
            calendar_days: Number of calendar days to look forward

        Returns:
            Number of trading days in the period
        """
        trading_days = 0
        current = start_date

        for _ in range(calendar_days):
            if self.calendar.is_trading_day(current):
                trading_days += 1
            current += timedelta(days=1)

        return trading_days

    def get_expected_bars_for_horizon(self, horizon: str) -> tuple[int | None, int | None]:
        """
        Get expected bar count and calendar days for a horizon label.

        Returns:
            (expected_bars, expected_calendar_days) or (None, None) if can't determine

        Examples:
            "3D" -> (2, 3)  # 3 calendar days ~ 2 trading days (accounts for weekend)
            "7D" -> (5, 7)  # 7 calendar days ~ 5 trading days (1 week)
            "30D" -> (21, 30)  # 30 calendar days ~ 21 trading days
            "INTRADAY" -> (None, None)  # Depends on bar_interval
        """
        horizon = horizon.upper()

        # Intraday is special - depends on bar_interval
        if horizon == "INTRADAY":
            return (None, None)

        # Parse horizon string for calendar days
        if horizon.endswith("D"):
            try:
                calendar_days = int(horizon[:-1])

                # Estimate trading days
                # Simple heuristic: trading_days ≈ calendar_days × (5/7) for daily
                # But more accurate to account for month/year patterns
                if calendar_days <= 10:
                    # Short horizons: assume one weekend in the period
                    expected_trading = max(1, calendar_days - 2 * (calendar_days // 7))
                elif calendar_days <= 40:
                    # Month-ish: use monthly average
                    expected_trading = int(calendar_days * (self.TRADING_DAYS_PER_MONTH / 30))
                else:
                    # Longer: use 5/7 rule
                    expected_trading = int(calendar_days * (5.0 / 7.0))

                return (expected_trading, calendar_days)
            except ValueError:
                return (None, None)

        # Month horizons
        if horizon.endswith("M"):
            try:
                months = int(horizon[:-1])
                calendar_days = months * 30  # Approximate
                expected_bars = int(months * self.TRADING_DAYS_PER_MONTH)
                return (expected_bars, calendar_days)
            except ValueError:
                return (None, None)

        # Year horizons
        if horizon.endswith("Y"):
            try:
                years = int(horizon[:-1])
                calendar_days = years * 365
                expected_bars = int(years * self.TRADING_DAYS_PER_YEAR)
                return (expected_bars, calendar_days)
            except ValueError:
                return (None, None)

        return (None, None)

    def validate_horizon_config(self, horizon: str) -> HorizonValidationResult:
        """
        Validate a single horizon configuration.

        Checks:
        1. Does horizon_bars make sense for the horizon label?
        2. Is bar_interval appropriate for the horizon?
        3. Are trading days vs calendar days accounted for?

        Returns:
            HorizonValidationResult with validation details
        """
        if horizon not in HORIZON_CONFIG:
            return HorizonValidationResult(
                horizon=horizon,
                is_valid=False,
                expected_bars=None,
                actual_bars=0,
                expected_calendar_days=None,
                discrepancy=f"Horizon '{horizon}' not found in HORIZON_CONFIG",
                recommendation="Add horizon to HORIZON_CONFIG",
            )

        config = HORIZON_CONFIG[horizon]
        actual_bars = config["horizon_bars"]
        bar_interval = config["bar_interval"]

        expected_bars, expected_calendar_days = self.get_expected_bars_for_horizon(horizon)

        # Intraday validation
        if horizon == HORIZON_INTRADAY:
            # For intraday, validate that bar_interval makes sense
            if bar_interval == "5m":
                # 6 bars × 5m = 30 minutes (reasonable intraday horizon)
                is_valid = True
                discrepancy = None
                recommendation = None
            else:
                is_valid = False
                discrepancy = f"Unexpected bar_interval '{bar_interval}' for INTRADAY"
                recommendation = "INTRADAY typically uses '5m' interval"

            return HorizonValidationResult(
                horizon=horizon,
                is_valid=is_valid,
                expected_bars=None,  # Depends on desired minutes
                actual_bars=actual_bars,
                expected_calendar_days=None,
                discrepancy=discrepancy,
                recommendation=recommendation,
            )

        # Daily+ horizons validation
        if expected_bars is None:
            return HorizonValidationResult(
                horizon=horizon,
                is_valid=False,
                expected_bars=None,
                actual_bars=actual_bars,
                expected_calendar_days=expected_calendar_days,
                discrepancy=f"Cannot determine expected bars for horizon label '{horizon}'",
                recommendation="Use standard horizon labels (e.g., 3D, 7D, 30D, 3M, 6M, 1Y)",
            )

        # Check if actual bars are within tolerance of expected
        diff = abs(actual_bars - expected_bars)
        is_valid = diff <= self.BAR_COUNT_TOLERANCE

        if not is_valid:
            discrepancy = (
                f"horizon_bars={actual_bars} does not match expected ~{expected_bars} trading days "
                f"for '{horizon}' (difference: {diff} bars, tolerance: {self.BAR_COUNT_TOLERANCE})"
            )
            if actual_bars > expected_bars:
                recommendation = (
                    f"Reduce horizon_bars to ~{expected_bars} to match {expected_calendar_days} calendar days, "
                    f"or rename horizon to better reflect {actual_bars} trading days"
                )
            else:
                recommendation = (
                    f"Increase horizon_bars to ~{expected_bars} to match {expected_calendar_days} calendar days, "
                    f"or rename horizon to better reflect {actual_bars} trading days"
                )
        else:
            discrepancy = None
            recommendation = None

        # Check bar_interval appropriateness
        if bar_interval != "1d" and expected_calendar_days and expected_calendar_days >= 3:
            discrepancy = (discrepancy or "") + f" | bar_interval='{bar_interval}' should be '1d' for multi-day horizons"
            is_valid = False

        return HorizonValidationResult(
            horizon=horizon,
            is_valid=is_valid,
            expected_bars=expected_bars,
            actual_bars=actual_bars,
            expected_calendar_days=expected_calendar_days,
            discrepancy=discrepancy,
            recommendation=recommendation,
        )

    def validate_all_horizons(self) -> dict[str, HorizonValidationResult]:
        """
        Validate all configured horizons.

        Returns:
            Dictionary mapping horizon name to validation result
        """
        results = {}
        for horizon in ALL_HORIZONS:
            results[horizon] = self.validate_horizon_config(horizon)
        return results

    def print_validation_report(self) -> None:
        """Print a human-readable validation report."""
        print("\n" + "=" * 80)
        print("HORIZON CONFIGURATION VALIDATION REPORT")
        print("=" * 80)

        results = self.validate_all_horizons()
        all_valid = all(r["is_valid"] for r in results.values())

        for horizon in ALL_HORIZONS:
            result = results[horizon]
            status = "✓ PASS" if result["is_valid"] else "✗ FAIL"

            print(f"\n{status} | {horizon}")
            print(f"  Configured: {result['actual_bars']} bars")

            if result["expected_bars"] is not None:
                print(f"  Expected: ~{result['expected_bars']} bars ({result['expected_calendar_days']} calendar days)")

            if result["discrepancy"]:
                print(f"  ⚠️  Issue: {result['discrepancy']}")

            if result["recommendation"]:
                print(f"  💡 Recommendation: {result['recommendation']}")

        print("\n" + "=" * 80)
        if all_valid:
            print("STATUS: ✓ ALL HORIZONS VALID")
        else:
            invalid_count = sum(1 for r in results.values() if not r["is_valid"])
            print(f"STATUS: ✗ {invalid_count}/{len(ALL_HORIZONS)} HORIZONS HAVE ISSUES")
        print("=" * 80 + "\n")

    def compute_actual_trading_days_sample(self, horizon: str, sample_date: date | None = None) -> int:
        """
        For a given horizon, compute actual trading days from a sample start date.

        This provides empirical validation of the configured horizon_bars.

        Args:
            horizon: Horizon key (e.g., "7D", "30D")
            sample_date: Starting date (defaults to today)

        Returns:
            Actual trading days counted from sample_date
        """
        if sample_date is None:
            sample_date = datetime.now().date()

        expected_bars, calendar_days = self.get_expected_bars_for_horizon(horizon)

        if calendar_days is None:
            # Intraday doesn't have calendar days
            return 0

        return self.count_trading_days(sample_date, calendar_days)


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    from config import configure_logging

    configure_logging(log_filename="horizon_validator_selftest.log")
    logger.info("Running horizon_validator.py self-test...")

    validator = HorizonValidator()

    # Print validation report
    validator.print_validation_report()

    # Test specific horizons
    print("\n=== DETAILED VALIDATION TESTS ===\n")

    # Test 7D horizon
    result_7d = validator.validate_horizon_config("7D")
    print(f"7D validation: {result_7d}")
    assert result_7d["horizon"] == "7D"
    assert result_7d["actual_bars"] == 5

    # Test 30D horizon
    result_30d = validator.validate_horizon_config("30D")
    print(f"\n30D validation: {result_30d}")
    assert result_30d["horizon"] == "30D"
    assert result_30d["actual_bars"] == 21

    # Test 1Y horizon
    result_1y = validator.validate_horizon_config("1Y")
    print(f"\n1Y validation: {result_1y}")
    assert result_1y["horizon"] == "1Y"
    assert result_1y["actual_bars"] == 252

    # Test actual trading days counting
    print("\n=== TRADING DAYS COUNTING TEST ===\n")
    test_date = date(2026, 9, 14)  # Monday
    trading_days_7 = validator.count_trading_days(test_date, 7)
    print(f"Trading days in 7 calendar days from 2026-09-14 (Mon): {trading_days_7} (expect ~5)")
    assert 4 <= trading_days_7 <= 5, f"Expected 4-5 trading days, got {trading_days_7}"

    trading_days_30 = validator.count_trading_days(test_date, 30)
    print(f"Trading days in 30 calendar days from 2026-09-14: {trading_days_30} (expect ~21)")
    assert 19 <= trading_days_30 <= 23, f"Expected ~21 trading days, got {trading_days_30}"

    print("\n✓ All tests passed")
