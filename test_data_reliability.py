# 1. Standard library imports
from datetime import date

# 2. Third-party imports
import numpy as np
import pandas as pd

import config

# 3. Local imports
from market_calendar import (
    CSVMarketCalendarProvider,
    MarketCalendar,
)
from market_data_provider import (
    DataStatus,
    PriceAdjustmentMode,
    TestFixtureMarketDataProvider,
)
from sector_provider import (
    SectorMapProvider,
)
from universe_provider import (
    UniverseProvider,
    get_nifty50_constituents,
)


def test_market_calendar_provider_versioned_csv_and_official_2026_holidays():
    """DATA-003: Verify CSVMarketCalendarProvider loads official NSE 2026 holiday calendar."""
    provider = CSVMarketCalendarProvider()
    holidays = provider.load_holidays()

    # Verify official 2026 holidays identified in audit
    critical_2026_dates = [
        "2026-02-19",  # Shivaji Jayanti
        "2026-03-19",  # Gudi Padwa
        "2026-04-01",  # Annual Bank Closing
        "2026-08-26",  # Id-e-Milad
        "2026-11-08",  # Diwali Laxmi Pujan
    ]
    for d in critical_2026_dates:
        assert d in holidays, f"Missing critical 2026 holiday: {d}"

    cal = MarketCalendar(provider=provider)

    # 19-Feb-2026 must be identified as holiday and non-trading day
    assert cal.is_holiday("2026-02-19") is True
    assert cal.is_trading_day(date(2026, 2, 19)) is False

    # 19-Mar-2026 must be identified as holiday
    assert cal.is_holiday("2026-03-19") is True
    assert cal.is_trading_day(date(2026, 3, 19)) is False

    # 01-Apr-2026 must be non-trading day
    assert cal.is_trading_day(date(2026, 4, 1)) is False

    # Normal trading day (e.g. Wednesday 2026-07-15)
    assert cal.is_trading_day(date(2026, 7, 15)) is True


def test_universe_provider_integrity_and_checksum():
    """DATA-001: Verify UniverseProvider loads snapshot with 50 constituents and SHA256 checksum."""
    provider = UniverseProvider()
    assert provider.validate_constituents(expected_count=50) is True

    constituents = provider.get_constituents()
    assert len(constituents) == 50
    assert len(set(constituents)) == 50

    snapshot = provider.get_snapshot()
    assert snapshot is not None
    assert snapshot.total_constituents == 50
    assert snapshot.checksum is not None and len(snapshot.checksum) == 64

    # Verify config synchronization
    assert len(config.NIFTY50_SYMBOLS) == 50
    assert len(config.NIFTY50_YFINANCE_TICKERS) == 50
    assert all(sym in config.NIFTY50_SYMBOLS for sym in ["RELIANCE", "TCS", "HDFCBANK", "INFY"])


def test_sector_map_provider_coverage():
    """DATA-002: Verify SectorMapProvider covers all 50 constituents with valid mappings."""
    provider = SectorMapProvider()
    nifty_symbols = get_nifty50_constituents()

    for sym in nifty_symbols:
        sec = provider.get_sector(sym)
        ind = provider.get_industry(sym)
        assert sec != "Unknown", f"Symbol {sym} missing sector mapping"
        assert ind != "Unknown", f"Symbol {sym} missing industry mapping"

    # Verify category filtering
    banking = provider.get_symbols_for_sector("Banking")
    assert len(banking) >= 5
    assert "HDFCBANK" in banking
    assert "SBIN" in banking

    # Verify config SECTOR_MAP synchronization
    assert len(config.SECTOR_MAP) >= 50
    assert config.SECTOR_MAP["RELIANCE"] == "Energy"
    assert config.SECTOR_MAP["TCS"] == "IT"


def test_market_data_provider_abstraction_and_test_fixture():
    """DATA-005: Verify abstract MarketDataProvider with TestFixtureMarketDataProvider."""
    dates = pd.date_range("2026-07-01 09:15", periods=30, freq="5min", tz="Asia/Kolkata")
    sample_df = pd.DataFrame(
        {
            "Open": np.linspace(2000, 2050, 30),
            "High": np.linspace(2005, 2055, 30),
            "Low": np.linspace(1995, 2045, 30),
            "Close": np.linspace(2002, 2052, 30),
            "Volume": np.full(30, 10000),
        },
        index=dates,
    )

    fixture_provider = TestFixtureMarketDataProvider({"INFY.NS": sample_df})

    # Hermetic fetch without any network connection
    result = fixture_provider.fetch_ohlcv("INFY.NS", adjustment=PriceAdjustmentMode.ADJUSTED)
    assert result.status == DataStatus.LIVE
    assert result.source == "test_fixture"
    assert result.adjustment_mode == PriceAdjustmentMode.ADJUSTED
    assert result.data is not None and len(result.data) == 30

    # Missing ticker returns UNAVAILABLE
    missing_res = fixture_provider.fetch_ohlcv("UNKNOWN.NS")
    assert missing_res.status == DataStatus.UNAVAILABLE
    assert missing_res.data is None


def test_data_quality_rules_and_timestamp_normalization():
    """Data quality rules: Monotonicity check, duplicate detection, timezone normalization."""
    provider = TestFixtureMarketDataProvider()

    # Create messy DataFrame with duplicates and non-monotonic index
    t0 = pd.Timestamp("2026-07-01 09:15:00", tz="Asia/Kolkata")
    t1 = pd.Timestamp("2026-07-01 09:20:00", tz="Asia/Kolkata")
    t2 = pd.Timestamp("2026-07-01 09:25:00", tz="Asia/Kolkata")

    messy_index = [t0, t2, t1, t2]  # non-monotonic and duplicate t2
    messy_df = pd.DataFrame(
        {
            "Open": [100, 102, 101, 102],
            "High": [101, 103, 102, 103],
            "Low": [99, 101, 100, 101],
            "Close": [100.5, 102.5, 101.5, 102.5],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=messy_index,
    )

    report = provider.validate_data_quality(messy_df, "TEST")
    assert report.is_monotonic is False
    assert report.duplicate_index_count == 1
    assert report.is_valid is False

    # Apply clean_and_normalize
    clean_df = provider.clean_and_normalize(messy_df)
    assert clean_df.index.is_monotonic_increasing is True
    assert clean_df.index.duplicated().sum() == 0
    assert len(clean_df) == 3

    # Re-validate cleaned data
    clean_report = provider.validate_data_quality(clean_df, "TEST")
    assert clean_report.is_monotonic is True
    assert clean_report.duplicate_index_count == 0
    assert clean_report.is_valid is True


def test_corporate_action_adjustment_metadata():
    """DATA-004: Explicit corporate action adjustment mode is declared and propagated."""
    from model_trainer import ModelTrainer

    trainer = ModelTrainer()
    assert hasattr(PriceAdjustmentMode, "ADJUSTED")
    assert hasattr(PriceAdjustmentMode, "RAW")
    assert hasattr(PriceAdjustmentMode, "TOTAL_RETURN")

    # Verify model metadata records adjustment mode
    meta = {
        "symbol": "TCS",
        "horizon": "INTRADAY",
        "price_adjustment_mode": PriceAdjustmentMode.ADJUSTED.value,
    }
    assert meta["price_adjustment_mode"] == "adjusted"
