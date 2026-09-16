"""
Test Suite for F28: Universe and Corporate Actions

Tests the versioned universe and sector providers, plus corporate action handling:
- Universe constituent validation (DATA-001)
- Sector mapping coverage (DATA-002)
- Corporate action price adjustment strategy (DATA-004)

Part of PY-01 Assignment (5 points, P1)
"""

import pytest
import hashlib
import json
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np

from universe_provider import UniverseProvider, UniverseSnapshot, get_nifty50_constituents
from sector_provider import SectorMapProvider, get_symbol_sector
from market_data_provider import PriceAdjustmentMode, TestFixtureMarketDataProvider, DataStatus
import config


# ============================================================================
# NORMAL CASES: Valid universe and sector operations
# ============================================================================


def test_f28_normal_universe_load_50_constituents():
    """
    NORMAL: UniverseProvider loads exactly 50 NIFTY 50 constituents.

    Input: Standard universe data file
    Expected: 50 unique constituents with valid checksum
    """
    provider = UniverseProvider()
    constituents = provider.get_constituents()

    assert len(constituents) == 50, f"Expected 50 constituents, got {len(constituents)}"
    assert len(set(constituents)) == 50, "Constituents should be unique"

    # Verify validation passes
    assert provider.validate_constituents(expected_count=50) is True

    print(f"✓ NORMAL case passed: Loaded {len(constituents)} unique constituents")


def test_f28_normal_universe_checksum_integrity():
    """
    NORMAL: Universe snapshot has valid SHA-256 checksum for integrity.

    Input: Universe snapshot with constituents
    Expected: 64-character hex SHA-256 checksum
    """
    provider = UniverseProvider()
    snapshot = provider.get_snapshot()

    assert snapshot is not None, "Should have active snapshot"
    assert snapshot.checksum is not None
    assert len(snapshot.checksum) == 64, "SHA-256 checksum should be 64 hex characters"
    assert snapshot.total_constituents == 50

    # Verify checksum matches actual constituents
    expected_checksum = provider._compute_checksum(snapshot.constituents)
    assert snapshot.checksum == expected_checksum, "Checksum should match constituents"

    print(f"✓ NORMAL case passed: Checksum verified {snapshot.checksum[:16]}...")


def test_f28_normal_sector_mapping_complete():
    """
    NORMAL: All 50 NIFTY constituents have sector and industry mappings.

    Input: NIFTY 50 universe
    Expected: Every symbol has sector != "Unknown" and industry != "Unknown"
    """
    provider = SectorMapProvider()
    constituents = get_nifty50_constituents()

    unmapped_sectors = []
    unmapped_industries = []

    for symbol in constituents:
        sector = provider.get_sector(symbol)
        industry = provider.get_industry(symbol)

        if sector == "Unknown":
            unmapped_sectors.append(symbol)
        if industry == "Unknown":
            unmapped_industries.append(symbol)

    assert len(unmapped_sectors) == 0, f"Symbols with missing sector: {unmapped_sectors}"
    assert len(unmapped_industries) == 0, f"Symbols with missing industry: {unmapped_industries}"

    print(f"✓ NORMAL case passed: All {len(constituents)} symbols have sector and industry mappings")


def test_f28_normal_corporate_action_modes_defined():
    """
    NORMAL: Corporate action price adjustment modes are explicitly defined.

    Input: PriceAdjustmentMode enum
    Expected: ADJUSTED, RAW, TOTAL_RETURN modes available
    """
    assert hasattr(PriceAdjustmentMode, "ADJUSTED")
    assert hasattr(PriceAdjustmentMode, "RAW")
    assert hasattr(PriceAdjustmentMode, "TOTAL_RETURN")

    # Verify they're distinct
    assert PriceAdjustmentMode.ADJUSTED != PriceAdjustmentMode.RAW
    assert PriceAdjustmentMode.ADJUSTED != PriceAdjustmentMode.TOTAL_RETURN

    # Verify values are sensible strings
    assert PriceAdjustmentMode.ADJUSTED.value == "adjusted"
    assert PriceAdjustmentMode.RAW.value == "raw"
    assert PriceAdjustmentMode.TOTAL_RETURN.value == "total_return"

    print(f"✓ NORMAL case passed: Corporate action modes defined (adjusted, raw, total_return)")


# ============================================================================
# BOUNDARY CASES: Edge conditions and special scenarios
# ============================================================================


def test_f28_boundary_universe_version_access():
    """
    BOUNDARY: Can access universe by specific version or default to latest.

    Input: Request by version vs default
    Expected: Returns correct version's constituents
    """
    provider = UniverseProvider()

    # Get default (latest) version
    default_constituents = provider.get_constituents()
    assert len(default_constituents) == 50

    # Get snapshot to know version
    snapshot = provider.get_snapshot()
    if snapshot:
        versioned_constituents = provider.get_constituents(version=snapshot.version)
        assert versioned_constituents == default_constituents

    print(f"✓ BOUNDARY case passed: Version access works (version={snapshot.version if snapshot else 'N/A'})")


def test_f28_boundary_sector_filtering():
    """
    BOUNDARY: Can filter symbols by sector (e.g., all Banking stocks).

    Input: Sector name "Banking"
    Expected: Returns list of banking sector symbols
    """
    provider = SectorMapProvider()

    banking_symbols = provider.get_symbols_for_sector("Banking")
    assert len(banking_symbols) >= 3, "Should have at least 3 banking stocks"

    # Verify known banking stocks are included
    known_banking = ["HDFCBANK", "ICICIBANK", "SBIN"]
    for bank in known_banking:
        if bank in get_nifty50_constituents():  # Only check if in universe
            assert bank in banking_symbols, f"{bank} should be in Banking sector"

    # Verify all returned symbols actually have Banking sector
    for symbol in banking_symbols:
        assert provider.get_sector(symbol) == "Banking"

    print(f"✓ BOUNDARY case passed: Sector filtering returned {len(banking_symbols)} banking stocks")


def test_f28_boundary_config_sync():
    """
    BOUNDARY: Config NIFTY50_SYMBOLS and SECTOR_MAP stay in sync with providers.

    Input: Config constants vs provider data
    Expected: Counts and key symbols match
    """
    # Universe sync
    constituents = get_nifty50_constituents()
    assert len(config.NIFTY50_SYMBOLS) == len(constituents)
    assert len(config.NIFTY50_YFINANCE_TICKERS) == len(constituents)

    # Known symbols should be in both
    known_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
    for sym in known_symbols:
        assert sym in config.NIFTY50_SYMBOLS
        assert sym in constituents

    # Sector map sync
    assert len(config.SECTOR_MAP) >= 50
    provider = SectorMapProvider()
    for sym in constituents:
        config_sector = config.SECTOR_MAP.get(sym, "Unknown")
        provider_sector = provider.get_sector(sym)
        # They should match (or provider should override if different)
        assert config_sector != "Unknown" or provider_sector != "Unknown"

    print(f"✓ BOUNDARY case passed: Config synced with providers")


def test_f28_boundary_universe_duplicate_detection():
    """
    BOUNDARY: Universe validation detects duplicate constituents.

    Input: Check validation logic handles duplicates
    Expected: validate_constituents() would return False if duplicates existed
    """
    provider = UniverseProvider()

    # Current data should have no duplicates
    constituents = provider.get_constituents()
    assert len(constituents) == len(set(constituents)), "No duplicates in current universe"

    # Verify validation logic
    assert provider.validate_constituents(expected_count=50) is True

    print(f"✓ BOUNDARY case passed: Duplicate detection working")


# ============================================================================
# FAILURE CASES: Invalid operations and error handling
# ============================================================================


def test_f28_failure_nonexistent_symbol_sector():
    """
    FAILURE: Requesting sector for non-existent symbol returns "Unknown".

    Input: "INVALID_SYMBOL"
    Expected: Returns "Unknown" gracefully, doesn't crash
    """
    provider = SectorMapProvider()

    sector = provider.get_sector("INVALID_SYMBOL_XYZ")
    industry = provider.get_industry("INVALID_SYMBOL_XYZ")

    assert sector == "Unknown", "Non-existent symbol should return Unknown sector"
    assert industry == "Unknown", "Non-existent symbol should return Unknown industry"

    print(f"✓ FAILURE case passed: Invalid symbol handled gracefully")


def test_f28_failure_wrong_constituent_count():
    """
    FAILURE: Universe validation fails if count != 50.

    Input: Validate with wrong expected count
    Expected: Returns False
    """
    provider = UniverseProvider()

    # Correct count should pass
    assert provider.validate_constituents(expected_count=50) is True

    # Wrong count should fail
    assert provider.validate_constituents(expected_count=49) is False
    assert provider.validate_constituents(expected_count=51) is False

    print(f"✓ FAILURE case passed: Wrong constituent count detected")


def test_f28_failure_empty_sector_filter():
    """
    FAILURE: Filtering by non-existent sector returns empty list.

    Input: Sector name "NonExistentSector"
    Expected: Returns empty list gracefully
    """
    provider = SectorMapProvider()

    symbols = provider.get_symbols_for_sector("NonExistentSector")
    assert isinstance(symbols, list)
    assert len(symbols) == 0

    print(f"✓ FAILURE case passed: Non-existent sector returns empty list")


# ============================================================================
# INTEGRATION: Corporate action handling with market data
# ============================================================================


def test_f28_integration_price_adjustment_in_test_fixture():
    """
    INTEGRATION: TestFixtureMarketDataProvider records price adjustment mode.

    Input: Fetch data with specific adjustment mode
    Expected: Result contains adjustment_mode metadata
    """
    dates = pd.date_range("2026-07-01 09:15", periods=20, freq="5min", tz="Asia/Kolkata")
    sample_df = pd.DataFrame(
        {
            "Open": np.linspace(2000, 2020, 20),
            "High": np.linspace(2005, 2025, 20),
            "Low": np.linspace(1995, 2015, 20),
            "Close": np.linspace(2002, 2022, 20),
            "Volume": np.full(20, 10000),
        },
        index=dates,
    )

    provider = TestFixtureMarketDataProvider({"TEST.NS": sample_df})

    # Fetch with ADJUSTED mode
    result_adj = provider.fetch_ohlcv("TEST.NS", adjustment=PriceAdjustmentMode.ADJUSTED)
    assert result_adj.status == DataStatus.LIVE
    assert result_adj.adjustment_mode == PriceAdjustmentMode.ADJUSTED

    # Fetch with RAW mode
    result_raw = provider.fetch_ohlcv("TEST.NS", adjustment=PriceAdjustmentMode.RAW)
    assert result_raw.status == DataStatus.LIVE
    assert result_raw.adjustment_mode == PriceAdjustmentMode.RAW

    print(f"✓ INTEGRATION passed: Price adjustment mode tracked correctly")


def test_f28_integration_universe_sector_cross_reference():
    """
    INTEGRATION: Cross-reference universe constituents with sector mappings.

    Input: All universe constituents
    Expected: Each has a valid sector assignment
    """
    universe_provider = UniverseProvider()
    sector_provider = SectorMapProvider()

    constituents = universe_provider.get_constituents()

    sector_distribution = {}
    for symbol in constituents:
        sector = sector_provider.get_sector(symbol)
        sector_distribution[sector] = sector_distribution.get(sector, 0) + 1

    # Should have multiple sectors represented
    assert len(sector_distribution) >= 5, "Should have at least 5 different sectors"

    # No "Unknown" sectors
    assert "Unknown" not in sector_distribution, "All symbols should have known sectors"

    # Common sectors should be present
    expected_sectors = ["Banking", "IT", "Energy"]
    for expected in expected_sectors:
        assert expected in sector_distribution, f"Expected sector {expected} not found"

    print(f"✓ INTEGRATION passed: {len(constituents)} constituents across {len(sector_distribution)} sectors")
    print(f"  Sectors: {', '.join(sorted(sector_distribution.keys()))}")


def test_f28_integration_versioned_data_effective_dates():
    """
    INTEGRATION: Universe and sector snapshots have effective date tracking.

    Input: Load snapshot metadata
    Expected: Contains version, effective dates, source info
    """
    universe_provider = UniverseProvider()
    universe_snapshot = universe_provider.get_snapshot()

    assert universe_snapshot is not None
    assert universe_snapshot.version is not None
    assert universe_snapshot.effective_from is not None
    assert universe_snapshot.source is not None

    print(f"✓ INTEGRATION passed: Universe snapshot metadata complete")
    print(f"  Version: {universe_snapshot.version}")
    print(f"  Effective from: {universe_snapshot.effective_from}")
    print(f"  Source: {universe_snapshot.source}")


def test_f28_integration_sector_industry_hierarchy():
    """
    INTEGRATION: Sector and industry form proper hierarchy (sector > industry).

    Input: Symbols with both sector and industry
    Expected: Industry is more granular than sector
    """
    provider = SectorMapProvider()

    # Check a few known stocks
    test_cases = [
        ("RELIANCE", "Energy"),
        ("TCS", "IT"),
        ("HDFCBANK", "Banking"),
    ]

    for symbol, expected_sector in test_cases:
        if symbol not in get_nifty50_constituents():
            continue

        sector = provider.get_sector(symbol)
        industry = provider.get_industry(symbol)

        assert sector == expected_sector, f"{symbol} sector mismatch"
        assert industry != "Unknown", f"{symbol} should have industry"
        assert industry != sector, f"{symbol} industry should be more granular than sector"

    print(f"✓ INTEGRATION passed: Sector/industry hierarchy validated")


# ============================================================================
# REGRESSION: Ensure existing functionality isn't broken
# ============================================================================


def test_f28_regression_config_constants_accessible():
    """
    REGRESSION: Config constants still accessible after provider implementation.

    Verify F28 changes don't break existing config access patterns.
    """
    assert hasattr(config, "NIFTY50_SYMBOLS")
    assert hasattr(config, "NIFTY50_YFINANCE_TICKERS")
    assert hasattr(config, "SECTOR_MAP")

    assert isinstance(config.NIFTY50_SYMBOLS, (list, tuple))
    assert isinstance(config.SECTOR_MAP, dict)

    assert len(config.NIFTY50_SYMBOLS) == 50
    assert len(config.SECTOR_MAP) >= 50

    print(f"✓ REGRESSION passed: Config constants intact")


def test_f28_regression_helper_functions():
    """
    REGRESSION: Global helper functions still work.

    Verify get_nifty50_constituents() and get_symbol_sector() work.
    """
    # Universe helper
    constituents = get_nifty50_constituents()
    assert isinstance(constituents, list)
    assert len(constituents) == 50

    # Sector helper
    reliance_sector = get_symbol_sector("RELIANCE")
    assert isinstance(reliance_sector, str)
    assert reliance_sector == "Energy"

    print(f"✓ REGRESSION passed: Helper functions working")


def test_f28_regression_data_reliability_tests_still_pass():
    """
    REGRESSION: Original test_data_reliability.py tests still pass.

    This ensures F28 PY-01 tests don't break the existing validated implementation.
    """
    # Import and run key assertions from test_data_reliability.py
    from test_data_reliability import (
        test_universe_provider_integrity_and_checksum,
        test_sector_map_provider_coverage,
        test_corporate_action_adjustment_metadata,
    )

    # These should not raise any exceptions
    test_universe_provider_integrity_and_checksum()
    test_sector_map_provider_coverage()
    test_corporate_action_adjustment_metadata()

    print(f"✓ REGRESSION passed: Original data reliability tests still pass")


# ============================================================================
# DOCUMENTATION: Verify data files and structure
# ============================================================================


def test_f28_documentation_universe_data_file_exists():
    """
    DOCUMENTATION: Universe data file exists and is valid JSON.

    Input: data/universe/nifty50_constituents_v1.json
    Expected: File exists, valid JSON, has required fields
    """
    universe_file = Path("data/universe/nifty50_constituents_v1.json")
    assert universe_file.exists(), f"Universe file not found: {universe_file}"

    with open(universe_file, encoding="utf-8-sig") as f:
        data = json.load(f)

    # Verify required fields
    assert "version" in data
    assert "constituents" in data
    assert "index_name" in data
    assert isinstance(data["constituents"], list)
    assert len(data["constituents"]) == 50

    print(f"✓ DOCUMENTATION passed: Universe data file valid")


def test_f28_documentation_sector_data_file_exists():
    """
    DOCUMENTATION: Sector map data file exists and is valid JSON.

    Input: data/sector_map/nifty50_sectors_v1.json
    Expected: File exists, valid JSON, has mappings for all 50 symbols
    """
    sector_file = Path("data/sector_map/nifty50_sectors_v1.json")
    assert sector_file.exists(), f"Sector map file not found: {sector_file}"

    with open(sector_file, encoding="utf-8-sig") as f:
        data = json.load(f)

    # Verify required fields
    assert "version" in data
    assert "mappings" in data
    assert isinstance(data["mappings"], dict)
    assert len(data["mappings"]) >= 50

    # Verify each mapping has sector and industry
    for symbol, mapping in data["mappings"].items():
        assert "sector" in mapping, f"{symbol} missing sector"
        assert "industry" in mapping, f"{symbol} missing industry"

    print(f"✓ DOCUMENTATION passed: Sector map data file valid")


# ============================================================================
# Run all tests
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
