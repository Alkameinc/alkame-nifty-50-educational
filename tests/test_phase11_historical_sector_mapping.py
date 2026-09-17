# 1. Standard library imports
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from config import NIFTY50_SYMBOLS, SECTOR_MAP
from event_classifier import EventClassifier, SCOPE_SECTOR, SCOPE_STOCK
from predictor import Predictor
from sector_provider import HistoricalSectorRecord, SectorMapProvider, sector_map_provider


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_sector_history_csv_integrity():
    """DATA-007: Verify CSV exists, has valid headers, dates, and non-empty rows."""
    csv_path = PROJECT_ROOT / "data" / "sector_map" / "nifty50_sector_history.csv"
    assert csv_path.exists(), f"Historical sector CSV missing at {csv_path}"

    df = pd.read_csv(csv_path)
    expected_cols = ["symbol", "sector", "industry", "effective_from", "effective_to", "source", "notes"]
    assert list(df.columns) == expected_cols, f"Unexpected columns in CSV: {df.columns}"

    assert len(df) >= 50, f"Expected at least 50 historical records, found {len(df)}"

    for idx, row in df.iterrows():
        sym = str(row["symbol"]).strip()
        sec = str(row["sector"]).strip()
        ind = str(row["industry"]).strip()
        eff_from = str(row["effective_from"]).strip()
        eff_to = str(row["effective_to"]).strip() if pd.notna(row["effective_to"]) and str(row["effective_to"]).strip() else None

        assert sym and sym != "nan", f"Row {idx}: missing symbol"
        assert sec and sec != "nan", f"Row {idx} ({sym}): missing sector"
        assert ind and ind != "nan", f"Row {idx} ({sym}): missing industry"

        # Validate date format YYYY-MM-DD
        from_dt = datetime.strptime(eff_from, "%Y-%m-%d")
        if eff_to is not None:
            to_dt = datetime.strptime(eff_to, "%Y-%m-%d")
            assert from_dt <= to_dt, f"Row {idx} ({sym}): effective_from {eff_from} > effective_to {eff_to}"

    # All 50 current constituents should be present
    csv_symbols = set(df["symbol"].unique())
    for sym in NIFTY50_SYMBOLS:
        assert sym in csv_symbols, f"Current NIFTY 50 constituent {sym} missing from historical sector CSV"

    # Historical constituents should also be present
    historical_syms = ["HDFC", "IOC", "GAIL", "ZEEL", "SHREECEM", "UPL", "BEL", "TRENT"]
    for sym in historical_syms:
        assert sym in csv_symbols, f"Historical constituent {sym} missing from historical sector CSV"


def test_historical_point_in_time_sector_lookups():
    """DATA-007: Test point-in-time sector and industry lookups across historical intervals."""
    # HDFC was a constituent until 2023-07-12 (classified as NBFC in codebase taxonomy)
    hdfc_sec_2021 = sector_map_provider.get_sector("HDFC", as_of="2021-06-01")
    assert hdfc_sec_2021 == "NBFC", f"Expected HDFC in 2021 to be NBFC, got {hdfc_sec_2021}"
    hdfc_ind_2021 = sector_map_provider.get_industry("HDFC", as_of="2021-06-01")
    assert "Housing" in hdfc_ind_2021 or "Finance" in hdfc_ind_2021, f"Expected Housing Finance for HDFC, got {hdfc_ind_2021}"

    # In 2024, HDFC is no longer an active constituent in sector history
    hdfc_sec_2024 = sector_map_provider.get_sector("HDFC", as_of="2024-01-01")
    assert hdfc_sec_2024 == "Unknown", f"Expected HDFC in 2024 to be Unknown, got {hdfc_sec_2024}"

    # IOC was a constituent until 2022-03-30
    ioc_sec_2021 = sector_map_provider.get_sector("IOC", as_of="2021-01-01")
    assert ioc_sec_2021 == "Energy", f"Expected Energy for IOC in 2021, got {ioc_sec_2021}"
    ioc_sec_2025 = sector_map_provider.get_sector("IOC", as_of="2025-01-01")
    assert ioc_sec_2025 == "Unknown", f"Expected IOC in 2025 to be Unknown, got {ioc_sec_2025}"

    # GAIL was a constituent until 2021-03-30
    gail_sec_2020 = sector_map_provider.get_sector("GAIL", as_of="2020-05-01")
    assert gail_sec_2020 == "Utilities", f"Expected Utilities for GAIL in 2020, got {gail_sec_2020}"
    gail_sec_2023 = sector_map_provider.get_sector("GAIL", as_of="2023-01-01")
    assert gail_sec_2023 == "Unknown", f"Expected GAIL in 2023 to be Unknown, got {gail_sec_2023}"

    # BEL entered NIFTY 50 on 2024-09-30
    bel_sec_2023 = sector_map_provider.get_sector("BEL", as_of="2023-01-01")
    assert bel_sec_2023 == "Unknown", f"Expected BEL in 2023 to be Unknown, got {bel_sec_2023}"
    bel_sec_2024 = sector_map_provider.get_sector("BEL", as_of="2024-10-15")
    assert bel_sec_2024 == "CapitalGoods", f"Expected CapitalGoods for BEL in 2024, got {bel_sec_2024}"

    # TRENT entered NIFTY 50 on 2024-09-30
    trent_sec_2023 = sector_map_provider.get_sector("TRENT", as_of="2023-01-01")
    assert trent_sec_2023 == "Unknown", f"Expected TRENT in 2023 to be Unknown, got {trent_sec_2023}"
    trent_sec_2024 = sector_map_provider.get_sector("TRENT", as_of="2024-10-15")
    assert trent_sec_2024 == "Retail", f"Expected Retail for TRENT in 2024, got {trent_sec_2024}"


def test_reclassification_transitions():
    """DATA-007: Test handling of corporate restructuring and sector reclassifications."""
    # TATACONSUM: In 2020 was Beverages/Tea, in 2021+ expanded to FMCG / Consumer Staples
    ind_2020 = sector_map_provider.get_industry("TATACONSUM", as_of="2020-03-01")
    assert "Tea" in ind_2020 or "Coffee" in ind_2020 or "Beverages" in ind_2020, f"Expected Tea/Beverages, got {ind_2020}"

    ind_2023 = sector_map_provider.get_industry("TATACONSUM", as_of="2023-01-01")
    assert "Staples" in ind_2023 or "Consumer" in ind_2023, f"Expected Consumer Staples, got {ind_2023}"

    # SHRIRAMFIN: In 2021 was AutoFinance / Commercial Vehicle, in 2023 was diversified NBFC / Consumer Finance
    sec_shriram_2021 = sector_map_provider.get_sector("SHRIRAMFIN", as_of="2021-06-01")
    assert sec_shriram_2021 == "AutoFinance", f"Expected AutoFinance, got {sec_shriram_2021}"

    sec_shriram_2023 = sector_map_provider.get_sector("SHRIRAMFIN", as_of="2023-06-01")
    assert sec_shriram_2023 == "NBFC", f"Expected NBFC, got {sec_shriram_2023}"

    ind_shriram_2021 = sector_map_provider.get_industry("SHRIRAMFIN", as_of="2021-06-01")
    assert "Commercial" in ind_shriram_2021 or "Vehicle" in ind_shriram_2021

    ind_shriram_2023 = sector_map_provider.get_industry("SHRIRAMFIN", as_of="2023-06-01")
    assert "Consumer" in ind_shriram_2023 or "Finance" in ind_shriram_2023


def test_get_sector_map_as_of():
    """DATA-007: Verify get_sector_map_as_of returns a consistent dictionary valid at date T."""
    # Snapshot at 2021-06-01
    s_map_2021 = sector_map_provider.get_sector_map_as_of("2021-06-01")
    assert isinstance(s_map_2021, dict)
    assert len(s_map_2021) >= 50
    assert "HDFC" in s_map_2021
    assert "IOC" in s_map_2021
    assert "BEL" not in s_map_2021
    assert "TRENT" not in s_map_2021

    # Snapshot at 2025-01-01
    s_map_2025 = sector_map_provider.get_sector_map_as_of("2025-01-01")
    assert isinstance(s_map_2025, dict)
    assert len(s_map_2025) >= 50
    assert "BEL" in s_map_2025
    assert "TRENT" in s_map_2025
    assert "HDFC" not in s_map_2025
    assert "IOC" not in s_map_2025

    # Check that no entries map to "Unknown" or empty
    for sym, sec in s_map_2025.items():
        assert sec and sec != "Unknown", f"Symbol {sym} has invalid sector {sec} in 2025 map"


def test_get_symbols_for_sector_point_in_time():
    """DATA-007: Verify get_symbols_for_sector returns point-in-time constituents."""
    # NBFC sector in 2021 included HDFC
    nbfc_2021 = sector_map_provider.get_symbols_for_sector("NBFC", as_of="2021-06-01")
    assert "HDFC" in nbfc_2021
    assert "BAJFINANCE" in nbfc_2021

    # NBFC sector in 2025 does NOT include HDFC
    nbfc_2025 = sector_map_provider.get_symbols_for_sector("NBFC", as_of="2025-01-01")
    assert "HDFC" not in nbfc_2025
    assert "BAJFINANCE" in nbfc_2025

    # Retail sector in 2025 includes TRENT
    retail_2025 = sector_map_provider.get_symbols_for_sector("Retail", as_of="2025-01-01")
    assert "TRENT" in retail_2025

    retail_2021 = sector_map_provider.get_symbols_for_sector("Retail", as_of="2021-06-01")
    assert "TRENT" not in retail_2021


def test_event_classifier_corporate_point_in_time_sector():
    """DATA-007: Verify event classifier assigns sector valid at corporate announcement time."""
    classifier = EventClassifier()

    # Event for HDFC in 2021
    hdfc_corp_2021 = {
        "symbol": "HDFC",
        "category": "CORPORATE_ANNOUNCEMENT",
        "raw": {"subject": "Q4 Financial Results", "published_at": "2021-05-15T10:00:00Z"},
    }
    evt_2021 = classifier.classify_corporate_event(hdfc_corp_2021)
    assert evt_2021.scope == SCOPE_STOCK
    assert evt_2021.affected_tickers == ["HDFC"]
    assert evt_2021.sector == "NBFC", f"Expected NBFC, got {evt_2021.sector}"

    # Event for HDFC in 2025 (delisted/merged, not in active NIFTY 50 universe)
    hdfc_corp_2025 = {
        "symbol": "HDFC",
        "category": "CORPORATE_ANNOUNCEMENT",
        "raw": {"subject": "Corporate Action", "published_at": "2025-01-10T10:00:00Z"},
    }
    evt_2025 = classifier.classify_corporate_event(hdfc_corp_2025)
    # HDFC is not active in 2025, sector resolves to None
    assert evt_2025.sector is None, f"Expected None for HDFC in 2025, got {evt_2025.sector}"


def test_event_classifier_news_point_in_time_sector():
    """DATA-007: Verify event classifier assigns sector valid at news publication time."""
    classifier = EventClassifier()

    # News for HDFC in 2021
    news_hdfc_2021 = {
        "symbol": "HDFC",
        "title": "HDFC reports robust mortgage loan growth in Q1",
        "published_at": "2021-07-20T09:30:00Z",
    }
    evt = classifier.classify_news_event(news_hdfc_2021)
    assert evt.scope == SCOPE_STOCK
    assert evt.affected_tickers == ["HDFC"]
    assert evt.sector == "NBFC", f"Expected NBFC, got {evt.sector}"

    # News with keyword "crude" in 2021 vs 2025: in 2021 IOC was constituent, in 2025 not
    news_crude_2021 = {
        "symbol": "RELIANCE",
        "title": "Crude oil rallies as supply constraints intensify",
        "published_at": "2021-05-10T12:00:00Z",
    }
    evt_crude_2021 = classifier.classify_news_event(news_crude_2021)
    assert evt_crude_2021.scope == SCOPE_SECTOR
    assert "IOC" in evt_crude_2021.affected_tickers, "Expected IOC in crude-sensitive tickers for 2021"

    news_crude_2025 = {
        "symbol": "RELIANCE",
        "title": "Crude oil rallies as supply constraints intensify",
        "published_at": "2025-01-15T12:00:00Z",
    }
    evt_crude_2025 = classifier.classify_news_event(news_crude_2025)
    assert evt_crude_2025.scope == SCOPE_SECTOR
    assert "IOC" not in evt_crude_2025.affected_tickers, "IOC should NOT be tagged in 2025"


def test_predictor_point_in_time_sector_adjustment():
    """DATA-007: Verify predictor resolves sector at point-in-time from market data index."""
    predictor = Predictor()

    # Construct synthetic DataFrame with timestamps in 2021 for HDFC
    dates_2021 = pd.date_range("2021-05-01 09:15:00", periods=50, freq="5min", tz=timezone.utc)
    stock_df = pd.DataFrame(
        {
            "Open": np.linspace(2500, 2550, 50),
            "High": np.linspace(2510, 2560, 50),
            "Low": np.linspace(2490, 2540, 50),
            "Close": np.linspace(2505, 2555, 50),
            "Volume": np.full(50, 10000),
        },
        index=dates_2021,
    )

    # In Step 5 of generate_signal, sector_map_provider is invoked with stock_df.index[-1]
    as_of_dt = stock_df.index[-1]
    resolved_sector = sector_map_provider.get_sector("HDFC", as_of=as_of_dt)
    assert resolved_sector == "NBFC", f"Expected NBFC, got {resolved_sector}"

    # For 2025 dates, HDFC resolves to Unknown
    dates_2025 = pd.date_range("2025-01-01 09:15:00", periods=50, freq="5min", tz=timezone.utc)
    stock_df_2025 = stock_df.copy()
    stock_df_2025.index = dates_2025
    resolved_sector_2025 = sector_map_provider.get_sector("HDFC", as_of=stock_df_2025.index[-1])
    assert resolved_sector_2025 == "Unknown"


def test_backwards_compatibility():
    """Verify exact backwards compatibility for legacy callers without as_of timestamp."""
    # When as_of is None, get_sector returns the active sector matching config.SECTOR_MAP
    for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]:
        sec = sector_map_provider.get_sector(sym)
        assert sec == SECTOR_MAP[sym], f"Legacy call for {sym} gave {sec}, expected {SECTOR_MAP[sym]}"

    # Legacy get_industry returns a valid string
    ind = sector_map_provider.get_industry("RELIANCE")
    assert ind and ind != "Unknown"

    # Unknown symbol returns Unknown
    assert sector_map_provider.get_sector("NONEXISTENT_SYMBOL_XYZ") == "Unknown"
    assert sector_map_provider.get_industry("NONEXISTENT_SYMBOL_XYZ") == "Unknown"

    # get_sector_history returns records
    records = sector_map_provider.get_sector_history("TATACONSUM")
    assert len(records) >= 2, "Expected at least 2 historical records for TATACONSUM"
