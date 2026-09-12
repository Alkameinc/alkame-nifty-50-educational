# 1. Standard library imports
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 2. Third-party imports
import numpy as np
import pandas as pd
import yfinance as yf

# 3. Local imports
from config import (
    BAR_INTERVAL,
    BAR_HISTORY_PERIOD,
    CACHE_DIR,
    MARKET_TIMEZONE,
    ensure_directories,
    configure_logging,
)
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------
class DataStatus(Enum):
    LIVE = "LIVE"
    CACHED_FRESH = "CACHED_FRESH"
    CACHED_STALE = "CACHED_STALE"
    UNAVAILABLE = "UNAVAILABLE"


class PriceAdjustmentMode(Enum):
    """Explicit corporate action adjustment strategy (DATA-004)."""
    ADJUSTED = "adjusted"          # Splits, dividends, and bonuses adjusted (default for ML returns & labels)
    RAW = "raw"                    # Unadjusted exchange execution prints
    TOTAL_RETURN = "total_return"  # Reinvested dividends + splits


@dataclass
class MarketDataResult:
    data: Optional[pd.DataFrame]
    status: DataStatus
    source: str
    adjustment_mode: PriceAdjustmentMode = PriceAdjustmentMode.ADJUSTED
    error: Optional[str] = None


@dataclass
class DataQualityReport:
    is_valid: bool
    is_monotonic: bool
    duplicate_index_count: int
    outlier_count: int
    timezone_aligned: bool
    issues: List[str]


# ---------------------------------------------------------------------------
# Abstract Base Provider
# ---------------------------------------------------------------------------
class MarketDataProvider(ABC):
    """
    Abstract Base Class for market data sources (DATA-005).
    Decouples core pipelines from specific vendors (yfinance, local cache, NSE broker APIs).
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
        adjustment: PriceAdjustmentMode = PriceAdjustmentMode.ADJUSTED,
    ) -> MarketDataResult:
        """Fetch intraday or daily OHLCV bars for a single ticker."""
        pass

    @abstractmethod
    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamental data (P/E, P/B, Market Cap, etc.)."""
        pass

    def validate_data_quality(
        self,
        df: Optional[pd.DataFrame],
        ticker: str,
        max_single_bar_return_pct: float = 50.0,
    ) -> DataQualityReport:
        """
        Applies data quality rules (Phase 2):
        1. Monotonic increasing timestamp index check
        2. Duplicate timestamp detection
        3. Extreme price outlier / jump rejection (>50% single-bar return)
        4. Timezone verification (Asia/Kolkata)
        """
        issues: List[str] = []
        if df is None or df.empty:
            return DataQualityReport(
                is_valid=False,
                is_monotonic=False,
                duplicate_index_count=0,
                outlier_count=0,
                timezone_aligned=False,
                issues=["DataFrame is None or empty"],
            )

        # 1. Missing required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # 2. Monotonic index check
        is_monotonic = bool(df.index.is_monotonic_increasing)
        if not is_monotonic:
            issues.append("Timestamps are not monotonically increasing")

        # 3. Duplicate timestamps
        dup_count = int(df.index.duplicated().sum())
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate timestamp index entries")

        # 4. Outlier jumps (e.g. erroneous 50%+ single bar spike)
        outlier_count = 0
        if "Close" in df.columns and len(df) > 1:
            pct_changes = df["Close"].pct_change().abs() * 100.0
            outlier_count = int((pct_changes > max_single_bar_return_pct).sum())
            if outlier_count > 0:
                issues.append(f"Found {outlier_count} single-bar returns exceeding {max_single_bar_return_pct}%")

        # 5. Timezone verification
        tz_aligned = True
        if hasattr(df.index, "tz") and df.index.tz is not None:
            tz_str = str(df.index.tz)
            if "Kolkata" not in tz_str and "IST" not in tz_str and "+05:30" not in tz_str:
                tz_aligned = False
                issues.append(f"Index timezone {tz_str} not aligned with Asia/Kolkata")

        is_valid = len(issues) == 0 or (len(issues) == 1 and outlier_count > 0 and outlier_count <= 2)
        return DataQualityReport(
            is_valid=is_valid,
            is_monotonic=is_monotonic,
            duplicate_index_count=dup_count,
            outlier_count=outlier_count,
            timezone_aligned=tz_aligned,
            issues=issues,
        )

    def clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes timezones to Asia/Kolkata, deduplicates, and sorts index."""
        if df is None or df.empty:
            return df

        clean_df = df.copy()
        # Drop duplicates
        clean_df = clean_df[~clean_df.index.duplicated(keep="last")]
        # Sort monotonic
        clean_df = clean_df.sort_index()

        # Normalize timezone
        if hasattr(clean_df.index, "tz") and clean_df.index.tz is not None:
            try:
                clean_df.index = clean_df.index.tz_convert("Asia/Kolkata")
            except Exception:
                pass
        else:
            try:
                clean_df.index = pd.to_datetime(clean_df.index).tz_localize("Asia/Kolkata")
            except Exception:
                pass

        return clean_df


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------
class YFinanceMarketDataProvider(MarketDataProvider):
    """Production/research market data provider backed by yfinance."""

    def __init__(self, cache_dir: Path = CACHE_DIR, max_retries: int = 3, retry_backoff: float = 2.0):
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        ensure_directories()

    def _cache_path(self, ticker: str, interval: str, adjustment: PriceAdjustmentMode) -> Path:
        safe_name = ticker.replace("=", "_").replace("^", "IDX_").replace(".", "_")
        return self.cache_dir / f"{safe_name}_{interval}_{adjustment.value}.csv"

    def _save_cache(self, ticker: str, df: pd.DataFrame, interval: str, adjustment: PriceAdjustmentMode) -> None:
        try:
            p = self._cache_path(ticker, interval, adjustment)
            df.to_csv(p)
        except Exception as e:
            logger.error(f"Failed to write cache for {ticker}: {e}")

    def _load_cache(self, ticker: str, interval: str, adjustment: PriceAdjustmentMode) -> Optional[pd.DataFrame]:
        p = self._cache_path(ticker, interval, adjustment)
        if not p.exists():
            # Check legacy cache path
            safe_name = ticker.replace("=", "_").replace("^", "IDX_").replace(".", "_")
            legacy_p = self.cache_dir / f"{safe_name}_{interval}.csv"
            if legacy_p.exists():
                p = legacy_p
            else:
                return None

        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return self.clean_and_normalize(df)
        except Exception as e:
            logger.error(f"Failed reading cache from {p}: {e}")
            return None

    def fetch_ohlcv(
        self,
        ticker: str,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
        adjustment: PriceAdjustmentMode = PriceAdjustmentMode.ADJUSTED,
    ) -> MarketDataResult:
        # Check cache first
        cached = self._load_cache(ticker, interval, adjustment)

        last_error = None
        auto_adjust = (adjustment == PriceAdjustmentMode.ADJUSTED)

        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=auto_adjust)
                if df is None or df.empty:
                    raise ValueError(f"Empty data returned for {ticker}")

                missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns {missing} for {ticker}")

                df = df[self.REQUIRED_COLUMNS].copy()
                df = self.clean_and_normalize(df)

                report = self.validate_data_quality(df, ticker)
                if not report.is_valid:
                    logger.warning(f"Data quality issues for {ticker}: {report.issues}")

                self._save_cache(ticker, df, interval, adjustment)
                health_registry.report("data_fetcher", ok=True, detail=f"Fetched {ticker} from Yahoo")
                return MarketDataResult(
                    data=df,
                    status=DataStatus.LIVE,
                    source="yfinance",
                    adjustment_mode=adjustment,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed for {ticker}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)

        # Fallback to cache
        if cached is not None:
            logger.warning(f"Live fetch failed for {ticker}, returning cached data.")
            return MarketDataResult(
                data=cached,
                status=DataStatus.CACHED_STALE,
                source="cache",
                adjustment_mode=adjustment,
                error=str(last_error),
            )

        health_registry.report("data_fetcher", ok=False, detail=f"No data for {ticker}", error=str(last_error))
        return MarketDataResult(
            data=None,
            status=DataStatus.UNAVAILABLE,
            source="none",
            adjustment_mode=adjustment,
            error=str(last_error),
        )

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        try:
            t = yf.Ticker(ticker)
            return t.info or {}
        except Exception as e:
            logger.warning(f"Failed fetching fundamentals for {ticker}: {e}")
            return {}


class LocalCacheMarketDataProvider(MarketDataProvider):
    """Offline / local-only provider for fully cached execution and tests."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir

    def fetch_ohlcv(
        self,
        ticker: str,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
        adjustment: PriceAdjustmentMode = PriceAdjustmentMode.ADJUSTED,
    ) -> MarketDataResult:
        safe_name = ticker.replace("=", "_").replace("^", "IDX_").replace(".", "_")
        p = self.cache_dir / f"{safe_name}_{interval}_{adjustment.value}.csv"
        if not p.exists():
            p = self.cache_dir / f"{safe_name}_{interval}.csv"

        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                clean_df = self.clean_and_normalize(df)
                return MarketDataResult(
                    data=clean_df,
                    status=DataStatus.CACHED_FRESH,
                    source="local_cache",
                    adjustment_mode=adjustment,
                )
            except Exception as e:
                return MarketDataResult(
                    data=None,
                    status=DataStatus.UNAVAILABLE,
                    source="local_cache",
                    adjustment_mode=adjustment,
                    error=str(e),
                )

        return MarketDataResult(
            data=None,
            status=DataStatus.UNAVAILABLE,
            source="local_cache",
            adjustment_mode=adjustment,
            error="Cache file not found",
        )

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        return {}


class TestFixtureMarketDataProvider(MarketDataProvider):
    """Deterministic, in-memory provider for hermetic automated tests (TEST-001)."""

    def __init__(self, fixtures: Optional[Dict[str, pd.DataFrame]] = None):
        self.fixtures: Dict[str, pd.DataFrame] = fixtures or {}

    def set_fixture(self, ticker: str, df: pd.DataFrame) -> None:
        self.fixtures[ticker] = df

    def fetch_ohlcv(
        self,
        ticker: str,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
        adjustment: PriceAdjustmentMode = PriceAdjustmentMode.ADJUSTED,
    ) -> MarketDataResult:
        if ticker in self.fixtures:
            df = self.fixtures[ticker].copy()
            clean_df = self.clean_and_normalize(df)
            return MarketDataResult(
                data=clean_df,
                status=DataStatus.LIVE,
                source="test_fixture",
                adjustment_mode=adjustment,
            )
        return MarketDataResult(
            data=None,
            status=DataStatus.UNAVAILABLE,
            source="test_fixture",
            adjustment_mode=adjustment,
            error=f"No fixture configured for {ticker}",
        )

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        return {
            "trailingPE": 22.5,
            "priceToBook": 3.2,
            "marketCap": 1500000000000,
            "trailingEps": 65.4,
            "dividendYield": 0.012,
            "fiftyTwoWeekHigh": 2800.0,
            "fiftyTwoWeekLow": 2100.0,
            "currentPrice": 2500.0,
            "sector": "Energy",
            "industry": "Oil & Gas",
        }


if __name__ == "__main__":
    configure_logging(log_filename="market_data_provider_selftest.log")
    print("\n=== MARKET DATA PROVIDER SELF-TEST ===")
    # 1. Test Fixture Provider
    dates = pd.date_range("2026-07-01 09:15", periods=50, freq="5min", tz="Asia/Kolkata")
    fixture_df = pd.DataFrame({
        "Open": np.linspace(100, 110, 50),
        "High": np.linspace(101, 111, 50),
        "Low": np.linspace(99, 109, 50),
        "Close": np.linspace(100.5, 110.5, 50),
        "Volume": np.full(50, 5000),
    }, index=dates)

    fixture_prov = TestFixtureMarketDataProvider({"RELIANCE.NS": fixture_df})
    res = fixture_prov.fetch_ohlcv("RELIANCE.NS", adjustment=PriceAdjustmentMode.ADJUSTED)
    assert res.status == DataStatus.LIVE
    assert res.data is not None and len(res.data) == 50
    assert res.adjustment_mode == PriceAdjustmentMode.ADJUSTED

    # 2. Data Quality Rules validation
    report = fixture_prov.validate_data_quality(res.data, "RELIANCE.NS")
    assert report.is_valid is True
    assert report.is_monotonic is True
    assert report.duplicate_index_count == 0
    print(f"Data Quality Report: valid={report.is_valid}, monotonic={report.is_monotonic}, duplicates={report.duplicate_index_count}")

    # 3. Fundamentals
    funds = fixture_prov.fetch_fundamentals("RELIANCE.NS")
    assert funds.get("trailingPE") == 22.5
    print(f"Fundamentals: P/E={funds.get('trailingPE')}, Sector={funds.get('sector')}")

    print("STATUS: PASS")
