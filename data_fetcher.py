# 1. Standard library imports
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass

# 2. Third-party imports
import pandas as pd
import yfinance as yf

# 3. Local imports
from config import (
    NIFTY50_YFINANCE_TICKERS,
    NIFTY_INDEX_TICKER,
    GLOBAL_TICKERS,
    BAR_INTERVAL,
    BAR_HISTORY_PERIOD,
    DATA_STALENESS_THRESHOLD_MINUTES,
    CACHE_DIR,
    to_yfinance_ticker,
    ensure_directories,
    configure_logging,
    DATA_STALENESS_THRESHOLD_TRADING_DAYS,
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME,
    MARKET_TIMEZONE,
)
from health_monitor import registry as health_registry
from market_data_provider import (
    DataStatus,
    PriceAdjustmentMode,
    MarketDataResult,
    MarketDataProvider,
    YFinanceMarketDataProvider,
    LocalCacheMarketDataProvider,
    TestFixtureMarketDataProvider,
    DataQualityReport,
)

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants & Models
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class DataFetcher:
    """
    Interface for OHLCV bars and fundamentals, wrapping an underlying
    MarketDataProvider (DATA-005) with caching, quality verification,
    and staleness detection.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, provider: Optional[MarketDataProvider] = None):
        self.cache_dir = cache_dir
        self.provider = provider or YFinanceMarketDataProvider(cache_dir=cache_dir)
        ensure_directories()

    def _cache_path(self, ticker: str, interval: str = BAR_INTERVAL) -> Path:
        safe_name = ticker.replace("=", "_").replace("^", "IDX_").replace(".", "_")
        return self.cache_dir / f"{safe_name}_{interval}.csv"

    def _save_cache(self, ticker: str, df: pd.DataFrame, interval: str = BAR_INTERVAL) -> None:
        try:
            df.to_csv(self._cache_path(ticker, interval=interval))
        except Exception as e:
            logger.error(f"Failed to write cache for {ticker} ({interval}): {e}")

    def _load_cache(self, ticker: str, interval: str = BAR_INTERVAL) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker, interval=interval)
        try:
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=True)
                if df.index.tz is not None:
                    try:
                        df.index = df.index.tz_convert("Asia/Kolkata")
                    except Exception:
                        pass
                logger.warning(f"Loaded stale CACHED data for {ticker} ({interval}) from {path}")
                return df
        except Exception as e:
            logger.error(f"Failed to load cache for {ticker} ({interval}): {e}")
        return None

    def fetch_ohlcv(
        self,
        ticker: str,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
        return_metadata: bool = False,
    ) -> Union[Optional[pd.DataFrame], MarketDataResult]:
        """
        Fetch OHLCV bars for a single ticker with retry logic. Returns None
        only if both live fetch and cache fallback fail — callers must handle
        that case (e.g. by suppressing signals for that ticker).
        """
        # FIRST: Check if we have a fresh cache
        cached = self._load_cache(ticker, interval=interval)
        if cached is not None and not self.check_staleness(cached, ticker):
            logger.info(f"Using fresh cache for {ticker}")
            if return_metadata:
                return MarketDataResult(data=cached, status=DataStatus.CACHED_FRESH, source="cache")
            return cached

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
                if df is None or df.empty:
                    raise ValueError(f"Empty DataFrame returned for {ticker}")

                missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing expected columns {missing_cols} for {ticker}")

                df = df[REQUIRED_COLUMNS].copy()
                self._save_cache(ticker, df, interval=interval)
                health_registry.report("data_fetcher", ok=True, detail=f"Fetched live data for {ticker}")
                if return_metadata:
                    return MarketDataResult(data=df, status=DataStatus.LIVE, source="yahoo")
                return df

            except Exception as e:
                last_error = e
                logger.error(
                    f"Attempt {attempt}/{MAX_RETRIES} failed fetching {ticker} "
                    f"(interval={interval}, period={period}): {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        logger.error(f"All {MAX_RETRIES} live fetch attempts failed for {ticker}: {last_error}")
        cached = self._load_cache(ticker, interval=interval)
        if cached is not None:
            if return_metadata:
                return MarketDataResult(data=cached, status=DataStatus.CACHED_STALE, source="cache")
            return cached

        logger.error(f"No cache available for {ticker} ({interval}) either — returning None.")
        health_registry.report("data_fetcher", ok=False, detail=f"No live or cached data for {ticker}", error=str(last_error))
        if return_metadata:
            return MarketDataResult(data=None, status=DataStatus.UNAVAILABLE, source="none")
        return None

    def fetch_daily_ohlcv(self, ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """Fetch full historical daily OHLCV bars for a single ticker."""
        return self.fetch_ohlcv(ticker, interval="1d", period=period)

    def fetch_daily_ohlcv_incremental(self, ticker: str, full_period: str = "5y") -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars using an incremental update strategy if cache exists."""
        cached = self._load_cache(ticker, interval="1d")
        if cached is None or cached.empty:
            return self.fetch_daily_ohlcv(ticker, period=full_period)
        
        last_cached_date = cached.index[-1]
        start_date = (last_cached_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        try:
            new_data = yf.Ticker(ticker).history(start=start_date, interval="1d", auto_adjust=False)
            if new_data is not None and not new_data.empty:
                missing_cols = [c for c in REQUIRED_COLUMNS if c not in new_data.columns]
                if not missing_cols:
                    new_data = new_data[REQUIRED_COLUMNS].copy()
                    combined = pd.concat([cached, new_data]).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    self._save_cache(ticker, combined, interval="1d")
                    return combined
        except Exception as e:
            logger.error(f"Incremental daily fetch failed for {ticker}, returning cached data: {e}")
            
        return cached

    def fetch_all_nifty50(
        self,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV bars for all 50 NIFTY constituents. Skips (with a logged error)
        any ticker that fails both live fetch and cache fallback."""
        results: Dict[str, pd.DataFrame] = {}
        for ticker in NIFTY50_YFINANCE_TICKERS:
            df = self.fetch_ohlcv(ticker, interval=interval, period=period)
            if df is not None:
                results[ticker] = df
            else:
                logger.error(f"Skipping {ticker} entirely — no live or cached data available.")
        logger.info(f"Fetched OHLCV for {len(results)}/{len(NIFTY50_YFINANCE_TICKERS)} NIFTY50 tickers.")
        return results

    def fetch_all_nifty50_daily(self, period: str = "5y") -> Dict[str, pd.DataFrame]:
        """Fetch daily OHLCV bars for all 50 NIFTY constituents incrementally."""
        results: Dict[str, pd.DataFrame] = {}
        for ticker in NIFTY50_YFINANCE_TICKERS:
            df = self.fetch_daily_ohlcv_incremental(ticker, full_period=period)
            if df is not None:
                results[ticker] = df
            else:
                logger.error(f"Skipping {ticker} entirely — no live or cached daily data available.")
        logger.info(f"Fetched daily OHLCV for {len(results)}/{len(NIFTY50_YFINANCE_TICKERS)} NIFTY50 tickers.")
        return results

    def fetch_global_tickers(
        self,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV bars for the global cross-asset tickers (DXY, Gold, Crude, VIX, etc.)."""
        results: Dict[str, pd.DataFrame] = {}
        for name, ticker in GLOBAL_TICKERS.items():
            df = self.fetch_ohlcv(ticker, interval=interval, period=period)
            if df is not None:
                results[name] = df
            else:
                logger.error(f"Skipping global ticker {name} ({ticker}) — no live or cached data available.")
        logger.info(f"Fetched OHLCV for {len(results)}/{len(GLOBAL_TICKERS)} global tickers.")
        return results

    def fetch_nifty_index(
        self,
        interval: str = BAR_INTERVAL,
        period: str = BAR_HISTORY_PERIOD,
    ) -> Optional[pd.DataFrame]:
        """Fetch the NIFTY 50 index itself — used as the baseline for edge/outperformance checks."""
        return self.fetch_ohlcv(NIFTY_INDEX_TICKER, interval=interval, period=period)

    def fetch_stock_fundamentals(self, ticker: str) -> Dict[str, object]:
        """Fetch fundamental metrics (P/E, P/B, Market Cap, EPS, Div Yield, 52W High/Low) with safe fallbacks."""
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            
            # Format Market Cap to INR Crores
            mcap = info.get("marketCap")
            mcap_cr = f"₹{mcap / 1e7:,.2f} Cr" if mcap else "N/A"
            
            pe = info.get("trailingPE") or info.get("forwardPE")
            pe_str = f"{pe:.2f}" if pe else "N/A"
            
            pb = info.get("priceToBook")
            pb_str = f"{pb:.2f}" if pb else "N/A"
            
            eps = info.get("trailingEps")
            eps_str = f"₹{eps:.2f}" if eps else "N/A"
            
            div = info.get("dividendYield")
            div_str = f"{div * 100:.2f}%" if div is not None else "N/A"
            
            high52 = info.get("fiftyTwoWeekHigh")
            low52 = info.get("fiftyTwoWeekLow")
            cmp_raw = info.get("currentPrice") or info.get("regularMarketPrice")

            # Calculate valuation assessment
            # Scoring factors:
            # 1. P/E vs typical baseline (under 18 attractive, 18-32 fair, >32 premium/growth)
            # 2. P/B vs baseline (under 2.5 attractive, 2.5-5.0 fair, >5.0 premium)
            # 3. 52-week range percentile (position between 52W low and 52W high)
            valuation_status = "FAIR PRICED"
            valuation_badge = "🟡 FAIR PRICED"
            valuation_details = []

            pe_val = float(pe) if pe and pe > 0 else None
            pb_val = float(pb) if pb and pb > 0 else None
            range_pct = None
            if cmp_raw and high52 and low52 and (high52 > low52):
                range_pct = max(0.0, min(1.0, (cmp_raw - low52) / (high52 - low52)))

            score = 0  # <0 towards underpriced, >0 towards overpriced
            if pe_val is not None:
                if pe_val < 18.0:
                    score -= 1
                    valuation_details.append(f"P/E ({pe_val:.1f}x) is below typical market average (<18x)")
                elif pe_val > 35.0:
                    score += 1
                    valuation_details.append(f"P/E ({pe_val:.1f}x) trades at a premium (>35x)")
                else:
                    valuation_details.append(f"P/E ({pe_val:.1f}x) is within reasonable range (18-35x)")

            if pb_val is not None:
                if pb_val < 2.0:
                    score -= 1
                    valuation_details.append(f"P/B ({pb_val:.1f}x) offers tangible asset cushion (<2.0x)")
                elif pb_val > 6.0:
                    score += 1
                    valuation_details.append(f"P/B ({pb_val:.1f}x) reflects high asset premium (>6.0x)")

            if range_pct is not None:
                if range_pct < 0.25:
                    score -= 1
                    valuation_details.append(f"Price is in lower quartile of 52W range ({range_pct*100:.0f}%)")
                elif range_pct > 0.85:
                    score += 1
                    valuation_details.append(f"Price is near 52W high ({range_pct*100:.0f}% of range)")

            if score <= -1:
                valuation_status = "UNDERPRICED"
                valuation_badge = "🟢 UNDERPRICED (Attractive Valuation)"
                valuation_rationale = "Valuation metrics suggest stock is trading at an attractive discount relative to earnings, assets, or historical range. " + " • ".join(valuation_details)
            elif score >= 2:
                valuation_status = "OVERPRICED"
                valuation_badge = "🔴 OVERPRICED (Premium Valuation)"
                valuation_rationale = "Valuation metrics reflect premium pricing or high growth expectations baked into price. " + " • ".join(valuation_details)
            else:
                valuation_status = "FAIR PRICED"
                valuation_badge = "🟡 FAIR PRICED (Fair Value)"
                valuation_rationale = "Valuation metrics align closely with historical and market baseline norms. " + " • ".join(valuation_details)

            return {
                "pe_ratio": pe_str,
                "price_to_book": pb_str,
                "market_cap": mcap_cr,
                "eps": eps_str,
                "dividend_yield": div_str,
                "fifty_two_week_high": f"₹{high52:,.2f}" if high52 else "N/A",
                "fifty_two_week_low": f"₹{low52:,.2f}" if low52 else "N/A",
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "valuation_status": valuation_status,
                "valuation_badge": valuation_badge,
                "valuation_rationale": valuation_rationale,
                "raw_pe": pe_val,
                "raw_pb": pb_val,
                "raw_range_pct": range_pct,
            }
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {ticker}: {e}")
            return {
                "pe_ratio": "N/A", "price_to_book": "N/A", "market_cap": "N/A",
                "eps": "N/A", "dividend_yield": "N/A", "fifty_two_week_high": "N/A",
                "fifty_two_week_low": "N/A", "sector": "N/A", "industry": "N/A",
                "valuation_status": "FAIR PRICED",
                "valuation_badge": "🟡 FAIR PRICED (Neutral Data)",
                "valuation_rationale": "Insufficient fundamental data to establish over/under-pricing. Evaluated as Neutral/Fair.",
                "raw_pe": None, "raw_pb": None, "raw_range_pct": None,
            }

    def is_market_open(self) -> bool:
        from market_calendar import is_market_open as _is_open
        return _is_open()

    def check_staleness(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Returns True if the data is STALE (last bar older than the configured
        threshold). Callers should suppress signals for a ticker when this is True.
        """
        try:
            if not self.is_market_open():
                return False # Market is closed, so data from last close is valid, not stale

            if df is None or df.empty:
                logger.warning(f"Staleness check: {ticker} has no data at all — treating as stale.")
                return True

            last_ts = df.index[-1]
            if last_ts.tzinfo is not None:
                now = pd.Timestamp.now(tz=last_ts.tzinfo)
            else:
                now = pd.Timestamp.now()

            age_minutes = (now - last_ts).total_seconds() / 60.0
            is_stale = age_minutes > DATA_STALENESS_THRESHOLD_MINUTES
            if is_stale:
                logger.warning(
                    f"{ticker} data is STALE: last bar is {age_minutes:.1f} minutes old "
                    f"(threshold={DATA_STALENESS_THRESHOLD_MINUTES}m)"
                )
            return is_stale
        except Exception as e:
            logger.error(f"Failed staleness check for {ticker}: {e}")
            return True  # fail safe: treat unknown state as stale/unsafe

    def check_staleness_daily(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Returns True if the daily data is STALE (last bar older than configured trading days).
        """
        try:
            if df is None or df.empty:
                logger.warning(f"Staleness check daily: {ticker} has no data at all — treating as stale.")
                return True

            last_ts = df.index[-1]
            if last_ts.tzinfo is not None:
                now = pd.Timestamp.now(tz=last_ts.tzinfo)
            else:
                now = pd.Timestamp.now()
            
            # Count trading days between last_ts and now
            trading_days = 0
            current = last_ts.normalize() + timedelta(days=1)
            now_norm = now.normalize()
            
            while current <= now_norm:
                if current.weekday() < 5:  # Monday to Friday
                    trading_days += 1
                current += timedelta(days=1)
                
            is_stale = trading_days > DATA_STALENESS_THRESHOLD_TRADING_DAYS
            
            if is_stale:
                logger.warning(
                    f"{ticker} daily data is STALE: last bar is {trading_days} trading days old "
                    f"(threshold={DATA_STALENESS_THRESHOLD_TRADING_DAYS})"
                )
            return is_stale
        except Exception as e:
            logger.error(f"Failed daily staleness check for {ticker}: {e}")
            return True


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="data_fetcher_selftest.log")
    logger.info("Running data_fetcher.py self-test...")

    fetcher = DataFetcher()
    test_symbol = "RELIANCE.NS"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== DATA FETCHER SELF-TEST RESULT ===")

        # Test 1: single ticker fetch
        df = fetcher.fetch_ohlcv(test_symbol)
        single_ok = df is not None and not df.empty
        print(f"Single ticker fetch ({test_symbol}): {'OK' if single_ok else 'FAILED'}"
              f" — rows={0 if df is None else len(df)}")

        # Test 1.5: fetch with metadata
        res = fetcher.fetch_ohlcv(test_symbol, return_metadata=True)
        meta_ok = isinstance(res, MarketDataResult) and res.status in DataStatus
        print(f"Fetch with metadata ({test_symbol}): {'OK' if meta_ok else 'FAILED'}"
              f" — status={getattr(res, 'status', 'UNKNOWN')}")

        # Test 2: staleness check runs without error
        if df is not None:
            stale = fetcher.check_staleness(df, test_symbol)
            print(f"Staleness check ran successfully. Stale={stale}")

        # Test 3: NIFTY index fetch
        index_df = fetcher.fetch_nifty_index()
        index_ok = index_df is not None and not index_df.empty
        print(f"NIFTY index fetch: {'OK' if index_ok else 'FAILED'}"
              f" — rows={0 if index_df is None else len(index_df)}")

        # Test 4: one global ticker fetch (Gold) to confirm the mapping works
        gold_df = fetcher.fetch_ohlcv(GLOBAL_TICKERS["GOLD"])
        gold_ok = gold_df is not None and not gold_df.empty
        print(f"Global ticker fetch (Gold, {GLOBAL_TICKERS['GOLD']}): {'OK' if gold_ok else 'FAILED'}"
              f" — rows={0 if gold_df is None else len(gold_df)}")

        # Test 5: cache fallback works by loading whatever we just cached
        cached = fetcher._load_cache(test_symbol)
        cache_ok = cached is not None and not cached.empty
        print(f"Cache read-back for {test_symbol}: {'OK' if cache_ok else 'FAILED'}")

        # Test 6: Daily bar fetch
        daily_df = fetcher.fetch_daily_ohlcv(test_symbol, period="1y")
        daily_ok = daily_df is not None and not daily_df.empty
        print(f"Daily ticker fetch ({test_symbol}): {'OK' if daily_ok else 'FAILED'}"
              f" — rows={0 if daily_df is None else len(daily_df)}")
        
        # Test 7: Incremental daily fetch
        daily_inc_df = fetcher.fetch_daily_ohlcv_incremental(test_symbol, full_period="1y")
        inc_ok = daily_inc_df is not None and len(daily_inc_df) >= len(daily_df) if daily_df is not None else False
        print(f"Incremental daily fetch ({test_symbol}): {'OK' if inc_ok else 'FAILED'}"
              f" — rows={0 if daily_inc_df is None else len(daily_inc_df)}")
        
        # Test 8: Cache collision test
        cache_5m = fetcher._cache_path(test_symbol, interval="5m").exists()
        cache_1d = fetcher._cache_path(test_symbol, interval="1d").exists()
        collision_ok = cache_5m and cache_1d
        print(f"Cache collision test (distinct files exist): {'OK' if collision_ok else 'FAILED'}")
        
        overall_pass = single_ok and meta_ok and index_ok and gold_ok and cache_ok and daily_ok and inc_ok and collision_ok
        print("STATUS: PASS" if overall_pass else "STATUS: FAIL — see details above")

        if not overall_pass:
            logger.error("data_fetcher.py self-test did not fully pass.")

    except Exception as e:
        logger.error(f"data_fetcher.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
