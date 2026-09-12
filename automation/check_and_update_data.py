import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BAR_HISTORY_PERIOD,
    BAR_INTERVAL,
    NIFTY_INDEX_TICKER,
    to_yfinance_ticker,
)
from data_fetcher import DataFetcher
from market_calendar import is_market_open, is_trading_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataSync")


def main():
    print("=" * 65)
    print("  ALKAME NIFTY 50 - REAL-TIME DATA FRESHNESS & SYNC")
    print("=" * 65)

    now = datetime.now()
    market_open = is_market_open()
    trading_day = is_trading_day(now.date())
    status_str = (
        "OPEN (Live Session)"
        if market_open
        else ("CLOSED (Trading Day, Off-hours)" if trading_day else "CLOSED (Weekend / Holiday)")
    )

    print(f"[*] NSE Market Status : {status_str}")
    print(f"[*] Current Timestamp : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 65)

    fetcher = DataFetcher()

    core_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    print("[*] Checking data freshness for benchmark index and core stocks...")

    # 1. Update Index
    print(f"  -> Benchmark Index ({NIFTY_INDEX_TICKER}):", end=" ", flush=True)
    idx_res = fetcher.fetch_ohlcv(
        NIFTY_INDEX_TICKER, interval=BAR_INTERVAL, period=BAR_HISTORY_PERIOD, return_metadata=True
    )
    if idx_res and idx_res.data is not None and not idx_res.data.empty:
        print(f"[{idx_res.status.value}] ({len(idx_res.data)} bars)")
    else:
        print("[EMPTY / WARNING]")

    # 2. Update Global Risk Tickers
    print("  -> Global Cross-Asset Tickers:", end=" ", flush=True)
    try:
        global_res = fetcher.fetch_global_tickers()
        print(f"[ACTIVE] ({len(global_res)} feeds synced)")
    except Exception as e:
        print(f"[WARNING] ({e})")

    # 3. Check and refresh core symbols
    stale_count = 0
    fresh_count = 0

    for sym in core_symbols:
        ticker = to_yfinance_ticker(sym)
        print(f"  -> {sym:10s} ({ticker:13s}):", end=" ", flush=True)
        cached = fetcher._load_cache(ticker, interval=BAR_INTERVAL)
        if cached is None or cached.empty or fetcher.check_staleness(cached, ticker):
            stale_count += 1
            print("STALE -> Updating live...", end=" ", flush=True)
            res = fetcher.fetch_ohlcv(ticker, interval=BAR_INTERVAL, period=BAR_HISTORY_PERIOD, return_metadata=True)
            if res and res.data is not None and not res.data.empty:
                print(f"[{res.status.value}]")
            else:
                print("[FALLBACK/UNAVAILABLE]")
        else:
            fresh_count += 1
            print(f"[FRESH] ({len(cached)} bars)")

    print("-" * 65)
    print(f"[*] Sync Complete: {fresh_count} Fresh, {stale_count} Updated/Synchronized.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
