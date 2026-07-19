import sys
import pandas as pd
from data_fetcher import DataFetcher
from backtester import Backtester
from config import NIFTY50_SYMBOLS, ALL_HORIZONS

def train_all():
    fetcher = DataFetcher()
    backtester = Backtester()
    
    from config import to_yfinance_ticker, NIFTY50_SYMBOLS, HORIZON_CONFIG
    
    symbols_to_train = NIFTY50_SYMBOLS

    print("=== ALKAME NIFTY 50: MODEL TRAINING ===")
    print(f"Training ensemble models for: {len(symbols_to_train)} stocks")

    for symbol in symbols_to_train:
        print(f"\n--- Processing {symbol} ---")
        for horizon in ALL_HORIZONS:
            print(f"Training {symbol} - {horizon}...")
            interval = HORIZON_CONFIG[horizon].get("bar_interval", "5m")
            period = HORIZON_CONFIG[horizon].get("history_period", "60d")
            
            if interval == "1d":
                stock_df = fetcher.fetch_daily_ohlcv(to_yfinance_ticker(symbol), period=period)
                index_df = fetcher.fetch_daily_ohlcv("^NSEI", period=period)
            else:
                stock_df = fetcher.fetch_ohlcv(to_yfinance_ticker(symbol), period=period)
                index_df = fetcher.fetch_nifty_index(period=period)
                
            if stock_df is None or stock_df.empty or index_df is None or index_df.empty:
                print(f"  -> FAILED: Could not fetch {period} {interval} data for {symbol}.")
                continue
                
            res = backtester.run_backtest_for_symbol(symbol, stock_df, index_df, horizon=horizon)
            if res.success:
                print(f"  -> SUCCESS! Edge: {res.edge_check_status}, Calibration: {res.calibration_status}")
            else:
                print(f"  -> FAILED: {res.error}")

if __name__ == "__main__":
    train_all()
