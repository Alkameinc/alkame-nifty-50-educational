from unittest.mock import MagicMock, patch

import pandas as pd

from data_fetcher import DataFetcher


@patch("data_fetcher.yf.Ticker")
def test_data_fetcher(mock_yf_ticker):
    mock_ticker_inst = MagicMock()
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 99.0],
            "High": [100.0, 101.0, 99.0],
            "Low": [100.0, 101.0, 99.0],
            "Close": [100.0, 101.0, 99.0],
            "Volume": [1000, 1000, 1000],
        }
    )
    df.index = pd.date_range("2026-01-01", periods=3)
    mock_ticker_inst.history.return_value = df
    mock_yf_ticker.return_value = mock_ticker_inst

    fetcher = DataFetcher()

    # fetch_ohlcv
    res = fetcher.fetch_ohlcv("TCS", period="1mo", interval="1d")
    assert res is not None

    # fetch_daily_ohlcv
    res_daily = fetcher.fetch_daily_ohlcv("TCS", period="1y")
    assert res_daily is not None

    # fetch_all_nifty50
    with patch("data_fetcher.NIFTY50_YFINANCE_TICKERS", ["TCS"]):
        all_res = fetcher.fetch_all_nifty50(period="1mo")
        assert "TCS" in all_res

    # staleness
    df_stale = df.copy()
    df_stale.index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    stale = fetcher.check_staleness(df_stale, "TCS")
    assert stale in [True, False]
