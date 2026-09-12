import logging
import pandas as pd
import pytest
from scheduler import Scheduler

from unittest.mock import patch


def test_scheduler_run_one_cycle_success():
    scheduler = Scheduler()
    scheduler.data_fetcher.is_market_open = lambda: False
    scheduler.get_cached_live_worthiness = lambda s, horizon="INTRADAY": None

    with patch.object(scheduler.predictor, "generate_multi_horizon_stream", return_value=[]):
        dates = pd.date_range("2026-07-01", periods=10)
        stock_df = pd.DataFrame(
            {
                "Close": [100.0] * 10,
                "High": [105.0] * 10,
                "Low": [95.0] * 10,
                "Open": [100.0] * 10,
                "Volume": [1000] * 10,
            },
            index=dates,
        )
        index_df = stock_df.copy()

        res = scheduler.run_one_cycle_for_symbol("RELIANCE", stock_df, index_df, return_structured=True)
        assert res is not None
        assert res.success is True
        assert res.status == "SUCCESS"


def test_scheduler_run_one_cycle_empty_data():
    scheduler = Scheduler()
    res = scheduler.run_one_cycle_for_symbol("RELIANCE", pd.DataFrame(), pd.DataFrame(), return_structured=True)
    assert res.success is False
    assert res.status == "DATA_UNAVAILABLE"
