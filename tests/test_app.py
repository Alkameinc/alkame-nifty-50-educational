import datetime
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

mock_st = MagicMock()
sys.modules["streamlit"] = mock_st
mock_st_runtime = MagicMock()
sys.modules["streamlit.runtime"] = mock_st_runtime
sys.modules["streamlit.runtime.scriptrunner"] = MagicMock()

from app import (
    format_action_label,
    format_confidence_display,
    format_events_for_table,
    prediction_records_to_dataframe,
    render_dashboard,
    risk_banner_style,
)
from event_classifier import Event
from predictor import ACTION_BUY, ACTION_HOLD, PredictionSignal


def get_dummy_signal():
    return PredictionSignal(
        timestamp=datetime.datetime.now().isoformat(),
        symbol="TCS",
        horizon="1D",
        action=ACTION_HOLD,
        model_predicted_class=ACTION_HOLD,
        raw_confidence=0.8,
        calibrated_confidence=0.75,
        target_price=110.0,
        stop_loss=90.0,
        suppressed=True,
        model_version="1.0",
        feature_version="1.0",
        risk_adjusted_confidence=0.75,
        agreement_fraction=1.0,
        downside_summary=[],
        upside_summary=[],
    )


def test_app_helpers():
    assert "BUY" in format_action_label(ACTION_BUY)

    sig = get_dummy_signal()
    assert "75%" in format_confidence_display(sig)

    sig.calibrated_confidence = None
    assert "not yet calibrated" in format_confidence_display(sig)

    assert risk_banner_style("NORMAL") == "success"
    assert risk_banner_style("CRISIS") == "error"

    e = Event(
        event_id="test",
        source="NEWS",
        event_type="type",
        timestamp=datetime.datetime.now(datetime.UTC),
        scope="test",
        affected_tickers=["TCS"],
        headline_or_label="head",
        sentiment_score=1.0,
        magnitude_estimate="HIGH",
    )
    assert len(format_events_for_table([e])) == 1

    df = prediction_records_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty

    class DummyRecord:
        def __init__(self):
            self.val = 1

    df = prediction_records_to_dataframe([DummyRecord()])
    assert not df.empty


@patch("app.st")
@patch("app.get_scheduler")
def test_render_dashboard(mock_get_sched, mock_st_local):
    mock_st_local.sidebar.selectbox.return_value = "TCS"
    mock_st_local.sidebar.button.return_value = False

    def mock_columns(n, *args, **kwargs):
        count = len(n) if isinstance(n, (list, tuple)) else n
        return [MagicMock() for _ in range(count)]

    mock_st_local.columns.side_effect = mock_columns

    def mock_tabs(labels, *args, **kwargs):
        return [MagicMock() for _ in labels]

    mock_st_local.tabs.side_effect = mock_tabs

    mock_scheduler = MagicMock()
    mock_scheduler.data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"Close": [100.0]})
    mock_scheduler.data_fetcher.fetch_nifty_index.return_value = pd.DataFrame({"Close": [100.0]})
    mock_scheduler.data_fetcher.fetch_stock_fundamentals.return_value = {"pe_ratio": 10.0}

    sig = get_dummy_signal()
    mock_scheduler.run_one_cycle_for_symbol.return_value = sig

    mock_snapshot = MagicMock()
    mock_snapshot.edge_check_result.alpha_pct = 5.0
    mock_snapshot.calibration_result.status = "CALIBRATED"
    mock_snapshot.calibration_result.expected_calibration_error = 0.1
    mock_scheduler.get_cached_live_worthiness.return_value = mock_snapshot

    mock_get_sched.return_value = mock_scheduler

    render_dashboard()


@patch("app.st")
@patch("app.get_scheduler")
def test_render_dashboard_refresh(mock_get_sched, mock_st_local):
    mock_st_local.sidebar.selectbox.return_value = "TCS"
    mock_st_local.sidebar.button.return_value = True  # refresh_backtest

    def mock_columns(n, *args, **kwargs):
        count = len(n) if isinstance(n, (list, tuple)) else n
        return [MagicMock() for _ in range(count)]

    mock_st_local.columns.side_effect = mock_columns

    def mock_tabs(labels, *args, **kwargs):
        return [MagicMock() for _ in labels]

    mock_st_local.tabs.side_effect = mock_tabs

    mock_scheduler = MagicMock()
    mock_scheduler.data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"Close": [100.0]})
    mock_scheduler.data_fetcher.fetch_nifty_index.return_value = pd.DataFrame({"Close": [100.0]})
    mock_scheduler.data_fetcher.fetch_stock_fundamentals.return_value = {"pe_ratio": 10.0}

    sig = get_dummy_signal()
    mock_scheduler.run_one_cycle_for_symbol.return_value = sig

    mock_snapshot = MagicMock()
    mock_snapshot.edge_check_result.alpha_pct = 5.0
    mock_snapshot.calibration_result.status = "CALIBRATED"
    mock_snapshot.calibration_result.expected_calibration_error = 0.1
    mock_scheduler.get_cached_live_worthiness.return_value = mock_snapshot

    mock_get_sched.return_value = mock_scheduler
    render_dashboard()
