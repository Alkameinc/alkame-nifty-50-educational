import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from api import app
from event_classifier import EventBatchResult, EventClassifier
from history_manager import HistoryManager
from predictor import PredictionContext, Predictor
from scheduler import EventContext, Scheduler


def _synthetic_bars(n: int = 40) -> pd.DataFrame:
    now_utc = datetime.now(UTC)
    dates = pd.date_range(end=now_utc, periods=n, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [105.0 + i for i in range(n)],
            "Low": [95.0 + i for i in range(n)],
            "Close": [102.0 + i for i in range(n)],
            "Volume": [1000 + i * 10 for i in range(n)],
        },
        index=dates,
    )
    return df


def test_event_source_unavailable_propagates_to_signal():
    """Verify EVENT_SOURCE_UNAVAILABLE never degrades into NO_EVENTS or EVENTS_AVAILABLE."""
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    mock_classifier = MagicMock(spec=EventClassifier)
    mock_classifier.classify_batch.return_value = EventBatchResult(
        events=[],
        status="EVENT_SOURCE_UNAVAILABLE",
        errors=["Data source connection timed out"],
    )

    predictor = Predictor(event_classifier=mock_classifier)
    predictor.data_fetcher.check_staleness = MagicMock(return_value=False)
    sig = predictor.generate_signal(
        symbol="RELIANCE",
        stock_df=stock_df,
        index_df=index_df,
        horizon="INTRADAY",
    )

    assert sig.event_source_status == "EVENT_SOURCE_UNAVAILABLE"
    assert sig.suppressed is True
    assert sig.action == "HOLD"
    assert any("Event sources are completely unavailable" in r for r in sig.suppression_reasons)


def test_no_events_is_distinct_from_unavailable():
    """Prove NO_EVENTS and EVENT_SOURCE_UNAVAILABLE are distinct across predictor layers."""
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    mock_classifier = MagicMock(spec=EventClassifier)
    mock_classifier.classify_batch.return_value = EventBatchResult(
        events=[],
        status="NO_EVENTS",
        errors=[],
    )

    predictor = Predictor(event_classifier=mock_classifier)
    predictor.data_fetcher.check_staleness = MagicMock(return_value=False)
    sig_no_events = predictor.generate_signal(
        symbol="RELIANCE",
        stock_df=stock_df,
        index_df=index_df,
        horizon="INTRADAY",
    )

    assert sig_no_events.event_source_status == "NO_EVENTS"
    assert sig_no_events.event_source_status != "EVENT_SOURCE_UNAVAILABLE"


def test_event_status_in_scheduler_cycle():
    """Verify event status correctly flows through a full scheduler cycle."""
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    scheduler = Scheduler()
    scheduler.data_fetcher.check_staleness = MagicMock(return_value=False)
    scheduler.get_event_context = MagicMock(
        return_value=EventContext(
            macro_events=None,
            corporate_events=None,
            news_articles=None,
            macro_status="EVENT_SOURCE_PARTIAL",
            corporate_status="NO_EVENTS",
            news_status="NO_EVENTS",
            status="EVENT_SOURCE_PARTIAL",
            as_of=datetime.now(UTC),
            refreshed_at=datetime.now(UTC),
            errors=["Macro service degraded"],
        )
    )

    res = scheduler.run_one_cycle_for_symbol(
        symbol="RELIANCE",
        stock_df=stock_df,
        index_df=index_df,
        return_structured=True,
    )

    assert res.success is True
    assert res.signal is not None
    for hor, sig in res.signal.signals.items():
        assert sig.event_source_status == "EVENT_SOURCE_PARTIAL"


def test_event_status_stored_in_history():
    """Verify event_source_status persists and is reconstructable from HistoryManager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_history.db"
        hm = HistoryManager(db_path=db_path)
        try:
            stock_df = _synthetic_bars()
            index_df = _synthetic_bars()

            predictor = Predictor()
            predictor.data_fetcher.check_staleness = MagicMock(return_value=False)
            ctx = PredictionContext(
                symbol="INFY",
                timestamp=datetime.now(UTC),
                market_data=stock_df,
                index_data=index_df,
                event_status="EVENT_SOURCE_PARTIAL",
            )
            multi_sig = predictor.predict_context(ctx, horizons=["INTRADAY"])
            sig = multi_sig.signals["INTRADAY"]
            assert sig.event_source_status == "EVENT_SOURCE_PARTIAL"

            pred_id = hm.save_prediction(sig)
            assert pred_id is not None

            records = hm.get_predictions(symbol="INFY", horizon="INTRADAY")
            assert len(records) > 0
            assert records[0].event_source_status == "EVENT_SOURCE_PARTIAL"

            reconstructed_sig = hm.get_prediction_signal_by_key(sig.prediction_key)
            assert reconstructed_sig is not None
            assert reconstructed_sig.event_source_status == "EVENT_SOURCE_PARTIAL"
        finally:
            hm.engine.dispose()


def test_event_source_partial_propagates_to_api():
    """Verify partial event source availability status is serialized to API response."""
    client = TestClient(app)

    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    mock_event_ctx = EventContext(
        macro_events=None,
        corporate_events=None,
        news_articles=None,
        macro_status="EVENT_SOURCE_PARTIAL",
        corporate_status="NO_EVENTS",
        news_status="NO_EVENTS",
        status="EVENT_SOURCE_PARTIAL",
        as_of=datetime.now(UTC),
        refreshed_at=datetime.now(UTC),
        errors=["Feed latency elevated"],
    )

    with patch("api.scheduler.get_event_context", return_value=mock_event_ctx), \
         patch("api.scheduler.data_fetcher.fetch_ohlcv", return_value=stock_df), \
         patch("api.scheduler.data_fetcher.fetch_nifty_index", return_value=index_df), \
         patch("api.scheduler.data_fetcher.check_staleness", return_value=False), \
         patch("api.scheduler.predictor.data_fetcher.check_staleness", return_value=False):

        response = client.get("/api/v1/signal/TCS")
        assert response.status_code == 200
        payload = response.json()
        assert "signals" in payload
        for hor, sig_data in payload["signals"].items():
            assert sig_data["event_source_status"] == "EVENT_SOURCE_PARTIAL"
