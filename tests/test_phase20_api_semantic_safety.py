from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app, humanize_reasoning
from api_schemas import HorizonSignalOut
from predictor import MultiHorizonSignal, PredictionSignal
from scheduler import CycleResult, EventContext

BANNED_DIRECTIVE_PHRASES = [
    "proceed with entry",
    "consider hedging",
    "reducing exposure",
    "place a",
    "buy now",
    "sell now",
    "execute",
    "take profit",
    "set stop",
]


def _synthetic_bars(n: int = 40) -> pd.DataFrame:
    now_utc = datetime.now(UTC)
    dates = pd.date_range(end=now_utc, periods=n, freq="5min")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [105.0 + i for i in range(n)],
            "Low": [95.0 + i for i in range(n)],
            "Close": [102.0 + i for i in range(n)],
            "Volume": [1000 + i * 10 for i in range(n)],
        },
        index=dates,
    )


@pytest.fixture
def mock_empty_event_context():
    return EventContext(
        macro_events=[],
        corporate_events=[],
        news_articles=[],
        macro_status="NO_EVENTS",
        corporate_status="NO_EVENTS",
        news_status="NO_EVENTS",
        status="NO_EVENTS",
        as_of=datetime.now(UTC),
        refreshed_at=datetime.now(UTC),
        errors=[],
    )


def test_api_003_model_score_distinct_from_calibrated_probability():
    """Verify model_score (raw ensemble confidence) is distinct from calibrated_probability."""
    schema = HorizonSignalOut(
        horizon="INTRADAY",
        action="HOLD",
        verdict_text="Model classified observation as neutral.",
        model_score=0.78,
        calibrated_probability=None,
        calibration_status="UNAVAILABLE",
        edge_status="NOT_VALIDATED",
        live_eligible=False,
    )
    assert schema.model_score == 0.78
    assert schema.calibrated_probability is None
    assert schema.confidence is None  # deprecated alias reflects calibrated probability


def test_api_003_uncalibrated_confidence_is_null_not_raw_score(mock_empty_event_context):
    """When calibration is not confirmed, confidence and calibrated_probability must be null, never raw score."""
    client = TestClient(app)
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    with (
        patch("api.scheduler.data_fetcher.fetch_ohlcv", return_value=stock_df),
        patch("api.scheduler.data_fetcher.fetch_nifty_index", return_value=index_df),
        patch("api.scheduler.data_fetcher.check_staleness", return_value=False),
        patch("api.scheduler.get_event_context", return_value=mock_empty_event_context),
        patch("api.scheduler.resolve_pending_outcomes", return_value=None),
        patch("api.scheduler.predictor.data_fetcher.check_staleness", return_value=False),
    ):

        response = client.get("/api/v1/signal/RELIANCE")
        assert response.status_code == 200
        payload = response.json()
        assert "signals" in payload
        for hor, sig in payload["signals"].items():
            assert sig["model_score"] >= 0.0
            if sig["calibration_status"] != "CALIBRATED":
                assert sig["calibrated_probability"] is None
                assert sig["confidence"] is None


def test_api_003_calibration_status_correct_when_calibrated(mock_empty_event_context):
    """Verify calibration_status is CALIBRATED when calibrated_confidence is present."""
    client = TestClient(app)
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    mock_sig = PredictionSignal(
        symbol="RELIANCE",
        timestamp=datetime.now(UTC),
        horizon="INTRADAY",
        action="BUY",
        model_predicted_class="UP",
        model_version="v1.0",
        feature_version="v1.0",
        raw_confidence=0.75,
        risk_adjusted_confidence=0.75,
        calibrated_confidence=0.71,
        agreement_fraction=1.0,
        downside_summary="Downside risk defined by support.",
        upside_summary="Upside momentum intact.",
        is_safe_to_trade_live=True,
        prediction_key="key123",
    )

    multi_sig = MultiHorizonSignal(
        symbol="RELIANCE",
        timestamp=datetime.now(UTC),
        signals={"INTRADAY": mock_sig},
        primary_action="BUY",
        primary_horizon="INTRADAY",
        reasoning=["Model proven with historical edge."],
        prediction_key="multi_key_123",
    )
    cycle_res = CycleResult(success=True, status="SUCCESS", symbol="RELIANCE", signal=multi_sig)

    with (
        patch("api.scheduler.data_fetcher.fetch_ohlcv", return_value=stock_df),
        patch("api.scheduler.data_fetcher.fetch_nifty_index", return_value=index_df),
        patch("api.scheduler.data_fetcher.check_staleness", return_value=False),
        patch("api.scheduler.get_event_context", return_value=mock_empty_event_context),
        patch("api.scheduler.resolve_pending_outcomes", return_value=None),
        patch("api.scheduler.run_one_cycle_for_symbol", return_value=cycle_res),
        patch("api.evaluate_model_validity") as mock_validity,
    ):

        val_res = MagicMock()
        val_res.is_live_eligible = True
        val_res.validity_status = "VALID"
        val_res.edge_status = "EDGE_CONFIRMED"
        mock_validity.return_value = val_res

        response = client.get("/api/v1/signal/RELIANCE")
        assert response.status_code == 200
        payload = response.json()
        sig_data = payload["signals"]["INTRADAY"]
        assert sig_data["calibration_status"] == "CALIBRATED"
        assert sig_data["calibrated_probability"] == 0.71
        assert sig_data["live_eligible"] is True


def test_api_003_calibration_status_unavailable_when_not_calibrated(mock_empty_event_context):
    """Verify calibration_status is UNAVAILABLE or INSUFFICIENT_DATA when uncalibrated."""
    client = TestClient(app)
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    with (
        patch("api.scheduler.data_fetcher.fetch_ohlcv", return_value=stock_df),
        patch("api.scheduler.data_fetcher.fetch_nifty_index", return_value=index_df),
        patch("api.scheduler.data_fetcher.check_staleness", return_value=False),
        patch("api.scheduler.get_event_context", return_value=mock_empty_event_context),
        patch("api.scheduler.resolve_pending_outcomes", return_value=None),
    ):

        response = client.get("/api/v1/signal/TCS")
        assert response.status_code == 200
        payload = response.json()
        for hor, sig in payload["signals"].items():
            if sig["calibrated_probability"] is None:
                assert sig["calibration_status"] in ("UNAVAILABLE", "INSUFFICIENT_DATA")


def test_api_004_no_directive_trading_language_in_verdict_text(mock_empty_event_context):
    """Verify that no banned directive trading phrases appear in any verdict_text."""
    client = TestClient(app)
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    with (
        patch("api.scheduler.data_fetcher.fetch_ohlcv", return_value=stock_df),
        patch("api.scheduler.data_fetcher.fetch_nifty_index", return_value=index_df),
        patch("api.scheduler.data_fetcher.check_staleness", return_value=False),
        patch("api.scheduler.get_event_context", return_value=mock_empty_event_context),
        patch("api.scheduler.resolve_pending_outcomes", return_value=None),
    ):

        response = client.get("/api/v1/signal/INFY")
        assert response.status_code == 200
        payload = response.json()
        for hor, sig in payload["signals"].items():
            v_text = sig["verdict_text"].lower()
            for phrase in BANNED_DIRECTIVE_PHRASES:
                assert phrase not in v_text, f"Banned phrase '{phrase}' found in verdict_text: {v_text}"


def test_api_004_no_directive_trading_language_in_reasoning():
    """Verify humanize_reasoning strips or avoids directive trading instructions."""
    raw_sample_reasons = [
        "Ensemble model lean: UP (raw confidence 0.65, model agreement 80%).",
        "Ensemble model lean: DOWN (raw confidence 0.58, model agreement 60%).",
        "Global risk level: NORMAL",
        "Global risk level: ELEVATED",
        "No specific events currently tagged as affecting this stock.",
        "Model leaned BUY, but confidence is not yet calibrated on enough history — forced to HOLD rather than act on an unproven strategy.",
    ]
    humanized = humanize_reasoning(raw_sample_reasons)
    for reason in humanized:
        r_lower = reason.lower()
        for phrase in BANNED_DIRECTIVE_PHRASES:
            assert phrase not in r_lower, f"Banned phrase '{phrase}' found in humanized reasoning: {reason}"


def test_api_004_verdict_uses_analytical_framing(mock_empty_event_context):
    """Ensure verdict text uses analytical framing describing classification rather than giving orders."""
    client = TestClient(app)
    stock_df = _synthetic_bars()
    index_df = _synthetic_bars()

    with (
        patch("api.scheduler.data_fetcher.fetch_ohlcv", return_value=stock_df),
        patch("api.scheduler.data_fetcher.fetch_nifty_index", return_value=index_df),
        patch("api.scheduler.data_fetcher.check_staleness", return_value=False),
        patch("api.scheduler.get_event_context", return_value=mock_empty_event_context),
        patch("api.scheduler.resolve_pending_outcomes", return_value=None),
    ):

        response = client.get("/api/v1/signal/HDFCBANK")
        assert response.status_code == 200
        payload = response.json()
        for hor, sig in payload["signals"].items():
            v_text = sig["verdict_text"]
            assert any(
                term in v_text for term in ["Model", "Holding", "neutral", "classified", "withheld", "Statistical"]
            )
