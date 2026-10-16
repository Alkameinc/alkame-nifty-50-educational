import os
import tempfile
from datetime import UTC, date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from api_schemas import MultiHorizonSignalResponse
from config import DEFAULT_DEV_ADMIN_KEY
from database import SessionLocal
from ensemble_manager import EnsembleManager
from event_classifier import EventClassifier
from feature_engineer import FeatureEngineer
from history_manager import HistoryManager
from macro_calendar import MacroEvent
from models import AuditLog
from predictor import PredictionContext, Predictor
from runtime_validator import (
    STATUS_EDGE_CONFIRMED,
    STATUS_SUFFICIENT,
    RuntimeValidator,
)
from scheduler import EventContext


@pytest.fixture
def fixed_market_data() -> pd.DataFrame:
    """180 bars of deterministic OHLCV across 2 days, timezone-aware UTC, DatetimeIndex."""
    rng = np.random.default_rng(101)
    base_price = 1000.0
    d1 = pd.date_range("2026-03-01 09:15:00", periods=90, freq="5min", tz=UTC)
    d2 = pd.date_range("2026-03-02 09:15:00", periods=90, freq="5min", tz=UTC)
    dates = d1.union(d2)
    rows = []
    price = base_price
    for _ in range(len(dates)):
        drift = rng.normal(0.2, 1.0)
        open_p = price
        close_p = open_p + drift
        high_p = max(open_p, close_p) + abs(rng.normal(0.5, 0.2))
        low_p = min(open_p, close_p) - abs(rng.normal(0.5, 0.2))
        volume = int(abs(rng.normal(25000, 5000)))
        rows.append([open_p, high_p, low_p, close_p, volume])
        price = close_p
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=dates)


@pytest.fixture
def fixed_index_data(fixed_market_data: pd.DataFrame) -> pd.DataFrame:
    """180 bars of deterministic NIFTY baseline OHLCV with same timestamps."""
    rng = np.random.default_rng(202)
    rows = []
    price = 22000.0
    for _ in range(len(fixed_market_data)):
        drift = rng.normal(0.1, 5.0)
        open_p = price
        close_p = open_p + drift
        high_p = max(open_p, close_p) + abs(rng.normal(2.0, 1.0))
        low_p = min(open_p, close_p) - abs(rng.normal(2.0, 1.0))
        volume = int(abs(rng.normal(100000, 20000)))
        rows.append([open_p, high_p, low_p, close_p, volume])
        price = close_p
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=fixed_market_data.index)


@pytest.fixture
def fixed_events() -> dict:
    """Fixed macro and news events."""
    macro = MacroEvent(
        event_date=date(2026, 3, 1),
        event_type="RBI_POLICY",
        label="RBI MPC Policy Stance",
        scope="SECTOR",
        sector_hint="Banking",
        impact_window_days_before=1,
        impact_window_days_after=1,
    )
    news = {
        "title": "Reliance accelerates solar power project commercial rollout",
        "symbol": "RELIANCE",
        "published_at": "2026-03-01T09:30:00+00:00",
        "sentiment_score": 0.35,
    }
    return {"macro": [macro], "news": [news]}


@pytest.fixture
def isolated_pipeline():
    """Deterministic, isolated pipeline components with in-memory SQLite and fresh ensemble."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "integration_pipeline.db"
        hm = HistoryManager(db_path=db_path)
        em = EnsembleManager()
        fe = FeatureEngineer()
        ec = EventClassifier()
        rv = RuntimeValidator()

        # Seed calibration & edge data
        rng = np.random.default_rng(42)
        confidences = rng.uniform(0.6, 0.85, size=150)
        correct = rng.uniform(0, 1, size=150) < confidences
        calib = rv.compute_calibration(pd.DataFrame({"confidence": confidences, "correct": correct}))

        dates = pd.date_range("2026-01-01", periods=100, freq="D", tz=UTC)
        strat_returns = pd.Series(np.full(100, 0.5), index=dates)
        base_returns = pd.Series(np.full(100, 0.1), index=dates)
        edge = rv.compute_edge_vs_baseline(strat_returns, base_returns)

        predictor = Predictor(
            ensemble_manager=em,
            feature_engineer=fe,
            event_classifier=ec,
            runtime_validator=rv,
        )
        predictor.data_fetcher.check_staleness = MagicMock(return_value=False)

        yield {
            "history_manager": hm,
            "ensemble_manager": em,
            "feature_engineer": fe,
            "event_classifier": ec,
            "runtime_validator": rv,
            "predictor": predictor,
            "calib": calib,
            "edge": edge,
        }
        hm.engine.dispose()


def test_integration_full_pipeline_deterministic(isolated_pipeline, fixed_market_data, fixed_index_data, fixed_events):
    """TEST-001 Stage 1: End-to-end data -> validation -> features -> model -> calibration -> edge -> risk -> predictor -> history."""
    predictor: Predictor = isolated_pipeline["predictor"]
    hm: HistoryManager = isolated_pipeline["history_manager"]
    calib = isolated_pipeline["calib"]
    edge = isolated_pipeline["edge"]

    assert calib.status == STATUS_SUFFICIENT
    assert edge.status == STATUS_EDGE_CONFIRMED

    ctx = PredictionContext(
        symbol="RELIANCE",
        timestamp=fixed_market_data.index[-1],
        market_data=fixed_market_data,
        index_data=fixed_index_data,
        macro_events=fixed_events["macro"],
        news_events=fixed_events["news"],
        data_status="LIVE",
        event_status="EVENTS_AVAILABLE",
        calibration_results={"INTRADAY": calib},
        edge_check_results={"INTRADAY": edge},
        as_of=fixed_market_data.index[-1],
    )

    multi_sig = predictor.predict_context(ctx, horizons=["INTRADAY"])
    assert "INTRADAY" in multi_sig.signals
    sig = multi_sig.signals["INTRADAY"]

    assert sig.symbol == "RELIANCE"
    assert sig.horizon == "INTRADAY"
    assert sig.action in ("BUY", "SELL", "HOLD")
    assert sig.model_predicted_class in ("UP", "DOWN", "FLAT")
    assert sig.raw_confidence > 0.0
    assert sig.downside_summary  # Charter requirement: never empty
    assert sig.upside_summary
    assert sig.prediction_key is not None

    pred_id = hm.save_prediction(sig)
    assert pred_id is not None
    saved = hm.get_prediction_by_key(sig.prediction_key)
    assert saved is not None
    assert saved.symbol == "RELIANCE"


def test_integration_no_future_data_in_features(fixed_market_data, fixed_index_data):
    """TEST-001 Stage 2: Enforce causal contract (shift >= 1) — changing future bar does not affect past features."""
    fe = FeatureEngineer()
    df_clean = fixed_market_data.copy()
    feat_original = fe.engineer_features_for_horizon(df_clean, fixed_index_data, horizon="INTRADAY")

    # Mutate the last row's close heavily
    df_mutated = fixed_market_data.copy()
    df_mutated.iloc[-1, df_mutated.columns.get_loc("Close")] = 99999.0
    feat_mutated = fe.engineer_features_for_horizon(df_mutated, fixed_index_data, horizon="INTRADAY")

    feat_cols = [c for c in feat_original.columns if c.endswith("_feat")]
    assert len(feat_cols) > 0

    # Row -2 (second to last) must be mathematically identical before and after mutating row -1
    for col in feat_cols:
        val_orig = feat_original[col].iloc[-2]
        val_mut = feat_mutated[col].iloc[-2]
        if pd.isna(val_orig):
            assert pd.isna(val_mut)
        else:
            assert np.isclose(val_orig, val_mut, atol=1e-8), f"Future data leakage detected in feature '{col}'!"


def test_integration_timestamps_are_utc_aware(isolated_pipeline, fixed_market_data, fixed_index_data, fixed_events):
    """TEST-001 Stage 3: DATA-004 contract — all internal timestamps must be timezone-aware UTC."""
    assert fixed_market_data.index.tz is not None
    assert str(fixed_market_data.index.tz) == "UTC"

    predictor: Predictor = isolated_pipeline["predictor"]
    ctx = PredictionContext(
        symbol="RELIANCE",
        timestamp=fixed_market_data.index[-1],
        market_data=fixed_market_data,
        index_data=fixed_index_data,
        as_of=fixed_market_data.index[-1],
    )
    multi_sig = predictor.predict_context(ctx, horizons=["INTRADAY"])
    sig = multi_sig.signals["INTRADAY"]

    assert sig.timestamp.tzinfo is not None
    assert sig.timestamp.tzinfo == UTC
    assert multi_sig.timestamp.tzinfo == UTC


def test_integration_feature_schema_complete(fixed_market_data, fixed_index_data):
    """TEST-001 Stage 4: Feature schema complete, ML columns end with _feat, latest row contains 0 NaNs."""
    fe = FeatureEngineer()
    engineered = fe.engineer_features_for_horizon(fixed_market_data, fixed_index_data, horizon="INTRADAY")
    assert engineered is not None and not engineered.empty

    feat_cols = [c for c in engineered.columns if c.endswith("_feat")]
    assert len(feat_cols) >= 10, f"Expected at least 10 ML feature columns, found {len(feat_cols)}"

    latest_row = engineered[feat_cols].iloc[[-1]]
    nan_count = latest_row.isna().sum().sum()
    assert nan_count == 0, f"Latest feature row contains {nan_count} NaNs passed to model!"


def test_integration_risk_state_applied(isolated_pipeline, fixed_market_data, fixed_index_data):
    """TEST-001 Stage 5: Global risk state applied to confidence adjustments."""
    predictor: Predictor = isolated_pipeline["predictor"]
    calib = isolated_pipeline["calib"]
    edge = isolated_pipeline["edge"]

    with patch.object(predictor.global_risk_monitor, "compute_composite_risk") as mock_risk:
        mock_reading = MagicMock()
        mock_reading.risk_level = "ELEVATED"
        mock_reading.dominant_driver = "VIX_SURGE"
        mock_risk.return_value = mock_reading

        with patch.object(predictor.global_risk_monitor, "get_confidence_multiplier", return_value=0.75):
            ctx = PredictionContext(
                symbol="RELIANCE",
                timestamp=fixed_market_data.index[-1],
                market_data=fixed_market_data,
                index_data=fixed_index_data,
                calibration_results={"INTRADAY": calib},
                edge_check_results={"INTRADAY": edge},
                as_of=fixed_market_data.index[-1],
            )
            multi_sig = predictor.predict_context(ctx, horizons=["INTRADAY"])
            sig = multi_sig.signals["INTRADAY"]

            assert sig.global_risk_level == "ELEVATED"
            assert sig.risk_adjusted_confidence <= sig.raw_confidence


def test_integration_event_status_propagated(isolated_pipeline, fixed_market_data, fixed_index_data):
    """TEST-001 Stage 6: Event status correctly recorded across PredictionSignal and MultiHorizonSignal."""
    predictor: Predictor = isolated_pipeline["predictor"]

    ctx = PredictionContext(
        symbol="RELIANCE",
        timestamp=fixed_market_data.index[-1],
        market_data=fixed_market_data,
        index_data=fixed_index_data,
        event_status="EVENT_SOURCE_PARTIAL",
        as_of=fixed_market_data.index[-1],
    )
    multi_sig = predictor.predict_context(ctx, horizons=["INTRADAY"])
    assert multi_sig.signals["INTRADAY"].event_source_status == "EVENT_SOURCE_PARTIAL"


def test_integration_audit_record_created():
    """TEST-001 Stage 7: Authenticated/admin actions create audit trail records in DB."""
    client = TestClient(app)
    admin_key = os.environ.get("ALKAME_ADMIN_KEY", DEFAULT_DEV_ADMIN_KEY)

    response = client.post(
        "/api/v1/risk/toggle?enabled=true",
        headers={"X-API-Key": admin_key},
    )
    assert response.status_code == 200

    # Verify audit record exists in database
    with SessionLocal() as db:
        logs = db.query(AuditLog).filter(AuditLog.action == "TOGGLE_RISK").all()
        assert len(logs) > 0
        last_log = logs[-1]
        assert last_log.client_role == "ADMIN"
        assert "enabled=True" in last_log.details


def test_integration_history_record_saved(isolated_pipeline, fixed_market_data, fixed_index_data):
    """TEST-001 Stage 8: History records are persistently saved and retrieved via unique prediction_key."""
    predictor: Predictor = isolated_pipeline["predictor"]
    hm: HistoryManager = isolated_pipeline["history_manager"]

    ctx = PredictionContext(
        symbol="RELIANCE",
        timestamp=fixed_market_data.index[-1],
        market_data=fixed_market_data,
        index_data=fixed_index_data,
        as_of=fixed_market_data.index[-1],
    )
    multi_sig = predictor.predict_context(ctx, horizons=["INTRADAY"])
    sig = multi_sig.signals["INTRADAY"]

    pred_id = hm.save_prediction(sig)
    assert pred_id is not None

    record = hm.get_prediction_by_key(sig.prediction_key)
    assert record is not None
    assert record.symbol == sig.symbol
    assert record.action == sig.action
    assert record.horizon == sig.horizon
    assert record.prediction_key == sig.prediction_key


def test_integration_api_response_schema_valid(fixed_market_data, fixed_index_data):
    """TEST-001 Stage 9: API endpoint returns valid, fully-formed schema conforming to MultiHorizonSignalResponse."""
    client = TestClient(app)

    mock_event_ctx = EventContext(
        macro_events=[],
        corporate_events=[],
        news_articles=[],
        macro_status="NO_EVENTS",
        corporate_status="NO_EVENTS",
        news_status="NO_EVENTS",
        status="NO_EVENTS",
        as_of=fixed_market_data.index[-1],
        refreshed_at=fixed_market_data.index[-1],
        errors=[],
    )

    with patch("api.scheduler.data_fetcher.fetch_ohlcv", return_value=fixed_market_data), \
         patch("api.scheduler.data_fetcher.fetch_nifty_index", return_value=fixed_index_data), \
         patch("api.scheduler.data_fetcher.check_staleness", return_value=False), \
         patch("api.scheduler.get_event_context", return_value=mock_event_ctx), \
         patch("api.scheduler.resolve_pending_outcomes", return_value=None):

        response = client.get("/api/v1/signal/RELIANCE")
        assert response.status_code == 200
        payload = response.json()

        # Validate with pydantic
        parsed = MultiHorizonSignalResponse(**payload)
        assert parsed.symbol == "RELIANCE"
        assert len(parsed.signals) > 0
        intraday = parsed.signals["INTRADAY"]
        assert intraday.action in ("BUY", "SELL", "HOLD")
        assert intraday.model_score >= 0.0


def test_integration_idempotent_same_input_same_output(isolated_pipeline, fixed_market_data, fixed_index_data):
    """TEST-001 Stage 10: 3 consecutive runs with identical inputs produce byte-for-byte identical output and keys."""
    predictor: Predictor = isolated_pipeline["predictor"]
    keys = []
    actions = []
    scores = []

    for _ in range(3):
        ctx = PredictionContext(
            symbol="RELIANCE",
            timestamp=fixed_market_data.index[-1],
            market_data=fixed_market_data.copy(),
            index_data=fixed_index_data.copy(),
            as_of=fixed_market_data.index[-1],
        )
        multi_sig = predictor.predict_context(ctx, horizons=["INTRADAY"])
        sig = multi_sig.signals["INTRADAY"]
        keys.append(sig.prediction_key)
        actions.append(sig.action)
        scores.append(sig.raw_confidence)

    assert len(set(keys)) == 1, f"Prediction keys diverged across identical runs: {keys}"
    assert len(set(actions)) == 1, f"Actions diverged across identical runs: {actions}"
    assert len(set(scores)) == 1, f"Scores diverged across identical runs: {scores}"
