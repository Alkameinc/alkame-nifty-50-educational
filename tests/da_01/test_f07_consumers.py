"""F07 consumer compatibility through actual validator, gate and application code.

Market/model/event inputs are deterministic fixtures; history and execution use
their real implementations. These tests do not repair scheduler model selection,
the fallback's missing bins, or scanner/scalping's absent evidence forwarding.
"""

import importlib.util
import socket
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
import ensemble_manager
import health_monitor
import history_manager
import scheduler as scheduler_module
from backtester import Backtester
from ensemble_manager import EnsemblePrediction
from event_classifier import EventBatchResult
from history_manager import HistoryManager
from model_trainer import ModelTrainer
from models import BacktestMetric
from predictor import PredictionSignal, Predictor
from runtime_validator import CalibrationResult, RuntimeValidator
from scalping import ScalpingEngine
from scanner import OpportunityScanner
from scheduler import Scheduler


@pytest.fixture(autouse=True)
def isolated_io(monkeypatch, tmp_path):
    """Keep real history/health writes in a temporary database; disallow network."""

    def no_network(*args, **kwargs):
        raise AssertionError("F07 consumer tests must not use external network services")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(config, "REQUIRED_DIRS", [tmp_path / "runtime"])
    manager = HistoryManager(db_path=tmp_path / "history.sqlite3")
    monkeypatch.setattr(health_monitor, "SessionLocal", manager.SessionLocal)
    yield manager
    manager.engine.dispose()


@pytest.fixture
def prices():
    index = pd.date_range("2026-09-01 09:15", periods=400, freq="5min")
    close = 100.0 + np.arange(len(index)) * 0.1
    return pd.DataFrame(
        {"Open": close, "High": close + 0.2, "Low": close - 0.2, "Close": close, "Volume": 1000},
        index=index,
    )


class StableModel:
    """A deterministic estimator fixture; real walk-forward code fits/votes it."""

    def fit(self, X, y):
        assert len(X) == len(y) and len(X) > 0
        return self

    def predict(self, X):
        return np.repeat("UP", len(X))


def make_backtester(prices, history, confidence=1.0):
    X = pd.DataFrame({"fixture_feature": np.arange(320)}, index=prices.index[:320])
    y = pd.Series("UP", index=X.index)
    # Retain actual chronological and walk-forward splitting. Only the prepared
    # model inputs, training service and estimator outputs are fixture boundaries.
    trainer = SimpleNamespace(
        prepare_dataset=lambda *args, **kwargs: (X, y, list(X.columns)),
        time_based_split=ModelTrainer.time_based_split,
        walk_forward_split=ModelTrainer.walk_forward_split,
    )
    ensemble = SimpleNamespace(
        model_trainer=trainer,
        train_ensemble_for_symbol=lambda *args, **kwargs: SimpleNamespace(success=True),
        predict=lambda symbol, rows, **kwargs: [
            EnsemblePrediction("UP", confidence, 1.0, {"fixture": "UP"}, "fixture-v1", "fixture-v1")
            for _ in range(len(rows))
        ],
        _build_base_estimators=lambda: {"gradient_boosting": StableModel(), "random_forest": StableModel()},
    )
    return Backtester(ensemble_manager=ensemble, runtime_validator=RuntimeValidator(), history_manager=history)


def make_predictor(prices):
    fetcher = SimpleNamespace(
        check_staleness=lambda *args, **kwargs: False,
        fetch_ohlcv=lambda *args, **kwargs: prices.copy(),
        fetch_nifty_index=lambda *args, **kwargs: prices.copy(),
        fetch_daily_ohlcv_incremental=lambda *args, **kwargs: prices.copy(),
    )
    features = SimpleNamespace(
        engineer_features_for_horizon=lambda *args, **kwargs: pd.DataFrame({"atr": [1.0], "rsi": [50.0]})
    )
    return Predictor(
        data_fetcher=fetcher,
        feature_engineer=features,
        ensemble_manager=SimpleNamespace(
            predict=lambda *args, **kwargs: [
                EnsemblePrediction("UP", 0.65, 1.0, {"fixture": "UP"}, "fixture-v1", "fixture-v1")
            ]
        ),
        event_classifier=SimpleNamespace(classify_batch=lambda **kwargs: EventBatchResult([], "NO_EVENTS")),
        global_risk_monitor=SimpleNamespace(
            compute_composite_risk=lambda: SimpleNamespace(risk_level="NORMAL", dominant_driver=None),
            get_confidence_multiplier=lambda *args: 1.0,
            get_toggle_state=lambda: SimpleNamespace(enabled=False),
        ),
        runtime_validator=RuntimeValidator(),
    )


def make_scheduler(prices, history):
    predictor = make_predictor(prices)
    return Scheduler(
        predictor=predictor,
        data_fetcher=predictor.data_fetcher,
        history_manager=history,
        backtester=make_backtester(prices, history),
        event_classifier=predictor.event_classifier,
        corporate_events_fetcher=SimpleNamespace(fetch_all_for_symbol=lambda *args: []),
        macro_calendar=SimpleNamespace(get_active_macro_events=lambda *args: []),
        news_sentiment_fetcher=SimpleNamespace(get_news_for_symbol=lambda *args: []),
    )


def seed_resolved_history(history, confidences):
    for i, confidence in enumerate(confidences):
        signal = PredictionSignal(
            symbol="RELIANCE",
            timestamp=datetime(2026, 9, 1, 10),
            horizon="INTRADAY",
            action="BUY",
            model_predicted_class="UP",
            model_version="fixture-v1",
            feature_version="fixture-v1",
            raw_confidence=0.65,
            risk_adjusted_confidence=confidence,
            calibrated_confidence=None,
            agreement_fraction=1.0,
            downside_summary="Fixture downside",
            upside_summary="Fixture upside",
        )
        prediction_id = history.save_prediction(signal, is_out_of_sample=True)
        assert prediction_id is not None
        assert history.resolve_outcome(prediction_id, "UP" if i < 39 else "DOWN")


def select_fixture_model_version(monkeypatch):
    # Existing Scheduler uses a dummy prediction's model_version, whose real
    # default UNKNOWN is excluded by HistoryManager. Supply model metadata at
    # that boundary to reach the real-history branch without changing F09.
    monkeypatch.setattr(ensemble_manager, "EnsemblePrediction", partial(EnsemblePrediction, model_version="fixture-v1"))


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
def test_legacy_calibration_constructors_keep_unknown_accounting(keyword):
    fields = dict(status="SUFFICIENT", n_samples=60, expected_calibration_error=0.0, is_well_calibrated=True, bins=[])
    result = CalibrationResult(**fields) if keyword else CalibrationResult(*fields.values())
    assert result.n_samples == 60
    assert result.input_count is None
    assert result.rejected_count is None
    assert result.rejection_reasons is None


@pytest.mark.parametrize("invalid", [False, True], ids=["valid-control", "invalid-model-confidence"])
def test_actual_held_out_backtest_consumes_calibration(prices, isolated_io, invalid):
    backtester = make_backtester(prices, isolated_io, np.nan if invalid else 1.0)
    result = backtester.run_backtest_for_symbol("RELIANCE", prices, prices)
    assert result.success, result.error
    assert result.n_test_predictions >= 50
    assert result.calibration_status == ("INSUFFICIENT_DATA" if invalid else "SUFFICIENT")
    if invalid:
        assert result.calibration_ece is None
        assert not result.is_live_worthy
        assert any("Only 0 historical predictions" in reason for reason in result.gate_reasons)
    else:
        assert result.calibration_ece == pytest.approx(0.0)
    assert set(result.cost_sensitivity) == {"OPTIMISTIC", "BASE", "PESSIMISTIC", "STRESS"}
    with isolated_io.SessionLocal() as db:
        persisted = db.query(BacktestMetric).filter_by(symbol="RELIANCE", horizon="INTRADAY").one()
        assert persisted.calibration_status == result.calibration_status
        assert persisted.calibration_ece == result.calibration_ece
        assert persisted.is_live_worthy == result.is_live_worthy


def test_actual_walk_forward_backtest_preserves_valid_voting_calibration(prices, isolated_io):
    backtester = make_backtester(prices, isolated_io)
    result = backtester.run_walk_forward_backtest("RELIANCE", prices, prices, n_splits=3)
    assert result.success, result.error
    assert len(result.walk_forward_folds) == 3
    assert result.n_test_predictions == sum(fold["test_size"] for fold in result.walk_forward_folds)
    assert result.n_test_predictions >= 50
    assert result.calibration_status == "SUFFICIENT"
    assert result.calibration_ece == pytest.approx(0.0)
    assert len(result.cost_sensitivity) == 4


def test_scheduler_fallback_preserves_unknown_accounting(prices, isolated_io):
    scheduler = make_scheduler(prices, isolated_io)
    snapshot = scheduler.refresh_live_worthiness("RELIANCE", prices, prices)
    calibration = snapshot.calibration_result
    assert calibration.status == "SUFFICIENT"
    assert calibration.n_samples >= 50
    assert calibration.input_count is None
    assert calibration.rejected_count is None
    assert calibration.rejection_reasons is None
    assert calibration.bins == []
    # Missing fallback bins are an existing F09 limitation, not evidence that
    # this manually constructed summary was validated by the F07 calculation.
    assert scheduler.predictor.runtime_validator.get_calibrated_confidence(0.65, calibration) is None
    assert scheduler.get_cached_live_worthiness("RELIANCE") is snapshot


@pytest.mark.parametrize("invalid", [False, True], ids=["valid-control", "invalid-real-history"])
def test_scheduler_real_history_reaches_actual_gate_and_predictor(prices, isolated_io, monkeypatch, invalid):
    select_fixture_model_version(monkeypatch)
    seed_resolved_history(isolated_io, [np.inf if invalid else 0.65] * 60)
    scheduler = make_scheduler(prices, isolated_io)
    snapshot = scheduler.refresh_live_worthiness("RELIANCE", prices, prices)
    calibration = snapshot.calibration_result
    assert calibration.status == ("INSUFFICIENT_DATA" if invalid else "SUFFICIENT")
    assert calibration.n_samples == (0 if invalid else 60)
    assert calibration.input_count == 60
    assert calibration.rejected_count == (60 if invalid else 0)
    assert sum(calibration.rejection_reasons.values()) == calibration.rejected_count
    assert scheduler.get_cached_live_worthiness("RELIANCE") is snapshot
    # Give the actual gate a confirmed edge calculated from stable return data.
    # This keeps suppression attributable to calibration and preserves F08.
    snapshot.edge_check_result = scheduler.predictor.runtime_validator.compute_edge_vs_baseline(
        pd.Series([0.5] * 60), pd.Series([0.0] * 60), slippage_bps=0, transaction_cost_bps=0
    )
    assert snapshot.edge_check_result.status == "EDGE_CONFIRMED"
    result = scheduler.run_one_cycle_for_symbol(
        "RELIANCE", prices, prices, macro_events=[], corporate_events=[], news_articles=[], return_structured=True
    )
    assert result.success, result.error
    signal = result.signal.signals["INTRADAY"]
    assert signal.model_predicted_class == "UP"
    assert signal.raw_confidence == 0.65
    assert signal.action == ("HOLD" if invalid else "BUY")
    assert signal.suppressed == invalid
    if invalid:
        assert signal.calibrated_confidence is None
        assert any("confidence is not yet calibrated" in reason for reason in signal.suppression_reasons)
    else:
        assert signal.calibrated_confidence == pytest.approx(0.65)


@pytest.fixture
def isolated_api(prices, isolated_io, monkeypatch):
    """Load actual HTTP routes with isolated construction and metrics storage."""
    import prometheus_client

    scheduler = make_scheduler(prices, isolated_io)
    metric_registry = prometheus_client.CollectorRegistry()
    spec = importlib.util.spec_from_file_location("f07_consumer_api", Path(__file__).resolve().parents[2] / "api.py")
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patches:
        # The API constructs its dependencies at import. Inject the real,
        # temporary-backed instances rather than constructing default stores.
        patches.setattr(scheduler_module, "Scheduler", lambda: scheduler)
        patches.setattr(history_manager, "HistoryManager", lambda: isolated_io)
        for name in ["Counter", "Gauge", "Histogram"]:
            patches.setattr(
                prometheus_client, name, partial(getattr(prometheus_client, name), registry=metric_registry)
            )
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("invalid", [False, True], ids=["valid-control", "invalid-evidence"])
def test_http_signal_serializes_actual_scheduler_calibration(isolated_api, prices, invalid):
    from fastapi.testclient import TestClient

    from scheduler import LiveWorthinessSnapshot

    validator = isolated_api.scheduler.predictor.runtime_validator
    calibration = validator.compute_calibration(
        pd.DataFrame({"confidence": [np.nan if invalid else 0.65] * 60, "correct": [1] * 39 + [0] * 21})
    )
    edge = validator.compute_edge_vs_baseline(
        pd.Series([0.5] * 60), pd.Series([0.0] * 60), slippage_bps=0, transaction_cost_bps=0
    )
    assert edge.status == "EDGE_CONFIRMED"
    isolated_api.scheduler._live_worthiness_cache[("RELIANCE", "INTRADAY")] = LiveWorthinessSnapshot(
        edge, calibration, datetime.now()
    )
    # TestClient handles HTTP in process; no server or external socket is used.
    response = TestClient(isolated_api.app).get("/api/v1/signal/RELIANCE")
    assert response.status_code == 200, response.text
    signal = response.json()["signals"]["INTRADAY"]
    assert signal["action"] == ("HOLD" if invalid else "BUY")
    assert signal["raw_confidence"] == 0.65
    assert signal["calibration_status"] == ("UNAVAILABLE" if invalid else "VALID")
    if invalid:
        assert signal["confidence"] is None
        assert signal["calibrated_confidence"] is None
        assert "Holding back" in signal["verdict_text"]
    else:
        assert signal["confidence"] == pytest.approx(0.65)
        assert signal["calibrated_confidence"] == pytest.approx(0.65)


def test_scanner_and_scalping_keep_actual_predictor_no_evidence_hold(prices, monkeypatch):
    import scalping
    import scanner

    predictor = make_predictor(prices)
    monkeypatch.setattr(scanner, "NIFTY50_SYMBOLS", ["RELIANCE"])
    monkeypatch.setattr(scalping, "NIFTY50_SYMBOLS", ["RELIANCE"])
    # These consumers do not pass evidence. Exercise that real limitation,
    # without manufacturing forwarding behavior in a mocked Predictor.
    summary = OpportunityScanner(predictor, predictor.data_fetcher).scan_with_summary()
    assert summary.scanned == 1
    assert summary.hold == 1
    assert summary.actionable == 0
    assert summary.data_error == 0
    assert summary.opportunities == []
    assert ScalpingEngine(predictor, predictor.data_fetcher, predictor.feature_engineer).find_opportunities() == []
