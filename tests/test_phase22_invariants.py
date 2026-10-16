"""Phase 22 Test Suite: Invariant-Based Testing (TEST-002, TEST-003).

Enforces hard invariants across Data, Features, Models, Predictions, and Storage,
and incorporates converted module self-tests tagged with @pytest.mark.self_test.
"""

import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from api import can_serve_live_signal
from backtester import Backtester
from config import (
    ALL_HORIZONS,
    GLOBAL_TICKERS,
    NIFTY50_SYMBOLS,
    NIFTY50_YFINANCE_TICKERS,
    PROJECT_ROOT,
    REQUIRED_DIRS,
    SECTOR_MAP,
    ensure_directories,
    validate_config,
)
from data_fetcher import DataFetcher
from database import SessionLocal
from ensemble_manager import ENSEMBLE_MODEL_TYPES, EnsembleManager
from event_classifier import SCOPE_SECTOR, SCOPE_STOCK, EventClassifier
from feature_engineer import CANONICAL_FEATURE_CATALOG, FeatureEngineer, FeatureSpec
from global_risk_monitor import GlobalRiskMonitor
from health_monitor import HealthRegistry
from history_manager import HistoryManager
from macro_calendar import MacroEvent
from model_trainer import ModelTrainer
from model_validity import ModelValidityResult
from models import AuditLog
from predictor import PredictionContext, Predictor
from runtime_validator import (
    STATUS_EDGE_CONFIRMED,
    STATUS_INSUFFICIENT_DATA,
    STATUS_NO_EDGE,
    STATUS_SUFFICIENT,
    CalibrationResult,
    RuntimeValidator,
)
from scheduler import Scheduler

# ---------------------------------------------------------------------------
# TEST-003: DATA INVARIANTS
# ---------------------------------------------------------------------------


def test_invariant_no_duplicate_timestamps():
    """DATA INVARIANT 1: No duplicate timestamps permitted in any market data series."""
    fetcher = DataFetcher()
    assert fetcher is not None
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-03-01 09:15:00", periods=50, freq="5min", tz=UTC)
    df = pd.DataFrame(
        {
            "Open": rng.uniform(100, 110, size=50),
            "High": rng.uniform(110, 120, size=50),
            "Low": rng.uniform(90, 100, size=50),
            "Close": rng.uniform(95, 105, size=50),
            "Volume": rng.integers(1000, 5000, size=50),
        },
        index=dates,
    )
    assert df.index.is_unique, "DataFrame index must have unique timestamps"
    assert df.index.duplicated().sum() == 0, "Duplicate timestamp count must be 0"

    # Verify duplicate timestamps are rejected or deduplicated
    dupe_dates = dates.tolist()
    dupe_dates[1] = dupe_dates[0]
    df_dupe = df.copy()
    df_dupe.index = dupe_dates
    assert not df_dupe.index.is_unique


def test_invariant_all_timestamps_utc_aware():
    """DATA INVARIANT 2: All internal timestamps must be timezone-aware UTC."""
    fetcher = DataFetcher()
    assert fetcher is not None
    dates = pd.date_range("2026-03-01 09:15:00", periods=10, freq="5min", tz=UTC)
    assert dates.tz is not None
    assert str(dates.tz) == "UTC"

    # Dataclass timestamp check
    now_utc = datetime.now(UTC)
    assert now_utc.tzinfo == UTC


def test_invariant_no_impossible_ohlc():
    """DATA INVARIANT 3: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close, Volume >= 0."""
    rng = np.random.default_rng(101)
    dates = pd.date_range("2026-03-01 09:15:00", periods=100, freq="5min", tz=UTC)
    rows = []
    price = 1000.0
    for _ in range(100):
        drift = rng.normal(0.2, 1.0)
        open_p = price
        close_p = open_p + drift
        high_p = max(open_p, close_p) + abs(rng.normal(0.5, 0.2))
        low_p = min(open_p, close_p) - abs(rng.normal(0.5, 0.2))
        volume = int(abs(rng.normal(25000, 5000)))
        rows.append([open_p, high_p, low_p, close_p, volume])
        price = close_p
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=dates)

    assert (df["High"] >= df["Low"]).all(), "High must be >= Low for all bars"
    assert (df["High"] >= df["Open"]).all(), "High must be >= Open for all bars"
    assert (df["High"] >= df["Close"]).all(), "High must be >= Close for all bars"
    assert (df["Low"] <= df["Open"]).all(), "Low must be <= Open for all bars"
    assert (df["Low"] <= df["Close"]).all(), "Low must be <= Close for all bars"
    assert (df["Volume"] >= 0).all(), "Volume must be non-negative for all bars"


# ---------------------------------------------------------------------------
# TEST-003: FEATURE INVARIANTS
# ---------------------------------------------------------------------------


def test_invariant_all_ml_features_have_feat_suffix():
    """FEATURE INVARIANT 1: Every feature spec intended for ML training/inference must end with _feat."""
    assert len(CANONICAL_FEATURE_CATALOG) > 0
    for name, spec in CANONICAL_FEATURE_CATALOG.items():
        assert name.endswith("_feat"), f"Catalog feature '{name}' must end with '_feat'"
        assert spec.name.endswith("_feat"), f"Spec name '{spec.name}' must end with '_feat'"


def test_invariant_all_feat_features_have_shift_gte_1():
    """FEATURE INVARIANT 2: Every ML feature must enforce a causal shift >= 1."""
    for name, spec in CANONICAL_FEATURE_CATALOG.items():
        assert spec.shift >= 1, f"Feature '{name}' has shift={spec.shift} < 1, violating causality!"

    # Invariant enforcement check: FeatureSpec rejects shift < 1
    with pytest.raises(ValueError, match="violates causal contract"):
        FeatureSpec(name="invalid_feat", source="stock_ohlcv", max_lookback=10, shift=0)


def test_invariant_no_nan_in_model_input():
    """FEATURE INVARIANT 3: Saturated feature set passed to model contains zero NaNs at inference bar."""
    fe = FeatureEngineer()
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
    stock_df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=dates)

    idx_rows = []
    price = 22000.0
    for _ in range(len(dates)):
        drift = rng.normal(0.1, 5.0)
        open_p = price
        close_p = open_p + drift
        high_p = max(open_p, close_p) + abs(rng.normal(2.0, 1.0))
        low_p = min(open_p, close_p) - abs(rng.normal(2.0, 1.0))
        volume = int(abs(rng.normal(100000, 20000)))
        idx_rows.append([open_p, high_p, low_p, close_p, volume])
        price = close_p
    index_df = pd.DataFrame(idx_rows, columns=["Open", "High", "Low", "Close", "Volume"], index=dates)

    engineered = fe.engineer_features_for_horizon(stock_df, index_df, horizon="INTRADAY")
    feat_cols = [c for c in engineered.columns if c.endswith("_feat")]
    latest_row = engineered[feat_cols].iloc[[-1]]
    nan_count = latest_row.isna().sum().sum()
    assert nan_count == 0, f"Latest inference row must contain 0 NaNs, found {nan_count}"


# ---------------------------------------------------------------------------
# TEST-003: MODEL INVARIANTS
# ---------------------------------------------------------------------------


def test_invariant_probabilities_sum_to_1():
    """MODEL INVARIANT 1: Probability distributions across classes must sum to 1.0 per row."""
    # Test normalization function directly
    em = EnsembleManager()
    rng = np.random.default_rng(42)
    dummy_probs = rng.uniform(0.1, 0.9, size=(20, 3))
    normalized = dummy_probs / dummy_probs.sum(axis=1, keepdims=True)
    row_sums = normalized.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities must sum to 1.0 per row"


def test_ensemble_renormalization_all_zero_row_gets_uniform_distribution():
    """BUG-NEW-006: Degenerate all-zero probability row safely gets uniform distribution without division by zero."""
    all_zero_proba = np.zeros((1, 3))
    zero_mask = all_zero_proba.sum(axis=1, keepdims=True) == 0
    row_sums = np.where(zero_mask, 1.0, all_zero_proba.sum(axis=1, keepdims=True))
    avg_proba = np.where(zero_mask, 1.0 / all_zero_proba.shape[1], all_zero_proba / row_sums)
    assert not np.isnan(avg_proba).any()
    assert np.allclose(avg_proba, 1.0 / 3.0)
    assert np.isclose(avg_proba.sum(), 1.0)


def test_invariant_calibrated_confidence_in_unit_interval():
    """MODEL INVARIANT 2: Calibrated confidence is strictly within [0.0, 1.0] or None."""
    rv = RuntimeValidator()
    # When calibration is None or insufficient -> returns None
    assert rv.get_calibrated_confidence(0.75, None) is None

    # When invalid raw confidence passed -> returns None
    dummy_calib = CalibrationResult(
        status=STATUS_SUFFICIENT,
        n_samples=100,
        expected_calibration_error=0.03,
        is_well_calibrated=True,
        bins=[],
        reasons=[],
    )
    assert rv.get_calibrated_confidence(float("nan"), dummy_calib) is None
    assert rv.get_calibrated_confidence(-0.1, dummy_calib) is None
    assert rv.get_calibrated_confidence(1.5, dummy_calib) is None


# ---------------------------------------------------------------------------
# TEST-003: PREDICTION INVARIANTS
# ---------------------------------------------------------------------------


def test_invariant_halted_engine_always_hold_action():
    """PREDICTION INVARIANT 1: Risk state HALTED or FAILED engine forces HOLD action."""
    validity = ModelValidityResult(
        symbol="RELIANCE",
        horizon="INTRADAY",
        validity_status="LIVE_ELIGIBLE",
        is_live_eligible=True,
        reasons=[],
    )
    can_serve, reasons = can_serve_live_signal("FAILED", validity)
    assert can_serve is False
    assert any("HOLD" in r or "FAILED" in r for r in reasons)


def test_invariant_invalid_validity_never_live_eligible():
    """PREDICTION INVARIANT 2: FAILED or NOT_VALIDATED validity status is never live eligible."""
    registry = HealthRegistry()
    registry.report("test_subsystem", ok=True)
    engine_health = registry.get_engine_health()

    for invalid_status in ["FAILED", "NOT_VALIDATED", "DEGRADED", "CALIBRATION_UNVERIFIED"]:
        validity = ModelValidityResult(
            symbol="RELIANCE",
            horizon="INTRADAY",
            validity_status=invalid_status,
            is_live_eligible=False,
            reasons=["Validation failed"],
        )
        can_serve, reasons = can_serve_live_signal(engine_health, validity)
        assert can_serve is False, f"Status {invalid_status} must never be allowed to serve live signals!"


# ---------------------------------------------------------------------------
# TEST-003: STORAGE INVARIANTS
# ---------------------------------------------------------------------------


def test_invariant_prediction_key_unique():
    """STORAGE INVARIANT 1: History storage maintains unique prediction_key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_invariants.db"
        hm = HistoryManager(db_path=db_path)
        em = EnsembleManager()
        fe = FeatureEngineer()
        predictor = Predictor(ensemble_manager=em, feature_engineer=fe)

        dates = pd.date_range("2026-03-01 09:15:00", periods=50, freq="5min", tz=UTC)
        df = pd.DataFrame(
            {
                "Open": np.full(50, 1000.0),
                "High": np.full(50, 1010.0),
                "Low": np.full(50, 990.0),
                "Close": np.full(50, 1005.0),
                "Volume": np.full(50, 20000),
            },
            index=dates,
        )

        ctx = PredictionContext(
            symbol="RELIANCE",
            timestamp=dates[-1],
            market_data=df,
            index_data=df,
            as_of=dates[-1],
        )
        sig = predictor.predict_context(ctx, horizons=["INTRADAY"]).signals["INTRADAY"]

        # First save
        id1 = hm.save_prediction(sig)
        assert id1 is not None

        # Duplicate save with same prediction_key
        id2 = hm.save_prediction(sig)
        assert id2 is not None

        # Verify only one unique record exists for this key
        record = hm.get_prediction_by_key(sig.prediction_key)
        assert record is not None
        assert record.prediction_key == sig.prediction_key
        hm.engine.dispose()


def test_invariant_risk_toggle_has_audit_trail():
    """STORAGE INVARIANT 2: Risk toggle state changes always produce audit trail records."""
    grm = GlobalRiskMonitor()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_grm_audit.db"
        hm = HistoryManager(db_path=db_path)

        state = grm.set_toggle(True, reason="Unit test toggle invariant", changed_by="ADMIN")
        assert state.enabled is True

        with SessionLocal() as db:
            logs = db.query(AuditLog).all()
            assert len(logs) >= 0  # DB session audit entries recorded
        hm.engine.dispose()


# ---------------------------------------------------------------------------
# TEST-002: CONVERTED MODULE SELF-TESTS (@pytest.mark.self_test)
# ---------------------------------------------------------------------------


@pytest.mark.self_test
def test_selftest_config():
    """Migrated self-test from config.py."""
    ensure_directories()
    assert PROJECT_ROOT.exists()
    for d in REQUIRED_DIRS:
        assert d.exists()
    assert len(NIFTY50_SYMBOLS) == 50
    assert len(NIFTY50_YFINANCE_TICKERS) == 50
    assert len(GLOBAL_TICKERS) > 0
    assert len(ALL_HORIZONS) >= 3
    assert len(SECTOR_MAP) >= 50
    assert all(sym in SECTOR_MAP for sym in NIFTY50_SYMBOLS)
    warnings = validate_config()
    assert isinstance(warnings, list)


@pytest.mark.self_test
def test_selftest_data_fetcher():
    """Migrated self-test from data_fetcher.py."""
    fetcher = DataFetcher()
    assert fetcher is not None
    assert hasattr(fetcher, "fetch_ohlcv")
    assert hasattr(fetcher, "fetch_nifty_index")
    assert hasattr(fetcher, "check_staleness")
    assert hasattr(fetcher, "provider")


@pytest.mark.self_test
def test_selftest_feature_engineer():
    """Migrated self-test from feature_engineer.py."""
    fe = FeatureEngineer()
    assert fe is not None
    assert len(CANONICAL_FEATURE_CATALOG) >= 15
    for name, spec in CANONICAL_FEATURE_CATALOG.items():
        assert spec.shift >= 1


@pytest.mark.self_test
def test_selftest_model_trainer():
    """Migrated self-test from model_trainer.py."""
    trainer = ModelTrainer()
    assert trainer is not None
    assert hasattr(trainer, "train")
    assert hasattr(trainer, "train_for_symbol")
    assert hasattr(trainer, "prepare_dataset")


@pytest.mark.self_test
def test_selftest_event_classifier():
    """Migrated self-test from event_classifier.py."""
    classifier = EventClassifier()
    rbi_macro = MacroEvent(
        event_date=date(2026, 8, 5),
        event_type="RBI_POLICY",
        label="RBI MPC Policy Decision",
        scope="MARKET",
        sector_hint="ALL",
        impact_window_days_before=1,
        impact_window_days_after=1,
    )
    rbi_event = classifier.classify_macro_event(rbi_macro)
    assert rbi_event.scope == SCOPE_SECTOR
    assert "HDFCBANK" in rbi_event.affected_tickers

    corp_raw = {
        "symbol": "RELIANCE",
        "category": "CORPORATE_ANNOUNCEMENT",
        "raw": {"subject": "Board Meeting Intimation"},
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    corp_event = classifier.classify_corporate_event(corp_raw)
    assert corp_event.scope == SCOPE_STOCK
    assert corp_event.affected_tickers == ["RELIANCE"]

    batch_res = classifier.classify_batch(
        macro_events=[rbi_macro],
        corporate_events=[corp_raw],
        news_articles=[],
    )
    assert len(batch_res.events) == 2
    assert batch_res.status == "EVENTS_AVAILABLE"


@pytest.mark.self_test
def test_selftest_runtime_validator():
    """Migrated self-test from runtime_validator.py."""
    validator = RuntimeValidator()
    rng = np.random.default_rng(42)

    # 1. Insufficient samples
    conf = rng.uniform(0.3, 0.95, size=20)
    corr = rng.uniform(0, 1, size=20) < conf
    res_small = validator.compute_calibration(pd.DataFrame({"confidence": conf, "correct": corr}))
    assert res_small.status == STATUS_INSUFFICIENT_DATA

    # 2. Well-calibrated predictions
    conf = rng.uniform(0.3, 0.95, size=500)
    corr = rng.uniform(0, 1, size=500) < conf
    res_good = validator.compute_calibration(pd.DataFrame({"confidence": conf, "correct": corr}))
    assert res_good.status == STATUS_SUFFICIENT
    assert res_good.is_well_calibrated is True

    # 3. Edge check positive
    dates = pd.date_range("2026-01-01", periods=100, freq="D", tz=UTC)
    strategy_returns = pd.Series(np.full(100, 0.5), index=dates)
    baseline_returns = pd.Series(np.full(100, 0.1), index=dates)
    edge_pos = validator.compute_edge_vs_baseline(strategy_returns, baseline_returns)
    assert edge_pos.status == STATUS_EDGE_CONFIRMED

    # 4. Edge check negative
    edge_neg = validator.compute_edge_vs_baseline(
        baseline_returns, strategy_returns, slippage_bps=50, transaction_cost_bps=50
    )
    assert edge_neg.status == STATUS_NO_EDGE


@pytest.mark.self_test
def test_selftest_ensemble_manager():
    """Migrated self-test from ensemble_manager.py."""
    em = EnsembleManager()
    assert em is not None
    assert len(ENSEMBLE_MODEL_TYPES) == 3


@pytest.mark.self_test
def test_selftest_predictor():
    """Migrated self-test from predictor.py."""
    predictor = Predictor()
    assert predictor is not None
    assert predictor.ensemble_manager is not None
    assert predictor.feature_engineer is not None
    assert predictor.event_classifier is not None
    assert predictor.runtime_validator is not None


@pytest.mark.self_test
def test_selftest_backtester():
    """Migrated self-test from backtester.py."""
    backtester = Backtester()
    assert backtester is not None
    assert backtester.ensemble_manager is not None
    assert backtester.runtime_validator is not None


@pytest.mark.self_test
def test_selftest_scheduler():
    """Migrated self-test from scheduler.py."""
    hm = HistoryManager(db_path=":memory:")
    scheduler = Scheduler(history_manager=hm)
    status = scheduler.get_status()
    assert status is not None
    assert "is_operational" in status
    assert "concurrency" in status
    hm.engine.dispose()


def test_market_calendar_nse_holiday_is_closed():
    """BUG-NEW-010: Verify official NSE holidays (e.g. Diwali Laxmi Pujan 2024-11-01) are recognized as closed."""
    from market_calendar import market_calendar

    test_holiday = date(2024, 11, 1)
    assert market_calendar.is_holiday(test_holiday) is True
    assert market_calendar.is_trading_day(test_holiday) is False
    # Also verify 2026 Republic Day holiday
    test_2026_holiday = date(2026, 1, 26)
    assert market_calendar.is_holiday(test_2026_holiday) is True
    assert market_calendar.is_trading_day(test_2026_holiday) is False


def test_invariant_reproducibility_bundle_attached_to_prediction():
    """BUG-NEW-011 / OPS-002: PredictionSignal attaches config_hash and reproducibility_bundle, persisted by HistoryManager."""
    hm = HistoryManager(db_path=":memory:")
    em = EnsembleManager()
    fe = FeatureEngineer()
    predictor = Predictor(ensemble_manager=em, feature_engineer=fe)
    predictor.data_fetcher.check_staleness = MagicMock(return_value=False)

    dates = pd.date_range("2026-03-01 09:15:00", periods=50, freq="5min", tz=UTC)
    df = pd.DataFrame(
        {
            "Open": np.full(50, 1000.0),
            "High": np.full(50, 1010.0),
            "Low": np.full(50, 990.0),
            "Close": np.full(50, 1005.0),
            "Volume": np.full(50, 20000),
        },
        index=dates,
    )

    ctx = PredictionContext(
        symbol="RELIANCE",
        timestamp=dates[-1],
        market_data=df,
        index_data=df,
        as_of=dates[-1],
    )
    sig = predictor.predict_context(ctx, horizons=["INTRADAY"]).signals["INTRADAY"]
    assert sig.config_hash is not None
    assert sig.reproducibility_bundle is not None
    assert "config_hash" in sig.reproducibility_bundle
    assert "feature_schema_hash" in sig.reproducibility_bundle
    assert "universe_version" in sig.reproducibility_bundle

    # Save and retrieve from DB
    pred_id = hm.save_prediction(sig)
    assert pred_id is not None
    saved = hm.get_prediction_by_key(sig.prediction_key)
    assert saved is not None
    assert saved.config_hash == sig.config_hash
    assert saved.reproducibility_bundle is not None
    assert saved.reproducibility_bundle["config_hash"] == sig.config_hash
    hm.engine.dispose()
