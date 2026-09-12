from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from config import HORIZON_3D, HORIZON_INTRADAY
from data_fetcher import DataFetcher, DataStatus, MarketDataResult
from event_classifier import EventBatchResult, EventClassifier

# Imports from codebase
from feature_engineer import FeatureEngineer
from history_manager import HistoryManager
from market_calendar import MarketCalendar
from predictor import ACTION_BUY, ACTION_HOLD, PredictionSignal
from runtime_validator import CalibrationBin, CalibrationResult, RuntimeValidator
from scanner import OpportunityScanner, ScanSummary


# ============================================================================
# 1. ORB Lookahead Leakage (P0 3.2 & Invariant A)
# ============================================================================
def test_orb_lookahead_leakage_and_future_mutation_invariance():
    fe = FeatureEngineer()
    # Create 5-minute bars for a single trading day (09:15 to 11:00)
    timestamps = pd.date_range("2026-07-14 09:15", "2026-07-14 11:00", freq="5min")
    n_bars = len(timestamps)

    df1 = pd.DataFrame(
        {
            "Open": [100.0] * n_bars,
            "High": [105.0] * n_bars,
            "Low": [95.0] * n_bars,
            "Close": [100.0] * n_bars,
            "Volume": [1000] * n_bars,
        },
        index=timestamps,
    )

    # 3 bars opening range: 09:15, 09:20, 09:25 (orb_bar_count=3 by default)
    or_high1, or_low1, breakout1 = fe.compute_opening_range_breakout(df1)

    # Bars within the opening range (first 3 bars) MUST have NaN for ORB levels
    assert pd.isna(or_high1.iloc[0]), "Opening bar 0 must not know ORB levels"
    assert pd.isna(or_high1.iloc[1]), "Opening bar 1 must not know ORB levels"
    assert pd.isna(or_high1.iloc[2]), "Opening bar 2 must not know ORB levels"
    # Starting at bar 3 (09:30), the ORB high should be known
    assert not pd.isna(or_high1.iloc[3]), "Post-ORB bar 3 must have ORB high"

    # Future-mutation invariance: Mutate bar 2's High (an extreme spike at 09:25)
    df2 = df1.copy()
    df2.loc[timestamps[2], "High"] = 200.0
    or_high2, or_low2, breakout2 = fe.compute_opening_range_breakout(df2)

    # Invariant: Mutating bar 2's price must NOT alter features at bar 0 or bar 1
    assert pd.isna(or_high2.iloc[0])
    assert pd.isna(or_high2.iloc[1])
    assert breakout2.iloc[0] == 0
    assert breakout2.iloc[1] == 0


# ============================================================================
# 2. Calibration Bin Indexing (P0 3.4)
# ============================================================================
def test_calibration_bin_explicit_boundary_lookup():
    rv = RuntimeValidator()
    # Construct calibration bins with an explicit gap: [0.0, 0.4] and [0.6, 1.0] (no [0.4, 0.6])
    bins = [
        CalibrationBin(
            lower_bound=0.0, upper_bound=0.4, count=100, mean_predicted_confidence=0.2, empirical_accuracy=0.35
        ),
        CalibrationBin(
            lower_bound=0.6, upper_bound=1.0, count=100, mean_predicted_confidence=0.8, empirical_accuracy=0.85
        ),
    ]
    cal_result = CalibrationResult(
        status="SUFFICIENT", n_samples=200, expected_calibration_error=0.05, is_well_calibrated=True, bins=bins
    )

    # Raw confidence in first bin
    c1 = rv.get_calibrated_confidence(0.2, cal_result)
    assert c1 == 0.35, f"Expected 0.35, got {c1}"

    # Raw confidence in gap: [0.4, 0.6] has no bin
    c_gap = rv.get_calibrated_confidence(0.5, cal_result)
    assert c_gap is None, f"Expected None for confidence falling into gap, got {c_gap}"

    # Raw confidence in second bin
    c2 = rv.get_calibrated_confidence(0.8, cal_result)
    assert c2 == 0.85, f"Expected 0.85, got {c2}"

    # Exact boundaries
    assert rv.get_calibrated_confidence(0.0, cal_result) == 0.35
    assert rv.get_calibrated_confidence(1.0, cal_result) == 0.85


# ============================================================================
# 3. Calibration Horizon & Model Version Isolation (P0 3.3 & Invariant D)
# ============================================================================
def test_calibration_isolation_by_horizon_and_model_version(tmp_path):
    db_path = tmp_path / "test_iso.sqlite3"
    hm = HistoryManager(db_path=db_path)

    base_sig = dict(
        symbol="TCS",
        timestamp=datetime.now(),
        action="BUY",
        model_predicted_class="UP",
        raw_confidence=0.8,
        risk_adjusted_confidence=0.8,
        calibrated_confidence=0.75,
        agreement_fraction=0.7,
        downside_summary="d",
        upside_summary="u",
        reasoning=[],
    )

    # 1. INTRADAY with model v1
    s1 = PredictionSignal(horizon=HORIZON_INTRADAY, model_version="v1.0", feature_version="v1", **base_sig)
    id1 = hm.save_prediction(s1)
    hm.resolve_outcome(id1, actual_class="UP")

    # 2. 3D with model v1
    s2 = PredictionSignal(horizon=HORIZON_3D, model_version="v1.0", feature_version="v1", **base_sig)
    id2 = hm.save_prediction(s2)
    hm.resolve_outcome(id2, actual_class="UP")

    # 3. INTRADAY with model v2
    s3 = PredictionSignal(horizon=HORIZON_INTRADAY, model_version="v2.0", feature_version="v2", **base_sig)
    id3 = hm.save_prediction(s3)
    hm.resolve_outcome(id3, actual_class="DOWN")

    # 4. Legacy/Unknown
    s4 = PredictionSignal(horizon=HORIZON_INTRADAY, model_version="UNKNOWN", feature_version="UNKNOWN", **base_sig)
    id4 = hm.save_prediction(s4)
    hm.resolve_outcome(id4, actual_class="UP")

    # Query for INTRADAY + v1.0
    cal_df_v1 = hm.build_calibration_dataset("TCS", horizon=HORIZON_INTRADAY, model_version="v1.0")
    assert len(cal_df_v1) == 1, "Must only match INTRADAY and v1.0"
    assert cal_df_v1["correct"].iloc[0] == 1

    # Query for 3D + v1.0
    cal_df_3d = hm.build_calibration_dataset("TCS", horizon=HORIZON_3D, model_version="v1.0")
    assert len(cal_df_3d) == 1, "Must only match 3D and v1.0"

    # Query for INTRADAY + v2.0
    cal_df_v2 = hm.build_calibration_dataset("TCS", horizon=HORIZON_INTRADAY, model_version="v2.0")
    assert len(cal_df_v2) == 1
    assert cal_df_v2["correct"].iloc[0] == 0

    # Ensure UNKNOWN rows are excluded
    cal_df_unk = hm.build_calibration_dataset("TCS", horizon=HORIZON_INTRADAY, model_version="UNKNOWN")
    assert len(cal_df_unk) == 0, "UNKNOWN/LEGACY rows must be excluded from calibration"


# ============================================================================
# 4. Data Freshness State & Provenance (P0 3.5 & Invariant B)
# ============================================================================
def test_data_freshness_explicit_state():
    fetcher = DataFetcher()
    result = fetcher.fetch_ohlcv("RELIANCE.NS", return_metadata=True)

    assert isinstance(result, MarketDataResult)
    assert result.status in [DataStatus.LIVE, DataStatus.CACHED_FRESH, DataStatus.CACHED_STALE, DataStatus.UNAVAILABLE]
    assert result.source.upper() in ["LIVE", "CACHE", "UNAVAILABLE"]
    if result.status == DataStatus.CACHED_STALE:
        assert result.is_stale is True


# ============================================================================
# 5. Market Calendar Authoritative Implementation (P0 3.6)
# ============================================================================
def test_market_calendar_unification():
    cal = MarketCalendar()
    tz = ZoneInfo("Asia/Kolkata")

    # Trading day
    tuesday = datetime(2026, 7, 14, 11, 0, tzinfo=tz)
    assert cal.is_market_open(tuesday) is True
    assert cal.is_trading_day(tuesday.date()) is True

    # Weekend
    saturday = datetime(2026, 7, 18, 11, 0, tzinfo=tz)
    assert cal.is_market_open(saturday) is False
    assert cal.is_trading_day(saturday.date()) is False

    # Exchange holiday
    holiday = datetime(2026, 1, 26, 11, 0, tzinfo=tz)
    assert cal.is_market_open(holiday) is False
    assert cal.is_trading_day(holiday.date()) is False

    # Scheduler and DataFetcher agree
    from scheduler import Scheduler

    assert Scheduler.is_market_open(tuesday) == cal.is_market_open(tuesday)
    assert Scheduler.is_market_open(holiday) == cal.is_market_open(holiday)


# ============================================================================
# 6. Event Feed Failure != No Events (P0 3.10 & Invariant E)
# ============================================================================
def test_event_classifier_fail_closed_on_source_error():
    classifier = EventClassifier()

    # Pass None for all sources to simulate provider failure
    batch = classifier.classify_batch(macro_events=None, corporate_events=None, news_articles=None)

    assert isinstance(batch, EventBatchResult)
    assert batch.status == "EVENT_SOURCE_UNAVAILABLE"
    assert len(batch.errors) > 0, "Source errors must be recorded"
    assert len(batch.events) == 0


# ============================================================================
# 7. Scanner Rejection Observability (P0 3.9 & Invariant F)
# ============================================================================
def test_scanner_rejection_summary():
    class MockFetcher:
        def fetch_ohlcv(self, ticker, **kwargs):
            return pd.DataFrame({"Close": [100.0]})

        def check_staleness(self, df, ticker):
            return False

    class MockPred:
        def generate_multi_horizon_signal(self, symbol, **kwargs):
            from predictor import MultiHorizonSignal

            action = ACTION_HOLD if symbol == "INFY" else ACTION_BUY
            sig = PredictionSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                horizon=HORIZON_INTRADAY,
                action=action,
                model_predicted_class="UP" if action == ACTION_BUY else "FLAT",
                model_version="v1",
                feature_version="v1",
                raw_confidence=0.8,
                risk_adjusted_confidence=0.8,
                calibrated_confidence=0.75,
                agreement_fraction=0.8,
                downside_summary="",
                upside_summary="",
                reasoning=[],
                is_safe_to_trade_live=(action == ACTION_BUY),
            )
            return MultiHorizonSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signals={HORIZON_INTRADAY: sig},
                primary_action=action,
                primary_horizon=HORIZON_INTRADAY,
                reasoning=[],
            )

    scanner = OpportunityScanner(predictor=MockPred(), data_fetcher=MockFetcher())
    summary = scanner.scan_with_summary(limit=3)

    assert isinstance(summary, ScanSummary)
    assert summary.scanned > 0
    assert summary.hold > 0
    # INFY must be recorded as rejected under 'hold'
    infy_rej = [r for r in summary.rejections if r.symbol == "INFY"]
    assert len(infy_rej) == 1
    assert infy_rej[0].category == "hold"


# ============================================================================
# 8. API Refresh Protection (P0 3.8 & Invariant G)
# ============================================================================
def test_api_refresh_rate_limiting():
    from api import refresh_backtest

    # First refresh succeeds or attempts fetch
    res1 = refresh_backtest("RELIANCE")

    # Immediate second refresh must trigger cooldown
    res2 = refresh_backtest("RELIANCE")
    assert res2.get("status") == "rejected"
    assert "cooldown active" in res2.get("reason", "").lower()


# ============================================================================
# 9. Risk-Based Position Sizing (P1 4.7)
# ============================================================================
def test_position_planner_risk_sizing():
    from position_planner import PositionPlanner
    from predictor import MultiHorizonSignal

    planner = PositionPlanner(portfolio_capital=10_00_000, max_position_pct=0.10)
    sig = MultiHorizonSignal(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        signals={},
        primary_action=ACTION_BUY,
        primary_horizon="1D",
        reasoning=[],
    )
    plan = planner.generate_plan(multi_signal=sig, current_price=2500.0, ma_level=2400.0, support_level=2300.0)

    assert plan.action == ACTION_BUY
    assert plan.planned_capital == 1_00_000.0
    assert plan.stop_price is not None and plan.stop_price < 2300.0
    assert plan.max_loss > 0.0
    assert plan.risk_pct > 0.0 and plan.risk_pct <= 2.0


# ============================================================================
# 10. Structured Scheduler Results (P1 4.10)
# ============================================================================
def test_scheduler_cycle_result():
    from scheduler import CycleResult, Scheduler

    sched = Scheduler()
    # When data is empty, returns structured failure
    res = sched.run_one_cycle_for_symbol("TEST", pd.DataFrame(), pd.DataFrame(), return_structured=True)
    assert isinstance(res, CycleResult)
    assert res.success is False
    assert res.status == "DATA_UNAVAILABLE"


# ============================================================================
# 11. Walk-Forward Split (P1 4.1)
# ============================================================================
def test_model_trainer_walk_forward_split():
    from model_trainer import ModelTrainer

    # 500 samples
    dates = pd.date_range("2026-01-01", periods=500, freq="1D")
    X = pd.DataFrame({"f1": range(500)}, index=dates)
    y = pd.Series(["UP"] * 500, index=dates)

    splits = list(ModelTrainer.walk_forward_split(X, y, n_splits=3, min_train_samples=200))
    assert len(splits) == 3
    for fold, X_tr, X_te, y_tr, y_te in splits:
        assert len(X_tr) >= 200
        assert len(X_te) > 0
        # Chronological order invariant: max train timestamp < min test timestamp
        assert X_tr.index.max() < X_te.index.min()


# ============================================================================
# 12. API Authentication (P0-001)
# ============================================================================
def test_api_requires_authentication():
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)
    # Unauthenticated mutating request must return 401
    r_no_auth = client.post("/api/risk/toggle?enabled=true")
    assert r_no_auth.status_code == 401, f"Expected 401, got {r_no_auth.status_code}"

    # Invalid key must return 401
    r_bad_key = client.post("/api/risk/toggle?enabled=true", headers={"X-API-Key": "invalid-key-xyz"})
    assert r_bad_key.status_code == 401, f"Expected 401, got {r_bad_key.status_code}"


# ============================================================================
# 13. API Role-Based Authorization (P0-001)
# ============================================================================
def test_api_rbac_permissions():
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)
    # Read-only key trying to perform mutating admin action must return 403
    r_readonly = client.post("/api/risk/toggle?enabled=true", headers={"X-API-Key": "dev-alkame-readonly-key"})
    assert r_readonly.status_code == 403, f"Expected 403, got {r_readonly.status_code}"

    # Admin key must succeed
    r_admin = client.post("/api/risk/toggle?enabled=true", headers={"X-API-Key": "dev-alkame-admin-key"})
    assert r_admin.status_code == 200, f"Expected 200, got {r_admin.status_code}"
    assert r_admin.json().get("status") == "success"


# ============================================================================
# 14. Strict CORS Configuration (P0-002)
# ============================================================================
def test_api_cors_no_wildcard_credentials():
    from config import CORS_ALLOW_CREDENTIALS, CORS_ALLOWED_ORIGINS

    if CORS_ALLOW_CREDENTIALS:
        assert "*" not in CORS_ALLOWED_ORIGINS, "Wildcard '*' origin is prohibited when CORS credentials are enabled."
    assert len(CORS_ALLOWED_ORIGINS) > 0


# ============================================================================
# 15. Purged Train/Test Boundary (P0-003)
# ============================================================================
def test_no_training_label_reaches_into_test_window():
    from model_trainer import ModelTrainer

    n_samples = 100
    horizon_bars = 6
    dates = pd.date_range("2026-01-01 09:15", periods=n_samples, freq="5min")
    X = pd.DataFrame({"f1_feat": range(n_samples)}, index=dates)
    y = pd.Series(["UP"] * n_samples, index=dates)

    X_tr, X_te, y_tr, y_te = ModelTrainer.time_based_split(X, y, test_fraction=0.20, purge_window=horizon_bars)

    test_start_idx = X.index.get_loc(X_te.index[0])
    train_end_idx = X.index.get_loc(X_tr.index[-1])

    # The gap between last train row and first test row must equal or exceed purge_window
    purge_gap = test_start_idx - train_end_idx
    assert (
        purge_gap >= horizon_bars + 1
    ), f"Purge gap ({purge_gap}) must be >= {horizon_bars + 1} bars to prevent forward label leakage."


# ============================================================================
# 16. Global Risk Monitor Fail-Closed State (P0-004)
# ============================================================================
def test_global_risk_monitor_fails_closed_when_drivers_unavailable():
    from global_risk_monitor import RISK_LEVEL_UNAVAILABLE, GlobalRiskMonitor

    class MockEmptyFetcher:
        def fetch_global_tickers(self, **kwargs):
            return {}  # Zero drivers available

    monitor = GlobalRiskMonitor(data_fetcher=MockEmptyFetcher())
    reading = monitor.compute_composite_risk()

    assert reading.risk_level == RISK_LEVEL_UNAVAILABLE, f"Expected {RISK_LEVEL_UNAVAILABLE}, got {reading.risk_level}"
    assert reading.is_available is False
    assert "unavailable" in reading.banner_message.lower()


# ============================================================================
# 17. Model Artifact Provenance & Lineage (P0-006)
# ============================================================================
def test_model_metadata_lineage_fields(tmp_path):
    import json

    from ensemble_manager import EnsembleManager

    em = EnsembleManager()
    dummy_models = {"dummy": None}
    feature_cols = ["f1_feat", "f2_feat"]

    # Test saving with full lineage
    em._save_ensemble(
        symbol="TEST_LINEAGE",
        fitted_models=dummy_models,
        feature_columns=feature_cols,
        per_model_accuracy={"dummy": 0.8},
        ensemble_accuracy=0.8,
        mean_agreement=1.0,
        horizon="INTRADAY",
        training_sample_count=150,
        training_data_start="2026-01-01",
        training_data_end="2026-01-10",
    )

    meta_path = em._ensemble_metadata_path("TEST_LINEAGE", "INTRADAY")
    assert meta_path.exists(), "Metadata file was not created"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Invariants for P0-006 model lineage
    required_lineage_fields = [
        "model_id",
        "symbol",
        "horizon",
        "model_version",
        "feature_version",
        "trained_at",
        "code_commit_sha",
        "training_sample_count",
        "training_seed",
        "feature_schema_hash",
        "environment",
    ]
    for field in required_lineage_fields:
        assert field in meta, f"Required lineage field '{field}' missing from metadata"

    assert len(meta["model_id"]) > 10, "model_id must be a non-empty UUID"
    assert "python_version" in meta["environment"]
    assert "sklearn_version" in meta["environment"]
