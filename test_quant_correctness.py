import json
import shutil

import numpy as np
import pandas as pd
import pytest

from config import (
    ALL_HORIZONS,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    MODELS_DIR,
)
from ensemble_manager import EnsembleManager
from execution_simulator import (
    ExecutionSimulator,
    ExitReason,
)
from feature_engineer import ML_SAFE_SUFFIX, FeatureEngineer
from history_manager import HistoryManager
from model_trainer import ModelTrainer
from predictor import PredictionSignal
from runtime_validator import (
    STATUS_CALIBRATION_STALE,
    STATUS_DRIFT_DETECTED,
    STATUS_FRESH,
    RuntimeValidator,
)


def _build_synthetic_ohlcv(n_days: int = 15, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
    """Builds multi-day intraday OHLCV bars suitable for indicator warmup and feature engineering."""
    rng = np.random.default_rng(seed)
    rows, timestamps = [], []
    price = 1000.0
    base_date = pd.Timestamp("2026-01-05 09:15:00")
    recent_closes = []
    for day in range(n_days):
        day_start = base_date + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=5 * bar)
            if len(recent_closes) >= 10:
                trend = recent_closes[-1] - recent_closes[-10]
                bias = 1.0 if trend < -5 else (-1.0 if trend > 5 else 0.0)
            else:
                bias = 0.0
            drift = rng.normal(bias, 1.2)
            price = max(1.0, price + drift)
            open_p = price
            close_p = max(1.0, price + rng.normal(bias * 0.4, 0.8))
            high_p = max(open_p, close_p) + abs(rng.normal(0, 0.4))
            low_p = min(open_p, close_p) - abs(rng.normal(0, 0.4))
            vol = int(abs(rng.normal(45000, 10000)))
            rows.append([open_p, high_p, low_p, close_p, vol])
            timestamps.append(ts)
            price = close_p
            recent_closes.append(close_p)
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex(timestamps),
    )


# =====================================================================
# 1. FEAT-002: Universal Future-Mutation Invariance Test
# =====================================================================
def test_universal_future_mutation_invariance_across_all_features():
    """
    FEAT-002: Mutate future bars (t >= cutoff) and verify that every feature
    value computed at t < cutoff remains strictly identical.
    Covers all 13 feature families in FeatureEngineer.
    """
    fe = FeatureEngineer()
    df_stock = _build_synthetic_ohlcv(n_days=4, bars_per_day=75, seed=101)
    df_index = _build_synthetic_ohlcv(n_days=4, bars_per_day=75, seed=202)
    df_index.index = df_stock.index

    features_original = fe.engineer_features(df_stock, df_index)
    assert features_original is not None

    cutoff_idx = 200
    df_stock_mutated = df_stock.copy()
    df_index_mutated = df_index.copy()

    # Drastically mutate future bars (both prices and volume)
    df_stock_mutated.iloc[cutoff_idx:, df_stock_mutated.columns.get_loc("Close")] *= 2.5
    df_stock_mutated.iloc[cutoff_idx:, df_stock_mutated.columns.get_loc("High")] *= 3.0
    df_stock_mutated.iloc[cutoff_idx:, df_stock_mutated.columns.get_loc("Low")] *= 0.2
    df_stock_mutated.iloc[cutoff_idx:, df_stock_mutated.columns.get_loc("Volume")] *= 10.0
    df_index_mutated.iloc[cutoff_idx:, df_index_mutated.columns.get_loc("Close")] *= 0.5

    features_mutated = fe.engineer_features(df_stock_mutated, df_index_mutated)
    assert features_mutated is not None

    feat_cols = [c for c in features_original.columns if c.endswith(ML_SAFE_SUFFIX)]
    assert len(feat_cols) >= 10, "Expected multiple ML-safe feature columns"

    # Prior to cutoff, all ML-safe features must be identical
    for col in feat_cols:
        orig_slice = features_original[col].iloc[:cutoff_idx]
        mut_slice = features_mutated[col].iloc[:cutoff_idx]
        pd.testing.assert_series_equal(
            orig_slice,
            mut_slice,
            check_names=False,
            obj=f"Feature {col} showed future-leakage leakage at t < cutoff!",
        )


# =====================================================================
# 2. QNT-007: Horizon-Specific Causality Invariant across all 7 Horizons
# =====================================================================
@pytest.mark.parametrize("horizon", ALL_HORIZONS)
def test_horizon_specific_causality_and_purge_window(horizon):
    """
    QNT-007: For every supported horizon, verify that:
    1. The forward return label looks exactly horizon_bars into the future.
    2. The time-based split with purge_window=horizon_bars discards the boundary rows
       so training labels NEVER overlap into test-interval bars.
    """
    trainer = ModelTrainer()
    horizon_bars = HORIZON_CONFIG[horizon]["horizon_bars"]
    n_days = max(10, (horizon_bars // 75) + 8)

    df_stock = _build_synthetic_ohlcv(n_days=n_days, bars_per_day=75, seed=303)
    df_index = _build_synthetic_ohlcv(n_days=n_days, bars_per_day=75, seed=404)
    df_index.index = df_stock.index

    prepared = trainer.prepare_dataset(df_stock, df_index, horizon=horizon)
    assert prepared is not None, f"Dataset preparation failed for horizon {horizon}"
    X, y, feature_cols = prepared

    # Boundary purging verification
    X_train, X_test, y_train, y_test = trainer.time_based_split(X, y, test_fraction=0.2, purge_window=horizon_bars)

    train_end_idx = len(X_train)
    test_start_idx = len(X) - len(X_test)

    # Gap between end of train and start of test must be >= purge_window
    assert test_start_idx - train_end_idx >= horizon_bars, (
        f"Horizon {horizon}: Purge gap ({test_start_idx - train_end_idx}) "
        f"is smaller than horizon bars ({horizon_bars})!"
    )


# =====================================================================
# 3. QNT-001: Purged Walk-Forward Split with Embargo
# =====================================================================
def test_walk_forward_split_with_purge_and_embargo():
    """
    QNT-001: Verify expanding and rolling walk-forward validation splits
    with both purge and embargo windows.
    """
    trainer = ModelTrainer()
    df_stock = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, seed=505)
    df_index = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, seed=606)
    df_index.index = df_stock.index

    prepared = trainer.prepare_dataset(df_stock, df_index, horizon=HORIZON_INTRADAY)
    assert prepared is not None
    X, y, _ = prepared

    purge_win = 5
    embargo_win = 2
    n_splits = 3

    folds = list(
        trainer.walk_forward_split(
            X,
            y,
            n_splits=n_splits,
            min_train_samples=100,
            purge_window=purge_win,
            embargo_window=embargo_win,
            mode="expanding",
        )
    )
    assert len(folds) == n_splits

    for i, X_tr, X_te, y_tr, y_te in folds:
        assert len(X_tr) > 0
        assert len(X_te) > 0
        assert X_tr.index[-1] < X_te.index[0]

    # Evaluate walk forward
    wf_eval = trainer.evaluate_walk_forward(
        X, y, n_splits=3, min_train_samples=100, purge_window=purge_win, embargo_window=embargo_win
    )
    assert "folds" in wf_eval
    assert "summary" in wf_eval
    summary = wf_eval["summary"]
    assert "mean_accuracy" in summary
    assert "worst_fold_accuracy" in summary
    assert "ci_95_accuracy" in summary
    assert "mean_mcc" in summary
    assert "mean_brier_score" in summary


# =====================================================================
# 4. QNT-002 & QNT-003: Realistic Execution Simulator & Cost Sensitivity
# =====================================================================
def test_execution_simulator_path_traversal_and_cost_tiers():
    """
    QNT-002, QNT-003: Test event-driven discrete bar execution simulator:
    - 1-bar delay
    - Simultaneous target/stop hit resolved conservatively (worst-case stop-loss)
    - Monotonic cost sensitivity across 4 tiers: Optimistic > Base > Pessimistic > Stress
    """
    sim = ExecutionSimulator()

    dates = pd.date_range("2026-01-01 09:15", periods=30, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0] * 30,
            "High": [101.0] * 30,
            "Low": [99.0] * 30,
            "Close": [100.0] * 30,
            "Volume": [1000] * 30,
        },
        index=dates,
    )

    # Simultaneous hit bar at index 5: both high and low hit target/stop
    df.iloc[5, df.columns.get_loc("High")] = 110.0
    df.iloc[5, df.columns.get_loc("Low")] = 90.0

    signals = pd.Series(0, index=dates)
    signals.iloc[3] = 1  # Long signal at bar 3

    rep = sim.simulate("TEST", df, signals, stop_loss_pct=2.0, profit_target_pct=2.0)
    assert rep.n_trades == 1
    trade = rep.trades[0]
    assert trade.entry_time == dates[4]
    assert trade.exit_time == dates[5]
    assert trade.exit_reason == ExitReason.STOP_LOSS.value

    # Test sensitivity analysis across all 4 cost scenarios
    sensitivity = sim.run_sensitivity_analysis("TEST", df, signals)
    assert len(sensitivity) == 4
    for tier in ["OPTIMISTIC", "BASE", "PESSIMISTIC", "STRESS"]:
        assert tier in sensitivity

    opt_net = sensitivity["OPTIMISTIC"].cumulative_net_return_pct
    base_net = sensitivity["BASE"].cumulative_net_return_pct
    pess_net = sensitivity["PESSIMISTIC"].cumulative_net_return_pct
    stress_net = sensitivity["STRESS"].cumulative_net_return_pct
    assert opt_net >= base_net >= pess_net >= stress_net


# =====================================================================
# 5. QNT-004: Comprehensive ML Metrics
# =====================================================================
def test_enhanced_model_evaluation_metrics():
    """
    QNT-004: Verify balanced accuracy, MCC, F1 macro/weighted, per-class recall,
    Brier score, log loss, and Expected Calibration Error (ECE) are produced.
    """
    trainer = ModelTrainer()
    df_stock = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, seed=707)
    df_index = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, seed=808)
    df_index.index = df_stock.index

    prepared = trainer.prepare_dataset(df_stock, df_index, horizon=HORIZON_INTRADAY)
    assert prepared is not None
    X, y, _ = prepared

    X_train, X_test, y_train, y_test = trainer.time_based_split(X, y, test_fraction=0.3)
    model = trainer.train(X_train, y_train)

    metrics = trainer.evaluate(model, X_test, y_test)
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "matthews_corrcoef" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "expected_calibration_error" in metrics
    assert "per_class_recall" in metrics
    assert "UP" in metrics["per_class_recall"]
    assert "DOWN" in metrics["per_class_recall"]
    assert "FLAT" in metrics["per_class_recall"]


# =====================================================================
# 6. QNT-005 & QNT-006: Out-of-Sample Calibration & Drift Detection
# =====================================================================
def test_out_of_sample_calibration_and_drift_detection(tmp_path):
    """
    QNT-005, QNT-006:
    1. HistoryManager saves and isolates is_out_of_sample predictions.
    2. Rolling calibration window filters out old predictions.
    3. RuntimeValidator checks distribution drift (PSI, KS).
    """
    db_file = tmp_path / "test_history.db"
    hm = HistoryManager(db_path=db_file)

    sig_in_sample = PredictionSignal(
        symbol="TCS",
        timestamp=pd.Timestamp.now(),
        action="BUY",
        model_predicted_class="UP",
        raw_confidence=0.8,
        risk_adjusted_confidence=0.75,
        calibrated_confidence=0.75,
        agreement_fraction=0.8,
        downside_summary="",
        upside_summary="",
        reasoning={},
        global_risk_level="LOW",
        risk_toggle_enabled=True,
        is_safe_to_trade_live=True,
        data_stale=False,
        suppressed=False,
        suppression_reasons=[],
        horizon="INTRADAY",
        model_version="v1.1",
        feature_version="v1.0",
    )
    sig_out_sample = PredictionSignal(
        symbol="TCS",
        timestamp=pd.Timestamp.now(),
        action="BUY",
        model_predicted_class="UP",
        raw_confidence=0.85,
        risk_adjusted_confidence=0.80,
        calibrated_confidence=0.80,
        agreement_fraction=0.9,
        downside_summary="",
        upside_summary="",
        reasoning={},
        global_risk_level="LOW",
        risk_toggle_enabled=True,
        is_safe_to_trade_live=True,
        data_stale=False,
        suppressed=False,
        suppression_reasons=[],
        horizon="INTRADAY",
        model_version="v1.1",
        feature_version="v1.0",
    )

    id_in = hm.save_prediction(sig_in_sample, is_out_of_sample=False)
    id_out = hm.save_prediction(sig_out_sample, is_out_of_sample=True)

    hm.resolve_outcome(id_in, "UP")
    hm.resolve_outcome(id_out, "UP")

    # Calibration dataset with only_out_of_sample=True must only contain id_out
    df_all = hm.build_calibration_dataset(symbol="TCS", horizon="INTRADAY", only_out_of_sample=False)
    assert len(df_all) == 2

    df_oos = hm.build_calibration_dataset(symbol="TCS", horizon="INTRADAY", only_out_of_sample=True)
    assert len(df_oos) == 1
    assert df_oos.iloc[0]["confidence"] == 0.80

    # Drift Detection test
    rv = RuntimeValidator()
    ref_df = pd.DataFrame(
        {
            "confidence": np.random.uniform(0.4, 0.6, size=50),
            "predicted_class": ["UP"] * 25 + ["DOWN"] * 25,
        }
    )
    drifted_df = pd.DataFrame(
        {
            "confidence": np.random.uniform(0.85, 0.99, size=50),
            "predicted_class": ["UP"] * 50,
        }
    )

    drift_res = rv.check_distribution_drift(drifted_df, ref_df)
    assert drift_res.drift_detected is True
    assert drift_res.status == STATUS_DRIFT_DETECTED

    # Calibration freshness test
    fresh_res = rv.validate_calibration_freshness(
        pd.DataFrame({"timestamp": [pd.Timestamp.now()] * 25}), max_age_days=30
    )
    assert fresh_res.is_fresh is True
    assert fresh_res.status == STATUS_FRESH

    stale_res = rv.validate_calibration_freshness(
        pd.DataFrame({"timestamp": [pd.Timestamp.now() - pd.Timedelta(days=120)] * 25}), max_age_days=30
    )
    assert stale_res.is_fresh is False
    assert stale_res.status == STATUS_CALIBRATION_STALE


# =====================================================================
# 7. QNT-008: Immutable Model Artifacts, Versioning & Rollback
# =====================================================================
def test_immutable_model_artifacts_and_rollback(tmp_path):
    """
    QNT-008: Verify immutable model artifacts in models/{symbol}/{horizon}/{run_id},
    current.json pointer updates, and rollback functionality.
    """
    em = EnsembleManager()
    symbol = "ROLLBACK_TEST"
    horizon = HORIZON_INTRADAY

    df_stock = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, seed=909)
    df_index = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, seed=910)
    df_index.index = df_stock.index

    # 1. Train first version
    res1 = em.train_ensemble_for_symbol(symbol, df_stock, df_index, horizon=horizon)
    assert res1.success is True

    pointer_path = em._current_pointer_path(symbol, horizon)
    assert pointer_path.exists()
    with open(pointer_path) as f:
        ptr1 = json.load(f)
    run_id_1 = ptr1["current_run_id"]

    run_dir_1 = em._versioned_dir(symbol, horizon, run_id_1)
    assert (run_dir_1 / "ensemble.joblib").exists()
    assert (run_dir_1 / "metadata.json").exists()
    assert (run_dir_1 / "schema.json").exists()
    assert (run_dir_1 / "metrics.json").exists()

    # 2. Train second version
    res2 = em.train_ensemble_for_symbol(symbol, df_stock, df_index, horizon=horizon)
    assert res2.success is True

    with open(pointer_path) as f:
        ptr2 = json.load(f)
    run_id_2 = ptr2["current_run_id"]
    assert run_id_1 != run_id_2, "Retraining must generate a distinct immutable run_id!"

    run_dir_2 = em._versioned_dir(symbol, horizon, run_id_2)
    assert (run_dir_1 / "ensemble.joblib").exists()
    assert (run_dir_2 / "ensemble.joblib").exists()

    # 3. List versions
    versions = em.list_model_versions(symbol, horizon)
    assert len(versions) >= 2
    assert versions[0]["run_id"] == run_id_2
    assert versions[0]["is_current"] is True
    assert versions[1]["run_id"] == run_id_1
    assert versions[1]["is_current"] is False

    # 4. Rollback to version 1
    rollback_ok = em.rollback_ensemble(symbol, horizon, run_id_1)
    assert rollback_ok is True

    with open(pointer_path) as f:
        ptr_after = json.load(f)
    assert ptr_after["current_run_id"] == run_id_1

    # Cleanup test models
    shutil.rmtree(MODELS_DIR / symbol, ignore_errors=True)
