# 1. Standard library imports
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from config import HORIZON_CONFIG, HORIZON_INTRADAY
from feature_engineer import (
    CANONICAL_FEATURE_CATALOG,
    ML_SAFE_SUFFIX,
    AlignmentCoverageReport,
    FeatureEngineer,
    FeatureSpec,
    validate_feature_contract,
)
from model_trainer import ModelTrainer, TemporalInformationInterval
from reference_level_engine import ReferenceLevelDeltas


def _generate_synthetic_market_data(
    num_bars: int = 400,
    base_price: float = 1000.0,
    seed: int = 42,
    freq_minutes: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic stock and index OHLCV with UTC timestamps."""
    rng = np.random.default_rng(seed)
    start_dt = datetime(2026, 6, 1, 9, 15, tzinfo=timezone.utc)
    timestamps = [start_dt + timedelta(minutes=i * freq_minutes) for i in range(num_bars)]

    # Stock
    prices = [base_price]
    for _ in range(num_bars - 1):
        prices.append(max(10.0, prices[-1] + rng.normal(0, 1.2)))
    stock_df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + abs(rng.normal(1.0, 0.5)) for p in prices],
            "Low": [p - abs(rng.normal(1.0, 0.5)) for p in prices],
            "Close": prices,
            "Volume": [int(abs(rng.normal(15000, 3000))) for _ in range(num_bars)],
        },
        index=pd.DatetimeIndex(timestamps),
    )

    # Index
    idx_prices = [24000.0]
    for _ in range(num_bars - 1):
        idx_prices.append(max(100.0, idx_prices[-1] + rng.normal(0, 5.0)))
    index_df = pd.DataFrame(
        {
            "Open": idx_prices,
            "High": [p + abs(rng.normal(5.0, 1.0)) for p in idx_prices],
            "Low": [p - abs(rng.normal(5.0, 1.0)) for p in idx_prices],
            "Close": idx_prices,
            "Volume": [int(abs(rng.normal(80000, 10000))) for _ in range(num_bars)],
        },
        index=pd.DatetimeIndex(timestamps),
    )

    return stock_df, index_df


# ============================================================================
# FEAT-001: Randomized Adversarial Future-Mutation Leakage Test Harness
# ============================================================================
def test_feat_001_adversarial_mutation_close_price():
    """
    FEAT-001: For random historical bar j, mutating future bar j Close price
    must produce zero change in all features computed for any bar i < j.
    """
    stock_df, index_df = _generate_synthetic_market_data(num_bars=300, seed=101)
    fe = FeatureEngineer()

    orig_feat = fe.engineer_features_for_horizon(stock_df, index_df, horizon=HORIZON_INTRADAY)
    assert orig_feat is not None
    ml_safe_cols = [c for c in orig_feat.columns if c.endswith(ML_SAFE_SUFFIX)]
    assert len(ml_safe_cols) > 10

    # Pick 5 random split points between bar 50 and bar 250
    rng = np.random.default_rng(202)
    split_positions = rng.integers(50, 250, size=5)

    for j in split_positions:
        mutated_stock = stock_df.copy()
        # Extreme shock to future bar j: 50% price crash
        mutated_stock.iloc[j, mutated_stock.columns.get_loc("Close")] *= 0.50
        mutated_stock.iloc[j, mutated_stock.columns.get_loc("High")] *= 0.50
        mutated_stock.iloc[j, mutated_stock.columns.get_loc("Low")] *= 0.50

        mut_feat = fe.engineer_features_for_horizon(mutated_stock, index_df, horizon=HORIZON_INTRADAY)
        assert mut_feat is not None

        # Invariant: For all i < j, all ML-safe features must be bit-for-bit identical within 1e-7
        diff = (orig_feat.iloc[:j][ml_safe_cols] - mut_feat.iloc[:j][ml_safe_cols]).abs()
        max_diff = diff.max().max()
        assert np.isnan(max_diff) or max_diff < 1e-7, (
            f"FEAT-001 Violation: Close price mutation at j={j} leaked backwards into features before j! "
            f"Max diff: {max_diff}"
        )


def test_feat_001_adversarial_mutation_volume():
    """FEAT-001: Mutating future bar j volume must produce zero change for i < j."""
    stock_df, index_df = _generate_synthetic_market_data(num_bars=300, seed=303)
    fe = FeatureEngineer()

    orig_feat = fe.engineer_features_for_horizon(stock_df, index_df, horizon=HORIZON_INTRADAY)
    ml_safe_cols = [c for c in orig_feat.columns if c.endswith(ML_SAFE_SUFFIX)]

    for j in [60, 120, 180]:
        mutated_stock = stock_df.copy()
        mutated_stock.iloc[j, mutated_stock.columns.get_loc("Volume")] *= 1000  # Massive spike

        mut_feat = fe.engineer_features_for_horizon(mutated_stock, index_df, horizon=HORIZON_INTRADAY)
        diff = (orig_feat.iloc[:j][ml_safe_cols] - mut_feat.iloc[:j][ml_safe_cols]).abs()
        max_diff = diff.max().max()
        assert np.isnan(max_diff) or max_diff < 1e-7, (
            f"FEAT-001 Violation: Volume mutation at j={j} leaked into features before j! Max diff: {max_diff}"
        )


def test_feat_001_adversarial_mutation_index():
    """FEAT-001: Mutating future index prices must produce zero change in stock features for i < j."""
    stock_df, index_df = _generate_synthetic_market_data(num_bars=300, seed=404)
    fe = FeatureEngineer()

    orig_feat = fe.engineer_features_for_horizon(stock_df, index_df, horizon=HORIZON_INTRADAY)
    ml_safe_cols = [c for c in orig_feat.columns if c.endswith(ML_SAFE_SUFFIX)]

    for j in [75, 150, 225]:
        mutated_index = index_df.copy()
        mutated_index.iloc[j, mutated_index.columns.get_loc("Close")] += 5000.0

        mut_feat = fe.engineer_features_for_horizon(stock_df, mutated_index, horizon=HORIZON_INTRADAY)
        diff = (orig_feat.iloc[:j][ml_safe_cols] - mut_feat.iloc[:j][ml_safe_cols]).abs()
        max_diff = diff.max().max()
        assert np.isnan(max_diff) or max_diff < 1e-7, (
            f"FEAT-001 Violation: Index mutation at j={j} leaked into features before j! Max diff: {max_diff}"
        )


# ============================================================================
# FEAT-002: Causal Feature Contracts at API / Type Boundary
# ============================================================================
def test_feat_002_feature_spec_causal_contract_validation():
    """FEAT-002: FeatureSpec rejects non-causal shifts (< 1)."""
    # Valid spec
    spec = FeatureSpec(
        name="test_feat",
        source="stock_ohlcv",
        max_lookback=20,
        shift=1,
    )
    assert spec.shift == 1
    assert spec.allowed_timestamp == "prior_bar_close"

    # Non-causal shift = 0 must raise ValueError
    with pytest.raises(ValueError, match="violates causal contract: shift=0 < 1"):
        FeatureSpec(
            name="lookahead_feat",
            source="stock_ohlcv",
            max_lookback=20,
            shift=0,
        )

    # Negative shift (future data) must raise ValueError
    with pytest.raises(ValueError, match="violates causal contract"):
        FeatureSpec(
            name="future_feat",
            source="stock_ohlcv",
            max_lookback=20,
            shift=-1,
        )


def test_feat_002_all_canonical_features_have_valid_specs():
    """FEAT-002: Every feature in CANONICAL_FEATURE_CATALOG has shift >= 1."""
    assert len(CANONICAL_FEATURE_CATALOG) >= 20
    for name, spec in CANONICAL_FEATURE_CATALOG.items():
        assert spec.shift >= 1, f"Feature {name} has non-causal shift={spec.shift}"
        assert name.endswith(ML_SAFE_SUFFIX), f"Feature {name} does not end with {ML_SAFE_SUFFIX}"


def test_feat_002_validate_feature_contract_catches_unregistered_and_raw():
    """FEAT-002: validate_feature_contract catches unregistered and raw unlagged features."""
    # Valid list
    valid_cols = ["rsi_feat", "macd_line_feat", "atr_feat"]
    assert validate_feature_contract(valid_cols) == []

    # Unregistered feature
    bad_cols = ["rsi_feat", "unknown_custom_feat"]
    errors = validate_feature_contract(bad_cols)
    assert len(errors) == 1
    assert "unknown_custom_feat" in errors[0]

    # Raw unlagged column in DataFrame
    raw_df = pd.DataFrame({"rsi": [50.0], "rsi_feat": [50.0]})
    raw_errors = validate_feature_contract(raw_df)
    assert len(raw_errors) >= 1
    assert "Raw unshifted feature columns present" in raw_errors[0]


def test_feat_002_model_trainer_prepare_dataset_enforces_contract():
    """FEAT-002: ModelTrainer.prepare_dataset rejects invalid feature contracts."""
    trainer = ModelTrainer()
    stock_df, index_df = _generate_synthetic_market_data(num_bars=200)

    # Normal dataset preparation succeeds
    res = trainer.prepare_dataset(stock_df, index_df, horizon=HORIZON_INTRADAY)
    assert res is not None
    X, y, cols = res
    assert len(cols) > 0

    # Injected rogue uncontracted feature in engineer output raises ValueError
    rogue_fe = FeatureEngineer()
    orig_engineer = rogue_fe.engineer_features_for_horizon

    def mocked_engineer(*args, **kwargs):
        df = orig_engineer(*args, **kwargs)
        df["unauthorized_lookahead_feat"] = 1.0  # Rogue column not in catalog
        return df

    rogue_fe.engineer_features_for_horizon = mocked_engineer
    rogue_trainer = ModelTrainer(feature_engineer=rogue_fe)

    with pytest.raises(ValueError, match="Feature contract violation"):
        rogue_trainer.prepare_dataset(stock_df, index_df, horizon=HORIZON_INTRADAY)


# ============================================================================
# FEAT-003: Cross-Series Alignment Coverage Minimum Enforcement
# ============================================================================
def test_feat_003_alignment_coverage_and_degradation():
    """FEAT-003: Alignment below threshold marks status DEGRADED or UNAVAILABLE."""
    stock_df, index_df = _generate_synthetic_market_data(num_bars=100)
    fe = FeatureEngineer()

    # 1. Full 100% overlap -> HEALTHY
    report_healthy = fe.check_alignment(stock_df, index_df)
    assert report_healthy.status == "HEALTHY"
    assert report_healthy.coverage_ratio == 1.0
    assert report_healthy.matched_count == 100

    # 2. 80% overlap -> DEGRADED (between 50% and 95%)
    degraded_index = index_df.iloc[:80]
    report_degraded = fe.check_alignment(stock_df, degraded_index)
    assert report_degraded.status == "DEGRADED"
    assert report_degraded.coverage_ratio == 0.80

    # 3. 30% overlap -> UNAVAILABLE (< 50%)
    broken_index = index_df.iloc[:30]
    report_unavail = fe.check_alignment(stock_df, broken_index)
    assert report_unavail.status == "UNAVAILABLE"
    assert report_unavail.coverage_ratio == 0.30

    # 4. Under UNAVAILABLE, engineer_features sets cross-series features to NaN
    out = fe.engineer_features(stock_df, broken_index)
    assert out is not None
    assert out["outperformance_pct"].isna().all()
    assert out["nifty_correlation"].isna().all()


# ============================================================================
# FEAT-004: Adaptive Deadband Causality
# ============================================================================
def test_feat_004_adaptive_deadband_causality():
    """
    FEAT-004: Adaptive deadband at timestamp t depends strictly on data <= t.
    Mutating prices after bar j must not change deadband at or before j.
    """
    trainer = ModelTrainer()
    stock_df, _ = _generate_synthetic_market_data(num_bars=300, seed=505)

    horizon_bars = 6
    deadband_default = 0.15

    orig_deadband = trainer.compute_adaptive_deadband(
        stock_df, horizon_bars=horizon_bars, deadband_pct_default=deadband_default
    )

    # Mutate future bar 200
    j = 200
    mutated_stock = stock_df.copy()
    mutated_stock.iloc[j:, mutated_stock.columns.get_loc("Close")] *= 1.50

    mut_deadband = trainer.compute_adaptive_deadband(
        mutated_stock, horizon_bars=horizon_bars, deadband_pct_default=deadband_default
    )

    # For all i < j, deadbands must be strictly identical
    diff = (orig_deadband.iloc[:j] - mut_deadband.iloc[:j]).abs().max()
    assert diff < 1e-9, f"FEAT-004 Violation: Deadband before j={j} was altered by future prices! Diff: {diff}"


def test_feat_004_adaptive_deadband_train_test_split_boundary():
    """
    FEAT-004: Modifying the test set has zero effect on training set adaptive deadbands.
    """
    trainer = ModelTrainer()
    stock_df, _ = _generate_synthetic_market_data(num_bars=400, seed=606)

    split_point = 250
    train_df = stock_df.iloc[:split_point].copy()

    # Calculate deadband on train_df directly
    train_deadband_isolated = trainer.compute_adaptive_deadband(train_df, horizon_bars=6, deadband_pct_default=0.15)

    # Calculate deadband on full dataset
    full_deadband = trainer.compute_adaptive_deadband(stock_df, horizon_bars=6, deadband_pct_default=0.15)

    # Verify that the train portion of full_deadband matches train_deadband_isolated
    diff = (train_deadband_isolated - full_deadband.iloc[:split_point]).abs().max()
    assert diff < 1e-9, f"FEAT-004 Violation: Full dataset deadband diverged from isolated train deadband! Diff: {diff}"


# ============================================================================
# FEAT-005: Purge and Embargo Boundaries for Cross-Validation & Splits
# ============================================================================
def test_feat_005_temporal_information_intervals():
    """FEAT-005: TemporalInformationInterval correctly constructs feature & label windows."""
    trainer = ModelTrainer()
    stock_df, _ = _generate_synthetic_market_data(num_bars=50)

    intervals = trainer.compute_information_intervals(stock_df, horizon_bars=6, max_lookback_bars=10)
    assert len(intervals) == 50

    # Observation 20
    obs = intervals[20]
    assert obs.timestamp == stock_df.index[20]
    assert obs.feature_start == stock_df.index[10]
    assert obs.feature_end == stock_df.index[20]
    assert obs.label_start == stock_df.index[20]
    assert obs.label_end == stock_df.index[26]

    # Check overlap with test set starting at bar 25
    test_start = stock_df.index[25]
    test_end = stock_df.index[49]
    # Observation 20 label extends to bar 26, which overlaps with test set [25, 49]
    assert obs.overlaps_with(test_start, test_end) is True

    # Observation 15 label extends to bar 21, which does NOT overlap with test set [25, 49]
    obs_safe = intervals[15]
    assert obs_safe.label_end == stock_df.index[21]
    assert obs_safe.overlaps_with(test_start, test_end) is False


def test_feat_005_split_isolation_and_purge_default():
    """
    FEAT-005: time_based_split automatically defaults purge_window to horizon_bars
    when horizon_bars is provided and purge_window is 0.
    """
    trainer = ModelTrainer()
    stock_df, index_df = _generate_synthetic_market_data(num_bars=300)
    prepared = trainer.prepare_dataset(stock_df, index_df, horizon=HORIZON_INTRADAY)
    assert prepared is not None
    X, y, _ = prepared

    horizon_bars = 6
    # Call with horizon_bars and purge_window=0
    X_tr, X_te, y_tr, y_te = trainer.time_based_split(
        X, y, test_fraction=0.20, purge_window=0, horizon_bars=horizon_bars
    )

    train_end_pos = len(X_tr)
    test_start_pos = len(X) - len(X_te)
    gap = test_start_pos - train_end_pos

    assert gap >= horizon_bars, (
        f"FEAT-005 Violation: Purge gap ({gap}) is less than horizon bars ({horizon_bars})!"
    )

    # Validate temporal isolation helper
    is_isolated, msg = trainer.validate_split_isolation(X_tr, X_te, horizon_bars=horizon_bars)
    assert is_isolated is True
    assert "Valid" in msg
