"""
tests/test_phase6_calibration.py
Phase 6 Calibration Verification Test Suite (CAL-001 to CAL-003).

Tests:
1. CAL-001: Sparse and non-contiguous calibration bins, numeric interval lookup, edge cases (NaN, Inf, extremes).
2. CAL-002: Adversarial calibration fixtures (overconfident, underconfident, constant 0.5, inverted confidence,
            severe class imbalance, small sample, concept drift, perfectly calibrated).
3. CAL-003: Strict 4-way chronological dataset separation (TRAIN -> CALIBRATION -> EDGE VALIDATION -> FINAL HOLDOUT)
            with purge window boundary isolation.
"""

import numpy as np
import pandas as pd
import pytest

from model_trainer import FourWaySplitResult, ModelTrainer
from runtime_validator import (
    STATUS_DRIFT_DETECTED,
    STATUS_INSUFFICIENT_DATA,
    STATUS_STABLE,
    STATUS_SUFFICIENT,
    CalibrationBin,
    CalibrationResult,
    RuntimeValidator,
)


# ============================================================================
# 1. CAL-001 — Sparse-Bin & Numeric Interval Lookup Tests
# ============================================================================
class TestCAL001SparseBinsAndIntervalLookup:
    """CAL-001: Exhaustively test sparse/non-contiguous calibration bins and numeric interval lookup."""

    @pytest.fixture
    def rv(self):
        return RuntimeValidator()

    def test_sparse_alternating_bins(self, rv):
        """
        bins = [populated, empty, populated, empty, populated]
        Lookup must return empirical accuracy for populated bins and None for empty bins.
        """
        bins = [
            CalibrationBin(
                lower_bound=0.0, upper_bound=0.2, count=50, mean_predicted_confidence=0.1, empirical_accuracy=0.15
            ),
            CalibrationBin(
                lower_bound=0.2, upper_bound=0.4, count=0, mean_predicted_confidence=0.0, empirical_accuracy=0.0
            ),
            CalibrationBin(
                lower_bound=0.4, upper_bound=0.6, count=60, mean_predicted_confidence=0.5, empirical_accuracy=0.52
            ),
            CalibrationBin(
                lower_bound=0.6, upper_bound=0.8, count=0, mean_predicted_confidence=0.0, empirical_accuracy=0.0
            ),
            CalibrationBin(
                lower_bound=0.8, upper_bound=1.0, count=70, mean_predicted_confidence=0.9, empirical_accuracy=0.88
            ),
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=180,
            expected_calibration_error=0.03,
            is_well_calibrated=True,
            bins=bins,
        )

        # In populated bin 0: [0.0, 0.2]
        assert rv.get_calibrated_confidence(0.1, cal_res) == 0.15

        # In empty bin 1: (0.2, 0.4] -> count == 0 -> returns None
        assert rv.get_calibrated_confidence(0.3, cal_res) is None

        # In populated bin 2: (0.4, 0.6]
        assert rv.get_calibrated_confidence(0.5, cal_res) == 0.52

        # In empty bin 3: (0.6, 0.8] -> count == 0 -> returns None
        assert rv.get_calibrated_confidence(0.7, cal_res) is None

        # In populated bin 4: (0.8, 1.0]
        assert rv.get_calibrated_confidence(0.9, cal_res) == 0.88

    def test_lookup_by_numeric_interval_not_array_position(self, rv):
        """
        Shuffling the bin array or ordering it arbitrarily must not change lookup results.
        Numeric intervals determine the bin, not list position.
        """
        bin_low = CalibrationBin(
            lower_bound=0.0, upper_bound=0.5, count=100, mean_predicted_confidence=0.25, empirical_accuracy=0.30
        )
        bin_high = CalibrationBin(
            lower_bound=0.5, upper_bound=1.0, count=100, mean_predicted_confidence=0.75, empirical_accuracy=0.80
        )

        # Pass in reverse order: [high, low]
        cal_res_reversed = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=200,
            expected_calibration_error=0.05,
            is_well_calibrated=True,
            bins=[bin_high, bin_low],
        )

        # Conf 0.2 must match bin_low (0.30), NOT bin_high even though bin_high is at index 0
        assert rv.get_calibrated_confidence(0.2, cal_res_reversed) == 0.30
        # Conf 0.8 must match bin_high (0.80)
        assert rv.get_calibrated_confidence(0.8, cal_res_reversed) == 0.80

    def test_empty_bins_all_empty(self, rv):
        """When all bins have count=0, any lookup must return None."""
        bins = [
            CalibrationBin(
                lower_bound=0.0, upper_bound=0.5, count=0, mean_predicted_confidence=0.0, empirical_accuracy=0.0
            ),
            CalibrationBin(
                lower_bound=0.5, upper_bound=1.0, count=0, mean_predicted_confidence=0.0, empirical_accuracy=0.0
            ),
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=0,
            expected_calibration_error=0.0,
            is_well_calibrated=True,
            bins=bins,
        )
        assert rv.get_calibrated_confidence(0.2, cal_res) is None
        assert rv.get_calibrated_confidence(0.8, cal_res) is None

    def test_one_populated_bin(self, rv):
        """Only one populated bin; inputs in that bin return accuracy, outside return None."""
        bins = [
            CalibrationBin(
                lower_bound=0.4, upper_bound=0.6, count=100, mean_predicted_confidence=0.5, empirical_accuracy=0.55
            )
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=100,
            expected_calibration_error=0.05,
            is_well_calibrated=True,
            bins=bins,
        )
        assert rv.get_calibrated_confidence(0.5, cal_res) == 0.55
        assert rv.get_calibrated_confidence(0.2, cal_res) is None
        assert rv.get_calibrated_confidence(0.8, cal_res) is None

    def test_all_bins_populated(self, rv):
        """10 contiguous populated bins; all queries in [0, 1] return their respective empirical accuracy."""
        bins = [
            CalibrationBin(
                lower_bound=i / 10.0,
                upper_bound=(i + 1) / 10.0,
                count=50,
                mean_predicted_confidence=(i + 0.5) / 10.0,
                empirical_accuracy=(i + 0.5) / 10.0,
            )
            for i in range(10)
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=500,
            expected_calibration_error=0.0,
            is_well_calibrated=True,
            bins=bins,
        )
        for i in range(10):
            query = (i + 0.5) / 10.0
            expected = (i + 0.5) / 10.0
            assert rv.get_calibrated_confidence(query, cal_res) == pytest.approx(expected, abs=1e-4)

    def test_extreme_confidence_boundaries(self, rv):
        """Exact 0.0 and 1.0 confidence must map to correct boundary bins."""
        bins = [
            CalibrationBin(
                lower_bound=0.0, upper_bound=0.5, count=50, mean_predicted_confidence=0.25, empirical_accuracy=0.20
            ),
            CalibrationBin(
                lower_bound=0.5, upper_bound=1.0, count=50, mean_predicted_confidence=0.75, empirical_accuracy=0.85
            ),
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=100,
            expected_calibration_error=0.05,
            is_well_calibrated=True,
            bins=bins,
        )
        assert rv.get_calibrated_confidence(0.0, cal_res) == 0.20
        assert rv.get_calibrated_confidence(1.0, cal_res) == 0.85

    def test_confidence_outside_expected_range_rejected(self, rv):
        """Confidences < 0.0 or > 1.0 must return None without clamping."""
        bins = [
            CalibrationBin(
                lower_bound=0.0, upper_bound=1.0, count=100, mean_predicted_confidence=0.5, empirical_accuracy=0.5
            )
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=100,
            expected_calibration_error=0.0,
            is_well_calibrated=True,
            bins=bins,
        )
        assert rv.get_calibrated_confidence(-0.01, cal_res) is None
        assert rv.get_calibrated_confidence(-1.0, cal_res) is None
        assert rv.get_calibrated_confidence(1.01, cal_res) is None
        assert rv.get_calibrated_confidence(2.5, cal_res) is None

    def test_nan_and_infinity_rejected(self, rv):
        """NaN, Inf, -Inf, and None must return None."""
        bins = [
            CalibrationBin(
                lower_bound=0.0, upper_bound=1.0, count=100, mean_predicted_confidence=0.5, empirical_accuracy=0.5
            )
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=100,
            expected_calibration_error=0.0,
            is_well_calibrated=True,
            bins=bins,
        )
        assert rv.get_calibrated_confidence(float("nan"), cal_res) is None
        assert rv.get_calibrated_confidence(np.nan, cal_res) is None
        assert rv.get_calibrated_confidence(float("inf"), cal_res) is None
        assert rv.get_calibrated_confidence(float("-inf"), cal_res) is None
        assert rv.get_calibrated_confidence(np.inf, cal_res) is None
        assert rv.get_calibrated_confidence(None, cal_res) is None
        assert rv.get_calibrated_confidence("invalid", cal_res) is None

    def test_min_bin_samples_gating(self, rv):
        """Bins with count < min_bin_samples must return None."""
        bins = [
            CalibrationBin(
                lower_bound=0.0, upper_bound=0.5, count=3, mean_predicted_confidence=0.25, empirical_accuracy=0.3
            ),
            CalibrationBin(
                lower_bound=0.5, upper_bound=1.0, count=50, mean_predicted_confidence=0.75, empirical_accuracy=0.8
            ),
        ]
        cal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=53,
            expected_calibration_error=0.05,
            is_well_calibrated=True,
            bins=bins,
        )
        # Default min_bin_samples=1 passes count=3
        assert rv.get_calibrated_confidence(0.2, cal_res, min_bin_samples=1) == 0.3
        # Strict min_bin_samples=5 rejects count=3
        assert rv.get_calibrated_confidence(0.2, cal_res, min_bin_samples=5) is None
        # count=50 passes min_bin_samples=5
        assert rv.get_calibrated_confidence(0.8, cal_res, min_bin_samples=5) == 0.8


# ============================================================================
# 2. CAL-002 — Adversarial Calibration Fixtures
# ============================================================================
class TestCAL002AdversarialFixtures:
    """CAL-002: Adversarial calibration fixtures and fail-safe behavior."""

    @pytest.fixture
    def rv(self):
        return RuntimeValidator(min_calibration_samples=50, ece_threshold=0.10)

    def test_fixture_small_sample_fails_safe(self, rv):
        """Fixture: N < min_calibration_samples returns INSUFFICIENT_DATA."""
        df = pd.DataFrame(
            {
                "confidence": [0.6, 0.7, 0.8, 0.65, 0.75],
                "correct": [True, True, False, True, False],
            }
        )
        res = rv.compute_calibration(df)
        assert res.status == STATUS_INSUFFICIENT_DATA
        assert res.is_well_calibrated is False
        assert res.expected_calibration_error is None
        assert any("Only 5" in r for r in res.reasons)

    def test_fixture_overconfident(self, rv):
        """
        Fixture: Model claims 0.92 confidence, but empirical accuracy is only ~0.35.
        Severe overconfidence must trigger high ECE and mark is_well_calibrated=False.
        """
        rng = np.random.default_rng(101)
        n = 200
        confidences = rng.uniform(0.85, 0.98, size=n)
        correct = rng.uniform(0, 1, size=n) < 0.35
        df = pd.DataFrame({"confidence": confidences, "correct": correct})

        res = rv.compute_calibration(df)
        assert res.status == STATUS_SUFFICIENT
        assert res.is_well_calibrated is False
        assert res.expected_calibration_error is not None
        assert res.expected_calibration_error > 0.40
        assert any("Overconfident" in r for r in res.reasons)

    def test_fixture_underconfident(self, rv):
        """
        Fixture: Model claims 0.55 confidence, but empirical accuracy is ~0.95.
        Severe underconfidence must trigger high ECE and mark is_well_calibrated=False.
        """
        rng = np.random.default_rng(102)
        n = 200
        confidences = rng.uniform(0.50, 0.60, size=n)
        correct = rng.uniform(0, 1, size=n) < 0.95
        df = pd.DataFrame({"confidence": confidences, "correct": correct})

        res = rv.compute_calibration(df)
        assert res.status == STATUS_SUFFICIENT
        assert res.is_well_calibrated is False
        assert res.expected_calibration_error is not None
        assert res.expected_calibration_error > 0.30
        assert any("Underconfident" in r for r in res.reasons)

    def test_fixture_constant_half_zero_resolution(self, rv):
        """
        Fixture: Model always outputs constant 0.50 (zero variance / zero resolution).
        Must be flagged as zero resolution and marked is_well_calibrated=False.
        """
        n = 100
        df = pd.DataFrame(
            {
                "confidence": [0.50] * n,
                "correct": [True, False] * (n // 2),
            }
        )
        res = rv.compute_calibration(df)
        assert res.status == STATUS_SUFFICIENT
        assert res.is_well_calibrated is False
        assert any("Zero resolution" in r for r in res.reasons)

    def test_fixture_inverted_confidence(self, rv):
        """
        Fixture: Inverted confidence (higher predicted confidence -> lower accuracy).
        E.g., conf 0.2 has 90% accuracy, conf 0.8 has 10% accuracy.
        Monotonicity check must detect inverted confidence and fail calibration.
        """
        rng = np.random.default_rng(103)
        n = 200
        # 100 low confidence samples with 90% accuracy
        c_low = rng.uniform(0.15, 0.30, size=100)
        y_low = rng.uniform(0, 1, size=100) < 0.90
        # 100 high confidence samples with 10% accuracy
        c_high = rng.uniform(0.70, 0.85, size=100)
        y_high = rng.uniform(0, 1, size=100) < 0.10

        df = pd.DataFrame(
            {
                "confidence": np.concatenate([c_low, c_high]),
                "correct": np.concatenate([y_low, y_high]),
            }
        )
        res = rv.compute_calibration(df)
        assert res.status == STATUS_SUFFICIENT
        assert res.is_well_calibrated is False
        assert any("Inverted confidence" in r for r in res.reasons)

    def test_fixture_severe_class_imbalance(self, rv):
        """
        Fixture: Target label has zero variance (e.g. 100% False).
        Must detect severe class imbalance and fail safe.
        """
        n = 100
        df = pd.DataFrame(
            {
                "confidence": np.linspace(0.2, 0.8, n),
                "correct": [False] * n,
            }
        )
        res = rv.compute_calibration(df)
        assert res.status == STATUS_SUFFICIENT
        assert res.is_well_calibrated is False
        assert any("Severe class imbalance" in r for r in res.reasons)

    def test_fixture_perfectly_calibrated(self, rv):
        """
        Fixture: Genuine well-calibrated predictions (p(correct) == confidence).
        Must produce low ECE <= 0.05 and pass is_well_calibrated=True.
        """
        rng = np.random.default_rng(42)
        n = 2000
        confidences = rng.uniform(0.1, 0.9, size=n)
        correct = rng.uniform(0, 1, size=n) < confidences
        df = pd.DataFrame({"confidence": confidences, "correct": correct})

        res = rv.compute_calibration(df)
        assert res.status == STATUS_SUFFICIENT
        assert res.is_well_calibrated is True
        assert res.expected_calibration_error is not None
        assert res.expected_calibration_error < 0.05
        assert any("well-calibrated" in r for r in res.reasons)

    def test_fixture_concept_drift(self, rv):
        """
        Fixture: Reference distribution vs shifted recent distribution.
        RuntimeValidator.check_distribution_drift must flag DRIFT_DETECTED.
        """
        rng = np.random.default_rng(202)
        # Reference: uniform in [0.2, 0.8]
        ref_df = pd.DataFrame({"confidence": rng.uniform(0.2, 0.8, size=200)})
        # Recent: shifted sharply to extreme high confidence [0.85, 0.99]
        rec_df = pd.DataFrame({"confidence": rng.uniform(0.85, 0.99, size=100)})

        drift_res = rv.check_distribution_drift(rec_df, ref_df)
        assert drift_res.status == STATUS_DRIFT_DETECTED
        assert drift_res.drift_detected is True
        assert drift_res.confidence_psi > 0.25

        # Also test insufficient samples for drift check (< 20 samples)
        small_rec = pd.DataFrame({"confidence": [0.5] * 10})
        small_drift = rv.check_distribution_drift(small_rec, ref_df)
        assert small_drift.status == STATUS_INSUFFICIENT_DATA
        assert small_drift.drift_detected is False


# ============================================================================
# 3. CAL-003 — Chronological Multi-Stage Dataset Segregation
# ============================================================================
class TestCAL003SeparateValidationDatasets:
    """
    CAL-003: Strictly chronological multi-stage dataset split:
    TRAIN -> CALIBRATION -> EDGE VALIDATION -> FINAL HOLDOUT.
    """

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(303)
        dates = pd.date_range("2025-01-01", periods=400, freq="D")
        X = pd.DataFrame(
            {
                "feat1": rng.normal(0, 1, 400),
                "feat2": rng.normal(5, 2, 400),
            },
            index=dates,
        )
        y = pd.Series(rng.choice(["UP", "DOWN", "FLAT"], size=400), index=dates)
        return X, y

    def test_chronological_4way_split_boundaries(self, sample_data):
        """
        Verify that 4-way chronological split partitions into:
        TRAIN (50%), CALIBRATION (20%), EDGE VALIDATION (15%), FINAL HOLDOUT (15%)
        with zero temporal overlap and non-empty partitions.
        """
        X, y = sample_data
        horizon_bars = 5
        res = ModelTrainer.chronological_4way_split(
            X,
            y,
            train_frac=0.50,
            cal_frac=0.20,
            edge_frac=0.15,
            holdout_frac=0.15,
            horizon_bars=horizon_bars,
        )

        assert isinstance(res, FourWaySplitResult)
        assert len(res.X_train) > 0
        assert len(res.X_cal) > 0
        assert len(res.X_edge) > 0
        assert len(res.X_holdout) > 0

        # Strict chronological ordering:
        # train_end < cal_start <= cal_end < edge_start <= edge_end < holdout_start <= holdout_end
        assert res.X_train.index[-1] < res.X_cal.index[0]
        assert res.X_cal.index[-1] < res.X_edge.index[0]
        assert res.X_edge.index[-1] < res.X_holdout.index[0]

        # Purge window gaps between partitions must be >= horizon_bars
        # In a daily series, index distance between train_end and cal_start is horizon_bars
        gap1_bars = (res.X_cal.index[0] - res.X_train.index[-1]).days
        gap2_bars = (res.X_edge.index[0] - res.X_cal.index[-1]).days
        gap3_bars = (res.X_holdout.index[0] - res.X_edge.index[-1]).days

        assert gap1_bars > horizon_bars
        assert gap2_bars > horizon_bars
        assert gap3_bars > horizon_bars

        # Zero index overlap across all pairs
        train_idx = set(res.X_train.index)
        cal_idx = set(res.X_cal.index)
        edge_idx = set(res.X_edge.index)
        holdout_idx = set(res.X_holdout.index)

        assert len(train_idx & cal_idx) == 0
        assert len(train_idx & edge_idx) == 0
        assert len(train_idx & holdout_idx) == 0
        assert len(cal_idx & edge_idx) == 0
        assert len(cal_idx & holdout_idx) == 0
        assert len(edge_idx & holdout_idx) == 0

    def test_validation_isolation_helper(self, sample_data):
        """validate_4way_split_isolation reports True on valid splits and False on corrupted splits."""
        X, y = sample_data
        res = ModelTrainer.chronological_4way_split(X, y, horizon_bars=4)
        ok, msg = ModelTrainer.validate_4way_split_isolation(res)
        assert ok is True
        assert "Valid 4-way chronological isolation" in msg

        # Corrupt split by injecting holdout row into train set
        corrupted_res = FourWaySplitResult(
            X_train=pd.concat([res.X_train, res.X_holdout.iloc[:1]]),
            y_train=pd.concat([res.y_train, res.y_holdout.iloc[:1]]),
            X_cal=res.X_cal,
            y_cal=res.y_cal,
            X_edge=res.X_edge,
            y_edge=res.y_edge,
            X_holdout=res.X_holdout,
            y_holdout=res.y_holdout,
        )
        ok_corrupt, msg_corrupt = ModelTrainer.validate_4way_split_isolation(corrupted_res)
        assert ok_corrupt is False
        assert "Chronological ordering violated" in msg_corrupt or "Overlap detected" in msg_corrupt

    def test_tuple_unpacking(self, sample_data):
        """FourWaySplitResult supports tuple unpacking into 8 objects."""
        X, y = sample_data
        X_tr, y_tr, X_cal, y_cal, X_ed, y_ed, X_ho, y_ho = ModelTrainer.chronological_4way_split(X, y)
        assert len(X_tr) == len(y_tr)
        assert len(X_cal) == len(y_cal)
        assert len(X_ed) == len(y_ed)
        assert len(X_ho) == len(y_ho)

    def test_invalid_parameters_fail_fast(self, sample_data):
        """Invalid fractions, lengths, or sample counts must raise ValueError."""
        X, y = sample_data
        # Fractions don't sum to 1.0
        with pytest.raises(ValueError, match="Split fractions must sum to 1.0"):
            ModelTrainer.chronological_4way_split(X, y, train_frac=0.4, cal_frac=0.2, edge_frac=0.1, holdout_frac=0.1)

        # Negative fraction
        with pytest.raises(ValueError, match="strictly positive"):
            ModelTrainer.chronological_4way_split(
                X, y, train_frac=-0.1, cal_frac=0.5, edge_frac=0.3, holdout_frac=0.3
            )

        # Mismatched length
        with pytest.raises(ValueError, match="equal length"):
            ModelTrainer.chronological_4way_split(X.iloc[:100], y)

        # Insufficient samples
        with pytest.raises(ValueError, match="Insufficient samples"):
            ModelTrainer.chronological_4way_split(X.iloc[:20], y.iloc[:20])
