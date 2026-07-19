# 1. Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import (
    MIN_CALIBRATION_SAMPLES,
    CALIBRATION_N_BINS,
    CALIBRATION_ECE_THRESHOLD,
    EDGE_CHECK_MIN_ALPHA_PCT,
    SLIPPAGE_BPS,
    SLIPPAGE_BPS,
    TRANSACTION_COST_BPS,
    configure_logging,
)
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
STATUS_SUFFICIENT = "SUFFICIENT"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUS_EDGE_CONFIRMED = "EDGE_CONFIRMED"
STATUS_NO_EDGE = "NO_EDGE"


@dataclass
class CalibrationBin:
    bin_range: str
    count: int
    mean_predicted_confidence: float
    empirical_accuracy: float


@dataclass
class CalibrationResult:
    status: str                      # SUFFICIENT | INSUFFICIENT_DATA
    n_samples: int
    expected_calibration_error: Optional[float]
    is_well_calibrated: bool
    bins: List[CalibrationBin] = field(default_factory=list)


@dataclass
class EdgeCheckResult:
    status: str                          # EDGE_CONFIRMED | NO_EDGE
    n_periods: int
    strategy_cumulative_return_pct: float
    baseline_cumulative_return_pct: float
    alpha_pct: float


@dataclass
class LiveGateResult:
    safe_to_show_calibrated_confidence: bool
    safe_to_treat_as_live_edge: bool
    reasons: List[str]


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class RuntimeValidator:
    """
    Enforces two hard rules before any signal reaches a user:
      1. A confidence score may only be displayed once it has been checked
         against real historical calibration data — an uncalibrated
         confidence number is treated as unsafe to show, not just imprecise.
      2. A strategy may only be treated as 'live-worthy' once it demonstrates
         a real, cost-adjusted edge over the NIFTY baseline.
    This class is intentionally decoupled from where the data comes from —
    it accepts prediction/outcome records and return series directly, so it
    works today with synthetic/test data and later with real records from
    history_manager.py and backtester.py without any changes here.
    """

    def __init__(
        self,
        min_calibration_samples: int = MIN_CALIBRATION_SAMPLES,
        n_bins: int = CALIBRATION_N_BINS,
        ece_threshold: float = CALIBRATION_ECE_THRESHOLD,
        min_alpha_pct: float = EDGE_CHECK_MIN_ALPHA_PCT,
    ):
        self.min_calibration_samples = min_calibration_samples
        self.n_bins = n_bins
        self.ece_threshold = ece_threshold
        self.min_alpha_pct = min_alpha_pct

    # -----------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------
    def compute_calibration(self, predictions_df: pd.DataFrame) -> CalibrationResult:
        """
        predictions_df must have columns: 'confidence' (float in [0,1]) and
        'correct' (bool — whether the predicted class actually happened).
        Builds a reliability table across n_bins confidence buckets and
        computes the Expected Calibration Error (ECE): the count-weighted
        average gap between predicted confidence and actual empirical accuracy.
        """
        try:
            n_samples = len(predictions_df)
            if n_samples < self.min_calibration_samples:
                logger.warning(
                    f"Only {n_samples} prediction records available, need >= "
                    f"{self.min_calibration_samples} to trust calibration — confidence scores "
                    "must be treated as unverified until more history accumulates."
                )
                return CalibrationResult(
                    status=STATUS_INSUFFICIENT_DATA, n_samples=n_samples,
                    expected_calibration_error=None, is_well_calibrated=False, bins=[],
                )

            df = predictions_df.copy()
            df["confidence"] = df["confidence"].clip(0.0, 1.0)
            bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
            df["bin"] = pd.cut(df["confidence"], bins=bin_edges, include_lowest=True)

            bins: List[CalibrationBin] = []
            ece = 0.0
            for bin_range, group in df.groupby("bin", observed=True):
                if len(group) == 0:
                    continue
                mean_conf = float(group["confidence"].mean())
                empirical_acc = float(group["correct"].mean())
                count = len(group)
                bins.append(CalibrationBin(
                    bin_range=str(bin_range), count=count,
                    mean_predicted_confidence=mean_conf, empirical_accuracy=empirical_acc,
                ))
                ece += (count / n_samples) * abs(mean_conf - empirical_acc)

            is_well_calibrated = ece <= self.ece_threshold
            health_registry.report("runtime_validator", ok=True, detail="Calibration computed")
            return CalibrationResult(
                status=STATUS_SUFFICIENT, n_samples=n_samples,
                expected_calibration_error=ece, is_well_calibrated=is_well_calibrated, bins=bins,
            )

        except Exception as e:
            logger.error(f"Failed computing calibration: {e}")
            health_registry.report("runtime_validator", ok=False, detail="Failed computing calibration", error=str(e))
            return CalibrationResult(
                status=STATUS_INSUFFICIENT_DATA, n_samples=len(predictions_df) if predictions_df is not None else 0,
                expected_calibration_error=None, is_well_calibrated=False, bins=[],
            )

    def get_calibrated_confidence(self, raw_confidence: float, calibration_result: CalibrationResult) -> Optional[float]:
        """
        Returns the historically-observed empirical accuracy for the bin that
        raw_confidence falls into — i.e. what confidence SHOULD actually be
        shown, based on real track record — or None if calibration isn't
        trustworthy yet (caller must suppress the confidence display entirely).
        """
        try:
            if calibration_result.status != STATUS_SUFFICIENT or not calibration_result.bins:
                return None
            raw_confidence = max(0.0, min(1.0, raw_confidence))
            bin_width = 1.0 / self.n_bins
            bin_index = min(int(raw_confidence / bin_width), len(calibration_result.bins) - 1)
            health_registry.report("runtime_validator", ok=True)
            return calibration_result.bins[bin_index].empirical_accuracy
        except Exception as e:
            logger.error(f"Failed getting calibrated confidence for raw={raw_confidence}: {e}")
            health_registry.report("runtime_validator", ok=False, detail="Failed getting calibrated confidence", error=str(e))
            return None

    # -----------------------------------------------------------------
    # Edge check vs NIFTY baseline
    # -----------------------------------------------------------------
    def compute_edge_vs_baseline(
        self,
        strategy_returns_pct: pd.Series,
        baseline_returns_pct: pd.Series,
        slippage_bps: float = SLIPPAGE_BPS,
        transaction_cost_bps: float = TRANSACTION_COST_BPS,
    ) -> EdgeCheckResult:
        """
        strategy_returns_pct: per-trade or per-period % returns of the strategy
        (BEFORE costs — costs are deducted here so every caller pays them the
        same way, rather than trusting each caller to have already done it).
        baseline_returns_pct: NIFTY's % return over the matching periods.
        """
        try:
            aligned_strategy, aligned_baseline = strategy_returns_pct.align(baseline_returns_pct, join="inner")
            n_periods = len(aligned_strategy)
            if n_periods == 0:
                logger.error("No overlapping periods between strategy and baseline returns.")
                return EdgeCheckResult(
                    status=STATUS_NO_EDGE, n_periods=0,
                    strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0,
                )

            total_cost_pct = (slippage_bps + transaction_cost_bps) / 100.0  # bps -> %
            net_strategy_returns = aligned_strategy - total_cost_pct

            strategy_cum = (np.prod(1 + net_strategy_returns / 100.0) - 1) * 100.0
            baseline_cum = (np.prod(1 + aligned_baseline / 100.0) - 1) * 100.0
            alpha = strategy_cum - baseline_cum

            status = STATUS_EDGE_CONFIRMED if alpha > self.min_alpha_pct else STATUS_NO_EDGE
            health_registry.report("runtime_validator", ok=True, detail="Edge check computed")
            return EdgeCheckResult(
                status=status, n_periods=n_periods,
                strategy_cumulative_return_pct=float(strategy_cum),
                baseline_cumulative_return_pct=float(baseline_cum),
                alpha_pct=float(alpha),
            )

        except Exception as e:
            logger.error(f"Failed computing edge vs baseline: {e}")
            health_registry.report("runtime_validator", ok=False, detail="Failed computing edge", error=str(e))
            return EdgeCheckResult(
                status=STATUS_NO_EDGE, n_periods=0,
                strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0,
            )

    # -----------------------------------------------------------------
    # Combined gate
    # -----------------------------------------------------------------
    def validate_before_live(
        self, calibration_result: CalibrationResult, edge_check_result: EdgeCheckResult
    ) -> LiveGateResult:
        reasons: List[str] = []

        safe_confidence = (
            calibration_result.status == STATUS_SUFFICIENT and calibration_result.is_well_calibrated
        )
        if calibration_result.status == STATUS_INSUFFICIENT_DATA:
            reasons.append(
                f"Only {calibration_result.n_samples} historical predictions available "
                f"(need >= {self.min_calibration_samples}) — confidence scores must be suppressed until more accumulate."
            )
        elif not calibration_result.is_well_calibrated:
            reasons.append(
                f"Expected Calibration Error {calibration_result.expected_calibration_error:.3f} exceeds "
                f"threshold {self.ece_threshold:.3f} — confidence scores are currently overconfident or underconfident."
            )

        safe_edge = edge_check_result.status == STATUS_EDGE_CONFIRMED
        if not safe_edge:
            reasons.append(
                f"Strategy alpha vs NIFTY baseline is {edge_check_result.alpha_pct:.3f}% "
                f"(need > {self.min_alpha_pct:.3f}%) — no confirmed edge, should not be treated as live-worthy."
            )

        if safe_confidence and safe_edge:
            reasons.append("Both calibration and edge checks passed — safe to treat as a live, trustworthy signal.")

        return LiveGateResult(
            safe_to_show_calibrated_confidence=safe_confidence,
            safe_to_treat_as_live_edge=safe_edge,
            reasons=reasons,
        )


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="runtime_validator_selftest.log")
    logger.info("Running runtime_validator.py self-test...")

    rng = np.random.default_rng(42)

    def _build_well_calibrated_predictions(n: int = 500) -> pd.DataFrame:
        """Confidence exactly matches the true probability of being correct —
        a genuinely well-calibrated set of predictions."""
        confidences = rng.uniform(0.3, 0.95, size=n)
        correct = rng.uniform(0, 1, size=n) < confidences
        return pd.DataFrame({"confidence": confidences, "correct": correct})

    def _build_overconfident_predictions(n: int = 500) -> pd.DataFrame:
        """Always claims high confidence (0.9) but is only right half the time —
        a deliberately badly-calibrated set of predictions."""
        confidences = np.full(n, 0.9)
        correct = rng.uniform(0, 1, size=n) < 0.5
        return pd.DataFrame({"confidence": confidences, "correct": correct})

    try:
        print("\n=== RUNTIME VALIDATOR SELF-TEST RESULT ===")
        validator = RuntimeValidator()

        # Test 1: insufficient samples
        small_df = _build_well_calibrated_predictions(n=20)
        small_result = validator.compute_calibration(small_df)
        print(f"Insufficient-samples case -> status={small_result.status}")
        assert small_result.status == STATUS_INSUFFICIENT_DATA

        # Test 2: well-calibrated predictions -> low ECE, is_well_calibrated True
        good_df = _build_well_calibrated_predictions(n=1000)
        good_result = validator.compute_calibration(good_df)
        print(f"Well-calibrated case -> status={good_result.status}, ECE={good_result.expected_calibration_error:.4f}, "
              f"is_well_calibrated={good_result.is_well_calibrated}")
        assert good_result.status == STATUS_SUFFICIENT
        assert good_result.is_well_calibrated is True

        # Test 3: overconfident predictions -> high ECE, is_well_calibrated False
        bad_df = _build_overconfident_predictions(n=1000)
        bad_result = validator.compute_calibration(bad_df)
        print(f"Overconfident case -> status={bad_result.status}, ECE={bad_result.expected_calibration_error:.4f}, "
              f"is_well_calibrated={bad_result.is_well_calibrated}")
        assert bad_result.is_well_calibrated is False
        assert bad_result.expected_calibration_error > good_result.expected_calibration_error

        # Test 4: get_calibrated_confidence returns None when data insufficient, a real number otherwise
        none_case = validator.get_calibrated_confidence(0.8, small_result)
        real_case = validator.get_calibrated_confidence(0.8, good_result)
        print(f"Calibrated confidence when insufficient data: {none_case}")
        print(f"Calibrated confidence when well-calibrated (raw=0.8): {real_case}")
        assert none_case is None
        assert real_case is not None and 0.0 <= real_case <= 1.0

        # Test 5: edge check — strategy clearly beats baseline
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        strategy_returns = pd.Series(rng.normal(0.15, 0.3, size=100), index=dates)  # positive drift
        baseline_returns = pd.Series(rng.normal(0.02, 0.3, size=100), index=dates)  # smaller drift
        edge_result_positive = validator.compute_edge_vs_baseline(strategy_returns, baseline_returns)
        print(f"Positive-edge case -> alpha={edge_result_positive.alpha_pct:.2f}%, status={edge_result_positive.status}")
        assert edge_result_positive.status == STATUS_EDGE_CONFIRMED

        # Test 6: edge check — strategy has NO real edge (identical distribution to baseline, heavy costs)
        no_edge_strategy = pd.Series(rng.normal(0.02, 0.3, size=100), index=dates)
        edge_result_negative = validator.compute_edge_vs_baseline(no_edge_strategy, baseline_returns,
                                                                    slippage_bps=50, transaction_cost_bps=50)
        print(f"No-edge case -> alpha={edge_result_negative.alpha_pct:.2f}%, status={edge_result_negative.status}")
        assert edge_result_negative.status == STATUS_NO_EDGE

        # Test 7: combined gate
        gate_good = validator.validate_before_live(good_result, edge_result_positive)
        gate_bad = validator.validate_before_live(bad_result, edge_result_negative)
        print(f"Combined gate (good calibration + positive edge): show_confidence={gate_good.safe_to_show_calibrated_confidence}, "
              f"live_edge={gate_good.safe_to_treat_as_live_edge}")
        print(f"Combined gate (bad calibration + no edge): show_confidence={gate_bad.safe_to_show_calibrated_confidence}, "
              f"live_edge={gate_bad.safe_to_treat_as_live_edge}")
        assert gate_good.safe_to_show_calibrated_confidence is True and gate_good.safe_to_treat_as_live_edge is True
        assert gate_bad.safe_to_show_calibrated_confidence is False and gate_bad.safe_to_treat_as_live_edge is False

        print("STATUS: PASS")
        logger.info("runtime_validator.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"runtime_validator.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"runtime_validator.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
