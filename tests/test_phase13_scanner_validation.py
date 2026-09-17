# 1. Standard library imports
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from config import HORIZON_INTRADAY
from execution_simulator import PHASE12_MANDATORY_SCENARIOS
from predictor import MultiHorizonSignal, PredictionSignal
from scanner import OpportunityScanner, ScanResult, ScanSummary
from scanner_validator import (
    ScannerCohortMetrics,
    ScannerRegimeMetrics,
    ScannerSignal,
    ScannerTurnoverMetrics,
    ScannerValidationReport,
    ScannerValidator,
    compute_ece_and_brier,
    compute_herfindahl_index,
)
from sector_provider import sector_map_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_mock_scan_period(
    timestamp: pd.Timestamp,
    symbols: list[str],
    conf_range: tuple[float, float] = (0.60, 0.90),
    hit_prob: float = 0.55,
    avg_gain: float = 0.005,
    avg_loss: float = 0.003,
    regime: str = "BULL_TREND",
    seed: int = 42,
) -> list[ScannerSignal]:
    """Generates a synthetic list of ScannerSignals for a single scan period."""
    rng = np.random.default_rng(seed)
    signals = []
    for sym in symbols:
        conf = float(rng.uniform(conf_range[0], conf_range[1]))
        score = conf * float(rng.uniform(0.75, 1.0))
        is_hit = bool(rng.random() < hit_prob)
        ret = float(rng.normal(avg_gain, 0.002)) if is_hit else float(-abs(rng.normal(avg_loss, 0.002)))
        signals.append(
            ScannerSignal(
                timestamp=timestamp,
                symbol=sym,
                sector=sector_map_provider.get_sector(sym, as_of=timestamp),
                action="BUY",
                predicted_class="UP",
                raw_confidence=conf,
                conviction_score=score,
                is_eligible=True,
                is_safe_to_trade_live=True,
                actual_class="UP" if is_hit else "DOWN",
                realized_forward_return_pct=ret,
                is_hit=is_hit,
                regime=regime,
            )
        )
    return signals


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_scanner_cohort_metrics_all_eligible_vs_top_n():
    """
    SCAN-001: The scanner's top-N selection is a separate statistical layer.
    Evaluate all eligible signals vs selected top-N signals and measure
    calibration, hit rate, class/sector distribution, turnover, and costs separately.
    """
    validator = ScannerValidator()
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "LT", "SBIN", "BHARTIARTL", "KOTAKBANK"]
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")

    periods = []
    for t in range(8):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        periods.append(_build_mock_scan_period(period_ts, symbols, hit_prob=0.60, seed=100 + t))

    report = validator.evaluate_scan_history(periods, top_n=3)

    assert isinstance(report, ScannerValidationReport)
    assert report.top_n == 3
    assert report.n_scan_periods == 8

    # Check that ALL_ELIGIBLE and TOP_3 are computed separately
    assert report.all_eligible_metrics.cohort_name == "ALL_ELIGIBLE"
    assert report.top_n_metrics.cohort_name == "TOP_3"
    assert report.all_eligible_metrics.n_signals == 80  # 8 periods * 10 symbols
    assert report.top_n_metrics.n_signals == 24  # 8 periods * 3 symbols

    # Metrics exist and are populated
    assert report.all_eligible_metrics.hit_rate_pct > 0
    assert report.top_n_metrics.hit_rate_pct > 0
    assert report.all_eligible_metrics.ece is not None
    assert report.top_n_metrics.ece is not None
    assert report.all_eligible_metrics.brier_score is not None
    assert report.top_n_metrics.brier_score is not None

    # Comparison summary exists
    assert "hit_rate_delta_pct" in report.comparison_summary
    assert "avg_return_delta_pct" in report.comparison_summary
    assert "mean_turnover_pct" in report.comparison_summary


def test_winners_curse_calibration_detection():
    """
    SCAN-001: Detect Winner's Curse.
    When ranking selects extreme-confidence items that turn out to be overconfident noise,
    top-N calibration must degrade and fail the calibration gate (DEGRADED_CALIBRATION).
    """
    validator = ScannerValidator(max_acceptable_top_n_ece=0.20, ece_degradation_threshold=0.10)
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "LT", "SBIN", "BHARTIARTL", "KOTAKBANK"]

    periods = []
    # Design an adversarial distribution where:
    # - Lower conviction items (conf ~0.55-0.65) have 60% hit rate (well-calibrated)
    # - Top conviction items (conf ~0.95) are overconfident noise with only 25% hit rate!
    for t in range(10):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        sigs = []
        for idx, sym in enumerate(symbols):
            if idx < 3:
                # Top 3 by score, but overconfident and wrong
                conf = 0.95
                score = 0.95
                is_hit = bool(t % 4 == 0)  # only 25% hit rate
                ret = 0.005 if is_hit else -0.005
            else:
                # Rest are moderately confident and mostly correct
                conf = 0.60
                score = 0.50
                is_hit = bool(t % 5 != 0)  # 80% hit rate
                ret = 0.004 if is_hit else -0.003

            sigs.append(
                ScannerSignal(
                    timestamp=period_ts,
                    symbol=sym,
                    sector=sector_map_provider.get_sector(sym, as_of=period_ts),
                    action="BUY",
                    predicted_class="UP",
                    raw_confidence=conf,
                    conviction_score=score,
                    is_eligible=True,
                    is_safe_to_trade_live=True,
                    actual_class="UP" if is_hit else "DOWN",
                    realized_forward_return_pct=ret,
                    is_hit=is_hit,
                    regime="BULL_TREND",
                )
            )
        periods.append(sigs)

    report = validator.evaluate_scan_history(periods, top_n=3)

    # Top-3 should have very high ECE (predicted 0.95 vs actual ~0.25 -> ECE ~ 0.70!)
    assert report.top_n_metrics.ece is not None
    assert report.top_n_metrics.ece > 0.20, f"Expected high ECE due to Winner's Curse, got {report.top_n_metrics.ece}"
    assert report.is_selection_valid is False
    assert report.validation_status == "DEGRADED_CALIBRATION"
    assert any("Winner's Curse" in r or "miscalibration" in r or "ECE" in r for r in report.gate_reasons)


def test_class_and_sector_distribution_and_concentration():
    """SCAN-001: Measure class distribution, sector distribution, and Herfindahl concentration index."""
    shares_diversified = {"IT": 0.20, "Banking": 0.20, "Energy": 0.20, "FMCG": 0.20, "Auto": 0.20}
    hhi_div = compute_herfindahl_index(shares_diversified)
    assert pytest.approx(hhi_div, abs=0.01) == 0.20

    shares_concentrated = {"Banking": 0.80, "IT": 0.20}
    hhi_conc = compute_herfindahl_index(shares_concentrated)
    assert pytest.approx(hhi_conc, abs=0.01) == 0.68

    shares_monopoly = {"Banking": 1.0}
    hhi_mono = compute_herfindahl_index(shares_monopoly)
    assert pytest.approx(hhi_mono, abs=0.01) == 1.0

    # Test within ScannerValidator
    validator = ScannerValidator()
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")
    
    # 5 stocks: 4 from Banking, 1 from IT
    banking_syms = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"]
    it_sym = ["TCS"]
    sigs = []
    for s in banking_syms:
        sigs.append(
            ScannerSignal(
                timestamp=base_ts,
                symbol=s,
                sector="Banking",
                action="BUY",
                predicted_class="UP",
                conviction_score=0.9,
                is_hit=True,
                realized_forward_return_pct=0.01,
            )
        )
    sigs.append(
        ScannerSignal(
            timestamp=base_ts,
            symbol="TCS",
            sector="IT",
            action="BUY",
            predicted_class="UP",
            conviction_score=0.5,
            is_hit=True,
            realized_forward_return_pct=0.01,
        )
    )

    metrics = validator.compute_cohort_metrics(sigs, cohort_name="TEST_COHORT")
    assert metrics.sector_distribution["Banking"] == 0.8
    assert metrics.sector_distribution["IT"] == 0.2
    assert metrics.herfindahl_sector_index == pytest.approx(0.68, abs=0.01)
    assert metrics.class_distribution["UP"] == 1.0


def test_turnover_and_churn_metrics():
    """SCAN-001: Measure rank stability, turnover, and churn rate between consecutive scans."""
    validator = ScannerValidator()

    # Case 1: Identical selections across 3 periods (0% turnover)
    static_history = [
        ["RELIANCE", "TCS", "INFY"],
        ["RELIANCE", "TCS", "INFY"],
        ["RELIANCE", "TCS", "INFY"],
    ]
    turnover_static = validator.compute_turnover(static_history, top_n=3)
    assert turnover_static.mean_turnover_pct == 0.0
    assert turnover_static.mean_retention_rate_pct == 100.0
    assert turnover_static.mean_churn_rate_pct == 0.0

    # Case 2: Completely disjoint selections (100% turnover)
    churn_history = [
        ["RELIANCE", "TCS", "INFY"],
        ["HDFCBANK", "ICICIBANK", "SBIN"],
        ["ITC", "LT", "BHARTIARTL"],
    ]
    turnover_churn = validator.compute_turnover(churn_history, top_n=3)
    assert turnover_churn.mean_turnover_pct == 100.0
    assert turnover_churn.mean_retention_rate_pct == 0.0
    assert turnover_churn.mean_churn_rate_pct == 100.0

    # Case 3: 1 symbol changes per period out of 3 (33.33% turnover)
    partial_history = [
        ["RELIANCE", "TCS", "INFY"],
        ["RELIANCE", "TCS", "HDFCBANK"],  # INFY replaced by HDFCBANK
        ["RELIANCE", "ICICIBANK", "HDFCBANK"],  # TCS replaced by ICICIBANK
    ]
    turnover_partial = validator.compute_turnover(partial_history, top_n=3)
    assert pytest.approx(turnover_partial.mean_turnover_pct, abs=0.1) == 33.33
    assert pytest.approx(turnover_partial.mean_retention_rate_pct, abs=0.1) == 66.67


def test_cost_sensitivity_across_phase12_tiers():
    """SCAN-001: Evaluate cost sensitivity across all 6 Phase 12 stress scenarios."""
    validator = ScannerValidator()
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")

    periods = []
    for t in range(5):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        periods.append(_build_mock_scan_period(period_ts, symbols, hit_prob=0.70, avg_gain=0.008, seed=50 + t))

    report = validator.evaluate_scan_history(periods, top_n=3)
    top_costs = report.top_n_metrics.cost_sensitivity

    for sc in PHASE12_MANDATORY_SCENARIOS:
        assert sc in top_costs, f"Missing Phase 12 scenario {sc} in top_n cost sensitivity"

    # Monotonic cost decay: BASE >= +25% cost >= +50% cost >= +100% cost
    assert top_costs["BASE"] >= top_costs["+25% cost"]
    assert top_costs["+25% cost"] >= top_costs["+50% cost"]
    assert top_costs["+50% cost"] >= top_costs["+100% cost"]


def test_friction_failure_gate():
    """
    SCAN-001: Critical Invariant:
    If top-N selection produces positive gross return but negative net return after
    friction and turnover, it must strictly fail validation (is_selection_valid = False).
    """
    validator = ScannerValidator()
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")

    # Generate trades with small positive gross gains (+0.05% per trade),
    # which will be completely consumed by round-trip transaction costs (~0.17% in BASE)
    periods = []
    for t in range(10):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        periods.append(
            _build_mock_scan_period(
                period_ts, symbols, hit_prob=0.85, avg_gain=0.05, avg_loss=0.01, seed=200 + t
            )
        )

    report = validator.evaluate_scan_history(periods, top_n=3)

    assert report.top_n_metrics.cumulative_gross_return_pct > 0.0, "Gross return should be positive"
    assert report.top_n_metrics.cumulative_net_return_pct <= 0.0, "Net return should be negative after costs"
    assert report.is_selection_valid is False
    assert report.validation_status == "FRICTION_FAILURE"
    assert any("live viability" in r or "net return is non-positive" in r for r in report.gate_reasons)


def test_regime_sensitivity_evaluation():
    """SCAN-001: Verify comparative metrics across market regimes."""
    validator = ScannerValidator()
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")

    # Periods across 3 distinct regimes
    periods = []
    # 1. BULL_TREND
    for t in range(4):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        periods.append(_build_mock_scan_period(period_ts, symbols, hit_prob=0.80, regime="BULL_TREND", seed=300 + t))
    # 2. BEAR_TREND
    for t in range(4, 8):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        periods.append(_build_mock_scan_period(period_ts, symbols, hit_prob=0.30, regime="BEAR_TREND", seed=300 + t))
    # 3. HIGH_VOLATILITY
    for t in range(8, 12):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        periods.append(_build_mock_scan_period(period_ts, symbols, hit_prob=0.50, regime="HIGH_VOLATILITY", seed=300 + t))

    report = validator.evaluate_scan_history(periods, top_n=3)

    reg_metrics = report.regime_metrics
    assert "BULL_TREND" in reg_metrics
    assert "BEAR_TREND" in reg_metrics
    assert "HIGH_VOLATILITY" in reg_metrics

    bull = reg_metrics["BULL_TREND"]
    bear = reg_metrics["BEAR_TREND"]

    # Bull regime should have higher hit rate than Bear regime
    assert bull.top_n_hit_rate_pct > bear.top_n_hit_rate_pct
    assert bull.top_n_count == 12  # 4 periods * 3 top_n
    assert bear.top_n_count == 12


def test_opportunity_scanner_validation_integration():
    """SCAN-001: Verify OpportunityScanner.validate_selection() integration."""
    mock_predictor = MagicMock()
    mock_fetcher = MagicMock()
    scanner = OpportunityScanner(predictor=mock_predictor, data_fetcher=mock_fetcher)

    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")

    periods = []
    for t in range(5):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        periods.append(_build_mock_scan_period(period_ts, symbols, hit_prob=0.65, seed=400 + t))

    report = scanner.validate_selection(periods, top_n=3)
    assert isinstance(report, ScannerValidationReport)
    assert report.top_n == 3
    assert report.all_eligible_metrics.n_signals == 25
    assert report.top_n_metrics.n_signals == 15


def test_backwards_compatibility():
    """SCAN-001: Ensure existing OpportunityScanner scan and scan_with_summary work unchanged."""
    mock_fetcher = MagicMock()
    mock_predictor = MagicMock()

    scanner = OpportunityScanner(mock_predictor, mock_fetcher)

    df = pd.DataFrame({"Close": [100.0, 101.0]})
    mock_fetcher.fetch_ohlcv.return_value = df
    mock_fetcher.check_staleness.return_value = False

    sig_1d = PredictionSignal(
        timestamp=pd.Timestamp("2026-01-01 09:15", tz="UTC"),
        symbol="TCS",
        horizon="1D",
        action="BUY",
        model_predicted_class="UP",
        raw_confidence=0.8,
        calibrated_confidence=0.8,
        target_price=110.0,
        stop_loss=90.0,
        suppressed=False,
        model_version="1.0",
        feature_version="1.0",
        risk_adjusted_confidence=0.8,
        agreement_fraction=1.0,
        downside_summary=[],
        upside_summary=[],
        is_safe_to_trade_live=True,
    )

    multi = MultiHorizonSignal(
        timestamp=pd.Timestamp("2026-01-01 09:15", tz="UTC"),
        symbol="TCS",
        primary_action="BUY",
        primary_horizon="1D",
        signals={"1D": sig_1d},
        reasoning=[],
    )
    mock_predictor.generate_multi_horizon_signal.return_value = multi

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("scanner.NIFTY50_SYMBOLS", ["TCS"])
        results = scanner.scan()
        assert len(results) == 1
        assert results[0].symbol == "TCS"
        assert hasattr(results[0], "sector")
        assert results[0].sector == "IT"

        summary = scanner.scan_with_summary()
        assert summary.scanned == 1
        assert summary.actionable == 1
        assert summary.opportunities[0].sector == "IT"
