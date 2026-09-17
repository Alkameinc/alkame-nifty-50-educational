# 1. Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, cast

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import SECTOR_MAP
from execution_simulator import (
    CostScenario,
    PHASE12_COST_SCENARIOS,
    PHASE12_MANDATORY_SCENARIOS,
    STANDARD_COST_SCENARIOS,
)
from sector_provider import sector_map_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ScannerSignal:
    """
    Represents a single stock evaluation within a scanner sweep.
    Contains model predictions, conviction ranking score, realized outcome, and regime context.
    """
    timestamp: pd.Timestamp
    symbol: str
    sector: str = "Unknown"
    action: str = "BUY"  # BUY, HOLD, SELL
    predicted_class: str = "UP"  # UP, DOWN, FLAT
    raw_confidence: float = 0.5
    conviction_score: float = 0.5
    is_eligible: bool = True
    is_safe_to_trade_live: bool = True
    actual_class: str | None = None  # UP, DOWN, FLAT
    realized_forward_return_pct: float | None = None
    is_hit: bool | None = None
    regime: str = "UNKNOWN"  # BULL_TREND, BEAR_TREND, HIGH_VOLATILITY, LOW_VOLATILITY_SIDEWAYS


@dataclass
class ScannerCohortMetrics:
    """
    Independent statistical metrics evaluated for a cohort of signals
    (e.g., ALL_ELIGIBLE vs TOP_N).
    """
    cohort_name: str
    n_signals: int
    hit_rate_pct: float
    average_return_pct: float
    median_return_pct: float
    win_loss_ratio: float
    ece: float | None
    brier_score: float | None
    class_distribution: dict[str, float]
    sector_distribution: dict[str, float]
    herfindahl_sector_index: float
    cumulative_gross_return_pct: float
    cumulative_net_return_pct: float
    cost_sensitivity: dict[str, float] = field(default_factory=dict)


@dataclass
class ScannerTurnoverMetrics:
    """Turnover and rank stability metrics for top-N selections over consecutive scans."""
    top_n: int
    n_periods: int
    mean_turnover_pct: float
    mean_retention_rate_pct: float
    mean_churn_rate_pct: float
    max_turnover_pct: float
    period_turnovers: list[float] = field(default_factory=list)


@dataclass
class ScannerRegimeMetrics:
    """Comparative performance breakdown for a specific market regime."""
    regime_name: str
    n_periods: int
    all_eligible_count: int
    top_n_count: int
    all_eligible_hit_rate_pct: float
    top_n_hit_rate_pct: float
    all_eligible_avg_return_pct: float
    top_n_avg_return_pct: float
    all_eligible_net_return_pct: float
    top_n_net_return_pct: float


@dataclass
class ScannerValidationReport:
    """
    SCAN-001: Comprehensive validation report comparing all eligible signals
    against selected top-N signals across calibration, edge, turnover, costs, and regimes.
    """
    top_n: int
    n_scan_periods: int
    all_eligible_metrics: ScannerCohortMetrics
    top_n_metrics: ScannerCohortMetrics
    turnover_metrics: ScannerTurnoverMetrics
    regime_metrics: dict[str, ScannerRegimeMetrics]
    is_selection_valid: bool
    validation_status: str  # VALIDATED, DEGRADED_CALIBRATION, FRICTION_FAILURE, REGIME_UNSTABLE, INSUFFICIENT_DATA
    gate_reasons: list[str] = field(default_factory=list)
    comparison_summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calibration & Statistical Math Utilities
# ---------------------------------------------------------------------------
def compute_ece_and_brier(
    confidences: list[float] | np.ndarray,
    hits: list[bool] | np.ndarray,
    n_bins: int = 10,
) -> tuple[float | None, float | None]:
    """
    Calculates Expected Calibration Error (ECE) and Brier Score.
    Uses equal-width probability bins between 0.0 and 1.0.
    """
    confs = np.asarray(confidences, dtype=float)
    corrects = np.asarray(hits, dtype=float)

    if len(confs) == 0 or len(corrects) == 0 or len(confs) != len(corrects):
        return None, None

    # Brier Score: Mean squared error of probabilities vs binary outcome (1.0 for hit, 0.0 for miss)
    brier_score = float(np.mean((confs - corrects) ** 2))

    # Equal-width bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confs)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = float(np.mean(corrects[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            bin_weight = bin_count / n_total
            ece += bin_weight * abs(bin_acc - bin_conf)

    return round(float(ece), 4), round(float(brier_score), 4)


def compute_herfindahl_index(shares: dict[str, float]) -> float:
    """
    Computes Herfindahl-Hirschman Index (HHI) for sector concentration: sum(s_i^2).
    Ranges from 1/N (perfect diversification) to 1.0 (100% single sector concentration).
    """
    if not shares:
        return 0.0
    total = sum(shares.values())
    if total <= 0:
        return 0.0
    norm_shares = [v / total for v in shares.values()]
    return round(float(sum(s**2 for s in norm_shares)), 4)


# ---------------------------------------------------------------------------
# ScannerValidator Class
# ---------------------------------------------------------------------------
class ScannerValidator:
    """
    SCAN-001: Statistical validator for top-N scanner selection.
    
    Evaluates:
      all eligible signals vs selected top-N signals
    Across all 6 mandated dimensions:
      1. Calibration (ECE & Brier score, Winner's Curse detection)
      2. Hit rate & returns
      3. Class & sector distribution (including Herfindahl concentration)
      4. Turnover (rank stability, churn rate)
      5. Cost sensitivity (net return decay under Phase 12 stress scenarios)
      6. Regime sensitivity (BULL_TREND, BEAR_TREND, HIGH_VOLATILITY, LOW_VOLATILITY_SIDEWAYS)
    """

    def __init__(
        self,
        cost_scenarios: dict[str, CostScenario] | None = None,
        ece_degradation_threshold: float = 0.10,
        max_acceptable_top_n_ece: float = 0.20,
    ):
        self.cost_scenarios = cost_scenarios or PHASE12_COST_SCENARIOS
        self.ece_degradation_threshold = ece_degradation_threshold
        self.max_acceptable_top_n_ece = max_acceptable_top_n_ece

    def detect_regime(
        self,
        index_slice: pd.DataFrame | None = None,
        lookback_bars: int = 20,
    ) -> str:
        """
        Detects market regime from index price history.
        Returns one of: BULL_TREND, BEAR_TREND, HIGH_VOLATILITY, LOW_VOLATILITY_SIDEWAYS.
        """
        if index_slice is None or len(index_slice) < max(5, lookback_bars // 2):
            return "UNKNOWN"

        try:
            close = index_slice["Close"] if "Close" in index_slice.columns else index_slice.iloc[:, 0]
            recent = close.tail(lookback_bars)
            if len(recent) < 2:
                return "UNKNOWN"

            ret = float((recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0])
            pct_changes = recent.pct_change().dropna()
            vol = float(pct_changes.std() * np.sqrt(252 * 75)) if len(pct_changes) > 1 else 0.0  # Approx intraday ann vol

            # Volatility check: High annualized intraday vol > 28%
            if vol > 0.28:
                return "HIGH_VOLATILITY"
            if ret > 0.004:
                return "BULL_TREND"
            if ret < -0.004:
                return "BEAR_TREND"
            return "LOW_VOLATILITY_SIDEWAYS"
        except Exception as e:
            logger.debug(f"Failed detecting regime: {e}")
            return "UNKNOWN"

    def compute_cohort_metrics(
        self,
        signals: list[ScannerSignal],
        cohort_name: str,
        cost_scenarios: dict[str, CostScenario] | None = None,
    ) -> ScannerCohortMetrics:
        """
        Measures calibration, hit rate, distributions, and Phase 12 cost sensitivity
        independently for a given cohort of signals.
        """
        scenarios = cost_scenarios or self.cost_scenarios
        n_signals = len(signals)

        if n_signals == 0:
            return ScannerCohortMetrics(
                cohort_name=cohort_name,
                n_signals=0,
                hit_rate_pct=0.0,
                average_return_pct=0.0,
                median_return_pct=0.0,
                win_loss_ratio=0.0,
                ece=None,
                brier_score=None,
                class_distribution={},
                sector_distribution={},
                herfindahl_sector_index=0.0,
                cumulative_gross_return_pct=0.0,
                cumulative_net_return_pct=0.0,
                cost_sensitivity={},
            )

        # 1. Hit rate and realized returns
        valid_hits = [s.is_hit for s in signals if s.is_hit is not None]
        hit_rate = float((sum(valid_hits) / len(valid_hits)) * 100.0) if valid_hits else 0.0

        returns = [s.realized_forward_return_pct for s in signals if s.realized_forward_return_pct is not None]
        avg_ret = float(np.mean(returns)) if returns else 0.0
        med_ret = float(np.median(returns)) if returns else 0.0

        pos_rets = [r for r in returns if r > 0]
        neg_rets = [abs(r) for r in returns if r < 0]
        win_loss_ratio = (
            round(float(np.mean(pos_rets) / np.mean(neg_rets)), 3)
            if pos_rets and neg_rets and np.mean(neg_rets) > 0
            else (99.0 if pos_rets else 0.0)
        )

        # 2. Calibration: ECE & Brier
        eval_confs = [s.raw_confidence for s in signals if s.is_hit is not None]
        eval_hits = [s.is_hit for s in signals if s.is_hit is not None]
        ece, brier = compute_ece_and_brier(eval_confs, eval_hits)

        # 3. Class distribution
        classes = [s.predicted_class for s in signals]
        class_dist = {cls: round(float(classes.count(cls) / n_signals), 4) for cls in set(classes)}

        # 4. Sector distribution and concentration
        sectors = [s.sector for s in signals if s.sector and s.sector != "Unknown"]
        if not sectors:
            # Fallback to symbol sector mapping if signal didn't populate sector
            sectors = [sector_map_provider.get_sector(s.symbol, as_of=s.timestamp) for s in signals]
        sec_dist = {sec: round(float(sectors.count(sec) / len(sectors)), 4) for sec in set(sectors)} if sectors else {}
        hhi = compute_herfindahl_index(sec_dist)

        # 5. Returns and Cost Sensitivity across Phase 12 Scenarios
        cum_gross = float(np.sum(returns)) if returns else 0.0

        cost_sensitivity: dict[str, float] = {}
        base_scenario = scenarios.get("BASE", STANDARD_COST_SCENARIOS["BASE"])
        base_cost_pct = base_scenario.total_cost_pct

        # Apply friction per trade: net return = sum(gross_i - cost_pct)
        cum_net_base = float(sum(r - base_cost_pct for r in returns)) if returns else 0.0

        # Mandated Phase 12 Scenarios: BASE, +25% cost, +50% cost, +100% cost, HIGH_SLIPPAGE, LOW_LIQUIDITY
        scenario_list = PHASE12_MANDATORY_SCENARIOS if all(s in scenarios for s in PHASE12_MANDATORY_SCENARIOS) else list(scenarios.keys())
        for sc_name in scenario_list:
            if sc_name in scenarios:
                sc = scenarios[sc_name]
                sc_net = float(sum(r - sc.total_cost_pct for r in returns)) if returns else 0.0
                cost_sensitivity[sc_name] = round(sc_net, 2)

        return ScannerCohortMetrics(
            cohort_name=cohort_name,
            n_signals=n_signals,
            hit_rate_pct=round(hit_rate, 2),
            average_return_pct=round(avg_ret, 3),
            median_return_pct=round(med_ret, 3),
            win_loss_ratio=win_loss_ratio,
            ece=ece,
            brier_score=brier,
            class_distribution=class_dist,
            sector_distribution=sec_dist,
            herfindahl_sector_index=hhi,
            cumulative_gross_return_pct=round(cum_gross, 2),
            cumulative_net_return_pct=round(cum_net_base, 2),
            cost_sensitivity=cost_sensitivity,
        )

    def compute_turnover(
        self,
        top_n_history: list[list[str]],
        top_n: int = 5,
    ) -> ScannerTurnoverMetrics:
        """
        Computes portfolio turnover, churn rate, and rank stability across consecutive scan periods.
        """
        n_periods = len(top_n_history)
        if n_periods <= 1:
            return ScannerTurnoverMetrics(
                top_n=top_n,
                n_periods=n_periods,
                mean_turnover_pct=0.0,
                mean_retention_rate_pct=100.0,
                mean_churn_rate_pct=0.0,
                max_turnover_pct=0.0,
                period_turnovers=[],
            )

        period_turnovers = []
        for i in range(1, n_periods):
            prev_set = set(top_n_history[i - 1])
            curr_set = set(top_n_history[i])
            if not prev_set and not curr_set:
                period_turnovers.append(0.0)
                continue

            # Fraction of positions replaced (turnover rate)
            max_size = max(len(prev_set), len(curr_set), 1)
            disjoint_new = len(curr_set - prev_set)
            turnover_fraction = (disjoint_new / max_size) * 100.0
            period_turnovers.append(turnover_fraction)

        mean_turnover = float(np.mean(period_turnovers)) if period_turnovers else 0.0
        max_turnover = float(np.max(period_turnovers)) if period_turnovers else 0.0
        mean_retention = max(0.0, 100.0 - mean_turnover)

        return ScannerTurnoverMetrics(
            top_n=top_n,
            n_periods=n_periods,
            mean_turnover_pct=round(mean_turnover, 2),
            mean_retention_rate_pct=round(mean_retention, 2),
            mean_churn_rate_pct=round(mean_turnover, 2),
            max_turnover_pct=round(max_turnover, 2),
            period_turnovers=[round(t, 2) for t in period_turnovers],
        )

    def compute_regime_sensitivity(
        self,
        eligible_by_regime: dict[str, list[ScannerSignal]],
        top_n_by_regime: dict[str, list[ScannerSignal]],
        cost_scenarios: dict[str, CostScenario] | None = None,
    ) -> dict[str, ScannerRegimeMetrics]:
        """
        Computes comparative hit rate and returns broken down across all active market regimes.
        """
        scenarios = cost_scenarios or self.cost_scenarios
        base_cost = scenarios.get("BASE", STANDARD_COST_SCENARIOS["BASE"]).total_cost_pct

        all_regimes = sorted(set(list(eligible_by_regime.keys()) + list(top_n_by_regime.keys())))
        results: dict[str, ScannerRegimeMetrics] = {}

        for regime in all_regimes:
            el_sigs = eligible_by_regime.get(regime, [])
            top_sigs = top_n_by_regime.get(regime, [])

            el_hits = [s.is_hit for s in el_sigs if s.is_hit is not None]
            el_hit_rate = float((sum(el_hits) / len(el_hits)) * 100.0) if el_hits else 0.0
            el_rets = [s.realized_forward_return_pct for s in el_sigs if s.realized_forward_return_pct is not None]
            el_avg_ret = float(np.mean(el_rets)) if el_rets else 0.0
            el_net_ret = float(sum(r - base_cost for r in el_rets)) if el_rets else 0.0

            top_hits = [s.is_hit for s in top_sigs if s.is_hit is not None]
            top_hit_rate = float((sum(top_hits) / len(top_hits)) * 100.0) if top_hits else 0.0
            top_rets = [s.realized_forward_return_pct for s in top_sigs if s.realized_forward_return_pct is not None]
            top_avg_ret = float(np.mean(top_rets)) if top_rets else 0.0
            top_net_ret = float(sum(r - base_cost for r in top_rets)) if top_rets else 0.0

            # Estimate number of distinct periods in this regime
            timestamps = {s.timestamp for s in (el_sigs + top_sigs)}

            results[regime] = ScannerRegimeMetrics(
                regime_name=regime,
                n_periods=len(timestamps),
                all_eligible_count=len(el_sigs),
                top_n_count=len(top_sigs),
                all_eligible_hit_rate_pct=round(el_hit_rate, 2),
                top_n_hit_rate_pct=round(top_hit_rate, 2),
                all_eligible_avg_return_pct=round(el_avg_ret, 3),
                top_n_avg_return_pct=round(top_avg_ret, 3),
                all_eligible_net_return_pct=round(el_net_ret, 2),
                top_n_net_return_pct=round(top_net_ret, 2),
            )

        return results

    def evaluate_selection_gate(
        self,
        all_eligible_metrics: ScannerCohortMetrics,
        top_n_metrics: ScannerCohortMetrics,
        turnover_metrics: ScannerTurnoverMetrics,
        regime_metrics: dict[str, ScannerRegimeMetrics],
    ) -> tuple[bool, str, list[str]]:
        """
        Evaluates strict statistical and economic validation gates:
        1. Winner's Curse / Calibration Gate
        2. Friction & Turnover Gate (positive gross, negative net)
        3. Regime Resilience Gate
        """
        gate_reasons: list[str] = []
        is_valid = True
        status = "VALIDATED"

        if top_n_metrics.n_signals == 0 or all_eligible_metrics.n_signals == 0:
            return False, "INSUFFICIENT_DATA", ["Insufficient signal count for scanner validation."]

        # 1. Winner's Curse / Calibration Gate
        # Never assume individual-model metrics remain valid after top-N selection.
        if top_n_metrics.ece is not None:
            if top_n_metrics.ece > self.max_acceptable_top_n_ece:
                is_valid = False
                status = "DEGRADED_CALIBRATION"
                gate_reasons.append(
                    f"Top-N selection exhibits severe miscalibration: ECE={top_n_metrics.ece:.3f} "
                    f"> max acceptable threshold {self.max_acceptable_top_n_ece:.3f} (Winner's Curse)."
                )
            elif (
                all_eligible_metrics.ece is not None
                and (top_n_metrics.ece - all_eligible_metrics.ece) > self.ece_degradation_threshold
            ):
                is_valid = False
                status = "DEGRADED_CALIBRATION"
                gate_reasons.append(
                    f"Top-N selection degrades calibration significantly: Top-N ECE ({top_n_metrics.ece:.3f}) "
                    f"exceeds eligible baseline ({all_eligible_metrics.ece:.3f}) by "
                    f"{top_n_metrics.ece - all_eligible_metrics.ece:.3f} > {self.ece_degradation_threshold:.3f}."
                )

        # 2. Friction & Turnover Gate: Never report only gross performance when evaluating viability
        if top_n_metrics.cumulative_gross_return_pct > 0.0 and top_n_metrics.cumulative_net_return_pct <= 0.0:
            is_valid = False
            if status == "VALIDATED":
                status = "FRICTION_FAILURE"
            gate_reasons.append(
                f"Top-N selection fails live viability: gross return is positive "
                f"(+{top_n_metrics.cumulative_gross_return_pct:.2f}%) but net return is non-positive "
                f"({top_n_metrics.cumulative_net_return_pct:.2f}%) after turnover ({turnover_metrics.mean_turnover_pct:.1f}%) "
                f"and transaction friction."
            )
        elif top_n_metrics.cumulative_net_return_pct <= 0.0:
            is_valid = False
            if status == "VALIDATED":
                status = "FRICTION_FAILURE"
            gate_reasons.append(
                f"Top-N selection produces non-positive net return ({top_n_metrics.cumulative_net_return_pct:.2f}%) "
                f"under BASE cost scenario."
            )

        # 3. Regime Resilience Gate: Detect catastrophic collapse in any major market regime
        for reg_name, reg_met in regime_metrics.items():
            if reg_name in ("BULL_TREND", "BEAR_TREND", "HIGH_VOLATILITY") and reg_met.top_n_count >= 5:
                if reg_met.top_n_hit_rate_pct < 25.0 and reg_met.top_n_net_return_pct < -5.0:
                    is_valid = False
                    if status == "VALIDATED":
                        status = "REGIME_UNSTABLE"
                    gate_reasons.append(
                        f"Top-N selection is unstable in {reg_name}: hit rate {reg_met.top_n_hit_rate_pct:.1f}% "
                        f"and net return {reg_met.top_n_net_return_pct:.2f}% represent catastrophic drawdown."
                    )

        if is_valid:
            gate_reasons.append(
                f"Scanner Top-{turnover_metrics.top_n} selection validated: "
                f"Hit Rate={top_n_metrics.hit_rate_pct:.1f}%, Net P&L=+{top_n_metrics.cumulative_net_return_pct:.2f}%, "
                f"ECE={top_n_metrics.ece if top_n_metrics.ece is not None else 0.0:.3f}, "
                f"Mean Turnover={turnover_metrics.mean_turnover_pct:.1f}%."
            )

        return is_valid, status, gate_reasons

    def evaluate_scan_history(
        self,
        scan_periods: list[list[ScannerSignal] | dict[str, Any]],
        index_df: pd.DataFrame | None = None,
        top_n: int = 5,
        cost_scenarios: dict[str, CostScenario] | None = None,
    ) -> ScannerValidationReport:
        """
        SCAN-001: Evaluates a series of chronological scanner sweeps.
        
        For each sweep:
          - Gathers all eligible signals meeting live safety criteria
          - Ranks by conviction score descending and slices the top-N
          - Tracks turnover between consecutive top-N portfolios
          - Classifies regime using index prices
        Produces independent statistical metrics for both ALL_ELIGIBLE and TOP_N.
        """
        all_eligible_signals: list[ScannerSignal] = []
        top_n_signals: list[ScannerSignal] = []
        top_n_symbol_history: list[list[str]] = []

        eligible_by_regime: dict[str, list[ScannerSignal]] = {}
        top_n_by_regime: dict[str, list[ScannerSignal]] = {}

        for period in scan_periods:
            period_signals: list[ScannerSignal] = []
            if isinstance(period, dict):
                raw_sigs = period.get("signals", [])
            else:
                raw_sigs = period

            for s in raw_sigs:
                if isinstance(s, ScannerSignal):
                    period_signals.append(s)

            if not period_signals:
                continue

            period_ts = period_signals[0].timestamp

            # Determine regime for this period if index_df is provided
            period_regime = "UNKNOWN"
            if index_df is not None and not index_df.empty:
                idx_slice = index_df.loc[:period_ts]
                period_regime = self.detect_regime(idx_slice)

            # Filter for eligible signals (actionable & live-safe)
            eligible = [
                s for s in period_signals
                if s.is_eligible and s.is_safe_to_trade_live and s.action == "BUY"
            ]

            # Populate sector and regime on each signal
            for s in eligible:
                if s.regime == "UNKNOWN" and period_regime != "UNKNOWN":
                    s.regime = period_regime
                if not s.sector or s.sector == "Unknown":
                    s.sector = sector_map_provider.get_sector(s.symbol, as_of=s.timestamp)

            # Sort eligible by conviction score descending to get Top-N
            eligible_sorted = sorted(eligible, key=lambda x: x.conviction_score, reverse=True)
            top_n_selected = eligible_sorted[:top_n]

            all_eligible_signals.extend(eligible)
            top_n_signals.extend(top_n_selected)

            # Track top-N symbol history for turnover calculation
            top_n_symbols = [s.symbol for s in top_n_selected]
            top_n_symbol_history.append(top_n_symbols)

            # Group by regime
            for s in eligible:
                reg = s.regime
                eligible_by_regime.setdefault(reg, []).append(s)
            for s in top_n_selected:
                reg = s.regime
                top_n_by_regime.setdefault(reg, []).append(s)

        # Compute cohort metrics independently
        all_eligible_metrics = self.compute_cohort_metrics(
            all_eligible_signals, cohort_name="ALL_ELIGIBLE", cost_scenarios=cost_scenarios
        )
        top_n_metrics = self.compute_cohort_metrics(
            top_n_signals, cohort_name=f"TOP_{top_n}", cost_scenarios=cost_scenarios
        )

        # Compute turnover metrics
        turnover_metrics = self.compute_turnover(top_n_symbol_history, top_n=top_n)

        # Compute regime sensitivity
        regime_metrics = self.compute_regime_sensitivity(
            eligible_by_regime, top_n_by_regime, cost_scenarios=cost_scenarios
        )

        # Evaluate gating
        is_valid, status, gate_reasons = self.evaluate_selection_gate(
            all_eligible_metrics, top_n_metrics, turnover_metrics, regime_metrics
        )

        # Comparison summary dictionary
        comparison_summary = {
            "hit_rate_delta_pct": round(top_n_metrics.hit_rate_pct - all_eligible_metrics.hit_rate_pct, 2),
            "avg_return_delta_pct": round(top_n_metrics.average_return_pct - all_eligible_metrics.average_return_pct, 3),
            "ece_delta": (
                round(top_n_metrics.ece - all_eligible_metrics.ece, 4)
                if top_n_metrics.ece is not None and all_eligible_metrics.ece is not None
                else None
            ),
            "herfindahl_concentration_delta": round(
                top_n_metrics.herfindahl_sector_index - all_eligible_metrics.herfindahl_sector_index, 4
            ),
            "mean_turnover_pct": turnover_metrics.mean_turnover_pct,
            "cost_sensitivity_top_n": top_n_metrics.cost_sensitivity,
        }

        return ScannerValidationReport(
            top_n=top_n,
            n_scan_periods=len(scan_periods),
            all_eligible_metrics=all_eligible_metrics,
            top_n_metrics=top_n_metrics,
            turnover_metrics=turnover_metrics,
            regime_metrics=regime_metrics,
            is_selection_valid=is_valid,
            validation_status=status,
            gate_reasons=gate_reasons,
            comparison_summary=comparison_summary,
        )


if __name__ == "__main__":
    print("ScannerValidator self-test...")
    validator = ScannerValidator()

    # Generate synthetic scan history
    np.random.seed(42)
    periods = []
    base_ts = pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata")
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "LT", "SBIN", "BHARTIARTL", "KOTAKBANK"]

    for t in range(10):
        period_ts = base_ts + pd.Timedelta(f"{t}h")
        sigs = []
        for sym in symbols:
            conf = float(np.random.uniform(0.55, 0.95))
            score = conf * float(np.random.uniform(0.7, 1.0))
            is_hit = bool(np.random.rand() < 0.60)
            ret = float(np.random.normal(0.003, 0.008) if is_hit else -np.random.normal(0.002, 0.006))
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
                    actual_class="UP" if is_hit else "FLAT",
                    realized_forward_return_pct=ret,
                    is_hit=is_hit,
                    regime="BULL_TREND" if t < 5 else "HIGH_VOLATILITY",
                )
            )
        periods.append(sigs)

    report = validator.evaluate_scan_history(periods, top_n=3)
    print(f"Validation Status: {report.validation_status}")
    print(f"Is Valid: {report.is_selection_valid}")
    print(f"All Eligible Hit Rate: {report.all_eligible_metrics.hit_rate_pct}%")
    print(f"Top-3 Hit Rate: {report.top_n_metrics.hit_rate_pct}%")
    print(f"Mean Turnover: {report.turnover_metrics.mean_turnover_pct}%")
    print(f"Gate Reasons: {report.gate_reasons}")
    assert report.all_eligible_metrics.n_signals == 100
    assert report.top_n_metrics.n_signals == 30
    assert report.turnover_metrics.n_periods == 10
    print("ScannerValidator self-test PASSED successfully!")
