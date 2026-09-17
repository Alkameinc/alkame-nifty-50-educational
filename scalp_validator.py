from __future__ import annotations

# 1. Standard library imports
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, cast

# 2. Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 3. Local imports
from config import (
    HORIZON_CONFIG,
    HORIZON_SCALP,
    SECTOR_MAP,
)
from sector_provider import sector_map_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ScalpLatencyConfig:
    """
    SCALP-001: Explicit latency configuration modeling order queue delay,
    network latency, and adverse selection price drag.
    """
    latency_ms: float = 100.0  # Execution delay in milliseconds
    base_fill_bps: float = 2.0  # Base slippage on market fill
    adverse_selection_bps: float = 4.0  # Additional adverse fill penalty on stop-outs
    execution_delay_ms: float | None = None

    def __post_init__(self) -> None:
        if self.execution_delay_ms is not None:
            self.latency_ms = self.execution_delay_ms

    @property
    def latency_drag_bps(self) -> float:
        """Latency drag: 1 bp per 100ms."""
        return (self.latency_ms / 100.0) * 1.0


@dataclass
class ScalpCostScenario:
    """
    SCALP-001: Scalping-specific cost scenario modeling tight spreads,
    high frequency turnover, and toxic flow on adverse fills.
    """
    name: str
    spread_bps: float = 2.0
    slippage_bps: float = 2.5
    brokerage_bps: float = 1.5
    stt_bps: float = 1.25  # Intraday equity STT
    exchange_charges_bps: float = 0.345
    gst_pct: float = 18.0
    stamp_duty_bps: float = 0.3
    sebi_turnover_bps: float = 0.1
    latency_ms: float = 50.0
    adverse_stop_bps: float = 0.0

    @property
    def round_trip_cost_pct(self) -> float:
        """Total round-trip friction as a percentage including spread and slippage."""
        brok_and_exch = self.brokerage_bps + self.exchange_charges_bps
        gst_bps = brok_and_exch * (self.gst_pct / 100.0)
        statutory_bps = (
            self.stt_bps + self.exchange_charges_bps + gst_bps + self.stamp_duty_bps + self.sebi_turnover_bps
        )
        total_slippage = (self.slippage_bps * 2.0) + self.spread_bps + self.adverse_stop_bps
        # Latency drag: 1 bp per 100ms
        latency_drag_bps = (self.latency_ms / 100.0) * 1.0
        total_bps = total_slippage + (self.brokerage_bps * 2.0) + statutory_bps + latency_drag_bps
        return round(float(total_bps / 100.0), 5)


# Mandatory Phase 14 Scalping Cost Scenarios
SCALPING_COST_SCENARIOS: dict[str, ScalpCostScenario] = {
    "SCALP_BASE": ScalpCostScenario(
        name="SCALP_BASE",
        spread_bps=2.0,
        slippage_bps=2.5,
        brokerage_bps=1.5,
        latency_ms=50.0,
        adverse_stop_bps=0.0,
    ),
    "SCALP_ADVERSE_SELECTION": ScalpCostScenario(
        name="SCALP_ADVERSE_SELECTION",
        spread_bps=3.0,
        slippage_bps=4.0,
        brokerage_bps=1.5,
        latency_ms=100.0,
        adverse_stop_bps=6.0,  # Toxic flow penalty on stops
    ),
    "SCALP_HIGH_LATENCY": ScalpCostScenario(
        name="SCALP_HIGH_LATENCY",
        spread_bps=3.0,
        slippage_bps=3.0,
        brokerage_bps=1.5,
        latency_ms=500.0,  # 500ms delay adds ~5 bps price drag
        adverse_stop_bps=2.0,
    ),
    "SCALP_STRESS": ScalpCostScenario(
        name="SCALP_STRESS",
        spread_bps=5.0,
        slippage_bps=6.0,
        brokerage_bps=2.0,
        latency_ms=750.0,
        adverse_stop_bps=8.0,
    ),
}


@dataclass
class ScalpCalibrationResult:
    """Independent calibration result for scalping predictions."""
    status: str  # SUFFICIENT, INSUFFICIENT_DATA, MISCALIBRATED
    expected_calibration_error: float | None
    brier_score: float | None
    is_well_calibrated: bool
    n_samples: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class ScalpEdgeResult:
    """Independent edge evaluation for scalping setups against zero-drift baseline."""
    n_trades: int
    win_rate_pct: float
    profit_factor: float
    gross_cumulative_return_pct: float
    net_cumulative_return_pct: float
    alpha_pct: float
    status: str  # STRONG_EDGE, MODERATE_EDGE, NO_EDGE, INSUFFICIENT_DATA
    reasons: list[str] = field(default_factory=list)


@dataclass
class ScalpValidationReport:
    """
    SCALP-001: Comprehensive validation document certifying whether a symbol's
    independent scalping model is safe and profitable to trade live.
    """
    symbol: str
    horizon: str
    is_scalp_valid: bool
    validation_status: str  # VALIDATED, MISCALIBRATED, INSUFFICIENT_EDGE, LATENCY_SENSITIVE, INSUFFICIENT_DATA
    calibration_result: ScalpCalibrationResult
    edge_result: ScalpEdgeResult
    cost_sensitivity: dict[str, float] = field(default_factory=dict)
    gate_reasons: list[str] = field(default_factory=list)
    model: ScalpModel | None = None

    @property
    def rejection_reasons(self) -> list[str]:
        """Convenience property listing reasons for gate rejection."""
        return self.gate_reasons if not self.is_scalp_valid else []


# ---------------------------------------------------------------------------
# Scalping Microstructure Dataset & Feature Engineering
# ---------------------------------------------------------------------------
class ScalpDataset:
    """
    SCALP-001: Specialized feature engineer and labeler for micro-horizon scalping.
    Computes order flow, spread proxies, and micro-momentum without lookahead.
    """

    @staticmethod
    def extract_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts micro-structure indicators tailored for 1-2 bar holding periods:
        - micro_spread_bps: high-low spread proxy
        - tick_volatility_pct: normalized bar range
        - order_imbalance_proxy: close location within bar range
        - micro_momentum_1b: 1-bar percentage return
        - micro_momentum_2b: 2-bar percentage return
        - volume_surge: volume relative to rolling 5-bar mean
        """
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing required OHLCV columns: {required_cols - set(df.columns)}")

        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        o = df["Open"]
        v = df["Volume"]

        bar_range = np.maximum(h - l, 1e-6)
        
        features = pd.DataFrame(index=df.index)
        features["micro_spread_bps"] = ((h - l) / c) * 10000.0 * 0.20
        features["tick_volatility_pct"] = ((h - l) / c) * 100.0
        features["order_imbalance_proxy"] = (c - o) / bar_range
        features["micro_momentum_1b"] = (c - c.shift(1)) / c.shift(1) * 100.0
        features["micro_momentum_2b"] = (c - c.shift(2)) / c.shift(2) * 100.0
        
        roll_vol = v.rolling(5).mean()
        features["volume_surge"] = np.where(roll_vol > 0, v / roll_vol, 1.0)
        
        return features

    @staticmethod
    def create_scalp_labels(
        close: pd.Series,
        horizon_bars: int = 2,
        deadband_pct: float = 0.05,
    ) -> pd.Series:
        """
        Generates fast forward returns:
        UP if forward return > +deadband_pct,
        DOWN if forward return < -deadband_pct,
        FLAT otherwise.
        """
        fwd_return = (close.shift(-horizon_bars) - close) / close * 100.0
        labels = pd.Series("FLAT", index=close.index)
        labels[fwd_return > deadband_pct] = "UP"
        labels[fwd_return < -deadband_pct] = "DOWN"
        # Mask out trailing unobserved rows
        labels.iloc[-horizon_bars:] = np.nan
        return labels


# ---------------------------------------------------------------------------
# Scalping Model
# ---------------------------------------------------------------------------
class ScalpModel:
    """
    SCALP-001: Dedicated lightweight micro-structure model trained specifically
    on high-frequency bar features. Independent from daily/30-minute macro ensembles.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            min_samples_leaf=10,
            random_state=random_state,
        )
        self.classes_ = np.array(["DOWN", "FLAT", "UP"])
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the scalping model on feature matrix X and target y."""
        valid_idx = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]

        if len(y_clean.unique()) < 2 or len(X_clean) < 30:
            raise ValueError("Insufficient data or class variety to train ScalpModel")

        self.model.fit(X_clean, y_clean)
        self.classes_ = self.model.classes_
        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities."""
        if not self.is_trained:
            raise RuntimeError("ScalpModel must be trained before predicting.")
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts top class."""
        if not self.is_trained:
            raise RuntimeError("ScalpModel must be trained before predicting.")
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# Scalping Validator
# ---------------------------------------------------------------------------
class ScalpValidator:
    """
    SCALP-001: Comprehensive validator for scalping isolation.
    Enforces independent calibration (ECE <= 0.08), latency drag, adverse stop slippage,
    and positive net edge above zero-drift benchmark.
    """

    def __init__(
        self,
        max_acceptable_ece: float = 0.08,
        min_scalp_trades: int = 30,
        min_win_rate_pct: float = 52.0,
        cost_scenarios: dict[str, ScalpCostScenario] | None = None,
    ):
        self.max_acceptable_ece = max_acceptable_ece
        self.min_scalp_trades = min_scalp_trades
        self.min_win_rate_pct = min_win_rate_pct
        self.cost_scenarios = cost_scenarios or SCALPING_COST_SCENARIOS
        self.last_model: ScalpModel | None = None

    def evaluate_calibration(
        self,
        confidences: list[float] | np.ndarray,
        hits: list[bool] | np.ndarray,
        n_bins: int = 10,
    ) -> ScalpCalibrationResult:
        """Independent calibration testing for scalping predictions."""
        confs = np.asarray(confidences, dtype=float)
        corrects = np.asarray(hits, dtype=float)
        reasons = []

        if len(confs) < 20 or len(corrects) < 20:
            return ScalpCalibrationResult(
                status="INSUFFICIENT_DATA",
                expected_calibration_error=None,
                brier_score=None,
                is_well_calibrated=False,
                n_samples=len(confs),
                reasons=["Insufficient samples for scalping calibration."],
            )

        brier = float(np.mean((confs - corrects) ** 2))
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(confs)

        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            in_bin = (confs >= bin_lower) & (confs <= bin_upper if i == n_bins - 1 else confs < bin_upper)
            bin_count = np.sum(in_bin)
            if bin_count > 0:
                bin_acc = float(np.mean(corrects[in_bin]))
                bin_conf = float(np.mean(confs[in_bin]))
                ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

        ece = round(float(ece), 4)
        brier = round(float(brier), 4)
        is_calibrated = ece <= self.max_acceptable_ece

        if is_calibrated:
            status = "SUFFICIENT"
            reasons.append(f"Scalp model is well-calibrated (ECE={ece:.4f} <= {self.max_acceptable_ece:.4f}).")
        else:
            status = "MISCALIBRATED"
            reasons.append(
                f"Scalp model miscalibrated: ECE={ece:.4f} exceeds strict scalping threshold {self.max_acceptable_ece:.4f}."
            )

        return ScalpCalibrationResult(
            status=status,
            expected_calibration_error=ece,
            brier_score=brier,
            is_well_calibrated=is_calibrated,
            n_samples=n_total,
            reasons=reasons,
        )

    def simulate_scalp_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        confidences: pd.Series,
        latency_cfg: ScalpLatencyConfig | None = None,
        cost_scenario: ScalpCostScenario | None = None,
        horizon_bars: int = 2,
    ) -> ScalpEdgeResult:
        """
        Simulates discrete execution of scalp setups with:
        - Latency delay price drag
        - Adverse selection slippage on losing trades
        - Scalping statutory friction
        """
        cfg = latency_cfg or ScalpLatencyConfig()
        scenario = cost_scenario or self.cost_scenarios["SCALP_BASE"]
        close = df["Close"]

        # Latency drag in percent: 1 bp per 100ms
        latency_drag_pct = ((cfg.latency_ms / 100.0) * 1.0) / 100.0
        base_cost_pct = scenario.round_trip_cost_pct

        trades_net = []
        trades_gross = []
        wins = 0

        for i in range(len(signals) - horizon_bars):
            sig = signals.iloc[i]
            if sig in ("BUY", "UP"):
                entry_p = close.iloc[i]
                exit_p = close.iloc[i + horizon_bars]
                
                # Gross return
                gross_ret = (exit_p - entry_p) / entry_p * 100.0
                
                # Apply friction: base cost + latency drag + adverse stop if loss
                is_loss = gross_ret < 0
                adverse_bps = cfg.adverse_selection_bps if is_loss else 0.0
                total_friction = base_cost_pct + latency_drag_pct + (adverse_bps / 100.0)
                net_ret = gross_ret - total_friction

                trades_gross.append(gross_ret)
                trades_net.append(net_ret)
                if net_ret > 0:
                    wins += 1

            elif sig in ("SELL", "DOWN"):
                entry_p = close.iloc[i]
                exit_p = close.iloc[i + horizon_bars]
                gross_ret = (entry_p - exit_p) / entry_p * 100.0
                is_loss = gross_ret < 0
                adverse_bps = cfg.adverse_selection_bps if is_loss else 0.0
                total_friction = base_cost_pct + latency_drag_pct + (adverse_bps / 100.0)
                net_ret = gross_ret - total_friction

                trades_gross.append(gross_ret)
                trades_net.append(net_ret)
                if net_ret > 0:
                    wins += 1

        n_trades = len(trades_net)
        reasons = []

        if n_trades == 0:
            return ScalpEdgeResult(
                n_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                gross_cumulative_return_pct=0.0,
                net_cumulative_return_pct=0.0,
                alpha_pct=0.0,
                status="INSUFFICIENT_DATA",
                reasons=["No scalp trades executed in backtest window."],
            )

        win_rate = (wins / n_trades) * 100.0
        pos_sum = sum(r for r in trades_net if r > 0)
        neg_sum = abs(sum(r for r in trades_net if r < 0))
        profit_factor = round(pos_sum / neg_sum, 3) if neg_sum > 0 else (99.0 if pos_sum > 0 else 0.0)
        cum_net = round(float(sum(trades_net)), 2)
        cum_gross = round(float(sum(trades_gross)), 2)

        # Baseline is zero drift
        alpha = cum_net

        has_edge = (n_trades >= self.min_scalp_trades) and (cum_net > 0.0) and (win_rate >= self.min_win_rate_pct or profit_factor >= 1.15)

        if has_edge:
            status = "STRONG_EDGE" if profit_factor >= 1.3 else "MODERATE_EDGE"
            reasons.append(
                f"Scalp edge validated: {n_trades} trades, Win Rate={win_rate:.1f}%, Profit Factor={profit_factor:.2f}, Net Return=+{cum_net:.2f}%."
            )
        else:
            status = "NO_EDGE"
            if n_trades < self.min_scalp_trades:
                reasons.append(f"Insufficient trade count ({n_trades} < {self.min_scalp_trades}).")
            if cum_net <= 0.0:
                reasons.append(f"Net return non-positive ({cum_net:.2f}%) after scalping friction and latency.")
            if win_rate < self.min_win_rate_pct and profit_factor < 1.15:
                reasons.append(f"Substandard win rate ({win_rate:.1f}%) and profit factor ({profit_factor:.2f}).")

        return ScalpEdgeResult(
            n_trades=n_trades,
            win_rate_pct=round(win_rate, 2),
            profit_factor=profit_factor,
            gross_cumulative_return_pct=cum_gross,
            net_cumulative_return_pct=cum_net,
            alpha_pct=round(alpha, 2),
            status=status,
            reasons=reasons,
        )

    def validate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        latency_cfg: ScalpLatencyConfig | None = None,
        cost_scenarios: dict[str, ScalpCostScenario] | None = None,
    ) -> ScalpValidationReport:
        """
        SCALP-001: End-to-end validation of scalping viability for a symbol.
        1. Prepares independent micro-structure dataset and 0.05% forward labels.
        2. Trains independent ScalpModel on chronological training split.
        3. Tests independent calibration on test split.
        4. Runs execution simulation across scalping cost and latency scenarios.
        5. Enforces safety gate.
        """
        cfg = latency_cfg or ScalpLatencyConfig()
        scenarios = cost_scenarios or self.cost_scenarios
        gate_reasons = []

        scalp_cfg = HORIZON_CONFIG.get(HORIZON_SCALP, {"horizon_bars": 2, "deadband_pct_default": 0.05})
        horizon_bars = scalp_cfg.get("horizon_bars", 2)
        deadband_pct = scalp_cfg.get("deadband_pct_default", 0.05)

        # 1. Dataset & Labels
        features = ScalpDataset.extract_microstructure_features(df)
        labels = ScalpDataset.create_scalp_labels(df["Close"], horizon_bars=horizon_bars, deadband_pct=deadband_pct)

        valid_idx = ~labels.isna() & ~features.isna().any(axis=1)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]

        if len(X) < 60:
            cal_res = ScalpCalibrationResult(
                status="INSUFFICIENT_DATA",
                expected_calibration_error=None,
                brier_score=None,
                is_well_calibrated=False,
                n_samples=len(X),
                reasons=["Insufficient data for scalping dataset preparation."],
            )
            edge_res = ScalpEdgeResult(
                n_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                gross_cumulative_return_pct=0.0,
                net_cumulative_return_pct=0.0,
                alpha_pct=0.0,
                status="INSUFFICIENT_DATA",
                reasons=["Insufficient data."],
            )
            return ScalpValidationReport(
                symbol=symbol,
                horizon=HORIZON_SCALP,
                is_scalp_valid=False,
                validation_status="INSUFFICIENT_DATA",
                calibration_result=cal_res,
                edge_result=edge_res,
                cost_sensitivity={},
                gate_reasons=["Dataset too short for scalping isolation."],
            )

        # 2. Chronological Split (70% train, 30% test)
        split_idx = int(len(X) * 0.70)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        test_df = df.loc[X_test.index]

        # 3. Model Training
        scalp_model = ScalpModel()
        try:
            scalp_model.train(X_train, y_train)
        except Exception as e:
            logger.error(f"ScalpModel training failed for {symbol}: {e}")
            cal_res = ScalpCalibrationResult(status="INSUFFICIENT_DATA", expected_calibration_error=None, brier_score=None, is_well_calibrated=False, n_samples=0)
            edge_res = ScalpEdgeResult(n_trades=0, win_rate_pct=0.0, profit_factor=0.0, gross_cumulative_return_pct=0.0, net_cumulative_return_pct=0.0, alpha_pct=0.0, status="INSUFFICIENT_DATA")
            return ScalpValidationReport(
                symbol=symbol,
                horizon=HORIZON_SCALP,
                is_scalp_valid=False,
                validation_status="INSUFFICIENT_DATA",
                calibration_result=cal_res,
                edge_result=edge_res,
                cost_sensitivity={},
                gate_reasons=[f"Training failed: {e}"],
            )

        # 4. Out-of-sample Predictions & Calibration
        probs = scalp_model.predict_proba(X_test)
        preds = scalp_model.predict(X_test)
        
        # Max predicted class probability
        confs = np.max(probs, axis=1)
        # Hit definition: predicted direction matched realized forward label
        hits = (preds == y_test.values) & (preds != "FLAT")
        eval_mask = preds != "FLAT"

        cal_res = self.evaluate_calibration(confs[eval_mask], hits[eval_mask])

        # 5. Execution Simulation & Cost Sensitivity
        pred_series = pd.Series(preds, index=X_test.index)
        conf_series = pd.Series(confs, index=X_test.index)

        edge_res = self.simulate_scalp_backtest(
            test_df, pred_series, conf_series, latency_cfg=cfg, cost_scenario=scenarios["SCALP_BASE"], horizon_bars=horizon_bars
        )

        cost_sensitivity = {}
        for sc_name, sc in scenarios.items():
            sc_edge = self.simulate_scalp_backtest(
                test_df, pred_series, conf_series, latency_cfg=cfg, cost_scenario=sc, horizon_bars=horizon_bars
            )
            cost_sensitivity[sc_name] = sc_edge.net_cumulative_return_pct

        # 6. Safety Gate
        is_valid = True
        status = "VALIDATED"

        if not cal_res.is_well_calibrated:
            is_valid = False
            status = "MISCALIBRATED"
            gate_reasons.extend(cal_res.reasons)

        if edge_res.status == "NO_EDGE" or edge_res.net_cumulative_return_pct <= 0.0:
            is_valid = False
            if status == "VALIDATED":
                status = "INSUFFICIENT_EDGE"
            gate_reasons.extend(edge_res.reasons)

        # Latency vulnerability check
        high_latency_ret = cost_sensitivity.get("SCALP_HIGH_LATENCY", -1.0)
        if high_latency_ret < -5.0:
            gate_reasons.append(
                f"Severe latency sensitivity: returns collapse to {high_latency_ret:.2f}% under 500ms delay."
            )

        if is_valid:
            gate_reasons.append(
                f"Scalping isolated and validated for {symbol}: Win Rate={edge_res.win_rate_pct:.1f}%, "
                f"Net P&L=+{edge_res.net_cumulative_return_pct:.2f}%, ECE={cal_res.expected_calibration_error:.3f}."
            )

        self.last_model = scalp_model
        return ScalpValidationReport(
            symbol=symbol,
            horizon=HORIZON_SCALP,
            is_scalp_valid=is_valid,
            validation_status=status,
            calibration_result=cal_res,
            edge_result=edge_res,
            cost_sensitivity=cost_sensitivity,
            gate_reasons=gate_reasons,
            model=scalp_model,
        )


if __name__ == "__main__":
    print("ScalpValidator self-test...")
    validator = ScalpValidator()

    # Generate synthetic 1m intraday data
    np.random.seed(42)
    n_bars = 400
    timestamps = pd.date_range("2026-01-05 09:15", periods=n_bars, freq="1min", tz="Asia/Kolkata")
    
    price = 1000.0
    prices = [price]
    for _ in range(n_bars - 1):
        price = max(10.0, price + np.random.normal(0.05, 0.4))
        prices.append(price)

    p_arr = np.array(prices)
    highs = p_arr + np.abs(np.random.normal(0, 0.3, n_bars))
    lows = p_arr - np.abs(np.random.normal(0, 0.3, n_bars))
    opens = np.roll(p_arr, 1)
    opens[0] = p_arr[0]
    volumes = np.random.randint(500, 20000, size=n_bars)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(highs, np.maximum(opens, p_arr)),
            "Low": np.minimum(lows, np.minimum(opens, p_arr)),
            "Close": p_arr,
            "Volume": volumes,
        },
        index=timestamps,
    )

    report = validator.validate_symbol("RELIANCE", df)
    print(f"Validation Status: {report.validation_status}")
    print(f"Is Scalp Valid: {report.is_scalp_valid}")
    print(f"Trades: {report.edge_result.n_trades}")
    print(f"Win Rate: {report.edge_result.win_rate_pct}%")
    print(f"Net Return: {report.edge_result.net_cumulative_return_pct}%")
    print(f"ECE: {report.calibration_result.expected_calibration_error}")
    print(f"Cost Sensitivity: {report.cost_sensitivity}")
    print(f"Gate Reasons: {report.gate_reasons}")
    print("ScalpValidator self-test completed successfully!")
