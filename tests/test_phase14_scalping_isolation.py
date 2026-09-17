# 1. Standard library imports
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from config import (
    EVENT_IMPACT_HORIZON,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    HORIZON_SCALP,
    HORIZON_TO_MA_PERIOD,
    HORIZON_TO_SR_METHOD,
    NIFTY50_SYMBOLS,
)
from predictor import ACTION_BUY, ACTION_SELL, PredictionSignal
from scalp_validator import (
    SCALPING_COST_SCENARIOS,
    ScalpCalibrationResult,
    ScalpCostScenario,
    ScalpDataset,
    ScalpEdgeResult,
    ScalpLatencyConfig,
    ScalpModel,
    ScalpValidationReport,
    ScalpValidator,
)
from scalping import ScalpSetup, ScalpingEngine


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------
def _generate_synthetic_1m_ohlcv(n_bars: int = 300, seed: int = 42, drift: float = 0.05) -> pd.DataFrame:
    """Generates synthetic 1-minute OHLCV data with realistic micro-volatility."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-05 09:15", periods=n_bars, freq="1min", tz="Asia/Kolkata")

    price = 1000.0
    records = []
    for ts in timestamps:
        ret = rng.normal(drift, 0.3)
        close = max(10.0, price + ret)
        high = max(price, close) + abs(rng.normal(0.1, 0.1))
        low = min(price, close) - abs(rng.normal(0.1, 0.1))
        open_ = price
        volume = int(rng.uniform(500, 3000))

        records.append({
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        })
        price = close

    return pd.DataFrame(records, index=timestamps)


# ---------------------------------------------------------------------------
# 1. Horizon Independence
# ---------------------------------------------------------------------------
def test_scalping_horizon_independence():
    """SCALP-001 Invariant 1: HORIZON_SCALP is strictly isolated from HORIZON_INTRADAY."""
    assert HORIZON_SCALP == "SCALP"
    assert HORIZON_SCALP in HORIZON_CONFIG
    assert HORIZON_INTRADAY in HORIZON_CONFIG

    scalp_cfg = HORIZON_CONFIG[HORIZON_SCALP]
    intra_cfg = HORIZON_CONFIG[HORIZON_INTRADAY]

    # Bar interval: 1m for scalping vs 5m for intraday
    assert scalp_cfg["bar_interval"] == "1m"
    assert intra_cfg["bar_interval"] == "5m"

    # Forward horizon bars: 2 bars (2 minutes) vs 6 bars (30 minutes)
    assert scalp_cfg["horizon_bars"] == 2
    assert intra_cfg["horizon_bars"] == 6

    # Deadband: 0.05% for micro-horizon vs 0.15% for intraday
    assert scalp_cfg["deadband_pct_default"] == 0.05
    assert intra_cfg["deadband_pct_default"] == 0.15

    # Lookback history: 15d for scalping vs 60d for intraday
    assert scalp_cfg["history_period"] == "15d"
    assert intra_cfg["history_period"] == "60d"

    # Supporting mapping independence
    assert HORIZON_TO_MA_PERIOD[HORIZON_SCALP] == 5
    assert HORIZON_TO_SR_METHOD[HORIZON_SCALP] == "bollinger"
    assert EVENT_IMPACT_HORIZON["ORDER_FLOW_IMBALANCE"] == HORIZON_SCALP


# ---------------------------------------------------------------------------
# 2. Dataset & Microstructure Features
# ---------------------------------------------------------------------------
def test_scalping_dataset_and_microstructure_features():
    """SCALP-001 Invariant 2 & 3: Microstructure feature extraction and 0.05% forward labels."""
    df = _generate_synthetic_1m_ohlcv(n_bars=80)

    # 1. Feature extraction
    features = ScalpDataset.extract_microstructure_features(df)
    expected_cols = {
        "micro_spread_bps",
        "tick_volatility_pct",
        "order_imbalance_proxy",
        "micro_momentum_1b",
        "micro_momentum_2b",
        "volume_surge",
    }
    assert expected_cols.issubset(set(features.columns))
    assert len(features) == len(df)

    expected_mom1 = (df["Close"].iloc[1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100.0
    assert np.isclose(features["micro_momentum_1b"].iloc[1], expected_mom1, atol=1e-4)

    # 2. Label creation
    labels = ScalpDataset.create_scalp_labels(df["Close"], horizon_bars=2, deadband_pct=0.05)
    assert len(labels) == len(df)
    assert labels.iloc[-2:].isna().all()
    valid_labels = labels.dropna().unique()
    assert set(valid_labels).issubset({"UP", "DOWN", "FLAT"})


# ---------------------------------------------------------------------------
# 3. Model Isolation
# ---------------------------------------------------------------------------
def test_scalping_model_isolation():
    """SCALP-001 Invariant 4: Dedicated fast estimator isolated from macro models."""
    model = ScalpModel(random_state=42)
    assert not model.is_trained

    dummy_X = pd.DataFrame({
        "micro_spread_bps": [1.0],
        "tick_volatility_pct": [0.1],
        "order_imbalance_proxy": [0.0],
        "micro_momentum_1b": [0.0],
        "micro_momentum_2b": [0.0],
        "volume_surge": [1.0],
    })
    with pytest.raises(RuntimeError):
        model.predict(dummy_X)

    df = _generate_synthetic_1m_ohlcv(n_bars=100)
    X = ScalpDataset.extract_microstructure_features(df).fillna(0.0)
    y = ScalpDataset.create_scalp_labels(df["Close"], horizon_bars=2, deadband_pct=0.05)

    model.train(X, y)
    assert model.is_trained

    probs = model.predict_proba(X.iloc[:5])
    assert probs.shape == (5, len(model.classes_))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    preds = model.predict(X.iloc[:5])
    assert len(preds) == 5
    for p in preds:
        assert p in model.classes_


# ---------------------------------------------------------------------------
# 4. Latency and Slippage Model
# ---------------------------------------------------------------------------
def test_scalping_latency_and_slippage_model():
    """SCALP-001 Invariant 5 and 7: Execution queue delay and adverse selection fill drag."""
    low_latency = ScalpLatencyConfig(execution_delay_ms=50.0, adverse_selection_bps=2.0)
    high_latency = ScalpLatencyConfig(execution_delay_ms=500.0, adverse_selection_bps=2.0)

    assert low_latency.latency_drag_bps == 0.5
    assert high_latency.latency_drag_bps == 5.0

    df = _generate_synthetic_1m_ohlcv(n_bars=120)
    pred_series = pd.Series(["UP" if i % 4 == 0 else "FLAT" for i in range(len(df))], index=df.index)
    conf_series = pd.Series([0.80 for _ in range(len(df))], index=df.index)

    validator = ScalpValidator()
    scenario = SCALPING_COST_SCENARIOS["SCALP_BASE"]

    edge_low = validator.simulate_scalp_backtest(
        df, pred_series, conf_series, latency_cfg=low_latency, cost_scenario=scenario, horizon_bars=2
    )
    edge_high = validator.simulate_scalp_backtest(
        df, pred_series, conf_series, latency_cfg=high_latency, cost_scenario=scenario, horizon_bars=2
    )

    assert edge_low.n_trades == edge_high.n_trades
    if edge_low.n_trades > 0:
        assert edge_low.net_cumulative_return_pct > edge_high.net_cumulative_return_pct


# ---------------------------------------------------------------------------
# 5. Cost Scenarios
# ---------------------------------------------------------------------------
def test_scalping_cost_scenarios():
    """SCALP-001 Invariant 6: Micro-horizon friction and adverse stop penalties."""
    required_scenarios = {"SCALP_BASE", "SCALP_ADVERSE_SELECTION", "SCALP_HIGH_LATENCY", "SCALP_STRESS"}
    assert required_scenarios.issubset(set(SCALPING_COST_SCENARIOS.keys()))

    base = SCALPING_COST_SCENARIOS["SCALP_BASE"]
    adverse = SCALPING_COST_SCENARIOS["SCALP_ADVERSE_SELECTION"]
    latency = SCALPING_COST_SCENARIOS["SCALP_HIGH_LATENCY"]
    stress = SCALPING_COST_SCENARIOS["SCALP_STRESS"]

    assert base.round_trip_cost_pct > 0.05
    assert adverse.adverse_stop_bps == 6.0
    assert latency.latency_ms == 500.0
    assert stress.round_trip_cost_pct > base.round_trip_cost_pct
    assert stress.round_trip_cost_pct > adverse.round_trip_cost_pct


# ---------------------------------------------------------------------------
# 6. Independent Calibration
# ---------------------------------------------------------------------------
def test_scalping_calibration_independent():
    """SCALP-001 Invariant 8: Dedicated calibration threshold (ECE <= 0.08)."""
    validator = ScalpValidator(max_acceptable_ece=0.08)

    np.random.seed(42)
    n = 100
    confs = np.linspace(0.5, 0.9, n)
    hits = (np.random.rand(n) < confs)

    cal_res = validator.evaluate_calibration(confs, hits, n_bins=5)
    assert cal_res.status == "SUFFICIENT"
    assert cal_res.expected_calibration_error is not None
    assert cal_res.brier_score is not None

    high_confs = np.full(50, 0.95)
    no_hits = np.zeros(50, dtype=bool)
    mis_res = validator.evaluate_calibration(high_confs, no_hits, n_bins=5)
    assert mis_res.status == "MISCALIBRATED"
    assert mis_res.is_well_calibrated is False
    assert mis_res.expected_calibration_error > 0.08

    tiny_res = validator.evaluate_calibration([0.8, 0.9], [True, False])
    assert tiny_res.status == "INSUFFICIENT_DATA"
    assert tiny_res.is_well_calibrated is False


# ---------------------------------------------------------------------------
# 7. Independent Edge Validation vs Zero-Drift Baseline
# ---------------------------------------------------------------------------
def test_scalping_edge_validation_vs_zero_drift():
    """SCALP-001 Invariant 9: Benchmark against zero-drift baseline."""
    validator = ScalpValidator(min_scalp_trades=30, min_win_rate_pct=52.0)
    df = _generate_synthetic_1m_ohlcv(n_bars=200, drift=0.1)
    preds = ["UP" if i % 3 == 0 else "FLAT" for i in range(len(df))]
    confs = [0.85 if i % 3 == 0 else 0.30 for i in range(len(df))]

    pred_s = pd.Series(preds, index=df.index)
    conf_s = pd.Series(confs, index=df.index)

    edge_res = validator.simulate_scalp_backtest(
        df, pred_s, conf_s,
        latency_cfg=ScalpLatencyConfig(50.0),
        cost_scenario=SCALPING_COST_SCENARIOS["SCALP_BASE"],
        horizon_bars=2,
    )
    assert edge_res.n_trades >= 30
    assert edge_res.status in ("STRONG_EDGE", "MODERATE_EDGE", "NO_EDGE")
    assert edge_res.gross_cumulative_return_pct >= edge_res.net_cumulative_return_pct


# ---------------------------------------------------------------------------
# 8. Safety Gate Suppression
# ---------------------------------------------------------------------------
def test_scalping_safety_gate_suppression():
    """Safety gate suppresses scalping setups when calibration or edge checks fail."""
    mock_predictor = MagicMock()
    mock_fetcher = MagicMock()
    mock_engineer = MagicMock()
    mock_validator = MagicMock()

    df = _generate_synthetic_1m_ohlcv(n_bars=80)
    mock_fetcher.fetch_ohlcv.return_value = df
    mock_fetcher.fetch_nifty_index.return_value = df
    mock_fetcher.check_staleness.return_value = False
    mock_engineer.engineer_features_for_horizon.return_value = pd.DataFrame({"atr": [2.0] * len(df)})

    mock_validator.validate_symbol.return_value = ScalpValidationReport(
        symbol="RELIANCE",
        horizon=HORIZON_SCALP,
        is_scalp_valid=False,
        validation_status="MISCALIBRATED",
        calibration_result=ScalpCalibrationResult("MISCALIBRATED", 0.15, 0.25, False, 100),
        edge_result=ScalpEdgeResult(35, 45.0, 0.85, -2.0, -5.0, -5.0, "NO_EDGE"),
        cost_sensitivity={},
        gate_reasons=["ECE 0.150 exceeds threshold 0.080.", "Negative net return -5.00%."],
    )

    engine = ScalpingEngine(mock_predictor, mock_fetcher, mock_engineer, scalp_validator=mock_validator)

    with patch("scalping.NIFTY50_SYMBOLS", ["RELIANCE"]):
        setups = engine.find_opportunities()
        assert len(setups) == 0

    mock_validator.validate_symbol.return_value = ScalpValidationReport(
        symbol="RELIANCE",
        horizon=HORIZON_SCALP,
        is_scalp_valid=True,
        validation_status="VALIDATED",
        calibration_result=ScalpCalibrationResult("SUFFICIENT", 0.04, 0.10, True, 100),
        edge_result=ScalpEdgeResult(40, 58.0, 1.4, 8.0, 4.5, 4.5, "STRONG_EDGE"),
        cost_sensitivity={},
        gate_reasons=["Validation passed."],
    )

    mock_sig = PredictionSignal(
        timestamp=pd.Timestamp.now(tz=timezone.utc),
        symbol="RELIANCE",
        horizon=HORIZON_SCALP,
        action=ACTION_BUY,
        model_predicted_class="UP",
        raw_confidence=0.85,
        risk_adjusted_confidence=0.85,
        calibrated_confidence=0.85,
        agreement_fraction=1.0,
        model_version="v1.0",
        feature_version="v1.0",
        downside_summary="",
        upside_summary="",
        reasoning=["Momentum positive"],
        is_safe_to_trade_live=True,
    )
    mock_predictor.generate_signal.return_value = mock_sig

    with patch("scalping.NIFTY50_SYMBOLS", ["RELIANCE"]):
        setups = engine.find_opportunities()
        assert len(setups) == 1
        assert setups[0].symbol == "RELIANCE"
        assert setups[0].action == ACTION_BUY
        assert setups[0].validation_report is not None
        assert setups[0].validation_report.is_scalp_valid is True


# ---------------------------------------------------------------------------
# 9. End-to-End Validation Report
# ---------------------------------------------------------------------------
def test_scalping_end_to_end_validation_report():
    """End-to-end test of ScalpValidator.validate_symbol generating full multi-scenario report."""
    validator = ScalpValidator()
    df = _generate_synthetic_1m_ohlcv(n_bars=350, drift=0.08)

    report = validator.validate_symbol("TCS", df)
    assert report.symbol == "TCS"
    assert report.horizon == HORIZON_SCALP
    assert report.calibration_result is not None
    assert report.edge_result is not None
    assert len(report.cost_sensitivity) == 4
    assert "SCALP_BASE" in report.cost_sensitivity
    assert "SCALP_ADVERSE_SELECTION" in report.cost_sensitivity
    assert "SCALP_HIGH_LATENCY" in report.cost_sensitivity
    assert "SCALP_STRESS" in report.cost_sensitivity
    assert len(report.gate_reasons) > 0


# ---------------------------------------------------------------------------
# 10. Backwards Compatibility
# ---------------------------------------------------------------------------
def test_scalping_backwards_compatibility():
    """Verify backwards compatibility for ScalpSetup and ScalpingEngine constructor."""
    setup = ScalpSetup(
        symbol="INFY",
        action="BUY",
        entry_price=1500.0,
        target_price=1510.0,
        stop_loss=1495.0,
        confidence=80.0,
        risk_reward_ratio=2.0,
        reasoning=["Test reasoning"],
    )
    assert setup.symbol == "INFY"
    assert setup.validation_report is None

    mock_predictor = MagicMock()
    mock_fetcher = MagicMock()
    mock_engineer = MagicMock()
    engine = ScalpingEngine(mock_predictor, mock_fetcher, mock_engineer)
    assert engine.scalp_validator is not None
    assert isinstance(engine.scalp_validator, ScalpValidator)
