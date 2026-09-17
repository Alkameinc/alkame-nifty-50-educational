import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import HORIZON_SCALP
from predictor import PredictionSignal
from scalp_validator import ScalpCalibrationResult, ScalpEdgeResult, ScalpValidationReport
from scalping import ScalpingEngine


def test_scalping_engine_find_opportunities():
    mock_predictor = MagicMock()
    mock_data_fetcher = MagicMock()
    mock_feature_engineer = MagicMock()
    mock_scalp_validator = MagicMock()

    # Mock stock data
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Volume": [100, 200, 300],
            "ATR_14": [1.0, 1.0, 1.0],
            "atr": [1.0, 1.0, 1.0],
        }
    )
    mock_data_fetcher.fetch_ohlcv.return_value = df
    mock_data_fetcher.fetch_nifty_index.return_value = df
    mock_data_fetcher.check_staleness.return_value = False

    # Mock features
    mock_feature_engineer.engineer_features_for_horizon.return_value = df

    # Mock signal
    mock_sig = PredictionSignal(
        timestamp=pd.Timestamp("2026-01-01 09:15", tz="UTC"),
        symbol="TCS",
        horizon=HORIZON_SCALP,
        action="BUY",
        model_predicted_class="BUY",
        raw_confidence=0.8,
        calibrated_confidence=0.8,
        target_price=103.5,
        stop_loss=101.0,
        suppressed=False,
        model_version="1.0",
        feature_version="1.0",
        risk_adjusted_confidence=0.8,
        agreement_fraction=1.0,
        downside_summary=[],
        upside_summary=[],
        is_safe_to_trade_live=True,
    )
    mock_predictor.generate_signal.return_value = mock_sig

    mock_scalp_validator.validate_symbol.return_value = ScalpValidationReport(
        symbol="TCS",
        horizon=HORIZON_SCALP,
        is_scalp_valid=True,
        validation_status="VALIDATED",
        calibration_result=ScalpCalibrationResult("SUFFICIENT", 0.04, 0.12, True, 100),
        edge_result=ScalpEdgeResult(50, 60.0, 1.4, 12.0, 8.0, 6.0, "STRONG_EDGE"),
        cost_sensitivity={},
        gate_reasons=["Validation passed."],
    )

    engine = ScalpingEngine(
        mock_predictor,
        mock_data_fetcher,
        mock_feature_engineer,
        scalp_validator=mock_scalp_validator,
    )

    with patch("scalping.NIFTY50_SYMBOLS", ["TCS"]):
        setups = engine.find_opportunities()
        assert len(setups) == 1
        assert setups[0].symbol == "TCS"
        assert setups[0].action == "BUY"
        assert setups[0].validation_report is not None
        assert setups[0].validation_report.is_scalp_valid is True

