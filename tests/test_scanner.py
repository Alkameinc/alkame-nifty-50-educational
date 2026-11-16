from unittest.mock import MagicMock, patch

import pandas as pd

from predictor import MultiHorizonSignal, PredictionSignal
from scanner import OpportunityScanner


def test_scanner():
    mock_fetcher = MagicMock()
    mock_predictor = MagicMock()

    scanner = OpportunityScanner(mock_predictor, mock_fetcher)

    df = pd.DataFrame({"Close": [100.0, 101.0]})
    mock_fetcher.fetch_ohlcv.return_value = df
    mock_fetcher.fetch_daily_ohlcv.return_value = df
    mock_fetcher.fetch_nifty_index.return_value = df
    mock_fetcher.check_staleness.return_value = False

    sig_1d = PredictionSignal(
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        horizon="1D",
        action="BUY",
        model_predicted_class="BUY",
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
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        primary_action="BUY",
        primary_horizon="1D",
        signals={"1D": sig_1d},
        reasoning=[],
    )
    mock_predictor.generate_multi_horizon_signal.return_value = multi

    with patch("scanner.NIFTY50_SYMBOLS", ["TCS"]):
        results = scanner.scan()
        assert len(results) == 1
        assert results[0].symbol == "TCS"

        summary = scanner.scan_with_summary()
        assert summary.scanned == 1
        assert summary.actionable == 1

        # Test error
        mock_predictor.generate_multi_horizon_signal.side_effect = Exception("Test")
        summary = scanner.scan_with_summary()
        assert summary.data_error == 1
