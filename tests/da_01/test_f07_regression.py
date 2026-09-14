"""Pre-fix F07 acceptance probes; no application implementation changes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ensemble_manager import EnsemblePrediction
from event_classifier import EventBatchResult
from predictor import Predictor
from runtime_validator import EdgeCheckResult, RuntimeValidator


def test_valid_pairs_preserve_count_weighted_calibration():
    result = RuntimeValidator().compute_calibration(
        pd.DataFrame({"confidence": [0.6] * 50, "correct": [1] * 30 + [0] * 20})
    )
    assert result.status == "SUFFICIENT"
    assert result.n_samples == 50
    assert result.expected_calibration_error == pytest.approx(0.0)
    assert result.is_well_calibrated


@pytest.mark.parametrize(
    "confidence,correct,expected_count",
    [
        ([np.nan] * 50, [1] * 50, 0),
        ([0.6] * 49 + [np.nan], [1] * 30 + [0] * 20, 49),
        ([np.inf] * 50, [1] * 50, 0),
        ([-0.2, 1.2] * 25, [0, 1] * 25, 0),
        ([0.5] * 50, [2, -1] * 25, 0),
        ([0.6] * 50, [None] * 25 + [1] * 25, 25),
    ],
    ids=["all-nan", "49-plus-invalid", "infinite", "out-of-range", "nonbinary", "missing-outcome"],
)
def test_invalid_pairs_cannot_establish_sufficient_evidence(confidence, correct, expected_count):
    result = RuntimeValidator().compute_calibration(pd.DataFrame({"confidence": confidence, "correct": correct}))
    assert result.status == "INSUFFICIENT_DATA"
    assert result.n_samples == expected_count
    assert not result.is_well_calibrated
    assert result.expected_calibration_error is None


def make_predictor():
    # Stable market/model inputs; the validator, gate and Predictor remain actual implementation.
    return Predictor(
        data_fetcher=SimpleNamespace(check_staleness=lambda *args, **kwargs: False),
        feature_engineer=SimpleNamespace(
            engineer_features_for_horizon=lambda *args, **kwargs: pd.DataFrame({"atr": [1.0], "rsi": [50.0]})
        ),
        ensemble_manager=SimpleNamespace(
            predict=lambda *args, **kwargs: [
                EnsemblePrediction("UP", 0.6, 1.0, {"fixture": "UP"}, "fixture", "fixture")
            ]
        ),
        event_classifier=SimpleNamespace(classify_batch=lambda **kwargs: EventBatchResult([], "NO_EVENTS")),
        global_risk_monitor=SimpleNamespace(
            compute_composite_risk=lambda: SimpleNamespace(risk_level="NORMAL", dominant_driver=None),
            get_confidence_multiplier=lambda *args: 1.0,
            get_toggle_state=lambda: SimpleNamespace(enabled=False),
        ),
        runtime_validator=RuntimeValidator(),
    )


@pytest.mark.parametrize("invalid", [False, True], ids=["valid-evidence", "invalid-evidence"])
def test_calibration_result_controls_actual_predictor_decision(invalid):
    predictor = make_predictor()
    calibration = predictor.runtime_validator.compute_calibration(
        pd.DataFrame(
            {
                "confidence": [np.nan if invalid else 0.6] * 50,
                "correct": [1] * 30 + [0] * 20,
            }
        )
    )
    frame = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000},
        index=pd.date_range("2026-09-14 09:15", periods=30, freq="5min"),
    )
    signal = predictor.generate_signal(
        "RELIANCE",
        frame,
        frame,
        calibration_result=calibration,
        edge_check_result=EdgeCheckResult("EDGE_CONFIRMED", 100, 5.0, 1.0, 4.0),
    )
    if invalid:
        assert signal.action == "HOLD"
        assert signal.calibrated_confidence is None
        assert signal.suppressed
    else:
        assert signal.action == "BUY"
        assert signal.calibrated_confidence == pytest.approx(0.6)
        assert not signal.suppressed
