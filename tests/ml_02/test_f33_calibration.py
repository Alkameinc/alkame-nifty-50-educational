from runtime_validator import CalibrationResult, RuntimeValidator


def test_risk_adjusted_confidence_is_not_a_calibrated_probability_without_support():
    validator = RuntimeValidator()

    result = CalibrationResult(
        status="INSUFFICIENT_DATA",
        n_samples=49,
        expected_calibration_error=None,
        is_well_calibrated=False,
        bins=[],
    )

    raw_confidence = 0.80
    risk_multiplier = 0.60
    risk_adjusted_confidence = raw_confidence * risk_multiplier

    calibrated = validator.get_calibrated_confidence(
        risk_adjusted_confidence,
        result,
    )

    assert risk_adjusted_confidence == 0.48
    assert calibrated is None


def test_calibrated_probability_requires_an_actual_calibration_bin():
    validator = RuntimeValidator()

    result = CalibrationResult(
        status="SUFFICIENT",
        n_samples=100,
        expected_calibration_error=0.05,
        is_well_calibrated=True,
        bins=[],
    )

    calibrated = validator.get_calibrated_confidence(0.48, result)

    assert calibrated is None