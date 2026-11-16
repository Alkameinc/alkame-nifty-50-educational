"""F07: usable paired evidence is the only support for calibration."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from runtime_validator import RuntimeValidator


def assert_accounting(result, submitted, accepted):
    assert result.input_count == submitted
    assert result.n_samples == accepted
    assert result.rejected_count == submitted - accepted
    assert sum(result.rejection_reasons.values()) == result.rejected_count


@pytest.mark.parametrize(
    "confidence,correct,reason",
    [
        (np.nan, 1, "nonfinite_confidence"),
        (np.inf, 1, "nonfinite_confidence"),
        (-np.inf, 1, "nonfinite_confidence"),
        (-0.1, 1, "out_of_range_confidence"),
        (1.1, 1, "out_of_range_confidence"),
        (None, 1, "invalid_confidence"),
        (pd.NA, 1, "invalid_confidence"),
        ("0.6", 1, "invalid_confidence"),
        ("bad", 1, "invalid_confidence"),
        (0.6 + 0j, 1, "invalid_confidence"),
        (0.6, np.nan, "nonfinite_outcome"),
        (0.6, np.inf, "nonfinite_outcome"),
        (0.6, -1, "nonbinary_outcome"),
        (0.6, 2, "nonbinary_outcome"),
        (0.6, 0.5, "nonbinary_outcome"),
        (0.6, None, "invalid_outcome"),
        (0.6, pd.NA, "invalid_outcome"),
        (0.6, "1", "invalid_outcome"),
        (0.6, 1 + 0j, "invalid_outcome"),
        (Decimal("1.00000000000000000001"), 1, "out_of_range_confidence"),
    ],
)
def test_invalid_pair_is_excluded_and_explained(confidence, correct, reason):
    # Object dtype preserves the actual malformed value instead of normalizing it in the fixture.
    frame = pd.DataFrame({"confidence": [confidence] * 50, "correct": [correct] * 50}, dtype=object)
    result = RuntimeValidator().compute_calibration(frame)
    assert result.status == "INSUFFICIENT_DATA"
    assert result.expected_calibration_error is None
    assert not result.is_well_calibrated
    assert result.bins == []
    assert_accounting(result, 50, 0)
    assert result.rejection_reasons == {reason: 50}


def test_mixed_rows_use_accepted_support_and_reconcile_without_mutation():
    # Hand oracle: 25 observations at .2 with accuracy .4; 75 at .8 with accuracy .4.
    # ECE = (25/100)*.2 + (75/100)*.4 = .35 (unweighted would be .30).
    valid = pd.DataFrame({"confidence": [0.2] * 25 + [0.8] * 75, "correct": [1] * 10 + [0] * 15 + [1] * 30 + [0] * 45})
    invalid = pd.DataFrame({"confidence": [np.nan, -0.1, 0.5, np.inf], "correct": [2, 1, None, np.nan]})
    frame = pd.concat([valid, invalid])  # Repeated indexes must not misalign paired rows.
    original = frame.copy(deep=True)
    result = RuntimeValidator().compute_calibration(frame)
    assert result.status == "SUFFICIENT"
    assert result.expected_calibration_error == pytest.approx(0.35)
    assert not result.is_well_calibrated
    assert_accounting(result, 104, 100)
    assert result.rejection_reasons == {
        "nonfinite_confidence": 2,
        "out_of_range_confidence": 1,
        "nonfinite_outcome": 1,
    }
    assert sum(b.count for b in result.bins) == 100
    assert len(result.bins) == 10
    assert sorted(b.count for b in result.bins if b.count) == [25, 75]
    pd.testing.assert_frame_equal(frame, original)
    assert result == RuntimeValidator().compute_calibration(frame)


@pytest.mark.parametrize("decimal_confidence,decimal_outcome", [(True, False), (False, True), (True, True)])
def test_valid_decimal_observations_preserve_existing_numeric_support(decimal_confidence, decimal_outcome):
    confidence = Decimal("0.6") if decimal_confidence else 0.6
    outcomes = [1] * 30 + [0] * 20
    if decimal_outcome:
        outcomes = [Decimal(value) for value in outcomes]
    frame = pd.DataFrame({"confidence": [confidence] * 50, "correct": outcomes})
    result = RuntimeValidator().compute_calibration(frame)
    assert result.status == "SUFFICIENT"
    assert result.expected_calibration_error == pytest.approx(0.0)
    assert result.is_well_calibrated
    assert_accounting(result, 50, 50)


@pytest.mark.parametrize("accepted", [0, 49, 50])
def test_threshold_depends_only_on_usable_pairs(accepted):
    frame = pd.DataFrame({"confidence": [0.6] * accepted + [np.nan] * 50, "correct": [1] * (accepted + 50)})
    result = RuntimeValidator().compute_calibration(frame)
    assert result.status == ("SUFFICIENT" if accepted >= 50 else "INSUFFICIENT_DATA")
    assert_accounting(result, accepted + 50, accepted)


@pytest.mark.parametrize("minimum", [0, 1, 50])
def test_no_support_never_passes_even_with_zero_configured_minimum(minimum):
    result = RuntimeValidator(min_calibration_samples=minimum).compute_calibration(
        pd.DataFrame({"confidence": [np.nan] * 50, "correct": [1] * 50})
    )
    assert result.status == "INSUFFICIENT_DATA"
    assert not result.is_well_calibrated
    assert result.expected_calibration_error is None
    assert_accounting(result, 50, 0)


@pytest.mark.parametrize("columns", [{}, {"confidence": [0.6] * 50}, {"correct": [1] * 50}])
def test_missing_columns_are_insufficient_with_reconciled_accounting(columns):
    frame = pd.DataFrame(columns)
    result = RuntimeValidator().compute_calibration(frame)
    assert result.status == "INSUFFICIENT_DATA"
    assert result.expected_calibration_error is None
    assert_accounting(result, len(frame), 0)
    assert result.rejection_reasons == ({"missing_columns": len(frame)} if len(frame) else {})


def test_empty_nullable_and_endpoint_values():
    empty = RuntimeValidator(min_calibration_samples=0).compute_calibration(
        pd.DataFrame(columns=["confidence", "correct"])
    )
    assert empty.status == "INSUFFICIENT_DATA"
    assert_accounting(empty, 0, 0)
    frame = pd.DataFrame(
        {
            "confidence": pd.Series([0.0] * 20 + [0.5] * 20 + [1.0] * 20 + [pd.NA], dtype="Float64"),
            "correct": pd.Series([False] * 20 + [True] * 10 + [False] * 10 + [True] * 20 + [pd.NA], dtype="boolean"),
        }
    )
    result = RuntimeValidator().compute_calibration(frame)
    assert result.status == "SUFFICIENT"
    assert result.expected_calibration_error == pytest.approx(0.0)
    assert result.is_well_calibrated
    assert_accounting(result, 61, 60)
    assert len(result.bins) == 10
    assert [b.count for b in result.bins if b.count] == [20, 20, 20]


@pytest.mark.parametrize("n_bins", [-1, 0])
def test_calculation_failure_cannot_certify_evidence(n_bins):
    frame = pd.DataFrame({"confidence": [0.6] * 50, "correct": [1] * 30 + [0] * 20})
    result = RuntimeValidator(n_bins=n_bins).compute_calibration(frame)
    assert result.status == "INSUFFICIENT_DATA"
    assert not result.is_well_calibrated
    assert result.expected_calibration_error is None
    assert_accounting(result, 50, 50)
