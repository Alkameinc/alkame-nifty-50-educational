from history_manager import HistoryManager
from predictor import PredictionSignal


def make_signal(timestamp):
    return PredictionSignal(
        symbol="RELIANCE",
        timestamp=timestamp,
        horizon="INTRADAY",
        action="BUY",
        model_predicted_class="UP",
        model_version="test-model",
        feature_version="test-features",
        raw_confidence=0.80,
        risk_adjusted_confidence=0.80,
        calibrated_confidence=None,
        agreement_fraction=1.0,
        downside_summary="test",
        upside_summary="test",
    )


def test_f38_resolved_prediction_is_not_returned_as_unresolved():
    hm = HistoryManager(db_path=":memory:")

    prediction_id = hm.save_prediction(
        make_signal("2026-09-04 15:30"),
        entry_timestamp="2026-09-04 15:30",
        entry_price=100.0,
    )

    assert prediction_id is not None

    unresolved_before = hm.get_predictions(
        symbol="RELIANCE",
        only_unresolved=True,
    )

    assert len(unresolved_before) == 1

    resolved = hm.resolve_outcome(
        prediction_id,
        "UP",
        entry_price=100.0,
        endpoint_price=101.0,
        entry_timestamp="2026-09-04 15:30",
        target_timestamp="2026-09-07 09:15",
        resolution_reason="verified_market_observations",
    )

    assert resolved is True

    unresolved_after = hm.get_predictions(
        symbol="RELIANCE",
        only_unresolved=True,
    )

    assert len(unresolved_after) == 0

    all_records = hm.get_predictions(
        symbol="RELIANCE",
        only_unresolved=False,
    )

    record = next(
        record
        for record in all_records
        if record.id == prediction_id
    )

    assert record.outcome_resolved is True
    assert record.outcome_actual_class == "UP"
    assert record.outcome_correct is True