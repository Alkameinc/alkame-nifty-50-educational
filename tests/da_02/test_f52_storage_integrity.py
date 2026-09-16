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


def test_f52_prediction_storage_remains_readable():
    hm = HistoryManager(db_path=":memory:")

    prediction_ids = []

    for i in range(10):
        prediction_id = hm.save_prediction(
            make_signal(f"2026-09-04 15:{30 + i:02d}"),
            entry_timestamp=f"2026-09-04 15:{30 + i:02d}",
            entry_price=100.0 + i,
        )

        assert prediction_id is not None
        prediction_ids.append(prediction_id)

    records = hm.get_predictions(
        symbol="RELIANCE",
        only_unresolved=True,
    )

    assert len(records) == 10

    stored_ids = {record.id for record in records}

    assert stored_ids == set(prediction_ids)


def test_f52_unresolved_records_are_preserved():
    hm = HistoryManager(db_path=":memory:")

    prediction_id = hm.save_prediction(
        make_signal("2026-09-04 15:30"),
        entry_timestamp="2026-09-04 15:30",
        entry_price=100.0,
    )

    assert prediction_id is not None

    records = hm.get_predictions(
        symbol="RELIANCE",
        only_unresolved=True,
    )

    record = next(
        record
        for record in records
        if record.id == prediction_id
    )

    assert record.outcome_resolved is False
    assert record.entry_price == 100.0
    assert record.symbol == "RELIANCE"