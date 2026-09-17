import pandas as pd

from history_manager import HistoryManager
from predictor import PredictionSignal
from scheduler import Scheduler


def _bars():
    index = pd.to_datetime([
        "2026-09-04 15:30",
        "2026-09-07 09:15",
        "2026-09-07 09:20",
        "2026-09-07 09:25",
        "2026-09-07 09:30",
        "2026-09-07 09:35",
        "2026-09-07 09:40",
        "2026-09-07 09:45",
    ])

    closes = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
    ]

    return pd.DataFrame({"Close": closes}, index=index)


def _prediction(hm):
    signal = PredictionSignal(
        symbol="RELIANCE",
        timestamp=pd.Timestamp("2026-09-04 15:30"),
        horizon="INTRADAY",
        action="BUY",
        model_predicted_class="UP",
        model_version="test-model",
        feature_version="test-features",
        raw_confidence=0.8,
        risk_adjusted_confidence=0.8,
        calibrated_confidence=None,
        agreement_fraction=1.0,
        downside_summary="test",
        upside_summary="test",
    )

    return hm.save_prediction(
        signal,
        entry_timestamp="2026-09-04 15:30",
        entry_price=100.0,
    )


def test_f11_requires_full_forward_horizon():
    hm = HistoryManager(db_path=":memory:")
    scheduler = Scheduler()
    scheduler.history_manager = hm

    prediction_id = _prediction(hm)
    assert prediction_id is not None

    incomplete_bars = _bars().iloc[:2]

    resolved = scheduler.resolve_pending_outcomes(
        "RELIANCE",
        incomplete_bars,
    )

    assert resolved == 0

    record = next(
        record
        for record in hm.get_predictions("RELIANCE")
        if record.id == prediction_id
    )

    assert record.outcome_resolved is False


def test_f11_resolves_after_full_forward_horizon():
    hm = HistoryManager(db_path=":memory:")
    scheduler = Scheduler()
    scheduler.history_manager = hm

    prediction_id = _prediction(hm)
    assert prediction_id is not None

    resolved = scheduler.resolve_pending_outcomes(
        "RELIANCE",
        _bars(),
    )

    assert resolved == 1

    record = next(
        record
        for record in hm.get_predictions("RELIANCE")
        if record.id == prediction_id
    )

    assert record.outcome_resolved is True
    assert record.outcome_actual_class == "UP"
    assert record.entry_price == 100.0