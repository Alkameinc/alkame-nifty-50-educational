import pandas as pd

from history_manager import HistoryManager
from predictor import PredictionSignal
from scheduler import Scheduler
from model_trainer import ModelTrainer


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
        100.1,
        100.2,
        100.3,
        100.4,
        100.45,
        100.5,
        100.5,
    ]

    return pd.DataFrame({"Close": closes}, index=index)


def _prediction(hm):
    signal = PredictionSignal(
        symbol="TCS",
        timestamp=pd.Timestamp("2026-09-04 15:30"),
        horizon="INTRADAY",
        action="BUY",
        model_predicted_class="FLAT",
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


def test_f12_equal_threshold_is_flat():
    hm = HistoryManager(db_path=":memory:")

    scheduler = Scheduler()
    scheduler.history_manager = hm

    prediction_id = _prediction(hm)

    assert prediction_id is not None

    original_deadband = ModelTrainer.compute_adaptive_deadband

    def fixed_deadband(self, df, horizon_bars, deadband_pct_default):
        return pd.Series(
            1.0,
            index=df.index,
            dtype=float,
        )

    ModelTrainer.compute_adaptive_deadband = fixed_deadband

    try:
        resolved = scheduler.resolve_pending_outcomes(
            "TCS",
            _bars(),
        )

        assert resolved == 1

        record = next(
            record
            for record in hm.get_predictions("TCS")
            if record.id == prediction_id
        )

        assert record.outcome_resolved is True
        assert record.outcome_actual_class == "FLAT"

    finally:
        ModelTrainer.compute_adaptive_deadband = original_deadband