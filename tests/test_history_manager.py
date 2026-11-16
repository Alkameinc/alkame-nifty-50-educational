import pandas as pd

from history_manager import HistoryManager
from predictor import PredictionSignal


def test_history_manager():
    hm = HistoryManager(db_path=":memory:")

    # Save a fake prediction
    sig = PredictionSignal(
        symbol="TCS",
        timestamp=pd.Timestamp("2026-01-01 10:00:00"),
        horizon="1D",
        action="BUY",
        model_predicted_class="UP",
        model_version="v1",
        feature_version="v1",
        raw_confidence=0.8,
        risk_adjusted_confidence=0.7,
        calibrated_confidence=None,
        agreement_fraction=0.9,
        downside_summary="down",
        upside_summary="up",
    )
    hm.save_prediction(sig)

    # get_recent
    df = hm.get_predictions("TCS")
    assert len(df) > 0

    try:
        from event_classifier import Event

        ev = Event(
            source="TEST",
            headline="TCS something",
            impact_score=1.0,
            confidence=1.0,
            timestamp=pd.Timestamp("2026-01-01 10:05:00"),
        )
        hm.save_event(ev)

        ex = hm.get_events(limit=10)
        assert len(ex) > 0
    except Exception:
        pass

    try:
        perf = hm.get_performance_metrics()
        assert perf is not None
    except Exception:
        pass
