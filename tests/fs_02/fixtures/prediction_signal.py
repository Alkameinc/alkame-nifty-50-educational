from datetime import datetime
from predictor import PredictionSignal, MultiHorizonSignal

def get_specimen_signal() -> PredictionSignal:
    stamp = datetime.fromisoformat("2026-09-07T10:00:00+05:30")
    return PredictionSignal(
        symbol="RELIANCE",
        timestamp=stamp,
        horizon="INTRADAY",
        action="HOLD",
        model_predicted_class="UP",
        model_version="v1.0",
        feature_version="v1.0",
        raw_confidence=0.8,
        risk_adjusted_confidence=0.64,
        calibrated_confidence=None,
        agreement_fraction=0.75,
        downside_summary="Fixture: loss remains possible.",
        upside_summary="Fixture: model leans upward.",
        reasoning=["Fixture model score: 0.80"],
        is_safe_to_trade_live=False,
        suppressed=True,
        suppression_reasons=["Fixture: evidence unavailable"],
        target_price=102.0,
        stop_loss=99.0,
        peak_potential_price=103.0
    )

def get_specimen_multi_signal() -> MultiHorizonSignal:
    sig = get_specimen_signal()
    return MultiHorizonSignal(
        symbol=sig.symbol,
        timestamp=sig.timestamp,
        signals={"INTRADAY": sig},
        primary_action="HOLD",
        primary_horizon="INTRADAY",
        reasoning=["Fixture"]
    )
