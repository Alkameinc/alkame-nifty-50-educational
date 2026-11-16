from narrative_builder import NarrativeBuilder, PriceLevelOutcome
from predictor import ACTION_BUY, ACTION_HOLD, MultiHorizonSignal, PredictionSignal


def test_narrative_builder():
    builder = NarrativeBuilder()

    sig_1d = PredictionSignal(
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        horizon="1D",
        action=ACTION_BUY,
        model_predicted_class=ACTION_BUY,
        raw_confidence=0.8,
        calibrated_confidence=0.75,
        target_price=110.0,
        stop_loss=90.0,
        suppressed=False,
        model_version="1.0",
        feature_version="1.0",
        risk_adjusted_confidence=0.75,
        agreement_fraction=1.0,
        downside_summary=[],
        upside_summary=[],
        is_safe_to_trade_live=True,
    )

    multi = MultiHorizonSignal(
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        primary_action=ACTION_BUY,
        primary_horizon="1D",
        signals={"1D": sig_1d},
        reasoning=["Test reasoning"],
    )

    narrative = builder.build_stock_narrative(multi)

    assert narrative is not None
    assert "TCS" in narrative
    assert "BUY" in narrative


def test_narrative_builder_hold():
    builder = NarrativeBuilder()

    sig_1d = PredictionSignal(
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        horizon="1D",
        action=ACTION_HOLD,
        model_predicted_class=ACTION_HOLD,
        raw_confidence=0.8,
        calibrated_confidence=0.75,
        target_price=110.0,
        stop_loss=90.0,
        suppressed=True,
        suppression_reasons=["Test"],
        model_version="1.0",
        feature_version="1.0",
        risk_adjusted_confidence=0.75,
        agreement_fraction=1.0,
        downside_summary=[],
        upside_summary=[],
        is_safe_to_trade_live=False,
    )

    multi = MultiHorizonSignal(
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        primary_action=ACTION_HOLD,
        primary_horizon="1D",
        signals={"1D": sig_1d},
        reasoning=["Test reasoning"],
    )

    narrative = builder.build_stock_narrative(multi)

    assert narrative is not None
    assert "TCS" in narrative
    assert "HOLD" in narrative


def test_narrative_builder_outcomes():
    builder = NarrativeBuilder()

    sig_1d = PredictionSignal(
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        horizon="1D",
        action=ACTION_BUY,
        model_predicted_class=ACTION_BUY,
        raw_confidence=0.8,
        calibrated_confidence=0.75,
        target_price=110.0,
        stop_loss=90.0,
        suppressed=False,
        model_version="1.0",
        feature_version="1.0",
        risk_adjusted_confidence=0.75,
        agreement_fraction=1.0,
        downside_summary=[],
        upside_summary=[],
        is_safe_to_trade_live=True,
    )

    multi = MultiHorizonSignal(
        timestamp="2026-01-01T00:00:00",
        symbol="TCS",
        primary_action=ACTION_BUY,
        primary_horizon="1D",
        signals={"1D": sig_1d},
        reasoning=["Test reasoning"],
    )

    outcome1 = PriceLevelOutcome(horizon="1D", predicted_level="105", confidence=0.8)

    narrative = builder.build_stock_narrative(multi, [outcome1])
    assert narrative is not None
