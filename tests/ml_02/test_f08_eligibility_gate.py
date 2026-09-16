from types import SimpleNamespace

from predictor import ACTION_BUY, ACTION_HOLD, ACTION_SELL, Predictor


def _gate(show_confidence, live_edge):
    return SimpleNamespace(
        safe_to_show_calibrated_confidence=show_confidence,
        safe_to_treat_as_live_edge=live_edge,
    )


def test_missing_evidence_forces_hold_but_preserves_model_direction():
    predictor = Predictor()

    gate = _gate(False, False)

    model_action = ACTION_BUY

    final_action = model_action
    if not gate.safe_to_treat_as_live_edge:
        final_action = ACTION_HOLD
    elif not gate.safe_to_show_calibrated_confidence:
        final_action = ACTION_HOLD

    assert final_action == ACTION_HOLD
    assert model_action == ACTION_BUY


def test_confirmed_edge_and_calibration_allow_model_action():
    predictor = Predictor()

    gate = _gate(True, True)

    model_actions = [ACTION_BUY, ACTION_SELL]

    for model_action in model_actions:
        final_action = model_action

        if not gate.safe_to_treat_as_live_edge:
            final_action = ACTION_HOLD
        elif not gate.safe_to_show_calibrated_confidence:
            final_action = ACTION_HOLD

        assert final_action == model_action


def test_missing_calibration_forces_hold_even_when_edge_is_confirmed():
    predictor = Predictor()

    gate = _gate(False, True)

    model_action = ACTION_SELL
    final_action = model_action

    if not gate.safe_to_treat_as_live_edge:
        final_action = ACTION_HOLD
    elif not gate.safe_to_show_calibrated_confidence:
        final_action = ACTION_HOLD

    assert final_action == ACTION_HOLD