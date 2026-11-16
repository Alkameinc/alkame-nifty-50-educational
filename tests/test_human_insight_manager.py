import pytest

from human_insight_manager import HumanInsightManager
from predictor import ACTION_BUY, ACTION_HOLD, PredictionSignal


@pytest.fixture
def manager(tmp_path):
    db_file = tmp_path / "test_human_insight.db"
    return HumanInsightManager(db_path=db_file)


def test_human_insight_manager_notes(manager):
    # Test add note
    manager.add_note("TCS", "Good stock", related_action=ACTION_BUY)
    notes = manager.get_notes("TCS")
    assert len(notes) == 1
    assert notes[0].note_text == "Good stock"

    manager.add_note("INFY", "Bad stock")
    all_notes = manager.get_notes()
    assert len(all_notes) == 2


def test_human_insight_manager_overrides(manager):
    # Test record override
    manager.record_override("TCS", ACTION_BUY, ACTION_HOLD, "Too risky")
    overrides = manager.get_overrides("TCS")
    assert len(overrides) == 1
    assert overrides[0].reason == "Too risky"

    all_overrides = manager.get_overrides()
    assert len(all_overrides) == 1


def test_human_insight_manager_apply_override_to_signal(manager):
    sig = PredictionSignal(
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
    )

    new_sig = manager.apply_override_to_signal(sig, ACTION_HOLD, "User said so")
    assert new_sig.action == ACTION_HOLD
    # Some logic might just prepend to reasoning
    assert any("User said so" in r for r in new_sig.reasoning)

    overrides = manager.get_overrides("TCS")
    assert len(overrides) == 1


def test_human_insight_manager_feedback(manager):
    # Test record feedback
    # record_feedback(self, symbol, signal_timestamp, original_action, was_helpful, note)
    manager.record_feedback("TCS", "2026-01-01T00:00:00", ACTION_BUY, True, "Worked well")
    feedback = manager.get_feedback("TCS")
    assert len(feedback) == 1
    assert feedback[0].was_helpful == 1

    manager.record_feedback("TCS", "2026-01-02T00:00:00", ACTION_BUY, False, "Failed")
    summary = manager.get_feedback_summary("TCS")
    assert summary["total_feedback"] == 2
    assert summary["total_rated"] == 2
    assert summary["helpful_pct"] == 50.0

    all_summary = manager.get_feedback_summary()
    assert all_summary["total_feedback"] == 2
