import pytest
import tempfile
from pathlib import Path
from human_insight_manager import HumanInsightManager
from predictor import PredictionSignal, ACTION_BUY, ACTION_HOLD
from tests.fs_02.fixtures.prediction_signal import get_specimen_signal

def test_f05_override_normal_case():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_human_insight.db"
        manager = HumanInsightManager(db_path=db_path)
        
        orig_sig = get_specimen_signal()
        orig_sig.horizon = "3D"
        orig_sig.action = ACTION_BUY
        
        overridden = manager.apply_override_to_signal(
            signal=orig_sig,
            overridden_action=ACTION_HOLD,
            reason="Reduce exposure",
            created_by="Alice"
        )
        
        # Verify effective decision and audit agree
        assert overridden.action == ACTION_HOLD
        assert overridden.horizon == "3D"
        assert overridden.symbol == "RELIANCE"
        assert any("HUMAN OVERRIDE by Alice" in r for r in overridden.reasoning)
        
        # Verify audit persistence
        overrides = manager.get_overrides("RELIANCE")
        assert len(overrides) == 1
        assert overrides[0].original_action == ACTION_BUY
        assert overrides[0].overridden_action == ACTION_HOLD
        assert overrides[0].reason == "Reduce exposure"
        assert overrides[0].created_by == "Alice"

def test_f05_override_boundary_context_preservation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_human_insight.db"
        manager = HumanInsightManager(db_path=db_path)
        
        orig_sig = get_specimen_signal()
        orig_sig.target_price = 102.0
        orig_sig.stop_loss = 99.0
        orig_sig.peak_potential_price = 103.0
        orig_sig.suppressed = True
        orig_sig.suppression_reasons = ["Test gate active"]
        
        overridden = manager.apply_override_to_signal(
            signal=orig_sig,
            overridden_action="SELL",
            reason="Fixture risk review",
            created_by="Bob"
        )
        
        # Check target/stop/peak preserved
        assert overridden.target_price == 102.0
        assert overridden.stop_loss == 99.0
        assert overridden.peak_potential_price == 103.0
        assert overridden.suppressed is True
        assert overridden.suppression_reasons == ["Test gate active"]

def test_f05_override_blank_reason_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_human_insight.db"
        manager = HumanInsightManager(db_path=db_path)
        
        orig_sig = get_specimen_signal()
        
        # Blank/whitespace reason should raise ValueError and not record
        with pytest.raises(ValueError, match="reason is mandatory"):
            manager.apply_override_to_signal(orig_sig, "SELL", "   ", "Alice")
            
        overrides = manager.get_overrides("RELIANCE")
        assert len(overrides) == 0

def test_f05_override_persistence_failure_simulation(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_human_insight.db"
        manager = HumanInsightManager(db_path=db_path)
        
        orig_sig = get_specimen_signal()
        
        # Monkeypatch record_override to simulate DB failure
        monkeypatch.setattr(manager, "record_override", lambda *args, **kwargs: None)
        
        with pytest.raises(ValueError, match="Failed to record override"):
            manager.apply_override_to_signal(orig_sig, "SELL", "Valid reason", "Alice")
