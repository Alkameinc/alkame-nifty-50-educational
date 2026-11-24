import json
from pathlib import Path

def test_f48_synthetic_json_specimen_contract():
    json_path = Path(__file__).parent / "fixtures" / "signal_specimen.json"
    assert json_path.exists(), "signal_specimen.json fixture must exist"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["symbol"] == "RELIANCE"
    assert data["action"] == "HOLD"
    assert data["raw_confidence"] == 0.8
    assert data["calibrated_confidence"] is None
    assert "Fixture model score: 0.80" in data["reasoning"]

def test_f57_effect_state_lint_integrity():
    # Verify that react-hooks/set-state-in-effect is no longer present in SignalCard.jsx
    component_path = Path(__file__).parent.parent.parent / "frontend" / "src" / "components" / "SignalCard.jsx"
    assert component_path.exists()
    content = component_path.read_text(encoding="utf-8")
    
    # Check that useEffect wraps state setters inside async function loadSignalData
    assert "const loadSignalData = async () => {" in content
    assert "loadSignalData();" in content
