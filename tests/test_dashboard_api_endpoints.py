from fastapi.testclient import TestClient

from api import app
from config import NIFTY50_SYMBOLS


def test_get_fundamentals_valid_and_invalid():
    client = TestClient(app)
    # Valid symbol
    sym = NIFTY50_SYMBOLS[0]
    res = client.get(f"/api/v1/fundamentals/{sym}")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == sym
    assert "cmp" in data
    assert "fundamentals" in data
    assert isinstance(data["fundamentals"], dict)

    # Invalid symbol
    res_err = client.get("/api/v1/fundamentals/INVALID_SYM")
    assert res_err.status_code == 404


def test_get_risk_reading():
    client = TestClient(app)
    res = client.get("/api/v1/risk/reading")
    assert res.status_code == 200
    data = res.json()
    assert "risk_level" in data
    assert "composite_score" in data
    assert "confidence_multiplier" in data
    assert "banner_message" in data
    assert "toggle_enabled" in data
    assert "indicators" in data


def test_notes_crud():
    client = TestClient(app)
    sym = NIFTY50_SYMBOLS[0]
    # Add note
    payload = {
        "note_text": "Dashboard test note for automated test",
        "related_action": "HOLD",
        "created_by": "test_analyst",
    }
    res_post = client.post(f"/api/v1/notes/{sym}", json=payload)
    assert res_post.status_code == 200
    assert res_post.json()["status"] == "success"
    assert "note_id" in res_post.json()

    # Get notes
    res_get = client.get(f"/api/v1/notes/{sym}")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["symbol"] == sym
    assert len(data["notes"]) > 0
    assert any("Dashboard test note" in n["note_text"] for n in data["notes"])


def test_override_crud():
    client = TestClient(app)
    sym = NIFTY50_SYMBOLS[0]
    # Record override
    payload = {
        "symbol": sym,
        "original_action": "HOLD",
        "overridden_action": "BUY",
        "reason": "Test audited override reason",
        "created_by": "test_trader",
    }
    res_post = client.post(f"/api/v1/override/{sym}", json=payload)
    assert res_post.status_code == 200
    data = res_post.json()
    assert data["symbol"] == sym
    assert data["overridden_action"] == "BUY"

    # Get active override
    res_get = client.get(f"/api/v1/override/{sym}")
    assert res_get.status_code == 200
    act = res_get.json()["active_override"]
    assert act is not None
    assert act["overridden_action"] == "BUY"
    assert act["reason"] == "Test audited override reason"


def test_get_history():
    client = TestClient(app)
    sym = NIFTY50_SYMBOLS[0]
    res = client.get(f"/api/v1/history/{sym}?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == sym
    assert "count" in data
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
