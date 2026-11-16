from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_tasks_lifecycle():
    # 1. Fetch initial tasks for GLOBAL
    resp = client.get("/api/v1/tasks/GLOBAL")
    assert resp.status_code == 200
    data = resp.json()
    assert "tasks" in data
    assert len(data["tasks"]) >= 1

    # 2. Add a new task for TCS
    new_task_payload = {
        "title": "Institutional test task for Brier calibration audit",
        "priority": "P0",
        "category": "Calibration",
        "symbol": "TCS",
        "due_date": "Today, 18:00 IST",
        "completed": False,
    }
    create_resp = client.post("/api/v1/tasks/TCS", json=new_task_payload)
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["title"] == new_task_payload["title"]
    assert created["priority"] == "P0"
    task_id = created["id"]

    # 3. Toggle task completion
    toggle_resp = client.put(f"/api/v1/tasks/{task_id}/toggle")
    assert toggle_resp.status_code == 200
    toggled = toggle_resp.json()
    assert toggled["completed"] is True

    # 4. Fetch tasks for TCS (should include the new task and GLOBAL tasks)
    tcs_tasks_resp = client.get("/api/v1/tasks/TCS")
    assert tcs_tasks_resp.status_code == 200
    tcs_tasks = tcs_tasks_resp.json()["tasks"]
    assert any(t["id"] == task_id for t in tcs_tasks)

    # 5. Delete task
    del_resp = client.delete(f"/api/v1/tasks/{task_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted_id"] == task_id

    # 6. Delete again should 404
    del_404_resp = client.delete(f"/api/v1/tasks/{task_id}")
    assert del_404_resp.status_code == 404


def test_stress_test_simulation():
    # 1. Normal benign shock
    payload = {
        "symbol": "ADANIENT",
        "crude_shock_pct": 5.0,
        "vix_shock_pct": 2.0,
        "inr_shock_pct": -1.0,
    }
    resp = client.post("/api/v1/risk/stress-test", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "ADANIENT"
    assert "base_cmp" in data
    assert "estimated_cmp" in data
    assert "estimated_pnl_pct" in data
    assert data["var_95_pct"] > 0
    assert data["var_99_pct"] > data["var_95_pct"]
    assert data["risk_regime"] in ("BENIGN", "MODERATE_STRESS", "ELEVATED_STRESS")
    assert "crude_beta" in data["factor_sensitivities"]

    # 2. Elevated stress shock
    severe_payload = {
        "symbol": "HDFCBANK",
        "crude_shock_pct": 20.0,
        "vix_shock_pct": 40.0,
        "inr_shock_pct": -5.0,
    }
    severe_resp = client.post("/api/v1/risk/stress-test", json=severe_payload)
    assert severe_resp.status_code == 200
    severe_data = severe_resp.json()
    assert severe_data["risk_regime"] in ("MODERATE_STRESS", "ELEVATED_STRESS")
