import pytest
from fastapi.testclient import TestClient

from api import app
from config import LOGS_DIR, configure_logging


@pytest.fixture
def client():
    return TestClient(app)


def test_metrics_endpoint(client):
    """Verify /metrics returns 200 and Prometheus metrics format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "nifty50_predictions_total" in response.text or "python_info" in response.text


def test_correlation_id_middleware(client):
    """Verify X-Correlation-ID header is propagated or auto-generated."""
    # Test auto-generation
    res1 = client.get("/api/v1/symbols")
    assert res1.status_code == 200
    assert "X-Correlation-ID" in res1.headers
    cid1 = res1.headers["X-Correlation-ID"]
    assert len(cid1) > 0

    # Test custom correlation ID propagation
    custom_cid = "test-corr-id-12345"
    res2 = client.get("/api/v1/symbols", headers={"X-Correlation-ID": custom_cid})
    assert res2.status_code == 200
    assert res2.headers.get("X-Correlation-ID") == custom_cid


def test_json_logging_configuration(monkeypatch):
    """Verify LOG_FORMAT=json initializes without throwing errors."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging(log_filename="test_json.log")
    log_file = LOGS_DIR / "test_json.log"
    assert log_file.exists()
