import sys
import time
from pathlib import Path

import pytest

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_symbols_endpoint_latency(client):
    """Verify /api/v1/symbols response latency is under 50ms."""
    start_time = time.perf_counter()
    response = client.get("/api/v1/symbols")
    latency_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200
    assert latency_ms < 200, f"Symbols endpoint latency {latency_ms:.2f}ms exceeded SLA threshold (200ms)"


def test_health_endpoint_latency(client):
    """Verify /api/v1/health response latency is under 200ms."""
    start_time = time.perf_counter()
    response = client.get("/api/v1/health")
    latency_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200
    assert latency_ms < 500, f"Health endpoint latency {latency_ms:.2f}ms exceeded SLA threshold (500ms)"


def test_metrics_endpoint_latency(client):
    """Verify /metrics endpoint response latency is under 100ms."""
    start_time = time.perf_counter()
    response = client.get("/metrics")
    latency_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200
    assert latency_ms < 200, f"Metrics endpoint latency {latency_ms:.2f}ms exceeded SLA threshold (200ms)"
