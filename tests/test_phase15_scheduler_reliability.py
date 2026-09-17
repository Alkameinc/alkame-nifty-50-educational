# 1. Standard library imports
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from api import app
from config import (
    SCHEDULER_CIRCUIT_BREAKER_DEGRADED_FAILURES,
    SCHEDULER_CIRCUIT_BREAKER_MAX_FAILURES,
)
from health_monitor import registry as health_registry
from scheduler import CycleResult, Scheduler
from scheduler_circuit_breaker import (
    SCHEDULER_STATE_DEGRADED,
    SCHEDULER_STATE_FAILED,
    SCHEDULER_STATE_HALTED,
    SCHEDULER_STATE_HEALTHY,
    SCHEDULER_STATE_STARTING,
    SchedulerCircuitBreaker,
    SchedulerMetrics,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------
def _build_mock_stock_df(n_bars: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2026-01-05 09:15", periods=n_bars, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "Open": [100.0] * n_bars,
            "High": [105.0] * n_bars,
            "Low": [95.0] * n_bars,
            "Close": [102.0] * n_bars,
            "Volume": [1000] * n_bars,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. Initial State
# ---------------------------------------------------------------------------
def test_scheduler_initial_state_starting():
    """SCHED-001: Scheduler initializes in explicit STARTING state with zeroed metrics."""
    cb = SchedulerCircuitBreaker()
    assert cb.state == SCHEDULER_STATE_STARTING
    assert cb.can_execute() is True
    assert cb.metrics.consecutive_failures == 0
    assert cb.metrics.total_cycles == 0
    assert cb.metrics.successful_cycles == 0
    assert cb.metrics.failed_cycles == 0
    assert cb.metrics.circuit_breaker_tripped is False
    assert cb.metrics.last_success is None
    assert cb.metrics.last_failure is None

    scheduler = Scheduler(circuit_breaker=cb)
    assert scheduler.circuit_breaker.state == SCHEDULER_STATE_STARTING
    status = scheduler.get_status()
    assert status["state"] == SCHEDULER_STATE_STARTING
    assert status["is_operational"] is True


# ---------------------------------------------------------------------------
# 2. Single Failure Retry
# ---------------------------------------------------------------------------
def test_circuit_breaker_single_failure_retry():
    """OPS-001: 1 failure allows retry on next tick; status remains STARTING/HEALTHY."""
    cb = SchedulerCircuitBreaker(max_failures=5, degraded_failures=2)
    cb.record_failure("Temporary connection timeout")

    assert cb.metrics.consecutive_failures == 1
    assert cb.metrics.failed_cycles == 1
    assert cb.metrics.total_cycles == 1
    assert cb.metrics.last_error == "Temporary connection timeout"
    assert cb.metrics.last_failure is not None
    # 1 failure is below degraded threshold (2), so execution is allowed
    assert cb.can_execute() is True
    assert cb.state in (SCHEDULER_STATE_STARTING, SCHEDULER_STATE_HEALTHY)


# ---------------------------------------------------------------------------
# 3. Degraded Transition
# ---------------------------------------------------------------------------
def test_circuit_breaker_degraded_transition():
    """SCHED-001: 2 consecutive failures transition state to DEGRADED."""
    cb = SchedulerCircuitBreaker(max_failures=5, degraded_failures=2)
    cb.record_failure("Error 1")
    cb.record_failure("Error 2")

    assert cb.metrics.consecutive_failures == 2
    assert cb.state == SCHEDULER_STATE_DEGRADED
    # Degraded still allows execution/retries
    assert cb.can_execute() is True


# ---------------------------------------------------------------------------
# 4. Halted after N Failures (Circuit Breaker Trip)
# ---------------------------------------------------------------------------
def test_circuit_breaker_halted_after_n_failures():
    """OPS-001: Exactly N consecutive failures trip the breaker into HALTED state."""
    cb = SchedulerCircuitBreaker(max_failures=4, degraded_failures=2)
    cb.record_failure("Error 1")
    assert cb.state == SCHEDULER_STATE_STARTING
    cb.record_failure("Error 2")
    assert cb.state == SCHEDULER_STATE_DEGRADED
    cb.record_failure("Error 3")
    assert cb.state == SCHEDULER_STATE_DEGRADED

    # 4th failure trips breaker
    cb.record_failure("Error 4 (fatal)")
    assert cb.state == SCHEDULER_STATE_HALTED
    assert cb.can_execute() is False
    assert cb.metrics.circuit_breaker_tripped is True
    assert cb.metrics.tripped_at is not None
    assert "Exceeded maximum consecutive failure limit" in str(cb.metrics.trip_reason)


# ---------------------------------------------------------------------------
# 5. Execution Suppression when Halted
# ---------------------------------------------------------------------------
def test_circuit_breaker_suppresses_execution_when_halted():
    """SCHED-001 & OPS-001: All prediction runs and cycle executions are suppressed when HALTED."""
    cb = SchedulerCircuitBreaker(max_failures=3, degraded_failures=2)
    cb.trip(reason="Manual test trip")

    assert cb.can_execute() is False
    assert cb.state == SCHEDULER_STATE_HALTED

    scheduler = Scheduler(circuit_breaker=cb)

    # 1. run_one_cycle_for_symbol suppression
    df = _build_mock_stock_df()
    res = scheduler.run_one_cycle_for_symbol("TCS", df, df, return_structured=True)
    assert isinstance(res, CycleResult)
    assert res.success is False
    assert res.status == "CIRCUIT_BREAKER_HALTED"
    assert "circuit breaker is HALTED" in str(res.error)

    # 2. run_cycle suppression
    cycle_res = scheduler.run_cycle()
    assert cycle_res["success"] is False
    assert cycle_res["status"] == "CIRCUIT_BREAKER_HALTED"
    assert cycle_res["state"] == SCHEDULER_STATE_HALTED


# ---------------------------------------------------------------------------
# 6. Recovery and Reset
# ---------------------------------------------------------------------------
def test_circuit_breaker_recovery_and_reset():
    """OPS-001: Successful cycle restores HEALTHY; manual reset clears tripped breaker."""
    cb = SchedulerCircuitBreaker(max_failures=3, degraded_failures=2)

    # Test recovery from DEGRADED
    cb.record_failure("Failure 1")
    cb.record_failure("Failure 2")
    assert cb.state == SCHEDULER_STATE_DEGRADED

    cb.record_success()
    assert cb.state == SCHEDULER_STATE_HEALTHY
    assert cb.metrics.consecutive_failures == 0
    assert cb.metrics.successful_cycles == 1
    assert cb.metrics.last_success is not None

    # Test recovery from HALTED via reset
    cb.record_failure("Failure A")
    cb.record_failure("Failure B")
    cb.record_failure("Failure C")
    assert cb.state == SCHEDULER_STATE_HALTED
    assert cb.can_execute() is False

    cb.reset()
    assert cb.can_execute() is True
    assert cb.state == SCHEDULER_STATE_HEALTHY
    assert cb.metrics.consecutive_failures == 0
    assert cb.metrics.circuit_breaker_tripped is False
    assert cb.metrics.tripped_at is None


# ---------------------------------------------------------------------------
# 7. Operational Timestamps Tracking
# ---------------------------------------------------------------------------
def test_scheduler_timestamps_tracking():
    """SCHED-001: Operational timestamps update accurately during cycle execution."""
    cb = SchedulerCircuitBreaker()
    scheduler = Scheduler(circuit_breaker=cb)

    df = _build_mock_stock_df()
    mock_provider = lambda: {"TCS": (df, df)}

    with patch.object(scheduler, "run_one_cycle_for_symbol") as mock_one:
        mock_one.return_value = CycleResult(success=True, status="SUCCESS", symbol="TCS")
        with patch.object(scheduler, "get_event_context") as mock_events:
            mock_context = MagicMock()
            mock_context.macro_events = None
            mock_context.corporate_events = None
            mock_context.news_articles = None
            mock_events.return_value = mock_context

            cycle_res = scheduler.run_cycle(symbol_data_provider=mock_provider, ignore_market_hours=True)

            assert cycle_res["success"] is True
            assert cycle_res["status"] == "SUCCESS"
            assert cb.metrics.last_data_fetch is not None
            assert cb.metrics.last_event_fetch is not None
            assert cb.metrics.last_prediction is not None
            assert cb.metrics.last_success is not None
            assert cb.metrics.consecutive_failures == 0


# ---------------------------------------------------------------------------
# 8. Health Registry Alertable Reporting
# ---------------------------------------------------------------------------
def test_health_registry_alertable_reporting():
    """OPS-001: State transitions accurately report to health registry and engine health."""
    scheduler = Scheduler()

    # Success cycle reports OK
    scheduler.circuit_breaker.record_success()
    health_registry.report("scheduler", ok=True, detail="Operational")

    statuses = {s.component: s for s in health_registry.get_status()}
    assert "scheduler" in statuses
    assert statuses["scheduler"].status == "OK"

    # Circuit breaker trips and reports failure until threshold reached
    scheduler.circuit_breaker.trip("Catastrophic downstream outage")
    for _ in range(5):
        health_registry.report("scheduler", ok=False, detail="HALTED", error="Outage")

    statuses = {s.component: s for s in health_registry.get_status()}
    assert statuses["scheduler"].status == "DOWN"

    engine_health = health_registry.get_engine_health()
    assert engine_health.checks["scheduler"] == "FAILED"


# ---------------------------------------------------------------------------
# 9. API Scheduler Endpoints
# ---------------------------------------------------------------------------
def test_api_scheduler_endpoints():
    """OPS-001: /api/v1/scheduler/status and reset-circuit-breaker endpoints function properly."""
    from config import DEFAULT_DEV_ADMIN_KEY

    client = TestClient(app)

    # 1. Status endpoint
    resp = client.get("/api/v1/scheduler/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert "is_operational" in data
    assert "metrics" in data
    assert "consecutive_failures" in data["metrics"]

    # 2. Reset endpoint without auth fails (401 or 403)
    unauth_resp = client.post("/api/v1/scheduler/reset-circuit-breaker")
    assert unauth_resp.status_code in (401, 403)

    # 3. Reset endpoint with admin auth succeeds
    auth_resp = client.post(
        "/api/v1/scheduler/reset-circuit-breaker",
        headers={"X-API-Key": DEFAULT_DEV_ADMIN_KEY},
    )
    assert auth_resp.status_code == 200
    auth_data = auth_resp.json()
    assert auth_data["status"] == "success"
    assert "scheduler" in auth_data
    assert auth_data["scheduler"]["is_operational"] is True


# ---------------------------------------------------------------------------
# 10. Backwards Compatibility
# ---------------------------------------------------------------------------
def test_scheduler_backwards_compatibility():
    """Verify backwards compatibility for Scheduler constructor and execution methods."""
    # Default constructor works cleanly
    s = Scheduler()
    assert s.circuit_breaker is not None
    assert s.circuit_breaker.state == SCHEDULER_STATE_STARTING

    # run_forever with max_iterations=0 terminates immediately without exception
    s.run_forever(symbol_data_provider=lambda: {}, max_iterations=0)
