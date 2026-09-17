# 1. Standard library imports
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any

# 2. Local imports
from config import (
    SCHEDULER_CIRCUIT_BREAKER_DEGRADED_FAILURES,
    SCHEDULER_CIRCUIT_BREAKER_MAX_FAILURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Constants (SCHED-001, OPS-001)
# ---------------------------------------------------------------------------
SCHEDULER_STATE_STARTING = "STARTING"
SCHEDULER_STATE_HEALTHY = "HEALTHY"
SCHEDULER_STATE_DEGRADED = "DEGRADED"
SCHEDULER_STATE_FAILED = "FAILED"
SCHEDULER_STATE_HALTED = "HALTED"

VALID_SCHEDULER_STATES = {
    SCHEDULER_STATE_STARTING,
    SCHEDULER_STATE_HEALTHY,
    SCHEDULER_STATE_DEGRADED,
    SCHEDULER_STATE_FAILED,
    SCHEDULER_STATE_HALTED,
}


# ---------------------------------------------------------------------------
# Operational Metrics Tracking
# ---------------------------------------------------------------------------
@dataclass
class SchedulerMetrics:
    """
    SCHED-001: Explicit tracking of operational metrics, timestamps, and error states.
    Ensures process liveness is not conflated with scheduler health.
    """
    consecutive_failures: int = 0
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    last_prediction: datetime | None = None
    last_data_fetch: datetime | None = None
    last_event_fetch: datetime | None = None
    last_error: str | None = None
    circuit_breaker_tripped: bool = False
    tripped_at: datetime | None = None
    trip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "consecutive_failures": self.consecutive_failures,
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_prediction": self.last_prediction.isoformat() if self.last_prediction else None,
            "last_data_fetch": self.last_data_fetch.isoformat() if self.last_data_fetch else None,
            "last_event_fetch": self.last_event_fetch.isoformat() if self.last_event_fetch else None,
            "last_error": self.last_error,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "tripped_at": self.tripped_at.isoformat() if self.tripped_at else None,
            "trip_reason": self.trip_reason,
        }


# ---------------------------------------------------------------------------
# Circuit Breaker Engine
# ---------------------------------------------------------------------------
class SchedulerCircuitBreaker:
    """
    OPS-001: Circuit breaker and alertable state machine governing scheduler execution.
    Policy:
      1 failure  -> retry (status remains STARTING/HEALTHY)
      2 failures -> DEGRADED
      N failures -> HALTED (circuit breaker trips, suspending cycle execution)
    """

    def __init__(
        self,
        max_failures: int | None = None,
        degraded_failures: int | None = None,
    ):
        self.max_failures = (
            max_failures if max_failures is not None else SCHEDULER_CIRCUIT_BREAKER_MAX_FAILURES
        )
        self.degraded_failures = (
            degraded_failures if degraded_failures is not None else SCHEDULER_CIRCUIT_BREAKER_DEGRADED_FAILURES
        )
        self.state = SCHEDULER_STATE_STARTING
        self.metrics = SchedulerMetrics()

    def record_success(self, timestamp: datetime | None = None) -> None:
        """Records a successful scheduler cycle and resets failure count."""
        now = timestamp or datetime.now(timezone.utc)
        self.metrics.consecutive_failures = 0
        self.metrics.successful_cycles += 1
        self.metrics.total_cycles += 1
        self.metrics.last_success = now
        self.metrics.last_error = None

        if self.state != SCHEDULER_STATE_HALTED:
            self.state = SCHEDULER_STATE_HEALTHY

    def record_failure(self, error: str, timestamp: datetime | None = None) -> None:
        """
        Records a failed cycle, updates error state, and transitions state machine.
        Trips breaker to HALTED if consecutive failures exceed max_failures.
        """
        now = timestamp or datetime.now(timezone.utc)
        self.metrics.consecutive_failures += 1
        self.metrics.failed_cycles += 1
        self.metrics.total_cycles += 1
        self.metrics.last_failure = now
        self.metrics.last_error = error

        if self.metrics.consecutive_failures >= self.max_failures:
            self.trip(
                reason=(
                    f"Exceeded maximum consecutive failure limit "
                    f"({self.metrics.consecutive_failures}/{self.max_failures}): {error}"
                ),
                timestamp=now,
            )
        elif self.metrics.consecutive_failures >= self.degraded_failures:
            self.state = SCHEDULER_STATE_DEGRADED
            logger.warning(
                f"Scheduler degraded: {self.metrics.consecutive_failures} consecutive failures. Error: {error}"
            )

    def trip(self, reason: str, timestamp: datetime | None = None) -> None:
        """Explicitly trips the circuit breaker to HALTED state."""
        now = timestamp or datetime.now(timezone.utc)
        self.state = SCHEDULER_STATE_HALTED
        self.metrics.circuit_breaker_tripped = True
        self.metrics.tripped_at = now
        self.metrics.trip_reason = reason
        logger.critical(f"Scheduler circuit breaker TRIPPED into HALTED state: {reason}")

    def reset(self) -> None:
        """Manually resets circuit breaker and failure counters to resume execution."""
        self.state = SCHEDULER_STATE_HEALTHY if self.metrics.successful_cycles > 0 else SCHEDULER_STATE_STARTING
        self.metrics.consecutive_failures = 0
        self.metrics.circuit_breaker_tripped = False
        self.metrics.tripped_at = None
        self.metrics.trip_reason = None
        self.metrics.last_error = None
        logger.info("Scheduler circuit breaker reset to operative state.")

    def can_execute(self) -> bool:
        """Returns True if scheduler is allowed to execute cycles, False if HALTED."""
        return self.state != SCHEDULER_STATE_HALTED

    def get_status(self) -> dict[str, Any]:
        """Returns complete serialized status payload."""
        return {
            "state": self.state,
            "is_operational": self.can_execute(),
            "max_failures": self.max_failures,
            "degraded_failures": self.degraded_failures,
            "metrics": self.metrics.to_dict(),
        }
