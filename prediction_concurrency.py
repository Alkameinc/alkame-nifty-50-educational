"""
prediction_concurrency.py
Phase 16: Concurrency & Idempotency Layer (SCHED-002, P1)

Enforces:
1. Deterministic Idempotency Key:
   prediction_key = hash(symbol, timestamp, model_version, feature_schema_hash)
2. In-Flight Execution Deduplication:
   Prevents two independent prediction cycles for the same key from executing simultaneously.
3. Thread-safe coordination across concurrent scheduler cycles, streaming runs, and API requests.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_timestamp_for_key(timestamp: datetime | str | pd.Timestamp) -> str:
    """Normalizes any datetime, string, or pd.Timestamp into a standard UTC ISO-8601 string."""
    if isinstance(timestamp, str):
        try:
            ts = pd.Timestamp(timestamp)
        except Exception:
            return timestamp.strip()
    else:
        ts = pd.Timestamp(timestamp)

    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_prediction_key(
    symbol: str,
    timestamp: datetime | str | pd.Timestamp,
    model_version: str,
    feature_schema_hash: str,
) -> str:
    """SCHED-002: Computes a deterministic SHA256 idempotency key:

    prediction_key = hash(symbol, timestamp, model_version, feature_schema_hash)
    """
    sym = symbol.strip().upper()
    ts_str = normalize_timestamp_for_key(timestamp)
    mv = (model_version or "UNKNOWN").strip()
    fsh = (feature_schema_hash or "").strip().lower()

    payload = f"{sym}:{ts_str}:{mv}:{fsh}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PredictionConcurrencyMetrics:
    total_requests: int = 0
    deduplicated_requests: int = 0
    simultaneous_waits: int = 0
    completed_executions: int = 0
    in_flight_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_requests": self.total_requests,
            "deduplicated_requests": self.deduplicated_requests,
            "simultaneous_waits": self.simultaneous_waits,
            "completed_executions": self.completed_executions,
            "in_flight_count": self.in_flight_count,
        }


class PredictionConcurrencyCoordinator:
    """Thread-safe coordinator for simultaneous prediction cycles and in-memory deduplication.

    Prevents multiple threads from running redundant model inference or feature
    engineering for the exact same (symbol, timestamp, model_version, feature_schema_hash).
    """

    def __init__(
        self,
        history_manager: Any | None = None,
        cache_ttl_seconds: float = 300.0,
        max_cache_entries: int = 2000,
    ):
        self.history_manager = history_manager
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_entries = max_cache_entries

        self._lock = threading.RLock()
        self._in_flight: dict[str, threading.Event] = {}
        self._in_flight_results: dict[str, Any] = {}
        self._in_flight_errors: dict[str, Exception] = {}
        self._recent_cache: dict[str, tuple[float, Any]] = {}
        self.metrics = PredictionConcurrencyMetrics()

    def get_metrics(self) -> dict[str, int]:
        with self._lock:
            self.metrics.in_flight_count = len(self._in_flight)
            return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        with self._lock:
            self.metrics = PredictionConcurrencyMetrics(in_flight_count=len(self._in_flight))

    def clear_cache(self) -> None:
        with self._lock:
            self._recent_cache.clear()
            self._in_flight_results.clear()
            self._in_flight_errors.clear()

    def _prune_cache_if_needed(self, now: float) -> None:
        """Prune expired cache entries and bound memory size."""
        if len(self._recent_cache) > self.max_cache_entries:
            expired = [k for k, (t, _) in self._recent_cache.items() if (now - t) > self.cache_ttl_seconds]
            for k in expired:
                self._recent_cache.pop(k, None)
            if len(self._recent_cache) > self.max_cache_entries:
                # Evict oldest 20%
                sorted_keys = sorted(self._recent_cache.keys(), key=lambda k: self._recent_cache[k][0])
                for k in sorted_keys[: len(sorted_keys) // 5]:
                    self._recent_cache.pop(k, None)

    def execute_or_wait(
        self,
        key: str,
        execute_fn: Callable[[], Any],
        timeout: float = 30.0,
    ) -> Any:
        """Executes execute_fn if this key is not currently computing and not cached.

        If currently computing in another thread, waits on that thread's execution
        and returns the same result. If already completed, returns the cached result.
        """
        now = time.time()

        with self._lock:
            self._prune_cache_if_needed(now)

            # 1. Check recent in-memory cache
            if key in self._recent_cache:
                saved_time, cached_result = self._recent_cache[key]
                if now - saved_time <= self.cache_ttl_seconds:
                    self.metrics.total_requests += 1
                    self.metrics.deduplicated_requests += 1
                    logger.debug(f"[SCHED-002] In-memory cache hit for key={key[:12]}")
                    return cached_result

            # 2. Check in-flight simultaneous execution
            if key in self._in_flight:
                self.metrics.total_requests += 1
                self.metrics.simultaneous_waits += 1
                self.metrics.deduplicated_requests += 1
                event = self._in_flight[key]
                logger.info(f"[SCHED-002] Awaiting simultaneous in-flight cycle for key={key[:12]}")
                # Wait outside lock
                wait_needed = True
            else:
                # 3. We are the executor
                self.metrics.total_requests += 1
                event = threading.Event()
                self._in_flight[key] = event
                self.metrics.in_flight_count = len(self._in_flight)
                wait_needed = False

        if wait_needed:
            finished = event.wait(timeout=timeout)
            if not finished:
                raise TimeoutError(
                    f"Timed out waiting for simultaneous cycle with prediction_key={key} after {timeout}s."
                )
            with self._lock:
                if key in self._in_flight_errors:
                    raise self._in_flight_errors[key]
                if key in self._in_flight_results:
                    return self._in_flight_results[key]
                if key in self._recent_cache:
                    return self._recent_cache[key][1]
                raise RuntimeError(f"Simultaneous cycle completed for key={key[:12]} but no result found.")

        # Executor path
        try:
            result = execute_fn()
            with self._lock:
                self._in_flight_results[key] = result
                self._recent_cache[key] = (time.time(), result)
                self.metrics.completed_executions += 1
            return result
        except Exception as e:
            with self._lock:
                self._in_flight_errors[key] = e
            logger.error(f"[SCHED-002] Error executing cycle for key={key[:12]}: {e}")
            raise
        finally:
            with self._lock:
                event.set()
                self._in_flight.pop(key, None)
                self.metrics.in_flight_count = len(self._in_flight)
