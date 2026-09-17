"""
tests/test_phase16_concurrency.py
Phase 16 Test Suite: Concurrency & Idempotency (SCHED-002, P1)

Verifies:
1. Deterministic idempotency key computation: hash(symbol, timestamp, model_version, feature_schema_hash).
2. Deduplication of simultaneous prediction cycles across concurrent threads.
3. Storage idempotency: duplicate prediction writes are suppressed and existing IDs returned.
4. Prediction retrieval and signal reconstruction by prediction_key.
5. In-flight synchronization and thread-safe exception handling without deadlock.
6. Scheduler integration: run_one_cycle_for_symbol deduplication and concurrency metrics.
7. REST API reporting of scheduler concurrency diagnostics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from config import DB_PATH
from history_manager import HistoryManager, PredictionRecord
from model_lineage import compute_feature_schema_hash
from models import Base
from prediction_concurrency import (
    PredictionConcurrencyCoordinator,
    PredictionConcurrencyMetrics,
    compute_prediction_key,
    normalize_timestamp_for_key,
)
from predictor import MultiHorizonSignal, PredictionSignal, Predictor
from scheduler import CycleResult, Scheduler


def _make_dummy_ohlcv(n_bars: int = 50, start_date: str = "2026-07-01 09:15:00") -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=n_bars, freq="5min", tz="Asia/Kolkata")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_p = close + np.random.randn(n_bars) * 0.1
    vol = np.random.randint(1000, 10000, size=n_bars)
    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": vol,
        },
        index=dates,
    )


@pytest.fixture
def temp_history_manager(tmp_path: Path):
    db_file = tmp_path / "test_concurrency.db"
    mgr = HistoryManager(db_path=db_file)
    return mgr


# ---------------------------------------------------------------------------
# 1. Deterministic Idempotency Key Computation
# ---------------------------------------------------------------------------
def test_prediction_key_deterministic_and_unique():
    ts1 = "2026-07-01T09:15:00Z"
    ts2 = pd.Timestamp("2026-07-01 09:15:00", tz="UTC")
    ts3 = datetime(2026, 7, 1, 9, 15, 0, tzinfo=timezone.utc)

    schema_hash = compute_feature_schema_hash(["rsi_feat", "sma_feat", "vol_feat"])

    key1 = compute_prediction_key("RELIANCE", ts1, "v1.0", schema_hash)
    key2 = compute_prediction_key("reliance", ts2, "v1.0", schema_hash)
    key3 = compute_prediction_key("RELIANCE", ts3, "v1.0", schema_hash)

    # Identical inputs across different timestamp formats produce identical 64-char key
    assert len(key1) == 64
    assert key1 == key2
    assert key1 == key3

    # Changing symbol produces distinct key
    key_tcs = compute_prediction_key("TCS", ts1, "v1.0", schema_hash)
    assert key_tcs != key1

    # Changing model version produces distinct key
    key_v2 = compute_prediction_key("RELIANCE", ts1, "v2.0", schema_hash)
    assert key_v2 != key1

    # Changing timestamp produces distinct key
    key_diff_time = compute_prediction_key("RELIANCE", "2026-07-01T09:20:00Z", "v1.0", schema_hash)
    assert key_diff_time != key1

    # Changing feature schema produces distinct key
    schema_hash2 = compute_feature_schema_hash(["rsi_feat", "macd_feat"])
    key_diff_schema = compute_prediction_key("RELIANCE", ts1, "v1.0", schema_hash2)
    assert key_diff_schema != key1


# ---------------------------------------------------------------------------
# 2. Storage Idempotency & Duplicate Write Prevention
# ---------------------------------------------------------------------------
def test_storage_idempotency_prevents_duplicate_records(temp_history_manager):
    hm = temp_history_manager
    ts = datetime(2026, 7, 1, 9, 15, 0, tzinfo=timezone.utc)
    key = compute_prediction_key("RELIANCE", ts, "v1.0", "abc123hash")

    sig = PredictionSignal(
        symbol="RELIANCE",
        timestamp=ts,
        horizon="INTRADAY",
        action="BUY",
        model_predicted_class="UP",
        model_version="v1.0",
        feature_version="v1.0",
        raw_confidence=0.75,
        risk_adjusted_confidence=0.70,
        calibrated_confidence=0.68,
        agreement_fraction=0.80,
        downside_summary="Downside risk 1%",
        upside_summary="Upside target 2%",
        prediction_key=key,
        feature_schema_hash="abc123hash",
    )

    # First write
    id1 = hm.save_prediction(sig)
    assert id1 is not None
    assert id1 > 0

    # Second write with exact same prediction_key
    id2 = hm.save_prediction(sig)
    assert id2 == id1, "Duplicate write should return the existing record ID"

    # Verify exactly one record exists in database
    records = hm.get_predictions(symbol="RELIANCE")
    assert len(records) == 1
    assert records[0].id == id1
    assert records[0].prediction_key == key


# ---------------------------------------------------------------------------
# 3. Retrieval and Signal Reconstruction
# ---------------------------------------------------------------------------
def test_get_prediction_by_key_and_signal_reconstruction(temp_history_manager):
    hm = temp_history_manager
    ts = datetime(2026, 7, 1, 9, 30, 0, tzinfo=timezone.utc)
    key = compute_prediction_key("INFY", ts, "v1.0", "infy_schema_hash")

    sig = PredictionSignal(
        symbol="INFY",
        timestamp=ts,
        horizon="INTRADAY",
        action="SELL",
        model_predicted_class="DOWN",
        model_version="v1.0",
        feature_version="v1.0",
        raw_confidence=0.65,
        risk_adjusted_confidence=0.60,
        calibrated_confidence=None,
        agreement_fraction=0.75,
        downside_summary="Downside expected",
        upside_summary="Upside limited",
        reasoning=["Trend breakdown", "High volume selling"],
        prediction_key=key,
        feature_schema_hash="infy_schema_hash",
    )

    hm.save_prediction(sig)

    # Query record by key
    rec = hm.get_prediction_by_key(key)
    assert rec is not None
    assert rec.symbol == "INFY"
    assert rec.prediction_key == key
    assert rec.action == "SELL"

    # Non-existent key returns None
    assert hm.get_prediction_by_key("non_existent_key") is None

    # Reconstruct PredictionSignal from stored record
    reconstructed_sig = hm.get_prediction_signal_by_key(key)
    assert reconstructed_sig is not None
    assert reconstructed_sig.symbol == "INFY"
    assert reconstructed_sig.action == "SELL"
    assert reconstructed_sig.prediction_key == key
    assert "Trend breakdown" in reconstructed_sig.reasoning


# ---------------------------------------------------------------------------
# 4. Coordinator Cache Hit & Sequential Deduplication
# ---------------------------------------------------------------------------
def test_coordinator_cache_hit_returns_existing():
    coordinator = PredictionConcurrencyCoordinator()
    key = "test_cache_key_123"

    call_count = 0

    def compute_fn():
        nonlocal call_count
        call_count += 1
        return {"status": "SUCCESS", "data": 42}

    # First execution
    res1 = coordinator.execute_or_wait(key, compute_fn)
    assert res1 == {"status": "SUCCESS", "data": 42}
    assert call_count == 1

    # Second execution: immediate cache hit, compute_fn not called
    res2 = coordinator.execute_or_wait(key, compute_fn)
    assert res2 == {"status": "SUCCESS", "data": 42}
    assert call_count == 1

    metrics = coordinator.get_metrics()
    assert metrics["total_requests"] == 2
    assert metrics["deduplicated_requests"] == 1
    assert metrics["completed_executions"] == 1


# ---------------------------------------------------------------------------
# 5. Multi-Threaded Simultaneous Execution Deduplication
# ---------------------------------------------------------------------------
def test_simultaneous_execution_deduplication_multithreaded():
    coordinator = PredictionConcurrencyCoordinator()
    key = "concurrent_execution_key_999"

    call_count = 0
    lock = threading.Lock()

    def slow_compute():
        nonlocal call_count
        time.sleep(0.1)  # Simulate expensive model inference
        with lock:
            call_count += 1
        return {"result_id": "computed_once"}

    results = []
    threads = []
    n_threads = 8

    def worker():
        res = coordinator.execute_or_wait(key, slow_compute)
        results.append(res)

    for _ in range(n_threads):
        t = threading.Thread(target=worker)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5.0)

    # Verify all threads finished
    assert len(results) == n_threads
    for r in results:
        assert r == {"result_id": "computed_once"}

    # Invariant: slow_compute was invoked EXACTLY ONCE
    assert call_count == 1

    metrics = coordinator.get_metrics()
    assert metrics["total_requests"] == n_threads
    assert metrics["completed_executions"] == 1
    assert metrics["simultaneous_waits"] == n_threads - 1
    assert metrics["deduplicated_requests"] == n_threads - 1


# ---------------------------------------------------------------------------
# 6. Simultaneous Execution Error Propagation Without Deadlock
# ---------------------------------------------------------------------------
def test_simultaneous_execution_propagates_exception():
    coordinator = PredictionConcurrencyCoordinator()
    key = "error_propagation_key_111"

    def faulty_compute():
        time.sleep(0.05)
        raise ValueError("Simulated model inference crash")

    errors = []
    threads = []
    n_threads = 4

    def worker():
        try:
            coordinator.execute_or_wait(key, faulty_compute)
        except Exception as e:
            errors.append(e)

    for _ in range(n_threads):
        t = threading.Thread(target=worker)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5.0)

    # All threads must receive the exception without hanging
    assert len(errors) == n_threads
    for err in errors:
        assert isinstance(err, ValueError)
        assert "Simulated model inference crash" in str(err)

    # Coordinator must not have leftover in_flight locks
    assert coordinator.get_metrics()["in_flight_count"] == 0


# ---------------------------------------------------------------------------
# 7. Scheduler Integration: run_one_cycle_for_symbol Deduplication
# ---------------------------------------------------------------------------
def test_scheduler_run_one_cycle_concurrency_deduplication(temp_history_manager):
    hm = temp_history_manager
    sched = Scheduler(history_manager=hm)

    df = _make_dummy_ohlcv(20)
    index_df = df.copy()

    # First cycle
    res1 = sched.run_one_cycle_for_symbol(
        "RELIANCE", df, index_df, macro_events=[], corporate_events=[], news_articles=[], return_structured=True
    )
    assert isinstance(res1, CycleResult)
    assert res1.success is True

    # Second cycle with exact same dataframe (same bar timestamp)
    res2 = sched.run_one_cycle_for_symbol(
        "RELIANCE", df, index_df, macro_events=[], corporate_events=[], news_articles=[], return_structured=True
    )
    assert isinstance(res2, CycleResult)
    assert res2.success is True

    # Verify coordinator registered the duplicate request
    metrics = sched.concurrency_coordinator.get_metrics()
    assert metrics["total_requests"] == 2
    assert metrics["deduplicated_requests"] == 1


# ---------------------------------------------------------------------------
# 8. Scheduler Concurrent Multi-Threaded Cycles
# ---------------------------------------------------------------------------
def test_scheduler_concurrent_threads_cycle_execution(temp_history_manager):
    hm = temp_history_manager
    sched = Scheduler(history_manager=hm)

    df = _make_dummy_ohlcv(20)
    index_df = df.copy()

    results = []
    threads = []
    n_threads = 5

    def worker():
        res = sched.run_one_cycle_for_symbol(
            "TCS", df, index_df, macro_events=[], corporate_events=[], news_articles=[], return_structured=True
        )
        results.append(res)

    for _ in range(n_threads):
        t = threading.Thread(target=worker)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    assert len(results) == n_threads
    for r in results:
        assert isinstance(r, CycleResult)
        assert r.success is True

    # Exactly 1 completed execution, 4 deduplicated/waited
    metrics = sched.concurrency_coordinator.get_metrics()
    assert metrics["completed_executions"] == 1
    assert metrics["deduplicated_requests"] == n_threads - 1


# ---------------------------------------------------------------------------
# 9. MultiHorizonSignal Prediction Key Population
# ---------------------------------------------------------------------------
def test_multi_horizon_signal_contains_prediction_keys(temp_history_manager):
    hm = temp_history_manager
    sched = Scheduler(history_manager=hm)

    df = _make_dummy_ohlcv(30)
    index_df = df.copy()

    res = sched.run_one_cycle_for_symbol(
        "INFY", df, index_df, macro_events=[], corporate_events=[], news_articles=[], return_structured=True
    )
    assert isinstance(res, CycleResult)
    assert res.success is True
    assert res.signal is not None

    multi_sig = res.signal
    assert multi_sig.prediction_key is not None
    assert len(multi_sig.prediction_key) == 64

    # Each individual horizon signal has its own prediction_key and feature_schema_hash
    for horizon, sig in multi_sig.signals.items():
        assert sig.prediction_key is not None
        assert len(sig.prediction_key) == 64
        assert sig.feature_schema_hash is not None


# ---------------------------------------------------------------------------
# 10. API Diagnostics: Scheduler Status Concurrency Metrics
# ---------------------------------------------------------------------------
def test_scheduler_status_api_reports_concurrency_metrics():
    client = TestClient(app)
    response = client.get("/api/v1/scheduler/status")
    assert response.status_code == 200
    data = response.json()

    assert "concurrency" in data
    concurrency_data = data["concurrency"]
    assert "total_requests" in concurrency_data
    assert "deduplicated_requests" in concurrency_data
    assert "simultaneous_waits" in concurrency_data
    assert "completed_executions" in concurrency_data
    assert "in_flight_count" in concurrency_data
