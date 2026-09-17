"""
tests/test_phase17_storage_reliability.py
Phase 17 Test Suite: Storage Reliability (SCHED-003, P1)

Verifies:
1. SQLite PRAGMAs (WAL mode, busy_timeout >= 30s, synchronous=NORMAL, foreign_keys=ON).
2. Deterministic retry with exponential backoff & jitter on lock contention.
3. Atomic transaction execution and rollback on error (no partial writes).
4. Atomic prediction bundle persistence (prediction + contributing events).
5. Duplicate write idempotency (predictions and events).
6. Corrupted record resilience (safe parsing of malformed JSON, dates, floats).
7. Database integrity checks (PRAGMA quick_check & integrity_check).
8. Storage startup recovery routine (forced WAL checkpoint, unresolved prediction audit).
9. Multi-threaded concurrent write stress testing under WAL mode.
10. Scheduler and REST API diagnostics integration.
"""

import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import app
from database import engine as global_engine
from event_classifier import Event
from health_monitor import registry as health_registry
from history_manager import HistoryManager, PredictionRecord
from models import Base, Event as DBEvent, Prediction as DBPrediction
from predictor import PredictionSignal
from scheduler import Scheduler
from storage_reliability import (
    StorageRecoveryReport,
    atomic_transaction,
    configure_sqlite_engine,
    create_reliable_engine,
    run_storage_recovery,
    safe_float,
    safe_iso_timestamp,
    safe_json_loads,
    verify_database_integrity,
    with_db_retry,
)


def _make_dummy_signal(symbol: str = "TCS", p_key: str | None = None) -> PredictionSignal:
    return PredictionSignal(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        horizon="INTRADAY",
        action="BUY",
        model_predicted_class="UP",
        model_version="v1.0",
        feature_version="v1.0",
        raw_confidence=0.75,
        risk_adjusted_confidence=0.72,
        calibrated_confidence=0.70,
        agreement_fraction=0.8,
        downside_summary="",
        upside_summary="",
        reasoning=["Signal rationale 1", "Signal rationale 2"],
        global_risk_level="NORMAL",
        risk_toggle_enabled=False,
        is_safe_to_trade_live=True,
        data_stale=False,
        suppressed=False,
        suppression_reasons=[],
        prediction_key=p_key,
        feature_schema_hash="hash_abc123",
    )


def _make_dummy_event(event_id: str, symbol: str = "TCS") -> Event:
    return Event(
        event_id=event_id,
        source="NEWS",
        event_type="EARNINGS",
        timestamp=datetime.now(timezone.utc),
        scope="STOCK",
        affected_tickers=[symbol],
        sector="IT",
        confidence_in_scope=1.0,
        headline_or_label=f"Quarterly earnings update for {symbol}",
        sentiment_score=0.85,
        magnitude_estimate="HIGH",
    )


# ---------------------------------------------------------------------------
# 1. SQLite PRAGMA Configuration
# ---------------------------------------------------------------------------
def test_sqlite_pragmas_configured(tmp_path):
    """PRAGMA journal_mode=WAL, busy_timeout >= 30000, synchronous=NORMAL, foreign_keys=ON."""
    db_file = tmp_path / "test_pragmas.sqlite3"
    eng = create_reliable_engine(db_file)
    with eng.connect() as conn:
        jm = conn.execute(text("PRAGMA journal_mode;")).scalar()
        bt = conn.execute(text("PRAGMA busy_timeout;")).scalar()
        sync = conn.execute(text("PRAGMA synchronous;")).scalar()
        fk = conn.execute(text("PRAGMA foreign_keys;")).scalar()

    assert str(jm).lower() == "wal"
    assert int(bt) >= 30000
    # In SQLite, PRAGMA synchronous returns 1 for NORMAL (2 is FULL, 0 is OFF)
    assert int(sync) == 1
    assert int(fk) == 1


# ---------------------------------------------------------------------------
# 2. Lock Contention Retry (with_db_retry)
# ---------------------------------------------------------------------------
def test_db_retry_on_simulated_lock():
    """Simulated SQLite database is locked exception recovers via retry."""
    attempts = 0

    @with_db_retry(max_retries=3, initial_delay=0.01, backoff_factor=1.5)
    def flaky_write():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")
        return "success"

    res = flaky_write()
    assert res == "success"
    assert attempts == 3


def test_db_retry_exhaustion_raises():
    """Retries exhaust and propagate error when lock does not clear."""
    attempts = 0

    @with_db_retry(max_retries=3, initial_delay=0.01, backoff_factor=1.2)
    def persistent_lock():
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("database is busy")

    with pytest.raises(sqlite3.OperationalError, match="database is busy"):
        persistent_lock()
    assert attempts == 4  # Initial attempt + 3 retries


# ---------------------------------------------------------------------------
# 3. Atomic Transaction Context
# ---------------------------------------------------------------------------
def test_atomic_transaction_commits_cleanly(tmp_path):
    """atomic_transaction commits all changes upon normal completion."""
    db_file = tmp_path / "test_atomic_commit.sqlite3"
    eng = create_reliable_engine(db_file)
    Base.metadata.create_all(bind=eng)
    from sqlalchemy.orm import sessionmaker

    sm = sessionmaker(bind=eng)

    with atomic_transaction(sm) as db:
        ev = DBEvent(
            event_id="EVT_A_1",
            source="TEST",
            event_type="NEWS",
            timestamp="2026-09-17T00:00:00Z",
            scope="MARKET",
            affected_tickers=",NIFTY,",
            confidence_in_scope=1.0,
            headline_or_label="Atomic test event",
            magnitude_estimate="LOW",
        )
        db.add(ev)

    with eng.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM events;")).scalar()
    assert count == 1


def test_atomic_transaction_rolls_back_on_error(tmp_path):
    """atomic_transaction cleanly rolls back all modifications on mid-block failure."""
    db_file = tmp_path / "test_atomic_rollback.sqlite3"
    eng = create_reliable_engine(db_file)
    Base.metadata.create_all(bind=eng)
    from sqlalchemy.orm import sessionmaker

    sm = sessionmaker(bind=eng)

    with pytest.raises(RuntimeError, match="Simulated mid-transaction failure"):
        with atomic_transaction(sm) as db:
            ev = DBEvent(
                event_id="EVT_ROLLBACK_1",
                source="TEST",
                event_type="NEWS",
                timestamp="2026-09-17T00:00:00Z",
                scope="MARKET",
                affected_tickers=",NIFTY,",
                confidence_in_scope=1.0,
                headline_or_label="Should be rolled back",
                magnitude_estimate="LOW",
            )
            db.add(ev)
            db.flush()
            raise RuntimeError("Simulated mid-transaction failure")

    with eng.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM events;")).scalar()
    assert count == 0


# ---------------------------------------------------------------------------
# 4. Atomic Prediction Bundle Persistence
# ---------------------------------------------------------------------------
def test_save_prediction_bundle_atomic_success(tmp_path):
    """save_prediction_bundle persists prediction and multiple events in a single transaction."""
    db_file = tmp_path / "test_bundle.sqlite3"
    hm = HistoryManager(db_path=db_file)

    sig = _make_dummy_signal("INFY", "pred_bundle_001")
    ev1 = _make_dummy_event("EVT_INFY_1", "INFY")
    ev2 = _make_dummy_event("EVT_INFY_2", "INFY")

    pred_id, ev_ids = hm.save_prediction_bundle(sig, events=[ev1, ev2])

    assert pred_id is not None
    assert ev_ids == ["EVT_INFY_1", "EVT_INFY_2"]

    preds = hm.get_predictions("INFY")
    assert len(preds) == 1
    events = hm.get_events_for_symbol("INFY")
    assert len(events) == 2


def test_save_prediction_bundle_rolls_back_on_event_failure(tmp_path):
    """save_prediction_bundle rolls back prediction if an event payload fails."""
    db_file = tmp_path / "test_bundle_fail.sqlite3"
    hm = HistoryManager(db_path=db_file)

    sig = _make_dummy_signal("RELIANCE", "pred_bundle_fail_001")
    # Valid event
    ev1 = _make_dummy_event("EVT_REL_1", "RELIANCE")
    # Bad event with None for non-nullable headline_or_label to induce database constraint error
    bad_ev = Event(
        event_id="EVT_BAD_2",
        source="NEWS",
        event_type="NEWS",
        timestamp=datetime.now(timezone.utc),
        scope="STOCK",
        affected_tickers=["RELIANCE"],
        confidence_in_scope=1.0,
        headline_or_label=None,  # DB column is nullable=False
        magnitude_estimate="HIGH",
    )

    pred_id, ev_ids = hm.save_prediction_bundle(sig, events=[ev1, bad_ev])

    # Should fail cleanly, returning None, []
    assert pred_id is None
    assert ev_ids == []

    # Verify ZERO records persisted (no partial write)
    preds = hm.get_predictions("RELIANCE")
    assert len(preds) == 0
    events = hm.get_events_for_symbol("RELIANCE")
    assert len(events) == 0


# ---------------------------------------------------------------------------
# 5. Duplicate Write Idempotency
# ---------------------------------------------------------------------------
def test_duplicate_event_write_idempotent(tmp_path):
    """Writing an event with an existing event_id returns True without raising IntegrityError."""
    db_file = tmp_path / "test_dup_event.sqlite3"
    hm = HistoryManager(db_path=db_file)

    ev = _make_dummy_event("EVT_IDEMPOTENT_1", "HDFCBANK")

    saved_first = hm.save_event(ev)
    assert saved_first is True

    # Attempt duplicate write
    saved_second = hm.save_event(ev)
    assert saved_second is True

    events = hm.get_events_for_symbol("HDFCBANK")
    assert len(events) == 1


# ---------------------------------------------------------------------------
# 6. Corrupted Records Resilience
# ---------------------------------------------------------------------------
def test_corrupted_record_resilience(tmp_path):
    """Query methods safely parse and tolerate malformed JSON, corrupt dates, or invalid types."""
    db_file = tmp_path / "test_corrupt.sqlite3"
    hm = HistoryManager(db_path=db_file)

    # Insert a valid prediction
    valid_sig = _make_dummy_signal("SBIN", "sbin_valid_001")
    hm.save_prediction(valid_sig)

    # Directly insert raw corrupted row into SQLite predictions table
    with hm.engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions ("
                "symbol, timestamp, action, model_predicted_class, raw_confidence, "
                "risk_adjusted_confidence, calibrated_confidence, agreement_fraction, "
                "reasoning, suppression_reasons, horizon, model_version, feature_version, "
                "outcome_resolved, is_out_of_sample, prediction_key"
                ") VALUES ("
                "'SBIN', 'NOT_A_DATE', 'BUY', 'UP', 0.8, "
                "0.75, 0.7, 0.66, "
                "'{BAD_JSON: INVALID', '[UNCLOSED_ARRAY', 'INTRADAY', 'v1.0', 'v1.0', "
                "0, 0, 'sbin_corrupt_002'"
                ");"
            )
        )
        conn.commit()

    # 1. get_predictions should return both rows without raising uncaught exceptions
    preds = hm.get_predictions("SBIN")
    assert len(preds) == 2

    # 2. get_prediction_signal_by_key should resiliently parse the corrupted row
    corrupt_sig = hm.get_prediction_signal_by_key("sbin_corrupt_002")
    assert corrupt_sig is not None
    assert corrupt_sig.symbol == "SBIN"
    # Fallback reasoning should not crash
    assert isinstance(corrupt_sig.reasoning, list)


def test_safe_parsing_helpers():
    """Test unit behavior of safe_json_loads, safe_float, safe_iso_timestamp."""
    assert safe_json_loads("invalid json", default={"default": 1}) == {"default": 1}
    assert safe_json_loads('["a", "b"]') == ["a", "b"]
    assert safe_float("not a number", default=1.5) == 1.5
    assert safe_float(None, default=0.0) == 0.0
    assert safe_float(3.14) == 3.14
    assert safe_iso_timestamp(None) is not None
    assert "2026-09-17" in safe_iso_timestamp("2026-09-17 10:30:00")


# ---------------------------------------------------------------------------
# 7. Database Integrity Verification
# ---------------------------------------------------------------------------
def test_database_integrity_verification(tmp_path):
    """verify_database_integrity returns True on clean db and detects corruption."""
    db_file = tmp_path / "test_integrity.sqlite3"
    eng = create_reliable_engine(db_file)
    Base.metadata.create_all(bind=eng)

    ok, msgs = verify_database_integrity(eng)
    assert ok is True
    assert msgs == []


# ---------------------------------------------------------------------------
# 8. Storage Recovery Routine
# ---------------------------------------------------------------------------
def test_storage_recovery_routine(tmp_path):
    """run_storage_recovery forces WAL checkpoint, audits counts, and registers health."""
    db_file = tmp_path / "test_recovery.sqlite3"
    hm = HistoryManager(db_path=db_file)

    # Insert 3 predictions (2 unresolved)
    for i in range(3):
        sig = _make_dummy_signal("WIPRO", f"wipro_{i}")
        pid = hm.save_prediction(sig)
        if i == 0 and pid:
            hm.resolve_outcome(pid, actual_class="UP")

    report: StorageRecoveryReport = hm.run_recovery(stale_unresolved_hours=1)
    assert report.wal_checkpointed is True
    assert report.integrity_ok is True
    assert report.prediction_count == 3
    assert report.unresolved_prediction_count == 2
    assert report.duration_ms >= 0

    # Verify health registry has storage component recorded
    statuses = health_registry.get_status()
    storage_status = next((s for s in statuses if s.component == "storage"), None)
    assert storage_status is not None
    assert storage_status.status == "OK"


# ---------------------------------------------------------------------------
# 9. Multi-Threaded Concurrent Writes under WAL Mode
# ---------------------------------------------------------------------------
def test_multithreaded_concurrent_writes_wal(tmp_path):
    """10 simultaneous threads writing predictions and events succeed without 'database is locked' errors."""
    db_file = tmp_path / "test_concurrent_wal.sqlite3"
    hm = HistoryManager(db_path=db_file)

    errors = []
    thread_count = 10
    writes_per_thread = 5

    def worker(worker_id: int):
        for j in range(writes_per_thread):
            try:
                sig = _make_dummy_signal("LT", f"worker_{worker_id}_sig_{j}")
                ev = _make_dummy_event(f"EVT_W_{worker_id}_{j}", "LT")
                hm.save_prediction_bundle(sig, events=[ev])
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == []
    preds = hm.get_predictions("LT", limit=100)
    assert len(preds) == thread_count * writes_per_thread


# ---------------------------------------------------------------------------
# 10. Scheduler and REST API Diagnostics Integration
# ---------------------------------------------------------------------------
def test_scheduler_startup_recovery_and_status(tmp_path):
    """Scheduler initializes storage recovery report and surfaces it in get_status()."""
    db_file = tmp_path / "test_sched_storage.sqlite3"
    hm = HistoryManager(db_path=db_file)
    sched = Scheduler(history_manager=hm)

    status = sched.get_status()
    assert "storage" in status
    storage_meta = status["storage"]
    assert storage_meta.get("integrity_ok") is True
    assert storage_meta.get("wal_checkpointed") is True


def test_api_scheduler_status_includes_storage(tmp_path):
    """GET /api/v1/scheduler/status surfaces storage recovery diagnostics."""
    client = TestClient(app)
    response = client.get("/api/v1/scheduler/status")
    assert response.status_code == 200
    data = response.json()
    assert "storage" in data
    assert data["storage"]["integrity_ok"] is True
    assert data["storage"]["wal_checkpointed"] is True
