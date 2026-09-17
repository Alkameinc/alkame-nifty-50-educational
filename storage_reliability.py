"""
storage_reliability.py
Storage Reliability Layer for Alkame Nifty 50 Educational (SCHED-003, P1)

Provides:
1. SQLite WAL mode & PRAGMA configuration (synchronous=NORMAL, busy_timeout=30s, foreign_keys=ON).
2. Reliable engine factory (`create_reliable_engine`).
3. Deterministic retry with exponential backoff & jitter for SQLite lock contention (`with_db_retry`).
4. Clean atomic transaction context manager (`atomic_transaction`).
5. Database integrity verification (`verify_database_integrity`).
6. Storage recovery routine (`run_storage_recovery`) executing WAL checkpointing, integrity validation,
   and table auditing on startup.
7. Corrupted record resilient deserializers (`safe_json_loads`, `safe_float`, `safe_iso_timestamp`).
"""

import functools
import json
import logging
import os
import random
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, TypeVar

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from config import DB_PATH

logger = logging.getLogger(__name__)

T = TypeVar("T")


def configure_sqlite_engine(engine: Engine) -> None:
    """Configures SQLite engine with WAL mode, busy timeout, and integrity pragmas.
    
    For SQLite connections:
    - PRAGMA journal_mode=WAL; (enables concurrent readers and writers)
    - PRAGMA synchronous=NORMAL; (reduces fsync bottlenecks while safe in WAL mode)
    - PRAGMA busy_timeout=30000; (30-second internal wait on lock contention)
    - PRAGMA foreign_keys=ON; (enforce relational integrity)
    """
    if not str(engine.url).startswith("sqlite"):
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            try:
                # WAL mode cannot be used with in-memory databases; SQLite handles this gracefully
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA busy_timeout=30000;")
                cursor.execute("PRAGMA foreign_keys=ON;")
            except Exception as e:
                logger.warning(f"Could not apply SQLite PRAGMAs: {e}")
            finally:
                cursor.close()


def create_reliable_engine(
    db_url_or_path: str | Path,
    echo: bool = False,
    pool_pre_ping: bool = True,
    **kwargs: Any,
) -> Engine:
    """Factory creating an SQLAlchemy engine configured for reliability."""
    if isinstance(db_url_or_path, Path) or not (
        str(db_url_or_path).startswith("sqlite:")
        or str(db_url_or_path).startswith("postgresql:")
        or str(db_url_or_path).startswith("postgres:")
    ):
        db_url = f"sqlite:///{db_url_or_path}"
    else:
        db_url = str(db_url_or_path)

    connect_args = kwargs.pop("connect_args", {})
    if db_url.startswith("sqlite"):
        connect_args.setdefault("check_same_thread", False)
        connect_args.setdefault("timeout", 30.0)

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        echo=echo,
        pool_pre_ping=pool_pre_ping,
        **kwargs,
    )
    configure_sqlite_engine(engine)
    return engine


def is_lock_contention_error(exc: BaseException) -> bool:
    """Detects whether an exception is caused by SQLite lock/busy contention."""
    msg = str(exc).lower()
    if isinstance(exc, (sqlite3.OperationalError, OperationalError, DBAPIError)):
        if "database is locked" in msg or "database is busy" in msg or "lock" in msg:
            return True
    if exc.__cause__ and is_lock_contention_error(exc.__cause__):
        return True
    return False


def with_db_retry(
    max_retries: int = 5,
    initial_delay: float = 0.05,
    backoff_factor: float = 2.0,
    max_delay: float = 2.0,
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator providing deterministic exponential backoff retry on SQLite lock contention."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if not is_lock_contention_error(exc) or attempt >= max_retries:
                        raise
                    attempt += 1
                    delay = min(max_delay, initial_delay * (backoff_factor ** (attempt - 1)))
                    if jitter:
                        delay += random.uniform(0.01, 0.05)
                    logger.warning(
                        f"[SCHED-003] SQLite lock contention detected in {func.__name__} "
                        f"(attempt {attempt}/{max_retries}). Retrying in {delay:.3f}s..."
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


@contextmanager
def atomic_transaction(session_factory: Callable[[], Session]) -> Generator[Session, None, None]:
    """Context manager providing strictly atomic transactions.
    
    Commits on successful block completion.
    Automatically rolls back on any exception before re-raising.
    Always closes the session cleanly.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        try:
            session.rollback()
        except Exception as rb_exc:
            logger.error(f"[SCHED-003] Failed during transaction rollback: {rb_exc}")
        raise
    finally:
        session.close()


def verify_database_integrity(engine: Engine) -> Tuple[bool, List[str]]:
    """Runs SQLite quick_check and integrity_check to verify database file health.
    
    Returns:
        (True, []) if database passed integrity checks.
        (False, error_messages) if corruption was detected.
    """
    if not str(engine.url).startswith("sqlite"):
        return True, []

    messages = []
    try:
        with engine.connect() as conn:
            # 1. Quick check (fast structural verification)
            qc_rows = conn.execute(text("PRAGMA quick_check;")).fetchall()
            for row in qc_rows:
                val = str(row[0]).strip()
                if val.lower() != "ok":
                    messages.append(f"quick_check failure: {val}")

            # 2. Integrity check (comprehensive verification if quick_check found anomalies)
            if messages:
                ic_rows = conn.execute(text("PRAGMA integrity_check;")).fetchall()
                for row in ic_rows:
                    val = str(row[0]).strip()
                    if val.lower() != "ok":
                        messages.append(f"integrity_check failure: {val}")

        return (len(messages) == 0, messages)
    except Exception as exc:
        err_msg = f"Integrity check execution error: {exc}"
        logger.error(f"[SCHED-003] {err_msg}")
        return False, [err_msg]


@dataclass
class StorageRecoveryReport:
    timestamp: str
    wal_checkpointed: bool
    checkpoint_log: Optional[str]
    integrity_ok: bool
    integrity_messages: List[str]
    prediction_count: int
    unresolved_prediction_count: int
    stale_unresolved_count: int
    event_count: int
    backtest_metrics_count: int
    duration_ms: float
    error: Optional[str] = None


def run_storage_recovery(
    engine: Engine,
    db_path: Optional[Path] = None,
    stale_unresolved_hours: int = 24,
) -> StorageRecoveryReport:
    """Executes startup storage recovery:
    
    1. Forces WAL checkpoint (TRUNCATE mode to flush and reset WAL journal).
    2. Runs PRAGMA quick_check to verify schema and b-tree integrity.
    3. Audits database counts (predictions, events, backtest metrics).
    4. Detects abandoned/stale unresolved predictions across system restarts.
    5. Reports storage health to health registry.
    """
    start_t = time.perf_counter()
    ts_now = datetime.now(timezone.utc).isoformat()

    wal_ok = False
    checkpoint_log = None
    integrity_ok = True
    integrity_msgs: List[str] = []
    pred_count = 0
    unresolved_count = 0
    stale_unresolved_count = 0
    event_count = 0
    metric_count = 0
    recovery_error = None

    try:
        # 1. Force WAL Checkpoint if SQLite
        if str(engine.url).startswith("sqlite"):
            try:
                with engine.connect() as conn:
                    # TRUNCATE checkpoints all frames and truncates the WAL file to zero length
                    res = conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE);")).fetchall()
                    wal_ok = True
                    checkpoint_log = str(res)
                    logger.info(f"[SCHED-003] Forced WAL checkpoint completed: {res}")
            except Exception as e:
                wal_ok = False
                checkpoint_log = f"WAL checkpoint failed: {e}"
                logger.warning(f"[SCHED-003] WAL checkpoint failed: {e}")
        else:
            wal_ok = True
            checkpoint_log = "Not SQLite; WAL checkpoint skipped."

        # 2. Verify Database Integrity
        integrity_ok, integrity_msgs = verify_database_integrity(engine)
        if not integrity_ok:
            logger.error(f"[SCHED-003] Database integrity verification FAILED: {integrity_msgs}")

        # 3. Audit counts from tables
        try:
            with engine.connect() as conn:
                # Count predictions
                try:
                    p_res = conn.execute(text("SELECT COUNT(*) FROM predictions;")).scalar()
                    pred_count = int(p_res) if p_res is not None else 0
                except Exception:
                    pred_count = 0

                # Count unresolved predictions
                try:
                    unres = conn.execute(
                        text("SELECT COUNT(*) FROM predictions WHERE outcome_resolved = 0 OR outcome_resolved IS NULL;")
                    ).scalar()
                    unresolved_count = int(unres) if unres is not None else 0
                except Exception:
                    unresolved_count = 0

                # Count stale unresolved predictions
                try:
                    cutoff = (datetime.now(timezone.utc) - timedelta(hours=stale_unresolved_hours)).isoformat()
                    stale_res = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM predictions WHERE (outcome_resolved = 0 OR outcome_resolved IS NULL) "
                            "AND timestamp < :cutoff;"
                        ),
                        {"cutoff": cutoff},
                    ).scalar()
                    stale_unresolved_count = int(stale_res) if stale_res is not None else 0
                except Exception:
                    stale_unresolved_count = 0

                # Count events
                try:
                    e_res = conn.execute(text("SELECT COUNT(*) FROM events;")).scalar()
                    event_count = int(e_res) if e_res is not None else 0
                except Exception:
                    event_count = 0

                # Count backtest metrics
                try:
                    m_res = conn.execute(text("SELECT COUNT(*) FROM backtest_metrics;")).scalar()
                    metric_count = int(m_res) if m_res is not None else 0
                except Exception:
                    metric_count = 0
        except Exception as e:
            recovery_error = str(e)
            logger.error(f"[SCHED-003] Table audit failed during storage recovery: {e}")

    except Exception as e:
        recovery_error = str(e)
        integrity_ok = False
        integrity_msgs.append(f"Unexpected recovery error: {e}")
        logger.error(f"[SCHED-003] Storage recovery failed unexpectedly: {e}")

    duration_ms = (time.perf_counter() - start_t) * 1000.0

    report = StorageRecoveryReport(
        timestamp=ts_now,
        wal_checkpointed=wal_ok,
        checkpoint_log=checkpoint_log,
        integrity_ok=integrity_ok,
        integrity_messages=integrity_msgs,
        prediction_count=pred_count,
        unresolved_prediction_count=unresolved_count,
        stale_unresolved_count=stale_unresolved_count,
        event_count=event_count,
        backtest_metrics_count=metric_count,
        duration_ms=round(duration_ms, 2),
        error=recovery_error,
    )

    # 4. Report health
    health_ok = integrity_ok and (recovery_error is None)
    detail_msg = (
        f"Storage online: WAL={wal_ok}, integrity={'PASS' if integrity_ok else 'FAIL'}, "
        f"predictions={pred_count} (unresolved={unresolved_count}, stale={stale_unresolved_count}), "
        f"events={event_count}, recovery_ms={report.duration_ms}"
    )
    try:
        from health_monitor import registry as health_registry

        health_registry.report(
            "storage",
            ok=health_ok,
            detail=detail_msg,
            error=recovery_error or ("; ".join(integrity_msgs) if integrity_msgs else None),
        )
    except Exception as rep_err:
        logger.warning(f"[SCHED-003] Health report failed: {rep_err}")

    return report


# ---------------------------------------------------------------------------
# Corrupted Record Resilient Deserializers
# ---------------------------------------------------------------------------
def safe_json_loads(val: Any, default: Any = None) -> Any:
    """Safely decodes a JSON string without raising uncaught exceptions on corrupted data."""
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    try:
        s_val = str(val).strip()
        if not s_val:
            return default
        return json.loads(s_val)
    except Exception as e:
        logger.warning(f"[SCHED-003] Resilient JSON deserialization failed for payload: {e}")
        return default


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts an input to float, falling back to default if corrupted/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if f == f else default  # handles NaN
    except Exception:
        return default


def safe_iso_timestamp(val: Any, default: Optional[str] = None) -> str:
    """Safely standardizes a timestamp string or object, returning an ISO string."""
    if val is None:
        return default or datetime.now(timezone.utc).isoformat()
    try:
        if isinstance(val, datetime):
            if val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)
            return val.isoformat()
        import pandas as pd
        ts = pd.Timestamp(val)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        return ts.isoformat()
    except Exception:
        return default or str(val)
