# Storage Architecture Evaluation: SQLite Concurrency Boundaries & PostgreSQL Migration Strategy

**Issue Reference:** `SCHED-003` (P1) — Phase 17 Storage Reliability  
**Project:** Alkame Nifty 50 Educational  
**Date:** September 2026  
**Status:** Architecture Design & Evaluation  

---

## 1. Executive Summary

Alkame Nifty 50 Educational currently utilizes **SQLite** via SQLAlchemy as its unified relational persistence engine (`nifty50_history.db`). SQLite provides an exceptional, zero-configuration local storage engine for research, local educational experimentation, and single-instance server processes.

Under **Phase 17 (`SCHED-003`)**, SQLite reliability has been hardened through:
- **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode=WAL;` separating readers from writers.
- **Concurrency PRAGMAs**: `PRAGMA synchronous=NORMAL;`, `PRAGMA busy_timeout=30000;`, and `PRAGMA foreign_keys=ON;`.
- **Deterministic Exponential Backoff Retry**: `with_db_retry` decorator mitigating transient write contention.
- **Atomic Transaction Boundaries**: `atomic_transaction` context manager and `save_prediction_bundle` eliminating partial writes.
- **Startup Storage Recovery**: `run_storage_recovery` forcing WAL checkpoints (`TRUNCATE`) and integrity audits.

However, as production requirements evolve to multi-worker API gateways (e.g. Gunicorn/Uvicorn with 4–8 worker processes), distributed microservices, or asynchronous Celery/RQ task queues, SQLite's single-writer architecture presents fundamental scalability and operational limits.

This document evaluates the operational boundaries of SQLite in financial forecasting pipelines and details the architectural migration plan to **PostgreSQL**.

---

## 2. SQLite Concurrency & Reliability Profile

### 2.1 Concurrency Model: WAL vs. Rollback Journal
In default SQLite `DELETE` or `ROLLBACK` journal mode, any write transaction acquires an exclusive lock on the entire database file, immediately blocking all other readers and writers.

In **Write-Ahead Logging (WAL)** mode (implemented in Phase 17):
- Changes are appended to an auxiliary `-wal` file rather than written directly to the database file.
- Readers read unmodified pages from the database file and modified pages from the WAL file using a shared memory index (`-shm`).
- **Advantage**: Readers do not block writers, and writers do not block readers.
- **Limitation**: **There can still only be one writer at a time.** SQLite strictly serializes all write transactions using a process/thread lock.

### 2.2 Concurrency Failure Modes Under Heavy Multi-Worker Loads
1. **Busy Timeout Exhaustion**:
   When multiple concurrent workers attempt simultaneous writes (e.g., 5 workers saving predictions + 2 background fetchers saving corporate filings + 1 admin toggling risk state), write requests queue up. If cumulative write duration exceeds `busy_timeout` (30 seconds), SQLite throws `OperationalError: database is locked`.
2. **Network Filesystem Corruption Risk**:
   SQLite WAL mode depends on POSIX shared memory primitives (`mmap`, `shm_open`) and fine-grained byte-range file locking. Running SQLite over network file systems (NFS, AWS EFS, SMB/CIFS, Azure Files) risks severe data corruption and random `database disk image is malformed` errors due to buggy remote lock semantics.
3. **Write Starvation & WAL Growth**:
   If reader processes maintain continuous, overlapping read transactions, SQLite cannot checkpoint the WAL file into the primary database file. The WAL file can grow unbounded (megabytes to gigabytes), degrading read performance and exhausting storage.
4. **Lack of Row-Level Locking**:
   Unlike client-server RDBMS engines, SQLite cannot lock individual rows (`SELECT ... FOR UPDATE`). Resolving race conditions requires application-level coordination locks or coarse database-wide lock waiting.

---

## 3. Workload Characterization in Alkame Nifty 50

| Workload Component | Concurrency Type | Frequency / Volume | SQLite WAL Suitability | PostgreSQL Suitability |
|---|---|---|---|---|
| **Scheduler Cycle** | Heavy Batch Write | Every 5 min (50 symbols) | High (Single Process) | Excellent |
| **Micro Scalping** | High-Frequency Write | Every 1–2 min bars | Medium (Lock contention risk) | Excellent |
| **Event Ingestion** | Asynchronous Write | 10–50 filings/hour | Medium (Transient locks) | Excellent |
| **Backtest / Calibration** | Bulk Read & Write | On-demand (100K+ rows) | Causes WAL growth stalls | High (Separate connections) |
| **REST API Serving** | High Concurrent Reads | Continuous (100+ req/sec) | Excellent (WAL readers) | Excellent |
| **Admin Risk Toggle / Audit** | Critical Point Write | Spontaneous | High with retry | Atomic across workers |
| **Multi-Worker API** | Multiple Process Writes | Parallel (4–16 workers) | **Poor (Lock queue saturation)** | **Native** |

---

## 4. Target Production Architecture: PostgreSQL Migration

To scale Alkame Nifty 50 for multi-worker containerized deployments (Kubernetes, AWS ECS, Docker Compose), the persistence layer should transition to PostgreSQL.

### 4.1 Architecture Diagram

```
                 +-----------------------------------+
                 |     Load Balancer (e.g. Nginx)    |
                 +-----------------+-----------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
+--------v-------+                                  +--------v-------+
|  API Worker 1  |                                  |  API Worker 2  |
|  (FastAPI)     |                                  |  (FastAPI)     |
+--------+-------+                                  +--------+-------+
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                     +-------------v-------------+
                     |    Scheduler / Background |
                     |    Worker (Ingestion)     |
                     +-------------+-------------+
                                   |
                     +-------------v-------------+
                     |  PgBouncer Connection     |
                     |  Pooler (Port 6432)       |
                     +-------------+-------------+
                                   |
                     +-------------v-------------+
                     |     PostgreSQL 16 Engine  |
                     |   (Primary / Read Replica)|
                     +---------------------------+
```

### 4.2 Key Advantages of PostgreSQL for Alkame Nifty 50
1. **Multi-Version Concurrency Control (MVCC) & Row-Level Locking**:
   Workers can update different predictions, events, and audit logs concurrently without blocking each other.
2. **Distributed Worker Safety**:
   Multiple physical containers/nodes can safely connect over TCP/IP without local filesystem locking constraints.
3. **Advisory Locks & Distributed Idempotency**:
   PostgreSQL native advisory locks (`pg_try_advisory_lock`) provide distributed lock coordination matching our `PredictionConcurrencyCoordinator` across independent servers.
4. **Native JSONB Queries**:
   Features and reasoning columns (`reasoning`, `suppression_reasons`, `dca_ladder`) can be stored as binary JSON (`JSONB`), enabling indexed queries inside JSON attributes (e.g. searching for predictions suppressed by specific volatility gates).
5. **Zero-Downtime Schema Migrations**:
   Using Alembic, migrations support transactional DDL with minimal table locking.

---

## 5. Dual-Mode Deployment & Configuration Plan

The system retains **100% backward compatibility** with SQLite for local development, tests, and educational usage, while enabling PostgreSQL via environment variables:

```python
# database.py
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

if DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        DATABASE_URL,
        pool_size=int(os.getenv("DB_POOL_SIZE", 20)),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", 10)),
        pool_pre_ping=True,
        pool_recycle=1800,
    )
else:
    # SQLite reliable engine (Phase 17)
    engine = create_reliable_engine(DATABASE_URL)
```

### 5.1 Deployment Matrix
- **Local Research / Self-Test / CI**: SQLite (WAL mode, file-based, zero infrastructure dependencies).
- **Staging / Production (Docker / K8s)**: PostgreSQL with PgBouncer connection pooling.

---

## 6. Conclusion & Recommendation

The Phase 17 storage reliability hardening provides complete data integrity, lock resilience, and crash safety for all single-node and multi-threaded SQLite operations.

For production multi-worker environments, the recommended path is setting `DATABASE_URL=postgresql+psycopg2://...`, which leverages our existing SQLAlchemy declarative models without any code refactoring in business logic.
