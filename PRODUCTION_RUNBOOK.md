# Alkame Nifty50 - Production Operational Runbook

This runbook outlines standard operating procedures for configuring, deploying, monitoring, and maintaining the Alkame Nifty50 Quantitative Engine and API in a production cloud environment.

---

## 1. Pre-Deployment Setup & Environment Configuration

1. **Environment Variables Template**:
   Copy `.env.example` to `.env` in the root directory:
   ```bash
   cp .env.example .env
   ```

2. **Configure Security Keys**:
   Open `.env` and set strong, randomly generated secrets:
   ```env
   ALKAME_API_AUTH_ENABLED=true
   ALKAME_ADMIN_KEY=prod-admin-<secure-random-token>
   ALKAME_READONLY_KEY=prod-readonly-<secure-random-token>
   LOG_FORMAT=json
   LOG_LEVEL=INFO
   ```

3. **Database URL Setup**:
   - For SQLite (Default): `DATABASE_URL=sqlite:///db/predictor.sqlite3`
   - For PostgreSQL: `DATABASE_URL=postgresql://<user>:<password>@<db-host>:5432/<dbname>`

---

## 2. Database Migrations & Maintenance

1. **Run Database Migrations**:
   Apply all Alembic migration scripts to upgrade schema to latest:
   ```bash
   alembic upgrade head
   ```

2. **Create Database Backup**:
   Run the automated backup script prior to any deployment or schema migration:
   ```bash
   python scripts/backup_db.py
   ```
   Backups are saved to `db/backups/predictor_backup_YYYYMMDD_HHMMSS.sqlite3` with an automatic retention policy (keeps 10 most recent backups).

---

## 3. Container Deployment (Docker Compose)

1. **Build and Start Multi-Container Stack**:
   Launch the `app` (FastAPI), `nginx` (Reverse Proxy), and `prometheus` (Metrics Server):
   ```bash
   docker compose up --build -d
   ```

2. **Verify Container Health**:
   Check that all 3 services are running cleanly:
   ```bash
   docker compose ps
   ```

3. **Inspect Application Logs**:
   ```bash
   docker compose logs -f app
   ```

---

## 4. Operational Monitoring & Health Verification

1. **Health Diagnostic Endpoint**:
   Check standard system component statuses:
   ```bash
   curl -H "X-API-Key: prod-admin-<secure-random-token>" http://localhost/api/v1/health
   ```
   *Expected Response*: `{"overall": "OK", "diagnostics": [...]}`

2. **Prometheus Metrics Endpoint**:
   Inspect quantitative domain metrics:
   ```bash
   curl http://localhost/metrics
   ```
   *Key Metrics Monitored*:
   - `nifty50_predictions_total`: Signal counts generated per symbol and horizon.
   - `nifty50_health_status`: Component status gauge (1.0 = OK, 0.0 = DEGRADED/FAIL).
   - `nifty50_model_inference_seconds`: Inference latency histograms.

3. **Prometheus Web UI**:
   Access Prometheus dashboard at `http://localhost:9090`.

---

## 5. Model Rollback & Emergency Procedures

1. **Rollback Model Version**:
   If a newly deployed model exhibits anomaly or degradation, rollback to a prior immutable model run ID using `EnsembleManager`:
   ```python
   from ensemble_manager import EnsembleManager
   em = EnsembleManager()
   em.rollback_ensemble(symbol="RELIANCE", horizon="INTRADAY", target_run_id="<target_run_id>")
   ```

2. **Global Risk Override**:
   To manually activate the global safety toggle via API:
   ```bash
   curl -X POST "http://localhost/api/v1/risk/toggle?enabled=true" \
     -H "X-API-Key: prod-admin-<secure-random-token>"
   ```

3. **Graceful Stack Shutdown**:
   ```bash
   docker compose down
   ```
