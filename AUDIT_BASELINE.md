# AUDIT_BASELINE.md — Phase 0 Baseline and Repository Inventory

**Generated:** 2026-09-16  
**Repository:** `https://github.com/Alkameinc/alkame-nifty-50-educational`  
**Purpose:** Official Phase 0 baseline snapshot prior to remediation execution.

---

## 1. System & Environment

- **commit:** `daf3dc5` (`daf3dc5 fix: resolve metadata load error in ensemble_manager and suppress pandas FutureWarnings during training`)
- **branch:** `main` (clean working tree aside from newly generated tracker artifacts)
- **python:** `3.12.10` (`win32`)
- **os:** Windows 11 (build 10.0.26100)
- **test runner:** `pytest-7.4.4` with plugins `anyio-4.14.2`, `base-url-2.1.0`, `cov-7.1.0`, `flask-1.3.0`, `playwright-0.4.0`, `qt-4.5.0`

---

## 2. Dependencies & Core Packages

| Package | Detected Version | Purpose |
|---|---|---|
| `yfinance` | 0.2.x | Market data provider |
| `pandas` | 2.2.x | Data analysis & manipulation |
| `numpy` | 2.x | Numerical computing |
| `scikit-learn` | 1.4.x | ML models (RandomForest, GradientBoosting) |
| `fastapi` | 0.115.x | REST API server |
| `starlette` | 0.41.x | ASGI framework |
| `uvicorn` | 0.34.x | ASGI web server |
| `sqlalchemy` | 2.0.x | Database ORM |
| `alembic` | 1.14.x | Schema migrations |
| `prometheus-client` | 0.21.x | Telemetry metrics |
| `joblib` | 1.4.x | Model artifact serialization |
| `streamlit` | 1.42.x | Dashboard UI |
| `vaderSentiment` | 3.3.x | News sentiment analyzer |
| `feedparser` | 6.0.x | RSS feed ingestion |

---

## 3. Test Suite & Coverage Baseline

- **tests collected:** 90 items across 25 modules
- **tests passed:** 90
- **tests failed:** 0
- **execution time:** 164.56s (0:02:44)
- **code coverage:** 57.3% total (3,715 statements covered out of 6,481 statements; 2,766 missed)
- **known failures:** 0 test failures; coverage fails current `fail_under=60` threshold in `pyproject.toml`
- **known warnings (63,431 total):**
  1. `PytestCollectionWarning`: `TestFixtureMarketDataProvider` in `market_data_provider.py:347` cannot be collected as a test class because it has an `__init__` constructor.
  2. `StarletteDeprecationWarning`: `fastapi.testclient.TestClient` deprecation notice regarding `httpx`.
  3. `DeprecationWarning: Bitwise inversion '~' on bool`: Deprecated in Python 3.16 (~41,000 occurrences in `test_p0_regressions.py`, `test_quant_correctness.py`, `test_scheduler.py` via pandas indexing).
  4. `DeprecationWarning: Setting the shape on a NumPy array`: Deprecated in NumPy 2.5 (~6,480 occurrences via `joblib.numpy_pickle`).
  5. `DeprecationWarning: The 'generic' unit for NumPy timedelta`: Deprecated in NumPy 2.x (~15,960 occurrences via `_build_synthetic_ohlcv` in test suites).

---

## 4. Static Analysis Baseline (`ruff check .`)

- **Total errors found:** 19 errors (17 auto-fixable)
- **Categories:**
  - `UP007` / `UP035`: Legacy `typing.Union` and `typing.Sequence` annotations in `alembic/versions/*.py`
  - `I001`: Unsorted import blocks in `api.py`, `alembic/versions/*.py`, `test_p0_regressions.py`
  - `UP017`: `datetime.now(timezone.utc)` vs `datetime.now(UTC)` in `api.py:116`
  - `E741`: Ambiguous variable name `l` in `api.py:570`
  - `B904`: Missing `raise ... from e` in `api.py:574`
  - `F401` / `UP015`: Unused import and superfluous mode in `patch_hashlib.py`

---

## 5. Production Module Import Inventory

All 31 production modules verified importable (`python -c "import <mod>"`):

- `config`: OK
- `data_fetcher`: OK
- `market_data_provider`: OK
- `market_calendar`: OK
- `universe_provider`: OK
- `sector_provider`: OK
- `macro_calendar`: OK
- `corporate_events_fetcher`: OK
- `news_sentiment_fetcher`: OK
- `event_classifier`: OK
- `feature_engineer`: OK
- `reference_level_engine`: OK
- `runtime_validator`: OK
- `model_trainer`: OK
- `ensemble_manager`: OK
- `predictor`: OK
- `narrative_builder`: OK
- `position_planner`: OK
- `global_risk_monitor`: OK
- `health_monitor`: OK
- `history_manager`: OK
- `human_insight_manager`: OK
- `execution_simulator`: OK
- `backtester`: OK
- `scanner`: OK
- `scalping`: OK
- `scheduler`: OK
- `api`: OK
- `app`: OK
- `database`: OK
- `models`: OK

---

## 6. Architecture & Entrypoint Catalog

### API routes (`api.py`):
- `GET /openapi.json`
- `GET /docs`
- `GET /docs/oauth2-redirect`
- `GET /redoc`
- `GET /metrics`
- `GET /api/v1/health`
- `GET /healthz`
- `GET /api/v1/symbols`
- `GET /api/v1/signal/{symbol}`
- `GET /api/v1/signal/stream/{symbol}`
- `POST /api/v1/signal/{symbol}/refresh`
- `GET /api/v1/scalping`
- `GET /api/v1/chart/{symbol}`
- `GET /api/v1/risk/toggle`
- `POST /api/v1/risk/toggle`
- `GET /api/v1/admin/audit-logs`
- `GET /`

### Training entrypoints:
- `model_trainer.py`: `ModelTrainer.train_for_symbol()`, `ModelTrainer.train_model()`
- `ensemble_manager.py`: `EnsembleManager.train_ensemble_for_symbol()`
- `train_all.py`: `train_all()`
- `backtester.py`: `Backtester.run_backtest_for_symbol()`, `Backtester.run_full_backtest()`

### Prediction entrypoints:
- `predictor.py`: `Predictor.predict_for_symbol()`, `Predictor.predict_multi_horizon()`
- `scheduler.py`: `Scheduler.run_one_cycle_for_symbol()`, `Scheduler.run_cycle()`
- `api.py`: `/api/v1/signal/{symbol}`, `/api/v1/signal/stream/{symbol}`, `/api/v1/signal/{symbol}/refresh`
- `scanner.py`: `OpportunityScanner.scan_universe()`
- `scalping.py`: `ScalpingEngine.find_opportunities()`

### Scheduler:
- `scheduler.py`: `Scheduler.start_loop()`, `Scheduler.run_cycle()`, `Scheduler.run_one_cycle_for_symbol()`

### Data providers:
- `market_data_provider.py`: `MarketDataProvider`, `YFinanceMarketDataProvider`, `TestFixtureMarketDataProvider`
- `data_fetcher.py`: `DataFetcher`
- `market_calendar.py`: `MarketCalendar`
- `universe_provider.py`: `UniverseProvider`
- `sector_provider.py`: `SectorMapProvider`

### Event providers:
- `macro_calendar.py`: `MacroCalendar`
- `corporate_events_fetcher.py`: `CorporateEventsFetcher`
- `news_sentiment_fetcher.py`: `NewsSentimentFetcher`
- `event_classifier.py`: `EventClassifier`

### Persistence & storage entrypoints:
- `database.py`: `engine`, `SessionLocal`, `get_db()`
- `models.py`: `PredictionRecord`, `AuditLog`, `Base`
- `history_manager.py`: `HistoryManager`
- `global_risk_monitor.py`: `data/global_risk_toggle_state.json`
- `ensemble_manager.py`: `models/{symbol}/{horizon}/{run_id}/` (`ensemble.joblib`, `metadata.json`, `schema.json`, `metrics.json`), `current.json`
- `data_fetcher.py`: `data/cache/*`

### Streamlit & UI entrypoints:
- `app.py`: Streamlit main dashboard
- `frontend/`: React / Vite application (`frontend/src/App.jsx`, `frontend/src/main.jsx`)

### Environment variables:
- `MARKETAUX_API_KEY`: API token for MarketAux financial news API
- `IMD_API_KEY`: API token for IMD monsoon weather API (optional)
- `ALKAME_API_AUTH_ENABLED`: Flag enabling/disabling API key authentication (default `"true"`)
- `ALKAME_ADMIN_KEY`: Primary administrative API key
- `ALKAME_READONLY_KEY`: Primary read-only API key
- `ALKAME_ADMIN_KEYS`: Comma-separated list of additional admin keys
- `ALKAME_READONLY_KEYS`: Comma-separated list of additional read-only keys
- `ALKAME_CORS_ORIGINS`: Comma-separated list of allowed CORS origins
- `LOG_FORMAT`: Logging format (`"text"` or `"json"`)
- `ENVIRONMENT` / `ALKAME_ENV`: Deployment environment (`"development"` or `"production"`)
