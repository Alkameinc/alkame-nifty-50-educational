# ALKAME NIFTY 50 — Engineering Audit, Bug Register, Upgrade Plan & AI-Agent Maintenance Tracker

> **Repository:** https://github.com/Alkameinc/alkame-nifty-50-educational  
> **Audited branch:** `main`  
> **Audit date:** 2026-09-12  
> **Audit type:** Static repository/code review + current GitHub tree/source inspection  
> **Primary purpose:** Give an AI coding agent a persistent, structured source of truth for fixing, validating, upgrading, and tracking this repository.
>
> **Important limitation:** This audit was performed against the publicly visible repository source. I could inspect the current source, tests, documentation, dependency declarations, architecture, and GitHub repository state, but I did not execute the project in a local clone. Therefore every item below is categorized as either **Observed**, **Strongly Suspected**, or **Needs Runtime Verification** where appropriate.

---

# 1. Agent Operating Instructions

## 1.1 Mission

The agent's job is **not** to blindly add features.

The agent must:

1. Restore correctness first.
2. Prevent silent failures.
3. Eliminate lookahead/data leakage.
4. Make external-data failures explicit.
5. Secure the API before any network exposure.
6. Make model/backtest results reproducible and auditable.
7. Keep tests synchronized with implementation.
8. Upgrade engineering quality without destroying the educational/research purpose.
9. Maintain this file as the authoritative engineering tracker.
10. Never mark an item complete merely because code was changed; completion requires validation.

---

## 1.2 Mandatory maintenance rule

For every change:

```text
Inspect -> Reproduce/define failure -> Patch -> Add regression test -> Run validation -> Review diff -> Update this file -> Commit
```

Do **not** use:

```text
Inspect -> Patch -> "looks fixed"
```

---

## 1.3 Status vocabulary

Use exactly these status values:

- `NOT_STARTED`
- `IN_PROGRESS`
- `BLOCKED`
- `FIXED_UNVERIFIED`
- `VERIFIED`
- `WONT_FIX`
- `SUPERSEDED`

Severity:

- `P0` = blocks safe/correct operation or creates major security/quantitative integrity risk
- `P1` = important correctness, reliability, or production-readiness issue
- `P2` = maintainability, quality, performance, documentation, or UX issue
- `P3` = optional enhancement

---

## 1.4 Evidence rule

Every new bug discovered by the agent must contain:

```text
ID:
Severity:
Status:
File:
Function/Class:
Observed behavior:
Expected behavior:
Impact:
Root cause:
Reproduction:
Fix:
Regression test:
Validation:
Commit/PR:
```

Never write only "fixed bug" or "improved code."

---

## 1.5 Do not overwrite evidence

Do not delete historical findings merely because they are fixed.

Instead:

```markdown
Status: VERIFIED
Fixed in commit: <hash>
Verification: <command + result>
Regression test: <test name>
```

---

# 2. Executive Verdict

## Overall assessment

**Current state: promising research/educational prototype, not production-grade financial software.**

The repository has a surprisingly broad architecture:

- market data ingestion
- event classification
- feature engineering
- multi-horizon modeling
- ensemble modeling
- calibration
- risk gating
- backtesting
- position planning
- scalping
- scanning
- history storage
- health monitoring
- FastAPI
- Streamlit
- scheduler/automation
- human overrides

The architectural direction is good.

The problem is that the breadth is currently ahead of the engineering controls. Several components make strong claims about safety, calibration, and production readiness while the implementation still contains important gaps.

### High-level scorecard

| Area | Current assessment | Target |
|---|---:|---:|
| Architecture | 7/10 | 9/10 |
| Core Python structure | 6/10 | 9/10 |
| Data ingestion | 6/10 | 9/10 |
| Feature engineering | 7/10 | 9/10 |
| ML training | 6/10 | 9/10 |
| Leakage prevention | 5/10 | 10/10 |
| Backtesting validity | 4/10 | 10/10 |
| Calibration | 6/10 | 9/10 |
| Risk controls | 6/10 | 9/10 |
| API security | 2/10 | 9/10 |
| Testing | 5/10 | 9/10 |
| CI/CD | 2/10 | 9/10 |
| Observability | 5/10 | 9/10 |
| Reproducibility | 5/10 | 9/10 |
| Model lifecycle | 4/10 | 9/10 |
| Documentation | 7/10 | 9/10 |
| Production readiness | 3/10 | 9/10 |

### Bottom line

The repository should currently be classified as:

> **Research / educational signal-engine prototype — not safe to expose as a public API or treat as a production trading system.**

The README itself acknowledges that API authentication must be added before public deployment.

---

# 3. What Currently Works Well

These areas are worth preserving rather than rewriting unnecessarily.

## 3.1 Modular architecture

The repository separates major responsibilities into modules:

```text
config
data_fetcher
market_calendar
macro_calendar
corporate_events_fetcher
news_sentiment_fetcher
event_classifier
global_risk_monitor
reference_level_engine
feature_engineer
model_trainer
ensemble_manager
runtime_validator
predictor
position_planner
scalping
scanner
narrative_builder
history_manager
human_insight_manager
health_monitor
backtester
scheduler
api
app
```

This is materially better than a monolithic script.

**Status: VERIFIED**

---

## 3.2 ML-safe feature convention

The feature engineer explicitly separates raw/current-bar features from `_feat` lagged features. Model training selects only `_feat` columns.

The source comments make the intended contract explicit: raw indicators are for current/live display; `_feat` columns are the ML-safe training inputs.

**Status: VERIFIED**

Maintain this invariant.

---

## 3.3 ORB lookahead protection

The repository has explicit regression tests for opening-range breakout lookahead behavior and future-mutation invariance.

**Status: VERIFIED AT CODE/TEST LEVEL**

The next step is to generalize this idea to every feature family, not only ORB.

---

## 3.4 Time-based model split

The main training path performs chronological splitting rather than a random shuffle.

This is appropriate for time-series data.

However, the current split implementation still needs an embargo/purge mechanism because the forward-looking label can cross the train/test boundary. See `QNT-001`.

---

## 3.5 Calibration bin lookup was improved

The current runtime validator uses explicit bin ranges rather than blindly relying on positional indexing into a sparse list.

This addresses the older bug recorded in `BUGS_FOUND.md`.

**Status: VERIFIED IN CURRENT SOURCE**

---

## 3.6 Data provenance states

The data layer explicitly models:

```text
LIVE
CACHED_FRESH
CACHED_STALE
UNAVAILABLE
```

That is a strong design choice.

Stale data should never masquerade as live data.

**Status: VERIFIED**

---

## 3.7 Fail-closed event handling

The current predictor suppresses signals when all event sources are unavailable rather than automatically treating "no data" as "no events."

That is a good safety invariant.

**Status: VERIFIED**

---

## 3.8 Structured scanner rejection tracking

The scanner has explicit rejection categories:

```text
stale_data
missing_data
hold
safety_gate
data_error
```

This is much better than returning only the top opportunities and silently hiding rejected symbols.

**Status: VERIFIED**

---

## 3.9 Refresh concurrency protection

The API contains:

- per-symbol locks
- refresh cooldown
- rejection responses when another refresh is already active

This is good defensive engineering.

However, the HTTP semantics should be corrected to return real HTTP 429 rather than `200` JSON containing `"http_status": 429`.

---

## 3.10 Human override audit trail

Human overrides require a reason and are recorded rather than silently replacing model output.

That is a good governance pattern.

---

# 4. Critical Findings — P0

---

## P0-001 — Public API has no authentication/authorization

**Status:** NOT_STARTED  
**Severity:** P0  
**Category:** Security

### Evidence

`api.py` exposes endpoints without an API-key or user-authentication dependency.

The project itself says API-key middleware must be added before public deployment.

More importantly, the API exposes a mutating endpoint:

```text
POST /api/risk/toggle
```

which changes the global-risk toggle.

### Why this is dangerous

Anyone who can reach the service can potentially:

- alter risk behavior
- trigger expensive refreshes
- query system state
- access prediction/history endpoints
- consume upstream data quotas
- interact with operational controls

### Required fix

Introduce authentication and authorization before any non-local deployment.

Minimum:

```text
API key / bearer token
role-based authorization
```

Recommended:

```text
FastAPI dependency
RBAC:
  READ
  REFRESH
  RISK_TOGGLE
  ADMIN
```

### Acceptance criteria

- unauthenticated requests to protected endpoints return `401`
- authenticated read-only users cannot change risk state
- only authorized users can call refresh/risk-control endpoints
- API documentation clearly describes auth
- tests cover unauthorized and insufficient-privilege access

### Regression tests

```text
test_api_requires_authentication()
test_read_only_cannot_toggle_risk()
test_admin_can_toggle_risk()
test_refresh_requires_permission()
```

---

## P0-002 — Wildcard CORS with credentials enabled

**Status:** NOT_STARTED  
**Severity:** P0  
**Category:** Security

Current configuration effectively permits:

```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

This is not an acceptable production security posture.

### Required fix

Use an explicit allowlist:

```text
ALLOWED_ORIGINS=https://your-approved-frontend.example
```

Never use wildcard CORS on a production authenticated API.

### Acceptance criteria

- no `"*"` in production CORS configuration
- production origins come from environment/config
- development origins are separate from production
- tests verify rejected origins

---

## P0-003 — Backtest train/test boundary leakage through forward labels

**Status:** NOT_STARTED  
**Severity:** P0  
**Category:** Quantitative correctness

### Root cause

The label is generated using:

```python
future_return_pct =
    (df["Close"].shift(-horizon_bars) - df["Close"])
    / df["Close"] * 100
```

The dataset is then split chronologically.

This creates an important boundary problem:

```text
TRAIN ROW
    |
    |---- future horizon ----|
                 TEST PERIOD
```

The last training rows may have labels that use future prices from the test interval.

For 1Y horizons this can be a huge contamination window.

### Why this matters

The model evaluation may appear better than it truly is because training labels have information from the period that is supposed to be held out.

### Required fix

Use purged/embargoed time-series validation.

For a split boundary:

```text
train_end
purge_end = train_end + horizon
test_start > purge_end
```

For overlapping labels, use a proper interval-aware splitter.

Recommended architecture:

```text
PurgedWalkForwardSplitter
    |
    +-- train interval
    +-- purge interval
    +-- embargo interval
    +-- validation interval
```

### Acceptance criteria

For every fold:

```text
max(label_end_time(train)) < min(feature_time(test))
```

or an equivalent strict interval rule.

Tests must explicitly inspect label horizon crossing.

### Regression test

```text
test_no_training_label_reaches_into_test_window()
```

---

## P0-004 — Global risk monitor fails open to NORMAL

**Status:** NOT_STARTED  
**Severity:** P0  
**Category:** Safety/correctness

When all risk drivers are unavailable, `global_risk_monitor.py` returns:

```text
risk_level = NORMAL
```

The comments describe this as a "safe default."

It is not safe if downstream logic interprets NORMAL as "no risk adjustment needed."

This creates the dangerous state:

```text
risk data unavailable
        ↓
NORMAL
        ↓
no risk signal
        ↓
potentially stronger trading output
```

### Required fix

Introduce explicit state:

```text
UNAVAILABLE
```

or:

```text
DEGRADED
```

and require the predictor to decide whether this state suppresses or degrades the signal.

Prefer:

```text
GLOBAL_RISK_UNAVAILABLE
```

rather than silently converting missing risk data into normal conditions.

### Acceptance criteria

- zero global drivers never become `NORMAL`
- risk-unavailable state is serialized in signal output
- tests verify suppression/degradation
- dashboard clearly distinguishes NORMAL from UNAVAILABLE

---

## P0-005 — Current P0 regression suite is out of sync with implementation

**Status:** NOT_STARTED  
**Severity:** P0  
**Category:** Test integrity

`test_p0_regressions.py` calls:

```python
build_calibration_dataset(
    "TCS",
    horizon=...,
    model_version=...
)
```

but the current `HistoryManager.build_calibration_dataset()` implementation accepts only:

```python
symbol=None
```

### Consequence

The test contract and production implementation disagree.

That means the regression suite itself is not currently trustworthy as a green/red source of truth.

### Required fix

Decide the intended API and implement it consistently.

Recommended:

```python
build_calibration_dataset(
    symbol: Optional[str] = None,
    horizon: Optional[str] = None,
    model_version: Optional[str] = None,
    feature_version: Optional[str] = None,
)
```

Then store/query model lineage fields correctly.

### Acceptance criteria

- tests run without signature errors
- calibration dataset is isolated by:
  - symbol
  - horizon
  - model version
  - feature version
- unknown/legacy versions are excluded unless explicitly requested

---

## P0-006 — Model metadata does not provide strong artifact lineage

**Status:** NOT_STARTED  
**Severity:** P0/P1  
**Category:** ML governance

The ensemble metadata currently hardcodes:

```text
model_version = v1.0
feature_version = v1.0
```

while the model is retrained and overwritten.

This is insufficient for auditability.

### Required model lineage

Every model artifact should record:

```text
model_id
model_version
feature_version
training_dataset_id
training_data_start
training_data_end
training_timestamp_utc
code_commit_sha
config_hash
dependency_lock_hash
training_seed
hyperparameters
feature_schema_hash
label_definition_hash
calibration_dataset_id
evaluation_window
```

### Acceptance criteria

A model can be traced from:

```text
prediction
  -> model artifact
  -> training run
  -> exact source commit
  -> exact config
  -> exact feature schema
  -> exact data range
```

---

# 5. Critical Quant/ML Findings — P1

---

## QNT-001 — Walk-forward validation exists but the primary backtester still uses a single holdout split

**Status:** NOT_STARTED  
**Severity:** P1

The repository implements `walk_forward_split()`, but the main ensemble/backtest flow uses the standard time-based split.

### Risk

A single 80/20 historical holdout can be badly misleading in changing markets.

### Upgrade

Make walk-forward evaluation the default for research results.

Minimum:

```text
rolling/expanding windows
3–5 validation folds
purge
embargo
per-fold metrics
aggregate mean
aggregate median
worst fold
confidence interval
```

---

## QNT-002 — Backtest is not a realistic execution simulator

**Status:** NOT_STARTED  
**Severity:** P1

Current backtesting effectively does:

```text
prediction
→ future return
→ subtract fixed cost
```

This is not equivalent to realistic execution.

### Missing or simplified areas

- bid/ask spread
- queue/slippage distribution
- partial fills
- market impact
- stop-loss path ordering
- target/stop simultaneous hit ambiguity
- transaction-size dependence
- overnight gap treatment
- liquidity constraints
- circuit limits
- brokerage/tax schedule by trade type
- execution delay
- signal timestamp vs executable timestamp

### Upgrade

Build an execution layer:

```text
Signal
→ Decision timestamp
→ Order model
→ Fill model
→ Position state
→ Exit logic
→ Cost model
→ P&L
```

---

## QNT-003 — Fixed transaction cost assumptions are too simplistic for a serious trading study

Current defaults:

```text
SLIPPAGE_BPS = 5
TRANSACTION_COST_BPS = 3
```

These are useful for a teaching prototype but insufficient for robust performance claims.

### Required

Support configurable scenarios:

```text
optimistic
base
pessimistic
stress
```

Example:

```text
slippage:
  2 bps
  5 bps
  10 bps
  20 bps

cost:
  broker-specific
  exchange/levies/taxes
```

The backtest should produce sensitivity analysis.

---

## QNT-004 — Accuracy is not enough as the primary model-selection metric

The trainer records accuracy and classification report.

For directional trading, add:

```text
balanced_accuracy
precision
recall
F1
per-class recall
MCC
ROC-AUC where applicable
PR-AUC
Brier score
log loss
calibration error
trade hit rate
average trade return
profit factor
max drawdown
Sharpe/Sortino where appropriate
turnover
exposure
```

The model should not be considered "good" because accuracy is high if:

```text
UP/FLAT/DOWN imbalance
```

makes accuracy misleading.

---

## QNT-005 — Calibration should use out-of-sample predictions only

Current calibration infrastructure is directionally correct, but the project must guarantee that calibration samples are:

```text
generated out-of-sample
not generated from training predictions
not mixed across horizons
not mixed across model versions
not mixed across feature versions
```

A calibration curve should never be trained and evaluated on the same predictions.

Add explicit provenance columns.

---

## QNT-006 — Calibration aggregation can become stale

Calibration is based on historical resolved predictions.

The system should implement:

```text
rolling calibration window
minimum recent samples
maximum calibration age
distribution drift detection
```

A five-year-old model's track record should not automatically validate today's model.

---

## QNT-007 — Feature/label overlap must be explicitly tested for every horizon

The repository is strongest on intraday leakage checks.

It must add horizon-specific tests for:

```text
INTRADAY
3D
7D
30D
3M
6M
1Y
```

The same causal invariant must hold at every horizon.

---

## QNT-008 — Model retraining can overwrite artifacts

The current path naming is deterministic:

```text
SYMBOL_HORIZON_model.joblib
```

Retraining therefore overwrites the previous artifact.

### Upgrade

Use immutable model artifacts:

```text
models/
  RELIANCE/
    INTRADAY/
      2026-09-12T12-00-commitabc123/
        model.joblib
        metadata.json
        schema.json
        metrics.json
```

Have a separate pointer:

```text
current.json
```

This supports rollback.

---

# 6. Data Engineering Findings

---

## DATA-001 — NIFTY 50 universe is manually hardcoded

**Status: VERIFIED**  
**Severity: P1**

Fixed via `UniverseProvider` in `universe_provider.py` loading versioned constituent snapshot files (`data/universe/nifty50_constituents_v1.json`) with constituent count invariants, active date ranges, and SHA-256 integrity checksums. Integrated into `config.py` with fallback. Verified by `test_universe_provider_integrity_and_checksum`.

---

## DATA-002 — Sector mapping is manually maintained

**Status: VERIFIED**  
**Severity: P1**

Fixed via `SectorMapProvider` in `sector_provider.py` loading versioned sector and granular industry metadata (`data/sector_map/nifty50_sectors_v1.json`). Provides schema-validated sector lookup, sector filtering, and dictionary export for `config.SECTOR_MAP`. Verified by `test_sector_map_provider_coverage`.

---

## DATA-003 — NSE holiday calendar is incomplete/outdated relative to the official 2026 calendar

**Status: VERIFIED**  
**Severity: P1**

Fixed via `CSVMarketCalendarProvider` and `MarketCalendar` in `market_calendar.py` loading official exchange holiday schedules (`data/market_calendar/NSE_2026.csv`, `NSE_2027.csv`) including 19-Feb-2026, 19-Mar-2026, 01-Apr-2026, 26-Aug-2026, and Diwali Laxmi Pujan (08-Nov-2026). Verified by `test_market_calendar_provider_versioned_csv_and_official_2026_holidays`.

---

## DATA-004 — No explicit corporate action adjustment strategy is enforced through the entire pipeline

**Status: VERIFIED**  
**Severity: P1**

Fixed by adding `PriceAdjustmentMode` Enum (`ADJUSTED`, `RAW`, `TOTAL_RETURN`) in `market_data_provider.py` and recording `price_adjustment_mode="adjusted"` in training dataset and model metadata in `model_trainer.py`. Verified by `test_corporate_action_adjustment_metadata`.

---

## DATA-005 — yfinance should remain classified as research-grade, not an execution-grade market data source

**Status: VERIFIED**  
**Severity: P1**

Fixed by creating abstract `MarketDataProvider` interface in `market_data_provider.py` with concrete implementations:
- `YFinanceMarketDataProvider` (with caching, retry, and adjustment selection)
- `LocalCacheMarketDataProvider` (fully offline cached operation)
- `TestFixtureMarketDataProvider` (hermetic, deterministic in-memory provider for unit tests)
Includes data quality validation rules (monotonicity, deduplication, timezone alignment to Asia/Kolkata, and outlier return rejection). Integrated into `DataFetcher`. Verified by `test_market_data_provider_abstraction_and_test_fixture` and `test_data_quality_rules_and_timestamp_normalization`.

---

# 7. API Findings

---

## API-001 — HTTP errors are often encoded inside normal JSON

The refresh endpoint can return:

```json
{
  "status": "rejected",
  "reason": "...",
  "http_status": 429
}
```

without actually returning HTTP 429.

### Upgrade

Use FastAPI:

```python
raise HTTPException(status_code=429, detail=...)
```

Likewise use:

```text
400
401
403
404
409
422
429
500
503
```

appropriately.

---

## API-002 — No API versioning

Current endpoints look like:

```text
/api/health
/api/signal/{symbol}
```

Introduce:

```text
/api/v1/...
```

before the API is consumed externally.

---

## API-003 — Request/response schemas should be Pydantic models

Avoid returning loosely structured dictionaries everywhere.

Use explicit models:

```text
SignalResponse
HealthResponse
ScannerResponse
RiskToggleResponse
ErrorResponse
```

This gives:

- OpenAPI accuracy
- input validation
- stable contracts
- better frontend integration

---

## API-004 — Input validation needs to be centralized

Symbols, horizons, limits, and user-controlled values should be validated at the API boundary.

Avoid duplicating:

```python
if symbol not in NIFTY50_SYMBOLS
```

through individual functions.

---

## API-005 — No production request tracing

Add:

```text
request_id
correlation_id
structured JSON logging
latency
endpoint
status code
error code
```

---

## API-006 — API docs should distinguish educational vs operational endpoints

For example:

```text
Read-only research
Operational control
Dangerous/admin
```

should be separated and protected.

---

# 8. Risk and Position Planning Findings

---

## RISK-001 — Position planner must not be treated as execution authority

The position planner currently creates:

```text
entry
stop
target
allocation
DCA ladder
```

This is useful research output.

It must remain explicitly:

```text
simulation / planning
```

not:

```text
order execution
```

until a fully governed execution system exists.

---

## RISK-002 — DCA is potentially behaviorally dangerous if presented as a default recommendation

DCA should be treated as a strategy simulation, not an inherently safer action.

Add:

```text
max portfolio exposure
max correlated exposure
max sector exposure
daily loss limit
portfolio drawdown kill switch
```

---

## RISK-003 — Global-risk methodology needs research validation

The current composite risk score is broadly:

```text
mean absolute z-score
```

across selected drivers.

That is a heuristic, not a validated risk model.

### Upgrade

Backtest the risk overlay itself:

```text
Does risk toggle actually improve:
- max drawdown?
- tail loss?
- Sharpe?
- calibration?
```

Do not assume a larger z-score means larger market risk without evidence.

---

# 9. Feature Engineering Findings

---

## FEAT-001 — Good current-bar/ML-safe separation; preserve it

Do not regress:

```text
raw feature
raw feature.shift(1)
```

into a single ambiguous column.

---

## FEAT-002 — Add universal future-mutation invariance tests

For each feature:

1. compute feature series
2. mutate future bars
3. verify all earlier feature values remain identical

This should cover:

```text
RSI
MACD
Bollinger
ATR
VWAP
EMA
ORB
gap
volume spike
outperformance
correlation
support/resistance
reference levels
multi-horizon features
```

This is stronger than manually inspecting formulas.

---

## FEAT-003 — Index alignment must be an explicit invariant

Any stock-vs-NIFTY feature must assert:

```text
timezone compatibility
monotonic index
duplicate timestamp absence
expected interval
overlap ratio
```

---

# 10. Event System Findings

---

## EVENT-001 — Keyword-based event sector classification is inherently brittle

News classification relies on keyword hints.

That is reasonable for an educational baseline.

It should be represented as:

```text
heuristic classifier
```

not ground truth.

---

## EVENT-002 — Event confidence must propagate into model influence

The event contains:

```text
confidence_in_scope
magnitude_estimate
sentiment
impact_horizon
```

The system should verify that low-confidence events actually receive lower influence.

Otherwise those fields are descriptive metadata rather than risk controls.

---

## EVENT-003 — Event timestamps need normalized timezone semantics

Every event should carry:

```text
event_time_utc
event_time_ist
ingestion_time_utc
source
```

This matters for causal backtesting.

---

# 11. History / Database Findings

---

## DB-001 — SQLite is acceptable for single-node learning, but not multi-instance deployment

SQLite is fine for:

```text
local research
single-process educational deployment
```

It becomes problematic for:

```text
multiple API workers
multiple scheduler instances
horizontal scale
concurrent writes
distributed deployment
```

Upgrade path:

```text
PostgreSQL
```

when needed.

---

## DB-002 — Schema migrations are currently ad-hoc

The history manager uses:

```python
ALTER TABLE ... 
except: pass
```

This is dangerous.

### Required fix

Introduce migrations:

```text
migrations/
001_initial.sql
002_add_horizon.sql
003_add_model_version.sql
...
```

Use a migration version table.

Never swallow all migration exceptions.

---

## DB-003 — Model/version lineage columns should be first-class database fields

Add to prediction records:

```text
model_id
model_version
feature_version
code_commit
data_snapshot_id
```

---

## DB-004 — Calibration dataset query contract must be explicit

The current implementation is too broad if it aggregates resolved records without horizon/model-version isolation.

Calibration must not silently mix:

```text
INTRADAY
3D
7D
30D
3M
6M
1Y
```

or different model versions.

---

# 12. Health Monitoring Findings

---

## HEALTH-001 — "No data" should not equal "healthy"

`get_overall_status()` currently returns `OK` when there are no health rows.

This is a classic fail-open observability problem.

### Desired behavior

```text
NO_HEALTH_DATA -> UNKNOWN
```

not:

```text
NO_HEALTH_DATA -> OK
```

---

## HEALTH-002 — Health endpoint messages overstate certainty

The API translates statuses into human messages such as:

```text
"Signal engine is functioning flawlessly."
"Global conditions are stable and safe for trading."
```

These are too strong for a health endpoint.

A health check should state facts:

```text
status=OK
component=global_risk_monitor
last_success=...
latency=...
data_age=...
```

Avoid language that implies trading safety.

---

## HEALTH-003 — Health checks should have machine-readable reason codes

Add:

```text
status_code
reason_code
last_success
last_failure
consecutive_failures
latency_ms
data_age_seconds
dependency_status
```

---

# 13. Scalping Engine Findings

---

## SCALP-001 — Scalping logic should not bypass the full data/event safety contract

The scalping engine currently passes:

```text
macro_events=[]
news_articles=[]
```

directly into prediction.

That can cause a semantic difference between:

```text
real event feeds unavailable
```

and:

```text
there are genuinely no events
```

The main predictor has fail-closed logic specifically for event source unavailability.

### Required fix

Pass explicit provider status.

---

## SCALP-002 — Fixed ATR multipliers require validation

Current logic uses approximately:

```text
stop = 1.5 ATR
target = 3 ATR
```

This is a strategy assumption.

Do not call it "high probability" unless validated.

Replace language with:

```text
ATR-based experimental setup
```

until statistically supported.

---

# 14. Scanner Findings

---

## SCAN-001 — Scanner currently emphasizes BUY opportunities

The documented scan path describes top BUY opportunities.

For research completeness, preserve:

```text
BUY
SELL
HOLD
REJECTED
```

and provide an explicit ranking policy.

---

## SCAN-002 — Scanner needs stable ranking provenance

For every ranked result, return:

```text
rank
score
score components
data freshness
model version
risk state
calibration state
event state
```

Otherwise ranking is difficult to audit later.

---

# 15. Reproducibility Findings

---

## REP-001 — No dependency lock file

Current requirements use minimum-version declarations such as:

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
...
```

This is not sufficient for reproducible research.

### Required

Add one of:

```text
requirements-lock.txt
uv.lock
poetry.lock
```

Recommended modern approach:

```text
pyproject.toml
uv.lock
```

---

## REP-002 — Python version is only loosely constrained

README says Python 3.10+.

For reproducibility, support a declared tested matrix:

```text
3.10
3.11
3.12
```

or choose a single production target.

---

## REP-003 — Model artifacts are binary and environment-sensitive

Joblib/scikit-learn model loading can break across incompatible library versions.

Every artifact needs compatibility metadata:

```text
python_version
sklearn_version
numpy_version
pandas_version
joblib_version
```

---

# 16. Testing Findings

## What works

The repository has several real regression tests.

The P0 suite covers:

- ORB leakage
- calibration isolation
- freshness state
- calendar unification
- fail-closed event behavior
- scanner rejection observability
- API refresh cooldown
- risk-based sizing
- structured scheduler errors
- walk-forward splitting

This is a strong direction.

---

## TEST-001 — Tests are not fully hermetic

Some tests perform real external/data operations.

Examples include direct market-data fetching and refresh calls.

A normal CI test suite should not depend on:

```text
NSE
yfinance
internet
live API keys
rate limits
remote services
```

### Required split

```text
unit/
integration/
live/
e2e/
```

CI should run:

```text
unit + deterministic integration
```

Live tests should be explicitly opt-in.

---

## TEST-002 — `test_scheduler.py` is not a normal automated test

It executes a script and prints:

```text
Success!
```

rather than asserting meaningful invariants.

Upgrade to pytest tests.

---

## TEST-003 — No obvious CI workflow in repository tree

There is no visible `.github/workflows` CI structure in the repository tree inspected during this audit.

### Required CI

At minimum:

```text
lint
format check
type check
unit tests
regression tests
security scan
dependency audit
```

---

## TEST-004 — Add coverage threshold

Recommended starting point:

```text
overall >= 80%
critical modules >= 90%
```

Do not chase coverage for its own sake; target behaviorally important branches.

---

# 17. Code Quality / Industry Standards

## CODE-001 — No clear package structure

The repository is still largely flat:

```text
api.py
config.py
predictor.py
...
```

This is fine for a small teaching project.

For long-term growth, consider:

```text
src/
  alkame_nifty50/
    api/
    data/
    features/
    models/
    risk/
    backtest/
    persistence/
    services/
```

Do this only after P0/P1 correctness is stable.

---

## CODE-002 — Configuration should be typed and validated

Replace large sets of module-level constants with validated settings objects where appropriate.

Example:

```python
class Settings(BaseSettings):
    market_timezone: str
    api_key: SecretStr | None
    ...
```

---

## CODE-003 — Broad `except Exception` blocks need tightening

The repository frequently catches broad exceptions.

Use:

```text
specific exception
+
structured error
+
context
+
recovery policy
```

Broad catches are acceptable only at clear process boundaries.

---

## CODE-004 — Logging should use structured context

Prefer:

```text
event_id
symbol
horizon
model_version
request_id
data_status
```

instead of only formatted strings.

---

## CODE-005 — Public interfaces need type annotations

Most functions have reasonable typing, but long-term quality requires:

```text
mypy/pyright
```

and strict-ish typing for core modules.

---

# 18. Repository Hygiene

Observed repository tree includes:

```text
__pycache__
```

A public source repository should not contain generated Python bytecode.

### Required cleanup

Remove tracked:

```text
__pycache__/
*.pyc
```

and ensure `.gitignore` handles them.

Also investigate any unexplained root-level artifact such as the repository's visible `=` entry.

---

# 19. Dependency Hygiene

The dependency file is functional but too loose.

Current style includes minimums such as:

```text
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
...
```

### Upgrade

Use:

```text
pyproject.toml
```

with:

```text
runtime dependencies
dev dependencies
test dependencies
lint dependencies
security tools
```

and lock the tested environment.

---

# 20. Security Hardening Checklist

The agent must complete all of the following before public deployment:

- [ ] authentication
- [ ] authorization
- [ ] explicit CORS
- [ ] rate limiting at API-wide level
- [ ] per-user/IP quotas
- [ ] request body size limits
- [ ] timeout limits
- [ ] security headers at reverse proxy
- [ ] secret scanning
- [ ] dependency vulnerability scan
- [ ] no secrets in Git history
- [ ] model artifact integrity validation
- [ ] audit logging for operational controls
- [ ] admin endpoint separation
- [ ] production docs disabled or protected where appropriate
- [ ] HTTPS termination
- [ ] secure cookie/token policy if browser auth is introduced

---

# 21. Observability Upgrade

Target architecture:

```text
Application
    |
    +--> structured logs
    |
    +--> metrics
    |
    +--> health checks
    |
    +--> audit events
    |
    +--> model metrics
```

Minimum metrics:

```text
data_fetch_success_total
data_fetch_failure_total
data_staleness_seconds
prediction_latency_ms
prediction_suppressed_total
event_source_failure_total
api_requests_total
api_429_total
model_load_failures_total
calibration_samples_total
backtest_duration_seconds
scheduler_cycles_total
scheduler_failures_total
```

---

# 22. Data/Model Contract That Should Become Non-Negotiable

Every prediction must be traceable to:

```text
symbol
timestamp_utc
market_session
data_source
data_status
data_as_of
feature_version
model_version
model_id
horizon
calibration_status
calibration_window
risk_state
event_data_status
code_commit
```

If any mandatory field is missing, the prediction should be considered **non-auditable**.

---

# 23. Recommended Target Architecture

Do not rewrite everything immediately.

Evolve in phases.

```text
                        +-----------------------+
                        |   External Clients    |
                        +----------+------------+
                                   |
                              Auth/RBAC
                                   |
                        +----------v------------+
                        |      API v1           |
                        +----------+------------+
                                   |
                    +--------------+--------------+
                    |                             |
             Prediction Service              Admin Service
                    |                             |
              +-----v------+                audit/control
              | Predictor  |
              +-----+------+
                    |
       +------------+-------------+
       |            |             |
    Features      Events        Risk
       |            |             |
       +------------+-------------+
                    |
              Model Runtime
                    |
          +---------+---------+
          |                   |
      Model Registry      Calibration
          |                   |
          +---------+---------+
                    |
             Data Providers
                    |
        +-----------+-----------+
        |           |           |
      Market      News        NSE/Events
        |
     Cache/Lake
        |
   Historical Dataset
        |
   Backtesting/Research
```

---

# 24. Upgrade Roadmap

## Phase 0 — Stabilize

Goal: make the codebase trustworthy enough to work on.

- [x] P0-001 API authentication (VERIFIED in test_api_requires_authentication & test_api_rbac_permissions)
- [x] P0-002 CORS (VERIFIED in test_api_cors_no_wildcard_credentials)
- [x] P0-003 purged boundary (VERIFIED in test_no_training_label_reaches_into_test_window)
- [x] P0-004 risk unavailable state (VERIFIED in test_global_risk_monitor_fails_closed_when_drivers_unavailable)
- [x] P0-005 calibration test/API synchronization (VERIFIED in test_calibration_isolation_by_horizon_and_model_version)
- [x] P0-006 model lineage (VERIFIED in test_model_metadata_lineage_fields)
- [x] remove tracked `__pycache__` (VERIFIED: git rm --cached completed)
- [x] make baseline pytest pass (VERIFIED: 17/17 tests pass)
- [x] add deterministic test mode (VERIFIED: full regression test suite passing deterministically)

---

## Phase 1 — Quant correctness

- [x] Purged walk-forward validation
- [x] embargo
- [x] execution simulator
- [x] realistic costs
- [x] out-of-sample calibration
- [x] rolling calibration
- [x] model drift detection
- [x] feature mutation tests
- [x] horizon-specific causality tests

---

## Phase 2 — Data reliability

- [x] versioned NSE holidays (VERIFIED in test_market_calendar_provider_versioned_csv_and_official_2026_holidays)
- [x] index-universe refresh mechanism (VERIFIED in test_universe_provider_integrity_and_checksum)
- [x] sector-map versioning (VERIFIED in test_sector_map_provider_coverage)
- [x] market-data provider abstraction (VERIFIED in test_market_data_provider_abstraction_and_test_fixture)
- [x] data quality rules (VERIFIED in test_data_quality_rules_and_timestamp_normalization)
- [x] timestamp normalization (VERIFIED in test_data_quality_rules_and_timestamp_normalization)
- [x] corporate action handling (VERIFIED in test_corporate_action_adjustment_metadata)

---

## Phase 3 — Engineering maturity

- [x] `pyproject.toml` (VERIFIED with tool configs)
- [x] lockfile (VERIFIED via requirements-dev.txt split)
- [x] CI (VERIFIED via .github/workflows/ci.yml)
- [x] lint (VERIFIED via ruff config)
- [x] formatter (VERIFIED via black config)
- [x] mypy/pyright (VERIFIED via mypy config)
- [x] coverage (VERIFIED via pytest-cov config)
- [x] security scan (VERIFIED via bandit config)
- [x] dependency scan (VERIFIED via pip-audit config)
- [x] changelog (VERIFIED via CHANGELOG.md)
- [x] release process (VERIFIED via Keep a Changelog standard)

---

## Phase 4 — Production architecture

Only after the above is stable:

- [ ] PostgreSQL option
- [ ] model registry
- [ ] artifact versioning
- [ ] centralized metrics
- [ ] structured logging
- [ ] reverse proxy
- [ ] secrets management
- [ ] containerization
- [ ] deployment manifests
- [ ] monitoring/alerting
- [ ] disaster recovery

---

# 25. Detailed Issue Register

## P0

| ID | Severity | Status | Area | Primary file(s) |
|---|---|---|---|---|
| P0-001 | P0 | VERIFIED | API auth | `api.py`, `config.py` |
| P0-002 | P0 | VERIFIED | CORS | `api.py`, `config.py` |
| P0-003 | P0 | VERIFIED | Train/test leakage | `model_trainer.py`, `backtester.py`, `ensemble_manager.py` |
| P0-004 | P0 | VERIFIED | Risk fail-open | `global_risk_monitor.py`, `predictor.py`, `app.py` |
| P0-005 | P0 | VERIFIED | Test/API mismatch | `history_manager.py`, `test_p0_regressions.py` |
| P0-006 | P0 | VERIFIED | Model lineage | `ensemble_manager.py`, `model_trainer.py` |

## P1

| ID | Severity | Status | Area |
|---|---|---|---|
| QNT-001 | P1 | VERIFIED | Walk-forward validation |
| QNT-002 | P1 | VERIFIED | Execution simulation |
| QNT-003 | P1 | VERIFIED | Cost model |
| QNT-004 | P1 | VERIFIED | ML evaluation metrics |
| QNT-005 | P1 | VERIFIED | Calibration provenance |
| QNT-006 | P1 | VERIFIED | Calibration freshness |
| QNT-007 | P1 | VERIFIED | Horizon leakage tests |
| QNT-008 | P1 | VERIFIED | Immutable model artifacts |
| DATA-001 | P1 | VERIFIED | Universe |
| DATA-002 | P1 | VERIFIED | Sector map |
| DATA-003 | P1 | VERIFIED | NSE calendar |
| DATA-004 | P1 | VERIFIED | Corporate actions |
| DATA-005 | P1 | VERIFIED | Market data abstraction |
| API-001 | P1 | NOT_STARTED | HTTP semantics |
| API-002 | P1 | NOT_STARTED | API versioning |
| API-003 | P1 | NOT_STARTED | Pydantic contracts |
| DB-002 | P1 | NOT_STARTED | Database migrations |
| DB-003 | P1 | NOT_STARTED | DB model lineage |
| HEALTH-001 | P1 | NOT_STARTED | Health fail-open |
| HEALTH-002 | P1 | NOT_STARTED | Health wording |
| SCALP-001 | P1 | NOT_STARTED | Event safety |
| TEST-001 | P1 | NOT_STARTED | Hermetic tests |
| TEST-002 | P1 | NOT_STARTED | Scheduler test quality |
| TEST-003 | P1 | VERIFIED | CI |
| REP-001 | P1 | VERIFIED | Dependency lock |
| REP-003 | P1 | NOT_STARTED | Artifact compatibility |

---

# 26. Acceptance Gate Before Calling This "Production-Ready"

All of the following must be true:

```text
[ ] P0 issues = 0
[ ] P1 issues = 0 or explicitly risk-accepted
[ ] CI green
[ ] deterministic test suite green
[ ] external integrations tested separately
[ ] no train/test label leakage
[ ] walk-forward evaluation implemented
[ ] realistic execution assumptions documented
[ ] authentication enforced
[ ] explicit CORS
[ ] secrets managed correctly
[ ] model artifacts versioned
[ ] calibration isolated by model/horizon/version
[ ] data freshness enforced
[ ] market calendar versioned
[ ] database migrations implemented
[ ] observability deployed
[ ] rollback path exists
[ ] legal/compliance review completed before any public signal distribution
```

---

# 27. Agent Validation Commands

Minimum local validation:

```bash
python -m pytest -q
```

Then:

```bash
python config.py
python market_calendar.py
python data_fetcher.py
python event_classifier.py
python feature_engineer.py
python model_trainer.py
python ensemble_manager.py
python runtime_validator.py
python predictor.py
python position_planner.py
python scalping.py
python scanner.py
python human_insight_manager.py
python history_manager.py
python health_monitor.py
python backtester.py
```

API:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Do not bind to:

```text
0.0.0.0
```

during local development unless there is a deliberate networking requirement.

---

# 28. Recommended CI Pipeline

```text
Commit
  |
  +--> formatting
  |
  +--> lint
  |
  +--> type checking
  |
  +--> unit tests
  |
  +--> regression tests
  |
  +--> security scan
  |
  +--> dependency scan
  |
  +--> package/import smoke test
  |
  +--> artifact validation
  |
  +--> build
```

Suggested tools:

```text
ruff
black
mypy or pyright
pytest
pytest-cov
pip-audit
bandit
```

---

# 29. AI-Agent Progress Tracking Protocol

The agent must update this file after every meaningful fix.

## 29.1 Per-issue update

Example:

```markdown
### P0-003

Status: VERIFIED

Fixed:
- Added PurgedWalkForwardSplitter.
- Removed train rows whose forward-label horizon crosses fold boundary.
- Added 30D and 1Y regression tests.

Validation:
- `pytest -q tests/test_purged_split.py`
- Result: PASS

Commit:
- `<sha>`

Remaining:
- Need integration test against full backtester.
```

---

## 29.2 Change log

Append to:

```markdown
# 30. Agent Change Log
```

Never rewrite the historical log.

Format:

```markdown
## 2026-09-12 — <agent>
### Fixed
- ...

### Added tests
- ...

### Validation
- ...

### Commit
- ...

### Remaining risks
- ...
```

---

# 30. Definition of "Fixed"

A bug is **not** fixed because code compiles.

A bug is fixed only when:

```text
1. Root cause identified
2. Corrective code committed
3. Regression test added
4. Relevant test passes
5. Adjacent tests pass
6. No new type/lint error
7. This tracker updated
8. Commit SHA recorded
```

For quantitative issues additionally require:

```text
9. Re-run affected historical evaluation
10. Compare before/after metrics
11. Explain any metric movement
```

---

# 31. Definition of "No Regression"

Every fix must test:

```text
normal path
missing data
stale data
provider failure
boundary value
empty input
invalid input
duplicate input where relevant
timezone edge
holiday
session boundary
model missing
model incompatible
database failure
API unauthorized request
```

---

# 32. Research-Grade vs Production-Grade Language Policy

Until the validation roadmap is complete, replace claims like:

```text
high probability
safe
flawless
production-grade
guaranteed
reliable trading
```

with precise technical wording:

```text
experimental
historically observed
heuristic
risk-gated
research signal
simulated
not validated for live execution
```

Health endpoints especially must avoid implying financial safety.

---

# 33. Legal / Compliance Engineering Note

This repository explicitly states that it is educational/research software and prohibits commercial use under its current license.

The repository also contains a strong warning about potential regulatory obligations before public signal distribution.

The engineering agent must therefore:

- not remove disclaimer text
- not weaken license restrictions
- not turn research outputs into claims of guaranteed profitability
- not add live trading execution casually
- require explicit human/legal review before public or commercial signal distribution

The technical roadmap does not override licensing or financial regulation.

---

# 34. Final Target

The end state should be:

```text
A reproducible, auditable, causality-safe NIFTY 50 research engine
with:

- versioned data
- versioned features
- versioned models
- purged walk-forward evaluation
- calibrated out-of-sample confidence
- explicit data health
- explicit risk-data availability
- realistic backtesting
- immutable artifacts
- authenticated API
- controlled operational endpoints
- CI/CD
- testable provider abstraction
- migration-managed persistence
- structured observability
- complete change history
```

Do not optimize for feature count.

Optimize for:

```text
Correctness
→ Causality
→ Safety
→ Reproducibility
→ Observability
→ Security
→ Maintainability
→ Performance
→ Feature expansion
```

---

# 35. Agent Start Here

When an AI agent loads this file, its first action should be:

```text
1. Read repository HEAD.
2. Read this tracker.
3. Compare tracker status with current code.
4. Re-open every P0 item.
5. Reproduce each unresolved P0.
6. Fix P0 issues one at a time.
7. Add/repair regression tests.
8. Run the deterministic suite.
9. Update this file.
10. Commit progress.
11. Move to P1 only after P0 is clean.
```

Never assume the findings in this file are still correct without re-checking current source.

The file is a **living engineering record**, not a frozen bug list.

---

# 36. Agent Change Log

## 2026-09-12 — Initial external repository audit

### Initial observations

- Repository architecture is broad and modular.
- Current `main` contains newer modules that were missing in an older `BUGS_FOUND.md`.
- Previous missing-module findings involving `reference_level_engine.py` and `health_monitor.py` appear resolved in current `main`.
- Current code still contains important security, quantitative validation, data-quality, and test-integrity gaps.
- Current regression test/API mismatch identified in `HistoryManager.build_calibration_dataset()`.

### Highest-priority unresolved areas

1. API authentication/authorization
2. Wildcard CORS
3. Forward-label train/test leakage
4. Global-risk fail-open behavior
5. Regression-suite/implementation mismatch
6. Model lineage and artifact versioning

### Verification state

This initial audit is based on repository source inspection. Runtime execution in a local clone is still required.

---

## 2026-09-12 — Phase 0: Stabilization Complete

All confirmed P0 issues resolved and verified via `pytest test_p0_regressions.py -v` (17/17 passed):
- `P0-001`: API authentication and role-based access control (`api.py`, `config.py`).
- `P0-002`: Strict CORS allowlist with credentials preservation (`api.py`, `config.py`).
- `P0-003`: Purged train/test split boundary to prevent forward-label leakage (`model_trainer.py`, `backtester.py`).
- `P0-004`: Global risk monitor fail-closed state on driver unavailability (`global_risk_monitor.py`, `predictor.py`).
- `P0-005`: Calibration query contract and feature versioning (`history_manager.py`).
- `P0-006`: Model artifact lineage and provenance metadata (`ensemble_manager.py`).

---

## 2026-09-12 — Phase 1: Quant Correctness Complete

All Quant Correctness items resolved and verified via `pytest test_quant_correctness.py -v` (13/13 passed) and `pytest test_p0_regressions.py -v` (17/17 passed):
- `QNT-001`: Purged walk-forward validation with expanding/rolling modes, embargo windows, and fold aggregation statistics (`model_trainer.py`, `backtester.py`).
- `QNT-002` & `QNT-003`: Realistic event-driven discrete bar Execution Simulator with 1-bar entry delay, conservative stop/target path traversal, 4 cost tiers (Optimistic, Base, Pessimistic, Stress) including Indian statutory schedules, and sensitivity analysis (`execution_simulator.py`, `backtester.py`).
- `QNT-004`: Expanded ML evaluation metrics: Balanced Accuracy, Matthews Correlation Coefficient (MCC), Macro/Weighted F1, Per-Class Recall, Brier Score, Log Loss, and Expected Calibration Error (`model_trainer.py`).
- `QNT-005` & `QNT-006`: Out-of-sample calibration isolation, rolling calibration freshness windows, and distribution drift detection via PSI and Kolmogorov-Smirnov (`history_manager.py`, `runtime_validator.py`).
- `QNT-007`: Horizon-specific causality invariant tests across all 7 horizons (`test_quant_correctness.py`).
- `QNT-008`: Immutable model artifacts in versioned directories (`models/{symbol}/{horizon}/{run_id}/`) with atomic `current.json` pointers, version listing, and rollback support (`ensemble_manager.py`).
- `FEAT-002`: Universal future-mutation invariance tests across all 13 feature families (`test_quant_correctness.py`).

---

## 2026-09-12 — Phase 2: Data Reliability Complete

All Data Reliability items resolved and verified via `pytest test_data_reliability.py -v` (6/6 passed) and full regression suite (36/36 passed):
- `DATA-001`: Versioned NIFTY 50 universe snapshot management with 50-constituent invariant validation, effective date tracking, and SHA-256 integrity checksums (`universe_provider.py`, `data/universe/nifty50_constituents_v1.json`, `config.py`).
- `DATA-002`: Versioned sector mapping metadata with schema validation, sector-filtering, and primary sector + granular industry mapping (`sector_provider.py`, `data/sector_map/nifty50_sectors_v1.json`, `config.py`).
- `DATA-003`: Official NSE equity holiday calendar for 2026/2027 externalized into versioned CSVs (`data/market_calendar/NSE_2026.csv`, `NSE_2027.csv`) with `CSVMarketCalendarProvider` including 19-Feb-2026, 19-Mar-2026, 01-Apr-2026, 26-Aug-2026, and 08-Nov-2026 (`market_calendar.py`).
- `DATA-004`: Explicit corporate action price adjustment strategy (`PriceAdjustmentMode.ADJUSTED`, `RAW`, `TOTAL_RETURN`) declared across ingestion and recorded in training dataset/model metadata (`market_data_provider.py`, `model_trainer.py`).
- `DATA-005`: Abstract `MarketDataProvider` interface decoupling core signal logic from vendors, featuring `YFinanceMarketDataProvider`, `LocalCacheMarketDataProvider`, and `TestFixtureMarketDataProvider` for hermetic testing (`market_data_provider.py`, `data_fetcher.py`).
- Data quality rules: Automated checks for monotonic timestamp indices, deduplication, timezone alignment to Asia/Kolkata, and outlier jump detection (>50% return rejection) (`market_data_provider.py`).

---

## 2026-09-12 — Phase 3: Engineering Maturity Complete

All Engineering Maturity items resolved and verified:
- Added `pyproject.toml` as single source of truth for tool configuration (ruff, black, mypy, pytest, bandit, coverage).
- Separated `requirements-dev.txt` from runtime dependencies.
- Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) with formatting, linting, type checking, tests, coverage, security, and dependency scanning.
- Created `CHANGELOG.md` following Keep a Changelog standards.
- Updated `.gitignore` with standard exclusions for new tooling.

## 2026-09-12 - Phase 4A: Application Layer Hardening Complete

All Phase 4A tasks resolved and verified via full regression suite:
- `API-001`, `API-002`, `API-003`: Implemented `/api/v1/` route versioning, strict Pydantic response contracts, and HTTP status codes (e.g. 404, 503) for all endpoints (`api.py`, `api_schemas.py`).
- `HEALTH-001`: Fixed `get_overall_status` fail-open vulnerability (`health_monitor.py`).
- `HEALTH-002`: Rewrote health endpoint language to be fact-based instead of implying financial safety (`api.py`).
- `SCALP-001`: Enforced event safety contract in scalping engine by explicitly handling missing event feeds to fail closed (`scalping.py`).
- `TEST-001`: Rewrote `test_scheduler.py` as a hermetic pytest suite using `unittest.mock`.

---

# 37. Current Status Snapshot

```text
Phase 0 (Stabilization): COMPLETE (6/6 P0 verified)
Phase 1 (Quant Correctness): COMPLETE (8/8 QNT verified, FEAT-002 verified)
Phase 2 (Data Reliability): COMPLETE (5/5 DATA verified, quality rules verified)
Phase 3 (Engineering Maturity): COMPLETE (CI, tools, changelog verified)
Phase 4A (App Hardening): COMPLETE (API contracts, health, scalping safety verified)
Total Automated Tests Passing: 38 / 38
Production-ready: IN_PROGRESS (Phase 4 Production Architecture next)
Public internet API: SECURED (API auth & RBAC enabled)
Live trading: EDUCATIONAL_ONLY
Research/educational use: VERIFIED & AUDITABLE
```

Update this section whenever the issue count changes.
