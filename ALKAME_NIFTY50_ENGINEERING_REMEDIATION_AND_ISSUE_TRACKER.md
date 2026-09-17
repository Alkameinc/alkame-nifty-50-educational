# Alkame Nifty 50 Educational — Engineering Remediation & Issue Tracking Plan

**Repository:** `https://github.com/Alkameinc/alkame-nifty-50-educational`  
**Purpose:** Master remediation plan for a local AI coding agent.  
**Scope:** Remaining bugs, dead ends, silent failures, pipeline integrity issues, security hardening, quantitative correctness, observability, testing, and deployment safety.

> **IMPORTANT:** This document is an execution plan, not permission to rewrite the architecture indiscriminately. Preserve working behavior unless a change is required by an issue below. Every change must be tested and recorded in the issue tracker.

---

# 1. Agent Operating Rules

The coding agent MUST follow these rules.

## 1.1 Before changing code

1. Read:
   - `README.md`
   - `pyproject.toml`
   - all existing `development*.md`, `BUGS_FOUND.md`, audit documents, and architecture documents
   - relevant source files and tests
2. Inspect the current branch before applying any old recommendation.
3. Do not assume an issue in an older audit is still present.
4. Mark an issue `OBSOLETE` if the current code already fixes it.
5. Never remove a safety gate merely because it makes tests or demos harder.
6. Never weaken validation to make a test pass.
7. Never introduce future-data access into feature engineering.
8. Never silently replace unavailable quantitative values with plausible-looking values.
9. Never commit real credentials, API keys, tokens, or secrets.
10. Do not make broad unrelated refactors while fixing an issue.

## 1.2 Every fix must include

For every issue:

- root-cause explanation
- code change
- regression test
- integration test where applicable
- documentation update if behavior changes
- issue tracker update
- verification command
- final status

## 1.3 Required status values

Use only:

- `OPEN`
- `IN_PROGRESS`
- `BLOCKED`
- `FIXED`
- `VERIFIED`
- `OBSOLETE`
- `WONT_FIX`

Do not mark an issue `VERIFIED` merely because the code compiles.

---

# 2. Severity Definitions

| Severity | Meaning |
|---|---|
| P0 | Security, data-integrity, leakage, production-safety, or research-validity blocker |
| P1 | Major correctness/reliability problem that can materially affect results |
| P2 | Important engineering or maintainability problem |
| P3 | Improvement / technical debt |

---

# 3. Master Issue Tracker

The agent MUST maintain this table as work progresses.

| ID | Area | Issue | Severity | Status | Owner | Test Added | Verified |
|---|---|---|---|---|---|---|---|
| SEC-001 | API Auth | Disabled auth currently maps callers to ADMIN | P0 | VERIFIED | AI | Yes | Yes |
| SEC-002 | Secrets | Default ADMIN/dev credentials exist in source | P0 | VERIFIED | AI | Yes | Yes |
| SEC-003 | API Auth | Audit every endpoint for explicit authorization | P0 | VERIFIED | AI | Yes | Yes |
| SEC-004 | Risk | Risk state is process-local/in-memory | P0 | VERIFIED | AI | Yes | Yes |
| SEC-005 | Multi-worker | Risk state can diverge between workers | P0 | VERIFIED | AI | Yes | Yes |
| SEC-006 | Metrics | `/metrics` exposure must be explicitly protected | P1 | VERIFIED | AI | Yes | Yes |
| SEC-007 | Rate Limit | In-memory rate limits are unsuitable for multi-worker deployment | P1 | VERIFIED | AI | Yes | Yes |
| SEC-008 | CORS | Restrict methods/headers/credential behavior | P1 | VERIFIED | AI | Yes | Yes |
| SEC-009 | Audit | Critical state mutations must not silently succeed when audit fails | P1 | VERIFIED | AI | Yes | Yes |
| SEC-010 | Correlation | Validate/sanitize client correlation IDs | P2 | VERIFIED | AI | Yes | Yes |
| API-001 | Architecture | Refactor duplicated/legacy API initialization and routes | P1 | OPEN | AI | No | No |
| API-002 | Consistency | API prediction path differs from scheduler context | P0 | VERIFIED | AI | Yes | Yes |
| API-003 | Semantics | Separate raw model score from calibrated probability | P1 | OPEN | AI | No | No |
| API-004 | UX/API | Remove overly directive trading language from analytical outputs | P1 | OPEN | AI | No | No |
| DATA-001 | Freshness | Plain DataFrame cache path can hide stale-data state | P0 | VERIFIED | AI | Yes | Yes |
| DATA-002 | Freshness | Cache freshness must understand market sessions | P1 | VERIFIED | AI | Yes | Yes |
| DATA-003 | Alignment | Detect stock/index timestamp alignment loss | P1 | VERIFIED | AI | Yes | Yes |
| DATA-004 | Time | Standardize internal timestamps on timezone-aware UTC | P1 | VERIFIED | AI | Yes | Yes |
| DATA-005 | Events | Invalid event timestamps must not become `now()` silently | P1 | VERIFIED | AI | Yes | Yes |
| DATA-006 | Universe | Historical backtests need historical NIFTY membership | P0 | VERIFIED | AI | Yes | Yes |
| DATA-007 | Sector | Historical analysis needs historical sector mappings | P1 | VERIFIED | AI | Yes | Yes |
| FEAT-001 | Leakage | Build randomized future-mutation leakage test | P0 | VERIFIED | AI | Yes | Yes |
| FEAT-002 | Leakage | Enforce causal feature contracts at API/type boundary | P0 | VERIFIED | AI | Yes | Yes |
| FEAT-003 | Alignment | Add minimum overlap/coverage assertions | P1 | VERIFIED | AI | Yes | Yes |
| FEAT-004 | Labels | Verify adaptive deadband only uses information available at t | P0 | VERIFIED | AI | Yes | Yes |
| FEAT-005 | Labels | Verify purge/embargo against feature lookback + label horizon | P0 | VERIFIED | AI | Yes | Yes |
| REF-001 | Quant | Reference-level failures return plausible values instead of unavailable state | P0 | VERIFIED | AI | Yes | Yes |
| REF-002 | Quant | Remove broad exception swallowing from reference-level engine | P1 | VERIFIED | AI | Yes | Yes |
| REF-003 | Quant | Validate support/resistance methodology and sensitivity | P2 | VERIFIED | AI | Yes | Yes |
| CAL-001 | Calibration | Exhaustively test sparse/non-contiguous calibration bins | P0 | VERIFIED | AI | Yes | Yes |
| CAL-002 | Calibration | Add adversarial calibration fixtures | P1 | VERIFIED | AI | Yes | Yes |
| CAL-003 | Calibration | Separate calibration data from strategy-edge validation | P0 | VERIFIED | AI | Yes | Yes |
| MODEL-001 | Lineage | Version model + feature schema + config + data | P0 | VERIFIED | AI | Yes | Yes |
| MODEL-002 | Config | Generate and persist configuration hash | P0 | VERIFIED | AI | Yes | Yes |
| MODEL-003 | Artifact | Store model hash and training metadata | P1 | VERIFIED | AI | Yes | Yes |
| MODEL-004 | Validity | Separate engine health from model validity | P0 | VERIFIED | AI | Yes | Yes |
| BACK-001 | Research | Prove backtest/live feature equivalence | P0 | VERIFIED | AI | Yes | Yes |
| BACK-002 | Research | Prove backtest/live signal equivalence | P0 | VERIFIED | AI | Yes | Yes |
| BACK-003 | Costs | Stress-test slippage and transaction costs | P1 | VERIFIED | AI | Yes | Yes |
| BACK-004 | Survivorship | Eliminate current-universe survivorship bias | P0 | VERIFIED | AI | Yes | Yes |
| SCAN-001 | Research | Validate top-N scanner selection separately | P1 | VERIFIED | AI | Yes | Yes |
| SCALP-001 | Research | Separate scalping validation from normal intraday validation | P0 | VERIFIED | AI | Yes | Yes |
| SCHED-001 | Reliability | Scheduler needs explicit failure/degraded/failed states | P1 | VERIFIED | AI | Yes | Yes |
| SCHED-002 | Concurrency | Prevent duplicate simultaneous prediction cycles | P1 | VERIFIED | AI | Yes | Yes |
| SCHED-003 | Storage | Handle SQLite lock/concurrency failures deterministically | P1 | VERIFIED | AI | Yes | Yes |
| EVENT-001 | Events | Preserve source/event availability semantics end-to-end | P1 | OPEN | AI | No | No |
| EVENT-002 | Events | Improve keyword/entity event classification | P2 | OPEN | AI | No | No |
| TEST-001 | Testing | Add full end-to-end integration test | P0 | OPEN | AI | No | No |
| TEST-002 | Testing | Replace reliance on self-tests with pytest integration suite | P1 | OPEN | AI | No | No |
| TEST-003 | Testing | Raise quality from line coverage to invariant coverage | P1 | OPEN | AI | No | No |
| DOC-001 | Documentation | Replace stale `BUGS_FOUND.md` with versioned audit history | P1 | OPEN | AI | No | No |
| DOC-002 | Documentation | Add `ENGINEERING_CONTRACT.md` for future AI agents | P1 | OPEN | AI | No | No |
| OPS-001 | Operations | Add scheduler circuit breaker and alertable health states | P1 | VERIFIED | AI | Yes | Yes |
| OPS-002 | Operations | Add reproducibility bundle to prediction records | P1 | OPEN | AI | No | No |

---

# 4. Phase 0 — Baseline and Repository Inventory

## Objective

Create a verified baseline before changing anything.

### Tasks

- [x] Run full test suite.
- [x] Record Python version.
- [x] Record dependency versions.
- [x] Record Git commit SHA.
- [x] Record test count.
- [x] Record coverage.
- [x] Run static analysis.
- [x] Run import checks for every production module.
- [x] Identify all API routes.
- [x] Identify all model training entry points.
- [x] Identify all prediction entry points.
- [x] Identify all data-fetch entry points.
- [x] Identify all event-fetch entry points.
- [x] Identify all persistence/storage entry points.
- [x] Identify all scheduler entry points.
- [x] Identify all Streamlit/UI entry points.
- [x] Identify all environment variables.

### Required output

Create:

```text
AUDIT_BASELINE.md
```

containing:

```text
commit:
python:
os:
dependencies:
tests:
coverage:
known failures:
known warnings:
API routes:
training entrypoints:
prediction entrypoints:
scheduler:
data providers:
event providers:
```

Do not fix application behavior in Phase 0 unless necessary to make the baseline executable.

---

# 5. Phase 1 — API Security Hardening

## SEC-001 — Authentication disabled must never mean ADMIN

### Current risk

Authentication-disabled mode must not create an ADMIN identity.

### Required behavior

Development:

```text
AUTH_DISABLED
→ explicitly DEVELOPMENT
→ read-only or restricted access
```

Production:

```text
AUTH_DISABLED
→ startup failure
```

### Acceptance criteria

- No anonymous request receives ADMIN privileges.
- Production cannot start without authentication.
- Regression tests cover:
  - auth enabled
  - auth disabled
  - production + auth disabled
  - development + auth disabled

---

## SEC-002 — Remove default secrets

Remove hard-coded fallback ADMIN credentials.

Required behavior:

```text
missing production secret
→ startup failure
```

Do not replace a secret with another known default.

### Tests

- missing secret test
- production startup test
- development explicit secret test
- secret scanning test

---

## SEC-003 — Endpoint authorization matrix

Create:

```text
docs/API_AUTHORIZATION.md
```

with:

| Endpoint | Method | Public | Read-only | Admin | Internal |
|---|---|---:|---:|---:|---:|

Every route must have an explicit classification.

No route may accidentally inherit privilege from global assumptions.

---

## SEC-004 / SEC-005 — Persistent shared risk state

Risk-control state must survive process restarts and remain identical across workers.

Preferred architecture:

```text
API worker A ─┐
API worker B ─┼──> shared persistent risk state
API worker C ─┘
```

Required metadata:

```text
enabled
changed_at
changed_by
version
reason
```

Use optimistic versioning where practical.

### Tests

- restart persistence
- two-process consistency
- concurrent update
- audit record creation

---

## SEC-006 — Protect metrics

Decide explicitly:

```text
private-only
```

or authenticated access.

Do not leave `/metrics` exposed accidentally.

---

## SEC-007 — Distributed rate limiting

If production supports multiple workers/processes:

```text
in-memory limiter
```

must not be the authoritative production limiter.

Use a shared backend or reverse-proxy/API-gateway limit.

Development fallback may remain local if clearly marked.

---

# 6. Phase 2 — Canonical Prediction Pipeline

## API-002 — One prediction context

There must be exactly one canonical prediction context builder.

Required conceptual API:

```python
PredictionContext(
    market_data=...,
    index_data=...,
    macro_events=...,
    corporate_events=...,
    news_events=...,
    timestamp=...,
    data_status=...,
    event_status=...,
)
```

Scheduler, API, scanner, Streamlit and other callers must use the same pipeline.

### Prohibited

```python
macro_events=[]
corporate_events=[]
news_articles=[]
```

as a shortcut for production prediction.

### Acceptance test

For identical fixtures:

```text
scheduler prediction == API prediction == replay prediction
```

---

# 7. Phase 3 — Data Reliability

## DATA-001 — No silent stale DataFrame

Replace ambiguous return types.

Preferred:

```python
MarketDataResult(
    data=DataFrame | None,
    status=LIVE | CACHED_FRESH | CACHED_STALE | UNAVAILABLE,
    fetched_at=...,
    source=...,
    age_seconds=...,
    error=...,
)
```

All production callers must inspect the result.

### Rule

```text
CACHED_STALE
→ never silently treated as LIVE
```

If stale data is allowed for a research-only path, that must be explicit.

---

## DATA-002 — Session-aware freshness

Freshness must account for:

- exchange trading sessions
- weekends
- holidays
- market open/close
- expected bar interval
- provider delay

Create tests around:

- Friday → Monday
- trading holiday
- pre-market
- post-market
- missing latest bar
- delayed provider response

---

## DATA-003 — Alignment coverage

For stock/index alignment calculate:

```text
alignment_ratio =
matched_timestamps / expected_stock_timestamps
```

Set an explicit threshold.

Below threshold:

```text
FEATURE_STATUS = DEGRADED/UNAVAILABLE
```

Do not silently continue.

---

## DATA-004 — UTC internally

All stored timestamps should be timezone-aware UTC.

Only convert to `Asia/Kolkata` at presentation boundaries.

Add a test rejecting naive datetimes in core pipeline objects.

---

## DATA-005 — Invalid event timestamps

Never:

```python
invalid_timestamp → datetime.now()
```

Instead:

```text
INVALID_TIMESTAMP
```

or reject the event.

An old event must never become a current event due to parsing failure.

---

# 8. Phase 4 — Leakage-Proof Feature Engineering

## FEAT-001 — Randomized adversarial leakage test

For multiple random historical positions `j`:

1. Copy dataset.
2. Mutate only future bar `j`.
3. Recompute features.
4. Assert all features before `j` are unchanged.

Repeat across:

- OHLC
- volume
- index
- reference levels
- rolling features
- event features
- market features
- volatility
- labels/deadbands

### Required property

For all `i < j`:

```text
features_original[i] == features_mutated[i]
```

within an explicitly defined numerical tolerance.

---

## FEAT-002 — Causal feature contract

Do not rely only on `_feat` naming.

Create a feature schema:

```python
FeatureSpec(
    name=...,
    source=...,
    max_lookback=...,
    shift=...,
    allowed_timestamp=...,
)
```

Training code must accept only validated feature schemas.

---

## FEAT-003 — Alignment coverage

Every cross-series feature must expose:

```text
source_count
matched_count
coverage_ratio
```

and have a defined minimum.

---

## FEAT-004 — Adaptive deadband causality

For every timestamp `t`:

```text
deadband[t]
```

must depend only on data available at or before `t`.

Create a boundary test specifically around train/test split.

---

## FEAT-005 — Purge and embargo

Training split logic must account for:

```text
maximum feature lookback
+
label horizon
+
embargo
```

Do not rely only on:

```text
max(train_timestamp) < min(test_timestamp)
```

Create a temporal information interval for each observation.

---

# 9. Phase 5 — Reference-Level Engine

## REF-001 — No plausible fallback values

If reference-level computation fails:

```text
status = UNAVAILABLE
levels = None
```

Never manufacture:

```text
support = current_price
resistance = current_price
MA = current_price
```

unless that is explicitly the mathematically intended value rather than a failure fallback.

---

## REF-002 — Exception handling

Separate:

```text
expected data failure
```

from:

```text
programming failure
```

Do not catch all exceptions and continue silently.

Log structured errors with:

```text
symbol
timestamp
operation
exception_type
message
```

---

## REF-003 — Method validation

Run sensitivity analysis across:

- lookback periods
- quantile thresholds
- volatility regimes
- symbols
- timeframes

Do not change the formula simply to improve historical performance.

The purpose is robustness validation, not curve fitting.

---

# 10. Phase 6 — Calibration

## CAL-001 — Sparse-bin tests

Explicitly test:

```text
bins = [populated, empty, populated, empty, populated]
```

Verify lookup by numeric interval, not array position.

Test:

- empty bins
- one populated bin
- all bins populated
- extreme confidence
- confidence outside expected range
- NaN
- infinity

---

## CAL-002 — Adversarial calibration fixtures

Create fixtures for:

```text
overconfident
underconfident
constant 0.5
perfectly calibrated
inverted confidence
severe class imbalance
small sample
concept drift
```

Calibration must fail safely where statistical evidence is insufficient.

---

## CAL-003 — Separate validation datasets

Do not use the same data for:

```text
training
calibration
edge validation
final holdout
```

unless explicitly justified.

Preferred:

```text
TRAIN
  ↓
CALIBRATION
  ↓
EDGE VALIDATION
  ↓
FINAL HOLDOUT
```

The final holdout must remain untouched until the evaluation stage.

---

# 11. Phase 7 — Model Lineage

Every trained model artifact must record:

```json
{
  "model_version": "...",
  "model_hash": "...",
  "feature_schema_hash": "...",
  "config_hash": "...",
  "dataset_version": "...",
  "dataset_start": "...",
  "dataset_end": "...",
  "universe_version": "...",
  "training_commit": "...",
  "python_version": "...",
  "dependency_lock_hash": "...",
  "training_timestamp": "..."
}
```

## MODEL-002 — Configuration hash

Changes to:

- horizons
- features
- thresholds
- model parameters
- risk settings
- calibration parameters

must change the configuration hash.

---

# 12. Phase 8 — Engine Health vs Model Validity

Create two separate states.

## ENGINE HEALTH

Examples:

```text
HEALTHY
DEGRADED
FAILED
```

Checks:

- API
- data
- events
- scheduler
- storage
- model file availability

## MODEL VALIDITY

Examples:

```text
UNTRAINED
TRAINED
CALIBRATED
VALIDATED
LIVE_ELIGIBLE
STALE
INVALID
```

A healthy engine must NOT automatically imply a valid model.

### Verification (MODEL-004)
- **Status:** `VERIFIED`
- **Module Created:** `model_validity.py` (`ModelValidityResult`, `evaluate_model_validity`, `can_serve_live_signal`)
- **Engine Checks:** 6 checks implemented in `HealthRegistry.get_engine_health()` (`api`, `data`, `events`, `scheduler`, `storage`, `model_file_availability`)
- **Model Validity States:** `UNTRAINED`, `TRAINED`, `CALIBRATED`, `VALIDATED`, `LIVE_ELIGIBLE`, `STALE`, `INVALID`
- **API Endpoints:**
  - `GET /api/v1/health` reports `engine_health` and `checks` dictionary.
  - `GET /api/v1/model-validity/{symbol}` reports granular model lifecycle state.
  - `GET /api/v1/signal/{symbol}` and stream strictly enforce `can_serve_live_signal`.
- **Test Suite:** `tests/test_phase8_engine_health_vs_model_validity.py` (22 tests passed).

---

# 13. Phase 9 — Backtest/Live Equivalence

This is a release-blocking research test.

For historical timestamp `T`:

```text
                 historical data
                       |
                       v
                replay/live path
                       |
                    SIGNAL A

                 historical data
                       |
                       v
                   backtest
                       |
                    SIGNAL B
```

Require:

```text
A == B
```

for:

- feature values
- feature schema
- event availability
- risk state
- calibration state
- final eligibility
- action/verdict

Allow only explicitly documented nondeterministic differences.

### Verification (BACK-001 & BACK-002)
- **Status:** `VERIFIED`
- **Feature Equivalence (`BACK-001`):**
  - Implemented `verify_feature_equivalence()` in `feature_engineer.py` testing batch backtest features vs sequential sliced live features.
  - Verified across multiple timestamps and multiple horizons (`INTRADAY`, `3D`, `30D`).
  - Confirmed column schema, ordering, dtypes, and numerical value parity within $\le 1e-9$ numerical tolerance.
  - Proved zero lookahead leakage across session day boundaries.
- **Signal Equivalence (`BACK-002`):**
  - Enhanced `Backtester` with `replay_point_in_time_signal()`, `run_signal_replay_backtest()`, and `verify_signal_equivalence()`.
  - Proved point-in-time signal identity between live `Predictor.generate_signal()` and backtest replay across: `action`, `model_predicted_class`, `raw_confidence`, `risk_adjusted_confidence`, `calibrated_confidence`, and `suppressed`.
  - Proved strict point-in-time event filtering ($t \le T$) preventing future events from leaking into backtest signals.
  - Documented allowed nondeterministic differences: wall-clock execution time (`now()`) in signal metadata and runtime latency.
- **Test Suite:** `tests/test_phase9_backtest_live_equivalence.py` (8/8 passed).

---

# 14. Phase 10 — Survivorship Bias

## DATA-006 / BACK-004

- **Status:** `VERIFIED`
- **Root Cause:**
  - `UniverseProvider` previously loaded only a single snapshot (`nifty50_constituents_v1.json`) from March 2024. The `as_of_date` parameter in `get_constituents` was completely ignored, silently applying today's constituents retroactively across all historical periods.
  - Backtesting and signal replay engines had no knowledge of historical index membership, resulting in severe survivorship bias (testing recent winners like `BEL`, `TRENT`, `SHRIRAMFIN`, `ADANIENT` in periods prior to their index inclusion, and excluding historical constituents like `UPL`, `HDFC`, `SHREECEM`, `IOC`, `GAIL`, `ZEEL`).
- **Remediation:**
  1. Created canonical point-in-time membership dataset `data/universe/nifty50_membership_history.csv` (`effective_from`, `effective_to`, `symbol`, `company`, `action`, `notes`) tracking all reconstitution events from 2020 through present, guaranteeing exactly 50 unique constituents on any query date $T$.
  2. Created versioned snapshots `nifty50_constituents_v1.json` (effective 2024-03-28 to 2024-09-29) and `nifty50_constituents_v2.json` (effective 2024-09-30 to present).
  3. Enhanced `UniverseProvider` (`universe_provider.py`):
     - `eligible_universe(as_of)`: Authoritative point-in-time constituent lookup returning exactly 50 constituents active at $T$.
     - `get_constituents_as_of(as_of)`: Direct alias to `eligible_universe`.
     - `get_constituents(version=None, as_of_date=None)`: Time-aware routing when `as_of_date` is supplied.
     - `is_member(symbol, as_of)`: Point-in-time membership check supporting strings, dates, datetimes, and Timestamps.
     - `get_membership_history(symbol=None)` & `get_membership_changes(start_date, end_date)`: Reconstitution history inspection.
  4. Enhanced `Backtester` (`backtester.py`):
     - Integrated `UniverseProvider`.
     - Supported `enforce_universe: bool = False` across `run_backtest_for_symbol`, `replay_point_in_time_signal`, and `run_signal_replay_backtest`.
     - When `enforce_universe=True`: Prohibits trades and suppresses signals on bars where `symbol ∉ eligible_universe(T)`, recording `n_universe_excluded_bars` and suppression reason `SURVIVORSHIP_BIAS_GATE`.
     - Added `run_universe_backtest()` and `quantify_survivorship_bias()` to measure portfolio-level distortion between biased and bias-free backtests.
- **Verification Command:**
  ```bash
  pytest tests/test_phase10_survivorship_bias.py -v
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase10_survivorship_bias.py` (8/8 passed).
  - Master Regression Suite: 154 passed across all 10 phases.

---

# 15. Phase 11 — Historical Sector Mapping

Sector mapping must also be time-aware:

```text
symbol
sector
effective_from
effective_to
source
```

Historical event analysis must use the sector valid at event time.

## DATA-007

- **Status:** `VERIFIED`
- **Root Cause:**
  - `SectorMapProvider` (`sector_provider.py`) previously only exposed static mapping dictionaries or snapshots that applied current constituent sector classifications unconditionally.
  - Event classifiers (`event_classifier.py`) and signal generators (`predictor.py`) accessed static `SECTOR_MAP` unconditionally, introducing lookahead and misclassification bias during historical event analysis, backtesting, and corporate restructuring (e.g. `TATACONSUM` transition from Beverages to FMCG; `SHRIRAMFIN` from AutoFinance to NBFC; historical constituents `HDFC`, `IOC`, `GAIL`, `ZEEL`, `SHREECEM`, `UPL`).
- **Remediation:**
  1. Created canonical point-in-time sector dataset `data/sector_map/nifty50_sector_history.csv` with schema (`symbol`, `sector`, `industry`, `effective_from`, `effective_to`, `source`, `notes`) tracking sector classifications and corporate restructuring transitions from 2020 through present.
  2. Enhanced `SectorMapProvider` (`sector_provider.py`):
     - `HistoricalSectorRecord` dataclass.
     - Automatic parsing and indexing of `nifty50_sector_history.csv`.
     - `get_sector(symbol, as_of=None, version=None)` & `get_sector_as_of(symbol, as_of)`: Point-in-time sector lookup (strict: returns `"Unknown"` if symbol not active at $T$).
     - `get_industry(symbol, as_of=None, version=None)` & `get_industry_as_of(symbol, as_of)`: Granular point-in-time industry description.
     - `get_sector_map_as_of(as_of)`: Full `{symbol: sector}` dictionary valid on query date $T$.
     - `get_symbols_for_sector(sector, as_of=None)`: Symbols belonging to a sector at timestamp $T$.
     - `get_sector_history(symbol=None)`: Historical classification inspection.
  3. Integrated Point-in-Time Sector Resolution in `EventClassifier` (`event_classifier.py`):
     - `classify_corporate_event`: Uses `sector_map_provider.get_sector(symbol, as_of=source_timestamp)`.
     - `classify_news_event`: Resolves `sector_val = sector_map_provider.get_sector(symbol, as_of=pub_at)` and point-in-time sector expansion via `_sectors_to_tickers(matched_sectors, as_of=pub_at)`.
     - `classify_macro_event`: Expands sector tickers at event date via `_sectors_to_tickers(sectors, as_of=macro_event.event_date)`.
  4. Integrated Point-in-Time Risk Multiplier Resolution in `Predictor` (`predictor.py`):
     - `generate_signal`: Resolves point-in-time sector using the latest bar timestamp in `stock_df.index[-1]` for risk multiplier lookups.
- **Verification Command:**
  ```bash
  pytest tests/test_phase11_historical_sector_mapping.py -v
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase11_historical_sector_mapping.py` (9/9 passed).
  - Master Regression Suite: 163 passed across all 11 phases in 92.10s.

---

# 16. Phase 12 — Costs and Slippage

Backtests must support configurable stress scenarios.

At minimum:

```text
BASE
+25% cost
+50% cost
+100% cost
HIGH_SLIPPAGE
LOW_LIQUIDITY
```

Separate:

```text
gross P&L
transaction costs
slippage
net P&L
```

Never report only gross performance when evaluating strategy viability.

- **Remediation Details (Issue `BACK-003` - VERIFIED):**
  1. Configurable Cost and Slippage Stress Scenarios in `ExecutionSimulator` (`execution_simulator.py`):
     - Added `create_scaled_cost_scenario()`.
     - Defined `PHASE12_COST_SCENARIOS`: `BASE`, `+25% cost`, `+50% cost`, `+100% cost`, `HIGH_SLIPPAGE` (25.0 bps slippage), and `LOW_LIQUIDITY` (35.0 bps slippage + 1.5x brokerage/exchange fee).
     - Programmatic aliases supported: `COST_PLUS_25`, `COST_PLUS_50`, `COST_PLUS_100`.
     - Retained 4-tier `STANDARD_COST_SCENARIOS` (`OPTIMISTIC`, `BASE`, `PESSIMISTIC`, `STRESS`) in `run_sensitivity_analysis()` for 100% backwards compatibility.
     - Added `run_stress_test_analysis()` executing all Phase 12 scenarios.
  2. Strict Decomposition of P&L in `SimulationReport` and `BacktestResult`:
     - Decomposed and tracked: `gross_pnl_pct`, `transaction_costs_pct`, `slippage_pct`, `net_pnl_pct`.
     - Invariant enforced: `gross_pnl_pct - transaction_costs_pct - slippage_pct == net_pnl_pct`.
     - Added `pnl_breakdown` and `is_viable` property to `SimulationReport`.
     - Added `is_viable_after_friction` and `pnl_breakdown` to `BacktestResult`.
  3. Viability Gate Enforced in `Backtester` (`backtester.py`):
     - Rejection rule: Never report only gross performance. If `net_pnl <= 0.0` or `sim_report.cumulative_net_return_pct <= 0.0`, `is_live_worthy` is strictly set to `False` and an explicit friction rejection reason is appended to `gate_reasons`.
     - Added `Backtester.run_stress_test()` API returning multi-scenario breakdown.
- **Verification Command:**
  ```bash
  pytest tests/test_phase12_costs_and_slippage.py -v
  pytest test_p0_regressions.py tests/test_phase1_security.py tests/test_phase2_canonical_pipeline.py tests/test_phase3_data_reliability.py tests/test_phase4_leakage.py tests/test_phase5_reference_levels.py tests/test_phase6_calibration.py tests/test_phase7_model_lineage.py tests/test_phase8_engine_health_vs_model_validity.py tests/test_phase9_backtest_live_equivalence.py tests/test_phase10_survivorship_bias.py tests/test_phase11_historical_sector_mapping.py tests/test_phase12_costs_and_slippage.py test_quant_correctness.py -v
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase12_costs_and_slippage.py` (8/8 passed in 18.55s).
  - Master Regression Suite: 171 passed across all 13 test suites in 105.50s (0 regressions).

---

# 17. Phase 13 — Scanner Validation

The scanner's top-N selection is a separate statistical layer.

Evaluate:

```text
all eligible signals
```

versus:

```text
selected top-N signals
```

Measure separately:

- calibration
- hit rate
- class distribution
- turnover
- cost sensitivity
- regime sensitivity

Do not assume individual-model metrics remain valid after ranking/selection.

- **Remediation Details (Issue `SCAN-001` - VERIFIED):**
  1. Dedicated Scanner Validation Module (`scanner_validator.py`):
     - Implemented `ScannerValidator` treating top-N selection as an independent statistical layer.
     - Compares `all eligible signals` vs `selected top-N signals` across all 6 mandated dimensions:
       - **Calibration**: Computes ECE and Brier score independently for both cohorts; detects Winner's Curse degradation.
       - **Hit Rate**: Computes win rate, average/median return, and win/loss ratio.
       - **Class & Sector Distribution**: Calculates class shares, point-in-time sector distribution, and Herfindahl-Hirschman Index (HHI) concentration score.
       - **Turnover**: Measures rank stability, churn rate, retention rate, and portfolio turnover friction between consecutive scan sweeps.
       - **Cost Sensitivity**: Evaluates net returns across all 6 Phase 12 stress scenarios (`BASE`, `+25% cost`, `+50% cost`, `+100% cost`, `HIGH_SLIPPAGE`, `LOW_LIQUIDITY`).
       - **Regime Sensitivity**: Breaks down performance across market regimes (`BULL_TREND`, `BEAR_TREND`, `HIGH_VOLATILITY`, `LOW_VOLATILITY_SIDEWAYS`).
  2. Strict Validation Gates:
     - Winner's Curse / Calibration Gate: Rejects selection if Top-N ECE > 0.20 or degrades significantly from eligible baseline (`DEGRADED_CALIBRATION`).
     - Friction & Turnover Gate: Rejects selection if Top-N produces positive gross return but non-positive net return after turnover and transaction costs (`FRICTION_FAILURE`).
     - Regime Resilience Gate: Detects catastrophic collapse in any major market regime (`REGIME_UNSTABLE`).
  3. Scanner Integration (`scanner.py`):
     - Added `validate_selection(historical_scans, index_df, top_n, cost_scenarios)` to `OpportunityScanner`.
     - Added point-in-time `sector` resolution to `ScanResult`.
     - Preserved 100% backwards compatibility for `scan()` and `scan_with_summary()`.
- **Verification Command:**
  ```bash
  pytest tests/test_phase13_scanner_validation.py -v
  pytest test_p0_regressions.py tests/test_phase1_security.py tests/test_phase2_canonical_pipeline.py tests/test_phase3_data_reliability.py tests/test_phase4_leakage.py tests/test_phase5_reference_levels.py tests/test_phase6_calibration.py tests/test_phase7_model_lineage.py tests/test_phase8_engine_health_vs_model_validity.py tests/test_phase9_backtest_live_equivalence.py tests/test_phase10_survivorship_bias.py tests/test_phase11_historical_sector_mapping.py tests/test_phase12_costs_and_slippage.py tests/test_phase13_scanner_validation.py test_quant_correctness.py -v
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase13_scanner_validation.py` (9/9 passed in 3.18s).
  - Master Regression Suite: 180 passed across all 14 test suites in 204.69s (0 regressions).

---

# 18. Phase 14 — Scalping Isolation

Scalping must have independent:

```text
model
dataset
labels
horizon
calibration
edge validation
cost model
slippage model
latency assumptions
```

Do not reuse ordinary intraday validation thresholds without evidence.

- **Remediation Details (Issue `SCALP-001` - VERIFIED):**
  1. **Independent Horizon & Configuration (`config.py`)**:
     - Introduced dedicated `HORIZON_SCALP = "SCALP"`.
     - Added `HORIZON_SCALP` to `HORIZON_CONFIG`: 1m bar interval, 2 forward horizon bars (2-minute holding period), tight 0.05% default deadband, 15-day rolling history, and 250 minimum samples.
     - Isolated from `HORIZON_INTRADAY` (5m bar interval, 6 bars / 30-minute holding period, 0.15% deadband).
     - Configured supporting mappings: `HORIZON_TO_MA_PERIOD[HORIZON_SCALP] = 5`, `HORIZON_TO_SR_METHOD[HORIZON_SCALP] = "bollinger"`, and `EVENT_IMPACT_HORIZON["ORDER_FLOW_IMBALANCE"] = HORIZON_SCALP`.
  2. **Dedicated Scalping Microstructure Dataset (`ScalpDataset` in `scalp_validator.py`)**:
     - Extracts 6 real-time microstructure features without lookahead:
       - `micro_spread_bps`: High-low spread proxy.
       - `tick_volatility_pct`: Normalized bar range.
       - `order_imbalance_proxy`: Close location within bar range `(Close - Open) / (High - Low)`.
       - `micro_momentum_1b`: 1-bar percentage return.
       - `micro_momentum_2b`: 2-bar percentage return.
       - `volume_surge`: Volume relative to 5-bar rolling average.
     - Fast forward labeling (`create_scalp_labels`): 2-bar forward returns with tight 0.05% threshold, trailing unobserved rows strictly NaN-masked.
  3. **Dedicated Scalping Model (`ScalpModel` in `scalp_validator.py`)**:
     - Lightweight `RandomForestClassifier` isolated from macro/sentiment ensembles.
     - Independent probability estimation for classes `["DOWN", "FLAT", "UP"]`.
  4. **Explicit Latency & Slippage Modeling (`ScalpLatencyConfig` & `ScalpCostScenario`)**:
     - Explicit execution delay (50ms to 500ms), applying 1 bp price drag per 100ms delay.
     - Adverse stop penalties (`adverse_stop_bps` = 6.0 to 8.0 bps) modeling toxic flow on stop-outs.
     - Four mandatory cost scenarios: `SCALP_BASE`, `SCALP_ADVERSE_SELECTION`, `SCALP_HIGH_LATENCY`, and `SCALP_STRESS`.
  5. **Independent Calibration & Edge Validation (`ScalpValidator`)**:
     - Dedicated calibration threshold: $ECE \le 0.08$. ECE > 0.08 or insufficient data flagged as `MISCALIBRATED`.
     - Discrete execution backtest against zero-drift baseline: requires $N \ge 30$, win rate $> 52\%$ or profit factor $> 1.15$, and positive net alpha after latency and statutory frictions.
     - Multi-scenario cost sensitivity analysis across all 4 scalping stress scenarios.
  6. **Engine Safety Gate & Modernization (`scalping.py`)**:
     - Refactored `ScalpingEngine` to validate each symbol via `ScalpValidator.validate_symbol(symbol, stock_df)`.
     - Suppresses unvalidated or miscalibrated symbols before signal emission.
     - Replaces wide intraday stops (3.0x ATR) with micro-horizon 1.0x ATR stops and 1.5x ATR targets.
     - Attached `ScalpValidationReport` to `ScalpSetup` while preserving 100% backwards compatibility for callers.
- **Verification Command:**
  ```bash
  pytest tests/test_phase14_scalping_isolation.py tests/test_scalping.py -v
  pytest tests/test_phase1_security.py tests/test_phase2_canonical_pipeline.py tests/test_phase3_data_reliability.py tests/test_phase4_leakage.py tests/test_phase5_reference_levels.py tests/test_phase6_calibration.py tests/test_phase7_model_lineage.py tests/test_phase8_engine_health_vs_model_validity.py tests/test_phase9_backtest_live_equivalence.py tests/test_phase10_survivorship_bias.py tests/test_phase11_historical_sector_mapping.py tests/test_phase12_costs_and_slippage.py tests/test_phase13_scanner_validation.py tests/test_phase14_scalping_isolation.py tests/test_scalping.py -q
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase14_scalping_isolation.py` (10/10 passed in 3.40s).
  - Legacy Suite: `tests/test_scalping.py` (1/1 passed in 3.47s).
  - Master Regression Suite: 161 passed across all 15 test suites in 169.51s (0 regressions).

---

# 19. Phase 15 — Scheduler Reliability

Implement explicit state:

```text
STARTING
HEALTHY
DEGRADED
FAILED
HALTED
```

Track:

```text
consecutive_failures
last_success
last_failure
last_prediction
last_data_fetch
last_event_fetch
```

## Circuit breaker

Example policy:

```text
1 failure  → retry
2 failures → degraded
N failures → halted
```

The exact N must be configurable and tested.

A running Python process is not equivalent to a healthy scheduler.

- **Remediation Details (Issues `SCHED-001`, `OPS-001` - VERIFIED):**
  1. **Configurable Circuit Breaker & Health Overrides (`config.py`)**:
     - Added `SCHEDULER_CIRCUIT_BREAKER_MAX_FAILURES = 5` (N consecutive failures to trip to `HALTED`).
     - Added `SCHEDULER_CIRCUIT_BREAKER_DEGRADED_FAILURES = 2` (consecutive failures to enter `DEGRADED`).
     - Added `"scheduler": {"degraded": 2, "down": 5}` to `HEALTH_THRESHOLD_OVERRIDES` ensuring complete alignment with `health_monitor.py`.
  2. **Scheduler Circuit Breaker & Metric Tracking (`scheduler_circuit_breaker.py`)**:
     - Explicit Lifecycle States: `SCHEDULER_STATE_STARTING`, `SCHEDULER_STATE_HEALTHY`, `SCHEDULER_STATE_DEGRADED`, `SCHEDULER_STATE_FAILED`, `SCHEDULER_STATE_HALTED`.
     - `SchedulerMetrics` dataclass tracking `consecutive_failures`, `total_cycles`, `successful_cycles`, `failed_cycles`, `last_success`, `last_failure`, `last_prediction`, `last_data_fetch`, `last_event_fetch`, `last_error`, `circuit_breaker_tripped`, `circuit_breaker_tripped_at`, `circuit_breaker_trip_reason`.
     - Deterministic State Transitions:
       - 0 failures -> `HEALTHY` (or `STARTING` initially).
       - 1 failure -> retry; state remains `STARTING` or `HEALTHY`.
       - >= 2 consecutive failures -> `DEGRADED`.
       - >= 5 consecutive failures -> `HALTED` (trips circuit breaker).
     - Strict execution guard `can_execute()` returning False when `HALTED`.
     - Recovery logic: single success resets `consecutive_failures` to 0 and transitions state back to `HEALTHY`.
     - Administrative manual reset via `reset()`.
  3. **Operational Scheduler Integration (`scheduler.py`)**:
     - Embedded `SchedulerCircuitBreaker` into `Scheduler`.
     - Registered initial scheduler health component in `health_registry`.
     - `run_one_cycle_for_symbol()` verifies `can_execute()`; immediately halts with `CycleResult(success=False, status="CIRCUIT_BREAKER_HALTED")` when halted.
     - Structured cycle execution in `run_cycle()` tracking `last_data_fetch`, `last_event_fetch`, `last_prediction` granularly.
     - On uncaught cycle exception, records failure in circuit breaker and logs comprehensive error.
     - On cycle completion, reports health status to `health_registry.report("scheduler", ok=..., detail=..., error=...)`.
     - Exposed `get_status()` and `reset_circuit_breaker()` methods.
  4. **API Endpoints & Health Alerting (`api.py`)**:
     - Added `GET /api/v1/scheduler/status` to expose real-time scheduler state machine, metrics, and timestamps.
     - Added `POST /api/v1/scheduler/reset-circuit-breaker` requiring `ADMIN` API key auth with audit logging.
     - Updated `translate_health_message()` with human-readable scheduler diagnostic summaries.
- **Verification Command:**
  ```bash
  pytest tests/test_phase15_scheduler_reliability.py test_scheduler.py tests/test_scheduler_extra.py -v
  pytest tests/test_phase1_security.py tests/test_phase2_canonical_pipeline.py tests/test_phase3_data_reliability.py tests/test_phase4_leakage.py tests/test_phase5_reference_levels.py tests/test_phase6_calibration.py tests/test_phase7_model_lineage.py tests/test_phase8_engine_health_vs_model_validity.py tests/test_phase9_backtest_live_equivalence.py tests/test_phase10_survivorship_bias.py tests/test_phase11_historical_sector_mapping.py tests/test_phase12_costs_and_slippage.py tests/test_phase13_scanner_validation.py tests/test_phase14_scalping_isolation.py tests/test_scalping.py tests/test_phase15_scheduler_reliability.py -q
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase15_scheduler_reliability.py` (10/10 passed in 4.85s).
  - Legacy Scheduler Suites: `test_scheduler.py` + `tests/test_scheduler_extra.py` (3/3 passed in 105.91s).
  - Master Regression Suite: 171 passed across all 16 test suites in 100.78s (0 regressions).

---

# 20. Phase 16 — Concurrency

Prevent two independent prediction cycles for the same:

```text
symbol + timestamp + model_version
```

from executing simultaneously.

Use an idempotency key:

```text
prediction_key =
hash(symbol, timestamp, model_version, feature_schema_hash)
```

If the same prediction is requested twice:

```text
return existing result
```

rather than creating duplicate state.

- **Remediation Details (Issue `SCHED-002` - VERIFIED):**
  1. **Deterministic Idempotency Key Formulation (`prediction_concurrency.py`)**:
     - Implemented `compute_prediction_key(symbol, timestamp, model_version, feature_schema_hash) -> str` which normalizes inputs and computes a 64-character SHA256 hex digest:
       `hashlib.sha256(f"{symbol.upper()}:{normalize_timestamp_for_key(timestamp)}:{model_version}:{feature_schema_hash}".encode("utf-8")).hexdigest()`.
     - Implemented `normalize_timestamp_for_key(timestamp)` ensuring all datetime objects, pandas Timestamps, and strings normalize to standard UTC ISO-8601 strings (`YYYY-MM-DDTHH:MM:SSZ`).
  2. **Thread-Safe In-Flight Concurrency Coordinator (`PredictionConcurrencyCoordinator`)**:
     - Coordinates concurrent prediction requests across threads using reentrant locks (`threading.RLock`) and synchronized execution events (`threading.Event`).
     - **In-flight Deduplication**: If multiple threads request a prediction cycle for the same key simultaneously, the first thread executes the cycle while subsequent threads wait on the event. Upon completion, the result is shared across all waiting callers, completely suppressing duplicate model inference.
     - **Sequential Deduplication & LRU/TTL Cache**: In-memory bounded cache (`_recent_cache`) immediately returns existing results for repeated requests with zero inference and zero database operations.
     - **Safe Error Handling**: Exceptions during cycle execution are recorded and propagated cleanly to all waiting callers without deadlocks or leaked in-flight tokens.
     - Tracks granular concurrency metrics: `total_requests`, `deduplicated_requests`, `simultaneous_waits`, `completed_executions`, and `in_flight_count`.
  3. **Idempotent Storage Layer (`models.py` & `history_manager.py`)**:
     - Added indexed `prediction_key` and `feature_schema_hash` columns to `Prediction` model.
     - Updated `PredictionRecord` dataclass to track `prediction_key` and `feature_schema_hash`.
     - In `HistoryManager.save_prediction()`: checks for existing records by `prediction_key` prior to insertion, returning the existing ID and skipping duplicate database writes. Catches concurrency collisions gracefully.
     - Added `get_prediction_by_key(prediction_key)` and `get_prediction_signal_by_key(prediction_key)` to retrieve records and reconstruct complete `PredictionSignal` instances from stored state.
  4. **Prediction & Scheduler Integration (`predictor.py`, `scheduler.py`, `api.py`)**:
     - `Predictor.generate_signal()` and `Predictor.generate_multi_horizon_signal()` attach deterministic `prediction_key` and `feature_schema_hash` to all emitted signals.
     - `Scheduler.run_one_cycle_for_symbol()` delegates cycle execution to `concurrency_coordinator.execute_or_wait()`, ensuring concurrent and repeated calls for the same bar return existing results.
     - `Scheduler.get_status()` surfaces real-time concurrency metrics, exposed via `GET /api/v1/scheduler/status`.
     - `api.py` and `api_schemas.py` expose `prediction_key` across API responses.
- **Verification Command:**
  ```bash
  pytest tests/test_phase16_concurrency.py -v
  pytest tests/test_phase1_security.py tests/test_phase2_canonical_pipeline.py tests/test_phase3_data_reliability.py tests/test_phase4_leakage.py tests/test_phase5_reference_levels.py tests/test_phase6_calibration.py tests/test_phase7_model_lineage.py tests/test_phase8_engine_health_vs_model_validity.py tests/test_phase9_backtest_live_equivalence.py tests/test_phase10_survivorship_bias.py tests/test_phase11_historical_sector_mapping.py tests/test_phase12_costs_and_slippage.py tests/test_phase13_scanner_validation.py tests/test_phase14_scalping_isolation.py tests/test_scalping.py tests/test_phase15_scheduler_reliability.py tests/test_phase16_concurrency.py -q
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase16_concurrency.py` (10/10 passed in 10.62s).
  - Legacy Scheduler Suites: `test_scheduler.py` + `tests/test_scheduler_extra.py` (3/3 passed in 108.13s).
  - Master Regression Suite: 181 passed across all 17 test suites in 117.25s (0 regressions).
  - Quantitative & P0 Regression Suite: 30 passed in 37.83s (0 regressions).

---

# 21. Phase 17 — Storage Reliability

Handle SQLite:

- lock contention
- duplicate writes
- partial writes
- corrupted records
- restart recovery

For production multi-worker deployment, evaluate moving critical shared state to a proper transactional database.

- **Remediation Details (Issue `SCHED-003` - VERIFIED):**
  1. **Dedicated Storage Reliability Layer (`storage_reliability.py`)**:
     - **SQLite WAL & Concurrency PRAGMAs**: Automatically attaches an SQLAlchemy `@event.listens_for(engine, "connect")` listener configuring:
       - `PRAGMA journal_mode=WAL;` (Write-Ahead Logging allowing concurrent readers without writer blocking).
       - `PRAGMA synchronous=NORMAL;` (reduces fsync bottlenecks while maintaining crash safety in WAL mode).
       - `PRAGMA busy_timeout=30000;` (30-second internal wait timeout to prevent premature lock failures).
       - `PRAGMA foreign_keys=ON;` (enforces relational integrity).
     - **Deterministic Lock Contention Retry (`with_db_retry`)**:
       - Exponential backoff decorator with randomized jitter ($t_{wait} = \text{initial\_delay} \times \text{backoff\_factor}^{\text{attempt}} + \text{jitter}$) intercepting `sqlite3.OperationalError` ("database is locked", "database is busy") up to 5 attempts.
     - **Atomic Transaction Context (`atomic_transaction`)**:
       - Reusable context manager guaranteeing commit on completion, clean rollback on exception, and session closure in `finally` block.
     - **Database Integrity Checking (`verify_database_integrity`)**:
       - Runs `PRAGMA quick_check;` and `PRAGMA integrity_check;`, returning `(True, [])` on healthy database or flagging corruption.
     - **Startup Storage Recovery Routine (`run_storage_recovery`)**:
       - Forces WAL checkpoint (`PRAGMA wal_checkpoint(TRUNCATE);`) to merge WAL frames and truncate auxiliary log.
       - Runs database integrity checks.
       - Audits table counts (`predictions`, `events`, `backtest_metrics`) and flags abandoned/stale unresolved predictions.
       - Reports storage component health to `health_registry.report("storage", ok=..., detail=...)`.
     - **Corrupted Record Resilience**:
       - Implemented `safe_json_loads`, `safe_float`, and `safe_iso_timestamp` to prevent malformed rows from crashing batch read operations.
  2. **Database Engine & History Manager Hardening (`database.py`, `history_manager.py`)**:
     - Configured global engine in `database.py` and custom engines in `HistoryManager` with `create_reliable_engine`.
     - Wrapped `save_prediction`, `save_event`, `save_backtest_result`, and `resolve_outcome` with `@with_db_retry`.
     - **Atomic Bundle Persistence**: Added `save_prediction_bundle(signal, narrative, dca_ladder, is_out_of_sample, events, prediction_key)` to persist predictions and contributing events in a single atomic transaction, preventing partial writes.
     - **Duplicate Event Write Idempotency**: Added duplicate detection on `event_id` in `save_event()`, returning existing success without raising `IntegrityError`.
     - **Resilient Deserialization**: Wrapped row parsing in `get_predictions()`, `get_prediction_by_key()`, `get_prediction_signal_by_key()`, and `get_events_for_symbol()` in safe deserializers so corrupted rows log warnings without crashing queries.
  3. **Scheduler & API Operational Integration (`scheduler.py`, `api.py`)**:
     - In `Scheduler.__init__`: executes startup storage recovery, checkpoints WAL, audits table integrity, and records `storage_recovery_report`.
     - In `Scheduler._execute_cycle()`: delegates to `save_prediction_bundle()` ensuring atomic persistence of predictions and associated events.
     - In `Scheduler.get_status()`: surfaces real-time storage diagnostics (`integrity_ok`, `wal_checkpointed`, `duration_ms`, `unresolved_prediction_count`).
     - In `api.py`: surfaces storage component health in `/api/v1/health` and `/api/v1/scheduler/status`.
  4. **Production Multi-Worker Architecture Document (`docs/STORAGE_ARCHITECTURE_EVALUATION.md`)**:
     - Detailed analysis of SQLite concurrency boundaries (single-writer serialization, busy timeout saturation, network filesystem locking hazards).
     - Target production architecture for PostgreSQL: connection pooling (PgBouncer), row-level locking (`SELECT ... FOR UPDATE`), distributed idempotency, Alembic migrations, and dual-mode environment switching (`DATABASE_URL=postgresql+psycopg2://...`).
- **Verification Command:**
  ```bash
  pytest tests/test_phase17_storage_reliability.py -v
  pytest tests/test_phase1_security.py tests/test_phase2_canonical_pipeline.py tests/test_phase3_data_reliability.py tests/test_phase4_leakage.py tests/test_phase5_reference_levels.py tests/test_phase6_calibration.py tests/test_phase7_model_lineage.py tests/test_phase8_engine_health_vs_model_validity.py tests/test_phase9_backtest_live_equivalence.py tests/test_phase10_survivorship_bias.py tests/test_phase11_historical_sector_mapping.py tests/test_phase12_costs_and_slippage.py tests/test_phase13_scanner_validation.py tests/test_phase14_scalping_isolation.py tests/test_scalping.py tests/test_phase15_scheduler_reliability.py tests/test_phase16_concurrency.py tests/test_phase17_storage_reliability.py -q
  ```
- **Results:**
  - Dedicated Suite: `tests/test_phase17_storage_reliability.py` (15/15 passed in 35.26s).
  - Master Regression Suite: 196 passed across all 18 test suites in 188.07s (0 regressions).
  - Quantitative & P0 Regression Suite: 91 passed in 50.32s (0 regressions).

---

# 22. Phase 18 — Event Pipeline

Event states must remain distinct:

```text
EVENTS_AVAILABLE
NO_EVENTS
EVENT_SOURCE_PARTIAL
EVENT_SOURCE_UNAVAILABLE
INVALID_EVENT_DATA
```

These states must propagate through:

```text
fetcher
→ classifier
→ predictor
→ scheduler
→ API
→ UI
→ history
```

Do not convert `SOURCE_UNAVAILABLE` into `NO_EVENTS`.

---

# 23. Phase 19 — Event Classification

Current keyword-based classification should be treated as a heuristic.

Improve incrementally:

```text
headline
 ↓
entity extraction
 ↓
company/sector/market mapping
 ↓
event category
 ↓
source confidence
 ↓
impact confidence
```

Do not use an NLP model simply for complexity. First add deterministic tests around current classification behavior.

---

# 24. Phase 20 — API Semantic Safety

The API should distinguish:

```json
{
  "model_score": 0.83,
  "calibrated_probability": null,
  "calibration_status": "UNAVAILABLE",
  "edge_status": "NOT_VALIDATED",
  "risk_status": "PASS",
  "live_eligible": false
}
```

Avoid presenting a raw model score as if it were a probability.

Prefer analytical wording such as:

```text
"Model classified the observation as UP."

"Calibration status: PASS."

"Live eligibility: FAIL — edge validation unavailable."
```

Avoid language that turns a research signal into an instruction to transact.

---

# 25. Phase 21 — Integration Test

Create a deterministic end-to-end fixture:

```text
fixed market data
fixed index data
fixed events
fixed model
fixed calibration
fixed config
fixed risk state
```

Then execute:

```text
data
→ validation
→ features
→ model
→ calibration
→ edge
→ risk
→ predictor
→ history
→ API
```

Assertions must cover:

- no future data
- correct timestamps
- expected feature schema
- correct model version
- correct risk state
- correct event state
- deterministic output
- audit record
- history record

---

# 26. Phase 22 — Invariant-Based Testing

Do not rely only on percentage coverage.

Add invariants.

## Data invariants

```text
no duplicate timestamps
timestamps sorted
timezone-aware
no impossible OHLC values
freshness status correct
```

## Feature invariants

```text
no future dependency
schema valid
alignment sufficient
NaN policy explicit
```

## Model invariants

```text
feature schema matches model
model version exists
config hash matches
probabilities valid
```

## Prediction invariants

```text
calibrated probability ∈ [0,1]
risk gate respected
failed validation cannot become LIVE_ELIGIBLE
```

## Storage invariants

```text
unique prediction key
immutable model lineage
audit trail exists
```

---

# 27. Phase 23 — Documentation Cleanup

## DOC-001

Do NOT keep an unversioned stale `BUGS_FOUND.md` as the source of truth.

Replace or transform it into:

```text
docs/audits/
    2026-09-16_initial_audit.md
    2026-09-16_remediation_plan.md
    2026-09-16_verification_report.md
```

Historical audits should never be rewritten to make them appear correct in hindsight.

---

# 28. `ENGINEERING_CONTRACT.md`

Create this file in the repository.

Minimum rules:

```text
1. No feature may use future information.
2. Any feature change requires a leakage test.
3. Any label change requires model-version change.
4. Any config change affecting predictions requires config-hash change.
5. Any model artifact must contain lineage metadata.
6. No production secret may have a source-code fallback.
7. Authentication disabled is never equivalent to ADMIN.
8. Risk state must be shared and persistent.
9. Stale data must never silently become LIVE data.
10. Missing quantitative values must use explicit unavailable states.
11. Backtest and live paths must share canonical feature/prediction logic.
12. Historical backtests must use historical universe membership.
13. Calibration and edge validation must use independent evidence.
14. Scheduler process health must be separated from pipeline health.
15. Every production endpoint must have an explicit authorization class.
16. Every bug fix requires a regression test.
17. Do not optimize historical performance by weakening correctness gates.
18. Do not remove a safety test without replacing it with an equivalent or stronger test.
```

---

# 29. AI Agent Issue Workflow

For every issue:

## Step A — Claim

Change:

```text
STATUS = IN_PROGRESS
```

Add:

```text
agent:
branch:
started:
```

## Step B — Investigate

Record:

```text
affected files:
root cause:
reproduction:
expected:
actual:
```

## Step C — Fix

Make the smallest correct change.

## Step D — Test

Add or update tests.

Run:

```bash
pytest
```

plus the narrowest relevant test first.

## Step E — Verify

Check:

```text
unit test
integration test
regression test
static analysis
import/build
```

## Step F — Record

Update the master tracker:

```text
STATUS = VERIFIED
Test Added = Yes
Verified = Yes
```

Add commit SHA.

---

# 30. Required Issue Record Format

For every significant issue, maintain a record like:

```markdown
## SEC-001 — Authentication Disabled → ADMIN

**Severity:** P0  
**Status:** OPEN  
**Owner:** AI Agent  
**Introduced:** Unknown  
**Detected:** 2026-09-16

### Problem

Authentication-disabled mode currently creates an ADMIN identity.

### Risk

Anonymous callers may obtain administrative capabilities.

### Reproduction

1. Set authentication disabled.
2. Start application.
3. Call an ADMIN endpoint without credentials.
4. Observe authorization result.

### Expected

Request is denied or receives explicitly restricted development privileges.

### Actual

Request can receive ADMIN privileges.

### Fix

Remove ADMIN fallback and fail closed in production.

### Tests

- [ ] auth disabled / development
- [ ] auth disabled / production
- [ ] admin endpoint without key
- [ ] valid admin key
- [ ] invalid admin key

### Verification

Command:

```bash
pytest tests/security -q
```

Result:

```text
PENDING
```

### Commit

```text
PENDING
```
```

---

# 31. Definition of Done

An issue is `VERIFIED` only when ALL applicable conditions are true:

- [ ] Root cause identified.
- [ ] Fix implemented.
- [ ] Regression test added.
- [ ] Existing tests pass.
- [ ] Relevant integration tests pass.
- [ ] No safety gate was weakened.
- [ ] Documentation updated.
- [ ] Issue tracker updated.
- [ ] Git commit recorded.
- [ ] No unrelated behavior changed.
- [ ] Security implications reviewed.
- [ ] Data/quantitative implications reviewed where applicable.

---

# 32. Release Gates

Do not declare the repository production-ready until all P0 issues are:

```text
VERIFIED
```

and the following release gates pass:

## Security

```text
[ ] no default production secrets
[ ] auth cannot degrade into ADMIN
[ ] every privileged route protected
[ ] risk state persistent/shared
[ ] metrics protected
[ ] rate limiting appropriate for deployment model
```

## Data

```text
[ ] freshness semantics verified
[ ] timezone semantics verified
[ ] alignment coverage verified
[ ] event failure states verified
[ ] historical universe available
[ ] historical sector mappings available
```

## ML

```text
[ ] randomized leakage tests pass
[ ] purge/embargo tests pass
[x] calibration tests pass
[x] independent edge validation
[x] model lineage complete
[x] config hash complete
```

## Research validity

```text
[ ] backtest/live feature equivalence proven
[ ] backtest/live signal equivalence proven
[ ] survivorship bias addressed
[ ] cost/slippage stress tests pass
[ ] scanner selection validated
[ ] scalping independently validated
```

## Operations

```text
[ ] scheduler circuit breaker
[ ] duplicate-cycle protection
[ ] storage concurrency handling
[ ] health state meaningful
[ ] alertable failure state
```

---

# 33. Final Verification Report

When remediation is complete, create:

```text
docs/audits/VERIFICATION_REPORT.md
```

with:

```markdown
# Alkame Nifty 50 — Verification Report

## Repository

Commit:
Date:
Python:
Environment:

## Test Summary

Tests:
Passed:
Failed:
Skipped:
Coverage:

## P0 Issues

| ID | Status | Evidence |
|---|---|---|

## P1 Issues

| ID | Status | Evidence |
|---|---|---|

## Security

Summary:

## Data Reliability

Summary:

## Leakage Testing

Summary:

## Model Validation

Summary:

## Backtest/Live Equivalence

Summary:

## Operational Reliability

Summary:

## Known Remaining Risks

1.
2.
3.

## Release Decision

Do not use a subjective score.

Use one of:

- `RELEASE BLOCKED`
- `RELEASE CANDIDATE`
- `VERIFIED FOR DOCUMENTED SCOPE`

The decision must be based on the release gates in this document.
```

---

# 34. Important Agent Constraint

This project is an educational quantitative research system.

Do not "fix" the project by:

- removing safety gates
- disabling validation
- replacing errors with fake values
- using current data to fill historical gaps
- using future data for better backtests
- changing thresholds solely to improve historical returns
- suppressing failing tests
- lowering coverage requirements to make CI pass
- deleting audit records
- hiding warnings
- weakening authentication
- adding default credentials
- silently falling back to synthetic market data in production

A passing pipeline is useful only if the pipeline remains **causal, reproducible, auditable, and honest about uncertainty**.

---

# 35. Recommended Execution Order

Execute in exactly this order unless a dependency requires otherwise:

```text
PHASE 0  → Baseline
PHASE 1  → API Security
PHASE 2  → Canonical Prediction Pipeline
PHASE 3  → Data Reliability
PHASE 4  → Leakage Protection
PHASE 5  → Reference Levels
PHASE 6  → Calibration
PHASE 7  → Model Lineage
PHASE 8  → Health vs Model Validity
PHASE 9  → Backtest/Live Equivalence
PHASE 10 → Historical Universe
PHASE 11 → Historical Sector Mapping
PHASE 12 → Cost/Slippage
PHASE 13 → Scanner
PHASE 14 → Scalping
PHASE 15 → Scheduler
PHASE 16 → Concurrency
PHASE 17 → Storage
PHASE 18 → Events
PHASE 19 → Event Classification
PHASE 20 → API Semantics
PHASE 21 → Integration Tests
PHASE 22 → Invariant Tests
PHASE 23 → Documentation
PHASE 24 → Final Verification
```

Do not jump directly to feature additions before completing P0 correctness/security work.

---

# 36. Final Agent Instruction

At the beginning of every session:

1. Read this file.
2. Read the repository's current status.
3. Select the highest-priority `OPEN` issue whose dependencies are satisfied.
4. Reproduce it before fixing it.
5. Fix only that issue or its directly required dependencies.
6. Add regression coverage.
7. Run tests.
8. Update this tracker.
9. Commit the change.
10. Continue to the next issue only after verification.

At the end of every session, print:

```text
SESSION SUMMARY

Issues attempted:
Issues fixed:
Issues verified:
Issues blocked:
Tests added:
Tests passed:
Tests failed:

Next recommended issue:
Current highest-risk unresolved issue:

Commit(s):
```

**Never claim an issue is fixed without evidence.**
