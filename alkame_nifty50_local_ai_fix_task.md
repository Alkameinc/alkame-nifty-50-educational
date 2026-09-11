# Local AI Agent Task — Audit, Fix, Test, and Raise PR

## Repository

Repository: `https://github.com/Alkameinc/alkame-nifty-50-educational`

## Objective

Perform a systematic engineering and ML-safety audit of the repository, fix the confirmed bugs and weaknesses listed below, add regression tests, update documentation, and raise a pull request.

This is an educational NIFTY-50 prediction/trading-research project. Do **not** introduce claims that the system is profitable or production-ready. Preserve the project's educational/safety-first intent.

---

# 1. Agent Operating Rules

1. Work on a new branch. Do not commit directly to `main`.
2. Inspect the current repository before changing anything.
3. Do not blindly implement every item below. Verify each issue against the current code first.
4. Preserve existing public APIs where practical. If a breaking change is necessary, document it.
5. Prefer small, reviewable commits.
6. Do not add secrets, API keys, credentials, personal data, generated databases, logs, caches, model binaries, or local environment files to git.
7. Do not weaken existing safety gates merely to make tests pass.
8. Do not silently swallow errors that should be actionable.
9. Run the existing tests before modifying code and record the baseline.
10. After modifications, run all available tests plus the new regression tests.
11. Run static checks/linting where available.
12. Update README/documentation when implementation behavior changes.
13. Before creating the PR, inspect `git diff`, `git status`, and the complete changed-file list.
14. The PR description must clearly distinguish:
   - bugs fixed,
   - methodological limitations,
   - tests added,
   - anything intentionally not changed.

---

# 2. Priority Classification

## P0 — Must Fix

These issues can create incorrect signals, misleading confidence, silent failures, or unsafe operational behavior.

- API exposes non-calibrated confidence.
- ORB feature lookahead leakage.
- Calibration mixes horizons/model versions.
- Calibration bin lookup can select the wrong bin.
- Data freshness/source is not explicit.
- Event-feed failure can look like "no events".
- Scheduler/data-fetcher have inconsistent market-open logic.
- FastAPI runtime dependencies/documentation mismatch.
- Public refresh endpoint lacks basic protection against uncontrolled expensive execution.
- Scanner silently discards rejection/failure reasons.

## P1 — Strongly Recommended

- Walk-forward validation.
- Independent validation/test separation.
- Better trading/backtest execution assumptions.
- Stronger edge/live-worthiness gates.
- Model/feature/calibration versioning.
- Risk-based position sizing.
- Event-aware scanner.
- Structured scheduler errors/results.
- API caching and refresh separation.
- Better exchange-calendar handling.

## P2 — Engineering Quality

- Lint/type checking.
- More adversarial tests.
- Better observability.
- SSE/API naming and semantics.
- Cleanup generated artifacts.
- Documentation consistency.

---

# 3. P0 Fixes

## 3.1 API must not expose uncalibrated confidence

### Problem

Inspect `api.py` and related signal serialization.

The predictor architecture distinguishes raw/risk-adjusted confidence from calibrated confidence, but API responses currently expose `risk_adjusted_confidence` as `"confidence"`.

This can cause consumers to interpret an uncalibrated score as a calibrated probability.

### Required behavior

The API must clearly distinguish:

- `raw_confidence`
- `risk_adjusted_confidence`
- `calibrated_confidence`
- `calibration_status`

If calibration is unavailable/invalid/not passed, do **not** label any raw score simply as `confidence`.

Prefer a response structure such as:

```json
{
  "confidence": null,
  "raw_confidence": 0.72,
  "risk_adjusted_confidence": 0.68,
  "calibrated_confidence": null,
  "calibration_status": "UNAVAILABLE"
}
```

If the project intentionally requires a single confidence field, it must only contain a calibrated value when calibration is valid.

### Tests

Add API tests proving:

1. uncalibrated signals do not expose risk-adjusted confidence as calibrated confidence;
2. calibrated signals expose calibrated confidence;
3. calibration failure is represented explicitly.

---

## 3.2 Fix ORB lookahead leakage

### Problem

Inspect `feature_engineer.py`.

Opening Range Breakout currently derives the opening-range high/low from the first N bars and can attach the resulting range information to bars inside the same opening window.

A bar inside the opening range must not know the future high/low that occurs later in that same range.

### Required behavior

If the opening range consists of the first `orb_bar_count` bars:

- Bars within the opening range must have no actionable ORB breakout feature.
- The opening range high/low becomes available only after the opening-range window is complete.
- The first eligible breakout observation must occur after the opening range has closed.
- Preserve the existing one-bar feature lag where required by the project's no-lookahead convention.

Example:

```text
09:15 — opening range
09:20 — opening range
09:25 — opening range closes

09:25/09:30 onward — ORB levels may be known, depending on bar semantics
```

Use the repository's actual interval/bar semantics rather than blindly applying these timestamps.

### Tests

Create a regression test that constructs synthetic OHLC data where the later opening-range bar contains an extreme high/low.

Assert that an earlier bar's ORB feature cannot change when that future opening-range bar is modified.

This is the key invariant:

> Changing future prices must not change a feature available at an earlier timestamp.

---

## 3.3 Separate calibration by horizon and model version

### Problem

Inspect `history_manager.py`, calibration code, model persistence, and prediction history.

Calibration data can currently be gathered by symbol without adequately separating prediction horizon and model identity.

A 5-minute prediction must not calibrate a 1-day prediction, and predictions generated by different model versions must not be mixed.

### Required behavior

Every persisted prediction used for calibration should carry enough metadata to identify at minimum:

```text
symbol
horizon
model_version
feature_version
prediction_timestamp
confidence
outcome/correctness
```

Calibration datasets must be filtered by:

```text
symbol + horizon + model_version
```

Feature version should also be included where feature changes can alter probability behavior.

### Backward compatibility

Existing historical rows without version metadata must not be silently mixed into new calibration datasets.

Possible safe approaches:

- mark them as `LEGACY/UNKNOWN` and exclude them from calibration; or
- migrate them explicitly if a deterministic version can be established.

Do not fabricate model versions.

### Tests

Add tests proving:

- different horizons do not share calibration data;
- different model versions do not share calibration data;
- legacy/unknown rows are excluded safely.

---

## 3.4 Fix calibration bin indexing

### Problem

Inspect `runtime_validator.py` and calibration structures.

Code that computes a numeric bin index and then indexes a list containing only non-empty calibration bins can map a confidence value to the wrong bin.

For example, if early bins are empty, list position does not equal numerical bin position.

### Required behavior

Calibration bins must retain explicit boundaries:

```text
lower_bound
upper_bound
sample_count
accuracy
calibrated_confidence
```

Lookup must use the actual confidence range, not the position of a compressed list.

### Tests

Construct calibration data with empty gaps between bins and prove that confidence values are assigned to the correct numerical bin.

Also test:

- confidence exactly 0;
- confidence exactly 1;
- confidence at bin boundaries;
- confidence slightly below/above boundaries.

---

## 3.5 Make data freshness/source an explicit state

### Problem

`DataFetcher` can fall back to cached data after a live-fetch failure.

Some downstream callers may not distinguish:

```text
LIVE
CACHE
STALE
UNAVAILABLE
```

This is unsafe because a dataframe alone does not communicate data provenance.

### Required behavior

Introduce an explicit data result/status object or equivalent metadata.

For example:

```python
@dataclass
class MarketDataResult:
    data: pd.DataFrame | None
    source: str
    fetched_at: datetime | None
    latest_bar_at: datetime | None
    is_stale: bool
    status: str
    error: str | None
```

Do not force a large architectural rewrite if a smaller compatible design is cleaner.

At minimum, every live-signal path must be able to distinguish:

```text
LIVE
CACHED_FRESH
CACHED_STALE
UNAVAILABLE
```

### Required invariant

A stale cache must never silently become an apparently live signal.

### Tests

Test:

1. live fetch succeeds;
2. live fetch fails and fresh cache is used;
3. live fetch fails and stale cache is returned;
4. stale data reaches predictor;
5. predictor correctly suppresses live trading/actionable output;
6. data-source/status remains visible to the caller.

---

## 3.6 Unify market-open/holiday logic

### Problem

`DataFetcher` and `Scheduler` have separate market-open logic.

One path may use weekday/time while another contains an NSE holiday list.

This can cause inconsistent decisions.

### Required behavior

Create one shared market-calendar abstraction.

It should answer:

```python
is_trading_day(date)
is_market_open(datetime)
next_market_open(datetime)
next_market_close(datetime)
```

Use timezone-aware datetimes.

Do not duplicate holiday logic across modules.

### Holiday data

Avoid a single hard-coded annual holiday list that silently becomes obsolete.

Use an appropriate exchange-calendar dependency if compatible with the educational project's dependency philosophy. If not, isolate the calendar data behind a clearly documented provider/configuration layer.

### Tests

Include:

- normal trading day;
- weekend;
- known exchange holiday;
- pre-open;
- market hours;
- post-close;
- timezone boundary;
- year transition.

---

## 3.7 Fix FastAPI dependency/documentation mismatch

### Problem

The implementation uses FastAPI, while documentation/dependency declarations may refer to Flask or omit FastAPI/Uvicorn.

### Required behavior

Verify:

- `requirements.txt` includes required runtime dependencies;
- README identifies FastAPI, not Flask;
- documented startup command actually works in a clean environment.

Typical command:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Use the repository's actual app module/name if different.

### Tests

Create a clean virtual environment and verify:

```bash
pip install -r requirements.txt
python -c "import fastapi"
python -c "import uvicorn"
```

Then perform a basic application import/startup test.

---

## 3.8 Protect expensive refresh operations

### Problem

A refresh endpoint can trigger data fetching, feature processing, model/backtest work, and persistence.

If publicly exposed without authentication/rate limiting, repeated requests can exhaust resources.

### Required behavior

At minimum:

- separate read from refresh;
- require an explicit refresh operation;
- prevent concurrent refreshes for the same symbol;
- enforce a cooldown/debounce;
- add reasonable request timeouts;
- return `429` or equivalent when a refresh is already running/cooling down.

If the project is intended only for local educational use, document that clearly and still avoid accidental unbounded execution.

Do not invent an authentication system larger than needed.

### Tests

Test repeated/concurrent refresh requests.

---

## 3.9 Scanner must preserve rejection/failure reasons

### Problem

Scanner currently focuses on actionable opportunities and may silently discard:

- stale data;
- missing models;
- calibration failures;
- safety-gate failures;
- event/data failures;
- HOLD signals.

### Required behavior

Return a structured scan summary.

Example:

```json
{
  "scanned": 50,
  "actionable": 5,
  "hold": 10,
  "stale": 12,
  "missing_model": 8,
  "calibration_unavailable": 5,
  "data_error": 3,
  "risk_suppressed": 7
}
```

Use categories that match the actual codebase.

Do not double-count a single stock unless the design explicitly documents multi-category accounting.

### Tests

Synthetic scanner tests should prove that rejected symbols remain observable in the summary.

---

## 3.10 Event-feed failures must not become "no events"

### Problem

An unavailable event/news source can currently produce an empty event result, which can be interpreted as "no events".

### Required behavior

Differentiate:

```text
EVENTS_AVAILABLE
NO_EVENTS
EVENT_SOURCE_UNAVAILABLE
EVENT_SOURCE_PARTIAL
```

Never convert an exception into an empty successful event set.

If event data is required for a safety gate, an unavailable event source should fail closed.

### Tests

Mock event provider failures and assert that the resulting status is unavailable/partial rather than "no events".

---

# 4. P1 Improvements

## 4.1 Implement walk-forward validation

Replace reliance on a single train/test split with chronological walk-forward evaluation where practical.

Example:

```text
Train window 1 -> Test window 1
Train window 2 -> Test window 2
Train window 3 -> Test window 3
...
```

Report:

- accuracy;
- precision/recall where relevant;
- calibration;
- cumulative return;
- alpha vs benchmark;
- drawdown;
- trade count;
- Sharpe/Sortino if meaningful;
- performance by horizon.

Do not use the final evaluation set to tune thresholds.

---

## 4.2 Separate training, validation, and final test data

Recommended flow:

```text
TRAIN
  ↓
VALIDATION / threshold selection
  ↓
CALIBRATION
  ↓
UNTOUCHED FINAL TEST
```

The final test set should not be used to tune the live-worthiness decision.

Document exactly which data is used for which purpose.

---

## 4.3 Strengthen the live-worthiness/edge gate

Do not consider:

```text
strategy_return > benchmark_return
```

alone sufficient.

Consider requiring a combination such as:

```text
minimum trade count
positive/acceptable alpha
acceptable maximum drawdown
acceptable risk-adjusted return
walk-forward consistency
calibration validity
data quality
model health
```

Do not invent arbitrary thresholds without documenting them.

If thresholds are educational placeholders, label them as such.

---

## 4.4 Improve backtest execution assumptions

Clearly distinguish the existing simulation from a true execution backtest.

Where feasible, add:

- entry price assumptions;
- exit price assumptions;
- spread;
- slippage;
- fees;
- position overlap;
- capital constraints;
- stop/target execution semantics;
- gap handling;
- turnover.

If full realism is out of scope, document the simplified model instead of overstating accuracy.

---

## 4.5 Add model and feature versioning

Persist:

```text
model_version
feature_version
training_data_end
training_timestamp
horizon
```

with model artifacts and prediction records.

A model version should change whenever material model/features/training logic changes.

Use deterministic version identifiers where possible, e.g.:

```text
<model-type>-<feature-version>-<training-date>-<commit>
```

---

## 4.6 Make calibration model-specific

Calibration should be tied to the exact probability-generating model/version and horizon.

If a new model is trained, old calibration must not automatically apply.

---

## 4.7 Improve position sizing

Current DCA allocation should not be described as full risk-based position sizing.

Where practical, incorporate:

```text
entry
stop distance
ATR/volatility
maximum portfolio risk
maximum capital allocation
```

Calculate maximum possible loss if every tranche fills.

The final plan should expose:

```text
planned_capital
max_loss
risk_pct
stop_price
```

Do not silently increase risk as price moves against the position.

---

## 4.8 Make scanner event-aware

Use the same relevant event/risk context as the normal predictor.

Do not allow the scanner to rank an opportunity using a materially different safety context from the detailed signal path.

---

## 4.9 Avoid repeated data fetching

Audit scanner/predictor multi-horizon execution.

Prefer:

```text
fetch once
  ↓
validate once
  ↓
reuse data
  ↓
generate all horizons
```

rather than repeatedly fetching the same underlying data.

Add caching with clear freshness semantics.

---

## 4.10 Structured scheduler results

Replace broad:

```python
except Exception:
    return None
```

where appropriate with structured status values.

Example:

```python
CycleResult(
    success=False,
    status="DATA_UNAVAILABLE",
    symbol=symbol,
    signal=None,
    error="..."
)
```

Keep exception details in logs while returning safe, useful status to callers.

---

# 5. P2 Engineering Cleanup

## 5.1 Repository hygiene

Remove generated/runtime artifacts from git where appropriate:

```text
__pycache__/
*.pyc
*.log
*.sqlite3
.env
cache directories
generated model artifacts
```

Do not delete intentionally tracked educational fixtures.

Update `.gitignore` accordingly.

Investigate any unexplained files/directories such as an accidental `=` artifact.

---

## 5.2 Linting

Add or run:

```bash
ruff check .
ruff format --check .
```

Fix obvious issues.

Remove duplicate imports such as duplicate configuration constants.

Do not perform unrelated mass formatting unless necessary.

---

## 5.3 Testing

Use `pytest` if compatible with the project.

Add tests for:

### Data

- stale cache;
- future timestamps;
- duplicate timestamps;
- out-of-order timestamps;
- NaNs;
- missing bars;
- timezone mismatch;
- market holiday.

### Feature leakage

- future-bar mutation invariance;
- ORB opening-range leakage;
- shifted feature invariants.

### ML

- single-class training;
- empty datasets;
- missing model;
- model-version mismatch;
- calibration boundaries;
- calibration gaps.

### Trading

- zero ATR;
- invalid stop;
- invalid target;
- extreme volatility;
- all DCA tranches filled;
- maximum-loss calculation.

### API

- invalid symbol;
- unavailable data;
- stale data;
- unavailable calibration;
- refresh cooldown;
- concurrent refresh;
- correct confidence fields.

---

# 6. Documentation Updates

Update README so that it accurately describes the current implementation.

Specifically verify:

- FastAPI vs Flask;
- API startup command;
- port;
- market calendar behavior;
- event-data limitations;
- cache behavior;
- calibration semantics;
- model versioning;
- backtest limitations;
- scanner behavior;
- whether confidence is calibrated;
- whether live trading is supported.

Do not claim:

```text
research-grade
production-ready
profitable
accurate
live-worthy
```

unless the implementation and validation genuinely justify those claims.

Prefer precise wording such as:

> Educational research framework for experimenting with NIFTY-50 prediction, validation, calibration, and risk-gated signal generation.

---

# 7. Required Safety Invariants

The following invariants must be encoded as tests wherever practical.

## Invariant A — No future information

For any timestamp `t`:

> Changing market/event data after `t` must not change features or predictions available at `t`.

---

## Invariant B — Stale data cannot masquerade as live

```text
LIVE != CACHED_FRESH != CACHED_STALE != UNAVAILABLE
```

Each state must remain distinguishable.

---

## Invariant C — Uncalibrated probability cannot be presented as calibrated

If calibration is invalid/unavailable:

```text
calibrated_confidence = null
calibration_status != VALID
```

---

## Invariant D — Calibration isolation

Calibration datasets must not mix:

```text
different horizons
different model versions
incompatible feature versions
```

---

## Invariant E — Event failure != no event

```text
source failure → EVENT_SOURCE_UNAVAILABLE
```

not:

```text
source failure → []
```

---

## Invariant F — Scanner failures remain observable

A scan must be able to explain why a symbol was excluded.

---

## Invariant G — Refresh cannot run unboundedly

Repeated refresh requests must be controlled.

---

# 8. Suggested Implementation Sequence

Use this order:

```text
1. Run baseline tests
2. Inspect repository and current branch
3. Fix API confidence semantics
4. Fix ORB leakage
5. Fix calibration structure/bin lookup
6. Add model/horizon/version isolation
7. Fix data freshness/status propagation
8. Unify market calendar
9. Fix event failure semantics
10. Fix scanner observability
11. Fix API dependencies/docs
12. Add refresh protection
13. Add regression tests
14. Run full test suite
15. Run lint/static checks
16. Update README
17. Inspect git diff
18. Commit
19. Push branch
20. Raise PR
```

Do not spend most of the task rewriting architecture before fixing the P0 issues.

---

# 9. Acceptance Criteria

The task is complete only when all of the following are true:

- [ ] Existing tests pass, or any pre-existing failures are documented.
- [ ] New regression tests cover all P0 bugs.
- [ ] ORB no-lookahead regression test passes.
- [ ] Calibration bins use explicit boundaries.
- [ ] Calibration is isolated by horizon/model version.
- [ ] API never labels uncalibrated confidence as calibrated confidence.
- [ ] Data provenance/freshness is explicit.
- [ ] Market-open/holiday logic has one authoritative implementation.
- [ ] Event source failures are distinguishable from empty event results.
- [ ] Scanner exposes meaningful rejection/failure counts.
- [ ] Refresh operations cannot be triggered without reasonable resource protection.
- [ ] FastAPI/Uvicorn dependencies and documentation are correct.
- [ ] Generated artifacts are not accidentally committed.
- [ ] README matches actual implementation.
- [ ] `ruff check .` passes, or remaining issues are documented.
- [ ] Full test suite passes.
- [ ] `git diff` has been reviewed.
- [ ] No secrets are present in the diff.
- [ ] A pull request is raised against `main`.

---

# 10. PR Requirements

## Branch

Use a descriptive branch name, for example:

```text
fix/ml-safety-and-runtime-audit
```

## Commit style

Prefer logical commits such as:

```text
fix: prevent ORB lookahead leakage
fix: isolate calibration by model and horizon
fix: expose explicit data freshness state
fix: correct API confidence semantics
test: add regression coverage for safety invariants
docs: align API and research limitations
chore: clean repository artifacts and lint issues
```

Squash commits if repository conventions require it.

## Pull Request title

Suggested:

```text
Fix ML leakage, calibration, data freshness, and runtime safety issues
```

## Pull Request body

The PR should include:

### Summary

What was fixed and why.

### P0 fixes

List each P0 issue addressed.

### Tests

List commands executed and results.

Example:

```text
pytest -q
ruff check .
ruff format --check .
```

### Methodology

Explain any changes affecting:

- feature leakage;
- calibration;
- backtesting;
- model validation.

### Remaining limitations

Be honest about anything not fully solved.

### Breaking changes

State explicitly whether API response fields or behavior changed.

---

# 11. Important: Do Not Overreach

This repository is educational.

Do not turn this task into an uncontrolled rewrite.

Do not:

- replace the entire ML stack;
- add a database server unless necessary;
- introduce cloud infrastructure;
- add paid APIs;
- add brokerage execution;
- enable real-money trading;
- remove safety gates;
- fabricate performance improvements;
- tune thresholds against the final test set;
- delete useful educational code merely because it is not production-grade.

Fix correctness first.

---

# 12. Final Agent Report

Before finishing, provide the PR with a concise final report containing:

```text
Repository:
Branch:
PR:

Baseline tests:
Final tests:
Lint:

P0 fixed:
- ...
- ...

P1 fixed:
- ...
- ...

P2 fixed:
- ...
- ...

Tests added:
- ...

Documentation updated:
- ...

Known remaining limitations:
- ...
```

The agent must not claim an issue is fixed unless there is either:

1. a code change that addresses it, and/or
2. a regression test proving the required behavior.

