# Bug Report — `alkame-nifty-50-educational`

Repo: https://github.com/Alkameinc/alkame-nifty-50-educational
Reviewed: `config.py`, `data_fetcher.py`, `feature_engineer.py`, `model_trainer.py`,
`ensemble_manager.py`, `predictor.py`, `runtime_validator.py`, `event_classifier.py`,
`requirements.txt`, `README.md`.

**Not yet line-audited in this pass** (recommend a follow-up review before relying on
them): `macro_calendar.py`, `corporate_events_fetcher.py`, `news_sentiment_fetcher.py`,
`global_risk_monitor.py`, `history_manager.py`, `human_insight_manager.py`,
`backtester.py`, `scheduler.py`, `app.py`. Several of these are imported by files below
and may inherit the same import-chain failures noted in Bug #1.

---

## 🔴 Critical / Blocking Bugs

### 1. Two Python modules are imported but do not exist in the repo

- `feature_engineer.py` has: `from reference_level_engine import ReferenceLevelDeltas`
- `event_classifier.py` has: `from health_monitor import registry as health_registry`

Neither `reference_level_engine.py` nor `health_monitor.py` exists anywhere in the
repository, and neither is a PyPI package in `requirements.txt`. This means:

```
python feature_engineer.py   # ModuleNotFoundError: No module named 'reference_level_engine'
python event_classifier.py   # ModuleNotFoundError: No module named 'health_monitor'
```

Because almost every downstream file imports `feature_engineer.py` (directly or via
`model_trainer.py` → `ensemble_manager.py` → `predictor.py`), **this single issue
breaks the entire pipeline from Phase 8 (feature engineering) onward** — you cannot
even run the "Verification" steps in the README past `event_classifier.py`.

**Fix needed:** either restore the missing `reference_level_engine.py` and
`health_monitor.py` source files (they were clearly written at some point — the code
calls very specific methods like `health_registry.report(name, ok=True/False,
detail=..., error=...)` and expects `ReferenceLevelDeltas` to have fields
`pct_from_moving_average`, `pct_from_support_band`, `pct_from_resistance_band`,
`pct_from_user_avg_cost`), or remove/stub these imports if the features were
intentionally descoped.

### 2. `config.py` is missing several constants that other modules require

The following names are imported from `config` but are **not defined** in
`config.py` as it currently exists in the repo:

| Missing constant | Required by |
|---|---|
| `HORIZON_INTRADAY` | `feature_engineer.py`, `model_trainer.py`, `event_classifier.py` |
| `HORIZON_30D` | `event_classifier.py` |
| `HORIZON_CONFIG` (dict, expects `["horizon_bars"]`, `["deadband_pct_default"]`, `["min_training_samples"]` per horizon) | `model_trainer.py` |
| `HORIZON_TO_MA_PERIOD` | `feature_engineer.py` |
| `HORIZON_TO_SR_METHOD` | `feature_engineer.py` |
| `EVENT_IMPACT_HORIZON` (dict, keyed by event type) | `event_classifier.py` |

Importing any of these modules currently raises `ImportError: cannot import name
'HORIZON_INTRADAY' from 'config'` (this would surface even if Bug #1 above is fixed
first).

**Fix needed:** add these constants/dicts to `config.py`. Based on how they're
consumed:
- `HORIZON_INTRADAY = "intraday"`, plus string keys for `"30D"`, `"3M"`, `"6M"`, `"1Y"`
  (referenced via a `lookback_map` in `feature_engineer.py`).
- `HORIZON_CONFIG` needs one entry per horizon with `horizon_bars`,
  `deadband_pct_default`, and `min_training_samples`.
- `HORIZON_TO_MA_PERIOD` / `HORIZON_TO_SR_METHOD` need one entry per non-intraday
  horizon (used with `.get(horizon, default)`, so missing entries degrade rather than
  crash, but intraday's own values are still undefined without `HORIZON_INTRADAY`).
- `EVENT_IMPACT_HORIZON` needs entries for at least: `RBI_POLICY`, `UNION_BUDGET`,
  `ELECTION`, `GEOPOLITICAL`, `CORPORATE_ANNOUNCEMENT`, `NEWS_HEADLINE`,
  `CLASSIFICATION_ERROR` (all read via `.get(..., default)` so it degrades to the
  default rather than crashing — but the default itself, `HORIZON_30D`/
  `HORIZON_INTRADAY`, doesn't exist either, per the row above).

---

## 🟠 Logic Bugs

### 3. `event_classifier.py` — unreachable code means success telemetry never fires

In **all three** classification methods (`classify_macro_event`,
`classify_corporate_event`, `classify_news_event`), the pattern is:

```python
return Event(
    ...
    raw=None,
)
health_registry.report("event_classifier", ok=True)
return evt
```

The `return Event(...)` statement exits the function immediately. The
`health_registry.report(...)` call and the second `return evt` are dead code that
can never execute — and `evt` is never even assigned, so if that line were ever
reached it would raise `NameError: name 'evt' is not defined`.

**Effect:** every *successful* classification silently fails to report health/ok
status (only the `except` branches report `ok=False`), so any monitoring/dashboard
built on `health_registry` will only ever see failures, never successes, for this
module.

**Fix:** assign the result to `evt` first, then report, then return it:

```python
evt = Event(...)
health_registry.report("event_classifier", ok=True)
return evt
```

### 4. `runtime_validator.py` — `get_calibrated_confidence` can return the wrong bin's accuracy

```python
bin_index = min(int(raw_confidence / bin_width), len(calibration_result.bins) - 1)
return calibration_result.bins[bin_index].empirical_accuracy
```

`calibration_result.bins` is built via `df.groupby("bin", observed=True)`, which
**only includes bins that actually contain samples**. If, say, confidence bin #4 (of
10) happens to have zero historical predictions, `bins` will have only 9 entries and
bin #5's data will shift into what `bins[4]` now points to. The positional index
computed from `raw_confidence / bin_width` assumes a dense, gap-free 0–9 bin array,
so once any bin in the middle of the range is empty, every confidence value at or
above that empty bin will be mapped to the *wrong* calibration bin, i.e. the wrong
empirical accuracy is shown/used from that point onward.

**Fix:** store bins in a fixed-size array indexed by their bin number (fill missing
bins with `None`/interpolate), or look bins up by their actual `pd.Interval` range
instead of positional order.

### 5. `model_trainer.py` — dead/no-op safety assertion in `prepare_dataset`

```python
feature_columns = [c for c in engineered.columns if c.endswith(ML_SAFE_SUFFIX)]
...
raw_leak = [c for c in feature_columns if not c.endswith(ML_SAFE_SUFFIX)]
assert not raw_leak, f"Non-lagged column(s) detected in feature set: {raw_leak}"
```

`feature_columns` was just filtered to *only* contain columns ending in
`ML_SAFE_SUFFIX` on the line above, so `raw_leak` is guaranteed to always be an empty
list — this assertion can never fire and provides no actual protection. It reads as
though it was meant to check something else (e.g. that none of the *raw* engineered
columns leaked into `X` some other way), but as written it's dead code that gives a
false sense of a safety net against lookahead/leakage.

**Fix:** clarify intent — if the goal is to guard against raw columns leaking in,
the check needs to inspect a different source list (e.g. everything selected into
`X`), not a value that was already filtered.

---

## 🟡 Design Notes / Minor Issues Worth Reviewing

### 6. `ensemble_manager.py` — soft-vote probabilities aren't renormalized after class padding

`_reindexed_proba()` zero-fills probability columns for any class a given model never
saw in training. When probability matrices are averaged across models
(`np.mean(list(proba_matrices.values()), axis=0)`), rows no longer necessarily sum to
1.0 if one model was missing a class. This won't break `argmax`-based predictions,
but the reported `confidence` value (`avg_proba[row_i, pred_idx[row_i]]`) will be
understated whenever any constituent model didn't see all label classes during
training — this is most likely to happen on small/imbalanced per-stock datasets.

### 7. Documented, not-yet-fixed limitations (from the README itself)

These are self-reported by the project and not new findings, but are worth tracking
alongside the above since they affect correctness:

- `scheduler.is_market_open()` checks weekday + time window only — **Indian exchange
  holidays are not modeled**, so the scheduler will treat NSE holidays as open trading
  days.
- `NIFTY50_SYMBOLS` and `SECTOR_MAP` in `config.py` need verification against NSE's
  next semi-annual index review — no code enforces freshness.
- Festive-window dates (used for demand-sensitive sector tagging) are approximate and
  need yearly verification (lunar calendar drift).
- `corporate_events_fetcher.py` is documented as unreliable from cloud/VPN IPs due to
  NSE bot protection — this isn't a code bug, but any CI/automated run from a cloud
  runner should expect this fetch to fail.

---

## Suggested Fix Order

1. Restore/stub `reference_level_engine.py` and `health_monitor.py` (Bug #1) — nothing
   downstream can even be imported until this is fixed.
2. Add the missing horizon-related constants and `EVENT_IMPACT_HORIZON` to
   `config.py` (Bug #2).
3. Re-run each module's built-in `python <file>.py` self-test (they're already written
   and fairly thorough) to confirm the import chain is clean end-to-end.
4. Fix the `event_classifier.py` unreachable-return bug (#3) and the calibration
   bin-lookup bug (#4) — both are silent correctness/observability issues, not crashes,
   so they won't be caught by the self-tests unless a test specifically checks bin
   sparsity or `health_registry` call counts.
5. Clarify/remove the dead assertion in `model_trainer.py` (#5).
6. Do a follow-up pass on the files not yet audited in this report, since several of
   them (`scheduler.py`, `app.py`, `backtester.py`, `history_manager.py`) likely import
   the same broken chain and haven't been checked line-by-line yet.
