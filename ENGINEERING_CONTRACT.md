# Engineering Contract — Alkame Nifty 50 Educational

This contract governs all future changes to this repository.
A change that violates any rule below MUST NOT be merged.

## Rules

1. No feature may use future information.
2. Any feature change requires a leakage test.
3. Any label change requires a model-version bump.
4. Any config change affecting predictions requires config-hash update.
5. Any model artifact must contain complete lineage metadata.
6. No production secret may have a source-code fallback.
7. Authentication disabled ≠ ADMIN.
8. Risk state must be shared and persistent across workers.
9. Stale data must never silently become LIVE data.
10. Missing quantitative values must use explicit unavailable states (never synthetic).
11. Backtest and live paths must share canonical feature/prediction logic.
12. Historical backtests must use historical universe membership.
13. Calibration and edge validation must use independent datasets.
14. Scheduler process health ≠ pipeline health.
15. Every production endpoint must have an explicit authorization class.
16. Every bug fix requires a regression test.
17. Do not optimize historical performance by weakening correctness gates.
18. Do not remove a safety test without replacing it with an equivalent or stronger test.
