# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 3 Engineering Maturity enhancements.
- `pyproject.toml` with tool configurations (ruff, black, mypy, pytest, bandit).
- `requirements-dev.txt` for dev tools.
- GitHub Actions CI workflow for linting, testing, and security scanning.
- This `CHANGELOG.md` file.

## [0.3.0] - 2026-09-12

### Added
- Phase 2: Data Reliability enhancements.
- Versioned NIFTY 50 universe snapshot management with checksums.
- Versioned sector mapping metadata.
- Official NSE equity holiday calendar for 2026/2027 in versioned CSVs.
- Explicit corporate action price adjustment strategy.
- Abstract `MarketDataProvider` interface with implementations (YFinance, LocalCache, TestFixture).
- Data quality rules for monotonic indices, deduplication, timezone alignment, and outlier rejection.

## [0.2.0] - 2026-09-12

### Added
- Phase 1: Quant Correctness enhancements.
- Purged walk-forward validation with expanding/rolling modes.
- Event-driven discrete bar Execution Simulator with 4 cost tiers.
- Expanded ML evaluation metrics (Balanced Accuracy, MCC, F1, Brier, LogLoss, ECE).
- Out-of-sample calibration isolation and freshness tracking.
- Horizon-specific causality invariant tests.
- Immutable versioned model artifacts with atomic pointers.

## [0.1.0] - 2026-09-12

### Added
- Phase 0: Stabilization enhancements.
- API authentication and role-based access control.
- Strict CORS allowlist with credentials preservation.
- Purged train/test split boundary to prevent forward-label leakage.
- Global risk monitor fail-closed state.
- Calibration query contract and feature versioning.
- Model artifact lineage and provenance metadata.
