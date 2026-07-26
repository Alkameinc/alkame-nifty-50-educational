# Data Analyst Notebooks — Alkame-Nifty50

> **Internship Project — DBERT Labs**
> This project was built as part of the DBERT Labs Internship Program.
> Program: https://dbert.online · Internships: https://internship.dbert.online
> Domain: Data Analyst
> Intern: [Your Name] · Batch start: [Month 2026]
> Base project: Alkame-Nifty50 (Alkameinc/alkame-nifty-50-educational)

## What's here

- `seed_sample_data.py` — generates deterministic synthetic prediction/event history so every
  notebook below is runnable immediately, without waiting for `scheduler.py` to accumulate real
  history. Run once: `python seed_sample_data.py`. Writes to `sample_history.sqlite3` (gitignored).
- `db_utils.py` — shared helper for safely loading either the sample database or the real one
  (`config.py`'s `DB_PATH`) into a pandas DataFrame.
- `notebooks/01_calibration_reliability.ipynb` — per-stock and per-sector reliability diagrams,
  using `runtime_validator.py`'s real `compute_calibration()`.
- `notebooks/02_backtest_sector_comparison.ipynb` — real backtests (via `backtester.py`) across
  NIFTY50 stocks, compared by sector.
- `notebooks/03_event_impact_study.ipynb` — whether event sentiment tagging (`event_classifier.py`)
  actually correlates with real forward price moves.
- `figures/` — exported chart PNGs referenced above.

## How to run

1. `pip install -r requirements-data-analyst.txt` (from inside `data_analyst/`, with the repo's
   main virtual environment already activated).
2. `python seed_sample_data.py` (from inside `data_analyst/`).
3. `jupyter lab` (from the repo root), then open any notebook in `notebooks/` and Run All Cells.

## Key findings (fill this in with your real results)

- Calibration: *(which sectors are/aren't well-calibrated, and by how much)*
- Backtest edge: *(which sectors/stocks currently show a real, cost-adjusted edge; which are
  currently not live-worthy per `runtime_validator.py`'s combined gate, and why)*
- Event impact: *(does sentiment-tagged event data correlate with real forward returns; sample
  size caveats)*

## Honest limitations of this analysis

- *(e.g. "backtest results in Notebook 02 are based on N days of yfinance 5-minute bars, per
  `config.py`'s BAR_HISTORY_PERIOD — a longer sample would give more statistical confidence")*
- *(e.g. "Notebook 03's event-impact findings are based on synthetic seeded data / a small number
  of real events — treat any correlation as preliminary until re-confirmed on more real history")*