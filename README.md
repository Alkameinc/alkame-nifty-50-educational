# Alkame-Nifty50 — Educational Repository

> **DBERT Labs Internship 2026**
> **Domain:** Data Analyst
> **Intern:** Ankur Sen
> **Program Details:** https://dbert.online · https://internship.dbert.online
> This repository is a component of the DBERT Labs structural training track. All modifications, upgrade branches, and submitted pull requests are developed under professional educational guidelines.

**NIFTY 50 Intraday Market Intelligence Engine**
*Human-in-the-loop · Calibration-gated · Event-aware · Scalping-ready · API-first*

> ⚠️ **Educational Repository** — This project is shared for learning purposes only. It does not constitute financial advice. Read the [Legal & Regulatory Notice](#legal--regulatory-notice) section before any use.

---

## Overview

Alkame-Nifty50 is an open-source, research-grade intraday signal engine for the NIFTY 50 index. It is built around four core design principles:

- **Risk-first**: `HOLD` with a reason is a valid — and often correct — output.
- **Calibration before confidence**: No signal confidence is ever surfaced until runtime calibration checks pass.
- **Human-in-the-loop**: Trader overrides, notes, and feedback are first-class citizens of the pipeline.
- **Modular and API-first**: Every major component is independently testable and externally accessible via a REST API layer.

The full methodology, regulatory rationale, and business charter live in [`ALKAME_NIFTY50_CHARTER.md`](./ALKAME_NIFTY50_CHARTER.md). Read it first — it explains *why* the system is built the way it is.

This project is built and maintained by interns and mentors of the **[DBERT AI Internship Program](https://internship.dbert.online)** — see the [DBERT Internship Program](#dbert-internship-program) section below to get involved.

---

## What's New — Recent Improvements

This release significantly expands the system beyond the original signal pipeline.

### 🆕 REST API Layer (`api.py`)
A Flask-based REST API now exposes core engine capabilities over HTTP. Enables integration with external dashboards, bots, and automation workflows without touching internal Python code directly.

### 🆕 Scalping Engine (`scalping.py`)
Dedicated short-horizon scalping logic for high-frequency intraday opportunities on NIFTY 50 constituents. Works alongside the main prediction pipeline with its own risk guardrails.

### 🆕 Market Scanner (`scanner.py`)
A multi-stock scanner that sweeps all NIFTY 50 constituents on a configurable cadence and surfaces top-ranked opportunities based on ensemble signal strength, volume anomalies, and event context.

### 🆕 Position Planner (`position_planner.py`)
Translates raw BUY/SELL signals into actionable position plans — entry price, target, stop-loss, and position sizing — based on configurable capital and risk parameters.

### 🆕 Narrative Builder (`narrative_builder.py`)
Generates human-readable, plain-English summaries of each signal decision explaining *why* the engine recommends BUY, SELL, or HOLD — covering technical factors, macro events, sentiment, and calibration status.

### 🆕 Reference Level Engine (`reference_level_engine.py`)
Computes and maintains dynamic support/resistance reference levels (weekly pivots, prior-day high/low, VWAP anchors) used by the predictor, position planner, and dashboard.

### 🆕 System Health Monitor (`health_monitor.py`)
Always-on health checks tracking data feed freshness, model staleness, API quota consumption, and pipeline latency. Surfaces warnings before failures cascade into bad signals.

### 🆕 Automation Directory (`automation/`)
Shell and Python scripts for unattended scheduled operation — cron-friendly wrappers around `scheduler.py`, log rotation helpers, and restart-on-failure scripts for always-on Linux/EC2 deployment.

### ⚙️ Improved Scheduler (`scheduler.py`)
Major expansion: now coordinates scanner sweeps, health monitoring, scalping runs, and reference level updates — not just the core prediction loop. Supports graceful shutdown and configurable symbol subsets.

### ⚙️ Enhanced Predictor (`predictor.py`)
Now integrates reference levels and narrative generation directly into the output object. Each prediction result includes structural context and a human-readable explanation alongside the BUY/SELL/HOLD signal.

### ⚙️ Expanded Feature Engineer (`feature_engineer.py`)
Additional technical indicators including volume-weighted signals and multi-timeframe momentum features. Strict no-lookahead discipline maintained across all new additions.

### ⚙️ Richer History Manager (`history_manager.py`)
Expanded schema captures position plan data, narrative summaries, and health snapshots alongside predictions — improving post-hoc analysis and outcome tracking significantly.

---

## Project Structure

| File / Directory | Purpose |
|---|---|
| `config.py` | Central configuration — tickers, sectors, thresholds, paths |
| `macro_calendar.py` | RBI/Budget/election/festive calendar (human-maintained CSV) |
| `data_fetcher.py` | yfinance OHLCV data with retry + cache fallback |
| `corporate_events_fetcher.py` | NSE corporate announcements, board meetings, block/bulk deals |
| `news_sentiment_fetcher.py` | marketaux + Google News RSS sentiment |
| `event_classifier.py` | Unifies all events, tags scope (MARKET / SECTOR / STOCK) |
| `global_risk_monitor.py` | Always-on composite risk score + human-gated toggle |
| `reference_level_engine.py` | Dynamic support/resistance levels, VWAP anchors, pivot points |
| `feature_engineer.py` | Technical indicators with multi-timeframe momentum; strict no-lookahead |
| `model_trainer.py` | Per-stock direction classifier, time-based split |
| `ensemble_manager.py` | Multi-model soft-voting ensemble with agreement scoring |
| `runtime_validator.py` | Calibration + edge-vs-NIFTY checks — the safety gate |
| `predictor.py` | Combines everything into one final BUY / SELL / HOLD signal with narrative |
| `position_planner.py` | Converts signals into entry/target/stop-loss/position-size plans |
| `scalping.py` | Short-horizon scalping signals for intraday high-frequency opportunities |
| `scanner.py` | Multi-stock sweep; ranks NIFTY 50 constituents by signal strength |
| `narrative_builder.py` | Generates plain-English signal explanations for each prediction |
| `human_insight_manager.py` | Trader notes, overrides, feedback (SQLite) |
| `history_manager.py` | Persists predictions, events, plans, narratives; resolves outcomes (SQLite) |
| `health_monitor.py` | Data feed freshness, model staleness, API quota, pipeline latency checks |
| `backtester.py` | Historical replay with slippage/costs, real alpha calculation |
| `scheduler.py` | Orchestrates the full pipeline on a market-hours loop |
| `api.py` | Flask REST API — signals, scanner results, and history over HTTP |
| `app.py` | Streamlit dashboard |
| `train_all.py` | Batch training script for all NIFTY 50 models |
| `automation/` | Cron wrappers, log rotation, restart-on-failure scripts for server deployment |

---

## Architecture at a Glance

```
Market Data (yfinance)          NSE Corporate Events
        │                               │
        ▼                               ▼
  data_fetcher.py          corporate_events_fetcher.py
        │                               │
        └──────────────┬────────────────┘
                       ▼
              event_classifier.py  ◄── macro_calendar.py
                       │                news_sentiment_fetcher.py
                       ▼                global_risk_monitor.py
             reference_level_engine.py
                       │
                       ▼
             feature_engineer.py
                       │
              ┌────────┴────────┐
              ▼                 ▼
        model_trainer.py   ensemble_manager.py
              │                 │
              └────────┬────────┘
                       ▼
              runtime_validator.py  (safety gate)
                       │
                       ▼
                  predictor.py
              ┌────────┴────────┐
              ▼                 ▼
      position_planner.py  narrative_builder.py
              │                 │
              └────────┬────────┘
                       ▼
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       app.py        api.py    history_manager.py
    (dashboard)   (REST API)   (SQLite storage)
                       │
              scanner.py / scalping.py
                       │
              health_monitor.py
                       │
              scheduler.py  ◄── automation/
```

---

## Requirements

- Python 3.10+
- A free [marketaux](https://www.marketaux.com) API key
- *(Optional)* An [IMD API key](https://api.imd.gov.in) for live monsoon data

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Set your API key as an environment variable. **Never hardcode keys into source files.**

```bash
# macOS / Linux
export MARKETAUX_API_KEY="your_key_here"

# Windows (Command Prompt)
set MARKETAUX_API_KEY=your_key_here

# Windows (PowerShell)
$env:MARKETAUX_API_KEY="your_key_here"
```

Without `IMD_API_KEY`, monsoon status falls back to manual entry via `macro_calendar.py`.

---

## Verification (Run Before First Use)

Each module includes a self-test. Run them in order to confirm your setup:

```bash
python config.py
python macro_calendar.py
python data_fetcher.py
python corporate_events_fetcher.py    # Run from a home/office connection — NSE blocks many cloud/VPN IPs
python news_sentiment_fetcher.py
python event_classifier.py
python global_risk_monitor.py
python reference_level_engine.py
python feature_engineer.py
python model_trainer.py
python ensemble_manager.py
python runtime_validator.py
python predictor.py
python position_planner.py
python narrative_builder.py
python scalping.py
python scanner.py
python human_insight_manager.py
python history_manager.py
python health_monitor.py
python backtester.py
python scheduler.py
python api.py                         # Starts the Flask REST API server
python app.py                         # Pure-logic self-test only; see Running section for dashboard
```

Each script prints `STATUS: PASS` or `STATUS: FAIL` at the end.

> **Note:** `data_fetcher.py`, `corporate_events_fetcher.py`, `news_sentiment_fetcher.py`, `global_risk_monitor.py`, `predictor.py`, and `scheduler.py` touch live external services and may take up to a minute or two.

---

## Running

**Terminal 1 — Signal Scheduler**

```bash
python scheduler.py
```

> Wire up `run_forever()` with a real `symbol_data_provider` callable (using `data_fetcher.py`) when ready for continuous operation. The integration point is intentionally left open — your symbol list, refresh cadence, and error-handling preferences for unattended operation are worth deciding deliberately.

**Terminal 2 — REST API Server**

```bash
python api.py
```

> Defaults to `http://localhost:5000`. Endpoints include signal retrieval, scanner results, health status, and prediction history. Refer to inline docstrings in `api.py` for full endpoint documentation. **Add API key middleware before any public or network-exposed deployment.**

**Terminal 3 — Dashboard**

```bash
streamlit run app.py
```

**Automated / Server Deployment**

```bash
# Use scripts in automation/ for unattended cron-based operation
# Example: run every 15 minutes during market hours
# */15 9-16 * * 1-5 /path/to/automation/run_scheduler.sh
```

---

## Known Limitations

- **NIFTY 50 constituent list** and sector map need verification against NSE's next semi-annual index review.
- **Festive-window dates** are approximate (lunar calendar shifts yearly) — verify each year.
- **Exchange holidays** are not yet modeled in `scheduler.is_market_open()` — it currently only checks weekday + time window.
- **`corporate_events_fetcher.py`** may be blocked from cloud/VPN IPs by NSE's bot protection — run from a normal home/office connection.
- **Scalping signals** carry inherently higher noise at very short timeframes; always validate against `runtime_validator.py` output before acting.
- **`api.py`** does not include authentication by default — add API key middleware before any public or network-exposed deployment.

---

## DBERT Internship Program

This repository — including the API layer, scalping engine, market scanner, position planner, narrative builder, and automation scripts — is built and maintained by student interns as part of the **[DBERT AI/ML Internship Program](https://internship.dbert.online)**, a hands-on remote internship for engineers who want real production experience instead of toy projects.

**Why do a DBERT internship?**

DBERT internships give you verifiable, portfolio-ready experience across the exact skill areas used in this project:

- **AI/ML Engineering Internship** — feature engineering, ensemble model training, calibration and backtesting on real market data
- **Full-Stack Development Internship** — Flask REST APIs, Streamlit dashboards, SQLite-backed data pipelines
- **Quantitative & Algorithmic Trading Internship** — signal design, scalping strategies, position planning, and risk modelling
- **DevOps & Cloud Deployment Internship** — AWS EC2, Nginx, cron automation, health monitoring, and CI-style deployment pipelines
- **Open-Source & LLM Internship tracks** — contributing to production-grade, open-source AI systems with real-world mentorship

Every DBERT intern works on live GitHub repositories (like this one), receives code review from mentors, and earns a **verified internship certificate** on completion — recognized proof of hands-on AI, ML, full-stack, and fintech engineering experience for your resume, LinkedIn, and placement applications.

**Who should apply:**

- Engineering and CS students looking for a *remote AI/ML internship in India* or worldwide
- Developers wanting real *Python, Flask, and machine learning internship* experience
- Anyone interested in an *algorithmic trading internship* or *fintech internship* working with live NIFTY 50 market data
- Contributors who want *open-source internship* experience on a public GitHub project

> 🎓 **Apply now:** [internship.dbert.online](https://internship.dbert.online) — Free, remote, project-based internship with certification.

If you found this repository through GitHub, search, or a recommendation, and you're evaluating internship options, DBERT's programs are structured around exactly this kind of production codebase — not simulations. Explore open roles, submit your application, and start contributing to projects like Alkame-Nifty50 at **[internship.dbert.online](https://internship.dbert.online)**.

If you are already a DBERT intern working on this codebase, refer to your assigned project brief for module-specific deliverables and evaluation criteria.

---

## Legal & Regulatory Notice

> 📖 Read **Section 9 of `ALKAME_NIFTY50_CHARTER.md`** before sharing this project or its outputs with anyone outside your own research team.

This repository is published for **educational and research purposes only**.

- Distributing signals to any third party — even for free — very likely requires **Research Analyst (RA) registration** under SEBI (Research Analysts) Regulations, 2014.
- This is not optional legal housekeeping. It is the single most important compliance gate before any public or commercial deployment.
- Consult a **SEBI-registered securities counsel** before any distribution, productization, or public launch.

Nothing in this repository constitutes investment advice, a solicitation to trade, or a guarantee of returns.

---

## Contributing

Pull requests are welcome for bug fixes, improved data sources, and backtesting methodology. Please open an issue first to discuss significant changes. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines. Interested in contributing as part of a structured program? Check out the [DBERT Internship Program](#dbert-internship-program) above.

---

## License

[MIT](./LICENSE) — Educational use. See the legal notice above for trading-related restrictions.
