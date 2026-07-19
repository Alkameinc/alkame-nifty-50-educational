# Alkame-Nifty50 — Educational Repository

> **DBERT Labs Internship 2026**
> **Domain:** Data Analyst
> **Intern:** Ankur Sen
> **Program Details:** https://dbert.online · https://internship.dbert.online
> This repository is a component of the DBERT Labs structural training track. All modifications, upgrade branches, and submitted pull requests are developed under professional educational guidelines.

**NIFTY 50 Intraday Market Intelligence Engine**
*Human-in-the-loop · Calibration-gated · Event-aware*

> ⚠️ **Educational Repository** — This project is shared for learning purposes only. It does not constitute financial advice. Read the [Legal & Regulatory Notice](#legal--regulatory-notice) section before any use.

---

## Overview

Alkame-Nifty50 is an open-source, research-grade intraday signal engine for the NIFTY 50 index. It is built around three core design principles:

- **Risk-first**: `HOLD` with a reason is a valid — and often correct — output.
- **Calibration before confidence**: No signal confidence is ever surfaced until runtime calibration checks pass.
- **Human-in-the-loop**: Trader overrides, notes, and feedback are first-class citizens of the pipeline.

The full methodology, regulatory rationale, and business charter live in [`ALKAME_NIFTY50_CHARTER.md`](./ALKAME_NIFTY50_CHARTER.md). Read it first — it explains *why* the system is built the way it is.

---

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | Central configuration — tickers, sectors, thresholds, paths |
| `macro_calendar.py` | RBI/Budget/election/festive calendar (human-maintained CSV) |
| `data_fetcher.py` | yfinance OHLCV data with retry + cache fallback |
| `corporate_events_fetcher.py` | NSE corporate announcements, board meetings, actions, block/bulk deals |
| `news_sentiment_fetcher.py` | marketaux + Google News RSS sentiment |
| `event_classifier.py` | Unifies all events, tags scope (MARKET / SECTOR / STOCK) |
| `global_risk_monitor.py` | Always-on composite risk score + human-gated toggle |
| `feature_engineer.py` | Technical indicators, strict no-lookahead discipline |
| `model_trainer.py` | Per-stock direction classifier, time-based split |
| `ensemble_manager.py` | Multi-model soft-voting ensemble with agreement scoring |
| `runtime_validator.py` | Calibration + edge-vs-NIFTY checks — the safety gate |
| `predictor.py` | Combines everything into one final BUY / SELL / HOLD signal |
| `human_insight_manager.py` | Trader notes, overrides, feedback (SQLite) |
| `history_manager.py` | Persists predictions/events, resolves outcomes (SQLite) |
| `backtester.py` | Historical replay with slippage/costs, real alpha calculation |
| `scheduler.py` | Orchestrates the full pipeline on a market-hours loop |
| `app.py` | Streamlit dashboard |

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
python feature_engineer.py
python model_trainer.py
python ensemble_manager.py
python runtime_validator.py
python predictor.py
python human_insight_manager.py
python history_manager.py
python backtester.py
python scheduler.py
python app.py                         # Pure-logic self-test only; see Running section for the real dashboard
```

Each script prints `STATUS: PASS` or `STATUS: FAIL` at the end.

> **Note:** `data_fetcher.py`, `corporate_events_fetcher.py`, `news_sentiment_fetcher.py`, `global_risk_monitor.py`, `predictor.py`, and `scheduler.py` touch live external services and may take up to a minute or two.

---

## Running

**Terminal 1 — Signal Scheduler**

```bash
python scheduler.py
```

> `scheduler.py` exposes the `Scheduler` class and its self-test. Wire up `run_forever()` with a real `symbol_data_provider` callable (using `data_fetcher.py`) when you're ready for continuous operation. The integration point is intentionally left open — your symbol list, refresh cadence, and error-handling preferences for unattended operation are worth deciding deliberately.

**Terminal 2 — Dashboard**

```bash
streamlit run app.py
```

---

## Known Limitations

- **NIFTY 50 constituent list** and sector map need verification against NSE's next semi-annual index review.
- **Festive-window dates** are approximate (lunar calendar shifts yearly) — verify each year.
- **Exchange holidays** are not yet modeled in `scheduler.is_market_open()` — it currently only checks weekday + time window.
- **`corporate_events_fetcher.py`** may be blocked from cloud/VPN IPs by NSE's bot protection — run from a normal home/office connection.

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

Pull requests are welcome for bug fixes, improved data sources, and backtesting methodology. Please open an issue first to discuss significant changes.

---

## License

[MIT](./LICENSE) — Educational use. See the legal notice above for trading-related restrictions.
