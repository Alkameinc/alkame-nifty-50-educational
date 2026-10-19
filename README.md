# Alkame-Nifty50 — Educational Market Intelligence Platform

> **DBERT Labs Internship 2026**  
> **Domain:** Quantitative Research & Data Analytics  
> **Program Details:** https://dbert.online · https://internship.dbert.online  
> This repository is a component of the DBERT Labs structural training track. All modifications, architectural designs, and research tools are developed under professional educational guidelines.

**NIFTY 50 Intraday Market Intelligence & Research Cockpit**  
*Educational · Multi-source Intelligence · Volume & Momentum Radar · SEBI Compliant · API-first*

---

> [!IMPORTANT]
> ### ⚖️ Statutory Legal & Regulatory Notice (SEBI Compliance)
> **This platform is strictly for academic, educational, and informational research purposes.**
> In accordance with the **Securities and Exchange Board of India (Research Analysts) Regulations, 2014** and **SEBI (Investment Advisers) Regulations, 2013**:
> - Alkame-Nifty50 **does NOT** provide any investment advice, trading tips, or financial recommendations.
> - Alkame-Nifty50 **does NOT** issue directional `BUY` or `SELL` trade calls, projected price targets, or algorithmic stop-loss recommendations.
> - All outputs, visual corridors, pivot levels, and technical indicators are mathematical computations derived purely from publicly available historical data and regulatory filings.
> - Any directional predictive modeling or machine learning weight sets are strictly quarantined locally for offline computational research (`_local_research/`) and are never distributed or served publicly.
> - Securities trading and derivatives involve substantial financial risk. Users must consult a SEBI-registered Investment Adviser before making any financial decisions.

---

## 🏛️ Overview

Alkame-Nifty50 is an open-source, research-grade **Market Intelligence & Quantitative Analytics Platform** for the National Stock Exchange (NSE) NIFTY 50 universe. 

Rather than issuing speculative price predictions or trading recommendations, Alkame unifies high-frequency market data, corporate filings, macro indicators, and factual technical metrics into a clean, institutional-grade analytical cockpit:

- **Descriptive Over Prescriptive**: Purely factual indicators (RSI-14, 20/50/200 EMAs, Bollinger Bands, ATR, VWAP, Camarilla pivots) without directional bias.
- **Multi-Source Context**: Integrates NSE regulatory announcements, corporate board meetings, RBI monetary policy events, global risk drivers (India VIX, Crude Oil, USD/INR), and news sentiment.
- **Market Breadth & Volume Radar**: Live constituent sweeps tracking volume surges vs. 20-day averages, advance/decline ratios, and 52-week proximity.
- **Offline Research Isolation**: Algorithmic models and backtesting harnesses are quarantined in `_local_research/` (excluded from git) to maintain a clean educational open-source repository.
- **Triple-Frontend Architecture**:
  1. **Institutional Cockpit HTML** (`alkame_dashboard.html`): High-contrast Dark/Light trading desk UI with zero build dependencies.
  2. **Modern React Vite Web App** (`frontend/`): Responsive modular SPA with Tailwind CSS, Lucide icons, and real-time polling.
  3. **Streamlit Analytics Studio** (`app.py`): Rapid desktop research tool for interactive filtering and exploratory analysis.

---

## 🚀 Key Features

### 1. Unified Market Insights Engine (`insights_engine.py`)
Central analytical service producing structured market intelligence (`GET /api/v1/insights/{symbol}`):
- **Technical Posture**: Factual RSI-14, moving average alignments (EMA 20, EMA 50, SMA 200), Bollinger Band %B & bandwidth, ATR volatility.
- **Mathematical Reference Channels**: Camarilla pivot floors and ceilings (H3, H4, L3, L4), standard floor pivots (P, R1, S1, R2, S2), and prior session high/low bounds.
- **Fundamentals & Valuation**: Trailing P/E, Price-to-Book, Market Cap, EPS, and Dividend Yield.
- **Corporate Filings**: Direct feed from NSE regulatory announcements, financial results schedules, and board actions.
- **Macro Risk Matrix**: RBI repo corridor, Union Budget schedule, and global contagion monitors (Brent crude, USD/INR, India VIX).
- **News Polarity**: Factual sentiment classification across audited financial news feeds.

### 2. Market Breadth & Intraday Volume Radar (`GET /api/v1/market-breadth`)
Sweeps all 50 NIFTY constituents to track real-time liquidity and momentum without trade signals:
- Advance / Decline / Unchanged constituent distribution.
- Volume surges where current session volume exceeds $1.5\times$ 20-day average.
- Proximity percentages to 52-week high and 52-week low.
- Constituent sorting by momentum and volume anomaly.

### 3. Institutional Research Cockpit (`alkame_dashboard.html`)
An enterprise dashboard designed for data analysts and institutional researchers:
- **Regime Posture Gate**: Descriptive summary of RSI, volume ratio, and Camarilla channels.
- **Multi-Timeframe Technical Matrix**: Intraday to multi-week trend and moving average posture.
- **Live Market Stream & Volatility Channels**: High-refresh canvas plotting real-time prices within VWAP and Camarilla boundaries.
- **System Diagnostics Matrix**: 16-subsystem health telemetry verifying data provider latency, database connections, and cache states.
- **Audit Ledger**: Comprehensive historical log with cryptographic hashes for reproducible research.

---

## 📁 Repository Structure

```
alkame-nifty-50-educational/
├── alkame_dashboard.html           # Institutional Cockpit (Standalone Dual-Theme HTML)
├── api.py                          # FastAPI REST Backend Layer
├── api_schemas.py                  # Pydantic Schemas for Requests & Responses
├── app.py                          # Streamlit Research Dashboard
├── config.py                       # System Configuration (SEBI_COMPLIANCE_MODE=True)
├── insights_engine.py              # Central Market Intelligence & Breadth Engine
├── prediction_types.py             # Neutral Educational Data Contracts & Schemas
├── corporate_events_fetcher.py     # NSE Announcements & Filings Collector
├── data_fetcher.py                 # Multi-Tier yfinance Data Fetcher with Cache
├── event_classifier.py             # Event Taxonomy (MARKET / SECTOR / STOCK)
├── global_risk_monitor.py          # Macro Risk Factors (VIX, Crude, USD/INR)
├── health_monitor.py               # 16-Subsystem Health Diagnostic Registry
├── history_manager.py              # SQLite Historical Data & Snapshot Store
├── human_insight_manager.py        # Analyst Notes & Research Task Store
├── macro_calendar.py               # Macroeconomic Event Calendar
├── news_sentiment_fetcher.py       # News Feed Parser & Polarity Scorer
├── reference_level_engine.py       # Camarilla & Standard Pivot Calculations
├── sector_provider.py              # NIFTY 50 Sector Classification Provider
│
├── frontend/                       # Modern React 19 + Vite 6 Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── StockInsightsCard.jsx   # Technical & Corporate Intelligence Card
│   │   │   └── MarketBreadthPanel.jsx  # Universe Volume & Momentum Radar
│   │   ├── App.jsx                     # Main Application Layout
│   │   └── api.js                      # API Client Layer
│   └── package.json
│
├── tests/                          # Automated Pytest Suite
│   ├── test_insights_engine.py     # Insights Engine & REST Endpoint Tests
│   ├── test_dashboard_api_endpoints.py # Core API Contract Tests
│   └── test_app.py                 # Streamlit UI Tests
│
└── _local_research/                # [GIT-IGNORED] Offline Machine Learning Research
    ├── models/                     # Trained Joblib/Pkl Weight Bundles
    ├── model_trainer.py            # Local ML Training Harness
    ├── backtester.py               # Quantitative Historical Backtester
    └── predictor.py                # Raw ML Directional Classifiers
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.11 or 3.12
- Node.js 18+ (for React frontend)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Alkameinc/alkame-nifty-50-educational.git
cd alkame-nifty-50-educational

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Launching the Services

#### Option A: FastAPI Backend & Institutional Cockpit (Recommended)
```bash
# Start FastAPI backend
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Once started:
- **Cockpit UI**: Open `http://127.0.0.1:8000/dashboard` in your browser.
- **API Swagger Docs**: Visit `http://127.0.0.1:8000/docs`.

#### Option B: React Vite Modern Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the React interface.

#### Option C: Streamlit Analytics Studio
```bash
streamlit run app.py
```
Open `http://localhost:8501` for exploratory analysis.

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/insights/{symbol}` | **Primary**: Factual technical indicators, Camarilla pivots, valuations, corporate filings, macro context, and news sentiment. |
| `GET` | `/api/v1/market-breadth` | **Primary**: Universe volume radar, volume surge ratios vs. 20D average, advances/declines, and 52W proximity. |
| `GET` | `/api/v1/fundamentals/{symbol}` | Valuation multiples, market capitalization, 52-week ranges, and sector data. |
| `GET` | `/api/v1/symbols` | List of 50 active NIFTY index constituent symbols. |
| `GET` | `/api/v1/chart/{symbol}` | OHLCV historical candlestick data formatted for charting. |
| `GET` | `/api/v1/health` | Comprehensive 16-subsystem health diagnostic matrix. |
| `GET` | `/api/v1/risk/reading` | Global risk monitor reading (India VIX, Crude Oil, USD/INR). |
| `GET` | `/api/v1/scalping` | Educational notice directing consumers to `/api/v1/market-breadth`. |
| `GET` | `/dashboard` | Serves the standalone Institutional Cockpit HTML application. |

---

## 🧪 Testing & Verification

Run the test suite to verify data pipelines and API schema integrity:
```bash
# Run insights engine and API tests
pytest tests/test_insights_engine.py tests/test_dashboard_api_endpoints.py tests/test_app.py -v

# Run React frontend build test
cd frontend && npm run build
```

---

## 🔬 Local Research Isolation

In strict compliance with open-source and regulatory boundaries, machine learning training workflows and predictive models are **not tracked in the Git repository**:
- Model artifacts (`models/*.joblib`, `models/*.pkl`, `models/*.json`) are excluded in `.gitignore`.
- Algorithmic research scripts (`model_trainer.py`, `backtester.py`, `scalping.py`, `ensemble_manager.py`) are backed up in `_local_research/` for offline mathematical experimentation.
- The public codebase remains 100% functional as an informational market intelligence platform without requiring proprietary model files.

---

## 🤝 DBERT Internship Program

This project is part of the **DBERT AI Internship Program**.  
- Program Portal: [https://internship.dbert.online](https://internship.dbert.online)  
- Organization: [https://dbert.online](https://dbert.online)  
- Track: Data Analyst & Quantitative Market Intelligence  

All contributions must follow the educational charter and SEBI non-advisory compliance rules outlined in this document.

---

## 📄 License
This repository is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
