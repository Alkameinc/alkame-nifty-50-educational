# Alkame Nifty50 API — Intern Verification Guide

> **Purpose:** This document is a step-by-step checklist for interns to verify that the Alkame Nifty50 API is running correctly on their local machine. Follow every section in order. All expected outputs are provided so you know exactly what to look for.

---

## Prerequisites

Before starting, make sure you have the following installed and set up:

| Tool | Minimum Version | Check Command |
|------|----------------|---------------|
| Python | 3.10+ | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| uvicorn | any | `uvicorn --version` |

### Clone the repo (if you haven't already)

```powershell
git clone https://github.com/Alkameinc/alkame-nifty-50-educational.git
cd alkame-nifty-50-educational
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Step 1 — Start the API Server

Open a **terminal** in the repo root and run:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

**Expected startup output (last few lines):**

```
INFO:macro_calendar:Loaded 9 macro calendar event(s) from ...
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

> [!IMPORTANT]
> If you see `SyntaxError` or `ImportError`, pull the latest changes with `git pull origin main` and try again.

Keep this terminal open. Open a **second terminal** for the verification commands below.

---

## Step 2 — Verify the Root Endpoint

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/ -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Expected response (HTTP 200):**

```json
{"message":"Welcome to Nifty50 API. Visit /docs for Swagger UI."}
```

> [!NOTE]
> If you get `{"detail":"Not Found"}`, the server may be running an old version. Pull latest and restart.

---

## Step 3 — Verify the Swagger UI (Interactive Docs)

Open your browser and navigate to:

```
http://127.0.0.1:8000/docs
```

**Expected:** A white page with the **Alkame Nifty50 API** title, a list of all endpoint groups (Health, Root, Signals, Admin, etc.) and **"Try it out"** buttons on each endpoint.

> [!WARNING]
> If the page is **blank/white with no content**, it means you're on an older version with a strict Content-Security-Policy that blocks the Swagger CDN scripts. Pull the latest code (`git pull origin main`) and restart the server.

You can also visit the ReDoc alternative docs at:

```
http://127.0.0.1:8000/redoc
```

---

## Step 4 — Verify the Health Check Endpoints

### 4a. Kubernetes-style probe (`/healthz`)

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/healthz -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Expected response (HTTP 200):**

```json
{
  "overall": "DEGRADED",
  "diagnostics": [
    {"component": "Runtime Validator", "status": "OK", "message": "Safety gate checks are executing normally."},
    {"component": "History Manager",   "status": "OK", "message": "Database operations are completing successfully."},
    {"component": "Ensemble Manager",  "status": "OK", "message": "Models loaded successfully."},
    {"component": "Data Fetcher",      "status": "OK", "message": "Market data providers are reachable."},
    {"component": "Predictor",         "status": "OK", "message": "Signal inference is operational."},
    ...
  ]
}
```

> [!NOTE]
> **`"overall": "DEGRADED"` is normal in a local dev environment.** Two components (`Corporate Events Fetcher` and `News Sentiment Fetcher`) show DEGRADED because they need a live `MARKETAUX_API_KEY` environment variable to reach external news APIs. All core components (DataFetcher, Predictor, EnsembleManager, HistoryManager, RuntimeValidator) must be `OK`. If any of those core ones are `DOWN`, flag it immediately.

### 4b. Full health endpoint (`/api/v1/health`)

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/v1/health -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Expected:** Same format as `/healthz` above.

---

## Step 5 — Verify Security Headers

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/ -UseBasicParsing | Select-Object -ExpandProperty Headers
```

**Every response must include all of the following headers:**

| Header | Expected Value |
|--------|---------------|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `X-XSS-Protection` | `1; mode=block` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` |
| `Content-Security-Policy` | Contains `default-src 'self'` and `cdn.jsdelivr.net` |

> [!CAUTION]
> If any of these headers are **missing**, do not merge code that removes them. These are production security requirements.

---

## Step 6 — Verify the Symbols Endpoint

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/v1/symbols -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Expected:** A JSON list of exactly 50 NIFTY50 stock symbols:

```json
{"symbols":["RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS",...]}
```

Check the count:

```powershell
$body = (Invoke-WebRequest -Uri http://127.0.0.1:8000/api/v1/symbols -UseBasicParsing).Content | ConvertFrom-Json
$body.symbols.Count
```

**Expected output:** `50`

---

## Step 7 — Verify the Admin Audit Logs Endpoint

This endpoint requires an API key. Use the default dev key:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/admin/audit-logs?limit=5" `
  -Headers @{"X-API-Key"="dev-alkame-admin-key"} `
  -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Expected response (HTTP 200):**

```json
{
  "status": "success",
  "logs": [
    {
      "id": 4,
      "timestamp": "2026-09-15T16:34:42.249791+00:00",
      "action": "TOGGLE_RISK",
      "resource": "global_risk",
      "status": "SUCCESS",
      "client_role": "ADMIN",
      "ip_address": "testclient"
    },
    ...
  ]
}
```

**Without an API key (testing auth):**

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/admin/audit-logs?limit=5" -UseBasicParsing
```

**Expected:** HTTP 401 with body:

```json
{"detail":"Invalid or missing API key. Provide valid key via 'X-API-Key' header or 'Authorization: Bearer <key>'."}
```

> [!IMPORTANT]
> The 401 response above is **correct and expected**. The API key system is working. Never expose `dev-alkame-admin-key` in production. In production the key is set via the `ALKAME_ADMIN_KEY` environment variable.

---

## Step 8 — Verify the Metrics Endpoint

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/metrics -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Expected:** Plain-text Prometheus metrics output, e.g.:

```
# HELP python_gc_objects_collected_total ...
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 1234.0
...
# HELP nifty50_predictions_total Total prediction signals generated
# TYPE nifty50_predictions_total counter
...
```

---

## Step 9 — Verify the Risk Toggle Endpoint

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/v1/risk/toggle -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Expected:**

```json
{"enabled":false,"reason":"","level_at_activation":null,"activated_at":null}
```

---

## Step 10 — Run the Full Test Suite

In a terminal with the **virtual environment active**, from the repo root:

```powershell
pytest -q
```

**Expected output:**

```
90 passed, X warnings in XX.XXs
```

> [!IMPORTANT]
> All **90 tests must pass**. If any tests fail, note the test name and error message and flag it in the team Slack channel before merging any code.

---

## Quick Verification Checklist

Print this checklist and tick off each item:

```
[ ] 1. Server starts without SyntaxError or ImportError
[ ] 2. GET /         → 200 {"message":"Welcome to Nifty50 API..."}
[ ] 3. /docs         → Swagger UI loads with endpoint list (not blank)
[ ] 4. GET /healthz  → 200 with all core components OK
[ ] 5. GET /api/v1/health → 200 with diagnostics
[ ] 6. Security headers present on every response
[ ] 7. GET /api/v1/symbols → 200 with 50 symbols
[ ] 8. GET /api/v1/admin/audit-logs (no key) → 401
[ ] 9. GET /api/v1/admin/audit-logs (with X-API-Key) → 200 with logs
[ ] 10. GET /metrics → 200 Prometheus text output
[ ] 11. GET /api/v1/risk/toggle → 200 with toggle state
[ ] 12. pytest -q → 90 passed, 0 failed
```

---

## Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Server won't start, `SyntaxError` | Old local code | `git pull origin main`, restart |
| `/docs` is blank | Old code with strict CSP | `git pull origin main`, restart |
| `/healthz` → 404 | Old code without the endpoint | `git pull origin main`, restart |
| Any endpoint → 401 | Missing `X-API-Key` header | Add `-Headers @{"X-API-Key"="dev-alkame-admin-key"}` |
| `Corporate Events Fetcher` DEGRADED | No `MARKETAUX_API_KEY` env var | Expected in dev — not a bug |
| `News Sentiment Fetcher` DEGRADED | No `MARKETAUX_API_KEY` env var | Expected in dev — not a bug |
| Tests fail unexpectedly | Dependency mismatch | `pip install -r requirements.txt` and retry |

---

## Environment Variables Reference

| Variable | Purpose | Default (dev only) |
|----------|---------|-------------------|
| `ALKAME_API_AUTH_ENABLED` | Enable/disable API key auth | `true` |
| `ALKAME_ADMIN_KEY` | Admin API key | `dev-alkame-admin-key` |
| `ALKAME_READONLY_KEY` | Read-only API key | `dev-alkame-readonly-key` |
| `MARKETAUX_API_KEY` | News/sentiment data provider | *(empty — degraded mode)* |
| `ALKAME_CORS_ORIGINS` | Allowed CORS origins | `http://localhost:8501,...` |

> [!CAUTION]
> **Never commit `.env` files or hardcoded API keys to git.** All secrets must come from environment variables. The default `dev-alkame-*` keys are for local development only and are not secret.

---



---

## Part 2: Verifying the React Frontend

Now that the backend is running perfectly on port 8000, we need to verify the new React frontend matches the updated API contract.

### 1. Start the Frontend Server

Open a **new terminal window** (keep the backend running in the first one) and navigate to the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

**Expected output:**
```
  VITE v6.0.3  ready in xxx ms

  ➜  Local:   http://localhost:3000/
```

### 2. Verify `.env.local` Configuration

If the frontend fails to load data, ensure your API key is correctly set so the proxy can authenticate. 
Create or check `frontend/.env.local`:

```
VITE_API_KEY=dev-alkame-admin-key
```
*(If you make changes to this file, you must restart the Vite server by pressing `Ctrl+C` and running `npm run dev` again).*

### 3. Frontend UI Checklist

Open `http://localhost:3000/` in your browser and verify the following components:

- [ ] **Dynamic Symbol Dropdown:** Clicking the "Select Stock" dropdown should show all 50 symbols fetched from the API.
- [ ] **Health Badge:** The top right corner should show a green badge saying "System Status: OK". Clicking it expands a diagnostic table.
- [ ] **Risk Banner:** If the backend `global_risk` toggle is manually enabled, a sticky red warning banner should appear at the top of the page.
- [ ] **Horizon Tabs:** You should see tabs for `INTRADAY`, `3D`, `7D`, `30D`, etc. Clicking them switches the active signal data without a page reload.
- [ ] **Mini Chart:** A responsive line chart should render below the price information, correctly plotting the last 100 price candles.
- [ ] **Scalping Panel:** If the `INTRADAY` tab is selected, a table titled "Intraday Scalping Opportunities" should appear at the bottom.
- [ ] **Narrative Summary:** An expandable "Overall Narrative" section should appear below the chart, providing human-readable AI analysis.

---

## Setting Up GPG-Verified Signatures for Commits

GitHub branch protection rules for this repository require all commits to be signed. Follow these steps to set up GPG signing:

### 1. Install GPG
- **Windows:** Download and install [Gpg4win](https://gpg4win.org/).
- **macOS:** Run `brew install gnupg`.
- **Linux (Ubuntu):** Run `sudo apt-get install gnupg`.

### 2. Generate a GPG Key
Open your terminal and run:
```powershell
gpg --full-generate-key
```
- Select **RSA and RSA** (default).
- Choose key size **4096**.
- Enter validity period (e.g., `1y` or `0` for no expiration).
- Enter your Name and the **exact email address** associated with your GitHub account.
- Set a secure passphrase.

### 3. Add the Key to GitHub
1. Find your key ID: `gpg --list-secret-keys --keyid-format=long`
   *(Look for the string after `rsa4096/`, e.g., `3AA5C34371567BD2`)*
2. Export the public key: `gpg --armor --export <YOUR_KEY_ID>`
3. Copy the output block (including `-----BEGIN PGP PUBLIC KEY BLOCK-----` to the end).
4. Go to **GitHub Settings** -> **SSH and GPG keys** -> **New GPG key**, and paste the block.

### 4. Configure Git to Use the Key
Run these commands in your terminal:
```powershell
git config --global user.signingkey <YOUR_KEY_ID>
git config --global commit.gpgsign true
```
Now, every time you run `git commit`, Git will automatically sign it using your GPG key.

## Who to Contact

| Issue | Contact |
|-------|---------|
| API not starting / test failures | Tech lead or senior dev |
| Frontend compilation errors | UI/UX team or senior dev |
| Git access / repo permissions | Alkame engineering team |
| API key for dev environment | Your onboarding manager |

---

*Document last updated: 15 September 2026 | Commit: `HEAD`*

