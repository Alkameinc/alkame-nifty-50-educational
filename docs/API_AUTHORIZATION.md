# API Endpoint Authorization Matrix

**Standard:** SEC-003 Endpoint Authorization Classification  
**Repository:** `https://github.com/Alkameinc/alkame-nifty-50-educational`  
**Last Updated:** 2026-09-16

Every API route implemented in the system is explicitly categorized in the matrix below. No route may accidentally inherit privileges from global or implicit assumptions.

---

## Authorization Classification Matrix

| Endpoint | Method | Public | Read-only | Admin | Internal | Description / Policy |
|---|---|:---:|:---:|:---:|:---:|---|
| `/` | `GET` | ✅ | - | - | - | Welcome root endpoint. Non-sensitive. |
| `/api/v1/health` | `GET` | ✅ | - | - | - | Health status & subsystem diagnostics. Safe liveness probe. |
| `/healthz` | `GET` | ✅ | - | - | - | Kubernetes-style liveness probe alias for `/api/v1/health`. |
| `/docs` | `GET` | ✅ | - | - | - | OpenAPI Swagger UI interactive documentation. |
| `/redoc` | `GET` | ✅ | - | - | - | ReDoc API specification documentation. |
| `/openapi.json` | `GET` | ✅ | - | - | - | Raw OpenAPI JSON schema. |
| `/metrics` | `GET` | - | - | ✅ | ✅ | Prometheus runtime telemetry metrics. Restricted to internal network (`127.0.0.1`, loopback, or scrapers with `ADMIN` API key). |
| `/api/v1/symbols` | `GET` | ✅ | ✅ | - | - | List of 50 active NIFTY index constituent symbols. |
| `/api/v1/chart/{symbol}` | `GET` | - | ✅ | ✅ | - | Historical bar data series for UI charting. |
| `/api/v1/signal/{symbol}` | `GET` | - | ✅ | ✅ | - | Multi-horizon prediction signal for requested symbol. |
| `/api/v1/signal/stream/{symbol}` | `GET` | - | ✅ | ✅ | - | Server-Sent Events (SSE) streaming predictions for symbol. |
| `/api/v1/scalping` | `GET` | - | ✅ | ✅ | - | Top intraday scalping opportunities. |
| `/api/v1/risk/toggle` | `GET` | - | ✅ | ✅ | - | Current state of human global risk toggle. |
| `/api/v1/risk/toggle` | `POST` | - | - | ✅ | - | State mutation: enables/disables human risk toggle. Requires `ADMIN` key; fails closed if audit log cannot be recorded. |
| `/api/v1/signal/{symbol}/refresh` | `POST` | - | ✅ | ✅ | - | Triggers on-demand live-worthiness refresh. Protected by symbol cooldown rate-limiting. |
| `/api/v1/admin/audit-logs` | `GET` | - | - | ✅ | - | Privileged endpoint to inspect historical audit log records. |

---

## Role Definitions

1. **Public (`Anonymous`)**: Accessible without API key or credentials. Limited strictly to non-sensitive liveness probes, welcome routes, and public documentation.
2. **Read-Only (`READ_ONLY`)**: Requires valid API key mapped to `READ_ONLY` or `ADMIN`. In development when `API_AUTH_ENABLED=false`, anonymous callers inherit `DEVELOPMENT_READONLY` allowing read-only access for local development without exposing admin mutations.
3. **Admin (`ADMIN`)**: Requires explicit valid key mapped to `ADMIN` role. Grants permission to mutate system state (e.g. risk toggle) and view administrative audit records.
4. **Internal (`Internal`)**: Loopback or private scraper address (`127.0.0.1`, `::1`, `localhost`), or verified `ADMIN` token.
