import json
import logging
import re
import uuid
from datetime import datetime, timezone

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Gauge, Histogram

from api_schemas import (
    ErrorResponse,
    HealthResponse,
    ModelValidityOut,
    MultiHorizonSignalResponse,
    SymbolsResponse,
)
from model_validity import can_serve_live_signal, evaluate_model_validity
from config import (
    API_AUTH_ENABLED,
    API_KEYS_ROLE_MAP,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOWED_HEADERS,
    CORS_ALLOWED_METHODS,
    CORS_ALLOWED_ORIGINS,
    IS_PRODUCTION,
    NIFTY50_SYMBOLS,
    to_yfinance_ticker,
)
from database import SessionLocal
from health_monitor import registry as health_registry
from history_manager import HistoryManager
from models import AuditLog
from scalping import ScalpingEngine
from scheduler import Scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alkame Nifty50 API", version="1.0.0")

# Security dependencies (SEC-001, SEC-002, SEC-003)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


class ClientAuth:
    def __init__(self, key: str, role: str):
        self.key = key
        self.role = role


def get_current_client(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_auth),
) -> ClientAuth:
    if not API_AUTH_ENABLED:
        if IS_PRODUCTION:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="CRITICAL SECURITY ERROR (SEC-001): Authentication cannot be disabled in production.",
            )
        # Development: explicitly DEVELOPMENT_READONLY, never ADMIN (SEC-001)
        return ClientAuth(key="disabled", role="DEVELOPMENT_READONLY")

    provided_key = None
    if api_key:
        provided_key = api_key
    elif bearer and bearer.credentials:
        provided_key = bearer.credentials

    if not provided_key or provided_key not in API_KEYS_ROLE_MAP:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide valid key via 'X-API-Key' header or 'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role = API_KEYS_ROLE_MAP[provided_key]
    return ClientAuth(key=provided_key, role=role)


def require_role(required_role: str):
    def role_checker(client: ClientAuth = Depends(get_current_client)) -> ClientAuth:
        if required_role == "ADMIN":
            if client.role != "ADMIN":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Action requires '{required_role}' privilege. Client has role '{client.role}'.",
                )
        elif required_role == "READ_ONLY":
            if client.role not in ("READ_ONLY", "ADMIN", "DEVELOPMENT_READONLY"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Action requires '{required_role}' privilege. Client has role '{client.role}'.",
                )
        return client

    return role_checker


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    # SEC-006: Protect /metrics from external unauthenticated access
    if request.url.path == "/metrics":
        client_host = request.client.host if request.client else ""
        is_internal = client_host in ("127.0.0.1", "::1", "testclient", "localhost")
        if not is_internal:
            api_key = request.headers.get("X-API-Key")
            auth_header = request.headers.get("Authorization", "")
            bearer_key = auth_header.replace("Bearer ", "").strip() if "Bearer " in auth_header else ""
            key = api_key or bearer_key
            if not key or API_KEYS_ROLE_MAP.get(key) != "ADMIN":
                return Response(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content="Forbidden: /metrics requires internal or ADMIN access.",
                )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "frame-ancestors 'none';"
    )
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


def log_audit_event(
    client: ClientAuth,
    action: str,
    resource: str,
    status: str,
    details: str,
    request: Request,
    fail_closed: bool = False,
):
    ip_address = request.client.host if request and request.client else "unknown"
    key_prefix = client.key[:6] if client.key else "none"
    try:
        with SessionLocal() as db:
            audit = AuditLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                client_key_prefix=key_prefix,
                client_role=client.role,
                action=action,
                resource=resource,
                status=status,
                details=details,
                ip_address=ip_address,
            )
            db.add(audit)
            db.commit()
    except Exception as e:
        logger.error(f"Failed to record audit log: {e}")
        if fail_closed:
            raise RuntimeError(f"Audit log recording failed: {e}") from e


# Strict CORS without wildcard credentials (SEC-008)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)


_SAFE_CORRELATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]{1,64}$")


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    # SEC-010: Validate/sanitize client correlation IDs
    raw_cid = request.headers.get("X-Correlation-ID")
    if raw_cid and _SAFE_CORRELATION_ID_PATTERN.match(raw_cid):
        correlation_id = raw_cid
    else:
        correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# Prometheus custom domain metrics (OBS-002)
PREDICTIONS_TOTAL = Counter(
    "nifty50_predictions_total",
    "Total prediction signals generated",
    ["symbol", "horizon", "action"],
)
SYSTEM_HEALTH_STATUS = Gauge(
    "nifty50_health_status",
    "Current health status by component (1=OK, 0=FAIL/DEGRADED)",
    ["component"],
)
MODEL_INFERENCE_SECONDS = Histogram(
    "nifty50_model_inference_seconds",
    "Model inference latency in seconds",
    ["symbol"],
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception as err:
    logger.warning(f"Could not initialize prometheus-fastapi-instrumentator: {err}")

scheduler = Scheduler()
history_manager = HistoryManager()
scalping_engine = ScalpingEngine(scheduler.predictor, scheduler.data_fetcher, scheduler.predictor.feature_engineer)


def translate_health_message(component: str, status: str) -> str:
    if status == "OK":
        messages = {
            "history_manager": "Database operations are completing successfully.",
            "data_fetcher": "Market data providers are reachable.",
            "runtime_validator": "Safety gate checks are executing normally.",
            "global_risk_monitor": "Risk thresholds are within standard operational limits.",
            "human_insight_manager": "Override subsystem is available.",
            "ensemble_manager": "Models loaded successfully.",
            "predictor": "Signal inference is operational.",
            "scheduler": "Periodic signal and outcome scheduling pipeline is operational.",
            "storage": "SQLite storage engine and WAL journal are healthy and verified.",
        }
        return messages.get(component, "Component is operational.")
    elif status == "DEGRADED":
        return "Component is experiencing elevated latency or partial failure."
    return "Component is offline or failing all checks."


def humanize_reasoning(reasons: list) -> list:
    humanized = []
    for r in reasons:
        if r.startswith("Ensemble model lean:"):
            # e.g. "Ensemble model lean: DOWN (raw confidence 0.47, model agreement 33%)."
            if "DOWN" in r:
                humanized.append("The AI models are predicting a downward trend.")
            elif "UP" in r:
                humanized.append("The AI models are predicting an upward trend.")
            else:
                humanized.append("The AI models are predicting a flat or sideways trend.")
        elif r.startswith("Global risk level:"):
            if "NORMAL" in r:
                humanized.append("Global risk is normal, so no penalties have been applied to this prediction.")
            else:
                humanized.append(
                    "Global risk is elevated, so we've lowered our confidence in this prediction to keep you safe."
                )
        elif "No specific events" in r:
            humanized.append("We haven't detected any major breaking news or events affecting this stock right now.")
        elif "No calibration or edge-check data" in r or "Model leaned" in r:
            if "forced to HOLD" in r:
                humanized.append(
                    "We forced a HOLD because this specific strategy hasn't proven itself against the NIFTY baseline yet. We prioritize safety over unproven trades."
                )
            else:
                humanized.append(
                    "We don't have enough historical proof that this pattern works yet, so we are staying cautious."
                )
        else:
            humanized.append(r)
    return list(dict.fromkeys(humanized))  # remove duplicates


@app.get("/api/v1/health", response_model=HealthResponse)
def get_health():
    overall = health_registry.get_overall_status()
    statuses = health_registry.get_status()
    engine_health_res = health_registry.get_engine_health()

    diagnostic = []
    if statuses:
        for s in statuses:
            SYSTEM_HEALTH_STATUS.labels(component=s.component).set(1.0 if s.status == "OK" else 0.0)
            diagnostic.append(
                {
                    "component": s.component.replace("_", " ").title(),
                    "status": s.status,
                    "message": translate_health_message(s.component, s.status),
                }
            )
    return {
        "overall": overall,
        "diagnostics": diagnostic,
        "engine_health": engine_health_res.status,
        "checks": engine_health_res.checks,
    }


@app.get("/api/v1/model-validity/{symbol}", response_model=ModelValidityOut)
def get_model_validity(symbol: str, horizon: str = "INTRADAY"):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    res = evaluate_model_validity(symbol=symbol, horizon=horizon)
    return ModelValidityOut(
        symbol=res.symbol,
        horizon=res.horizon,
        validity_status=res.validity_status,
        is_live_eligible=res.is_live_eligible,
        model_version=res.model_version,
        trained_at=res.trained_at,
        age_days=res.age_days,
        calibration_status=res.calibration_status,
        edge_status=res.edge_status,
        reasons=res.reasons,
        metadata=res.metadata,
    )


@app.get("/healthz", tags=["Health"], response_model=HealthResponse)
def healthz():
    """Kubernetes-style liveness probe - alias for /api/v1/health."""
    return get_health()


@app.get("/api/v1/symbols", response_model=SymbolsResponse)
def get_symbols():
    return {"symbols": NIFTY50_SYMBOLS}


@app.get(
    "/api/v1/signal/{symbol}",
    response_model=MultiHorizonSignalResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_signal(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")

    event_context = scheduler.get_event_context(symbol)
    context = scheduler.build_prediction_context(
        symbol,
        macro_events=event_context.macro_events,
        corporate_events=event_context.corporate_events,
        news_articles=event_context.news_articles,
    )
    stock_df = context.market_data
    index_df = context.index_data

    if stock_df is None or not isinstance(stock_df, pd.DataFrame) or stock_df.empty:
        raise HTTPException(status_code=503, detail=f"Could not fetch data for {symbol}")

    scheduler.resolve_pending_outcomes(symbol, stock_df)
    cycle_res = scheduler.run_one_cycle_for_symbol(
        symbol,
        stock_df,
        index_df,
        macro_events=event_context.macro_events,
        corporate_events=event_context.corporate_events,
        news_articles=event_context.news_articles,
    )
    multi_signal = getattr(cycle_res, "signal", cycle_res)
    if multi_signal is None or not hasattr(multi_signal, "signals") or not multi_signal.signals:
        raise HTTPException(status_code=503, detail=f"No signal available for {symbol}")

    # We can fetch narrative once
    recent_records = history_manager.get_predictions(symbol, limit=1)
    narrative = (
        recent_records[0].narrative
        if recent_records and getattr(recent_records[0], "narrative", None)
        else "No narrative available."
    )

    engine_health_res = health_registry.get_engine_health()

    all_horizons_data = {}
    for hor, sig in multi_signal.signals.items():
        validity_res = evaluate_model_validity(symbol, hor)
        can_serve, serve_reasons = can_serve_live_signal(engine_health_res, validity_res)

        sig_action = sig.action
        calibrated_conf = getattr(sig, "calibrated_confidence", None)
        raw_reasoning = list(sig.reasoning)

        if not can_serve:
            sig_action = "HOLD"
            calibrated_conf = None
            raw_reasoning.extend(serve_reasons)
            if not validity_res.is_live_eligible:
                verdict_text = f"Holding: Model for {hor} is {validity_res.validity_status} and not eligible for live execution."
            else:
                verdict_text = "Holding: Engine health is FAILED; live execution halted."
        else:
            if sig_action == "BUY":
                verdict_text = "Strong opportunity identified. Proceed with entry according to your risk parameters."
            elif sig_action == "SELL":
                verdict_text = "Warning: Downward pressure detected. Consider hedging or reducing exposure."
            else:
                if sig.suppressed:
                    verdict_text = "Holding back: We don't have enough historical proof that this pattern works yet."
                else:
                    verdict_text = "No clear edge detected. Better to stay out and wait for a higher-probability setup."

        PREDICTIONS_TOTAL.labels(symbol=symbol, horizon=hor, action=sig_action).inc()

        events = []
        for e in sig.contributing_events:
            events.append({"type": e.event_type, "label": e.headline_or_label, "sentiment": e.sentiment_score})

        all_horizons_data[hor] = {
            "horizon": hor,
            "action": sig_action,
            "verdict_text": verdict_text,
            "confidence": calibrated_conf,
            "raw_confidence": getattr(sig, "raw_confidence", 0.0),
            "risk_adjusted_confidence": getattr(sig, "risk_adjusted_confidence", 0.0),
            "calibrated_confidence": calibrated_conf,
            "calibration_status": "VALID" if calibrated_conf is not None else "UNAVAILABLE",
            "current_price": float(stock_df["Close"].iloc[-1]) if not stock_df.empty else None,
            "target_price": sig.target_price,
            "stop_loss": sig.stop_loss,
            "peak_potential_price": getattr(sig, "peak_potential_price", None),
            "downside_summary": sig.downside_summary,
            "upside_summary": sig.upside_summary,
            "events": events,
            "reasoning": humanize_reasoning(raw_reasoning),
            "prediction_key": getattr(sig, "prediction_key", None),
        }

    return {
        "symbol": symbol,
        "narrative": narrative,
        "signals": all_horizons_data,
        "prediction_key": getattr(multi_signal, "prediction_key", None),
    }


@app.get("/api/v1/signal/stream/{symbol}")
def stream_signal(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")

    event_context = scheduler.get_event_context(symbol)
    context = scheduler.build_prediction_context(
        symbol,
        macro_events=event_context.macro_events,
        corporate_events=event_context.corporate_events,
        news_articles=event_context.news_articles,
    )
    stock_df = context.market_data
    index_df = context.index_data

    if stock_df is None or not isinstance(stock_df, pd.DataFrame) or stock_df.empty:
        raise HTTPException(status_code=503, detail=f"Could not fetch data for {symbol}")

    scheduler.resolve_pending_outcomes(symbol, stock_df)
    engine_health_res = health_registry.get_engine_health()

    def generate():
        stream = scheduler.run_cycle_stream_for_symbol(
            symbol,
            stock_df,
            index_df,
            macro_events=event_context.macro_events,
            corporate_events=event_context.corporate_events,
            news_articles=event_context.news_articles,
        )
        for sig in stream:
            validity_res = evaluate_model_validity(symbol, sig.horizon)
            can_serve, serve_reasons = can_serve_live_signal(engine_health_res, validity_res)

            sig_action = sig.action
            calibrated_conf = getattr(sig, "calibrated_confidence", None)
            raw_reasoning = list(getattr(sig, "reasoning", []))

            if not can_serve:
                sig_action = "HOLD"
                calibrated_conf = None
                raw_reasoning.extend(serve_reasons)
                if not validity_res.is_live_eligible:
                    verdict_text = f"Holding: Model for {sig.horizon} is {validity_res.validity_status} and not eligible for live execution."
                else:
                    verdict_text = "Holding: Engine health is FAILED; live execution halted."
            else:
                if sig_action == "BUY":
                    verdict_text = "Strong opportunity identified. Proceed with entry according to your risk parameters."
                elif sig_action == "SELL":
                    verdict_text = "Warning: Downward pressure detected. Consider hedging or reducing exposure."
                else:
                    if getattr(sig, "suppressed", False):
                        verdict_text = "Holding back: We don't have enough historical proof that this pattern works yet."
                    else:
                        verdict_text = "No clear edge detected. Better to stay out and wait for a higher-probability setup."

            events = []
            if getattr(sig, "contributing_events", None):
                for e in sig.contributing_events:
                    events.append(
                        {
                            "type": getattr(e, "event_type", ""),
                            "label": getattr(e, "headline_or_label", ""),
                            "sentiment": getattr(e, "sentiment_score", 0.0),
                        }
                    )

            data = {
                "horizon": sig.horizon,
                "action": sig_action,
                "verdict_text": verdict_text,
                "confidence": calibrated_conf,
                "raw_confidence": getattr(sig, "raw_confidence", 0.0),
                "risk_adjusted_confidence": getattr(
                    sig, "risk_adjusted_confidence", getattr(sig, "raw_confidence", 0.0)
                ),
                "calibrated_confidence": calibrated_conf,
                "calibration_status": (
                    "VALID" if calibrated_conf is not None else "UNAVAILABLE"
                ),
                "current_price": float(stock_df["Close"].iloc[-1]) if not stock_df.empty else None,
                "target_price": getattr(sig, "target_price", None),
                "stop_loss": getattr(sig, "stop_loss", None),
                "peak_potential_price": getattr(sig, "peak_potential_price", None),
                "downside_summary": getattr(sig, "downside_summary", ""),
                "upside_summary": getattr(sig, "upside_summary", ""),
                "events": events,
                "reasoning": humanize_reasoning(raw_reasoning),
            }
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


import threading
import time

# --- Refresh protection state ---
_refresh_locks: dict[str, threading.Lock] = {}  # symbol -> threading.Lock
_refresh_last_time: dict[str, float] = {}  # symbol -> float (epoch)
_REFRESH_COOLDOWN_SECONDS = 60  # Minimum seconds between refreshes for the same symbol


def _get_refresh_lock(symbol: str) -> threading.Lock:
    if symbol not in _refresh_locks:
        _refresh_locks[symbol] = threading.Lock()
    return _refresh_locks[symbol]


@app.post("/api/v1/signal/{symbol}/refresh")
def refresh_backtest(
    symbol: str, request: Request, response: Response = Response(), client: ClientAuth = Depends(get_current_client)
):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")

    lock = _get_refresh_lock(symbol)
    if not lock.acquire(blocking=False):
        if response is not None:
            response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"status": "rejected", "reason": "A refresh is already running for this symbol.", "http_status": 429}

    try:
        last = _refresh_last_time.get(symbol, 0)
        elapsed = time.time() - last
        if elapsed < _REFRESH_COOLDOWN_SECONDS:
            remaining = int(_REFRESH_COOLDOWN_SECONDS - elapsed)
            if response is not None:
                response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
            return {
                "status": "rejected",
                "reason": f"Refresh cooldown active. Try again in {remaining}s.",
                "http_status": 429,
            }

        yf_ticker = to_yfinance_ticker(symbol)
        raw_stock = scheduler.data_fetcher.fetch_ohlcv(yf_ticker)
        raw_index = scheduler.data_fetcher.fetch_nifty_index()
        stock_df = getattr(raw_stock, "df", raw_stock)
        index_df = getattr(raw_index, "df", raw_index)
        if (
            isinstance(stock_df, pd.DataFrame)
            and isinstance(index_df, pd.DataFrame)
            and not stock_df.empty
            and not index_df.empty
        ):
            scheduler.refresh_live_worthiness(symbol, stock_df, index_df)
            _refresh_last_time[symbol] = time.time()
            return {"status": "success"}
        return {"error": "Failed to fetch data"}
    finally:
        lock.release()


@app.get("/api/v1/scalping")
def get_scalping():
    setups = scalping_engine.find_opportunities(limit=5)
    result = []
    for s in setups:
        result.append(
            {
                "symbol": s.symbol,
                "action": s.action,
                "entry": s.entry_price,
                "target": s.target_price,
                "stop": s.stop_loss,
                "confidence": s.confidence,
                "rr": s.risk_reward_ratio,
            }
        )
    return {"setups": result}


from config import HORIZON_CONFIG


@app.get("/api/v1/chart/{symbol}")
def get_chart(symbol: str, horizon: str = "INTRADAY"):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    yf_ticker = to_yfinance_ticker(symbol)

    cfg = HORIZON_CONFIG.get(horizon, HORIZON_CONFIG["INTRADAY"])
    interval = str(cfg["bar_interval"])
    period = str(cfg["history_period"])

    raw_stock = scheduler.data_fetcher.fetch_ohlcv(yf_ticker, interval=interval, period=period)
    stock_df = getattr(raw_stock, "df", raw_stock)
    if stock_df is None or not isinstance(stock_df, pd.DataFrame) or stock_df.empty:
        return {"error": "No data"}

    # Return last 100 bars for charting
    recent = stock_df.tail(100)
    chart_data = []
    for idx, row in recent.iterrows():
        # Format date differently based on interval
        if interval == "5m" or interval == "1m":
            time_str = idx.strftime("%d %b %H:%M") if hasattr(idx, "strftime") else str(idx)
        else:
            time_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)

        chart_data.append(
            {"time": time_str, "close": float(row["Close"]), "volume": int(row["Volume"]) if "Volume" in row else 0}
        )
    return {"chart": chart_data}


@app.get("/api/v1/risk/toggle")
def get_risk_toggle():
    state = scheduler.predictor.global_risk_monitor.get_toggle_state()
    return {
        "enabled": state.enabled,
        "reason": state.reason,
        "level_at_activation": state.level_at_activation,
        "activated_at": state.activated_at,
    }


@app.post("/api/v1/risk/toggle")
def set_risk_toggle(enabled: bool, request: Request, client: ClientAuth = Depends(require_role("ADMIN"))):
    reason = f"Toggled by {client.role} ({client.key[:6]}...) via API"
    try:
        log_audit_event(
            client, "TOGGLE_RISK", "global_risk", "SUCCESS", f"Set enabled={enabled}", request, fail_closed=True
        )
    except Exception as e:
        logger.error(f"Audit failure aborting risk toggle mutation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Critical state mutation aborted because audit logging failed.",
        ) from e

    state = scheduler.predictor.global_risk_monitor.set_toggle(enabled, reason, changed_by=client.role)
    return {"status": "success", "enabled": state.enabled}


@app.get("/api/v1/scheduler/status")
def get_scheduler_status():
    """SCHED-001 / OPS-001: Exposes real-time scheduler state and circuit breaker status."""
    return scheduler.get_status()


@app.post("/api/v1/scheduler/reset-circuit-breaker")
def reset_scheduler_circuit_breaker(request: Request, client: ClientAuth = Depends(require_role("ADMIN"))):
    """SCHED-001 / OPS-001: Administratively resets a tripped scheduler circuit breaker."""
    try:
        log_audit_event(
            client,
            "RESET_CIRCUIT_BREAKER",
            "scheduler",
            "SUCCESS",
            "Reset scheduler circuit breaker",
            request,
            fail_closed=True,
        )
    except Exception as e:
        logger.error(f"Audit failure aborting circuit breaker reset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Critical state mutation aborted because audit logging failed.",
        ) from e

    status_data = scheduler.reset_circuit_breaker()
    return {"status": "success", "scheduler": status_data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


@app.get("/api/v1/admin/audit-logs", tags=["Admin"])
def get_audit_logs(limit: int = 50, client: ClientAuth = Depends(require_role("ADMIN"))):
    try:
        with SessionLocal() as db:
            logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
            return {
                "status": "success",
                "logs": [
                    {
                        "id": l.id,
                        "timestamp": l.timestamp,
                        "action": l.action,
                        "resource": l.resource,
                        "status": l.status,
                        "client_role": l.client_role,
                        "ip_address": l.ip_address,
                    }
                    for l in logs
                ],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch audit logs")


@app.get("/", tags=["Root"])
def read_root():
    """Welcome endpoint."""
    return {"message": "Welcome to Nifty50 API. Visit /docs for Swagger UI."}
