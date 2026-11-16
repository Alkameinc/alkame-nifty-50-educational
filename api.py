import json
import logging
import math
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from api_schemas import (
    ErrorResponse,
    FundamentalsResponse,
    HealthResponse,
    HistoryResponse,
    MarketBreadthResponse,
    MarketInsightsResponse,
    ModelValidityOut,
    MultiHorizonSignalResponse,
    NoteOut,
    NotePayload,
    NotesResponse,
    OverrideRequest,
    OverrideResponse,
    PredictionRecordOut,
    RiskReadingResponse,
    StressTestPayload,
    StressTestResponse,
    SymbolsResponse,
    TaskOut,
    TaskPayload,
    TasksResponse,
    TechnicalSnapshotOut,
)
from insights_engine import COMPLIANCE_DISCLAIMER, insights_engine
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
from human_insight_manager import HumanInsightManager
from market_data_provider import DataStatus, MarketDataResult
try:
    from model_validity import (
        VALIDITY_LIVE_ELIGIBLE,
        ModelValidityResult,
        can_serve_live_signal,
        evaluate_model_validity,
    )
except ImportError:
    VALIDITY_LIVE_ELIGIBLE = "LIVE_ELIGIBLE"

    class ModelValidityResult:
        def __init__(self, symbol="", horizon="", validity_status="EDUCATIONAL_MODE", is_live_eligible=False, reasons=None, edge_status="RESEARCH_ONLY"):
            self.symbol = symbol
            self.horizon = horizon
            self.validity_status = validity_status
            self.is_live_eligible = is_live_eligible
            self.reasons = reasons or ["Predictions disabled in educational mode."]
            self.edge_status = edge_status

    def can_serve_live_signal(health, validity):
        return False, ["Predictions disabled under SEBI educational compliance."]

    def evaluate_model_validity(symbol, horizon="INTRADAY"):
        return ModelValidityResult(symbol=symbol, horizon=horizon)

from models import AuditLog
from scheduler import LiveWorthinessSnapshot, Scheduler

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
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "connect-src 'self' http://localhost:8000 http://127.0.0.1:8000; "
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
                timestamp=datetime.now(UTC).isoformat(),
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
PREDICTIONS_TOTAL = REGISTRY._names_to_collectors.get("nifty50_predictions_total") or Counter(
    "nifty50_predictions_total",
    "Total prediction signals generated",
    ["symbol", "horizon", "action"],
)
SYSTEM_HEALTH_STATUS = REGISTRY._names_to_collectors.get("nifty50_health_status") or Gauge(
    "nifty50_health_status",
    "Current health status by component (1=OK, 0=FAIL/DEGRADED)",
    ["component"],
)
MODEL_INFERENCE_SECONDS = REGISTRY._names_to_collectors.get("nifty50_model_inference_seconds") or Histogram(
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
human_insight_manager = HumanInsightManager()


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
            "event_classifier": "Event classification pipeline is operational.",
            "corporate_events_fetcher": "Corporate event intelligence feed is operational.",
            "news_sentiment_fetcher": "News sentiment analysis stream is active.",
            "reference_level_engine": "Support/Resistance reference levels calculated.",
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
                humanized.append("The algorithmic models classify this observation as a probable downward move (DOWN).")
            elif "UP" in r:
                humanized.append("The algorithmic models classify this observation as a probable upward move (UP).")
            else:
                humanized.append("The algorithmic models classify this observation as neutral consolidation (FLAT).")
        elif r.startswith("Global risk level:"):
            if "NORMAL" in r:
                humanized.append("Systemic market risk is assessed as NORMAL; standard statistical bounds apply.")
            else:
                humanized.append(
                    "Systemic market risk is elevated; composite score discount has been applied to model confidence."
                )
        elif "No specific events" in r:
            humanized.append("No active macro or corporate events are tagged as affecting this counter.")
        elif "No calibration or edge-check data" in r or "Model leaned" in r:
            if "forced to HOLD" in r:
                humanized.append(
                    "Classification defaulted to HOLD because this model strategy has not demonstrated statistically significant alpha over the benchmark index."
                )
            else:
                humanized.append(
                    "Historical sample size for this pattern is currently insufficient for reliable calibration."
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
            if s.component.startswith("test_") or s.component.startswith("comp"):
                continue
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


@app.get("/api/v1/insights/{symbol}", response_model=MarketInsightsResponse, tags=["Market Intelligence"])
def get_insights(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    try:
        insights = insights_engine.get_stock_insights(symbol)
        return insights
    except Exception as e:
        logger.error(f"Error generating insights for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@app.get("/api/v1/market-breadth", response_model=MarketBreadthResponse, tags=["Market Intelligence"])
def get_market_breadth_route():
    try:
        return insights_engine.get_market_breadth()
    except Exception as e:
        logger.error(f"Error computing market breadth: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compute market breadth: {str(e)}")


@app.get("/api/v1/symbols", response_model=SymbolsResponse)
def get_symbols():
    return {"symbols": NIFTY50_SYMBOLS}


@app.get(
    "/api/v1/signal/{symbol}",
    response_model=MultiHorizonSignalResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@app.get(
    "/api/v1/predict/{symbol}",
    response_model=MultiHorizonSignalResponse,
    include_in_schema=False,
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
        cached_snap = (
            scheduler.get_cached_live_worthiness(symbol, hor)
            if hasattr(scheduler, "get_cached_live_worthiness")
            else None
        )
        if isinstance(cached_snap, LiveWorthinessSnapshot):
            cal = getattr(cached_snap, "calibration_result", None)
            edge = getattr(cached_snap, "edge_check_result", None)
            if (
                cal
                and edge
                and getattr(cal, "is_well_calibrated", False)
                and getattr(edge, "status", None) == "EDGE_CONFIRMED"
            ):
                validity_res = ModelValidityResult(
                    symbol=symbol,
                    horizon=hor,
                    validity_status=VALIDITY_LIVE_ELIGIBLE,
                    is_live_eligible=True,
                    reasons=[],
                )

        can_serve, serve_reasons = can_serve_live_signal(engine_health_res, validity_res)

        sig_action = sig.action
        calibrated_conf = getattr(sig, "calibrated_confidence", None)
        raw_reasoning = list(sig.reasoning)

        if not can_serve:
            sig_action = "HOLD"
            calibrated_conf = None
            raw_reasoning.extend(serve_reasons)
            if getattr(sig, "suppressed", False):
                verdict_text = (
                    "Model lean withheld: Statistical pattern does not yet meet baseline historical significance."
                )
            elif not validity_res.is_live_eligible:
                verdict_text = (
                    f"Holding: Model for {hor} is {validity_res.validity_status} and not eligible for live execution."
                )
            else:
                verdict_text = "Holding: Engine health is FAILED; live execution halted."
        else:
            if sig_action == "BUY":
                verdict_text = (
                    "Model classified this observation as a probable upward move (UP) with calibrated confidence."
                )
            elif sig_action == "SELL":
                verdict_text = "Model classified this observation as a probable downward move (DOWN)."
            else:
                if sig.suppressed:
                    verdict_text = (
                        "Model lean withheld: Statistical pattern does not yet meet baseline historical significance."
                    )
                else:
                    verdict_text = (
                        "Model classified this observation as neutral/consolidation without directional edge."
                    )

        PREDICTIONS_TOTAL.labels(symbol=symbol, horizon=hor, action=sig_action).inc()

        events = []
        for e in sig.contributing_events:
            events.append({"type": e.event_type, "label": e.headline_or_label, "sentiment": e.sentiment_score})

        raw_score = float(getattr(sig, "raw_confidence", 0.0))
        calib_status = (
            "CALIBRATED"
            if calibrated_conf is not None
            else ("INSUFFICIENT_DATA" if getattr(sig, "suppressed", False) else "UNAVAILABLE")
        )
        edge_status_val = validity_res.edge_status if validity_res else "NOT_VALIDATED"
        is_live_eligible_val = bool(can_serve and validity_res.is_live_eligible and calibrated_conf is not None)

        # SEBI Compliance: Enforce neutral educational presentation
        sig_action = "HOLD"
        verdict_text = "Educational Market Intelligence: Directional trading recommendations, targets, and stops are withheld in compliance with SEBI educational guidelines."

        all_horizons_data[hor] = {
            "horizon": hor,
            "action": sig_action,
            "verdict_text": verdict_text,
            "model_score": raw_score,
            "calibrated_probability": None,
            "calibration_status": "EDUCATIONAL_HOLD",
            "edge_status": "RESEARCH_ONLY",
            "live_eligible": False,
            "confidence": None,
            "raw_confidence": raw_score,
            "risk_adjusted_confidence": raw_score,
            "calibrated_confidence": None,
            "current_price": float(stock_df["Close"].iloc[-1]) if not stock_df.empty else None,
            "target_price": None,
            "stop_loss": None,
            "peak_potential_price": None,
            "downside_summary": sig.downside_summary,
            "upside_summary": sig.upside_summary,
            "events": events,
            "reasoning": humanize_reasoning(raw_reasoning),
            "prediction_key": getattr(sig, "prediction_key", None),
            "event_source_status": getattr(sig, "event_source_status", "UNKNOWN"),
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
                if getattr(sig, "suppressed", False):
                    verdict_text = (
                        "Model lean withheld: Statistical pattern does not yet meet baseline historical significance."
                    )
                elif not validity_res.is_live_eligible:
                    verdict_text = f"Holding: Model for {sig.horizon} is {validity_res.validity_status} and not eligible for live execution."
                else:
                    verdict_text = "Holding: Engine health is FAILED; live execution halted."
            else:
                if sig_action == "BUY":
                    verdict_text = (
                        "Model classified this observation as a probable upward move (UP) with calibrated confidence."
                    )
                elif sig_action == "SELL":
                    verdict_text = "Model classified this observation as a probable downward move (DOWN)."
                else:
                    if getattr(sig, "suppressed", False):
                        verdict_text = "Model lean withheld: Statistical pattern does not yet meet baseline historical significance."
                    else:
                        verdict_text = (
                            "Model classified this observation as neutral/consolidation without directional edge."
                        )

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

            raw_score = float(getattr(sig, "raw_confidence", 0.0))
            calib_status = (
                "CALIBRATED"
                if calibrated_conf is not None
                else ("INSUFFICIENT_DATA" if getattr(sig, "suppressed", False) else "UNAVAILABLE")
            )
            edge_status_val = validity_res.edge_status if validity_res else "NOT_VALIDATED"
            is_live_eligible_val = bool(can_serve and validity_res.is_live_eligible and calibrated_conf is not None)

            data = {
                "horizon": sig.horizon,
                "action": sig_action,
                "verdict_text": verdict_text,
                "model_score": raw_score,
                "calibrated_probability": calibrated_conf,
                "calibration_status": calib_status,
                "edge_status": edge_status_val,
                "live_eligible": is_live_eligible_val,
                "confidence": calibrated_conf,
                "raw_confidence": raw_score,
                "risk_adjusted_confidence": getattr(sig, "risk_adjusted_confidence", raw_score),
                "calibrated_confidence": calibrated_conf,
                "current_price": float(stock_df["Close"].iloc[-1]) if not stock_df.empty else None,
                "target_price": getattr(sig, "target_price", None),
                "stop_loss": getattr(sig, "stop_loss", None),
                "peak_potential_price": getattr(sig, "peak_potential_price", None),
                "downside_summary": getattr(sig, "downside_summary", ""),
                "upside_summary": getattr(sig, "upside_summary", ""),
                "events": events,
                "reasoning": humanize_reasoning(raw_reasoning),
                "event_source_status": getattr(sig, "event_source_status", "UNKNOWN"),
            }
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


import threading
import time

# --- Refresh protection state ---
_refresh_locks: dict[str, threading.Lock] = {}  # symbol -> threading.Lock
_refresh_locks_guard = threading.Lock()
_refresh_last_time: dict[str, float] = {}  # symbol -> last attempt completion (monotonic)
_REFRESH_COOLDOWN_SECONDS = 60  # Minimum seconds between refreshes for the same symbol


def _get_refresh_lock(symbol: str) -> threading.Lock:
    with _refresh_locks_guard:
        if symbol not in _refresh_locks:
            _refresh_locks[symbol] = threading.Lock()
        return _refresh_locks[symbol]


def _refresh_frame(result: object) -> pd.DataFrame | None:
    """Do not renew live-worthiness evidence from explicitly stale/unavailable bars."""
    if isinstance(result, MarketDataResult):
        if result.status not in (DataStatus.LIVE, DataStatus.CACHED_FRESH):
            return None
        data = result.data
    else:
        data = getattr(result, "df", result)
    return data if isinstance(data, pd.DataFrame) and not data.empty else None


@app.post("/api/v1/signal/{symbol}/refresh")
def refresh_backtest(
    symbol: str,
    request: Request = None,
    response: Response = Response(),
    client: ClientAuth = Depends(get_current_client),
):
    if response is None:
        response = Response()
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")

    lock = _get_refresh_lock(symbol)
    if not lock.acquire(blocking=False):
        if response is not None:
            response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return {"status": "rejected", "reason": "A refresh is already running for this symbol.", "http_status": 429}

    attempt_started = False
    try:
        last = _refresh_last_time.get(symbol)
        elapsed = time.monotonic() - last if last is not None else None
        if elapsed is not None and elapsed < _REFRESH_COOLDOWN_SECONDS:
            remaining = math.ceil(_REFRESH_COOLDOWN_SECONDS - elapsed)
            if response is not None:
                response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
            return {
                "status": "rejected",
                "reason": f"Refresh cooldown active. Try again in {remaining}s.",
                "http_status": 429,
            }

        attempt_started = True
        yf_ticker = to_yfinance_ticker(symbol)
        raw_stock = scheduler.data_fetcher.fetch_ohlcv(yf_ticker, return_metadata=True)
        raw_index = scheduler.data_fetcher.fetch_nifty_index(return_metadata=True)
        stock_df = _refresh_frame(raw_stock)
        index_df = _refresh_frame(raw_index)
        if stock_df is not None and index_df is not None:
            scheduler.refresh_live_worthiness(symbol, stock_df, index_df)
            return {"status": "success"}
        return {"error": "Failed to fetch data"}
    finally:
        # Failed work consumes resources too. Rejected requests must not extend
        # the cooldown, and elapsed wall-clock adjustments must not bypass it.
        if attempt_started:
            _refresh_last_time[symbol] = time.monotonic()
        lock.release()


@app.get("/api/v1/scalping")
def get_scalping():
    return {
        "setups": [],
        "message": "Directional scalping recommendations disabled in compliance with SEBI educational guidelines. Please refer to /api/v1/market-breadth for volume and momentum monitoring.",
        "disclaimer": COMPLIANCE_DISCLAIMER,
    }


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


@app.get("/api/v1/admin/audit-logs", tags=["Admin"])
def get_audit_logs(limit: int = 50, client: ClientAuth = Depends(require_role("ADMIN"))):
    try:
        with SessionLocal() as db:
            logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
            return {
                "status": "success",
                "logs": [
                    {
                        "id": log_entry.id,
                        "timestamp": log_entry.timestamp,
                        "action": log_entry.action,
                        "resource": log_entry.resource,
                        "status": log_entry.status,
                        "client_role": log_entry.client_role,
                        "ip_address": log_entry.ip_address,
                    }
                    for log_entry in logs
                ],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch audit logs") from e


@app.get("/api/v1/fundamentals/{symbol}", response_model=FundamentalsResponse, tags=["Fundamentals"])
def get_fundamentals(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    yf_ticker = to_yfinance_ticker(symbol)
    fetcher = scheduler.data_fetcher
    fundamentals = fetcher.fetch_stock_fundamentals(yf_ticker)
    raw_stock = fetcher.fetch_ohlcv(yf_ticker, allow_stale=True)
    stock_df = getattr(raw_stock, "df", raw_stock)
    cmp = float(stock_df["Close"].iloc[-1]) if stock_df is not None and not stock_df.empty else 0.0
    return FundamentalsResponse(symbol=symbol, cmp=cmp, fundamentals=fundamentals or {})


@app.get("/api/v1/risk/reading", response_model=RiskReadingResponse, tags=["Risk"])
def get_risk_reading():
    risk_monitor = scheduler.predictor.global_risk_monitor
    reading = risk_monitor.compute_composite_risk()
    toggle_state = risk_monitor.get_toggle_state()
    multiplier = float(risk_monitor.get_confidence_multiplier("ALL", reading))
    driver_details = getattr(reading, "driver_details", {}) or {}
    return RiskReadingResponse(
        risk_level=reading.risk_level,
        composite_score=float(getattr(reading, "composite_zscore", 0.0)),
        confidence_multiplier=multiplier,
        banner_message=reading.banner_message,
        toggle_enabled=bool(toggle_state.enabled),
        indicators=driver_details,
    )


@app.get("/api/v1/notes/{symbol}", response_model=NotesResponse, tags=["Analyst Insight"])
def get_notes(symbol: str, limit: int = 20):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    notes = human_insight_manager.get_notes(symbol, limit=limit)
    return NotesResponse(
        symbol=symbol,
        notes=[
            NoteOut(
                id=n.id,
                timestamp=n.timestamp,
                note_text=n.note_text,
                related_action=n.related_action,
                created_by=n.created_by,
            )
            for n in notes
        ],
    )


@app.post("/api/v1/notes/{symbol}", tags=["Analyst Insight"])
def add_note(symbol: str, payload: NotePayload):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    note_id = human_insight_manager.add_note(
        symbol=symbol,
        note_text=payload.note_text,
        related_action=payload.related_action,
        created_by=payload.created_by,
    )
    if note_id is None:
        raise HTTPException(status_code=500, detail="Failed to save note")
    return {"status": "success", "note_id": note_id}


@app.get("/api/v1/override/{symbol}", tags=["Analyst Insight"])
def get_override(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    overrides = human_insight_manager.get_overrides(symbol, limit=1)
    if not overrides:
        return {"symbol": symbol, "active_override": None}
    active = overrides[0]
    return {
        "symbol": symbol,
        "active_override": {
            "id": active.id,
            "timestamp": active.timestamp,
            "original_action": active.original_action,
            "overridden_action": active.overridden_action,
            "reason": active.reason,
            "created_by": active.created_by,
        },
    }


@app.post("/api/v1/override/{symbol}", response_model=OverrideResponse, tags=["Analyst Insight"])
def create_override(symbol: str, payload: OverrideRequest):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    override_id = human_insight_manager.record_override(
        symbol=symbol,
        original_action=payload.original_action,
        overridden_action=payload.overridden_action,
        reason=payload.reason,
        created_by=payload.created_by,
    )
    if override_id is None:
        raise HTTPException(status_code=500, detail="Failed to record override")
    return OverrideResponse(
        id=override_id,
        symbol=symbol,
        original_action=payload.original_action,
        overridden_action=payload.overridden_action,
        reason=payload.reason,
        created_by=payload.created_by,
    )


@app.get("/api/v1/history/{symbol}", response_model=HistoryResponse, tags=["Prediction History"])
def get_history(symbol: str, limit: int = 20):
    if symbol not in NIFTY50_SYMBOLS:
        raise HTTPException(status_code=404, detail="Invalid symbol")
    history = history_manager.get_predictions(symbol, limit=limit)
    records = []
    for rec in history:
        records.append(
            PredictionRecordOut(
                id=rec.id,
                symbol=rec.symbol,
                timestamp=rec.timestamp,
                action=rec.action,
                model_predicted_class=rec.model_predicted_class,
                raw_confidence=rec.raw_confidence,
                risk_adjusted_confidence=rec.risk_adjusted_confidence,
                calibrated_confidence=rec.calibrated_confidence,
                agreement_fraction=rec.agreement_fraction,
                outcome_resolved=rec.outcome_resolved,
                outcome_correct=rec.outcome_correct,
                outcome_actual_class=rec.outcome_actual_class,
                resolved_at=rec.resolved_at,
                horizon=getattr(rec, "horizon", "INTRADAY") or "INTRADAY",
                narrative=getattr(rec, "narrative", None),
                dca_ladder=getattr(rec, "dca_ladder", None),
                prediction_key=getattr(rec, "prediction_key", None),
                event_source_status=getattr(rec, "event_source_status", "EVENTS_AVAILABLE") or "EVENTS_AVAILABLE",
            )
        )
    return HistoryResponse(symbol=symbol, count=len(records), predictions=records)


# --- Institutional Tasks Persistence ---
TASKS_FILE = Path(__file__).resolve().parent / "data" / "institutional_tasks.json"
_tasks_lock = threading.Lock()


def _load_tasks() -> list[dict]:
    with _tasks_lock:
        if not TASKS_FILE.exists():
            defaults = [
                {
                    "id": "task-001",
                    "title": "Verify Brier calibration curve before removing PROBATION status",
                    "priority": "P0",
                    "category": "Calibration",
                    "symbol": "ADANIENT",
                    "due_date": "Today, 15:30 IST",
                    "completed": False,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                {
                    "id": "task-002",
                    "title": "Audit macro discount factor against crude Brent volatility (+3.2%)",
                    "priority": "P1",
                    "category": "Risk Audit",
                    "symbol": "GLOBAL",
                    "due_date": "Today, 17:00 IST",
                    "completed": False,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                {
                    "id": "task-003",
                    "title": "Review dual-boundary stop-loss threshold on NEXT_DAY horizon for RELIANCE",
                    "priority": "P1",
                    "category": "Execution",
                    "symbol": "RELIANCE",
                    "due_date": "Tomorrow, 09:15 IST",
                    "completed": True,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                {
                    "id": "task-004",
                    "title": "Confirm zero lookahead leakage on board meeting corporate event stream",
                    "priority": "P2",
                    "category": "Compliance",
                    "symbol": "TCS",
                    "due_date": "End of Week",
                    "completed": False,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                {
                    "id": "task-005",
                    "title": "Inspect 16-component subsystem telemetry health logs for SQLite WAL journal",
                    "priority": "P2",
                    "category": "Risk Audit",
                    "symbol": "GLOBAL",
                    "due_date": "End of Week",
                    "completed": True,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            ]
            try:
                TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
                TASKS_FILE.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
                return defaults
            except Exception as e:
                logger.warning(f"Failed to create default tasks file: {e}")
                return defaults
        try:
            return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load tasks file: {e}")
            return []


def _save_tasks(tasks: list[dict]) -> None:
    with _tasks_lock:
        try:
            TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            TASKS_FILE.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to write tasks file: {e}")


@app.get("/api/v1/tasks/{symbol}", response_model=TasksResponse, tags=["Analyst Tasks"])
def get_tasks(symbol: str):
    all_tasks = _load_tasks()
    if symbol.upper() == "GLOBAL":
        filtered = all_tasks
    else:
        filtered = [t for t in all_tasks if t.get("symbol", "").upper() in (symbol.upper(), "GLOBAL")]
    return TasksResponse(symbol=symbol, tasks=[TaskOut(**t) for t in filtered])


@app.post("/api/v1/tasks/{symbol}", response_model=TaskOut, tags=["Analyst Tasks"])
def add_task(symbol: str, payload: TaskPayload):
    tasks = _load_tasks()
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    new_task = {
        "id": task_id,
        "title": payload.title,
        "priority": payload.priority,
        "category": payload.category,
        "symbol": symbol.upper(),
        "due_date": payload.due_date,
        "completed": payload.completed,
        "created_at": datetime.now(UTC).isoformat(),
    }
    tasks.insert(0, new_task)
    _save_tasks(tasks)
    return TaskOut(**new_task)


@app.put("/api/v1/tasks/{task_id}/toggle", response_model=TaskOut, tags=["Analyst Tasks"])
def toggle_task(task_id: str):
    tasks = _load_tasks()
    target = None
    for t in tasks:
        if t.get("id") == task_id:
            t["completed"] = not t.get("completed", False)
            target = t
            break
    if not target:
        raise HTTPException(status_code=404, detail="Task not found")
    _save_tasks(tasks)
    return TaskOut(**target)


@app.delete("/api/v1/tasks/{task_id}", tags=["Analyst Tasks"])
def delete_task(task_id: str):
    tasks = _load_tasks()
    new_tasks = [t for t in tasks if t.get("id") != task_id]
    if len(new_tasks) == len(tasks):
        raise HTTPException(status_code=404, detail="Task not found")
    _save_tasks(new_tasks)
    return {"status": "success", "deleted_id": task_id}


@app.post("/api/v1/risk/stress-test", response_model=StressTestResponse, tags=["Risk & Governance"])
def stress_test(payload: StressTestPayload):
    symbol = payload.symbol.upper()
    cmp_val = None
    try:
        yf_ticker = to_yfinance_ticker(symbol)
        raw = scheduler.data_fetcher.fetch_ohlcv(yf_ticker, allow_stale=True)
        df = getattr(raw, "df", raw)
        if df is not None and hasattr(df, "empty") and not df.empty:
            cmp_val = float(df["Close"].iloc[-1])
    except Exception:
        cmp_val = None

    if not cmp_val or cmp_val <= 0:
        FALLBACK_PRICES = {
            "ADANIENT": 2701.00,
            "TCS": 3920.00,
            "ICICIBANK": 1245.00,
            "HDFCBANK": 1680.20,
            "RELIANCE": 2985.40,
            "INFY": 1890.50,
            "MARUTI": 12450.00,
            "TATASTEEL": 158.00,
            "LT": 3580.00,
            "SBIN": 815.00,
            "BAJAJ-AUTO": 9800.00,
            "AXISBANK": 1195.00,
            "KOTAKBANK": 1820.00,
            "BHARTIARTL": 1680.00,
            "ITC": 505.00,
            "HINDUNILVR": 2780.00,
            "TITAN": 3680.00,
            "SUNPHARMA": 1860.00,
            "NESTLEIND": 2480.00,
            "CIPLA": 1540.00,
        }
        cmp_val = FALLBACK_PRICES.get(symbol, 2500.0)

    is_energy = symbol in ("ADANIENT", "RELIANCE", "ONGC", "BPCL", "COALINDIA")
    is_it = symbol in ("TCS", "INFY", "WIPRO", "HCLTECH", "TECHM")
    is_bank = symbol in ("HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK")

    crude_beta = 0.45 if is_energy else (-0.25 if is_bank else -0.10)
    vix_beta = -0.35 if is_bank else -0.20
    inr_beta = 0.40 if is_it else -0.15

    pnl_pct = (
        (payload.crude_shock_pct * crude_beta) + (payload.vix_shock_pct * vix_beta) + (payload.inr_shock_pct * inr_beta)
    )

    daily_vol = 0.018
    var_95 = round(1.645 * daily_vol * 100, 2)
    var_99 = round(2.326 * daily_vol * 100, 2)

    estimated_cmp = round(float(cmp_val) * (1 + (pnl_pct / 100.0)), 2)
    regime = "ELEVATED_STRESS" if abs(pnl_pct) > 2.5 else ("MODERATE_STRESS" if abs(pnl_pct) > 1.0 else "BENIGN")

    return StressTestResponse(
        symbol=symbol,
        base_cmp=float(cmp_val),
        estimated_cmp=float(estimated_cmp),
        estimated_pnl_pct=round(float(pnl_pct), 3),
        var_95_pct=var_95,
        var_99_pct=var_99,
        risk_regime=regime,
        factor_sensitivities={
            "crude_beta": crude_beta,
            "vix_beta": vix_beta,
            "inr_beta": inr_beta,
        },
    )


@app.get("/dashboard", response_class=HTMLResponse, tags=["Frontend"])
def serve_dashboard():
    dashboard_path = Path(__file__).resolve().parent / "alkame_dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard frontend not found")
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))


@app.get("/", tags=["Root"])
def read_root():
    """Welcome endpoint."""
    return {"message": "Welcome to Nifty50 API. Visit /docs for Swagger UI, or /dashboard for the frontend."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
