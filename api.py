from fastapi import FastAPI, Query, Security, HTTPException, status, Depends, Response
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import logging
from config import (
    NIFTY50_SYMBOLS, HORIZON_INTRADAY, ALL_HORIZONS, to_yfinance_ticker,
    API_AUTH_ENABLED, API_KEYS_ROLE_MAP, CORS_ALLOWED_ORIGINS, CORS_ALLOW_CREDENTIALS,
    DEFAULT_DEV_API_KEY
)
from scheduler import Scheduler
from scalping import ScalpingEngine
from history_manager import HistoryManager
from health_monitor import registry as health_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alkame Nifty50 API", version="1.0.0")

# Security dependencies (P0-001)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)

class ClientAuth:
    def __init__(self, key: str, role: str):
        self.key = key
        self.role = role

def get_current_client(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth)
) -> ClientAuth:
    if not API_AUTH_ENABLED:
        return ClientAuth(key="disabled", role="ADMIN")
    
    provided_key = None
    if api_key:
        provided_key = api_key
    elif bearer and bearer.credentials:
        provided_key = bearer.credentials
        
    if not provided_key or provided_key not in API_KEYS_ROLE_MAP:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide valid key via 'X-API-Key' header or 'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    role = API_KEYS_ROLE_MAP[provided_key]
    return ClientAuth(key=provided_key, role=role)

def require_role(required_role: str):
    def role_checker(client: ClientAuth = Depends(get_current_client)) -> ClientAuth:
        if required_role == "ADMIN" and client.role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Action requires '{required_role}' privilege. Client has role '{client.role}'."
            )
        return client
    return role_checker

# Strict CORS without wildcard credentials (P0-002)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = Scheduler()
history_manager = HistoryManager()
scalping_engine = ScalpingEngine(scheduler.predictor, scheduler.data_fetcher, scheduler.predictor.feature_engineer)

def translate_health_message(component: str, status: str) -> str:
    if status == "OK":
        messages = {
            "history_manager": "Records verified and synced successfully.",
            "data_fetcher": "Live market data is streaming perfectly.",
            "runtime_validator": "All strict safety rules are currently passing.",
            "global_risk_monitor": "Global conditions are stable and safe for trading.",
            "human_insight_manager": "Manual override controls are online.",
            "ensemble_manager": "AI models are loaded and ready to analyze.",
            "predictor": "Signal engine is functioning flawlessly."
        }
        return messages.get(component, "All system checks passed.")
    elif status == "DEGRADED":
        return "Experiencing slight delays in data, but recovering."
    return "Currently offline or unresponsive, checking connections."

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
                humanized.append("Global risk is elevated, so we've lowered our confidence in this prediction to keep you safe.")
        elif "No specific events" in r:
            humanized.append("We haven't detected any major breaking news or events affecting this stock right now.")
        elif "No calibration or edge-check data" in r or "Model leaned" in r:
            if "forced to HOLD" in r:
                humanized.append("We forced a HOLD because this specific strategy hasn't proven itself against the NIFTY baseline yet. We prioritize safety over unproven trades.")
            else:
                humanized.append("We don't have enough historical proof that this pattern works yet, so we are staying cautious.")
        else:
            humanized.append(r)
    return list(dict.fromkeys(humanized)) # remove duplicates

@app.get("/api/health")
def get_health():
    overall = health_registry.get_overall_status()
    statuses = health_registry.get_status()
    
    diagnostic = []
    if statuses:
        for s in statuses:
            diagnostic.append({
                "component": s.component.replace('_', ' ').title(),
                "status": s.status,
                "message": translate_health_message(s.component, s.status)
            })
    return {"overall": overall, "diagnostics": diagnostic}

@app.get("/api/symbols")
def get_symbols():
    return {"symbols": NIFTY50_SYMBOLS}

@app.get("/api/signal/{symbol}")
def get_signal(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        return {"error": "Invalid symbol"}
    
    yf_ticker = to_yfinance_ticker(symbol)
    stock_df = scheduler.data_fetcher.fetch_ohlcv(yf_ticker)
    index_df = scheduler.data_fetcher.fetch_nifty_index()
    
    if stock_df is None or stock_df.empty:
        return {"error": f"Could not fetch data for {symbol}"}
        
    scheduler.resolve_pending_outcomes(symbol, stock_df)
    multi_signal = scheduler.run_one_cycle_for_symbol(symbol, stock_df, index_df, macro_events=[], corporate_events=[], news_articles=[])
    if multi_signal is None or not getattr(multi_signal, 'signals', None):
        return {"error": f"No signal available for {symbol}"}

    # We can fetch narrative once
    recent_records = history_manager.get_predictions(symbol, limit=1)
    narrative = recent_records[0].narrative if recent_records and getattr(recent_records[0], 'narrative', None) else "No narrative available."

    all_horizons_data = {}
    for hor, sig in multi_signal.signals.items():
        if sig.action == "BUY":
            verdict_text = "Strong opportunity identified. Proceed with entry according to your risk parameters."
        elif sig.action == "SELL":
            verdict_text = "Warning: Downward pressure detected. Consider hedging or reducing exposure."
        else:
            if sig.suppressed:
                verdict_text = "Holding back: We don't have enough historical proof that this pattern works yet."
            else:
                verdict_text = "No clear edge detected. Better to stay out and wait for a higher-probability setup."

        events = []
        for e in sig.contributing_events:
            events.append({
                "type": e.event_type,
                "label": e.headline_or_label,
                "sentiment": e.sentiment_score
            })
            
        all_horizons_data[hor] = {
            "horizon": hor,
            "action": sig.action,
            "verdict_text": verdict_text,
            "confidence": getattr(sig, 'calibrated_confidence', None),
            "raw_confidence": getattr(sig, 'raw_confidence', 0.0),
            "risk_adjusted_confidence": getattr(sig, 'risk_adjusted_confidence', 0.0),
            "calibrated_confidence": getattr(sig, 'calibrated_confidence', None),
            "calibration_status": "VALID" if getattr(sig, 'calibrated_confidence', None) is not None else "UNAVAILABLE",
            "current_price": float(stock_df["Close"].iloc[-1]) if not stock_df.empty else None,
            "target_price": sig.target_price,
            "stop_loss": sig.stop_loss,
            "peak_potential_price": getattr(sig, 'peak_potential_price', None),
            "downside_summary": sig.downside_summary,
            "upside_summary": sig.upside_summary,
            "events": events,
            "reasoning": humanize_reasoning(sig.reasoning)
        }

    return {
        "symbol": symbol,
        "narrative": narrative,
        "signals": all_horizons_data
    }

@app.get("/api/signal/stream/{symbol}")
def stream_signal(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        return {"error": "Invalid symbol"}
    
    yf_ticker = to_yfinance_ticker(symbol)
    stock_df = scheduler.data_fetcher.fetch_ohlcv(yf_ticker)
    index_df = scheduler.data_fetcher.fetch_nifty_index()
    
    if stock_df is None or stock_df.empty:
        return {"error": f"Could not fetch data for {symbol}"}
        
    scheduler.resolve_pending_outcomes(symbol, stock_df)
    
    def generate():
        stream = scheduler.run_cycle_stream_for_symbol(symbol, stock_df, index_df, macro_events=[], corporate_events=[], news_articles=[])
        for sig in stream:
            if sig.action == "BUY":
                verdict_text = "Strong opportunity identified. Proceed with entry according to your risk parameters."
            elif sig.action == "SELL":
                verdict_text = "Warning: Downward pressure detected. Consider hedging or reducing exposure."
            else:
                if getattr(sig, 'suppressed', False):
                    verdict_text = "Holding back: We don't have enough historical proof that this pattern works yet."
                else:
                    verdict_text = "No clear edge detected. Better to stay out and wait for a higher-probability setup."

            events = []
            if getattr(sig, 'contributing_events', None):
                for e in sig.contributing_events:
                    events.append({
                        "type": getattr(e, 'event_type', ''),
                        "label": getattr(e, 'headline_or_label', ''),
                        "sentiment": getattr(e, 'sentiment_score', 0.0)
                    })
                
            data = {
                "horizon": sig.horizon,
                "action": sig.action,
                "verdict_text": verdict_text,
                "confidence": getattr(sig, 'calibrated_confidence', None),
                "raw_confidence": getattr(sig, 'raw_confidence', 0.0),
                "risk_adjusted_confidence": getattr(sig, 'risk_adjusted_confidence', getattr(sig, 'raw_confidence', 0.0)),
                "calibrated_confidence": getattr(sig, 'calibrated_confidence', None),
                "calibration_status": "VALID" if getattr(sig, 'calibrated_confidence', None) is not None else "UNAVAILABLE",
                "current_price": float(stock_df["Close"].iloc[-1]) if not stock_df.empty else None,
                "target_price": getattr(sig, 'target_price', None),
                "stop_loss": getattr(sig, 'stop_loss', None),
                "peak_potential_price": getattr(sig, 'peak_potential_price', None),
                "downside_summary": getattr(sig, 'downside_summary', ""),
                "upside_summary": getattr(sig, 'upside_summary', ""),
                "events": events,
                "reasoning": humanize_reasoning(getattr(sig, 'reasoning', []))
            }
            yield f"data: {json.dumps(data)}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")


import time
import threading

# --- Refresh protection state ---
_refresh_locks: dict = {}          # symbol -> threading.Lock
_refresh_last_time: dict = {}      # symbol -> float (epoch)
_REFRESH_COOLDOWN_SECONDS = 60     # Minimum seconds between refreshes for the same symbol

def _get_refresh_lock(symbol: str) -> threading.Lock:
    if symbol not in _refresh_locks:
        _refresh_locks[symbol] = threading.Lock()
    return _refresh_locks[symbol]


@app.post("/api/signal/{symbol}/refresh")
def refresh_backtest(
    symbol: str,
    response: Response = Response(),
    client: ClientAuth = Depends(get_current_client)
):
    if symbol not in NIFTY50_SYMBOLS:
        return {"error": "Invalid symbol"}

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
            return {"status": "rejected", "reason": f"Refresh cooldown active. Try again in {remaining}s.", "http_status": 429}

        yf_ticker = to_yfinance_ticker(symbol)
        stock_df = scheduler.data_fetcher.fetch_ohlcv(yf_ticker)
        index_df = scheduler.data_fetcher.fetch_nifty_index()
        if stock_df is not None and index_df is not None:
            scheduler.refresh_live_worthiness(symbol, stock_df, index_df)
            _refresh_last_time[symbol] = time.time()
            return {"status": "success"}
        return {"error": "Failed to fetch data"}
    finally:
        lock.release()

@app.get("/api/scalping")
def get_scalping():
    setups = scalping_engine.find_opportunities(limit=5)
    result = []
    for s in setups:
        result.append({
            "symbol": s.symbol,
            "action": s.action,
            "entry": s.entry_price,
            "target": s.target_price,
            "stop": s.stop_loss,
            "confidence": s.confidence,
            "rr": s.risk_reward_ratio
        })
    return {"setups": result}

from config import HORIZON_CONFIG

@app.get("/api/chart/{symbol}")
def get_chart(symbol: str, horizon: str = "INTRADAY"):
    if symbol not in NIFTY50_SYMBOLS:
        return {"error": "Invalid symbol"}
    yf_ticker = to_yfinance_ticker(symbol)
    
    cfg = HORIZON_CONFIG.get(horizon, HORIZON_CONFIG["INTRADAY"])
    interval = cfg["bar_interval"]
    period = cfg["history_period"]
    
    stock_df = scheduler.data_fetcher.fetch_ohlcv(yf_ticker, interval=interval, period=period)
    if stock_df is None or stock_df.empty:
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
            
        chart_data.append({
            "time": time_str,
            "close": float(row["Close"]),
            "volume": int(row["Volume"]) if "Volume" in row else 0
        })
    return {"chart": chart_data}

@app.get("/api/risk/toggle")
def get_risk_toggle():
    state = scheduler.predictor.global_risk_monitor.get_toggle_state()
    return {
        "enabled": state.enabled,
        "reason": state.reason,
        "level_at_activation": state.level_at_activation,
        "activated_at": state.activated_at
    }

@app.post("/api/risk/toggle")
def set_risk_toggle(enabled: bool, client: ClientAuth = Depends(require_role("ADMIN"))):
    # In a real app, you might take the reason from the request body.
    # For now, we'll just toggle it with a generic reason.
    reason = f"Toggled by {client.role} ({client.key[:6]}...) via API"
    state = scheduler.predictor.global_risk_monitor.set_toggle(enabled, reason)
    return {"status": "success", "enabled": state.enabled}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
