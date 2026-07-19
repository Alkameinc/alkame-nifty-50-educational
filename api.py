from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import logging
from config import NIFTY50_SYMBOLS, HORIZON_INTRADAY, ALL_HORIZONS, to_yfinance_ticker
from scheduler import Scheduler
from scalping import ScalpingEngine
from history_manager import HistoryManager
from health_monitor import registry as health_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alkame Nifty50 API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
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
            "confidence": sig.risk_adjusted_confidence,
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
                "confidence": getattr(sig, 'risk_adjusted_confidence', getattr(sig, 'raw_confidence', 0.0)),
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


@app.post("/api/signal/{symbol}/refresh")
def refresh_backtest(symbol: str):
    if symbol not in NIFTY50_SYMBOLS:
        return {"error": "Invalid symbol"}
    yf_ticker = to_yfinance_ticker(symbol)
    stock_df = scheduler.data_fetcher.fetch_ohlcv(yf_ticker)
    index_df = scheduler.data_fetcher.fetch_nifty_index()
    if stock_df is not None and index_df is not None:
        scheduler.refresh_live_worthiness(symbol, stock_df, index_df)
        return {"status": "success"}
    return {"error": "Failed to fetch data"}

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
def set_risk_toggle(enabled: bool):
    # In a real app, you might take the reason from the request body.
    # For now, we'll just toggle it with a generic reason.
    reason = "User toggled via dashboard" if enabled else "User disabled via dashboard"
    state = scheduler.predictor.global_risk_monitor.set_toggle(enabled, reason)
    return {"status": "success", "enabled": state.enabled}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
