import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import configure_logging
from backend.routers import signals, history, overrides

configure_logging(log_filename="backend_api.log")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Alkame-Nifty50 API",
    description=(
        "Decision-support API for NIFTY 50 signals. Human-in-the-loop, calibration-gated, "
        "risk-first — see ALKAME_NIFTY50_CHARTER.md for the full methodology. "
        "This API never places trades and does not constitute investment advice."
    ),
    version="0.1.0",
)

# CORS: allows the React frontend (running on a different port, e.g. 5173) to call this API.
# In local development this is safe to leave open. Before deploying anywhere real, tighten
# allow_origins to your actual frontend's domain — see Part 7.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router)
app.include_router(history.router)
app.include_router(overrides.router)


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is up — hit this first when debugging."""
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Alkame-Nifty50 API is running. See /docs for interactive documentation.",
    }