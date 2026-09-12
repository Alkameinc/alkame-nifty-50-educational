from datetime import datetime

from pydantic import BaseModel, Field


class EventOut(BaseModel):
    event_id: str
    scope: str  # MARKET | SECTOR | STOCK
    headline_or_label: str
    sentiment_score: float | None = None
    magnitude_estimate: str


class SignalResponse(BaseModel):
    symbol: str
    timestamp: datetime
    action: str  # BUY | SELL | HOLD
    model_predicted_class: str  # UP | DOWN | FLAT
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: float | None = Field(
        default=None,
        description="Null until enough real history exists to trust this number — "
        "the frontend MUST show a plain warning instead of a fake percentage when this is null.",
    )
    agreement_fraction: float
    downside_summary: str
    upside_summary: str
    reasoning: list[str] = []
    contributing_events: list[EventOut] = []
    global_risk_level: str
    risk_toggle_enabled: bool
    is_safe_to_trade_live: bool
    data_stale: bool
    suppressed: bool
    suppression_reasons: list[str] = []


class PredictionRecordOut(BaseModel):
    id: int
    symbol: str
    timestamp: str
    action: str
    model_predicted_class: str
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: float | None = None
    agreement_fraction: float
    outcome_resolved: bool
    outcome_correct: bool | None = None
    outcome_actual_class: str | None = None
    resolved_at: str | None = None


class HistoryResponse(BaseModel):
    symbol: str
    count: int
    predictions: list[PredictionRecordOut]


class OverrideRequest(BaseModel):
    """What the frontend sends when a trader overrides a signal."""

    symbol: str
    original_action: str
    overridden_action: str
    reason: str = Field(..., min_length=1, description="Mandatory — an empty reason is rejected.")
    created_by: str = "default_trader"


class OverrideResponse(BaseModel):
    id: int
    symbol: str
    original_action: str
    overridden_action: str
    reason: str
    created_by: str


class ErrorResponse(BaseModel):
    detail: str
