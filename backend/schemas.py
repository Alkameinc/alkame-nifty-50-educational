from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class EventOut(BaseModel):
    event_id: str
    scope: str                     # MARKET | SECTOR | STOCK
    headline_or_label: str
    sentiment_score: Optional[float] = None
    magnitude_estimate: str


class SignalResponse(BaseModel):
    symbol: str
    timestamp: datetime
    action: str                    # BUY | SELL | HOLD
    model_predicted_class: str      # UP | DOWN | FLAT
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: Optional[float] = Field(
        default=None,
        description="Null until enough real history exists to trust this number — "
                     "the frontend MUST show a plain warning instead of a fake percentage when this is null.",
    )
    agreement_fraction: float
    downside_summary: str
    upside_summary: str
    reasoning: List[str] = []
    contributing_events: List[EventOut] = []
    global_risk_level: str
    risk_toggle_enabled: bool
    is_safe_to_trade_live: bool
    data_stale: bool
    suppressed: bool
    suppression_reasons: List[str] = []


class PredictionRecordOut(BaseModel):
    id: int
    symbol: str
    timestamp: str
    action: str
    model_predicted_class: str
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: Optional[float] = None
    agreement_fraction: float
    outcome_resolved: bool
    outcome_correct: Optional[bool] = None
    outcome_actual_class: Optional[str] = None
    resolved_at: Optional[str] = None


class HistoryResponse(BaseModel):
    symbol: str
    count: int
    predictions: List[PredictionRecordOut]


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
