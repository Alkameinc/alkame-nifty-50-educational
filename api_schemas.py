from pydantic import BaseModel, Field


class EventOut(BaseModel):
    type: str
    label: str
    sentiment: float | None = None


class HorizonSignalOut(BaseModel):
    horizon: str
    action: str
    verdict_text: str
    confidence: float | None = None
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: float | None = None
    calibration_status: str
    current_price: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None
    peak_potential_price: float | None = None
    downside_summary: str
    upside_summary: str
    events: list[EventOut]
    reasoning: list[str]


class MultiHorizonSignalResponse(BaseModel):
    symbol: str
    narrative: str
    signals: dict[str, HorizonSignalOut]


class HealthDiagnosticOut(BaseModel):
    component: str
    status: str
    message: str


class HealthResponse(BaseModel):
    overall: str
    diagnostics: list[HealthDiagnosticOut]


class SymbolsResponse(BaseModel):
    symbols: list[str]


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
    horizon: str = "INTRADAY"
    narrative: str | None = None
    dca_ladder: str | None = None


class HistoryResponse(BaseModel):
    symbol: str
    count: int
    predictions: list[PredictionRecordOut]


class OverrideRequest(BaseModel):
    symbol: str
    original_action: str
    overridden_action: str
    reason: str = Field(..., min_length=1)
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
