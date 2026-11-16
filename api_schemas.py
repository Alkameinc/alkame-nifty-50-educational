from typing import Any

from pydantic import BaseModel, Field


class EventOut(BaseModel):
    type: str
    label: str
    sentiment: float | None = None


class HorizonSignalOut(BaseModel):
    horizon: str
    action: str
    verdict_text: str
    model_score: float = 0.0
    calibrated_probability: float | None = None
    calibration_status: str = "UNAVAILABLE"
    edge_status: str = "NOT_VALIDATED"
    live_eligible: bool = False
    confidence: float | None = Field(None, deprecated=True)
    raw_confidence: float = 0.0
    risk_adjusted_confidence: float = 0.0
    calibrated_confidence: float | None = None
    current_price: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None
    peak_potential_price: float | None = None
    downside_summary: str = ""
    upside_summary: str = ""
    events: list[EventOut] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    prediction_key: str | None = None
    event_source_status: str = "EVENTS_AVAILABLE"


class MultiHorizonSignalResponse(BaseModel):
    symbol: str
    narrative: str
    signals: dict[str, HorizonSignalOut]
    prediction_key: str | None = None


class HealthDiagnosticOut(BaseModel):
    component: str
    status: str
    message: str


class HealthResponse(BaseModel):
    overall: str
    diagnostics: list[HealthDiagnosticOut]
    engine_health: str = "HEALTHY"
    checks: dict[str, str] = Field(default_factory=dict)


class ModelValidityOut(BaseModel):
    symbol: str
    horizon: str
    validity_status: str
    is_live_eligible: bool
    model_version: str | None = None
    trained_at: str | None = None
    age_days: float | None = None
    calibration_status: str
    edge_status: str
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    prediction_key: str | None = None
    event_source_status: str = "EVENTS_AVAILABLE"


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


class NotePayload(BaseModel):
    note_text: str = Field(..., min_length=1)
    related_action: str | None = None
    created_by: str = "default_trader"


class NoteOut(BaseModel):
    id: int
    timestamp: str
    note_text: str
    related_action: str | None = None
    created_by: str


class NotesResponse(BaseModel):
    symbol: str
    notes: list[NoteOut]


class FundamentalsResponse(BaseModel):
    symbol: str
    cmp: float
    fundamentals: dict[str, Any] = Field(default_factory=dict)


class RiskReadingResponse(BaseModel):
    risk_level: str
    composite_score: float
    confidence_multiplier: float
    banner_message: str
    toggle_enabled: bool
    indicators: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


class TaskPayload(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = "P1"
    category: str = "Risk Audit"
    symbol: str = "GLOBAL"
    due_date: str | None = None
    completed: bool = False


class TaskOut(BaseModel):
    id: str
    title: str
    priority: str
    category: str
    symbol: str
    due_date: str | None = None
    completed: bool
    created_at: str


class TasksResponse(BaseModel):
    symbol: str
    tasks: list[TaskOut]


class StressTestPayload(BaseModel):
    symbol: str
    crude_shock_pct: float = 0.0
    vix_shock_pct: float = 0.0
    inr_shock_pct: float = 0.0


class StressTestResponse(BaseModel):
    symbol: str
    base_cmp: float
    estimated_cmp: float
    estimated_pnl_pct: float
    var_95_pct: float
    var_99_pct: float
    risk_regime: str
    factor_sensitivities: dict[str, float]


class TechnicalSnapshotOut(BaseModel):
    symbol: str
    cmp: float
    rsi_14: float
    rsi_status: str
    ema_20: float | None = None
    ema_50: float | None = None
    sma_200: float | None = None
    trend_posture: str
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    bollinger_pct_b: float | None = None
    bollinger_bandwidth: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    volume_latest: int
    volume_20d_sma: float | None = None
    volume_ratio: float | None = None
    volume_surge: bool
    vwap: float | None = None
    vwap_relation: str
    pivots_camarilla: dict[str, float] = Field(default_factory=dict)
    pivots_standard: dict[str, float] = Field(default_factory=dict)
    prior_day_high: float | None = None
    prior_day_low: float | None = None
    prior_day_close: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    pct_from_52w_high: float | None = None
    pct_from_52w_low: float | None = None


class MarketInsightsResponse(BaseModel):
    symbol: str
    timestamp: str
    cmp: float
    day_change_pct: float
    technical: TechnicalSnapshotOut
    valuation: dict[str, Any]
    corporate_events: list[dict[str, Any]]
    macro_context: dict[str, Any]
    news_sentiment: dict[str, Any]
    disclaimer: str


class ConstituentBreadthItem(BaseModel):
    symbol: str
    cmp: float
    day_change_pct: float
    volume: int
    volume_ratio: float
    volume_surge: bool
    rsi_14: float
    pct_from_52w_high: float
    sector: str


class MarketBreadthResponse(BaseModel):
    timestamp: str
    total_constituents: int
    advances: int
    declines: int
    unchanged: int
    advance_decline_ratio: float
    volume_surges: list[ConstituentBreadthItem]
    top_gainers: list[ConstituentBreadthItem]
    top_losers: list[ConstituentBreadthItem]
    all_constituents: list[ConstituentBreadthItem]
    disclaimer: str
