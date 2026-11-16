"""
Educational Data Contracts & Signal Specifications for Alkame-Nifty50.

In compliance with SEBI (Research Analysts) Regulations, 2014 and SEBI (Investment
Advisers) Regulations, 2013, Alkame operates in Educational & Informational Market
Intelligence Mode. All directional trading signals, target prices, and stop losses
are withheld.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any
import pandas as pd

ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"

CLASS_TO_ACTION = {
    "UP": ACTION_BUY,
    "DOWN": ACTION_SELL,
    "FLAT": ACTION_HOLD,
}


@dataclass
class PredictionSignal:
    symbol: str
    timestamp: datetime
    horizon: str
    action: str = ACTION_HOLD
    model_predicted_class: str = "FLAT"
    model_version: str = "1.0-educational"
    feature_version: str = "1.0-educational"
    raw_confidence: float = 0.5
    risk_adjusted_confidence: float = 0.5
    calibrated_confidence: float | None = None
    agreement_fraction: float = 1.0
    downside_summary: str = "Descriptive technical reference: Camarilla L3 floor."
    upside_summary: str = "Descriptive technical reference: Camarilla H3 ceiling."
    reasoning: list[str] = field(default_factory=lambda: ["Informational market research: no directional advice provided."])
    contributing_events: list = field(default_factory=list)
    global_risk_level: str = "NORMAL"
    risk_toggle_enabled: bool = False
    is_safe_to_trade_live: bool = False
    data_stale: bool = False
    suppressed: bool = True
    suppression_reasons: list[str] = field(default_factory=lambda: ["SEBI educational compliance mode"])
    target_price: float | None = None
    stop_loss: float | None = None
    peak_potential_price: float | None = None
    model_id: str | None = None
    code_commit: str | None = None
    data_snapshot_id: str | None = None
    prediction_key: str | None = None
    feature_schema_hash: str | None = None
    entry_price: float | None = None
    data_version: str = "UNKNOWN"
    label_definition_version: str = "direction-adaptive-deadband-v1"
    event_source_status: str = "EVENTS_AVAILABLE"
    config_hash: str | None = None
    reproducibility_bundle: dict | None = None

    def __post_init__(self):
        if self.timestamp is not None:
            if isinstance(self.timestamp, str):
                try:
                    ts = pd.to_datetime(self.timestamp)
                    if ts.tz is None:
                        ts = ts.tz_localize("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    self.timestamp = ts.to_pydatetime()
                except Exception:
                    pass
            elif hasattr(self.timestamp, "tzinfo") and self.timestamp.tzinfo is None:
                self.timestamp = self.timestamp.replace(tzinfo=UTC)


@dataclass
class MultiHorizonSignal:
    symbol: str
    timestamp: datetime
    signals: dict[str, PredictionSignal] = field(default_factory=dict)
    primary_action: str = ACTION_HOLD
    primary_horizon: str = "INTRADAY"
    reasoning: list[str] = field(default_factory=lambda: ["Educational market intelligence summary"])
    prediction_key: str | None = None

    def __post_init__(self):
        if self.timestamp is not None:
            if isinstance(self.timestamp, str):
                try:
                    ts = pd.to_datetime(self.timestamp)
                    if ts.tz is None:
                        ts = ts.tz_localize("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    self.timestamp = ts.to_pydatetime()
                except Exception:
                    pass
            elif hasattr(self.timestamp, "tzinfo") and self.timestamp.tzinfo is None:
                self.timestamp = self.timestamp.replace(tzinfo=UTC)


@dataclass
class PredictionContext:
    symbol: str
    timestamp: datetime
    market_data: pd.DataFrame
    index_data: pd.DataFrame | None = None
    macro_events: list | None = None
    corporate_events: list[dict] | None = None
    news_events: list[dict] | None = None
    data_status: str = "LIVE"
    event_status: str = "EVENTS_AVAILABLE"
    calibration_results: dict[str, Any] | None = None
    edge_check_results: dict[str, Any] | None = None
    as_of: datetime | None = None


@dataclass
class CalibrationResult:
    brier_score: float = 0.2
    is_well_calibrated: bool = False
    expected_calibration_error: float = 0.05
    status: str = "EDUCATIONAL_HOLD"


@dataclass
class EdgeCheckResult:
    alpha_pct: float = 0.0
    p_value: float = 1.0
    status: str = "RESEARCH_ONLY"
