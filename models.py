from sqlalchemy import Column, Integer, String, Float, Boolean
from database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(String, nullable=False)
    action = Column(String, nullable=False)
    model_predicted_class = Column(String, nullable=False)
    raw_confidence = Column(Float, nullable=False)
    risk_adjusted_confidence = Column(Float, nullable=False)
    calibrated_confidence = Column(Float)
    agreement_fraction = Column(Float, nullable=False)
    downside_summary = Column(String)
    upside_summary = Column(String)
    reasoning = Column(String)
    global_risk_level = Column(String)
    risk_toggle_enabled = Column(Boolean)
    is_safe_to_trade_live = Column(Boolean)
    data_stale = Column(Boolean)
    suppressed = Column(Boolean)
    suppression_reasons = Column(String)
    outcome_resolved = Column(Boolean, default=False)
    outcome_correct = Column(Boolean)
    outcome_actual_class = Column(String)
    resolved_at = Column(String)
    horizon = Column(String, default="INTRADAY")
    narrative = Column(String)
    dca_ladder = Column(String)
    model_version = Column(String, default="UNKNOWN")
    feature_version = Column(String, default="UNKNOWN")
    is_out_of_sample = Column(Boolean, default=False)


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, index=True, nullable=False)
    source = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    affected_tickers = Column(String, nullable=False)
    sector = Column(String)
    confidence_in_scope = Column(Float, nullable=False)
    headline_or_label = Column(String, nullable=False)
    sentiment_score = Column(Float)
    magnitude_estimate = Column(String, nullable=False)


class HealthStatus(Base):
    __tablename__ = "health_status"

    component = Column(String, primary_key=True, index=True)
    status = Column(String)
    last_success_at = Column(String)
    last_error = Column(String)
    last_error_at = Column(String)
    consecutive_failures = Column(Integer)
    detail = Column(String)


class BacktestMetric(Base):
    __tablename__ = "backtest_metrics"

    symbol = Column(String, primary_key=True)
    horizon = Column(String, primary_key=True)
    strategy_cumulative_return_pct = Column(Float)
    baseline_cumulative_return_pct = Column(Float)
    alpha_pct = Column(Float)
    edge_check_status = Column(String)
    calibration_status = Column(String)
    calibration_ece = Column(Float)
    is_live_worthy = Column(Boolean)
    updated_at = Column(String)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_version = Column(String, primary_key=True)
    symbol = Column(String, primary_key=True)
    horizon = Column(String, primary_key=True)
    created_at = Column(String)
    git_commit = Column(String)
    python_version = Column(String)
    os_platform = Column(String)
    sklearn_version = Column(String)
    pandas_version = Column(String)
    joblib_version = Column(String)
    features_hash = Column(String)
    artifact_path = Column(String)
    is_active = Column(Boolean)
