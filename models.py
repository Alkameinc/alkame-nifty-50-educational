from sqlalchemy import Boolean, Column, Float, Integer, String

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
    model_id = Column(String)
    code_commit = Column(String)
    data_snapshot_id = Column(String)
    is_out_of_sample = Column(Boolean, default=False)
    prediction_key = Column(String, index=True, nullable=True)
    feature_schema_hash = Column(String, nullable=True)

    # DA-02 lineage/provenance. Nullable fields keep legacy rows readable while
    # making unknown provenance explicit rather than inventing facts.
    generation_id = Column(String, index=True)
    data_version = Column(String)
    label_definition_version = Column(String)
    entry_timestamp = Column(String)
    entry_price = Column(Float)
    target_timestamp = Column(String)
    outcome_entry_price = Column(Float)
    outcome_endpoint_price = Column(Float)
    outcome_resolution_status = Column(String, default="PENDING", server_default="PENDING")
    outcome_resolution_reason = Column(String)
    outcome_entry_timestamp = Column(String)
    outcome_target_timestamp = Column(String)
    delivery_count = Column(Integer, default=1, server_default="1")


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

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, unique=True, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    horizon = Column(String, index=True, nullable=False)
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


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    client_key_prefix = Column(String)
    client_role = Column(String)
    action = Column(String, nullable=False)
    resource = Column(String)
    status = Column(String)  # SUCCESS | FAILED | REJECTED
    details = Column(String)
    ip_address = Column(String)


class RiskState(Base):
    __tablename__ = "risk_state"

    id = Column(Integer, primary_key=True, default=1)
    enabled = Column(Boolean, nullable=False, default=False)
    reason = Column(String, default="")
    level_at_activation = Column(String, nullable=True)
    activated_at = Column(String, nullable=True)
    updated_at = Column(String, nullable=False)
    changed_by = Column(String, default="system")
    version = Column(Integer, nullable=False, default=1)

