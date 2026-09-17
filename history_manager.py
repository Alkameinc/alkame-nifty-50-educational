import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import sessionmaker

from config import DB_PATH, configure_logging, ensure_directories

TICKER_DELIMITER = ","
from database import SessionLocal
from event_classifier import Event
from health_monitor import registry as health_registry
from models import BacktestMetric as DBBacktestMetric
from models import Event as DBEvent
from models import Prediction as DBPrediction
from predictor import PredictionSignal
from storage_reliability import (
    atomic_transaction,
    create_reliable_engine,
    run_storage_recovery,
    safe_float,
    safe_iso_timestamp,
    safe_json_loads,
    with_db_retry,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    id: int
    symbol: str
    horizon: str
    model_version: str
    feature_version: str
    model_id: str | None
    code_commit: str | None
    data_snapshot_id: str | None
    narrative: str
    dca_ladder: str
    timestamp: str
    action: str
    model_predicted_class: str
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: float | None
    agreement_fraction: float
    outcome_resolved: bool
    outcome_correct: bool | None
    outcome_actual_class: str | None
    resolved_at: str | None
    is_out_of_sample: bool = False
    prediction_key: str | None = None
    feature_schema_hash: str | None = None


@dataclass
class EventRecord:
    id: int
    event_id: str
    source: str
    event_type: str
    timestamp: str
    scope: str
    affected_tickers: list[str]
    sector: str | None
    confidence_in_scope: float
    headline_or_label: str
    sentiment_score: float | None
    magnitude_estimate: str


class HistoryManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        if db_path and str(db_path) != str(DB_PATH):
            from database import Base

            self.engine = create_reliable_engine(db_path)
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        else:
            from database import engine as default_engine

            self.engine = default_engine
            self.SessionLocal = SessionLocal
        ensure_directories()

    @with_db_retry(max_retries=5, initial_delay=0.05)
    def save_prediction(
        self,
        signal: PredictionSignal,
        narrative: str | None = None,
        dca_ladder: dict | None = None,
        is_out_of_sample: bool = False,
        prediction_key: str | None = None,
    ) -> int | None:
        p_key = prediction_key or getattr(signal, "prediction_key", None)
        f_hash = getattr(signal, "feature_schema_hash", None)
        try:
            with atomic_transaction(self.SessionLocal) as db:
                # SCHED-002: Check for existing prediction before insert
                if p_key:
                    existing = db.query(DBPrediction).filter(DBPrediction.prediction_key == p_key).first()
                    if existing:
                        logger.info(
                            f"[SCHED-002] Prediction already exists with key {p_key[:12]} (id={existing.id}). "
                            "Skipping duplicate write and returning existing record."
                        )
                        return int(existing.id)

                dca_ladder_str = (
                    json.dumps(dca_ladder) if isinstance(dca_ladder, dict) else str(dca_ladder) if dca_ladder else None
                )
                prediction = DBPrediction(
                    symbol=signal.symbol,
                    timestamp=safe_iso_timestamp(signal.timestamp),
                    action=signal.action,
                    model_predicted_class=signal.model_predicted_class,
                    raw_confidence=safe_float(signal.raw_confidence),
                    risk_adjusted_confidence=safe_float(signal.risk_adjusted_confidence),
                    calibrated_confidence=(
                        safe_float(signal.calibrated_confidence) if signal.calibrated_confidence is not None else None
                    ),
                    agreement_fraction=safe_float(signal.agreement_fraction),
                    downside_summary=signal.downside_summary,
                    upside_summary=signal.upside_summary,
                    reasoning=(
                        json.dumps(signal.reasoning) if isinstance(signal.reasoning, list) else str(signal.reasoning)
                    ),
                    global_risk_level=signal.global_risk_level,
                    risk_toggle_enabled=signal.risk_toggle_enabled,
                    is_safe_to_trade_live=signal.is_safe_to_trade_live,
                    data_stale=signal.data_stale,
                    suppressed=signal.suppressed,
                    suppression_reasons=json.dumps(signal.suppression_reasons) if signal.suppression_reasons else None,
                    horizon=signal.horizon,
                    narrative=narrative,
                    dca_ladder=dca_ladder_str,
                    model_version=signal.model_version,
                    feature_version=signal.feature_version,
                    model_id=signal.model_id,
                    code_commit=signal.code_commit,
                    data_snapshot_id=signal.data_snapshot_id,
                    is_out_of_sample=is_out_of_sample,
                    prediction_key=p_key,
                    feature_schema_hash=f_hash,
                )
                db.add(prediction)
                db.flush()
                pred_id = int(prediction.id) if prediction and prediction.id is not None else None

            health_registry.report("history_manager", ok=True)
            return pred_id
        except Exception as e:
            # If insert collided in race condition, query and return existing record
            if p_key:
                try:
                    with self.SessionLocal() as db_retry:
                        existing = db_retry.query(DBPrediction).filter(DBPrediction.prediction_key == p_key).first()
                        if existing:
                            logger.info(
                                f"[SCHED-002] Race condition resolved for key {p_key[:12]}, returning existing id={existing.id}"
                            )
                            return int(existing.id)
                except Exception:
                    pass
            logger.error(f"Failed saving prediction for {signal.symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving prediction", error=str(e))
            return None

    @with_db_retry(max_retries=5, initial_delay=0.05)
    def save_prediction_bundle(
        self,
        signal: PredictionSignal,
        narrative: str | None = None,
        dca_ladder: dict | None = None,
        is_out_of_sample: bool = False,
        events: list[Event] | None = None,
        prediction_key: str | None = None,
    ) -> tuple[int | None, list[str]]:
        """SCHED-003: Persists a prediction and its associated contributing events in a single atomic transaction.
        
        Guarantees zero partial writes: if any event fails, the entire transaction rolls back,
        preventing orphaned predictions or half-persisted states.
        """
        p_key = prediction_key or getattr(signal, "prediction_key", None)
        f_hash = getattr(signal, "feature_schema_hash", None)
        saved_event_ids: list[str] = []
        try:
            with atomic_transaction(self.SessionLocal) as db:
                # 1. Prediction deduplication check
                pred_id = None
                if p_key:
                    existing = db.query(DBPrediction).filter(DBPrediction.prediction_key == p_key).first()
                    if existing:
                        logger.info(
                            f"[SCHED-002/SCHED-003] Bundle prediction already exists with key {p_key[:12]} (id={existing.id})."
                        )
                        pred_id = int(existing.id)

                if pred_id is None:
                    dca_ladder_str = (
                        json.dumps(dca_ladder) if isinstance(dca_ladder, dict) else str(dca_ladder) if dca_ladder else None
                    )
                    prediction = DBPrediction(
                        symbol=signal.symbol,
                        timestamp=safe_iso_timestamp(signal.timestamp),
                        action=signal.action,
                        model_predicted_class=signal.model_predicted_class,
                        raw_confidence=safe_float(signal.raw_confidence),
                        risk_adjusted_confidence=safe_float(signal.risk_adjusted_confidence),
                        calibrated_confidence=(
                            safe_float(signal.calibrated_confidence) if signal.calibrated_confidence is not None else None
                        ),
                        agreement_fraction=safe_float(signal.agreement_fraction),
                        downside_summary=signal.downside_summary,
                        upside_summary=signal.upside_summary,
                        reasoning=(
                            json.dumps(signal.reasoning) if isinstance(signal.reasoning, list) else str(signal.reasoning)
                        ),
                        global_risk_level=signal.global_risk_level,
                        risk_toggle_enabled=signal.risk_toggle_enabled,
                        is_safe_to_trade_live=signal.is_safe_to_trade_live,
                        data_stale=signal.data_stale,
                        suppressed=signal.suppressed,
                        suppression_reasons=json.dumps(signal.suppression_reasons) if signal.suppression_reasons else None,
                        horizon=signal.horizon,
                        narrative=narrative,
                        dca_ladder=dca_ladder_str,
                        model_version=signal.model_version,
                        feature_version=signal.feature_version,
                        model_id=signal.model_id,
                        code_commit=signal.code_commit,
                        data_snapshot_id=signal.data_snapshot_id,
                        is_out_of_sample=is_out_of_sample,
                        prediction_key=p_key,
                        feature_schema_hash=f_hash,
                    )
                    db.add(prediction)
                    db.flush()
                    pred_id = int(prediction.id) if prediction and prediction.id is not None else None

                # 2. Persist contributing events atomically in the same transaction
                if events:
                    for ev in events:
                        existing_ev = db.query(DBEvent).filter(DBEvent.event_id == ev.event_id).first()
                        if existing_ev:
                            saved_event_ids.append(ev.event_id)
                            continue

                        tickers_str = TICKER_DELIMITER + TICKER_DELIMITER.join(ev.affected_tickers) + TICKER_DELIMITER
                        db_event = DBEvent(
                            event_id=ev.event_id,
                            source=ev.source,
                            event_type=ev.event_type,
                            timestamp=safe_iso_timestamp(ev.timestamp),
                            scope=ev.scope,
                            affected_tickers=tickers_str,
                            sector=ev.sector,
                            confidence_in_scope=safe_float(ev.confidence_in_scope),
                            headline_or_label=ev.headline_or_label,
                            sentiment_score=safe_float(ev.sentiment_score) if ev.sentiment_score is not None else None,
                            magnitude_estimate=ev.magnitude_estimate,
                        )
                        db.add(db_event)
                        saved_event_ids.append(ev.event_id)

            health_registry.report("history_manager", ok=True)
            return pred_id, saved_event_ids
        except Exception as e:
            logger.error(f"[SCHED-003] Failed saving prediction bundle for {signal.symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving prediction bundle", error=str(e))
            return None, []

    @with_db_retry(max_retries=5, initial_delay=0.05)
    def resolve_outcome(self, prediction_id: int, actual_class: str) -> bool:
        try:
            with atomic_transaction(self.SessionLocal) as db:
                prediction = db.query(DBPrediction).filter(DBPrediction.id == prediction_id).first()
                if prediction:
                    prediction.outcome_resolved = True  # type: ignore[assignment]
                    prediction.outcome_correct = prediction.model_predicted_class == actual_class  # type: ignore[assignment]
                    prediction.outcome_actual_class = actual_class  # type: ignore[assignment]
                    prediction.resolved_at = datetime.now(timezone.utc).isoformat()  # type: ignore[assignment]
            health_registry.report("history_manager", ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed resolving outcome for prediction_id={prediction_id}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed resolving outcome", error=str(e))
            return False

    def get_predictions(
        self,
        symbol: str | None = None,
        horizon: str | None = None,
        limit: int = 200,
        only_unresolved: bool = False,
        model_version: str | None = None,
        feature_version: str | None = None,
        only_out_of_sample: bool = False,
    ) -> list[PredictionRecord]:
        try:
            with self.SessionLocal() as db:
                query = db.query(DBPrediction)
                if symbol:
                    query = query.filter(DBPrediction.symbol == symbol)
                if horizon:
                    query = query.filter(DBPrediction.horizon == horizon)
                if only_unresolved:
                    query = query.filter(DBPrediction.outcome_resolved.is_(False))
                if model_version:
                    query = query.filter(DBPrediction.model_version == model_version)
                if feature_version:
                    query = query.filter(DBPrediction.feature_version == feature_version)
                if only_out_of_sample:
                    query = query.filter(DBPrediction.is_out_of_sample.is_(True))

                rows = query.order_by(DBPrediction.id.desc()).limit(limit).all()

                records = []
                for row in rows:
                    try:
                        records.append(
                            PredictionRecord(
                                id=int(row.id),
                                symbol=str(row.symbol or "UNKNOWN"),
                                horizon=str(row.horizon or "INTRADAY"),
                                model_version=str(row.model_version or "UNKNOWN"),
                                feature_version=str(row.feature_version or "UNKNOWN"),
                                model_id=str(row.model_id) if row.model_id else None,
                                code_commit=str(row.code_commit) if row.code_commit else None,
                                data_snapshot_id=str(row.data_snapshot_id) if row.data_snapshot_id else None,
                                narrative=str(row.narrative) if row.narrative else "",
                                dca_ladder=str(row.dca_ladder) if row.dca_ladder else "",
                                timestamp=safe_iso_timestamp(row.timestamp),
                                action=str(row.action or "HOLD"),
                                model_predicted_class=str(row.model_predicted_class or "FLAT"),
                                raw_confidence=safe_float(row.raw_confidence, 0.0),
                                risk_adjusted_confidence=safe_float(row.risk_adjusted_confidence, 0.0),
                                calibrated_confidence=(
                                    safe_float(row.calibrated_confidence) if row.calibrated_confidence is not None else None
                                ),
                                agreement_fraction=safe_float(row.agreement_fraction, 0.0),
                                outcome_resolved=bool(row.outcome_resolved),
                                outcome_correct=bool(row.outcome_correct) if row.outcome_correct is not None else None,
                                outcome_actual_class=str(row.outcome_actual_class) if row.outcome_actual_class else None,
                                resolved_at=safe_iso_timestamp(row.resolved_at) if row.resolved_at else None,
                                is_out_of_sample=bool(row.is_out_of_sample),
                                prediction_key=str(row.prediction_key) if getattr(row, "prediction_key", None) else None,
                                feature_schema_hash=str(row.feature_schema_hash) if getattr(row, "feature_schema_hash", None) else None,
                            )
                        )
                    except Exception as parse_err:
                        logger.warning(f"[SCHED-003] Skipping corrupted prediction row id={getattr(row, 'id', None)}: {parse_err}")
            health_registry.report("history_manager", ok=True)
            return records
        except Exception as e:
            logger.error(f"Failed getting predictions: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed getting predictions", error=str(e))
            return []

    def get_prediction_by_key(self, prediction_key: str) -> PredictionRecord | None:
        """SCHED-002: Retrieve a prediction record by its unique idempotency key."""
        try:
            with self.SessionLocal() as db:
                row = db.query(DBPrediction).filter(DBPrediction.prediction_key == prediction_key).first()
                if not row:
                    return None
                return PredictionRecord(
                    id=int(row.id),
                    symbol=str(row.symbol or "UNKNOWN"),
                    horizon=str(row.horizon or "INTRADAY"),
                    model_version=str(row.model_version or "UNKNOWN"),
                    feature_version=str(row.feature_version or "UNKNOWN"),
                    model_id=str(row.model_id) if row.model_id else None,
                    code_commit=str(row.code_commit) if row.code_commit else None,
                    data_snapshot_id=str(row.data_snapshot_id) if row.data_snapshot_id else None,
                    narrative=str(row.narrative) if row.narrative else "",
                    dca_ladder=str(row.dca_ladder) if row.dca_ladder else "",
                    timestamp=safe_iso_timestamp(row.timestamp),
                    action=str(row.action or "HOLD"),
                    model_predicted_class=str(row.model_predicted_class or "FLAT"),
                    raw_confidence=safe_float(row.raw_confidence, 0.0),
                    risk_adjusted_confidence=safe_float(row.risk_adjusted_confidence, 0.0),
                    calibrated_confidence=(
                        safe_float(row.calibrated_confidence) if row.calibrated_confidence is not None else None
                    ),
                    agreement_fraction=safe_float(row.agreement_fraction, 0.0),
                    outcome_resolved=bool(row.outcome_resolved),
                    outcome_correct=bool(row.outcome_correct) if row.outcome_correct is not None else None,
                    outcome_actual_class=str(row.outcome_actual_class) if row.outcome_actual_class else None,
                    resolved_at=safe_iso_timestamp(row.resolved_at) if row.resolved_at else None,
                    is_out_of_sample=bool(row.is_out_of_sample),
                    prediction_key=str(row.prediction_key) if getattr(row, "prediction_key", None) else None,
                    feature_schema_hash=str(row.feature_schema_hash) if getattr(row, "feature_schema_hash", None) else None,
                )
        except Exception as e:
            logger.error(f"Failed getting prediction by key={prediction_key}: {e}")
            return None

    def get_prediction_signal_by_key(self, prediction_key: str) -> PredictionSignal | None:
        """SCHED-002: Reconstructs a full PredictionSignal from stored database record."""
        try:
            with self.SessionLocal() as db:
                row = db.query(DBPrediction).filter(DBPrediction.prediction_key == prediction_key).first()
                if not row:
                    return None

                try:
                    ts = pd.Timestamp(row.timestamp).to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    ts = datetime.now(timezone.utc)

                reasoning = safe_json_loads(row.reasoning, [])
                if not isinstance(reasoning, list):
                    reasoning = [str(reasoning)]

                suppression_reasons = safe_json_loads(row.suppression_reasons, [])
                if not isinstance(suppression_reasons, list):
                    suppression_reasons = [str(suppression_reasons)]

                return PredictionSignal(
                    symbol=str(row.symbol),
                    timestamp=ts,
                    horizon=str(row.horizon),
                    action=str(row.action),
                    model_predicted_class=str(row.model_predicted_class),
                    model_version=str(row.model_version),
                    feature_version=str(row.feature_version),
                    raw_confidence=safe_float(row.raw_confidence, 0.0),
                    risk_adjusted_confidence=safe_float(row.risk_adjusted_confidence, 0.0),
                    calibrated_confidence=safe_float(row.calibrated_confidence) if row.calibrated_confidence is not None else None,
                    agreement_fraction=safe_float(row.agreement_fraction, 0.0),
                    downside_summary=str(row.downside_summary or ""),
                    upside_summary=str(row.upside_summary or ""),
                    reasoning=reasoning,
                    global_risk_level=str(row.global_risk_level or "NORMAL"),
                    risk_toggle_enabled=bool(row.risk_toggle_enabled),
                    is_safe_to_trade_live=bool(row.is_safe_to_trade_live),
                    data_stale=bool(row.data_stale),
                    suppressed=bool(row.suppressed),
                    suppression_reasons=suppression_reasons,
                    model_id=str(row.model_id) if row.model_id else None,
                    code_commit=str(row.code_commit) if row.code_commit else None,
                    data_snapshot_id=str(row.data_snapshot_id) if row.data_snapshot_id else None,
                    prediction_key=str(row.prediction_key) if row.prediction_key else None,
                    feature_schema_hash=str(row.feature_schema_hash) if getattr(row, "feature_schema_hash", None) else None,
                )
        except Exception as e:
            logger.error(f"Failed reconstructing prediction signal for key={prediction_key}: {e}")
            return None

    def build_calibration_dataset(
        self,
        symbol: str | None = None,
        horizon: str | None = None,
        model_version: str | None = None,
        feature_version: str | None = None,
        only_out_of_sample: bool = False,
        max_age_days: int | None = None,
    ) -> pd.DataFrame:
        try:
            records = self.get_predictions(
                symbol=symbol,
                horizon=horizon,
                model_version=model_version,
                feature_version=feature_version,
                limit=100_000,
                only_unresolved=False,
                only_out_of_sample=only_out_of_sample,
            )
            cutoff_ts = None
            if max_age_days is not None and max_age_days > 0:
                cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

            data = []
            for r in records:
                if not r.outcome_resolved or r.outcome_correct is None or r.risk_adjusted_confidence is None:
                    continue
                if r.model_version == "UNKNOWN":
                    continue
                if cutoff_ts and r.resolved_at and r.resolved_at < cutoff_ts:
                    continue
                data.append({"confidence": float(r.risk_adjusted_confidence), "correct": int(r.outcome_correct)})

            health_registry.report("history_manager", ok=True)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed building calibration dataset for symbol={symbol}: {e}")
            health_registry.report(
                "history_manager", ok=False, detail="Failed building calibration dataset", error=str(e)
            )
            return pd.DataFrame()

    @with_db_retry(max_retries=5, initial_delay=0.05)
    def save_event(self, event: Event) -> bool:
        try:
            with atomic_transaction(self.SessionLocal) as db:
                # SCHED-003: Deduplicate writes on existing event_id
                existing = db.query(DBEvent).filter(DBEvent.event_id == event.event_id).first()
                if existing:
                    logger.info(
                        f"[SCHED-003] Event already exists with event_id={event.event_id}. Skipping duplicate write."
                    )
                    return True

                tickers_str = TICKER_DELIMITER + TICKER_DELIMITER.join(event.affected_tickers) + TICKER_DELIMITER
                db_event = DBEvent(
                    event_id=event.event_id,
                    source=event.source,
                    event_type=event.event_type,
                    timestamp=safe_iso_timestamp(event.timestamp),
                    scope=event.scope,
                    affected_tickers=tickers_str,
                    sector=event.sector,
                    confidence_in_scope=safe_float(event.confidence_in_scope),
                    headline_or_label=event.headline_or_label,
                    sentiment_score=safe_float(event.sentiment_score) if event.sentiment_score is not None else None,
                    magnitude_estimate=event.magnitude_estimate,
                )
                db.add(db_event)
            health_registry.report("history_manager", ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed saving event {event.event_id}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving event", error=str(e))
            return False

    def get_events_for_symbol(self, symbol: str, limit: int = 50) -> list[EventRecord]:
        try:
            with self.SessionLocal() as db:
                search_str = f"%{TICKER_DELIMITER}{symbol}{TICKER_DELIMITER}%"
                rows = (
                    db.query(DBEvent)
                    .filter(DBEvent.affected_tickers.like(search_str))
                    .order_by(DBEvent.id.desc())
                    .limit(limit)
                    .all()
                )

                records = []
                for row in rows:
                    try:
                        tickers = [t for t in str(row.affected_tickers).split(TICKER_DELIMITER) if t]
                        records.append(
                            EventRecord(
                                id=int(row.id),
                                event_id=str(row.event_id),
                                source=str(row.source),
                                event_type=str(row.event_type),
                                timestamp=safe_iso_timestamp(row.timestamp),
                                scope=str(row.scope),
                                affected_tickers=tickers,
                                sector=str(row.sector) if row.sector else None,
                                confidence_in_scope=safe_float(row.confidence_in_scope, 0.0),
                                headline_or_label=str(row.headline_or_label),
                                sentiment_score=safe_float(row.sentiment_score) if row.sentiment_score is not None else None,
                                magnitude_estimate=str(row.magnitude_estimate),
                            )
                        )
                    except Exception as ev_err:
                        logger.warning(f"[SCHED-003] Skipping corrupted event row id={getattr(row, 'id', None)}: {ev_err}")
            health_registry.report("history_manager", ok=True)
            return records
        except Exception as e:
            logger.error(f"Failed fetching events for symbol={symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed fetching events", error=str(e))
            return []

    @with_db_retry(max_retries=5, initial_delay=0.05)
    def save_backtest_result(
        self,
        symbol: str,
        horizon: str,
        strategy_ret: float,
        base_ret: float,
        alpha: float,
        edge: str,
        calib: str,
        ece: float | None,
        live_worthy: bool,
    ) -> bool:
        try:
            with atomic_transaction(self.SessionLocal) as db:
                row = db.query(DBBacktestMetric).filter_by(symbol=symbol, horizon=horizon).first()
                if row:
                    row.strategy_cumulative_return_pct = safe_float(strategy_ret)  # type: ignore[assignment]
                    row.baseline_cumulative_return_pct = safe_float(base_ret)  # type: ignore[assignment]
                    row.alpha_pct = safe_float(alpha)  # type: ignore[assignment]
                    row.edge_check_status = edge  # type: ignore[assignment]
                    row.calibration_status = calib  # type: ignore[assignment]
                    row.calibration_ece = safe_float(ece) if ece is not None else None  # type: ignore[assignment]
                    row.is_live_worthy = bool(live_worthy)  # type: ignore[assignment]
                    row.updated_at = datetime.now(timezone.utc).isoformat()  # type: ignore[assignment]
                else:
                    metric = DBBacktestMetric(
                        symbol=symbol,
                        horizon=horizon,
                        strategy_cumulative_return_pct=safe_float(strategy_ret),
                        baseline_cumulative_return_pct=safe_float(base_ret),
                        alpha_pct=safe_float(alpha),
                        edge_check_status=edge,
                        calibration_status=calib,
                        calibration_ece=safe_float(ece) if ece is not None else None,
                        is_live_worthy=bool(live_worthy),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )
                    db.add(metric)
            health_registry.report("history_manager", ok=True, detail=f"Saved backtest metrics for {symbol} {horizon}")
            return True
        except Exception as e:
            logger.error(f"Failed saving backtest metrics for {symbol} {horizon}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving backtest metrics", error=str(e))
            return False

    def run_recovery(self, stale_unresolved_hours: int = 24):
        """SCHED-003: Runs storage startup recovery on the associated engine."""
        return run_storage_recovery(self.engine, db_path=self.db_path, stale_unresolved_hours=stale_unresolved_hours)


if __name__ == "__main__":
    import os

    from config import DB_DIR

    configure_logging(log_filename="history_manager_selftest.log")
    logger.info("Running history_manager.py self-test...")

    test_db_path = DB_DIR / "test_history_selftest.sqlite3"
    if test_db_path.exists():
        try:
            os.remove(test_db_path)
        except:
            pass

    test_symbol = "M&M"

    try:
        print("\n=== HISTORY MANAGER SELF-TEST RESULT ===")
        manager = HistoryManager(db_path=test_db_path)

        for i in range(5):
            actual = "UP" if i < 3 else "FLAT"
            s = PredictionSignal(
                symbol=test_symbol,
                timestamp=datetime.now(timezone.utc),
                action="BUY",
                model_predicted_class="UP",
                raw_confidence=0.72 + (i * 0.01),
                risk_adjusted_confidence=0.70 + (i * 0.01),
                calibrated_confidence=None,
                agreement_fraction=0.66,
                downside_summary="Some downside.",
                upside_summary="Some upside.",
                reasoning=["Model said UP."],
                global_risk_level="NORMAL",
                risk_toggle_enabled=False,
                is_safe_to_trade_live=False,
                data_stale=False,
                suppressed=False,
                suppression_reasons=[],
                horizon="INTRADAY",
                model_version="v1.0",
                feature_version="v1.0",
            )
            pid = manager.save_prediction(s)
            if pid is not None:
                manager.resolve_outcome(pid, actual_class=actual)

                if i == 0:
                    print(f"Prediction saved: id={pid}")
                    resolved_ok = manager.resolve_outcome(pid, actual_class="UP")
                    predictions = manager.get_predictions(test_symbol)
                    print(f"Outcome resolved: {resolved_ok}, outcome_correct={predictions[0].outcome_correct}")
                    assert resolved_ok and predictions[0].outcome_correct is True

        calibration_df = manager.build_calibration_dataset(test_symbol, horizon="INTRADAY", model_version="v1.0")
        print(f"Calibration dataset built: {len(calibration_df)} rows, columns={list(calibration_df.columns)}")
        assert len(calibration_df) == 5
        assert set(calibration_df.columns) == {"confidence", "correct"}
        assert calibration_df["correct"].sum() == 3

        event_for_symbol = Event(
            event_id="EVT_TEST_1",
            source="CORPORATE",
            event_type="CORPORATE_ANNOUNCEMENT",
            timestamp=datetime.now(timezone.utc),
            scope="STOCK",
            affected_tickers=[test_symbol, "HDFCBANK"],
            sector="Auto",
            confidence_in_scope=1.0,
            headline_or_label="M&M announces EV partnership",
            sentiment_score=0.8,
            magnitude_estimate="HIGH",
        )
        event_not_for_symbol = Event(
            event_id="EVT_TEST_2",
            source="CORPORATE",
            event_type="CORPORATE_ANNOUNCEMENT",
            timestamp=datetime.now(timezone.utc),
            scope="STOCK",
            affected_tickers=["MARUTI", "TMPV"],
            sector="Auto",
            confidence_in_scope=1.0,
            headline_or_label="Board meeting for Maruti",
            sentiment_score=0.1,
            magnitude_estimate="MEDIUM",
        )
        manager.save_event(event_for_symbol)
        manager.save_event(event_not_for_symbol)

        # Duplicate event save test (SCHED-003)
        dup_saved = manager.save_event(event_for_symbol)
        assert dup_saved is True

        events_for_symbol = manager.get_events_for_symbol(test_symbol)
        print(f"Events correctly matched for {test_symbol}: {len(events_for_symbol)}")
        assert len(events_for_symbol) == 1
        assert events_for_symbol[0].event_id == "EVT_TEST_1"

        metrics_saved = manager.save_backtest_result(
            symbol=test_symbol,
            horizon="INTRADAY",
            strategy_ret=15.5,
            base_ret=10.0,
            alpha=5.5,
            edge="PASS",
            calib="PASS",
            ece=0.03,
            live_worthy=True,
        )
        assert metrics_saved is True

        # Bundle test (SCHED-003)
        bundle_signal = PredictionSignal(
            symbol="INFY",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            model_predicted_class="UP",
            raw_confidence=0.85,
            risk_adjusted_confidence=0.82,
            calibrated_confidence=0.80,
            agreement_fraction=0.9,
            downside_summary="",
            upside_summary="",
            reasoning=["Bundle reasoning"],
            global_risk_level="NORMAL",
            risk_toggle_enabled=False,
            is_safe_to_trade_live=True,
            data_stale=False,
            suppressed=False,
            suppression_reasons=[],
            horizon="INTRADAY",
            model_version="v1.0",
            feature_version="v1.0",
        )
        bundle_event = Event(
            event_id="EVT_BUNDLE_1",
            source="NEWS",
            event_type="EARNINGS",
            timestamp=datetime.now(timezone.utc),
            scope="STOCK",
            affected_tickers=["INFY"],
            sector="IT",
            confidence_in_scope=1.0,
            headline_or_label="INFY Q2 results beat estimates",
            sentiment_score=0.9,
            magnitude_estimate="HIGH",
        )
        b_pred_id, b_ev_ids = manager.save_prediction_bundle(bundle_signal, events=[bundle_event])
        assert b_pred_id is not None
        assert b_ev_ids == ["EVT_BUNDLE_1"]

        # Run recovery
        rep = manager.run_recovery()
        assert rep.integrity_ok is True
        print(f"Recovery executed: WAL={rep.wal_checkpointed}, predictions={rep.prediction_count}")

        print("STATUS: PASS")
        logger.info("history_manager.py self-test passed.")
    except AssertionError as ae:
        logger.error(f"history_manager.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL - {ae}")
    except Exception as e:
        logger.error(f"history_manager.py self-test crashed: {e}")
        print(f"STATUS: FAIL - {e}")
        import traceback

        traceback.print_exc()
