import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text
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

logger = logging.getLogger(__name__)

LABEL_DEFINITION_VERSION = "direction-adaptive-deadband-v1"


from dataclasses import dataclass


@dataclass
class PredictionRecord:
    id: int
    symbol: str
    horizon: str
    model_version: str
    feature_version: str
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
    generation_id: str = "UNKNOWN"
    data_version: str = "UNKNOWN"
    label_definition_version: str = "UNKNOWN"
    entry_timestamp: str | None = None
    entry_price: float | None = None
    target_timestamp: str | None = None
    outcome_entry_price: float | None = None
    outcome_endpoint_price: float | None = None
    outcome_resolution_status: str = "PENDING"
    outcome_resolution_reason: str | None = None
    outcome_entry_timestamp: str | None = None
    outcome_target_timestamp: str | None = None
    delivery_count: int = 1
    data_stale: bool = False
    suppressed: bool = False


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
        ensure_directories()
        if db_path and str(db_path) != str(DB_PATH):
            from sqlalchemy import create_engine
            from database import Base
            self.engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False, "timeout": 30},
            )
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
        else:
            self.SessionLocal = SessionLocal
            from database import engine
            self.engine = engine
        self._configure_sqlite()
        self._migrate_schema()

    def _configure_sqlite(self) -> None:
        if self.engine.dialect.name != "sqlite":
            return
        try:
            with self.engine.begin() as conn:
                conn.execute(text("PRAGMA busy_timeout=30000"))
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
        except Exception as exc:
            logger.error("Failed configuring SQLite storage: %s", exc)
            raise

    def _migrate_schema(self) -> None:
        """Upgrade legacy SQLite schemas and fail loudly if verification fails."""
        if self.engine.dialect.name != "sqlite":
            return
        from database import Base
        Base.metadata.create_all(bind=self.engine)
        prediction_columns = {
            "generation_id": "TEXT",
            "data_version": "TEXT",
            "label_definition_version": "TEXT",
            "entry_timestamp": "TEXT",
            "entry_price": "REAL",
            "target_timestamp": "TEXT",
            "outcome_entry_price": "REAL",
            "outcome_endpoint_price": "REAL",
            "outcome_resolution_status": "TEXT DEFAULT 'PENDING'",
            "outcome_resolution_reason": "TEXT",
            "outcome_entry_timestamp": "TEXT",
            "outcome_target_timestamp": "TEXT",
            "delivery_count": "INTEGER NOT NULL DEFAULT 1",
        }
        with self.engine.begin() as conn:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(predictions)"))}
            for name, definition in prediction_columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {name} {definition}"))
            conn.execute(text("UPDATE predictions SET generation_id='UNKNOWN' WHERE generation_id IS NULL OR generation_id=''"))
            conn.execute(text("UPDATE predictions SET data_version='UNKNOWN' WHERE data_version IS NULL OR data_version=''"))
            conn.execute(text("UPDATE predictions SET label_definition_version='UNKNOWN' WHERE label_definition_version IS NULL OR label_definition_version=''"))
            conn.execute(text("UPDATE predictions SET outcome_resolution_status=CASE WHEN outcome_resolved=1 THEN 'RESOLVED' ELSE 'PENDING' END"))
            missing = set(prediction_columns) - {row[1] for row in conn.execute(text("PRAGMA table_info(predictions)"))}
            if missing:
                raise RuntimeError(f"Prediction schema migration incomplete: {sorted(missing)}")

            bt_info = list(conn.execute(text("PRAGMA table_info(backtest_metrics)")))
            if bt_info:
                bt_cols = {row[1] for row in bt_info}
                pk_cols = {row[1] for row in bt_info if row[5]}
                if "run_id" not in bt_cols or pk_cols == {"symbol", "horizon"}:
                    conn.execute(text("ALTER TABLE backtest_metrics RENAME TO backtest_metrics_legacy"))
                    conn.execute(text("""CREATE TABLE backtest_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL UNIQUE,
                        symbol TEXT NOT NULL,
                        horizon TEXT NOT NULL,
                        strategy_cumulative_return_pct REAL,
                        baseline_cumulative_return_pct REAL,
                        alpha_pct REAL,
                        edge_check_status TEXT,
                        calibration_status TEXT,
                        calibration_ece REAL,
                        is_live_worthy BOOLEAN,
                        updated_at TEXT
                    )"""))
                    conn.execute(text("""INSERT INTO backtest_metrics
                        (run_id, symbol, horizon, strategy_cumulative_return_pct,
                         baseline_cumulative_return_pct, alpha_pct, edge_check_status,
                         calibration_status, calibration_ece, is_live_worthy, updated_at)
                        SELECT 'legacy-' || symbol || '-' || horizon, symbol, horizon,
                               strategy_cumulative_return_pct, baseline_cumulative_return_pct,
                               alpha_pct, edge_check_status, calibration_status,
                               calibration_ece, is_live_worthy, updated_at
                        FROM backtest_metrics_legacy"""))
                    conn.execute(text("DROP TABLE backtest_metrics_legacy"))
                if "run_id" not in {row[1] for row in conn.execute(text("PRAGMA table_info(backtest_metrics)"))}:
                    raise RuntimeError("Backtest schema migration incomplete: run_id missing")

    def save_prediction(
        self,
        signal: PredictionSignal,
        narrative: str | None = None,
        dca_ladder: dict | None = None,
        is_out_of_sample: bool = False,
        generation_id: str | None = None,
        data_version: str | None = None,
        label_definition_version: str | None = None,
        entry_timestamp: str | None = None,
        entry_price: float | None = None,
    ) -> int | None:
        try:
            with self.SessionLocal() as db:
                dca_ladder_str = (
                    json.dumps(dca_ladder) if isinstance(dca_ladder, dict) else str(dca_ladder) if dca_ladder else None
                )
                entry_ts = entry_timestamp or str(signal.timestamp)
                entry_px = entry_price if entry_price is not None else getattr(signal, "entry_price", None)
                seed = {"symbol": signal.symbol, "timestamp": str(signal.timestamp), "horizon": signal.horizon,
                        "model_version": signal.model_version, "feature_version": signal.feature_version,
                        "action": signal.action, "predicted_class": signal.model_predicted_class,
                        "entry_timestamp": entry_ts, "entry_price": entry_px}
                stable_generation_id = generation_id or "forecast-" + hashlib.sha256(
                    json.dumps(seed, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:24]
                existing = db.query(DBPrediction).filter(DBPrediction.generation_id == stable_generation_id).first()
                if existing:
                    existing.delivery_count = int(existing.delivery_count or 1) + 1
                    db.commit()
                    return int(existing.id)
                prediction = DBPrediction(
                    symbol=signal.symbol,
                    timestamp=str(signal.timestamp),
                    action=signal.action,
                    model_predicted_class=signal.model_predicted_class,
                    raw_confidence=float(signal.raw_confidence),
                    risk_adjusted_confidence=float(signal.risk_adjusted_confidence),
                    calibrated_confidence=(
                        float(signal.calibrated_confidence) if signal.calibrated_confidence is not None else None
                    ),
                    agreement_fraction=float(signal.agreement_fraction),
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
                    is_out_of_sample=is_out_of_sample,
                    generation_id=stable_generation_id,
                    data_version=data_version or getattr(signal, "data_version", None) or "UNKNOWN",
                    label_definition_version=label_definition_version or getattr(signal, "label_definition_version", None) or LABEL_DEFINITION_VERSION,
                    entry_timestamp=entry_ts,
                    entry_price=float(entry_px) if entry_px is not None else None,
                    outcome_resolution_status="PENDING",
                    delivery_count=1,
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
            health_registry.report("history_manager", ok=True)
            return int(prediction.id) if prediction and prediction.id is not None else None
        except Exception as e:
            logger.error(f"Failed saving prediction for {signal.symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving prediction", error=str(e))
            return None

    def resolve_outcome(self, prediction_id: int, actual_class: str, *, entry_price: float | None = None,
                        endpoint_price: float | None = None, entry_timestamp: str | None = None,
                        target_timestamp: str | None = None, resolution_reason: str | None = None) -> bool:
        try:
            with self.SessionLocal() as db:
                prediction = db.query(DBPrediction).filter(DBPrediction.id == prediction_id).first()
                if not prediction:
                    return False
                if bool(prediction.outcome_resolved):
                    return True
                prediction.outcome_resolved = True  # type: ignore[assignment]
                prediction.outcome_correct = prediction.model_predicted_class == actual_class  # type: ignore[assignment]
                prediction.outcome_actual_class = actual_class  # type: ignore[assignment]
                prediction.resolved_at = datetime.now().isoformat()  # type: ignore[assignment]
                prediction.outcome_entry_price = entry_price  # type: ignore[assignment]
                prediction.outcome_endpoint_price = endpoint_price  # type: ignore[assignment]
                prediction.outcome_entry_timestamp = entry_timestamp  # type: ignore[assignment]
                prediction.outcome_target_timestamp = target_timestamp  # type: ignore[assignment]
                prediction.target_timestamp = target_timestamp  # type: ignore[assignment]
                prediction.outcome_resolution_status = "RESOLVED"  # type: ignore[assignment]
                prediction.outcome_resolution_reason = resolution_reason or "verified_market_observations"  # type: ignore[assignment]
                db.commit()
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

                order = DBPrediction.id.asc() if only_unresolved else DBPrediction.id.desc()
                rows = query.order_by(order).limit(limit).all()

                records = []
                for row in rows:
                    records.append(
                        PredictionRecord(
                            id=int(row.id),
                            symbol=str(row.symbol),
                            horizon=str(row.horizon),
                            model_version=str(row.model_version),
                            feature_version=str(row.feature_version),
                            narrative=str(row.narrative) if row.narrative else "",
                            dca_ladder=str(row.dca_ladder) if row.dca_ladder else "",
                            timestamp=str(row.timestamp),
                            action=str(row.action),
                            model_predicted_class=str(row.model_predicted_class),
                            raw_confidence=float(row.raw_confidence),
                            risk_adjusted_confidence=float(row.risk_adjusted_confidence),
                            calibrated_confidence=(
                                float(row.calibrated_confidence) if row.calibrated_confidence is not None else None
                            ),
                            agreement_fraction=float(row.agreement_fraction),
                            outcome_resolved=bool(row.outcome_resolved),
                            outcome_correct=bool(row.outcome_correct) if row.outcome_correct is not None else None,
                            outcome_actual_class=str(row.outcome_actual_class) if row.outcome_actual_class else None,
                            resolved_at=str(row.resolved_at) if row.resolved_at else None,
                            is_out_of_sample=bool(row.is_out_of_sample),
                            generation_id=str(row.generation_id) if row.generation_id else "UNKNOWN",
                            data_version=str(row.data_version) if row.data_version else "UNKNOWN",
                            label_definition_version=str(row.label_definition_version) if row.label_definition_version else "UNKNOWN",
                            entry_timestamp=str(row.entry_timestamp) if row.entry_timestamp else None,
                            entry_price=float(row.entry_price) if row.entry_price is not None else None,
                            target_timestamp=str(row.target_timestamp) if row.target_timestamp else None,
                            outcome_entry_price=float(row.outcome_entry_price) if row.outcome_entry_price is not None else None,
                            outcome_endpoint_price=float(row.outcome_endpoint_price) if row.outcome_endpoint_price is not None else None,
                            outcome_resolution_status=str(row.outcome_resolution_status or "PENDING"),
                            outcome_resolution_reason=str(row.outcome_resolution_reason) if row.outcome_resolution_reason else None,
                            outcome_entry_timestamp=str(row.outcome_entry_timestamp) if row.outcome_entry_timestamp else None,
                            outcome_target_timestamp=str(row.outcome_target_timestamp) if row.outcome_target_timestamp else None,
                            delivery_count=int(row.delivery_count or 1),
                            data_stale=bool(row.data_stale),
                            suppressed=bool(row.suppressed),
                        )
                    )
            health_registry.report("history_manager", ok=True)
            return records
        except Exception as e:
            logger.error(f"Failed getting predictions: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed getting predictions", error=str(e))
            return []

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
                cutoff_ts = (datetime.now() - timedelta(days=max_age_days)).isoformat()

            data = []
            for r in records:
                if not r.outcome_resolved or r.outcome_correct is None or r.risk_adjusted_confidence is None:
                    continue
                if r.model_version == "UNKNOWN" or r.feature_version == "UNKNOWN":
                    continue
                if r.generation_id == "UNKNOWN" or r.label_definition_version == "UNKNOWN":
                    continue
                if r.data_stale:
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

    def save_event(self, event: Event) -> bool:
        try:
            with self.SessionLocal() as db:
                tickers_str = TICKER_DELIMITER + TICKER_DELIMITER.join(event.affected_tickers) + TICKER_DELIMITER
                db_event = DBEvent(
                    event_id=event.event_id,
                    source=event.source,
                    event_type=event.event_type,
                    timestamp=str(event.timestamp),
                    scope=event.scope,
                    affected_tickers=tickers_str,
                    sector=event.sector,
                    confidence_in_scope=float(event.confidence_in_scope),
                    headline_or_label=event.headline_or_label,
                    sentiment_score=float(event.sentiment_score) if event.sentiment_score is not None else None,
                    magnitude_estimate=event.magnitude_estimate,
                )
                db.add(db_event)
                db.commit()
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
                    tickers = [t for t in str(row.affected_tickers).split(TICKER_DELIMITER) if t]
                    records.append(
                        EventRecord(
                            id=int(row.id),
                            event_id=str(row.event_id),
                            source=str(row.source),
                            event_type=str(row.event_type),
                            timestamp=str(row.timestamp),
                            scope=str(row.scope),
                            affected_tickers=tickers,
                            sector=str(row.sector) if row.sector else None,
                            confidence_in_scope=float(row.confidence_in_scope),
                            headline_or_label=str(row.headline_or_label),
                            sentiment_score=float(row.sentiment_score) if row.sentiment_score is not None else None,
                            magnitude_estimate=str(row.magnitude_estimate),
                        )
                    )
            health_registry.report("history_manager", ok=True)
            return records
        except Exception as e:
            logger.error(f"Failed fetching events for symbol={symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed fetching events", error=str(e))
            return []

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
            with self.SessionLocal() as db:
                run_id = hashlib.sha256(
                    f"{symbol}|{horizon}|{strategy_ret}|{base_ret}|{alpha}|{edge}|{calib}|{ece}|{live_worthy}|{datetime.now().isoformat()}".encode()
                ).hexdigest()[:24]
                metric = DBBacktestMetric(
                    run_id=run_id, symbol=symbol, horizon=horizon,
                    strategy_cumulative_return_pct=strategy_ret,
                    baseline_cumulative_return_pct=base_ret, alpha_pct=alpha,
                    edge_check_status=edge, calibration_status=calib,
                    calibration_ece=ece, is_live_worthy=live_worthy,
                    updated_at=datetime.now().isoformat(),
                )
                db.add(metric)
                db.commit()
            health_registry.report("history_manager", ok=True, detail=f"Saved backtest metrics for {symbol} {horizon}")
            return True
        except Exception as e:
            logger.error(f"Failed saving backtest metrics for {symbol} {horizon}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving backtest metrics", error=str(e))
            return False


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
            actual = "UP" if i < 2 else "FLAT"
            s = PredictionSignal(
                symbol=test_symbol,
                timestamp=datetime.now(),
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
            timestamp=datetime.now(),
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
            timestamp=datetime.now(),
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
