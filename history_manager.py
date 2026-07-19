# 1. Standard library imports
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 2. Third-party imports
import pandas as pd

# 3. Local imports
from config import DB_PATH, ensure_directories, configure_logging
from predictor import PredictionSignal
from event_classifier import Event
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
SQLITE_TIMEOUT_SECONDS = 10
TICKER_DELIMITER = ","  # affected_tickers stored as ',TICKER1,TICKER2,' so LIKE matching can't false-positive on substrings


@dataclass
class PredictionRecord:
    id: int
    symbol: str
    horizon: str
    narrative: str
    dca_ladder: str
    timestamp: str
    action: str
    model_predicted_class: str
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: Optional[float]
    agreement_fraction: float
    outcome_resolved: bool
    outcome_correct: Optional[bool]
    outcome_actual_class: Optional[str]
    resolved_at: Optional[str]


@dataclass
class EventRecord:
    id: int
    event_id: str
    source: str
    event_type: str
    timestamp: str
    scope: str
    affected_tickers: List[str]
    sector: Optional[str]
    confidence_in_scope: float
    headline_or_label: str
    sentiment_score: Optional[float]
    magnitude_estimate: str


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class HistoryManager:
    """
    Persists predictions and classified events to the shared SQLite database,
    and resolves prediction outcomes once the actual future price move is
    known. build_calibration_dataset() is the key integration point: it turns
    real resolved predictions into the (confidence, correct) table that
    runtime_validator.py's calibration check needs — replacing the synthetic
    data used everywhere else in this build with genuine track record.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        ensure_directories()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=SQLITE_TIMEOUT_SECONDS)

    def _init_db(self) -> None:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    model_predicted_class TEXT NOT NULL,
                    raw_confidence REAL NOT NULL,
                    risk_adjusted_confidence REAL NOT NULL,
                    calibrated_confidence REAL,
                    agreement_fraction REAL NOT NULL,
                    downside_summary TEXT,
                    upside_summary TEXT,
                    reasoning TEXT,
                    global_risk_level TEXT,
                    risk_toggle_enabled INTEGER,
                    is_safe_to_trade_live INTEGER,
                    data_stale INTEGER,
                    suppressed INTEGER,
                    suppression_reasons TEXT,
                    outcome_resolved INTEGER DEFAULT 0,
                    outcome_correct INTEGER,
                    outcome_actual_class TEXT,
                    resolved_at TEXT,
                    horizon TEXT DEFAULT 'INTRADAY',
                    narrative TEXT,
                    dca_ladder TEXT
                )
            """)
            # Try adding new columns if table already exists without them
            try:
                cursor.execute("ALTER TABLE predictions ADD COLUMN horizon TEXT DEFAULT 'INTRADAY'")
            except: pass
            try:
                cursor.execute("ALTER TABLE predictions ADD COLUMN narrative TEXT")
            except: pass
            try:
                cursor.execute("ALTER TABLE predictions ADD COLUMN dca_ladder TEXT")
            except: pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    affected_tickers TEXT NOT NULL,
                    sector TEXT,
                    confidence_in_scope REAL,
                    headline_or_label TEXT,
                    sentiment_score REAL,
                    magnitude_estimate TEXT,
                    recorded_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    strategy_cumulative_return_pct REAL NOT NULL,
                    baseline_cumulative_return_pct REAL NOT NULL,
                    alpha_pct REAL NOT NULL,
                    edge_check_status TEXT NOT NULL,
                    calibration_status TEXT NOT NULL,
                    calibration_ece REAL,
                    is_live_worthy INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, horizon)
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed initializing history database at {self.db_path}: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()

    # -----------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------
    def save_prediction(self, signal: PredictionSignal, narrative: Optional[str] = None, dca_ladder: Optional[dict] = None) -> Optional[int]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            dca_ladder_str = json.dumps(dca_ladder) if dca_ladder else None
            cursor.execute("""
                INSERT INTO predictions (
                    symbol, timestamp, action, model_predicted_class, raw_confidence,
                    risk_adjusted_confidence, calibrated_confidence, agreement_fraction,
                    downside_summary, upside_summary, reasoning, global_risk_level,
                    risk_toggle_enabled, is_safe_to_trade_live, data_stale, suppressed, suppression_reasons,
                    horizon, narrative, dca_ladder
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol, signal.timestamp.isoformat(), signal.action, signal.model_predicted_class,
                signal.raw_confidence, signal.risk_adjusted_confidence, signal.calibrated_confidence,
                signal.agreement_fraction, signal.downside_summary, signal.upside_summary,
                json.dumps(signal.reasoning), signal.global_risk_level, int(signal.risk_toggle_enabled),
                int(signal.is_safe_to_trade_live), int(signal.data_stale), int(signal.suppressed),
                json.dumps(signal.suppression_reasons), signal.horizon, narrative, dca_ladder_str
            ))

            conn.commit()
            prediction_id = cursor.lastrowid
            logger.info(f"Saved prediction #{prediction_id} for {signal.symbol} ({signal.action}).")
            health_registry.report("history_manager", ok=True, detail=f"Saved prediction for {signal.symbol}")
            return prediction_id
        except Exception as e:
            logger.error(f"Failed saving prediction for {signal.symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving prediction", error=str(e))
            return None
        finally:
            if conn is not None:
                conn.close()

    def resolve_outcome(self, prediction_id: int, actual_class: str) -> bool:
        """Marks a prediction resolved once the true future outcome (UP/DOWN/FLAT) is known."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT model_predicted_class FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            if row is None:
                logger.error(f"No prediction found with id={prediction_id} to resolve.")
                return False
            predicted_class = row[0]
            correct = int(predicted_class == actual_class)
            cursor.execute("""
                UPDATE predictions
                SET outcome_resolved = 1, outcome_correct = ?, outcome_actual_class = ?, resolved_at = ?
                WHERE id = ?
            """, (correct, actual_class, datetime.now().isoformat(), prediction_id))
            conn.commit()
            logger.info(f"Resolved prediction #{prediction_id}: predicted={predicted_class}, actual={actual_class}, correct={bool(correct)}")
            health_registry.report("history_manager", ok=True, detail=f"Resolved prediction {prediction_id}")
            return True
        except Exception as e:
            logger.error(f"Failed resolving outcome for prediction #{prediction_id}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed resolving outcome", error=str(e))
            return False
        finally:
            if conn is not None:
                conn.close()

    def get_predictions(self, symbol: Optional[str] = None, limit: int = 200,
                         only_unresolved: bool = False) -> List[PredictionRecord]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = ("SELECT id, symbol, timestamp, action, model_predicted_class, raw_confidence, "
                      "risk_adjusted_confidence, calibrated_confidence, agreement_fraction, outcome_resolved, "
                      "outcome_correct, outcome_actual_class, resolved_at, horizon, narrative, dca_ladder FROM predictions WHERE 1=1")
            params = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if only_unresolved:
                query += " AND outcome_resolved = 0"
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            records = []
            for row in rows:

                records.append(PredictionRecord(
                    id=row[0], symbol=row[1], timestamp=row[2], action=row[3], model_predicted_class=row[4],
                    raw_confidence=row[5], risk_adjusted_confidence=row[6], calibrated_confidence=row[7],
                    agreement_fraction=row[8], outcome_resolved=bool(row[9]),
                    outcome_correct=None if row[10] is None else bool(row[10]),
                    outcome_actual_class=row[11], resolved_at=row[12],
                    horizon=row[13] if len(row) > 13 else 'INTRADAY',
                    narrative=row[14] if len(row) > 14 else None,
                    dca_ladder=row[15] if len(row) > 15 else None
                ))

            health_registry.report("history_manager", ok=True)
            return records
        except Exception as e:
            logger.error(f"Failed fetching predictions for symbol={symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed fetching predictions", error=str(e))
            return []
        finally:
            if conn is not None:
                conn.close()

    def build_calibration_dataset(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Builds the (confidence, correct) DataFrame that runtime_validator.py's
        compute_calibration() expects, from REAL resolved predictions. Uses
        risk_adjusted_confidence (the confidence actually shown pre-calibration)
        rather than raw_confidence, since that's what's being calibrated.
        """
        try:
            records = self.get_predictions(symbol=symbol, limit=100_000, only_unresolved=False)
            resolved = [r for r in records if r.outcome_resolved and r.outcome_correct is not None]
            if not resolved:
                health_registry.report("history_manager", ok=True)
                return pd.DataFrame(columns=["confidence", "correct"])
            df = pd.DataFrame({
                "confidence": [r.risk_adjusted_confidence for r in resolved],
                "correct": [r.outcome_correct for r in resolved],
            })
            health_registry.report("history_manager", ok=True)
            return df
        except Exception as e:
            logger.error(f"Failed building calibration dataset for symbol={symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed building calibration dataset", error=str(e))
            return pd.DataFrame(columns=["confidence", "correct"])

    # -----------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------
    def save_event(self, event: Event) -> Optional[int]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            # Wrapped in delimiters so a later LIKE '%,SYMBOL,%' match can't false-positive on substrings.
            tickers_str = TICKER_DELIMITER + TICKER_DELIMITER.join(event.affected_tickers) + TICKER_DELIMITER
            cursor.execute("""
                INSERT INTO events (
                    event_id, source, event_type, timestamp, scope, affected_tickers, sector,
                    confidence_in_scope, headline_or_label, sentiment_score, magnitude_estimate, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.source, event.event_type, event.timestamp.isoformat(), event.scope,
                tickers_str, event.sector, event.confidence_in_scope, event.headline_or_label,
                event.sentiment_score, event.magnitude_estimate, datetime.now().isoformat(),
            ))
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"Saved event #{row_id} ({event.event_type}, scope={event.scope}).")
            health_registry.report("history_manager", ok=True, detail=f"Saved event {event.event_id}")
            return row_id
        except Exception as e:
            logger.error(f"Failed saving event {event.event_id}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving event", error=str(e))
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_events_for_symbol(self, symbol: str, limit: int = 100) -> List[EventRecord]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            # Delimiter-wrapped LIKE match: ',SYMBOL,' must appear exactly, so 'TCS' can never
            # accidentally match a row whose tickers include something like 'CTCS' or 'TCSX'.
            pattern = f"%{TICKER_DELIMITER}{symbol}{TICKER_DELIMITER}%"
            cursor.execute("""
                SELECT id, event_id, source, event_type, timestamp, scope, affected_tickers, sector,
                       confidence_in_scope, headline_or_label, sentiment_score, magnitude_estimate
                FROM events WHERE affected_tickers LIKE ? ORDER BY id DESC LIMIT ?
            """, (pattern, limit))
            rows = cursor.fetchall()
            records = []
            for row in rows:
                tickers = [t for t in row[6].strip(TICKER_DELIMITER).split(TICKER_DELIMITER) if t]
                records.append(EventRecord(
                    id=row[0], event_id=row[1], source=row[2], event_type=row[3], timestamp=row[4],
                    scope=row[5], affected_tickers=tickers, sector=row[7], confidence_in_scope=row[8],
                    headline_or_label=row[9], sentiment_score=row[10], magnitude_estimate=row[11],
                ))
            health_registry.report("history_manager", ok=True)
            return records
        except Exception as e:
            logger.error(f"Failed fetching events for symbol={symbol}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed fetching events", error=str(e))
            return []
        finally:
            if conn is not None:
                conn.close()

    # -----------------------------------------------------------------
    # Backtest Metrics
    # -----------------------------------------------------------------
    def save_backtest_result(self, symbol: str, horizon: str, strategy_ret: float, base_ret: float, alpha: float, edge: str, calib: str, ece: Optional[float], live_worthy: bool) -> bool:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtest_metrics (
                    symbol, horizon, strategy_cumulative_return_pct, baseline_cumulative_return_pct,
                    alpha_pct, edge_check_status, calibration_status, calibration_ece, is_live_worthy, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, horizon) DO UPDATE SET
                    strategy_cumulative_return_pct=excluded.strategy_cumulative_return_pct,
                    baseline_cumulative_return_pct=excluded.baseline_cumulative_return_pct,
                    alpha_pct=excluded.alpha_pct,
                    edge_check_status=excluded.edge_check_status,
                    calibration_status=excluded.calibration_status,
                    calibration_ece=excluded.calibration_ece,
                    is_live_worthy=excluded.is_live_worthy,
                    updated_at=excluded.updated_at
            """, (
                symbol, horizon, strategy_ret, base_ret, alpha, edge, calib, ece, int(live_worthy), datetime.now().isoformat()
            ))
            conn.commit()
            health_registry.report("history_manager", ok=True, detail=f"Saved backtest metrics for {symbol} {horizon}")
            return True
        except Exception as e:
            logger.error(f"Failed saving backtest metrics for {symbol} {horizon}: {e}")
            health_registry.report("history_manager", ok=False, detail="Failed saving backtest metrics", error=str(e))
            return False
        finally:
            if conn is not None:
                conn.close()


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from config import DB_DIR

    configure_logging(log_filename="history_manager_selftest.log")
    logger.info("Running history_manager.py self-test...")

    test_db_path = DB_DIR / "test_history_selftest.sqlite3"
    if test_db_path.exists():
        os.remove(test_db_path)

    test_symbol = "M&M"  # deliberately chosen: shares no risky substrings with other NIFTY50 tickers,
                          # a good stress test for the delimiter-wrapped LIKE matching below

    try:
        print("\n=== HISTORY MANAGER SELF-TEST RESULT ===")
        manager = HistoryManager(db_path=test_db_path)

        # --- Predictions + outcome resolution ---
        signal = PredictionSignal(
            symbol=test_symbol, timestamp=datetime.now(), action="BUY", model_predicted_class="UP",
            raw_confidence=0.72, risk_adjusted_confidence=0.70, calibrated_confidence=None, agreement_fraction=0.66,
            downside_summary="Some downside.", upside_summary="Some upside.", reasoning=["Model said UP."],
        )
        pred_id = manager.save_prediction(signal)
        print(f"Prediction saved: id={pred_id}")
        assert pred_id is not None

        resolved_ok = manager.resolve_outcome(pred_id, actual_class="UP")
        predictions = manager.get_predictions(test_symbol)
        print(f"Outcome resolved: {resolved_ok}, outcome_correct={predictions[0].outcome_correct}")
        assert resolved_ok and predictions[0].outcome_correct is True

        # Save several more predictions with resolved outcomes to build a calibration dataset
        pattern = [(0.9, "UP", "UP"), (0.9, "UP", "DOWN"), (0.3, "DOWN", "UP"), (0.6, "FLAT", "FLAT")]
        for conf, predicted, actual in pattern:
            s = PredictionSignal(
                symbol=test_symbol, timestamp=datetime.now(), action="BUY", model_predicted_class=predicted,
                raw_confidence=conf, risk_adjusted_confidence=conf, calibrated_confidence=None, agreement_fraction=0.5,
                downside_summary="d", upside_summary="u", reasoning=[],
            )
            pid = manager.save_prediction(s)
            manager.resolve_outcome(pid, actual_class=actual)

        calibration_df = manager.build_calibration_dataset(test_symbol)
        print(f"Calibration dataset built: {len(calibration_df)} rows, columns={list(calibration_df.columns)}")
        assert len(calibration_df) == 5  # the first one + 4 more
        assert set(calibration_df.columns) == {"confidence", "correct"}
        assert calibration_df["correct"].sum() == 3  # UP/UP (first) + UP/UP + FLAT/FLAT = 3 correct

        # --- Events + delimiter-safe symbol matching ---
        event_for_symbol = Event(
            event_id="EVT_TEST_1", source="CORPORATE", event_type="CORPORATE_ANNOUNCEMENT",
            timestamp=datetime.now(), scope="STOCK", affected_tickers=[test_symbol, "HDFCBANK"],
            sector="Auto", confidence_in_scope=1.0, headline_or_label="Board meeting for M&M",
            sentiment_score=0.1, magnitude_estimate="MEDIUM",
        )
        event_not_for_symbol = Event(
            event_id="EVT_TEST_2", source="CORPORATE", event_type="CORPORATE_ANNOUNCEMENT",
            timestamp=datetime.now(), scope="STOCK", affected_tickers=["MARUTI", "TMPV"],
            sector="Auto", confidence_in_scope=1.0, headline_or_label="Board meeting for Maruti",
            sentiment_score=0.1, magnitude_estimate="MEDIUM",
        )
        manager.save_event(event_for_symbol)
        manager.save_event(event_not_for_symbol)

        events_for_symbol = manager.get_events_for_symbol(test_symbol)
        print(f"Events correctly matched for {test_symbol}: {len(events_for_symbol)} "
              f"(expected 1, must NOT include the Maruti-only event)")
        assert len(events_for_symbol) == 1
        assert events_for_symbol[0].event_id == "EVT_TEST_1"

        print("STATUS: PASS")
        logger.info("history_manager.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"history_manager.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"history_manager.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
    finally:
        if test_db_path.exists():
            os.remove(test_db_path)
