# 1. Standard library imports
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 2. Third-party imports
# (none required — stdlib sqlite3 only)

# 3. Local imports
from config import DB_PATH, ensure_directories, configure_logging
from predictor import PredictionSignal
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
SQLITE_TIMEOUT_SECONDS = 10
DEFAULT_CREATED_BY = "default_trader"


@dataclass
class NoteRecord:
    id: int
    symbol: str
    timestamp: str
    note_text: str
    related_action: Optional[str]
    created_by: str


@dataclass
class OverrideRecord:
    id: int
    symbol: str
    timestamp: str
    original_action: str
    overridden_action: str
    reason: str
    created_by: str


@dataclass
class FeedbackRecord:
    id: int
    symbol: str
    signal_timestamp: str
    original_action: str
    was_helpful: Optional[bool]
    outcome_notes: str
    rated_at: str


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class HumanInsightManager:
    """
    Stores everything a human trader adds on top of the model's own output:
    free-text notes, explicit overrides of a signal's final action (with a
    mandatory reason, same audit-trail discipline as the global risk toggle),
    and feedback on whether a past signal was actually helpful. All of this
    is the raw material a future calibration/retraining pass would use — this
    file only handles storage and retrieval, not automatic model updates.
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
                CREATE TABLE IF NOT EXISTS human_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    note_text TEXT NOT NULL,
                    related_action TEXT,
                    created_by TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS human_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    original_action TEXT NOT NULL,
                    overridden_action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_by TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_timestamp TEXT NOT NULL,
                    original_action TEXT NOT NULL,
                    was_helpful INTEGER,
                    outcome_notes TEXT,
                    rated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed initializing human insight database at {self.db_path}: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()

    # -----------------------------------------------------------------
    # Notes
    # -----------------------------------------------------------------
    def add_note(self, symbol: str, note_text: str, related_action: Optional[str] = None,
                 created_by: str = DEFAULT_CREATED_BY) -> Optional[int]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO human_notes (symbol, timestamp, note_text, related_action, created_by) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, datetime.now().isoformat(), note_text, related_action, created_by),
            )
            conn.commit()
            note_id = cursor.lastrowid
            logger.info(f"Added note #{note_id} for {symbol}: {note_text[:60]}")
            health_registry.report("human_insight_manager", ok=True, detail=f"Added note for {symbol}")
            return note_id
        except Exception as e:
            logger.error(f"Failed adding note for {symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed adding note", error=str(e))
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_notes(self, symbol: Optional[str] = None, limit: int = 50) -> List[NoteRecord]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT id, symbol, timestamp, note_text, related_action, created_by "
                    "FROM human_notes WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, symbol, timestamp, note_text, related_action, created_by "
                    "FROM human_notes ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            health_registry.report("human_insight_manager", ok=True)
            return [NoteRecord(*row) for row in rows]
        except Exception as e:
            logger.error(f"Failed fetching notes for symbol={symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed fetching notes", error=str(e))
            return []
        finally:
            if conn is not None:
                conn.close()

    # -----------------------------------------------------------------
    # Overrides
    # -----------------------------------------------------------------
    def record_override(self, symbol: str, original_action: str, overridden_action: str, reason: str,
                         created_by: str = DEFAULT_CREATED_BY) -> Optional[int]:
        if not reason or not reason.strip():
            logger.error(f"Refusing to record override for {symbol} without a reason — reason is mandatory.")
            return None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO human_overrides (symbol, timestamp, original_action, overridden_action, reason, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, datetime.now().isoformat(), original_action, overridden_action, reason, created_by),
            )
            conn.commit()
            override_id = cursor.lastrowid
            logger.info(
                f"Recorded override #{override_id} for {symbol}: {original_action} -> {overridden_action} "
                f"(reason: {reason[:60]})"
            )
            health_registry.report("human_insight_manager", ok=True, detail=f"Recorded override for {symbol}")
            return override_id
        except Exception as e:
            logger.error(f"Failed recording override for {symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed recording override", error=str(e))
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_overrides(self, symbol: Optional[str] = None, limit: int = 50) -> List[OverrideRecord]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT id, symbol, timestamp, original_action, overridden_action, reason, created_by "
                    "FROM human_overrides WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, symbol, timestamp, original_action, overridden_action, reason, created_by "
                    "FROM human_overrides ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            health_registry.report("human_insight_manager", ok=True)
            return [OverrideRecord(*row) for row in rows]
        except Exception as e:
            logger.error(f"Failed fetching overrides for symbol={symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed fetching overrides", error=str(e))
            return []
        finally:
            if conn is not None:
                conn.close()

    def apply_override_to_signal(
        self, signal: PredictionSignal, overridden_action: str, reason: str, created_by: str = DEFAULT_CREATED_BY,
    ) -> PredictionSignal:
        """
        Records the override for audit purposes AND returns a new
        PredictionSignal reflecting the human's final call — the model's
        original output is preserved in reasoning, never silently erased.
        """
        self.record_override(signal.symbol, signal.action, overridden_action, reason, created_by)
        updated_reasoning = list(signal.reasoning) + [
            f"HUMAN OVERRIDE by {created_by}: changed action from {signal.action} to {overridden_action}. "
            f"Reason: {reason}"
        ]
        return PredictionSignal(
            symbol=signal.symbol, timestamp=signal.timestamp, action=overridden_action,
            model_predicted_class=signal.model_predicted_class, raw_confidence=signal.raw_confidence,
            risk_adjusted_confidence=signal.risk_adjusted_confidence,
            calibrated_confidence=signal.calibrated_confidence, agreement_fraction=signal.agreement_fraction,
            downside_summary=signal.downside_summary, upside_summary=signal.upside_summary,
            reasoning=updated_reasoning, contributing_events=signal.contributing_events,
            global_risk_level=signal.global_risk_level, risk_toggle_enabled=signal.risk_toggle_enabled,
            is_safe_to_trade_live=signal.is_safe_to_trade_live, data_stale=signal.data_stale,
            suppressed=signal.suppressed, suppression_reasons=signal.suppression_reasons,
        )

    # -----------------------------------------------------------------
    # Feedback
    # -----------------------------------------------------------------
    def record_feedback(self, symbol: str, signal_timestamp: str, original_action: str,
                         was_helpful: Optional[bool], outcome_notes: str = "") -> Optional[int]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            helpful_int = None if was_helpful is None else int(was_helpful)
            cursor.execute(
                "INSERT INTO signal_feedback (symbol, signal_timestamp, original_action, was_helpful, "
                "outcome_notes, rated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, signal_timestamp, original_action, helpful_int, outcome_notes, datetime.now().isoformat()),
            )
            conn.commit()
            feedback_id = cursor.lastrowid
            logger.info(f"Recorded feedback #{feedback_id} for {symbol} ({original_action}): helpful={was_helpful}")
            health_registry.report("human_insight_manager", ok=True, detail=f"Recorded feedback for {symbol}")
            return feedback_id
        except Exception as e:
            logger.error(f"Failed recording feedback for {symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed recording feedback", error=str(e))
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_feedback(self, symbol: Optional[str] = None, limit: int = 50) -> List[FeedbackRecord]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT id, symbol, signal_timestamp, original_action, was_helpful, outcome_notes, rated_at "
                    "FROM signal_feedback WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, symbol, signal_timestamp, original_action, was_helpful, outcome_notes, rated_at "
                    "FROM signal_feedback ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            records = []
            for row in rows:
                helpful = None if row[4] is None else bool(row[4])
                records.append(FeedbackRecord(row[0], row[1], row[2], row[3], helpful, row[5], row[6]))
            health_registry.report("human_insight_manager", ok=True)
            return records
        except Exception as e:
            logger.error(f"Failed fetching feedback for symbol={symbol}: {e}")
            health_registry.report("human_insight_manager", ok=False, detail="Failed fetching feedback", error=str(e))
            return []
        finally:
            if conn is not None:
                conn.close()

    def get_feedback_summary(self, symbol: Optional[str] = None) -> dict:
        records = self.get_feedback(symbol=symbol, limit=10_000)
        rated = [r for r in records if r.was_helpful is not None]
        if not rated:
            return {"total_feedback": len(records), "total_rated": 0, "helpful_pct": None}
        helpful_count = sum(1 for r in rated if r.was_helpful)
        return {
            "total_feedback": len(records),
            "total_rated": len(rated),
            "helpful_pct": round(100.0 * helpful_count / len(rated), 1),
        }


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from config import DB_DIR

    configure_logging(log_filename="human_insight_manager_selftest.log")
    logger.info("Running human_insight_manager.py self-test...")

    test_db_path = DB_DIR / "test_human_insights_selftest.sqlite3"
    if test_db_path.exists():
        os.remove(test_db_path)  # start clean so counts below are deterministic

    test_symbol = "RELIANCE"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== HUMAN INSIGHT MANAGER SELF-TEST RESULT ===")
        manager = HumanInsightManager(db_path=test_db_path)

        # Notes
        note_id = manager.add_note(test_symbol, "Watching for RBI policy reaction this week.", related_action="HOLD")
        notes = manager.get_notes(test_symbol)
        print(f"Note added and retrieved: id={note_id}, count={len(notes)}, text='{notes[0].note_text if notes else None}'")
        assert note_id is not None and len(notes) == 1

        # Overrides — mandatory reason enforcement
        rejected_override = manager.record_override(test_symbol, "BUY", "HOLD", reason="")
        print(f"Override with empty reason correctly rejected: {rejected_override is None}")
        assert rejected_override is None

        override_id = manager.record_override(
            test_symbol, "BUY", "HOLD", reason="Waiting for board meeting outcome before acting."
        )
        overrides = manager.get_overrides(test_symbol)
        print(f"Valid override recorded: id={override_id}, count={len(overrides)}, "
              f"{overrides[0].original_action if overrides else None} -> {overrides[0].overridden_action if overrides else None}")
        assert override_id is not None and len(overrides) == 1

        # apply_override_to_signal — build a minimal fake PredictionSignal and override it
        fake_signal = PredictionSignal(
            symbol=test_symbol, timestamp=datetime.now(), action="BUY", model_predicted_class="UP",
            raw_confidence=0.7, risk_adjusted_confidence=0.7, calibrated_confidence=0.65, agreement_fraction=0.66,
            downside_summary="Some downside.", upside_summary="Some upside.", reasoning=["Model said BUY."],
        )
        overridden_signal = manager.apply_override_to_signal(
            fake_signal, overridden_action="HOLD", reason="Human wants to wait for confirmation."
        )
        print(f"Signal action after override: {overridden_signal.action} (was {fake_signal.action})")
        print(f"Override reasoning preserved in signal: "
              f"{'HUMAN OVERRIDE' in overridden_signal.reasoning[-1]}")
        assert overridden_signal.action == "HOLD"
        assert "HUMAN OVERRIDE" in overridden_signal.reasoning[-1]

        # Feedback + summary calculation
        manager.record_feedback(test_symbol, datetime.now().isoformat(), "BUY", was_helpful=True, outcome_notes="Worked out well.")
        manager.record_feedback(test_symbol, datetime.now().isoformat(), "SELL", was_helpful=False, outcome_notes="Missed the reversal.")
        manager.record_feedback(test_symbol, datetime.now().isoformat(), "HOLD", was_helpful=None, outcome_notes="Not rated yet.")
        summary = manager.get_feedback_summary(test_symbol)
        print(f"Feedback summary: {summary}")
        assert summary["total_feedback"] == 3
        assert summary["total_rated"] == 2
        assert summary["helpful_pct"] == 50.0  # 1 of 2 rated was helpful

        print("STATUS: PASS")
        logger.info("human_insight_manager.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"human_insight_manager.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"human_insight_manager.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
    finally:
        if test_db_path.exists():
            os.remove(test_db_path)  # leave no test artifacts behind
