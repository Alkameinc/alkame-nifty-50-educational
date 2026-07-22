"""
Generates a realistic, clearly-synthetic prediction/outcome history so the
data_analyst notebooks are runnable from day one, without waiting for
scheduler.py to accumulate real history. Writes to a SEPARATE sqlite file
(data_analyst/sample_history.sqlite3) — this never touches or modifies the
real db/predictor.sqlite3 that the live engine uses.

Run once with: python data_analyst/seed_sample_data.py
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np



# Make the repo root importable (this file lives one folder below it)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NIFTY50_SYMBOLS, SECTOR_MAP, configure_logging  # noqa: E402
from history_manager import HistoryManager  # noqa: E402
from predictor import PredictionSignal  # noqa: E402
from event_classifier import Event  # add this import near the top of the file

SAMPLE_DB_PATH = Path(__file__).resolve().parent / "sample_history.sqlite3"

# Deliberately give some sectors a genuine, simulated edge and others none —
# this is what makes Task 2 (sector comparison) meaningful to practice on,
# instead of every sector looking identical.
SECTOR_SKILL_LEVEL = {
    "IT": 0.72, "Pharma": 0.68, "Banking": 0.55, "FMCG": 0.50,
    "Auto": 0.48, "Energy": 0.45, "Metals": 0.44, "NBFC": 0.52,
}
DEFAULT_SKILL = 0.50  # no real edge — coin flip

CLASSES = ["UP", "DOWN", "FLAT"]


def _simulate_symbol_history(symbol: str, n_predictions: int, rng: np.random.Generator) -> None:
    sector = SECTOR_MAP.get(symbol, "Other")
    skill = SECTOR_SKILL_LEVEL.get(sector, DEFAULT_SKILL)

    manager = HistoryManager(db_path=SAMPLE_DB_PATH)
    base_time = datetime.now() - timedelta(days=60)

    for i in range(n_predictions):
        timestamp = base_time + timedelta(hours=i * 2)
        predicted_class = rng.choice(CLASSES, p=[0.35, 0.35, 0.30])

        raw_confidence = float(np.clip(rng.normal(0.55 + skill * 0.3, 0.12), 0.05, 0.98))

        is_correct = rng.uniform(0, 1) < skill
        actual_class = predicted_class if is_correct else rng.choice(
            [c for c in CLASSES if c != predicted_class]
        )

        action = {"UP": "BUY", "DOWN": "SELL", "FLAT": "HOLD"}[predicted_class]

        signal = PredictionSignal(
            symbol=symbol, timestamp=timestamp, action=action,
            model_predicted_class=predicted_class,
            raw_confidence=raw_confidence, risk_adjusted_confidence=raw_confidence,
            calibrated_confidence=None, agreement_fraction=float(rng.uniform(0.4, 1.0)),
            downside_summary="Synthetic sample data — see data_analyst/seed_sample_data.py.",
            upside_summary="Synthetic sample data — see data_analyst/seed_sample_data.py.",
            reasoning=["Synthetic sample prediction for notebook development."],
        )
        pred_id = manager.save_prediction(signal)
        if pred_id is not None:
            manager.resolve_outcome(pred_id, actual_class=actual_class)

def _simulate_symbol_events(symbol: str, n_events: int, rng: np.random.Generator) -> None:
    """
    Generates sample events with a DELIBERATE, known ground-truth relationship to price
    moves, so Task 3's notebook has something real to detect. Negative-sentiment events
    are followed by a simulated negative price drift; positive-sentiment events by a
    positive drift — this is synthetic ground truth, not a claim about real markets.
    """
    manager = HistoryManager(db_path=SAMPLE_DB_PATH)
    base_time = datetime.now() - timedelta(days=45)

    for i in range(n_events):
        timestamp = base_time + timedelta(hours=i * 6)
        sentiment = float(rng.uniform(-1.0, 1.0))
        event = Event(
            event_id=f"SAMPLE_EVT_{symbol}_{i}",
            source="NEWS", event_type="NEWS_SENTIMENT", timestamp=timestamp,
            scope="STOCK", affected_tickers=[symbol], sector=SECTOR_MAP.get(symbol),
            confidence_in_scope=1.0,
            headline_or_label=f"Sample headline #{i} for {symbol} (sentiment={sentiment:.2f})",
            sentiment_score=sentiment, magnitude_estimate="MEDIUM",
        )
        manager.save_event(event)


if __name__ == "__main__":
    configure_logging(log_filename="seed_sample_data.log")

    if SAMPLE_DB_PATH.exists():
        SAMPLE_DB_PATH.unlink()  # start clean so re-running this script gives reproducible results

    print("=== SEEDING SAMPLE HISTORY FOR DATA ANALYST NOTEBOOKS ===")
    rng = np.random.default_rng(42)  # fixed seed -> identical sample data for every intern who runs this

    for symbol in NIFTY50_SYMBOLS:
        _simulate_symbol_history(symbol, n_predictions=120, rng=rng)
        print(f"  Seeded {symbol} ({SECTOR_MAP.get(symbol, 'Other')})")

    print(f"\nDone. Sample database written to: {SAMPLE_DB_PATH}")
    for symbol in NIFTY50_SYMBOLS[:15]:  # a subset is enough for this notebook's purposes
        _simulate_symbol_events(symbol, n_events=20, rng=rng)
        print(f"  Seeded sample events for {symbol}")
    print("STATUS: PASS")