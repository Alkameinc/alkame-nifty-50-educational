"""
Shared helper so every notebook connects to a SQLite database the exact same
way, and so switching from sample data to the real live database is a
one-line change in each notebook, not a copy-pasted block re-written three times.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH  # noqa: E402 — the REAL live database path, from config.py

SAMPLE_DB_PATH = Path(__file__).resolve().parent / "sample_history.sqlite3"


def load_predictions(db_path: Path = SAMPLE_DB_PATH) -> pd.DataFrame:
    """Loads the full `predictions` table as a DataFrame. Pass db_path=DB_PATH
    (imported above) once you want to analyze the real live history instead
    of the synthetic sample."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        conn.close()


def load_events(db_path: Path = SAMPLE_DB_PATH) -> pd.DataFrame:
    """Loads the full `events` table as a DataFrame."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM events", conn)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        conn.close()