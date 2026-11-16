#!/usr/bin/env python3
"""
Database Backup & Maintenance Script for Alkame Nifty50.
Creates timestamped backups of SQLite/PostgreSQL databases before migrations or restarts.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DB_DIR, DB_PATH, ensure_directories

BACKUP_DIR = DB_DIR / "backups"


def create_backup() -> Path | None:
    """Creates a timestamped snapshot backup of the current database file."""
    ensure_directories()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if using SQLite
    if DB_PATH.exists():
        backup_filename = f"predictor_backup_{timestamp}.sqlite3"
        target_path = BACKUP_DIR / backup_filename
        try:
            shutil.copy2(DB_PATH, target_path)
            print(f"[SUCCESS] SQLite database backed up to: {target_path}")
            return target_path
        except Exception as e:
            print(f"[ERROR] Failed to backup SQLite database: {e}", file=sys.stderr)
            return None
    else:
        print(f"[WARNING] Database file {DB_PATH} not found. Skipping SQLite backup.")
        return None


def cleanup_old_backups(keep_count: int = 10) -> None:
    """Retains only the N most recent backups to prevent disk space exhaustion."""
    if not BACKUP_DIR.exists():
        return

    backups = sorted(BACKUP_DIR.glob("predictor_backup_*.sqlite3"), key=os.path.getmtime)
    if len(backups) > keep_count:
        to_delete = backups[:-keep_count]
        for old_backup in to_delete:
            try:
                old_backup.unlink()
                print(f"[CLEANUP] Deleted old backup: {old_backup.name}")
            except Exception as e:
                print(f"[WARNING] Failed deleting old backup {old_backup.name}: {e}")


if __name__ == "__main__":
    print(f"=== Starting Database Backup Process [{datetime.now().isoformat()}] ===")
    backup_file = create_backup()
    if backup_file:
        cleanup_old_backups(keep_count=10)
        print("=== Database Backup Process Completed Successfully ===")
    else:
        print("=== Database Backup Completed with Warnings/Errors ===")
