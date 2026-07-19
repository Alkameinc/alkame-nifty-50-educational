# 1. Standard library imports
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# 2. Third-party imports
import pandas as pd

# 3. Local imports
from config import (
    DB_PATH,
    HEALTH_DEGRADED_THRESHOLD,
    HEALTH_DOWN_THRESHOLD,
    HEALTH_THRESHOLD_OVERRIDES,
    configure_logging
)

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class HealthStatus:
    component: str
    status: str            # "OK" | "DEGRADED" | "DOWN"
    last_success_at: Optional[datetime]
    last_error: Optional[str]
    last_error_at: Optional[datetime]
    consecutive_failures: int
    detail: str            # free-text

# ---------------------------------------------------------------------------
# 6. Classes
# ---------------------------------------------------------------------------
class HealthRegistry:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS health_status (
                        component TEXT PRIMARY KEY,
                        status TEXT,
                        last_success_at TEXT,
                        last_error TEXT,
                        last_error_at TEXT,
                        consecutive_failures INTEGER,
                        detail TEXT
                    )
                """)
        except Exception as e:
            logger.error(f"Failed to initialize health_status table: {e}")

    def report(self, component: str, ok: bool, detail: str = "", error: Optional[str] = None) -> HealthStatus:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM health_status WHERE component = ?", (component,))
                row = cursor.fetchone()
                
                now_str = datetime.now().isoformat()
                
                if row:
                    _, curr_status, curr_last_success, curr_last_error, curr_last_error_at, curr_failures, curr_detail = row
                    consecutive_failures = 0 if ok else (curr_failures + 1)
                    last_success_at = now_str if ok else curr_last_success
                    last_error = error if not ok else curr_last_error
                    last_error_at = now_str if not ok else curr_last_error_at
                else:
                    consecutive_failures = 0 if ok else 1
                    last_success_at = now_str if ok else None
                    last_error = error if not ok else None
                    last_error_at = now_str if not ok else None
                
                # Determine status
                overrides = HEALTH_THRESHOLD_OVERRIDES.get(component, {})
                degraded_thresh = overrides.get("degraded", HEALTH_DEGRADED_THRESHOLD)
                down_thresh = overrides.get("down", HEALTH_DOWN_THRESHOLD)
                
                if consecutive_failures >= down_thresh:
                    status = "DOWN"
                elif consecutive_failures >= degraded_thresh:
                    status = "DEGRADED"
                else:
                    status = "OK"

                cursor.execute("""
                    INSERT INTO health_status (component, status, last_success_at, last_error, last_error_at, consecutive_failures, detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(component) DO UPDATE SET
                        status = excluded.status,
                        last_success_at = excluded.last_success_at,
                        last_error = excluded.last_error,
                        last_error_at = excluded.last_error_at,
                        consecutive_failures = excluded.consecutive_failures,
                        detail = excluded.detail
                """, (component, status, last_success_at, last_error, last_error_at, consecutive_failures, detail))
                conn.commit()
                
                def parse_dt(dt_str):
                    return datetime.fromisoformat(dt_str) if dt_str else None
                
                return HealthStatus(
                    component=component,
                    status=status,
                    last_success_at=parse_dt(last_success_at),
                    last_error=last_error,
                    last_error_at=parse_dt(last_error_at),
                    consecutive_failures=consecutive_failures,
                    detail=detail
                )
        except Exception as e:
            logger.error(f"Failed to report health status for {component}: {e}")
            return HealthStatus(
                component=component, status="UNKNOWN", last_success_at=None,
                last_error=str(e), last_error_at=datetime.now(), consecutive_failures=0, detail="Failed to write to DB"
            )

    def get_status(self, component: Optional[str] = None) -> List[HealthStatus]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if component:
                    cursor.execute("SELECT * FROM health_status WHERE component = ?", (component,))
                    rows = cursor.fetchall()
                else:
                    cursor.execute("SELECT * FROM health_status")
                    rows = cursor.fetchall()
                
                statuses = []
                for row in rows:
                    comp, stat, ls, le, lea, cf, det = row
                    statuses.append(HealthStatus(
                        component=comp,
                        status=stat,
                        last_success_at=datetime.fromisoformat(ls) if ls else None,
                        last_error=le,
                        last_error_at=datetime.fromisoformat(lea) if lea else None,
                        consecutive_failures=cf,
                        detail=det
                    ))
                return statuses
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return []

    def get_overall_status(self) -> str:
        statuses = self.get_status()
        if not statuses:
            return "OK"
        status_levels = [s.status for s in statuses]
        if "DOWN" in status_levels:
            return "DOWN"
        if "DEGRADED" in status_levels:
            return "DEGRADED"
        return "OK"


# ---------------------------------------------------------------------------
# 7. Global Instance
# ---------------------------------------------------------------------------
registry = HealthRegistry()

# ---------------------------------------------------------------------------
# 8. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from config import DB_DIR
    
    configure_logging(log_filename="health_monitor_selftest.log")
    logger.info("Running health_monitor.py self-test...")
    
    test_db_path = DB_DIR / "test_health_selftest.sqlite3"
    if test_db_path.exists():
        os.remove(test_db_path)
        
    try:
        registry = HealthRegistry(db_path=test_db_path)
        
        # Test: report a success, confirm status == "OK"
        st = registry.report("test_comp", ok=True, detail="first success")
        assert st.status == "OK"
        assert st.consecutive_failures == 0
        
        # Test: report HEALTH_DEGRADED_THRESHOLD consecutive failures, confirm status == "DEGRADED"
        for i in range(HEALTH_DEGRADED_THRESHOLD):
            st = registry.report("test_comp", ok=False, error=f"error {i}")
        assert st.status == "DEGRADED"
        assert st.consecutive_failures == HEALTH_DEGRADED_THRESHOLD
        
        # Test: report up to HEALTH_DOWN_THRESHOLD, confirm status == "DOWN"
        for i in range(HEALTH_DOWN_THRESHOLD - HEALTH_DEGRADED_THRESHOLD):
            st = registry.report("test_comp", ok=False, error=f"error {HEALTH_DEGRADED_THRESHOLD+i}")
        assert st.status == "DOWN"
        assert st.consecutive_failures == HEALTH_DOWN_THRESHOLD
        
        # Test: report a success again, confirm consecutive_failures resets to 0 and status returns to "OK"
        st = registry.report("test_comp", ok=True, detail="recovered")
        assert st.status == "OK"
        assert st.consecutive_failures == 0
        
        # Test: confirm get_overall_status correctly returns the worst status
        registry.report("comp1", ok=False)  # assuming 1 is OK because degraded is 2 by default
        registry.report("comp2", ok=False) 
        registry.report("comp2", ok=False)  # comp2 is now DEGRADED
        assert registry.get_overall_status() == "DEGRADED"
        
        for i in range(HEALTH_DOWN_THRESHOLD):
            registry.report("comp3", ok=False) # comp3 is now DOWN
            
        assert registry.get_overall_status() == "DOWN"
        
        print("STATUS: PASS")
        logger.info("health_monitor.py self-test passed.")
    except AssertionError as ae:
        logger.error(f"health_monitor.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"health_monitor.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
    finally:
        if test_db_path.exists():
            try:
                os.remove(test_db_path)
            except:
                pass
