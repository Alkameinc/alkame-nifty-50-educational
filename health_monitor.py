# 1. Standard library imports
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy import text as sa_text

# 2. Third-party imports
# 3. Local imports
from config import (
    HEALTH_DEGRADED_THRESHOLD,
    HEALTH_DOWN_THRESHOLD,
    HEALTH_THRESHOLD_OVERRIDES,
    configure_logging,
)
from database import SessionLocal
from models import HealthStatus as DBHealthStatus

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants & Dataclasses (MODEL-004)
# ---------------------------------------------------------------------------
ENGINE_HEALTH_HEALTHY = "HEALTHY"
ENGINE_HEALTH_DEGRADED = "DEGRADED"
ENGINE_HEALTH_FAILED = "FAILED"


@dataclass
class HealthStatus:
    component: str
    status: str  # "OK" | "DEGRADED" | "DOWN"
    last_success_at: datetime | None
    last_error: str | None
    last_error_at: datetime | None
    consecutive_failures: int
    detail: str  # free-text


@dataclass
class EngineHealthResult:
    """
    MODEL-004: Evaluates operational infrastructure across 6 distinct checks:
    api, data, events, scheduler, storage, model_file_availability.
    Decoupled from statistical model validity.
    """

    status: str  # HEALTHY | DEGRADED | FAILED
    checks: dict[str, str]  # check name -> "HEALTHY" | "DEGRADED" | "FAILED"
    components: list[HealthStatus]
    summary: str
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 6. Classes
# ---------------------------------------------------------------------------
class HealthRegistry:
    def __init__(self, db_path=None):
        pass

    def report(self, component: str, ok: bool, detail: str = "", error: str | None = None) -> HealthStatus:
        try:
            with SessionLocal() as db:
                row = db.query(DBHealthStatus).filter(DBHealthStatus.component == component).first()

                now_str = datetime.now().isoformat()

                last_success_at: str | None = None
                last_error: str | None = None
                last_error_at: str | None = None
                consecutive_failures: int = 0

                if row:
                    consecutive_failures = 0 if ok else (int(row.consecutive_failures) + 1)
                    last_success_at = now_str if ok else (str(row.last_success_at) if row.last_success_at else None)
                    last_error = error if not ok else (str(row.last_error) if row.last_error else None)
                    last_error_at = now_str if not ok else (str(row.last_error_at) if row.last_error_at else None)
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

                if row:
                    row.consecutive_failures = consecutive_failures  # type: ignore[assignment]
                    row.last_success_at = last_success_at  # type: ignore[assignment]
                    row.last_error = last_error  # type: ignore[assignment]
                    row.last_error_at = last_error_at  # type: ignore[assignment]
                    row.status = status  # type: ignore[assignment]
                    row.detail = detail  # type: ignore[assignment]
                else:
                    row = DBHealthStatus(
                        component=component,
                        status=status,
                        last_success_at=last_success_at,
                        last_error=last_error,
                        last_error_at=last_error_at,
                        consecutive_failures=consecutive_failures,
                        detail=detail,
                    )
                    db.add(row)

                db.commit()

                return HealthStatus(
                    component=component,
                    status=status,
                    last_success_at=datetime.fromisoformat(last_success_at) if last_success_at else None,
                    last_error=last_error,
                    last_error_at=datetime.fromisoformat(last_error_at) if last_error_at else None,
                    consecutive_failures=consecutive_failures,
                    detail=detail,
                )
        except Exception as e:
            logger.error(f"Failed to report health status for {component}: {e}")
            return HealthStatus(
                component=component,
                status="UNKNOWN",
                last_success_at=None,
                last_error=str(e),
                last_error_at=None,
                consecutive_failures=0,
                detail="Failed to write to DB",
            )

    def get_status(self, component: str | None = None) -> list[HealthStatus]:
        try:
            with SessionLocal() as db:
                if component:
                    rows = db.query(DBHealthStatus).filter(DBHealthStatus.component == component).all()
                else:
                    rows = db.query(DBHealthStatus).all()

                statuses = []
                for row in rows:
                    statuses.append(
                        HealthStatus(
                            component=str(row.component),
                            status=str(row.status),
                            last_success_at=(
                                datetime.fromisoformat(str(row.last_success_at)) if row.last_success_at else None
                            ),
                            last_error=str(row.last_error) if row.last_error else None,
                            last_error_at=datetime.fromisoformat(str(row.last_error_at)) if row.last_error_at else None,
                            consecutive_failures=int(row.consecutive_failures),
                            detail=str(row.detail),
                        )
                    )
                return statuses
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return []

    def get_overall_status(self) -> str:
        statuses = self.get_status()
        if not statuses:
            return "UNKNOWN"
        status_levels = [s.status for s in statuses]
        if "DOWN" in status_levels:
            return "DOWN"
        if "DEGRADED" in status_levels:
            return "DEGRADED"
        return "OK"

    def get_engine_health(self) -> EngineHealthResult:
        """
        MODEL-004: Performs infrastructure health checks across 6 distinct checks:
        1. api: DB connectivity & API infrastructure responsiveness.
        2. data: data_fetcher reachability.
        3. events: corporate_events_fetcher, news_sentiment_fetcher, event_classifier.
        4. scheduler: scheduler execution status.
        5. storage: DB accessibility and writable data/ storage directories.
        6. model_file_availability: models/ directory accessibility and readability.
        """
        checks: dict[str, str] = {}
        reasons: list[str] = []

        statuses = self.get_status()
        comp_map = {s.component: s for s in statuses}

        # 1. API Check
        try:
            with SessionLocal() as db:
                db.execute(sa_text("SELECT 1"))
            checks["api"] = ENGINE_HEALTH_HEALTHY
        except Exception as e:
            checks["api"] = ENGINE_HEALTH_FAILED
            reasons.append(f"API infrastructure DB connection failed: {e}")

        # 2. Data Check (DataFetcher)
        df_status = comp_map.get("data_fetcher")
        if df_status is None or df_status.status == "OK":
            checks["data"] = ENGINE_HEALTH_HEALTHY
        elif df_status.status == "DEGRADED":
            checks["data"] = ENGINE_HEALTH_DEGRADED
            reasons.append(f"Market data provider degraded: {df_status.detail or df_status.last_error}")
        else:
            checks["data"] = ENGINE_HEALTH_FAILED
            reasons.append(f"Market data provider offline: {df_status.detail or df_status.last_error}")

        # 3. Events Check (Corporate events, news sentiment, event classifier)
        event_comps = ["corporate_events_fetcher", "news_sentiment_fetcher", "event_classifier"]
        event_statuses = [comp_map[c].status for c in event_comps if c in comp_map]
        if "DOWN" in event_statuses:
            checks["events"] = ENGINE_HEALTH_DEGRADED
            reasons.append("One or more event fetchers/classifiers offline.")
        elif "DEGRADED" in event_statuses:
            checks["events"] = ENGINE_HEALTH_DEGRADED
            reasons.append("Event subsystem operating in degraded mode.")
        else:
            checks["events"] = ENGINE_HEALTH_HEALTHY

        # 4. Scheduler Check
        sched_status = comp_map.get("scheduler")
        if sched_status is None or sched_status.status == "OK":
            checks["scheduler"] = ENGINE_HEALTH_HEALTHY
        elif sched_status.status == "DEGRADED":
            checks["scheduler"] = ENGINE_HEALTH_DEGRADED
            reasons.append("Scheduler cycle execution experiencing delays.")
        else:
            checks["scheduler"] = ENGINE_HEALTH_FAILED
            reasons.append("Scheduler execution offline.")

        # 5. Storage Check (DB & writable disk directories)
        try:
            import config

            db_ok = config.DB_PATH.parent.exists()
            data_ok = config.DATA_DIR.exists() and os.access(config.DATA_DIR, os.W_OK)
            if db_ok and data_ok:
                checks["storage"] = ENGINE_HEALTH_HEALTHY
            else:
                checks["storage"] = ENGINE_HEALTH_FAILED
                reasons.append("Storage directory write permission or existence check failed.")
        except Exception as e:
            checks["storage"] = ENGINE_HEALTH_FAILED
            reasons.append(f"Storage accessibility check failed: {e}")

        # 6. Model File Availability Check (models/ directory accessible and readable)
        try:
            import config

            models_dir = config.MODELS_DIR
            if models_dir.exists() and os.access(models_dir, os.R_OK):
                checks["model_file_availability"] = ENGINE_HEALTH_HEALTHY
            else:
                checks["model_file_availability"] = ENGINE_HEALTH_FAILED
                reasons.append("Model storage directory does not exist or is not readable.")
        except Exception as e:
            checks["model_file_availability"] = ENGINE_HEALTH_FAILED
            reasons.append(f"Model storage accessibility check failed: {e}")

        # Overall Status Resolution:
        # Critical checks: api, data, storage, model_file_availability
        critical_failed = any(
            checks.get(c) == ENGINE_HEALTH_FAILED
            for c in ["api", "data", "storage", "model_file_availability"]
        )

        check_vals = list(checks.values())
        if critical_failed:
            overall_status = ENGINE_HEALTH_FAILED
            summary = "Engine health is FAILED: critical infrastructure components are unavailable."
        elif ENGINE_HEALTH_DEGRADED in check_vals or ENGINE_HEALTH_FAILED in check_vals:
            overall_status = ENGINE_HEALTH_DEGRADED
            summary = "Engine health is DEGRADED: operating with partial subsystem availability."
        else:
            overall_status = ENGINE_HEALTH_HEALTHY
            summary = "Engine health is HEALTHY: all infrastructure checks operational."

        return EngineHealthResult(
            status=overall_status,
            checks=checks,
            components=statuses,
            summary=summary,
            reasons=reasons,
        )


# ---------------------------------------------------------------------------
# 7. Global Instance
# ---------------------------------------------------------------------------
registry = HealthRegistry()

# ---------------------------------------------------------------------------
# 8. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import tempfile

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # For tests, we use a separate test db, so we patch SessionLocal
    from database import Base

    configure_logging(log_filename="health_monitor_selftest.log")
    logger.info("Running health_monitor.py self-test...")

    fd, test_db_path = tempfile.mkstemp(suffix=".sqlite3")

    os.close(fd)
    engine = create_engine(f"sqlite:///{test_db_path}")

    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    import health_monitor

    health_monitor.SessionLocal = TestSessionLocal

    try:
        print("\n=== HEALTH MONITOR SELF-TEST RESULT ===")

        # Test: empty DB should return UNKNOWN (fixed fail-open)
        assert registry.get_overall_status() == "UNKNOWN"

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
            registry.report("comp3", ok=False)  # comp3 is now DOWN

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
        try:
            os.remove(test_db_path)
        except:
            pass
