from datetime import datetime, timedelta
from types import SimpleNamespace

from runtime_validator import CalibrationResult
from scheduler import LiveWorthinessSnapshot


def _snapshot(refreshed_at):
    return LiveWorthinessSnapshot(
        edge_check_result=SimpleNamespace(status="EDGE_CONFIRMED"),
        calibration_result=CalibrationResult(
            status="SUFFICIENT",
            n_samples=100,
            expected_calibration_error=0.05,
            is_well_calibrated=True,
            bins=[{"accuracy": 0.8}],
        ),
        refreshed_at=refreshed_at,
    )


def test_cache_identity_is_isolated_by_symbol_and_horizon():
    from scheduler import Scheduler

    scheduler = Scheduler()
    now = datetime.now()
    intraday = _snapshot(now)
    daily = _snapshot(now)

    scheduler._live_worthiness_cache[("RELIANCE", "INTRADAY")] = intraday
    scheduler._live_worthiness_cache[("RELIANCE", "3D")] = daily

    assert scheduler.get_cached_live_worthiness("RELIANCE", "INTRADAY") is intraday
    assert scheduler.get_cached_live_worthiness("RELIANCE", "3D") is daily
    assert scheduler.get_cached_live_worthiness("TCS", "INTRADAY") is None


def test_expired_snapshot_is_not_usable():
    from scheduler import Scheduler

    scheduler = Scheduler()
    old = _snapshot(datetime(2026, 9, 6, 9, 0))
    scheduler._live_worthiness_cache[("RELIANCE", "INTRADAY")] = old

    cached = scheduler.get_cached_live_worthiness("RELIANCE", "INTRADAY")

    # Documents the F10 requirement: an expired snapshot must not be returned.
    assert cached is None
