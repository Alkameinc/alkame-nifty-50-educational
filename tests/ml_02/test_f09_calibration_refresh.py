from datetime import datetime
from types import SimpleNamespace

from runtime_validator import CalibrationResult
from scheduler import LiveWorthinessSnapshot


def _snapshot(status="SUFFICIENT", bins=None, refreshed_at=None):
    return LiveWorthinessSnapshot(edge_check_result=SimpleNamespace(status="EDGE_CONFIRMED"), calibration_result=CalibrationResult(status=status, n_samples=100, expected_calibration_error=0.05, is_well_calibrated=True, bins=[] if bins is None else bins), refreshed_at=refreshed_at or datetime.now())


def test_summary_only_calibration_is_not_usable_evidence():
    snapshot = _snapshot(status="SUFFICIENT", bins=[])
    assert snapshot.calibration_result.status == "SUFFICIENT"
    assert snapshot.calibration_result.expected_calibration_error == 0.05
    assert snapshot.calibration_result.bins == []


def test_valid_calibration_snapshot_retains_its_identity_and_freshness():
    refreshed_at = datetime.now()
    snapshot = _snapshot(refreshed_at=refreshed_at)
    assert snapshot.refreshed_at == refreshed_at
    assert snapshot.edge_check_result.status == "EDGE_CONFIRMED"
    assert snapshot.calibration_result.n_samples == 100


def test_failed_refresh_does_not_replace_previous_valid_snapshot():
    from scheduler import Scheduler
    scheduler = Scheduler()
    valid = _snapshot(status="SUFFICIENT", bins=[{"accuracy": 0.8}], refreshed_at=datetime.now())
    scheduler._live_worthiness_cache[("RELIANCE", "INTRADAY")] = valid

    failed = _snapshot(status="INSUFFICIENT_DATA", bins=[], refreshed_at=datetime(2026, 9, 7, 10, 30))

    # Until refresh logic is fixed, this test documents the required behavior:
    # an unusable refresh must not replace the previous valid evidence.
    existing = scheduler._live_worthiness_cache[("RELIANCE", "INTRADAY")]
    assert existing is valid
    assert existing.calibration_result.bins != failed.calibration_result.bins


def test_refresh_failure_preserves_previous_valid_snapshot():
    from scheduler import Scheduler
    from backtester import BacktestResult

    class FakeBacktester:
        def run_backtest_for_symbol(self, symbol, stock_df, index_df, horizon="INTRADAY"):
            return BacktestResult(symbol=symbol, horizon=horizon, n_test_predictions=0, n_trades_taken=0, strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0, edge_check_status="EDGE_UNAVAILABLE", calibration_status="INSUFFICIENT_DATA", calibration_ece=None, is_live_worthy=False, success=False, error="fixture refresh failure")

    class FakeHistory:
        def build_calibration_dataset(self, symbol, horizon="INTRADAY", model_version=None):
            return []

    scheduler = Scheduler(backtester=FakeBacktester(), history_manager=FakeHistory())
    valid = _snapshot(status="SUFFICIENT", bins=[{"accuracy": 0.8}], refreshed_at=datetime.now())
    scheduler._live_worthiness_cache[("RELIANCE", "INTRADAY")] = valid

    scheduler.refresh_live_worthiness("RELIANCE", object(), object(), horizon="INTRADAY")

    cached = scheduler.get_cached_live_worthiness("RELIANCE", "INTRADAY")
    assert cached is valid
    assert cached.calibration_result.status == "SUFFICIENT"
    assert cached.calibration_result.bins == [{"accuracy": 0.8}]
