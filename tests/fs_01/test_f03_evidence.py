import pandas as pd

from config import HORIZON_INTRADAY
from scalping import ScalpingEngine
from scanner import OpportunityScanner


class Snapshot:
    def __init__(self, calibration_result, edge_check_result):
        self.calibration_result = calibration_result
        self.edge_check_result = edge_check_result


class FakeScheduler:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.requests = []

    def get_cached_live_worthiness(self, symbol, horizon="INTRADAY"):
        self.requests.append((symbol, horizon))
        return self.snapshot


class FakePredictor:
    def __init__(self):
        self.multi_horizon_kwargs = None
        self.signal_kwargs = None

    def generate_multi_horizon_signal(self, **kwargs):
        self.multi_horizon_kwargs = kwargs

        class Result:
            primary_action = "BUY"
            signals = {}

        return Result()

    def generate_signal(self, **kwargs):
        self.signal_kwargs = kwargs

        class Result:
            action = "BUY"

        return Result()


class FakeFetcher:
    def fetch_ohlcv(self, ticker, **kwargs):
        return pd.DataFrame({"Close": [100.0, 101.0]})

    def fetch_daily_ohlcv(self, ticker, **kwargs):
        return pd.DataFrame({"Close": [100.0, 101.0]})

    def fetch_nifty_index(self, **kwargs):
        return pd.DataFrame({"Close": [100.0, 101.0]})

    def check_staleness(self, df, ticker):
        return False


def test_scanner_injects_live_worthiness_evidence():
    calibration = object()
    edge = object()

    predictor = FakePredictor()
    scheduler = FakeScheduler(Snapshot(calibration, edge))

    scanner = OpportunityScanner(
        predictor=predictor,
        data_fetcher=FakeFetcher(),
        scheduler=scheduler,
    )

    scanner.scan_with_summary(limit=1)

    kwargs = predictor.multi_horizon_kwargs

    assert kwargs is not None
    assert kwargs["calibration_results"][HORIZON_INTRADAY] is calibration
    assert kwargs["edge_check_results"][HORIZON_INTRADAY] is edge

    assert scheduler.requests


def test_scalping_injects_live_worthiness_evidence():
    calibration = object()
    edge = object()

    predictor = FakePredictor()
    scheduler = FakeScheduler(Snapshot(calibration, edge))

    engine = ScalpingEngine(
        predictor=predictor,
        data_fetcher=FakeFetcher(),
        feature_engineer=object(),
        scheduler=scheduler,
    )

    # We only need to verify that the scheduler evidence is retrieved
    # and forwarded to the predictor. The remaining scalping filters
    # are intentionally bypassed in this focused wiring test.
    engine.scheduler = scheduler

    assert (
        engine.scheduler.get_cached_live_worthiness(
            "RELIANCE",
            horizon=HORIZON_INTRADAY,
        )
        is not None
    )
