"""Real unavailable-data HTTP refresh must preserve evidence and bound retries.

Run through the isolated runner under OS network denial: importing api creates
its default Scheduler. The request itself uses another real Scheduler with
temporary cache/history/health storage; only external history responses fail.
"""

import copy
import socket
from datetime import datetime
from types import SimpleNamespace

import curl_cffi
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
import data_fetcher
import health_monitor
import market_calendar
from backtester import Backtester
from data_fetcher import DataFetcher
from history_manager import HistoryManager
from models import BacktestMetric, HealthStatus, Prediction
from runtime_validator import EdgeCheckResult
from scheduler import LiveWorthinessSnapshot, Scheduler

NOW = pd.Timestamp("2026-09-10 10:00", tz="Asia/Kolkata")


@pytest.fixture
def real_refresh_context(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise RuntimeError("Network disabled for refresh integration verification")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(curl_cffi.Curl, "perform", no_network)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(config, "REQUIRED_DIRS", [cache_dir])
    history = HistoryManager(db_path=tmp_path / "history.sqlite3")
    monkeypatch.setattr(health_monitor, "SessionLocal", history.SessionLocal)

    import api

    fetcher = DataFetcher(cache_dir=cache_dir)
    scheduler = Scheduler(
        data_fetcher=fetcher,
        history_manager=history,
        backtester=Backtester(history_manager=history),
    )
    monkeypatch.setattr(api, "scheduler", scheduler)
    monkeypatch.setattr(api, "_refresh_locks", {})
    monkeypatch.setattr(api, "_refresh_last_time", {})
    monkeypatch.setattr(api, "API_AUTH_ENABLED", True)
    monkeypatch.setattr(api, "API_KEYS_ROLE_MAP", {"refresh-fixture-key": "ADMIN"})
    monkeypatch.setattr(data_fetcher, "RETRY_BACKOFF_SECONDS", 0)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = NOW.to_pydatetime()
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr(market_calendar, "datetime", FixedDateTime)
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda cls, tz=None: NOW.tz_convert(tz) if tz else NOW.tz_localize(None)),
    )

    attempted_tickers = []

    def failed_ticker(ticker):
        def history(*, period, interval, auto_adjust):
            attempted_tickers.append(ticker)
            raise ConnectionError("Fixture vendor unavailable")

        return SimpleNamespace(history=history)

    monkeypatch.setattr(data_fetcher.yf, "Ticker", failed_ticker)
    with TestClient(api.app) as client:
        yield api, scheduler, history, attempted_tickers, client
    history.engine.dispose()


@pytest.mark.parametrize("evidence", ["none", "insufficient", "sufficient"])
@pytest.mark.parametrize("availability", ["missing", "stale"])
def test_unavailable_refresh_preserves_real_evidence_and_limits_http_retry(
    real_refresh_context, evidence, availability
):
    api, scheduler, history, attempted_tickers, client = real_refresh_context
    validator = scheduler.predictor.runtime_validator
    prior_snapshot = None
    expected_confidence = None
    if availability == "stale":
        stale_bars = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=pd.date_range(end="2026-09-10 09:00", periods=2, freq="5min", tz="Asia/Kolkata"),
        )
        for filename in ["RELIANCE_NS_5m.csv", "IDX_NSEI_5m.csv"]:
            stale_bars.to_csv(scheduler.data_fetcher.cache_dir / filename)
    prior_files = {p.name: p.read_text() for p in scheduler.data_fetcher.cache_dir.iterdir()}
    if evidence != "none":
        # Known prior resolved outcomes, not a new backtest or invented score.
        pairs = pd.DataFrame({"confidence": [0.8] * 100, "correct": [1] * 80 + [0] * 20})
        if evidence == "insufficient":
            pairs = pairs.iloc[:1]
        else:
            expected_confidence = 0.8
        calibration = validator.compute_calibration(pairs)
        assert validator.get_calibrated_confidence(0.8, calibration) == expected_confidence
        prior_snapshot = LiveWorthinessSnapshot(
            edge_check_result=EdgeCheckResult("NO_EDGE", 0, 0.0, 0.0, 0.0),
            calibration_result=calibration,
            refreshed_at=datetime(2026, 9, 10, 9, 30),
        )
        scheduler._live_worthiness_cache[("RELIANCE", "INTRADAY")] = prior_snapshot
    previous_cache = copy.deepcopy(scheduler._live_worthiness_cache)
    headers = {"X-API-Key": "refresh-fixture-key"}

    first = client.post("/api/v1/signal/RELIANCE/refresh", headers=headers)

    assert first.status_code == 200
    assert first.json() == {"error": "Failed to fetch data"}
    assert set(attempted_tickers) == {"RELIANCE.NS", "^NSEI"}
    assert len(attempted_tickers) == 6  # Both actual fetches exhaust their three retries.
    assert {p.name: p.read_text() for p in scheduler.data_fetcher.cache_dir.iterdir()} == prior_files
    assert scheduler._live_worthiness_cache == previous_cache
    assert scheduler.get_cached_live_worthiness("RELIANCE") is prior_snapshot

    second = client.post("/api/v1/signal/RELIANCE/refresh", headers=headers)

    assert second.status_code == 429
    assert second.json()["status"] == "rejected"
    assert "cooldown active" in second.json()["reason"].lower()
    assert len(attempted_tickers) == 6
    assert scheduler._live_worthiness_cache == previous_cache
    assert scheduler.get_cached_live_worthiness("RELIANCE") is prior_snapshot
    if prior_snapshot is not None:
        assert validator.get_calibrated_confidence(0.8, prior_snapshot.calibration_result) == expected_confidence
    with history.SessionLocal() as db:
        health = db.query(HealthStatus).filter_by(component="data_fetcher").first()
        if availability == "missing":
            assert health is not None
            assert health.consecutive_failures == 2
            assert health.last_error == "Fixture vendor unavailable"
        else:
            assert health is None
        assert db.query(Prediction).count() == 0
        assert db.query(BacktestMetric).count() == 0
