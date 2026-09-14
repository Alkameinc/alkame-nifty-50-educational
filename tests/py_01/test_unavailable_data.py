"""PY-01: truthful no-data metadata with real fetch/cache/health behavior.

Only vendor responses and time are controlled. CSV cache operations, retry and
validation branches, freshness checks and SQLite health writes remain real.
Run through the isolated verification runner beneath OS-level network denial.
"""

import socket
from datetime import datetime
from types import SimpleNamespace

import curl_cffi
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
import data_fetcher
import health_monitor
import market_calendar
from data_fetcher import DataFetcher
from market_data_provider import DataStatus, MarketDataResult
from models import Base, HealthStatus

SYMBOL = "RELIANCE.NS"
NOW = pd.Timestamp("2026-09-10 10:00", tz="Asia/Kolkata")


@pytest.fixture
def isolated_fetcher(monkeypatch, tmp_path):
    """Keep all files and health records temporary; deny network in Python too."""

    def no_network(*args, **kwargs):
        raise RuntimeError("Network disabled for unavailable-data verification")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(curl_cffi.Curl, "perform", no_network)

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(config, "REQUIRED_DIRS", [cache_dir])
    engine = create_engine(f"sqlite:///{tmp_path / 'health.sqlite3'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(health_monitor, "SessionLocal", sessions)
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
    fetcher = DataFetcher(cache_dir=cache_dir)
    yield fetcher, sessions
    engine.dispose()


def bars(last="2026-09-10 09:55"):
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.date_range(end=last, periods=2, freq="5min", tz="Asia/Kolkata"),
    )


def vendor_response(monkeypatch, response):
    """Control the external history call while retaining DataFetcher's logic."""

    def history(*, period, interval, auto_adjust):
        if isinstance(response, Exception):
            raise response
        return response.copy() if response is not None else None

    monkeypatch.setattr(data_fetcher.yf, "Ticker", lambda ticker: SimpleNamespace(history=history))


@pytest.fixture(params=["exception", "none", "empty", "missing_columns"])
def failed_response(request):
    cases = {
        "exception": (ConnectionError("vendor connection refused"), "vendor connection refused"),
        "none": (None, "Empty DataFrame returned"),
        "empty": (bars().iloc[:0], "Empty DataFrame returned"),
        "missing_columns": (bars().drop(columns=["Close"]), "Missing expected columns ['Close']"),
    }
    return cases[request.param]


@pytest.mark.parametrize("interval", ["5m", "1d"])
def test_failed_fetch_without_cache_has_explicit_unavailable_source(
    isolated_fetcher, monkeypatch, failed_response, interval
):
    fetcher, _ = isolated_fetcher
    response, _ = failed_response
    vendor_response(monkeypatch, response)

    result = fetcher.fetch_ohlcv(SYMBOL, interval=interval, period="1mo", return_metadata=True)

    assert isinstance(result, MarketDataResult)
    assert result.data is None
    assert result.status == DataStatus.UNAVAILABLE
    assert result.source.upper() == "UNAVAILABLE"
    assert result.is_stale is False
    assert not list(fetcher.cache_dir.iterdir())


def test_failed_fetch_without_cache_preserves_failure_reason(isolated_fetcher, monkeypatch, failed_response):
    fetcher, sessions = isolated_fetcher
    response, reason = failed_response
    vendor_response(monkeypatch, response)

    result = fetcher.fetch_ohlcv(SYMBOL, return_metadata=True)

    assert result.error is not None
    assert reason in result.error
    with sessions() as db:
        health = db.query(HealthStatus).filter_by(component="data_fetcher").one()
        assert health.last_error == result.error
        assert health.consecutive_failures == 1


@pytest.mark.parametrize("interval", ["5m", "1d"])
def test_live_fetch_keeps_data_and_writes_interval_cache(isolated_fetcher, monkeypatch, interval):
    fetcher, _ = isolated_fetcher
    expected = bars()
    vendor_response(monkeypatch, expected)

    result = fetcher.fetch_ohlcv(SYMBOL, interval=interval, period="1mo", return_metadata=True)

    assert result.status == DataStatus.LIVE
    assert result.source == "yahoo"
    assert result.error is None
    assert result.is_stale is False
    pd.testing.assert_frame_equal(result.data, expected)
    cached = pd.read_csv(fetcher.cache_dir / f"RELIANCE_NS_{interval}.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(cached.reset_index(drop=True), expected.reset_index(drop=True))


def test_fresh_cache_remains_usable_when_vendor_is_unavailable(isolated_fetcher, monkeypatch):
    fetcher, _ = isolated_fetcher
    expected = bars()
    expected.to_csv(fetcher.cache_dir / "RELIANCE_NS_5m.csv")
    vendor_response(monkeypatch, ConnectionError("vendor unavailable"))

    result = fetcher.fetch_ohlcv(SYMBOL, return_metadata=True)

    assert result.status == DataStatus.CACHED_FRESH
    assert result.source == "cache"
    assert result.is_stale is False
    pd.testing.assert_frame_equal(result.data, expected, check_freq=False)


def test_stale_cache_fallback_is_preserved_when_live_fetch_fails(isolated_fetcher, monkeypatch):
    fetcher, _ = isolated_fetcher
    expected = bars(last="2026-09-10 09:00")
    expected.to_csv(fetcher.cache_dir / "RELIANCE_NS_5m.csv")
    vendor_response(monkeypatch, ConnectionError("vendor unavailable"))

    result = fetcher.fetch_ohlcv(SYMBOL, return_metadata=True)

    assert result.status == DataStatus.CACHED_STALE
    assert result.source == "cache"
    assert result.is_stale is True
    pd.testing.assert_frame_equal(result.data, expected, check_freq=False)


def test_original_provenance_regression_accepts_stale_cache(isolated_fetcher, monkeypatch):
    import test_p0_regressions

    fetcher, _ = isolated_fetcher
    bars(last="2026-09-10 09:00").to_csv(fetcher.cache_dir / "RELIANCE_NS_5m.csv")
    vendor_response(monkeypatch, ConnectionError("vendor unavailable"))
    monkeypatch.setattr(test_p0_regressions, "DataFetcher", lambda: fetcher)

    test_p0_regressions.test_data_freshness_explicit_state()


@pytest.mark.parametrize("interval", ["5m", "1d"])
def test_failed_legacy_fetch_still_returns_none(isolated_fetcher, monkeypatch, interval):
    fetcher, _ = isolated_fetcher
    vendor_response(monkeypatch, ConnectionError("vendor unavailable"))

    assert fetcher.fetch_ohlcv(SYMBOL, interval=interval, period="1mo") is None


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_index_wrapper_preserves_legacy_data_and_can_return_metadata(
    isolated_fetcher, monkeypatch, metadata, available
):
    fetcher, _ = isolated_fetcher
    expected = bars()

    def index_ticker(ticker):
        assert ticker == "^NSEI"

        def history(*, interval, period, auto_adjust):
            assert (interval, period, auto_adjust) == ("1d", "1mo", False)
            if not available:
                raise ConnectionError("index vendor unavailable")
            return expected.copy()

        return SimpleNamespace(history=history)

    monkeypatch.setattr(data_fetcher.yf, "Ticker", index_ticker)
    if metadata:
        result = fetcher.fetch_nifty_index(interval="1d", period="1mo", return_metadata=True)
        assert isinstance(result, MarketDataResult)
        assert result.status == (DataStatus.LIVE if available else DataStatus.UNAVAILABLE)
        if not available:
            assert result.error == "index vendor unavailable"
        frame = result.data
    else:
        frame = fetcher.fetch_nifty_index(interval="1d", period="1mo")
    if available:
        pd.testing.assert_frame_equal(frame, expected)
    else:
        assert frame is None
