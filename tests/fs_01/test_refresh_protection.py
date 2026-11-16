"""Live-worthiness refreshes must stay bounded even when dependencies fail."""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException, Response

from market_data_provider import DataStatus, MarketDataResult


@pytest.fixture
def context(monkeypatch):
    import api

    clock = SimpleNamespace(monotonic=1000.0, wall=1000.0)
    frame = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0], "Volume": [1000]})
    state = SimpleNamespace(stock=frame, index=frame, completed=[], fetched=[], failure=None, delay=0.0)

    def fetch_stock(ticker, return_metadata=False):
        state.fetched.append(ticker)
        clock.monotonic += state.delay
        clock.wall += state.delay
        if state.failure == "fetch":
            raise RuntimeError("Fixture fetch failure")
        return state.stock

    def refresh(symbol, stock, index):
        if state.failure == "refresh":
            raise RuntimeError("Fixture refresh failure")
        state.completed.append((symbol, stock, index))

    monkeypatch.setattr(api, "_refresh_locks", {})
    monkeypatch.setattr(api, "_refresh_last_time", {})
    monkeypatch.setattr(api, "time", SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.monotonic))
    monkeypatch.setattr(
        api,
        "scheduler",
        SimpleNamespace(
            data_fetcher=SimpleNamespace(fetch_ohlcv=fetch_stock, fetch_nifty_index=lambda **kwargs: state.index),
            refresh_live_worthiness=refresh,
        ),
    )
    return api, clock, state


def invoke(api, symbol="RELIANCE"):
    response = Response()
    result = api.refresh_backtest(symbol, response=response)
    return result, response.status_code


@pytest.mark.parametrize("missing", ["stock", "index"])
def test_failed_fetch_still_enforces_cooldown(context, missing):
    api, clock, state = context
    setattr(state, missing, None)
    first, _ = invoke(api)
    assert first == {"error": "Failed to fetch data"}
    second, code = invoke(api)
    assert code == 429
    assert second["status"] == "rejected"
    assert "cooldown active" in second["reason"].lower()
    assert state.completed == []
    assert len(state.fetched) == 1


@pytest.mark.parametrize("failure", ["fetch", "refresh"])
def test_exception_releases_lock_and_retains_cooldown(context, failure):
    api, clock, state = context
    state.failure = failure
    with pytest.raises(RuntimeError, match="Fixture"):
        invoke(api)
    assert not api._get_refresh_lock("RELIANCE").locked()
    result, code = invoke(api)
    assert code == 429
    assert "cooldown active" in result["reason"].lower()
    state.failure = None
    clock.monotonic += 60
    clock.wall += 60
    assert invoke(api) == ({"status": "success"}, 200)


def test_successful_refresh_and_exact_expiry_without_extending_rejections(context):
    api, clock, state = context
    assert invoke(api) == ({"status": "success"}, 200)
    recorded = api._refresh_last_time["RELIANCE"]
    for elapsed in [10.0, 30.0, 59.75]:
        clock.monotonic = clock.wall = 1000.0 + elapsed
        result, code = invoke(api)
        assert code == 429
        assert result["status"] == "rejected"
        assert api._refresh_last_time["RELIANCE"] == recorded
    assert "1s" in result["reason"]
    clock.monotonic = clock.wall = 1060.0
    assert invoke(api) == ({"status": "success"}, 200)
    assert len(state.completed) == 2


@pytest.mark.parametrize("unavailable", [False, True])
def test_cooldown_starts_after_admitted_attempt_finishes(context, unavailable):
    api, clock, state = context
    state.delay = 75.0
    if unavailable:
        state.stock = None
    first, _ = invoke(api)
    assert first == ({"error": "Failed to fetch data"} if unavailable else {"status": "success"})
    assert api._refresh_last_time["RELIANCE"] == 1075.0
    assert invoke(api)[1] == 429


def test_first_attempt_is_allowed_at_monotonic_zero(context):
    api, clock, state = context
    clock.monotonic = clock.wall = 0.0
    assert invoke(api) == ({"status": "success"}, 200)
    assert invoke(api)[1] == 429


@pytest.mark.parametrize("wall_jump", [-3600.0, 3600.0])
def test_wall_clock_changes_do_not_change_cooldown(context, wall_jump):
    api, clock, state = context
    assert invoke(api) == ({"status": "success"}, 200)
    clock.wall += wall_jump
    assert invoke(api)[1] == 429
    clock.monotonic += 60
    assert invoke(api) == ({"status": "success"}, 200)


def test_concurrent_same_symbol_rejected_and_other_symbol_can_progress(context, monkeypatch):
    api, clock, state = context
    entered, release = threading.Event(), threading.Event()
    normal_fetch = api.scheduler.data_fetcher.fetch_ohlcv

    def blocked_fetch(ticker, **kwargs):
        if ticker == "RELIANCE.NS":
            entered.set()
            assert release.wait(timeout=5), "Test failed to release fixture fetch"
        return normal_fetch(ticker, **kwargs)

    monkeypatch.setattr(api.scheduler.data_fetcher, "fetch_ohlcv", blocked_fetch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoke, api)
        try:
            assert entered.wait(timeout=5)
            rejected, code = invoke(api)
            assert code == 429
            assert "already running" in rejected["reason"]
            assert pool.submit(invoke, api, "TCS").result(timeout=5) == ({"status": "success"}, 200)
        finally:
            release.set()
        assert first.result(timeout=5) == ({"status": "success"}, 200)
    assert sorted(item[0] for item in state.completed) == ["RELIANCE", "TCS"]


def test_first_concurrent_requests_share_one_symbol_lock(context, monkeypatch):
    api, _, _ = context
    entered, release = threading.Event(), threading.Event()
    count = 0

    def slow_lock_factory():
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(timeout=5)
        return threading.Lock()

    # Widen the real lock-construction race without replacing the registry code.
    monkeypatch.setattr(api, "threading", SimpleNamespace(Lock=slow_lock_factory))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(api._get_refresh_lock, "RELIANCE")
        try:
            assert entered.wait(timeout=5)
            second = pool.submit(api._get_refresh_lock, "RELIANCE")
            try:
                second.result(timeout=0.2)
            except TimeoutError:
                pass  # Correct atomic initialization waits for the first constructor.
        finally:
            release.set()
        assert first.result(timeout=5) is second.result(timeout=5)
    assert count == 1


@pytest.mark.parametrize("data_status", list(DataStatus))
@pytest.mark.parametrize("side", ["stock", "index"])
def test_explicit_metadata_can_only_refresh_from_available_fresh_data(context, data_status, side):
    api, _, state = context
    stock, index = state.stock, state.index
    state.stock = MarketDataResult(stock, DataStatus.LIVE, "fixture")
    state.index = MarketDataResult(index, DataStatus.LIVE, "fixture")
    getattr(state, side).status = data_status
    result, _ = invoke(api)
    if data_status in (DataStatus.LIVE, DataStatus.CACHED_FRESH):
        assert result == {"status": "success"}
        assert state.completed[0][1] is stock
        assert state.completed[0][2] is index
    else:
        assert result == {"error": "Failed to fetch data"}
        assert state.completed == []


def test_legacy_frame_wrappers_remain_usable(context):
    api, _, state = context
    stock, index = state.stock, state.index
    state.stock = SimpleNamespace(df=stock)
    state.index = SimpleNamespace(df=index)
    assert invoke(api) == ({"status": "success"}, 200)
    assert state.completed[0][1] is stock
    assert state.completed[0][2] is index


def test_http_repeated_unavailable_refresh_returns_429(context, monkeypatch):
    from fastapi.testclient import TestClient

    api, _, state = context
    state.stock = None
    monkeypatch.setitem(
        api.app.dependency_overrides, api.get_current_client, lambda: api.ClientAuth("fixture", "ADMIN")
    )
    with TestClient(api.app) as client:
        first = client.post("/api/v1/signal/RELIANCE/refresh")
        assert first.json() == {"error": "Failed to fetch data"}
        second = client.post("/api/v1/signal/RELIANCE/refresh")
    assert second.status_code == 429
    assert second.json()["status"] == "rejected"
    assert state.completed == []


def test_invalid_symbol_cannot_start_or_reserve_refresh(context):
    api, _, state = context
    with pytest.raises(HTTPException) as caught:
        invoke(api, "NOT_A_SYMBOL")
    assert caught.value.status_code == 404
    assert state.fetched == []
    assert api._refresh_last_time == {}
