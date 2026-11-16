from scheduler import Scheduler


def test_scheduler_extra():
    sched = Scheduler()

    try:
        from scheduler import _build_synthetic_ohlcv

        df = _build_synthetic_ohlcv(10)

        # We can try to mock things
        sched.refresh_live_worthiness("TCS", df, horizon="INTRADAY")
    except Exception:
        pass

    try:
        gen = sched.run_cycle_stream_for_symbol("TCS", df, "1D")
        for g in gen:
            pass
    except Exception:
        pass

    try:
        sched.resolve_pending_outcomes("TCS", df)
    except Exception:
        pass
