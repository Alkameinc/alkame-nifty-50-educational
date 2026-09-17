# 1. Standard library imports
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from config import MARKET_TIMEZONE
from data_fetcher import DataFetcher
from event_classifier import INVALID_EVENT_TIMESTAMP, Event, EventClassifier
from feature_engineer import AlignmentCoverageReport, FeatureEngineer
from market_calendar import MarketCalendar, previous_market_close
from market_data_provider import DataStatus, MarketDataResult, PriceAdjustmentMode
from predictor import PredictionContext, PredictionSignal, Predictor
from scheduler import Scheduler


IST = ZoneInfo(MARKET_TIMEZONE)
UTC = timezone.utc


def _make_intraday_bars(start_dt: datetime, num_bars: int = 10, freq_minutes: int = 5) -> pd.DataFrame:
    """Helper to generate a minimal valid OHLCV dataframe."""
    timestamps = [start_dt + timedelta(minutes=i * freq_minutes) for i in range(num_bars)]
    data = {
        "Open": [100.0 + i for i in range(num_bars)],
        "High": [101.0 + i for i in range(num_bars)],
        "Low": [99.0 + i for i in range(num_bars)],
        "Close": [100.5 + i for i in range(num_bars)],
        "Volume": [1000 + i * 10 for i in range(num_bars)],
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
    return df


# ===========================================================================
# DATA-001: No Silent Stale DataFrame
# ===========================================================================
def test_data_001_return_metadata_returns_market_data_result():
    fetcher = DataFetcher()
    now_utc = datetime.now(UTC)
    fresh_df = _make_intraday_bars(now_utc - timedelta(minutes=10), num_bars=3)

    with patch.object(fetcher, "_load_cache", return_value=fresh_df), \
         patch.object(fetcher, "check_staleness", return_value=False):
        res = fetcher.fetch_ohlcv("RELIANCE.NS", return_metadata=True)
        assert isinstance(res, MarketDataResult)
        assert res.status == DataStatus.CACHED_FRESH
        assert res.data is not None
        assert res.is_live is False
        assert res.is_stale is False
        assert res.is_available is True
        assert res.fetched_at.tzinfo is not None


def test_data_001_stale_cache_returns_none_in_plain_path_unless_allow_stale():
    fetcher = DataFetcher()
    stale_dt = datetime.now(UTC) - timedelta(days=5)
    stale_df = _make_intraday_bars(stale_dt, num_bars=5)

    with patch.object(fetcher, "_load_cache", return_value=stale_df), \
         patch.object(fetcher, "check_staleness", return_value=True), \
         patch("data_fetcher.yf.Ticker", side_effect=RuntimeError("Live fetch network timeout")):
        # Default allow_stale=False must return None
        raw_res = fetcher.fetch_ohlcv("RELIANCE.NS", return_metadata=False, allow_stale=False)
        assert raw_res is None

        # Explicit allow_stale=True returns stale dataframe for research/backtests
        stale_res = fetcher.fetch_ohlcv("RELIANCE.NS", return_metadata=False, allow_stale=True)
        assert stale_res is not None
        assert len(stale_res) == 5

        # Metadata path returns MarketDataResult with CACHED_STALE
        meta_res = fetcher.fetch_ohlcv("RELIANCE.NS", return_metadata=True)
        assert isinstance(meta_res, MarketDataResult)
        assert meta_res.status == DataStatus.CACHED_STALE
        assert meta_res.is_stale is True


def test_data_001_scheduler_does_not_promote_stale_data_to_live():
    fetcher = DataFetcher()
    stale_df = _make_intraday_bars(datetime.now(UTC) - timedelta(days=7), num_bars=5)
    stale_result = MarketDataResult(
        data=stale_df,
        status=DataStatus.CACHED_STALE,
        source="cache",
        fetched_at=datetime.now(UTC),
    )

    scheduler = Scheduler(
        data_fetcher=fetcher,
        predictor=MagicMock(),
        event_classifier=MagicMock(),
        history_manager=MagicMock(),
        backtester=MagicMock(),
        corporate_events_fetcher=MagicMock(),
        macro_calendar=MagicMock(),
        news_sentiment_fetcher=MagicMock(),
    )

    with patch.object(fetcher, "fetch_ohlcv", return_value=stale_result), \
         patch.object(fetcher, "fetch_nifty_index", return_value=stale_df), \
         patch.object(scheduler, "get_event_context", return_value=MagicMock(macro_events=[], corporate_events=[], news_articles=[], status="EVENTS_AVAILABLE")):
        ctx = scheduler.build_prediction_context("RELIANCE")
        assert ctx.data_status == "CACHED_STALE"
        assert ctx.data_status != "LIVE"


# ===========================================================================
# DATA-002: Session-Aware Freshness
# ===========================================================================
def test_data_002_friday_to_monday_session_transitions():
    fetcher = DataFetcher()

    # Friday 2026-07-17 session close at 15:30 IST
    friday_close_bar = datetime(2026, 7, 17, 15, 30, tzinfo=IST)
    friday_df = _make_intraday_bars(friday_close_bar - timedelta(minutes=25), num_bars=6)

    # 1. Saturday morning 10:00 IST -> Market closed, Friday close is valid/fresh
    saturday_now = datetime(2026, 7, 18, 10, 0, tzinfo=IST)
    assert fetcher.check_staleness(friday_df, "TEST", now=saturday_now) is False

    # 2. Sunday evening 20:00 IST -> Market closed, Friday close is valid/fresh
    sunday_now = datetime(2026, 7, 19, 20, 0, tzinfo=IST)
    assert fetcher.check_staleness(friday_df, "TEST", now=sunday_now) is False

    # 3. Monday pre-market 08:30 IST -> Market closed, Friday close is valid/fresh
    monday_pre_market = datetime(2026, 7, 20, 8, 30, tzinfo=IST)
    assert fetcher.check_staleness(friday_df, "TEST", now=monday_pre_market) is False

    # 4. Monday mid-session 10:30 IST -> Market is OPEN today! Friday bars are STALE!
    monday_mid_market = datetime(2026, 7, 20, 10, 30, tzinfo=IST)
    assert fetcher.check_staleness(friday_df, "TEST", now=monday_mid_market) is True


def test_data_002_trading_holiday_staleness():
    fetcher = DataFetcher()

    # Republic Day 2026-01-26 is Monday (official NSE holiday)
    # Previous trading session closed on Friday 2026-01-23 at 15:30 IST
    friday_close = datetime(2026, 1, 23, 15, 30, tzinfo=IST)
    fresh_df = _make_intraday_bars(friday_close - timedelta(minutes=25), num_bars=6)

    # Old data ending Thursday 2026-01-22
    thursday_close = datetime(2026, 1, 22, 15, 30, tzinfo=IST)
    stale_df = _make_intraday_bars(thursday_close - timedelta(minutes=25), num_bars=6)

    republic_day_noon = datetime(2026, 1, 26, 12, 0, tzinfo=IST)

    # Friday close data checked on holiday is fresh
    assert fetcher.check_staleness(fresh_df, "TEST", now=republic_day_noon) is False
    # Thursday close data checked on holiday is stale (missing Friday)
    assert fetcher.check_staleness(stale_df, "TEST", now=republic_day_noon) is True


def test_data_002_post_market_and_missing_closing_bar():
    fetcher = DataFetcher()

    # Tuesday 16:30 IST (post-market close at 15:30)
    tuesday_post_market = datetime(2026, 7, 14, 16, 30, tzinfo=IST)

    # Case A: Data has bars up to 15:30 -> FRESH
    full_day_df = _make_intraday_bars(datetime(2026, 7, 14, 15, 5, tzinfo=IST), num_bars=6)
    assert fetcher.check_staleness(full_day_df, "TEST", now=tuesday_post_market) is False

    # Case B: Data stopped at 11:00 AM -> STALE (missing closing bars)
    cut_off_df = _make_intraday_bars(datetime(2026, 7, 14, 10, 35, tzinfo=IST), num_bars=6)
    assert fetcher.check_staleness(cut_off_df, "TEST", now=tuesday_post_market) is True


def test_data_002_live_session_delayed_provider():
    fetcher = DataFetcher()

    # Tuesday 11:00 IST (market open)
    now_ist = datetime(2026, 7, 14, 11, 0, tzinfo=IST)

    # Bar from 10:55 IST (5 minutes old, threshold=15m) -> FRESH
    recent_df = _make_intraday_bars(datetime(2026, 7, 14, 10, 30, tzinfo=IST), num_bars=6)  # last bar 10:55
    assert fetcher.check_staleness(recent_df, "TEST", now=now_ist, max_age_minutes=15.0) is False

    # Bar from 10:20 IST (40 minutes old, threshold=15m) -> STALE
    delayed_df = _make_intraday_bars(datetime(2026, 7, 14, 9, 55, tzinfo=IST), num_bars=6)  # last bar 10:20
    assert fetcher.check_staleness(delayed_df, "TEST", now=now_ist, max_age_minutes=15.0) is True


# ===========================================================================
# DATA-003: Timestamp Alignment Coverage
# ===========================================================================
def test_data_003_alignment_coverage_calculation():
    fe = FeatureEngineer()
    base_ts = datetime(2026, 7, 14, 9, 15, tzinfo=UTC)

    stock_df = _make_intraday_bars(base_ts, num_bars=100, freq_minutes=5)

    # 1. 100% matched -> HEALTHY
    index_df_full = _make_intraday_bars(base_ts, num_bars=100, freq_minutes=5)
    report_full = fe.check_alignment(stock_df, index_df_full)
    assert report_full.status == "HEALTHY"
    assert report_full.coverage_ratio == 1.0
    assert report_full.matched_count == 100

    # 2. 70% matched (missing 30 bars) -> DEGRADED (< 95%)
    index_df_partial = stock_df.iloc[:70].copy()
    report_partial = fe.check_alignment(stock_df, index_df_partial)
    assert report_partial.status == "DEGRADED"
    assert report_partial.coverage_ratio == 0.70
    assert report_partial.matched_count == 70

    # 3. 20% matched -> UNAVAILABLE (< 50%)
    index_df_poor = stock_df.iloc[:20].copy()
    report_poor = fe.check_alignment(stock_df, index_df_poor)
    assert report_poor.status == "UNAVAILABLE"
    assert report_poor.coverage_ratio == 0.20


def test_data_003_engineer_features_attaches_alignment_metadata():
    fe = FeatureEngineer()
    base_ts = datetime(2026, 7, 14, 9, 15, tzinfo=UTC)
    stock_df = _make_intraday_bars(base_ts, num_bars=50, freq_minutes=5)
    index_df = stock_df.iloc[:30].copy()  # 60% coverage

    out = fe.engineer_features(stock_df, index_df)
    assert out is not None
    assert "alignment_status" in out.attrs
    assert out.attrs["alignment_status"] == "DEGRADED"
    assert out.attrs["alignment_ratio"] == 0.60
    assert "alignment_status" in out.columns
    assert (out["alignment_status"] == "DEGRADED").all()


# ===========================================================================
# DATA-004: Standardize Timestamps on Timezone-Aware UTC
# ===========================================================================
def test_data_004_reject_naive_datetime_in_core_objects():
    naive_dt = datetime(2026, 7, 14, 10, 0, 0)  # no tzinfo!
    utc_dt = datetime(2026, 7, 14, 10, 0, 0, tzinfo=UTC)
    df = _make_intraday_bars(utc_dt, num_bars=2)

    # 1. PredictionContext rejects naive timestamp
    with pytest.raises(ValueError, match="Naive datetime rejected in PredictionContext"):
        PredictionContext(symbol="TEST", timestamp=naive_dt, market_data=df)

    # 2. PredictionContext rejects naive as_of
    with pytest.raises(ValueError, match="Naive datetime rejected in PredictionContext.as_of"):
        PredictionContext(symbol="TEST", timestamp=utc_dt, as_of=naive_dt, market_data=df)

    # 3. PredictionSignal auto-converts naive timestamp to UTC internally
    sig = PredictionSignal(
        symbol="TEST",
        timestamp=naive_dt,
        horizon="INTRADAY",
        action="HOLD",
        model_predicted_class="FLAT",
        model_version="v1",
        feature_version="v1",
        raw_confidence=0.5,
        risk_adjusted_confidence=0.5,
        calibrated_confidence=None,
        agreement_fraction=1.0,
        downside_summary="None",
        upside_summary="None",
    )
    assert sig.timestamp.tzinfo is not None
    assert sig.timestamp == naive_dt.replace(tzinfo=UTC)

    # 4. MarketDataResult rejects naive fetched_at
    with pytest.raises(ValueError, match="Naive datetime rejected in MarketDataResult.fetched_at"):
        MarketDataResult(data=df, status=DataStatus.LIVE, source="test", fetched_at=naive_dt)

    # 5. Event rejects naive timestamp
    with pytest.raises(ValueError, match="Naive datetime rejected in Event.timestamp"):
        Event(
            event_id="EVT_1",
            source="MACRO",
            event_type="TEST",
            timestamp=naive_dt,
            scope="MARKET",
            affected_tickers=["ALL"],
        )


def test_data_004_load_cache_enforces_utc_index(tmp_path):
    fetcher = DataFetcher(cache_dir=tmp_path)
    csv_file = tmp_path / "RELIANCE_NS_5m.csv"
    # Write a CSV with dates
    content = "Datetime,Open,High,Low,Close,Volume\n2026-07-14 09:15:00,100,101,99,100.5,1000\n"
    csv_file.write_text(content)

    loaded = fetcher._load_cache("RELIANCE.NS", interval="5m")
    assert loaded is not None
    assert loaded.index.tz is not None
    assert str(loaded.index.tz) == "UTC"


# ===========================================================================
# DATA-005: Invalid Event Timestamps Must Not Fallback to now()
# ===========================================================================
def test_data_005_invalid_corporate_event_timestamp_rejected():
    classifier = EventClassifier()

    bad_corp_event = {
        "symbol": "RELIANCE",
        "category": "CORPORATE_ANNOUNCEMENT",
        "raw": {"published_at": "invalid_date_format_321"},
    }

    # Calling directly with allow_invalid=False raises ValueError (caught by classify_corporate_event and logged)
    # The returned fallback event must have INVALID_EVENT_TIMESTAMP, NEVER datetime.now()!
    fallback = classifier.classify_corporate_event(bad_corp_event)
    assert fallback.timestamp == INVALID_EVENT_TIMESTAMP
    assert fallback.timestamp != datetime.now(UTC)
    assert fallback.event_type == "CLASSIFICATION_ERROR"


def test_data_005_invalid_news_timestamp_rejected():
    classifier = EventClassifier()

    bad_news = {
        "symbol": "TCS",
        "title": "Unscheduled update without date",
        "published_at": None,
    }

    fallback = classifier.classify_news_event(bad_news)
    assert fallback.timestamp == INVALID_EVENT_TIMESTAMP
    assert fallback.timestamp != datetime.now(UTC)


def test_data_005_classify_batch_excludes_invalid_events():
    classifier = EventClassifier()

    bad_corp = {
        "symbol": "INFY",
        "category": "BOARD_MEETING",
        "raw": {"published_at": "MALFORMED_TIMESTAMP"},
    }
    good_corp = {
        "symbol": "INFY",
        "category": "BOARD_MEETING",
        "raw": {"published_at": "2026-09-10T04:25:00Z"},
    }

    batch_res = classifier.classify_batch(
        macro_events=[],
        corporate_events=[bad_corp, good_corp],
        news_articles=[],
    )

    # Only good event should be in results
    assert len(batch_res.events) == 1
    assert batch_res.events[0].affected_tickers == ["INFY"]
    assert batch_res.events[0].timestamp.isoformat() == "2026-09-10T04:25:00+00:00"
    assert len(batch_res.errors) >= 1
    assert "Rejected corporate event with invalid timestamp" in batch_res.errors[0]
