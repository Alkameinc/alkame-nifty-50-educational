import ast
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from event_classifier import EventClassifier
from macro_calendar import MacroEvent
from predictor import ACTION_HOLD, MultiHorizonSignal, PredictionSignal
from scheduler import Scheduler

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "f23_event_context.json"


class FixtureMacroCalendar:
    def __init__(self, events):
        self.events = events
        self._last_query_ok = True

    def get_active_macro_events(self, check_date=None):
        return self.events


class FixtureCorporateFetcher:
    def __init__(self, events, status="EVENTS_AVAILABLE", errors=None):
        self.events = events
        self._last_fetch_status = status
        self._last_fetch_errors = errors or []

    def fetch_all_for_symbol(self, symbol):
        return list(self.events)


class FixtureNewsFetcher:
    def __init__(self, articles, status="EVENTS_AVAILABLE", errors=None):
        self.articles = articles
        self._last_fetch_status = status
        self._last_fetch_errors = errors or []

    def get_news_for_symbol(self, symbol):
        return list(self.articles)


class RecordingPredictor:
    def __init__(self):
        self.received = None

    def generate_multi_horizon_signal(
        self,
        symbol,
        horizons,
        stock_df,
        index_df,
        macro_events=None,
        corporate_events=None,
        news_articles=None,
        calibration_results=None,
        edge_check_results=None,
    ):
        self.received = {
            "macro_events": macro_events,
            "corporate_events": corporate_events,
            "news_articles": news_articles,
        }
        signal = PredictionSignal(
            symbol=symbol,
            timestamp=datetime.fromisoformat("2026-09-10T04:30:00+00:00"),
            horizon="INTRADAY",
            action=ACTION_HOLD,
            model_predicted_class="FLAT",
            model_version="fixture",
            feature_version="fixture",
            raw_confidence=0.5,
            risk_adjusted_confidence=0.5,
            calibrated_confidence=None,
            agreement_fraction=0.5,
            downside_summary="fixture",
            upside_summary="fixture",
        )
        return MultiHorizonSignal(
            symbol=symbol,
            timestamp=signal.timestamp,
            signals={"INTRADAY": signal},
            primary_action=ACTION_HOLD,
            primary_horizon="INTRADAY",
            reasoning=["fixture"],
        )


class FixtureHistory:
    def save_prediction(self, signal):
        return 1

    def save_event(self, event):
        return 1


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def make_scheduler(macro, corporate, news, predictor=None):
    return Scheduler(
        data_fetcher=object(),
        predictor=predictor or RecordingPredictor(),
        event_classifier=EventClassifier(macro_calendar=macro),
        history_manager=FixtureHistory(),
        backtester=object(),
        corporate_events_fetcher=corporate,
        macro_calendar=macro,
        news_sentiment_fetcher=news,
    )


def make_macro_events(items):
    return [MacroEvent(**{**item, "event_date": datetime.fromisoformat(item["event_date"]).date()}) for item in items]


def test_f23_boundary_and_source_context():
    fixture = load_fixture()
    normal = fixture["normal"]
    as_of = datetime.fromisoformat(fixture["signal_as_of"].replace("Z", "+00:00"))

    macro = FixtureMacroCalendar(make_macro_events(normal["macro_events"]))
    corporate = FixtureCorporateFetcher(normal["corporate_events"])
    news = FixtureNewsFetcher(normal["news_articles"])
    scheduler = make_scheduler(macro, corporate, news)

    context = scheduler.get_event_context("RELIANCE", as_of=as_of)

    assert context.status == "EVENTS_AVAILABLE"
    assert context.macro_status == "EVENTS_AVAILABLE"
    assert context.corporate_status == "EVENTS_AVAILABLE"
    assert context.news_status == "EVENTS_AVAILABLE"
    assert len(context.macro_events) == 1
    assert context.corporate_events[0]["symbol"] == "RELIANCE"
    assert [a["title"] for a in context.news_articles] == ["RELIANCE eligible news fixture"]

    classified = EventClassifier().classify_corporate_event(context.corporate_events[0])
    assert classified.source == "CORPORATE"
    assert classified.affected_tickers == ["RELIANCE"]
    assert classified.timestamp.isoformat() == "2026-09-10T04:25:00+00:00"

    news_event = EventClassifier().classify_news_event(context.news_articles[0])
    assert news_event.source == "NEWS"
    assert news_event.affected_tickers == ["RELIANCE"]


def test_f23_scheduler_passes_same_context_to_predictor():
    fixture = load_fixture()
    normal = fixture["normal"]
    as_of = datetime.fromisoformat(fixture["signal_as_of"].replace("Z", "+00:00"))
    macro = FixtureMacroCalendar(make_macro_events(normal["macro_events"]))
    corporate = FixtureCorporateFetcher(normal["corporate_events"])
    news = FixtureNewsFetcher(normal["news_articles"])
    predictor = RecordingPredictor()
    scheduler = make_scheduler(macro, corporate, news, predictor=predictor)

    stock_df = pd.DataFrame({"Close": [100.0]})
    index_df = pd.DataFrame({"Close": [100.0]})
    context = scheduler.get_event_context("RELIANCE", as_of=as_of)
    scheduler.run_one_cycle_for_symbol(
        "RELIANCE",
        stock_df,
        index_df,
        macro_events=context.macro_events,
        corporate_events=context.corporate_events,
        news_articles=context.news_articles,
    )

    assert predictor.received["macro_events"] == context.macro_events
    assert predictor.received["corporate_events"] == context.corporate_events
    assert predictor.received["news_articles"] == context.news_articles


def test_f23_scheduler_auto_collects_when_callers_omit_event_lists():
    fixture = load_fixture()
    normal = fixture["normal"]
    as_of = datetime.fromisoformat(fixture["signal_as_of"].replace("Z", "+00:00"))
    macro = FixtureMacroCalendar(make_macro_events(normal["macro_events"]))
    corporate = FixtureCorporateFetcher(normal["corporate_events"])
    news = FixtureNewsFetcher(normal["news_articles"])
    predictor = RecordingPredictor()
    scheduler = make_scheduler(macro, corporate, news, predictor=predictor)
    context = scheduler.get_event_context("RELIANCE", as_of=as_of)
    scheduler.get_event_context = lambda symbol: context

    stock_df = pd.DataFrame({"Close": [100.0]})
    index_df = pd.DataFrame({"Close": [100.0]})
    scheduler.run_one_cycle_for_symbol("RELIANCE", stock_df, index_df)

    assert predictor.received["corporate_events"] == context.corporate_events
    assert predictor.received["news_articles"] == context.news_articles


def test_f23_failure_distinguishes_unavailable_corporate_from_verified_empty_news():
    fixture = load_fixture()
    normal = fixture["normal"]
    failure = fixture["failure"]
    as_of = datetime.fromisoformat(fixture["signal_as_of"].replace("Z", "+00:00"))

    macro = FixtureMacroCalendar(make_macro_events(normal["macro_events"]))
    corporate = FixtureCorporateFetcher([], status=failure["corporate_status"], errors=failure["corporate_errors"])
    news = FixtureNewsFetcher([], status=failure["news_status"])
    scheduler = make_scheduler(macro, corporate, news)

    context = scheduler.get_event_context("RELIANCE", as_of=as_of)

    assert context.status == "EVENT_SOURCE_PARTIAL"
    assert context.macro_status == "EVENTS_AVAILABLE"
    assert context.macro_events
    assert context.corporate_status == "EVENT_SOURCE_UNAVAILABLE"
    assert context.corporate_events is None
    assert context.news_status == "NO_EVENTS"
    assert context.news_articles == []


def test_f23_real_callers_request_event_context():
    root = Path(__file__).parents[2]
    api_source = (root / "api.py").read_text(encoding="utf-8")
    app_source = (root / "app.py").read_text(encoding="utf-8")
    scheduler_source = (root / "scheduler.py").read_text(encoding="utf-8")

    for source in (api_source, app_source, scheduler_source):
        assert "get_event_context(" in source
        assert "macro_events=event_context.macro_events" in source
        assert "corporate_events=event_context.corporate_events" in source
        assert "news_articles=event_context.news_articles" in source

    ast.parse(api_source)
    ast.parse(app_source)
    ast.parse(scheduler_source)
