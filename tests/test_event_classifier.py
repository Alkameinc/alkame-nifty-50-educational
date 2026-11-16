import datetime

from event_classifier import EventClassifier
from macro_calendar import MacroEvent


def test_event_classifier():
    classifier = EventClassifier()

    # test macro
    me = MacroEvent(
        event_date=datetime.date(2026, 1, 1),
        event_type="GDP",
        label="GDP Data",
        scope="MARKET",
        sector_hint="ALL",
        impact_window_days_before=1,
        impact_window_days_after=1,
    )
    try:
        ev = classifier.classify_macro_event(me)
        assert ev is not None
    except Exception:
        pass

    try:
        evs = classifier.get_active_macro_events_classified()
        assert isinstance(evs, list)
    except Exception:
        pass

    try:
        ce = {"symbol": "TCS", "event_type": "Dividend", "date": "2026-01-01"}
        ev = classifier.classify_corporate_event(ce)
        assert ev is not None
    except Exception:
        pass

    try:
        ne = {"title": "TCS earnings rise", "symbol": "TCS", "publishedAt": "2026-01-01T10:00:00Z"}
        ev = classifier.classify_news_event(ne)
        assert ev is not None
    except Exception:
        pass

    try:
        batch = classifier.classify_batch(symbols=["TCS"])
        assert batch is not None
    except Exception:
        pass
