from types import SimpleNamespace

from predictor import Predictor


def _event(sentiment_score):
    return SimpleNamespace(
        sentiment_score=sentiment_score,
        headline_or_label="Fixture event",
    )


def test_unknown_sentiment_is_not_treated_as_negative_or_positive():
    predictor = Predictor()

    event = _event(None)

    negative_events = [
        e for e in [event] if e.sentiment_score is not None and e.sentiment_score < 0
    ]
    positive_events = [
        e for e in [event] if e.sentiment_score is not None and e.sentiment_score > 0
    ]

    assert negative_events == []
    assert positive_events == []


def test_known_sentiment_is_classified_correctly():
    predictor = Predictor()

    negative = _event(-0.4)
    positive = _event(0.6)

    negative_events = [
        e for e in [negative, positive]
        if e.sentiment_score is not None and e.sentiment_score < 0
    ]
    positive_events = [
        e for e in [negative, positive]
        if e.sentiment_score is not None and e.sentiment_score > 0
    ]

    assert negative_events == [negative]
    assert positive_events == [positive]