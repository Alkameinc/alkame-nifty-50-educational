from unittest.mock import MagicMock, patch

import pytest

from news_sentiment_fetcher import SOURCE_GOOGLE_RSS, SOURCE_MARKETAUX, NewsSentimentFetcher


@pytest.fixture
def fetcher():
    return NewsSentimentFetcher()


def test_sentiment_label(fetcher):
    assert fetcher._sentiment_label(0.1) == "POSITIVE"
    assert fetcher._sentiment_label(-0.1) == "NEGATIVE"
    assert fetcher._sentiment_label(0.0) == "NEUTRAL"


def test_is_stale(fetcher):
    import datetime

    from config import NEWS_STALENESS_HOURS

    now = datetime.datetime.now(datetime.UTC)
    assert fetcher._is_stale(now) is False
    assert fetcher._is_stale(now - datetime.timedelta(hours=NEWS_STALENESS_HOURS + 1)) is True
    assert fetcher._is_stale(None) is True


@patch("news_sentiment_fetcher.requests.get")
@patch("news_sentiment_fetcher.MARKETAUX_API_KEY", "dummy_key")
def test_fetch_marketaux_success(mock_get, fetcher):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "title": "Good news",
                "url": "http://test",
                "published_at": "2026-01-01T00:00:00Z",
                "entities": [{"sentiment_score": 0.5}],
            }
        ]
    }
    mock_get.return_value = mock_response

    res = fetcher._fetch_marketaux("TCS")
    assert len(res) == 1
    assert res[0]["title"] == "Good news"
    assert res[0]["source"] == SOURCE_MARKETAUX


@patch("news_sentiment_fetcher.requests.get")
@patch("news_sentiment_fetcher.MARKETAUX_API_KEY", "dummy_key")
def test_fetch_marketaux_failure(mock_get, fetcher):
    mock_get.side_effect = Exception("API error")
    with patch("news_sentiment_fetcher.MAX_RETRIES", 1), patch("news_sentiment_fetcher.RETRY_BACKOFF_SECONDS", 0):
        res = fetcher._fetch_marketaux("TCS")
        assert res == []


@patch("news_sentiment_fetcher.feedparser.parse")
def test_fetch_google_rss_success(mock_parse, fetcher):
    mock_feed = MagicMock()
    mock_feed.entries = [
        {"title": "Good news for TCS", "link": "http://test", "published_parsed": (2026, 1, 1, 0, 0, 0, 0, 0, 0)}
    ]
    mock_parse.return_value = mock_feed

    res = fetcher._fetch_google_rss("TCS")
    assert len(res) == 1
    assert res[0]["title"] == "Good news for TCS"
    assert res[0]["source"] == SOURCE_GOOGLE_RSS


@patch("news_sentiment_fetcher.feedparser.parse")
def test_fetch_google_rss_failure(mock_parse, fetcher):
    mock_parse.side_effect = Exception("Parse error")
    with patch("news_sentiment_fetcher.MAX_RETRIES", 1), patch("news_sentiment_fetcher.RETRY_BACKOFF_SECONDS", 0):
        res = fetcher._fetch_google_rss("TCS")
        assert res == []


@patch("news_sentiment_fetcher.NewsSentimentFetcher._fetch_marketaux")
@patch("news_sentiment_fetcher.NewsSentimentFetcher._fetch_google_rss")
def test_get_news_for_symbol(mock_google, mock_marketaux, fetcher):
    # Try marketaux success
    mock_marketaux.return_value = [{"title": "Marketaux news"}]
    res = fetcher.get_news_for_symbol("TCS")
    assert len(res) == 1
    assert res[0]["title"] == "Marketaux news"

    # Try marketaux fail, fallback to google
    mock_marketaux.return_value = []
    mock_google.return_value = [{"title": "Google news"}]
    res = fetcher.get_news_for_symbol("TCS")
    assert len(res) == 1
    assert res[0]["title"] == "Google news"

    # Both fail
    mock_marketaux.return_value = []
    mock_google.return_value = []
    res = fetcher.get_news_for_symbol("TCS")
    assert res == []


@patch("news_sentiment_fetcher.NIFTY50_SYMBOLS", ["TCS", "INFY"])
@patch("news_sentiment_fetcher.NewsSentimentFetcher.get_news_for_symbol")
def test_get_news_for_all_nifty50(mock_get_news, fetcher):
    mock_get_news.return_value = [{"title": "News"}]
    res = fetcher.get_news_for_all_nifty50()
    assert "TCS" in res
    assert "INFY" in res
    assert len(res["TCS"]) == 1

    # Test error fallback
    mock_get_news.side_effect = Exception("Test error")
    res = fetcher.get_news_for_all_nifty50()
    assert res["TCS"] == []
