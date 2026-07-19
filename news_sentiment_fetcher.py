# 1. Standard library imports
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import quote

# 2. Third-party imports
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 3. Local imports
from config import (
    MARKETAUX_API_KEY,
    MARKETAUX_BASE_URL,
    MARKETAUX_COUNTRY,
    NEWS_FETCH_LIMIT,
    NEWS_STALENESS_HOURS,
    GOOGLE_NEWS_RSS_BASE,
    NIFTY50_SYMBOLS,
    configure_logging,
)
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 10

SOURCE_MARKETAUX = "marketaux"
SOURCE_GOOGLE_RSS = "google_news_rss"

SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class NewsSentimentFetcher:
    """
    Fetches news headlines relevant to a stock, tries marketaux first
    (has a built-in sentiment score and India-market tagging), and falls
    back to free Google News RSS with our own VADER sentiment scoring if
    marketaux is unavailable, unconfigured, or returns nothing.

    Every article is normalized to:
    {symbol, title, url, published_at, source, sentiment_score, sentiment_label, is_stale}
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    @staticmethod
    def _sentiment_label(score: float) -> str:
        if score >= SENTIMENT_POSITIVE_THRESHOLD:
            return "POSITIVE"
        if score <= SENTIMENT_NEGATIVE_THRESHOLD:
            return "NEGATIVE"
        return "NEUTRAL"

    @staticmethod
    def _is_stale(published_at: Optional[datetime]) -> bool:
        if published_at is None:
            return True
        try:
            now = datetime.now(timezone.utc)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            age_hours = (now - published_at).total_seconds() / 3600.0
            return age_hours > NEWS_STALENESS_HOURS
        except Exception as e:
            logger.error(f"Failed staleness check on published_at={published_at}: {e}")
            return True

    def _fetch_marketaux(self, symbol: str, company_name: Optional[str] = None) -> List[Dict]:
        """Fetch news + sentiment from marketaux for a given symbol."""
        if not MARKETAUX_API_KEY:
            logger.warning("MARKETAUX_API_KEY not set — skipping marketaux, will use RSS fallback.")
            return []

        search_term = company_name or symbol
        params = {
            "api_token": MARKETAUX_API_KEY,
            "search": search_term,
            "countries": MARKETAUX_COUNTRY,
            "limit": NEWS_FETCH_LIMIT,
            "language": "en",
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(MARKETAUX_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                resp.raise_for_status()
                payload = resp.json()
                articles = payload.get("data", [])

                normalized = []
                for a in articles:
                    published_raw = a.get("published_at")
                    published_at = None
                    if published_raw:
                        try:
                            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                        except Exception:
                            published_at = None

                    # marketaux provides per-entity sentiment; fall back to article-level if present
                    sentiment_score = 0.0
                    entities = a.get("entities", [])
                    matched_entity = next(
                        (e for e in entities if search_term.lower() in str(e.get("name", "")).lower()
                         or search_term.lower() in str(e.get("symbol", "")).lower()),
                        None,
                    )
                    if matched_entity and matched_entity.get("sentiment_score") is not None:
                        sentiment_score = float(matched_entity["sentiment_score"])

                    normalized.append({
                        "symbol": symbol,
                        "title": a.get("title", ""),
                        "url": a.get("url", ""),
                        "published_at": published_at,
                        "source": SOURCE_MARKETAUX,
                        "sentiment_score": sentiment_score,
                        "sentiment_label": self._sentiment_label(sentiment_score),
                        "is_stale": self._is_stale(published_at),
                    })
                return normalized

            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt}/{MAX_RETRIES} failed fetching marketaux news for {symbol}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        logger.error(f"marketaux fetch failed for {symbol} after {MAX_RETRIES} attempts: {last_error}")
        return []

    def _fetch_google_rss(self, symbol: str, company_name: Optional[str] = None) -> List[Dict]:
        """Fallback: fetch headlines from free Google News RSS and score sentiment ourselves."""
        search_term = company_name or symbol
        query = quote(f"{search_term} NSE stock")
        url = f"{GOOGLE_NEWS_RSS_BASE}?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(url)
            if getattr(feed, "bozo", False) and feed.entries == []:
                logger.error(f"Google News RSS parse issue for {symbol}: {getattr(feed, 'bozo_exception', 'unknown')}")

            normalized = []
            for entry in feed.entries[:NEWS_FETCH_LIMIT]:
                title = entry.get("title", "")
                link = entry.get("link", "")
                published_at = None
                if entry.get("published_parsed"):
                    try:
                        published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        published_at = None

                vs = self._vader.polarity_scores(title)
                sentiment_score = vs.get("compound", 0.0)

                normalized.append({
                    "symbol": symbol,
                    "title": title,
                    "url": link,
                    "published_at": published_at,
                    "source": SOURCE_GOOGLE_RSS,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": self._sentiment_label(sentiment_score),
                    "is_stale": self._is_stale(published_at),
                })
            return normalized

        except Exception as e:
            logger.error(f"Failed fetching Google News RSS for {symbol}: {e}")
            return []

    def get_news_for_symbol(self, symbol: str, company_name: Optional[str] = None) -> List[Dict]:
        """
        Get news for one symbol: try marketaux first, fall back to RSS if
        marketaux is unconfigured, errors, or returns zero articles.
        """
        articles = self._fetch_marketaux(symbol, company_name)
        if articles:
            health_registry.report("news_sentiment_fetcher", ok=True, detail="served via marketaux")
            return articles

        logger.info(f"marketaux returned no results for {symbol}, falling back to Google News RSS.")
        rss_articles = self._fetch_google_rss(symbol, company_name)
        if rss_articles:
            health_registry.report("news_sentiment_fetcher", ok=True, detail="served via google_news_rss fallback")
        else:
            health_registry.report("news_sentiment_fetcher", ok=False, detail="both marketaux and google_news_rss failed")
        return rss_articles

    def get_news_for_all_nifty50(self) -> Dict[str, List[Dict]]:
        """Fetch news for every NIFTY50 symbol. Intended for a scheduled cadence,
        not every prediction cycle, to respect API rate limits."""
        results: Dict[str, List[Dict]] = {}
        for symbol in NIFTY50_SYMBOLS:
            try:
                results[symbol] = self.get_news_for_symbol(symbol)
            except Exception as e:
                logger.error(f"Failed fetching news for {symbol}: {e}")
                health_registry.report("news_sentiment_fetcher", ok=False, detail=f"Failed fetching news for {symbol}", error=str(e))
                results[symbol] = []
        total = sum(len(v) for v in results.values())
        logger.info(f"Fetched {total} news article(s) across {len(NIFTY50_SYMBOLS)} symbols.")
        return results


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="news_sentiment_fetcher_selftest.log")
    logger.info("Running news_sentiment_fetcher.py self-test...")

    test_symbol = "RELIANCE"  # single test symbol allowed in the __main__ block only
    test_company_name = "Reliance Industries"

    try:
        print("\n=== NEWS SENTIMENT FETCHER SELF-TEST RESULT ===")
        fetcher = NewsSentimentFetcher()

        if not MARKETAUX_API_KEY:
            print("NOTE: MARKETAUX_API_KEY is not set — this test will exercise the Google News RSS fallback path.")

        articles = fetcher.get_news_for_symbol(test_symbol, test_company_name)
        print(f"Articles found for {test_symbol}: {len(articles)}")

        schema_ok = True
        required_keys = {"symbol", "title", "url", "published_at", "source", "sentiment_score",
                          "sentiment_label", "is_stale"}
        for a in articles[:5]:
            missing = required_keys - set(a.keys())
            if missing:
                schema_ok = False
                print(f"  MISSING KEYS in article: {missing}")
            else:
                print(f"  [{a['source']}] {a['sentiment_label']} ({a['sentiment_score']:.2f}) — {a['title'][:70]}")

        print(f"All articles correctly schema'd: {schema_ok}")
        print("STATUS: PASS" if schema_ok else "STATUS: FAIL — schema mismatch")

    except Exception as e:
        logger.error(f"news_sentiment_fetcher.py self-test failed: {e}")
        print(f"STATUS: FAIL — {e}")
