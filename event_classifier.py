# 1. Standard library imports
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

# 2. Third-party imports
# (none required)

# 3. Local imports
from config import (
    NIFTY50_SYMBOLS,
    SECTOR_MAP,
    CRUDE_SENSITIVE_SECTORS,
    INR_WEAKNESS_BENEFICIARY_SECTORS,
    INR_WEAKNESS_HURT_SECTORS,
    MONSOON_SENSITIVE_SECTORS,
    RATE_SENSITIVE_SECTORS,
    EVENT_IMPACT_HORIZON,
    HORIZON_INTRADAY,
    HORIZON_30D,
    configure_logging,
)
from macro_calendar import MacroCalendar, MacroEvent
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
SCOPE_MARKET = "MARKET"
SCOPE_SECTOR = "SECTOR"
SCOPE_STOCK = "STOCK"
VALID_SCOPES = {SCOPE_MARKET, SCOPE_SECTOR, SCOPE_STOCK}

# Reverse index: sector -> [symbols], built once from config.SECTOR_MAP
SECTOR_TO_SYMBOLS: Dict[str, List[str]] = {}
for _sym, _sector in SECTOR_MAP.items():
    SECTOR_TO_SYMBOLS.setdefault(_sector, []).append(_sym)

# Keyword hints used to map a macro event type or a news headline to sectors,
# when the source doesn't already give us an explicit sector hint.
MACRO_EVENT_TYPE_TO_SECTORS = {
    "RBI_POLICY": RATE_SENSITIVE_SECTORS,
    "GDP_RELEASE": ["ALL"],
    "UNION_BUDGET": ["ALL"],
    "ELECTION": ["ALL"],
    "FESTIVE_WINDOW": None,   # sector comes from the macro event's own sector_hint field
    "MONSOON_STATUS": MONSOON_SENSITIVE_SECTORS,
    "FDI_FLOW_RELEASE": ["ALL"],
    "GEOPOLITICAL": ["ALL"],
    "QUARTERLY_EARNINGS": None,
    "REGULATORY_CHANGE": None,
    "MACRO_OTHER": None,
}

# Simple keyword -> sector hints for news headline classification.
# Deliberately conservative: if nothing matches, the article stays STOCK-scoped
# to the symbol it was fetched for, rather than being guessed as MARKET-wide.
NEWS_KEYWORD_SECTOR_HINTS = {
    "crude": CRUDE_SENSITIVE_SECTORS,
    "oil price": CRUDE_SENSITIVE_SECTORS,
    "rupee": None,  # handled specially (direction-dependent), see _classify_currency_headline
    "monsoon": MONSOON_SENSITIVE_SECTORS,
    "rainfall": MONSOON_SENSITIVE_SECTORS,
    "interest rate": RATE_SENSITIVE_SECTORS,
    "repo rate": RATE_SENSITIVE_SECTORS,
    "war": ["ALL"],
    "sanction": ["ALL"],
    "blockade": ["ALL"],
    "tariff": ["ALL"],
}


@dataclass
class Event:
    """Unified event schema used across the whole system, regardless of
    which source (macro calendar, corporate announcement, or news) it came from."""
    event_id: str
    source: str                          # "MACRO", "CORPORATE", "NEWS"
    event_type: str                      # e.g. "RBI_POLICY", "CORPORATE_ANNOUNCEMENT", "NEWS_HEADLINE"
    timestamp: datetime
    scope: str                           # MARKET | SECTOR | STOCK
    affected_tickers: List[str]          # explicit list, never implied
    sector: Optional[str] = None
    confidence_in_scope: float = 1.0     # 1.0 = certain, lower = ambiguous, needs human review
    headline_or_label: str = ""
    sentiment_score: Optional[float] = None
    magnitude_estimate: str = "MEDIUM"   # "LOW" | "MEDIUM" | "HIGH"
    impact_horizon: str = HORIZON_INTRADAY
    raw: Optional[dict] = field(default=None, repr=False)

    def needs_human_review(self, low_confidence_threshold: float = 0.5) -> bool:
        return self.confidence_in_scope < low_confidence_threshold


class EventClassifier:
    """
    Converts raw items from macro_calendar.py, corporate_events_fetcher.py,
    and news_sentiment_fetcher.py into unified Event objects with a mandatory
    scope tag and explicit affected-ticker list.
    """

    def __init__(self, macro_calendar: Optional[MacroCalendar] = None):
        self.macro_calendar = macro_calendar or MacroCalendar()
        self._event_counter = 0

    def _next_event_id(self, prefix: str) -> str:
        self._event_counter += 1
        return f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._event_counter}"

    def _sectors_to_tickers(self, sectors: List[str]) -> List[str]:
        """Expand a list of sector names (or ['ALL']) into a flat ticker list."""
        if not sectors:
            return []
        if "ALL" in sectors:
            return list(NIFTY50_SYMBOLS)
        tickers: List[str] = []
        for sector in sectors:
            tickers.extend(SECTOR_TO_SYMBOLS.get(sector, []))
        return sorted(set(tickers))

    # -----------------------------------------------------------------
    # Macro calendar events -> Event objects
    # -----------------------------------------------------------------
    def classify_macro_event(self, macro_event: MacroEvent) -> Event:
        try:
            sectors = MACRO_EVENT_TYPE_TO_SECTORS.get(macro_event.event_type)
            if sectors is None:
                # Fall back to whatever sector_hint the calendar entry itself specifies
                sectors = macro_event.sector_list()

            scope = macro_event.scope if macro_event.scope in VALID_SCOPES else SCOPE_MARKET
            if scope == SCOPE_STOCK:
                tickers = sectors
            else:
                tickers = self._sectors_to_tickers(sectors)
                # If expansion only touches a subset (not literally ALL), this is really SECTOR scope
                if "ALL" not in sectors and scope == SCOPE_MARKET:
                    scope = SCOPE_SECTOR

            return Event(
                event_id=self._next_event_id("MACRO"),
                source="MACRO",
                event_type=macro_event.event_type,
                timestamp=datetime.combine(macro_event.event_date, datetime.min.time()),
                scope=scope,
                affected_tickers=tickers,
                sector=sectors[0] if scope == SCOPE_SECTOR and sectors and sectors[0] != "ALL" else None,
                confidence_in_scope=1.0,  # macro calendar entries are human-curated, high confidence
                headline_or_label=macro_event.label,
                sentiment_score=None,
                magnitude_estimate="HIGH" if macro_event.event_type in
                    {"RBI_POLICY", "UNION_BUDGET", "ELECTION", "GEOPOLITICAL"} else "MEDIUM",
                impact_horizon=EVENT_IMPACT_HORIZON.get(macro_event.event_type, HORIZON_30D),
                raw=None,
            )
            health_registry.report("event_classifier", ok=True)
            return evt
        except Exception as e:
            logger.error(f"Failed classifying macro event {macro_event}: {e}")
            health_registry.report("event_classifier", ok=False, detail="Failed classifying macro event", error=str(e))
            return self._fallback_event("MACRO", macro_event.label)

    def get_active_macro_events_classified(self, check_date: Optional[date] = None) -> List[Event]:
        try:
            raw_events = self.macro_calendar.get_active_macro_events(check_date)
            res = [self.classify_macro_event(e) for e in raw_events]
            health_registry.report("event_classifier", ok=True)
            return res
        except Exception as e:
            logger.error(f"Failed getting active classified macro events: {e}")
            health_registry.report("event_classifier", ok=False, detail="Failed getting active classified macro events", error=str(e))
            return []

    # -----------------------------------------------------------------
    # Corporate events -> Event objects (always STOCK scope, single symbol)
    # -----------------------------------------------------------------
    def classify_corporate_event(self, corporate_event: dict) -> Event:
        try:
            symbol = corporate_event.get("symbol")
            category = corporate_event.get("category", "CORPORATE_ANNOUNCEMENT")
            raw = corporate_event.get("raw", {})
            label = raw.get("subject") or raw.get("desc") or raw.get("purpose") or category

            affected = [symbol] if symbol else []
            scope = SCOPE_STOCK if symbol else SCOPE_MARKET  # market-wide block deals have no single symbol

            return Event(
                event_id=self._next_event_id("CORP"),
                source="CORPORATE",
                event_type=category,
                timestamp=datetime.now(),
                scope=scope,
                affected_tickers=affected,
                sector=SECTOR_MAP.get(symbol) if symbol else None,
                confidence_in_scope=1.0,  # sourced directly from NSE, high confidence
                headline_or_label=str(label),
                sentiment_score=None,
                magnitude_estimate="MEDIUM",
                impact_horizon=EVENT_IMPACT_HORIZON.get(category, EVENT_IMPACT_HORIZON.get("CORPORATE_ANNOUNCEMENT", HORIZON_INTRADAY)),
                raw=raw,
            )
            health_registry.report("event_classifier", ok=True)
            return evt
        except Exception as e:
            logger.error(f"Failed classifying corporate event {corporate_event}: {e}")
            health_registry.report("event_classifier", ok=False, detail="Failed classifying corporate event", error=str(e))
            return self._fallback_event("CORPORATE", str(corporate_event))

    # -----------------------------------------------------------------
    # News events -> Event objects (STOCK by default, escalated to SECTOR/MARKET
    # only when the headline text matches a known keyword hint)
    # -----------------------------------------------------------------
    def classify_news_event(self, news_article: dict) -> Event:
        try:
            symbol = news_article.get("symbol")
            title = news_article.get("title", "")
            title_lower = title.lower()

            matched_sectors: Optional[List[str]] = None
            for keyword, sectors in NEWS_KEYWORD_SECTOR_HINTS.items():
                if keyword in title_lower:
                    if keyword == "rupee":
                        matched_sectors = self._classify_currency_headline(title_lower)
                    else:
                        matched_sectors = sectors
                    break

            if matched_sectors:
                scope = SCOPE_MARKET if "ALL" in matched_sectors else SCOPE_SECTOR
                tickers = self._sectors_to_tickers(matched_sectors)
                confidence = 0.6  # keyword-matched macro-relevance is inherently less certain than a direct tag
                sector_val = None if scope == SCOPE_MARKET else matched_sectors[0]
            else:
                # No macro keyword matched -> stays scoped to the single stock it was fetched for
                scope = SCOPE_STOCK
                tickers = [symbol] if symbol else []
                confidence = 1.0 if symbol else 0.3
                sector_val = SECTOR_MAP.get(symbol) if symbol else None

            return Event(
                event_id=self._next_event_id("NEWS"),
                source="NEWS",
                event_type="NEWS_HEADLINE",
                timestamp=news_article.get("published_at") or datetime.now(),
                scope=scope,
                affected_tickers=tickers,
                sector=sector_val,
                confidence_in_scope=confidence,
                headline_or_label=title,
                sentiment_score=news_article.get("sentiment_score"),
                magnitude_estimate="HIGH" if scope == SCOPE_MARKET else "MEDIUM",
                impact_horizon=EVENT_IMPACT_HORIZON.get("NEWS_HEADLINE", HORIZON_INTRADAY),
                raw=news_article,
            )
            health_registry.report("event_classifier", ok=True)
            return evt
        except Exception as e:
            logger.error(f"Failed classifying news event {news_article}: {e}")
            health_registry.report("event_classifier", ok=False, detail="Failed classifying news event", error=str(e))
            return self._fallback_event("NEWS", news_article.get("title", ""))

    @staticmethod
    def _classify_currency_headline(title_lower: str) -> List[str]:
        """USD/INR moves affect exporters and importers in opposite directions —
        this needs its own logic rather than a flat sector list."""
        if "weak" in title_lower or "depreciat" in title_lower or "falls" in title_lower:
            # Weak rupee: exporters (IT, Pharma) benefit, importers (Energy, Auto) hurt.
            # We tag both sets; predictor.py is responsible for applying direction correctly per sector.
            return list(set(INR_WEAKNESS_BENEFICIARY_SECTORS + INR_WEAKNESS_HURT_SECTORS))
        if "strong" in title_lower or "appreciat" in title_lower or "gains" in title_lower:
            return list(set(INR_WEAKNESS_BENEFICIARY_SECTORS + INR_WEAKNESS_HURT_SECTORS))
        return ["ALL"]

    def _fallback_event(self, source: str, label: str) -> Event:
        """Used only when classification itself throws — produces a safe,
        low-confidence, STOCK-scoped-to-nothing event that human review will catch."""
        return Event(
            event_id=self._next_event_id(f"{source}_FALLBACK"),
            source=source,
            event_type="CLASSIFICATION_ERROR",
            timestamp=datetime.now(),
            scope=SCOPE_STOCK,
            affected_tickers=[],
            sector=None,
            confidence_in_scope=0.0,
            headline_or_label=label,
            sentiment_score=None,
            magnitude_estimate="LOW",
            impact_horizon=EVENT_IMPACT_HORIZON.get("CLASSIFICATION_ERROR", HORIZON_INTRADAY),
            raw=None,
        )

    def classify_batch(
        self,
        macro_events: Optional[List[MacroEvent]] = None,
        corporate_events: Optional[List[dict]] = None,
        news_articles: Optional[List[dict]] = None,
    ) -> List[Event]:
        """Classify a mixed batch from all three sources into one unified Event list."""
        results: List[Event] = []
        try:
            for m in (macro_events or []):
                results.append(self.classify_macro_event(m))
            for c in (corporate_events or []):
                results.append(self.classify_corporate_event(c))
            for n in (news_articles or []):
                results.append(self.classify_news_event(n))
            health_registry.report("event_classifier", ok=True)
        except Exception as e:
            logger.error(f"Failed classifying event batch: {e}")
            health_registry.report("event_classifier", ok=False, detail="Failed classifying event batch", error=str(e))
        return results


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="event_classifier_selftest.log")
    logger.info("Running event_classifier.py self-test...")

    try:
        classifier = EventClassifier()
        print("\n=== EVENT CLASSIFIER SELF-TEST RESULT ===")

        # Test 1: macro event (RBI policy) -> should be SECTOR scope (rate-sensitive sectors only)
        rbi_macro = MacroEvent(
            event_date=date(2026, 8, 5), event_type="RBI_POLICY", label="RBI MPC Policy Decision",
            scope="MARKET", sector_hint="ALL", impact_window_days_before=1, impact_window_days_after=1,
        )
        rbi_event = classifier.classify_macro_event(rbi_macro)
        print(f"RBI macro event -> scope={rbi_event.scope}, affected_tickers={len(rbi_event.affected_tickers)}")
        assert rbi_event.scope == SCOPE_SECTOR, "Expected RBI policy to be scoped to rate-sensitive sectors"
        assert "HDFCBANK" in rbi_event.affected_tickers, "Expected banking stocks tagged for RBI event"
        assert "INFY" not in rbi_event.affected_tickers, "IT stocks should NOT be tagged for a pure rate event"

        # Test 2: corporate event -> should be STOCK scope, single ticker
        corp_raw = {"symbol": "RELIANCE", "category": "CORPORATE_ANNOUNCEMENT",
                    "raw": {"subject": "Board Meeting Intimation"}, "fetched_at": datetime.now().isoformat()}
        corp_event = classifier.classify_corporate_event(corp_raw)
        print(f"Corporate event -> scope={corp_event.scope}, affected_tickers={corp_event.affected_tickers}")
        assert corp_event.scope == SCOPE_STOCK
        assert corp_event.affected_tickers == ["RELIANCE"]

        # Test 3: news headline with a sector keyword (crude) -> should escalate to SECTOR scope
        news_crude = {"symbol": "RELIANCE", "title": "Crude oil prices spike 6% overnight on supply fears",
                      "published_at": datetime.now(), "sentiment_score": -0.4}
        news_event = classifier.classify_news_event(news_crude)
        print(f"Crude news event -> scope={news_event.scope}, sectors touched via tickers={len(news_event.affected_tickers)}")
        assert news_event.scope == SCOPE_SECTOR
        assert "MARUTI" in news_event.affected_tickers  # Auto sector should be tagged for crude sensitivity

        # Test 4: plain single-stock news with no macro keyword -> should stay STOCK scope
        news_plain = {"symbol": "TCS", "title": "TCS wins new multi-year contract with European client",
                      "published_at": datetime.now(), "sentiment_score": 0.5}
        news_plain_event = classifier.classify_news_event(news_plain)
        print(f"Plain company news -> scope={news_plain_event.scope}, affected_tickers={news_plain_event.affected_tickers}")
        assert news_plain_event.scope == SCOPE_STOCK
        assert news_plain_event.affected_tickers == ["TCS"]

        # Test 5: batch classification combines all sources correctly
        batch = classifier.classify_batch(
            macro_events=[rbi_macro], corporate_events=[corp_raw], news_articles=[news_crude, news_plain],
        )
        print(f"Batch classification produced {len(batch)} unified events (expected 4)")
        assert len(batch) == 4

        print("STATUS: PASS")
        logger.info("event_classifier.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"event_classifier.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — assertion error: {ae}")
    except Exception as e:
        logger.error(f"event_classifier.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
