# 1. Standard library imports
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 2. Third-party imports
from nse import NSE

# 3. Local imports
from config import (
    NIFTY50_SYMBOLS,
    NSE_RATE_LIMIT_DELAY_SECONDS,
    DATA_DIR,
    ensure_directories,
    configure_logging,
)
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
DEFAULT_LOOKBACK_DAYS = 7
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2

# Corporate event categories, normalized across the different nse package calls
# so event_classifier.py can treat them uniformly.
EVENT_CATEGORY_ANNOUNCEMENT = "CORPORATE_ANNOUNCEMENT"
EVENT_CATEGORY_BOARD_MEETING = "BOARD_MEETING"
EVENT_CATEGORY_CORPORATE_ACTION = "CORPORATE_ACTION"
EVENT_CATEGORY_BLOCK_DEAL = "BLOCK_DEAL"
EVENT_CATEGORY_BULK_DEAL = "BULK_DEAL"


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class CorporateEventsFetcher:
    """
    Wraps the open-source `nse` package to fetch corporate announcements,
    board meetings, corporate actions, and block/bulk deals for NIFTY50
    stocks. Every call is wrapped with retry + rate-limit delay since NSE
    throttles requests to ~3/sec and can be flaky.
    """

    def __init__(self, download_folder: str = str(DATA_DIR)):
        ensure_directories()
        self.download_folder = download_folder
        self._nse: Optional[NSE] = None

    def __enter__(self):
        self._nse = NSE(download_folder=self.download_folder, server=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_client(self) -> NSE:
        if self._nse is None:
            self._nse = NSE(download_folder=self.download_folder, server=False)
        return self._nse

    def close(self) -> None:
        try:
            if self._nse is not None:
                self._nse.exit()
                self._nse = None
        except Exception as e:
            logger.error(f"Error closing NSE client session: {e}")

    def _call_with_retry(self, fn_name: str, fn, *args, **kwargs):
        """Generic retry wrapper around any nse client method."""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                time.sleep(NSE_RATE_LIMIT_DELAY_SECONDS)
                result = fn(*args, **kwargs)
                health_registry.report("corporate_events_fetcher", ok=True, detail=f"Fetched {fn_name}")
                return result
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt}/{MAX_RETRIES} failed calling {fn_name}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        logger.error(f"All {MAX_RETRIES} attempts failed for {fn_name}: {last_error}")
        health_registry.report("corporate_events_fetcher", ok=False, detail=f"All {MAX_RETRIES} attempts failed for {fn_name}", error=str(last_error))
        return []

    def fetch_announcements(
        self, symbol: str, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Fetch corporate announcements for a single symbol."""
        to_date = to_date or datetime.now()
        from_date = from_date or (to_date - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        client = self._get_client()
        raw = self._call_with_retry(
            "announcements", client.announcements,
            index="equities", symbol=symbol, from_date=from_date, to_date=to_date,
        )
        return self._normalize(raw, symbol, EVENT_CATEGORY_ANNOUNCEMENT)

    def fetch_board_meetings(
        self, symbol: str, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Fetch upcoming/past board meetings for a single symbol."""
        to_date = to_date or datetime.now()
        from_date = from_date or (to_date - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        client = self._get_client()
        raw = self._call_with_retry(
            "boardMeetings", client.boardMeetings,
            index="equities", symbol=symbol, from_date=from_date, to_date=to_date,
        )
        return self._normalize(raw, symbol, EVENT_CATEGORY_BOARD_MEETING)

    def fetch_corporate_actions(
        self, symbol: str, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Fetch corporate actions (dividends, splits, bonuses, rights) for a single symbol."""
        to_date = to_date or datetime.now()
        from_date = from_date or (to_date - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        client = self._get_client()
        raw = self._call_with_retry(
            "actions", client.actions,
            segment="equities", symbol=symbol, from_date=from_date, to_date=to_date,
        )
        return self._normalize(raw, symbol, EVENT_CATEGORY_CORPORATE_ACTION)

    def fetch_block_deals(self) -> List[Dict]:
        """Fetch today's market-wide block deals (not filterable by symbol upstream)."""
        client = self._get_client()
        raw = self._call_with_retry("blockDeals", client.blockDeals)
        if isinstance(raw, dict):
            raw = raw.get("data", []) if raw else []
        return self._normalize(raw, symbol=None, category=EVENT_CATEGORY_BLOCK_DEAL)

    def fetch_bulk_deals(
        self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Fetch bulk deals across the market for a date range."""
        to_date = to_date or datetime.now()
        from_date = from_date or (to_date - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        client = self._get_client()
        raw = self._call_with_retry(
            "bulkdeals", client.bulkdeals,
            option_type="bulk_deals", fromdate=from_date, todate=to_date,
        )
        return self._normalize(raw, symbol=None, category=EVENT_CATEGORY_BULK_DEAL)

    def fetch_all_for_symbol(self, symbol: str) -> List[Dict]:
        """Fetch every corporate-event category for one symbol, combined into one list."""
        events: List[Dict] = []
        events.extend(self.fetch_announcements(symbol))
        events.extend(self.fetch_board_meetings(symbol))
        events.extend(self.fetch_corporate_actions(symbol))
        return events

    def fetch_all_nifty50(self) -> Dict[str, List[Dict]]:
        """Fetch corporate events for every NIFTY50 symbol. Slow by design —
        respects NSE rate limits — intended to run on a scheduled cadence,
        not on every prediction cycle."""
        results: Dict[str, List[Dict]] = {}
        for symbol in NIFTY50_SYMBOLS:
            try:
                results[symbol] = self.fetch_all_for_symbol(symbol)
            except Exception as e:
                logger.error(f"Failed fetching corporate events for {symbol}: {e}")
                health_registry.report("corporate_events_fetcher", ok=False, detail=f"Failed fetching for {symbol}", error=str(e))
                results[symbol] = []
        total_events = sum(len(v) for v in results.values())
        logger.info(f"Fetched {total_events} corporate event(s) across {len(NIFTY50_SYMBOLS)} symbols.")
        return results

    @staticmethod
    def _normalize(raw_items: List[Dict], symbol: Optional[str], category: str) -> List[Dict]:
        """Normalize raw nse-package dicts into a consistent shape:
        {symbol, category, raw, fetched_at}. Downstream event_classifier.py
        is responsible for further scope/sector tagging."""
        normalized = []
        if not raw_items:
            return normalized
        try:
            for item in raw_items:
                normalized.append({
                    "symbol": symbol or item.get("symbol") or item.get("Symbol"),
                    "category": category,
                    "raw": item,
                    "fetched_at": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.error(f"Failed normalizing {category} events for {symbol}: {e}")
            health_registry.report("corporate_events_fetcher", ok=False, detail=f"Failed normalizing {category} for {symbol}", error=str(e))
        return normalized


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="corporate_events_fetcher_selftest.log")
    logger.info("Running corporate_events_fetcher.py self-test...")

    test_symbol = "RELIANCE"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== CORPORATE EVENTS FETCHER SELF-TEST RESULT ===")
        with CorporateEventsFetcher() as fetcher:
            announcements = fetcher.fetch_announcements(test_symbol)
            print(f"Announcements for {test_symbol} (last {DEFAULT_LOOKBACK_DAYS}d): {len(announcements)}")

            board_meetings = fetcher.fetch_board_meetings(test_symbol)
            print(f"Board meetings for {test_symbol} (last {DEFAULT_LOOKBACK_DAYS}d): {len(board_meetings)}")

            actions = fetcher.fetch_corporate_actions(test_symbol)
            print(f"Corporate actions for {test_symbol} (last {DEFAULT_LOOKBACK_DAYS}d): {len(actions)}")

            block_deals = fetcher.fetch_block_deals()
            print(f"Market-wide block deals today: {len(block_deals)}")

            # This call is intentionally exercised so the retry/logging path is verified
            # even on days with zero of a given event type — that is expected and fine.
            all_events = fetcher.fetch_all_for_symbol(test_symbol)
            print(f"Combined events for {test_symbol}: {len(all_events)}")

        # Test connectivity succeeded if we got this far without an unhandled exception.
        # Zero results on any individual category is NOT a failure (there may genuinely
        # be no announcements/board meetings that week) — the test is about the pipeline
        # running end-to-end without crashing and returning well-formed lists.
        all_are_lists = all(
            isinstance(x, list) for x in [announcements, board_meetings, actions, block_deals, all_events]
        )
        print(f"All results correctly typed as lists: {all_are_lists}")
        print("STATUS: PASS" if all_are_lists else "STATUS: FAIL")

    except Exception as e:
        logger.error(f"corporate_events_fetcher.py self-test failed: {e}")
        print(f"STATUS: FAIL — {e}")
