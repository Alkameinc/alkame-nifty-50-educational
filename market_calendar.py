# 1. Standard library imports
import csv
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# 2. Local imports
from config import (
    MARKET_CLOSE_TIME,
    MARKET_OPEN_TIME,
    MARKET_TIMEZONE,
    configure_logging,
)

# 3. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 4. NSE Holidays List & Provider
# ---------------------------------------------------------------------------
# Official NSE Equity Holidays for 2026 (and common annual fixed holidays)
# Reference: https://www.nseindia.com/resources/exchange-communication-holidays
OFFICIAL_NSE_2026_HOLIDAYS: set[str] = {
    "2026-01-15",  # Municipal Corporation Election - Maharashtra
    "2026-01-26",  # Republic Day
    "2026-02-19",  # Chhatrapati Shivaji Maharaj Jayanti
    "2026-03-03",  # Holi
    "2026-03-19",  # Gudi Padwa
    "2026-03-26",  # Shri Ram Navami
    "2026-03-31",  # Shri Mahavir Jayanti
    "2026-04-01",  # Annual Bank Closing
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id
    "2026-06-26",  # Muharram
    "2026-08-15",  # Independence Day (Saturday)
    "2026-08-26",  # Id-e-Milad
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-08",  # Diwali Laxmi Pujan (Muhurat Trading / Non-regular)
    "2026-11-10",  # Diwali-Balipratipada
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}

NSE_HOLIDAYS_SET: set[str] = {
    # 2025 (Reference / fallback)
    "2025-01-26",
    "2025-08-15",
    "2025-10-02",
    "2025-12-25",
    # 2026 Official NSE Equity Holidays (Full official list)
    *OFFICIAL_NSE_2026_HOLIDAYS,
    # 2027 Key fixed holidays
    "2027-01-26",
    "2027-03-22",
    "2027-03-26",
    "2027-04-14",
    "2027-04-19",
    "2027-05-01",
    "2027-08-15",
    "2027-10-02",
    "2027-10-11",
    "2027-10-29",
    "2027-11-01",
    "2027-11-14",
    "2027-12-25",
}


class MarketCalendarProvider:
    """Abstract interface for market holiday and schedule providers."""

    def load_holidays(self) -> set[str]:
        raise NotImplementedError


class CSVMarketCalendarProvider(MarketCalendarProvider):
    """Loads versioned market holiday CSV files from data/market_calendar/ directory."""

    def __init__(self, calendar_dir: Path | None = None):
        self.calendar_dir = calendar_dir or (Path(__file__).parent / "data" / "market_calendar")

    def load_holidays(self) -> set[str]:
        holidays: set[str] = set()
        if not self.calendar_dir.exists():
            logger.warning(f"Calendar directory {self.calendar_dir} not found. Falling back to built-in holidays.")
            return set(NSE_HOLIDAYS_SET)

        csv_files = list(self.calendar_dir.glob("NSE_*.csv"))
        if not csv_files:
            logger.warning(f"No NSE_*.csv found in {self.calendar_dir}. Falling back to built-in holidays.")
            return set(NSE_HOLIDAYS_SET)

        for csv_path in sorted(csv_files):
            try:
                with open(csv_path, encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    file_holiday_count = 0
                    for row in reader:
                        date_str = (row.get("date") or row.get("\ufeffdate") or "").strip()
                        # Regular trading holidays are excluded from market trading
                        if date_str:
                            holidays.add(date_str)
                            file_holiday_count += 1
                logger.info(f"Loaded {file_holiday_count} holidays from {csv_path.name}")
            except Exception as e:
                logger.error(f"Failed loading market calendar CSV from {csv_path}: {e}")

        # Ensure baseline fallback holidays are also included
        holidays.update(NSE_HOLIDAYS_SET)
        return holidays


class MarketCalendar:
    """Authoritative exchange calendar for NSE Equity market operations.

    Provides unified market-open, trading-day, holiday, and session transition
    logic across data fetching, scheduling, and validation.
    """

    def __init__(
        self,
        holidays: set[str] | None = None,
        provider: MarketCalendarProvider | None = None,
        timezone_str: str = MARKET_TIMEZONE,
    ):
        if holidays is not None:
            self.holidays = holidays
        else:
            p = provider or CSVMarketCalendarProvider()
            try:
                self.holidays = p.load_holidays()
            except Exception as e:
                logger.warning(f"Error loading holidays from provider: {e}; falling back to static set.")
                self.holidays = set(NSE_HOLIDAYS_SET)

        self.tz = ZoneInfo(timezone_str)

    def is_holiday(self, d: date | datetime | str) -> bool:
        """Return True if the date is a recognized exchange holiday."""
        if isinstance(d, datetime):
            d_str = d.strftime("%Y-%m-%d")
        elif isinstance(d, date):
            d_str = d.strftime("%Y-%m-%d")
        else:
            d_str = str(d)
        return d_str in self.holidays

    def is_trading_day(self, d: date | datetime | str) -> bool:
        """Return True if `d` is a valid NSE trading day (weekday and non-holiday)."""
        if isinstance(d, datetime):
            check_date = d.date()
        elif isinstance(d, date):
            check_date = d
        else:
            check_date = datetime.strptime(str(d), "%Y-%m-%d").date()

        # Check weekend: Monday=0, Sunday=6
        if check_date.weekday() >= 5:
            return False

        return not self.is_holiday(check_date)

    def is_market_open(self, now: datetime | None = None) -> bool:
        """Check if NSE equity market is open right now (or at specified datetime).

        Rules:
        - Must be a trading day (weekday and non-holiday)
        - Must be between MARKET_OPEN_TIME and MARKET_CLOSE_TIME in IST
        """
        try:
            if now is None:
                current_dt = datetime.now(self.tz)
            else:
                current_dt = now.astimezone(self.tz) if now.tzinfo is not None else now.replace(tzinfo=self.tz)

            if not self.is_trading_day(current_dt.date()):
                return False

            current_time = current_dt.time()
            return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME
        except Exception as e:
            logger.error(f"Error checking is_market_open: {e}")
            return False

    def next_market_open(self, dt: datetime | None = None) -> datetime:
        """Return the next NSE market open timestamp (timezone-aware in IST)."""
        if dt is None:
            current_dt = datetime.now(self.tz)
        else:
            current_dt = dt.astimezone(self.tz) if dt.tzinfo is not None else dt.replace(tzinfo=self.tz)

        check_date = current_dt.date()

        # If today is a trading day and it's before open, market opens today
        if self.is_trading_day(check_date) and current_dt.time() < MARKET_OPEN_TIME:
            return datetime.combine(check_date, MARKET_OPEN_TIME, tzinfo=self.tz)

        # Otherwise, search forward day by day
        for offset in range(1, 30):
            next_date = check_date + timedelta(days=offset)
            if self.is_trading_day(next_date):
                return datetime.combine(next_date, MARKET_OPEN_TIME, tzinfo=self.tz)

        raise RuntimeError("No trading day found in next 30 days.")

    def next_market_close(self, dt: datetime | None = None) -> datetime:
        """Return the market close timestamp for the current or next open trading session."""
        if dt is None:
            current_dt = datetime.now(self.tz)
        else:
            current_dt = dt.astimezone(self.tz) if dt.tzinfo is not None else dt.replace(tzinfo=self.tz)

        check_date = current_dt.date()

        # If today is a trading day and market hasn't closed yet, close is today
        if self.is_trading_day(check_date) and current_dt.time() <= MARKET_CLOSE_TIME:
            return datetime.combine(check_date, MARKET_CLOSE_TIME, tzinfo=self.tz)

        # Otherwise, find next open and return its close
        next_open = self.next_market_open(current_dt)
        return datetime.combine(next_open.date(), MARKET_CLOSE_TIME, tzinfo=self.tz)


# Global default instance
market_calendar = MarketCalendar()


# Standalone helper functions
def is_holiday(d: date | datetime | str) -> bool:
    return market_calendar.is_holiday(d)


def is_trading_day(d: date | datetime | str) -> bool:
    return market_calendar.is_trading_day(d)


def is_market_open(now: datetime | None = None) -> bool:
    return market_calendar.is_market_open(now)


def next_market_open(dt: datetime | None = None) -> datetime:
    return market_calendar.next_market_open(dt)


def next_market_close(dt: datetime | None = None) -> datetime:
    return market_calendar.next_market_close(dt)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="market_calendar_selftest.log")
    logger.info("Running market_calendar.py self-test...")

    print("\n=== MARKET CALENDAR SELF-TEST ===")
    cal = MarketCalendar()
    tz = ZoneInfo(MARKET_TIMEZONE)

    # 1. Normal trading day
    tuesday_10am = datetime(2026, 7, 14, 10, 0, tzinfo=tz)
    print(f"Normal trading day (Tuesday 10:00): is_open={cal.is_market_open(tuesday_10am)} (expect True)")
    assert cal.is_market_open(tuesday_10am) is True
    assert cal.is_trading_day(tuesday_10am.date()) is True

    # 2. Weekend
    saturday_10am = datetime(2026, 7, 18, 10, 0, tzinfo=tz)
    print(f"Weekend (Saturday 10:00): is_open={cal.is_market_open(saturday_10am)} (expect False)")
    assert cal.is_market_open(saturday_10am) is False
    assert cal.is_trading_day(saturday_10am.date()) is False

    # 3. Known exchange holiday (Republic Day: 2026-01-26 is Monday)
    republic_day = datetime(2026, 1, 26, 10, 0, tzinfo=tz)
    print(f"Holiday (Republic Day 2026-01-26): is_open={cal.is_market_open(republic_day)} (expect False)")
    assert cal.is_market_open(republic_day) is False
    assert cal.is_trading_day(republic_day.date()) is False

    # 4. Pre-open (08:30 IST)
    tuesday_830am = datetime(2026, 7, 14, 8, 30, tzinfo=tz)
    print(f"Pre-open (Tuesday 08:30): is_open={cal.is_market_open(tuesday_830am)} (expect False)")
    assert cal.is_market_open(tuesday_830am) is False

    # 5. Post-close (16:00 IST)
    tuesday_4pm = datetime(2026, 7, 14, 16, 0, tzinfo=tz)
    print(f"Post-close (Tuesday 16:00): is_open={cal.is_market_open(tuesday_4pm)} (expect False)")
    assert cal.is_market_open(tuesday_4pm) is False

    # 6. next_market_open from Saturday
    nxt_open = cal.next_market_open(saturday_10am)
    print(f"Next open from Saturday 2026-07-18: {nxt_open} (expect Monday 2026-07-20 09:15)")
    assert nxt_open == datetime(2026, 7, 20, 9, 15, tzinfo=tz)

    # 7. next_market_close from Tuesday morning
    nxt_close = cal.next_market_close(tuesday_10am)
    print(f"Next close from Tuesday 10:00: {nxt_close} (expect Tuesday 2026-07-14 15:30)")
    assert nxt_close == datetime(2026, 7, 14, 15, 30, tzinfo=tz)

    # 8. next_market_close from Tuesday post-close
    nxt_close_post = cal.next_market_close(tuesday_4pm)
    print(f"Next close from Tuesday 16:00: {nxt_close_post} (expect Wednesday 2026-07-15 15:30)")
    assert nxt_close_post == datetime(2026, 7, 15, 15, 30, tzinfo=tz)

    # 9. Timezone boundary (UTC input converting to IST)
    # Tuesday 10:00 IST is Tuesday 04:30 UTC
    utc_dt = datetime(2026, 7, 14, 4, 30, tzinfo=ZoneInfo("UTC"))
    assert cal.is_market_open(utc_dt) is True

    # 10. Year transition (2026-12-31 to 2027-01-01)
    new_year_eve = datetime(2026, 12, 31, 16, 0, tzinfo=tz)
    nxt_open_ny = cal.next_market_open(new_year_eve)
    assert nxt_open_ny.date() == date(2027, 1, 1)

    print("STATUS: PASS")
