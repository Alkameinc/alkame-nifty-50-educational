from datetime import UTC, date, datetime, timedelta, timezone
from unittest import TestCase
from zoneinfo import ZoneInfo

from market_calendar import MarketCalendar


class TestMarketCalendarTimezone(TestCase):
    def setUp(self):
        self.calendar = MarketCalendar(holidays={"2026-07-14"})

    def test_utc_evening_is_a_holiday_after_midnight_in_ist(self):
        timestamp = datetime(2026, 7, 13, 20, tzinfo=UTC)
        self.assertTrue(self.calendar.is_holiday(timestamp))
        self.assertFalse(self.calendar.is_trading_day(timestamp))

    def test_utc_sunday_evening_is_a_trading_day_in_ist(self):
        timestamp = datetime(2026, 7, 12, 20, tzinfo=UTC)
        self.assertFalse(self.calendar.is_holiday(timestamp))
        self.assertTrue(self.calendar.is_trading_day(timestamp))

    def test_source_weekend_can_still_be_a_trading_day_in_ist(self):
        east_of_exchange = timezone(timedelta(hours=14))
        timestamp = datetime(2026, 7, 18, 1, tzinfo=east_of_exchange)
        self.assertTrue(self.calendar.is_trading_day(timestamp))

    def test_dates_strings_and_naive_datetimes_keep_their_calendar_date(self):
        for value in (date(2026, 7, 14), "2026-07-14", datetime(2026, 7, 14, 1)):
            with self.subTest(value=value):
                self.assertTrue(self.calendar.is_holiday(value))
                self.assertFalse(self.calendar.is_trading_day(value))

    def test_custom_exchange_timezone_is_used(self):
        calendar = MarketCalendar(holidays={"2026-07-14"}, timezone_str="America/New_York")
        timestamp = datetime(2026, 7, 15, 2, tzinfo=UTC)
        self.assertTrue(calendar.is_holiday(timestamp))
        self.assertFalse(calendar.is_trading_day(timestamp))

    def test_session_helpers_use_the_same_exchange_day(self):
        timestamp = datetime(2026, 7, 13, 20, tzinfo=UTC)
        ist = ZoneInfo("Asia/Kolkata")
        self.assertFalse(self.calendar.is_market_open(timestamp))
        self.assertEqual(self.calendar.next_market_open(timestamp), datetime(2026, 7, 15, 9, 15, tzinfo=ist))
        self.assertEqual(self.calendar.next_market_close(timestamp), datetime(2026, 7, 15, 15, 30, tzinfo=ist))
        self.assertEqual(self.calendar.previous_market_close(timestamp), datetime(2026, 7, 13, 15, 30, tzinfo=ist))
