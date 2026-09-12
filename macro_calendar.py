# 1. Standard library imports
import csv
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

# 2. Third-party imports
# (none required)
# 3. Local imports
from config import (
    MACRO_CALENDAR_PATH,
    MONSOON_SENSITIVE_SECTORS,
    configure_logging,
    ensure_directories,
)

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "event_date",
    "event_type",
    "label",
    "scope",
    "sector_hint",
    "impact_window_days_before",
    "impact_window_days_after",
    "notes",
]

# Event types this module understands. Anything else is treated as a generic
# "MACRO_OTHER" event so the system never silently drops an entry a human added.
KNOWN_EVENT_TYPES = {
    "RBI_POLICY",
    "GDP_RELEASE",
    "UNION_BUDGET",
    "ELECTION",
    "FESTIVE_WINDOW",
    "MONSOON_STATUS",
    "FDI_FLOW_RELEASE",
    "MACRO_OTHER",
}

# Seed data — known, publicly published dates. Traders should review/update this
# each time a new RBI calendar or election schedule is announced.
SEED_EVENTS = [
    # RBI MPC FY2026-27 calendar (published by RBI, confirmed April-Dec 2026 dates)
    {
        "event_date": "2026-04-08",
        "event_type": "RBI_POLICY",
        "label": "RBI MPC Policy Decision",
        "scope": "MARKET",
        "sector_hint": "Banking,NBFC,Insurance,ConsumerDurables,Auto",
        "impact_window_days_before": 1,
        "impact_window_days_after": 1,
        "notes": "Bi-monthly MPC decision",
    },
    {
        "event_date": "2026-06-05",
        "event_type": "RBI_POLICY",
        "label": "RBI MPC Policy Decision",
        "scope": "MARKET",
        "sector_hint": "Banking,NBFC,Insurance,ConsumerDurables,Auto",
        "impact_window_days_before": 1,
        "impact_window_days_after": 1,
        "notes": "Bi-monthly MPC decision",
    },
    {
        "event_date": "2026-08-05",
        "event_type": "RBI_POLICY",
        "label": "RBI MPC Policy Decision",
        "scope": "MARKET",
        "sector_hint": "Banking,NBFC,Insurance,ConsumerDurables,Auto",
        "impact_window_days_before": 1,
        "impact_window_days_after": 1,
        "notes": "Bi-monthly MPC decision",
    },
    {
        "event_date": "2026-10-07",
        "event_type": "RBI_POLICY",
        "label": "RBI MPC Policy Decision",
        "scope": "MARKET",
        "sector_hint": "Banking,NBFC,Insurance,ConsumerDurables,Auto",
        "impact_window_days_before": 1,
        "impact_window_days_after": 1,
        "notes": "Pre-festive policy review",
    },
    {
        "event_date": "2026-12-04",
        "event_type": "RBI_POLICY",
        "label": "RBI MPC Policy Decision",
        "scope": "MARKET",
        "sector_hint": "Banking,NBFC,Insurance,ConsumerDurables,Auto",
        "impact_window_days_before": 1,
        "impact_window_days_after": 1,
        "notes": "Final MPC meeting of calendar year",
    },
    # Union Budget — fixed annual date
    {
        "event_date": "2027-02-01",
        "event_type": "UNION_BUDGET",
        "label": "Union Budget Presentation",
        "scope": "MARKET",
        "sector_hint": "ALL",
        "impact_window_days_before": 2,
        "impact_window_days_after": 2,
        "notes": "Fixed annual date",
    },
    # Festive windows — approximate, verify exact dates yearly (lunar calendar shifts)
    {
        "event_date": "2026-10-20",
        "event_type": "FESTIVE_WINDOW",
        "label": "Diwali demand window",
        "scope": "SECTOR",
        "sector_hint": "Auto,FMCG,ConsumerDurables",
        "impact_window_days_before": 14,
        "impact_window_days_after": 3,
        "notes": "High retail/auto/consumer-durable demand period, verify exact date yearly",
    },
]


@dataclass
class MacroEvent:
    event_date: date
    event_type: str
    label: str
    scope: str  # "MARKET" | "SECTOR" | "STOCK"
    sector_hint: str  # comma-separated sector names, or "ALL"
    impact_window_days_before: int
    impact_window_days_after: int
    notes: str = ""

    def affects_date(self, check_date: date) -> bool:
        """Whether this event's impact window covers the given date."""
        window_start = self.event_date - timedelta(days=self.impact_window_days_before)
        window_end = self.event_date + timedelta(days=self.impact_window_days_after)
        return window_start <= check_date <= window_end

    def sector_list(self) -> list[str]:
        if self.sector_hint.strip().upper() == "ALL":
            return ["ALL"]
        return [s.strip() for s in self.sector_hint.split(",") if s.strip()]


class MacroCalendar:
    """
    Manages the macro/calendar event store backed by a CSV file so traders can
    edit it directly (in Excel or a text editor) without touching code.
    """

    def __init__(self, csv_path: Path = MACRO_CALENDAR_PATH):
        self.csv_path = csv_path
        self._events: list[MacroEvent] = []
        self._load_or_seed()

    def _load_or_seed(self) -> None:
        try:
            ensure_directories()
            if not self.csv_path.exists():
                logger.info(f"No macro calendar found at {self.csv_path}, seeding with defaults.")
                self._write_rows(SEED_EVENTS)
            self._events = self._read_rows()
            logger.info(f"Loaded {len(self._events)} macro calendar event(s) from {self.csv_path}")
        except Exception as e:
            logger.error(f"Failed to load or seed macro calendar: {e}")
            self._events = []

    def _write_rows(self, rows: list[dict]) -> None:
        f = None
        try:
            f = open(self.csv_path, mode="w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed writing macro calendar CSV at {self.csv_path}: {e}")
            raise
        finally:
            if f is not None:
                f.close()

    def _read_rows(self) -> list[MacroEvent]:
        events: list[MacroEvent] = []
        f = None
        try:
            f = open(self.csv_path, newline="", encoding="utf-8")
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    event_type = row["event_type"].strip() if row.get("event_type") else "MACRO_OTHER"
                    if event_type not in KNOWN_EVENT_TYPES:
                        logger.warning(
                            f"Unrecognized event_type '{event_type}' in macro calendar, " "treating as MACRO_OTHER."
                        )
                        event_type = "MACRO_OTHER"

                    events.append(
                        MacroEvent(
                            event_date=datetime.strptime(row["event_date"].strip(), "%Y-%m-%d").date(),
                            event_type=event_type,
                            label=row.get("label", "").strip(),
                            scope=row.get("scope", "MARKET").strip().upper() or "MARKET",
                            sector_hint=row.get("sector_hint", "ALL").strip() or "ALL",
                            impact_window_days_before=int(row.get("impact_window_days_before") or 0),
                            impact_window_days_after=int(row.get("impact_window_days_after") or 0),
                            notes=row.get("notes", "").strip(),
                        )
                    )
                except Exception as row_err:
                    logger.error(f"Skipping malformed macro calendar row {row}: {row_err}")
        except Exception as e:
            logger.error(f"Failed reading macro calendar CSV at {self.csv_path}: {e}")
        finally:
            if f is not None:
                f.close()
        return events

    def reload(self) -> None:
        """Re-read the CSV from disk — call this if a trader edited it mid-session."""
        try:
            self._events = self._read_rows()
            logger.info(f"Reloaded macro calendar: {len(self._events)} event(s).")
        except Exception as e:
            logger.error(f"Failed to reload macro calendar: {e}")

    def add_event(self, macro_event: MacroEvent) -> None:
        """Append a new event to memory and persist it to the CSV."""
        try:
            self._events.append(macro_event)
            self._write_rows([asdict_with_date_str(macro_event) for macro_event in self._events])
            logger.info(f"Added macro event: {macro_event.label} on {macro_event.event_date}")
        except Exception as e:
            logger.error(f"Failed to add macro event {macro_event}: {e}")
            raise

    def get_active_macro_events(self, check_date: date | None = None) -> list[MacroEvent]:
        """Return all macro events whose impact window covers check_date (default: today)."""
        check_date = check_date or date.today()
        try:
            return [e for e in self._events if e.affects_date(check_date)]
        except Exception as e:
            logger.error(f"Failed computing active macro events for {check_date}: {e}")
            return []

    def get_all_events(self) -> list[MacroEvent]:
        return list(self._events)


def asdict_with_date_str(macro_event: MacroEvent) -> dict:
    """Serialize a MacroEvent to a CSV-writable dict (date -> ISO string)."""
    d = asdict(macro_event)
    d["event_date"] = macro_event.event_date.strftime("%Y-%m-%d")
    return d


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="macro_calendar_selftest.log")
    logger.info("Running macro_calendar.py self-test...")

    try:
        calendar = MacroCalendar()
        all_events = calendar.get_all_events()
        print("\n=== MACRO CALENDAR SELF-TEST RESULT ===")
        print(f"Total events loaded: {len(all_events)}")

        # Test 1: active events on a known RBI policy date
        test_date = date(2026, 8, 5)
        active_on_rbi_day = calendar.get_active_macro_events(test_date)
        print(f"Active events on {test_date} (known RBI policy date): {len(active_on_rbi_day)}")
        for e in active_on_rbi_day:
            print(f"  - {e.label} ({e.event_type}, scope={e.scope}, sectors={e.sector_list()})")

        # Test 2: a quiet date with no events nearby
        quiet_date = date(2026, 1, 15)
        active_on_quiet_day = calendar.get_active_macro_events(quiet_date)
        print(f"Active events on {quiet_date} (expected quiet day): {len(active_on_quiet_day)}")

        # Test 3: add a new manual event (e.g. monsoon status update) and confirm it persists
        new_event = MacroEvent(
            event_date=date.today(),
            event_type="MONSOON_STATUS",
            label="Monsoon status: Normal (test entry)",
            scope="SECTOR",
            sector_hint=",".join(MONSOON_SENSITIVE_SECTORS),
            impact_window_days_before=0,
            impact_window_days_after=30,
            notes="Added by self-test",
        )
        calendar.add_event(new_event)
        calendar.reload()
        today_events = calendar.get_active_macro_events(date.today())
        found = any(e.label == new_event.label for e in today_events)
        print(f"Manual monsoon event added and reloaded successfully: {found}")

        assert len(all_events) >= len(SEED_EVENTS), "Seed events did not load correctly"
        assert found, "Newly added manual event was not found after reload"

        print("STATUS: PASS")
        logger.info("macro_calendar.py self-test passed.")

    except Exception as e:
        logger.error(f"macro_calendar.py self-test failed: {e}")
        print(f"STATUS: FAIL — {e}")
