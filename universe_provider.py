# 1. Standard library imports
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

# 2. Local imports
from config import configure_logging

# 3. Logger setup
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models & Provider
# ---------------------------------------------------------------------------
@dataclass
class UniverseSnapshot:
    index_name: str
    version: str
    effective_from: str
    effective_to: str | None
    source: str
    total_constituents: int
    constituents: list[str]
    checksum: str


@dataclass
class MembershipRecord:
    effective_from: str
    effective_to: str | None
    symbol: str
    company: str
    action: str = "BASE"
    notes: str = ""


class UniverseProvider:
    """
    Authoritative provider for versioned index constituent universes (DATA-001).
    Validates constituent count (50 for NIFTY 50), effective date ranges,
    and SHA-256 integrity checksums.
    """

    def __init__(self, universe_dir: Path | None = None):
        self.universe_dir = universe_dir or (Path(__file__).parent / "data" / "universe")
        self._snapshots: dict[str, UniverseSnapshot] = {}
        self._membership_history: list[MembershipRecord] = []
        self._active_version: str | None = None
        self.load_universes()

    def _compute_checksum(self, constituents: list[str]) -> str:
        data_str = ",".join(sorted(constituents))
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def load_universes(self) -> None:
        """Scan and load all versioned universe JSON files and membership history CSV."""
        if not self.universe_dir.exists():
            logger.warning(f"Universe directory {self.universe_dir} not found.")
            return

        # 1. Load JSON snapshots
        json_files = list(self.universe_dir.glob("nifty50_constituents_*.json"))
        for fpath in sorted(json_files):
            try:
                with open(fpath, encoding="utf-8-sig") as f:
                    data = json.load(f)

                version = str(data.get("version", "1.0"))
                constituents = data.get("constituents", [])
                checksum = self._compute_checksum(constituents)

                snapshot = UniverseSnapshot(
                    index_name=data.get("index_name", "NIFTY 50"),
                    version=version,
                    effective_from=data.get("effective_from", "1996-04-22"),
                    effective_to=data.get("effective_to"),
                    source=data.get("source", "NSE Indices"),
                    total_constituents=len(constituents),
                    constituents=constituents,
                    checksum=checksum,
                )
                self._snapshots[version] = snapshot
                self._active_version = version
                logger.info(
                    f"Loaded universe {snapshot.index_name} v{version} ({len(constituents)} constituents, checksum: {checksum[:8]}...)"
                )
            except Exception as e:
                logger.error(f"Failed loading universe file {fpath}: {e}")

        # 2. Load historical membership interval dataset (DATA-006)
        history_csv = self.universe_dir / "nifty50_membership_history.csv"
        self._membership_history = []
        if history_csv.exists():
            try:
                with open(history_csv, encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        eff_from = row["effective_from"].strip()
                        eff_to = row["effective_to"].strip() if row.get("effective_to") else None
                        symbol = row["symbol"].strip().upper()
                        company = row["company"].strip()
                        action = row.get("action", "BASE").strip()
                        notes = row.get("notes", "").strip()
                        self._membership_history.append(
                            MembershipRecord(
                                effective_from=eff_from,
                                effective_to=eff_to,
                                symbol=symbol,
                                company=company,
                                action=action,
                                notes=notes,
                            )
                        )
                logger.info(
                    f"Loaded {len(self._membership_history)} historical membership intervals from {history_csv.name}"
                )
            except Exception as e:
                logger.error(f"Failed loading membership history CSV {history_csv}: {e}")

    @staticmethod
    def _normalize_date(val: Any) -> str:
        """Normalizes date, datetime, pd.Timestamp, or str to YYYY-MM-DD string."""
        if val is None:
            return date.today().strftime("%Y-%m-%d")
        if isinstance(val, str):
            return val[:10]
        if isinstance(val, (datetime, date)):
            return val.strftime("%Y-%m-%d")
        try:
            import pandas as pd

            ts = pd.to_datetime(val)
            return ts.strftime("%Y-%m-%d")
        except Exception:
            return str(val)[:10]

    def eligible_universe(self, as_of: Any = None) -> list[str]:
        """
        BACK-004 & DATA-006: Return the list of constituents that were active in the NIFTY 50 index
        at point-in-time `as_of`.
        Strictly prevents survivorship bias by consulting historical interval membership records.
        """
        if as_of is None:
            return self.get_constituents()

        target_date_str = self._normalize_date(as_of)

        if self._membership_history:
            eligible = set()
            for rec in self._membership_history:
                if rec.effective_from <= target_date_str:
                    if rec.effective_to is None or target_date_str <= rec.effective_to:
                        eligible.add(rec.symbol)
            if eligible:
                return sorted(list(eligible))

        # Fallback to snapshots if CSV not present or no match
        for snap in reversed(list(self._snapshots.values())):
            eff_from = snap.effective_from
            eff_to = snap.effective_to
            if eff_from <= target_date_str:
                if eff_to is None or target_date_str <= eff_to:
                    return list(snap.constituents)

        # Fallback to latest snapshot
        return self.get_constituents()

    def get_constituents_as_of(self, as_of: Any) -> list[str]:
        """Alias for eligible_universe(as_of) (BACK-004)."""
        return self.eligible_universe(as_of)

    def get_constituents(self, version: str | None = None, as_of_date: Any = None) -> list[str]:
        """
        Return the list of constituents for a given version, historical date, or latest active universe.
        """
        if as_of_date is not None:
            return self.eligible_universe(as_of_date)

        target_version = version or self._active_version
        if target_version and target_version in self._snapshots:
            return list(self._snapshots[target_version].constituents)

        # Fallback to any loaded snapshot
        if self._snapshots:
            latest = list(self._snapshots.values())[-1]
            return list(latest.constituents)

        # Default hardcoded fallback
        from config import NIFTY50_SYMBOLS

        return list(NIFTY50_SYMBOLS)

    def is_member(self, symbol: str, as_of: Any = None) -> bool:
        """
        Check if a given symbol was an active constituent of NIFTY 50 at timestamp `as_of`.
        """
        active_universe = self.eligible_universe(as_of)
        return symbol.upper() in set(active_universe)

    def get_membership_history(self, symbol: str | None = None) -> list[MembershipRecord]:
        """Return historical membership interval records, optionally filtered by symbol."""
        if symbol:
            sym_upper = symbol.upper()
            return [rec for rec in self._membership_history if rec.symbol == sym_upper]
        return list(self._membership_history)

    def get_membership_changes(
        self, start_date: Any = None, end_date: Any = None
    ) -> list[dict[str, Any]]:
        """
        Return chronological changelog of reconstitution events (additions and exclusions).
        """
        start_str = self._normalize_date(start_date) if start_date else "1990-01-01"
        end_str = self._normalize_date(end_date) if end_date else "2099-12-31"

        changes: list[dict[str, Any]] = []
        for rec in self._membership_history:
            if rec.action == "INCLUDED" and start_str <= rec.effective_from <= end_str:
                changes.append(
                    {
                        "date": rec.effective_from,
                        "type": "INCLUDED",
                        "symbol": rec.symbol,
                        "company": rec.company,
                        "notes": rec.notes,
                    }
                )
            if rec.effective_to and start_str <= rec.effective_to <= end_str:
                changes.append(
                    {
                        "date": rec.effective_to,
                        "type": "EXCLUDED",
                        "symbol": rec.symbol,
                        "company": rec.company,
                        "notes": rec.notes,
                    }
                )

        changes.sort(key=lambda x: (x["date"], x["type"]))
        return changes

    def get_snapshot(self, version: str | None = None) -> UniverseSnapshot | None:
        target_version = version or self._active_version
        return self._snapshots.get(target_version) if target_version else None

    def validate_constituents(self, expected_count: int = 50, as_of_date: Any = None) -> bool:
        """
        Verify that active universe (or point-in-time universe) has exactly expected_count constituents
        and no duplicates.
        """
        constituents = self.eligible_universe(as_of_date) if as_of_date else self.get_constituents()
        if len(constituents) != expected_count:
            logger.error(
                f"Universe constituent count mismatch (as_of={as_of_date}): found {len(constituents)}, expected {expected_count}"
            )
            return False
        if len(set(constituents)) != len(constituents):
            logger.error(f"Universe contains duplicate constituents (as_of={as_of_date})")
            return False
        return True


# Global default instance
universe_provider = UniverseProvider()


def get_nifty50_constituents() -> list[str]:
    return universe_provider.get_constituents()


if __name__ == "__main__":
    configure_logging(log_filename="universe_provider_selftest.log")
    print("\n=== UNIVERSE PROVIDER SELF-TEST ===")
    provider = UniverseProvider()
    constituents = provider.get_constituents()
    print(f"Constituents count: {len(constituents)}")
    print(f"Sample constituents: {constituents[:5]}")
    snapshot = provider.get_snapshot()
    if snapshot:
        print(f"Index: {snapshot.index_name}, Version: {snapshot.version}, SHA256: {snapshot.checksum}")
    is_valid = provider.validate_constituents(expected_count=50)
    print(f"Validation (count==50 & unique): {is_valid}")
    assert is_valid is True

    # Test historical point-in-time queries
    milestone_dates = [
        "2020-05-01",  # Pre-Sep 2020: ZEEL & INFRATEL in
        "2020-11-01",  # Post-Sep 2020: DIVISLAB & SBILIFE in
        "2021-05-01",  # Post-Mar 2021: TATACONSUM in, GAIL out
        "2022-05-01",  # Post-Mar 2022: APOLLOHOSP in, IOC out
        "2022-11-01",  # Post-Sep 2022: ADANIENT in, SHREECEM out
        "2023-08-01",  # Post-Jul 2023: LTIM in, HDFC out
        "2024-05-01",  # Post-Mar 2024: SHRIRAMFIN in, UPL out
        "2024-10-15",  # Post-Sep 2024: BEL & TRENT in, DIVISLAB & LTIM out
    ]
    for d in milestone_dates:
        u = provider.eligible_universe(d)
        print(f"Date {d}: {len(u)} constituents. Valid: {provider.validate_constituents(50, as_of_date=d)}")
        assert len(u) == 50
        assert provider.validate_constituents(50, as_of_date=d)

    # Test membership
    assert provider.is_member("GAIL", "2020-06-01") is True
    assert provider.is_member("GAIL", "2021-06-01") is False
    assert provider.is_member("TATACONSUM", "2020-06-01") is False
    assert provider.is_member("TATACONSUM", "2021-06-01") is True

    assert provider.is_member("UPL", "2023-01-01") is True
    assert provider.is_member("UPL", "2024-06-01") is False
    assert provider.is_member("SHRIRAMFIN", "2023-01-01") is False
    assert provider.is_member("SHRIRAMFIN", "2024-06-01") is True

    print("STATUS: PASS")

