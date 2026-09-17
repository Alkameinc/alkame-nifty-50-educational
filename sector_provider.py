# 1. Standard library imports
import csv
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
class SectorMapSnapshot:
    version: str
    effective_date: str
    source: str
    total_symbols: int
    mappings: dict[str, dict[str, str]]


@dataclass
class HistoricalSectorRecord:
    symbol: str
    sector: str
    industry: str
    effective_from: str
    effective_to: str | None
    source: str
    notes: str = ""


class SectorMapProvider:
    """
    Authoritative provider for versioned sector classifications (DATA-002 & DATA-007).
    Maps symbols to primary sector and granular industry with schema validation,
    supporting point-in-time historical sector resolution for event studies and replay backtests.
    """

    def __init__(self, sector_dir: Path | None = None):
        self.sector_dir = sector_dir or (Path(__file__).parent / "data" / "sector_map")
        self._snapshots: dict[str, SectorMapSnapshot] = {}
        self._history: list[HistoricalSectorRecord] = []
        self._active_version: str | None = None
        self.load_sector_maps()

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

    def load_sector_maps(self) -> None:
        """Scan and load all versioned sector map JSON files and historical CSV."""
        if not self.sector_dir.exists():
            logger.warning(f"Sector map directory {self.sector_dir} not found.")
            return

        # 1. Load JSON snapshots
        json_files = list(self.sector_dir.glob("nifty50_sectors_*.json"))
        for fpath in sorted(json_files):
            try:
                with open(fpath, encoding="utf-8-sig") as f:
                    data = json.load(f)

                version = str(data.get("version", "1.0"))
                mappings = data.get("mappings", {})

                snapshot = SectorMapSnapshot(
                    version=version,
                    effective_date=data.get("effective_date", "2024-03-28"),
                    source=data.get("source", "NSE Sector Classification"),
                    total_symbols=len(mappings),
                    mappings=mappings,
                )
                self._snapshots[version] = snapshot
                self._active_version = version
                logger.info(f"Loaded sector map v{version} ({len(mappings)} mappings)")
            except Exception as e:
                logger.error(f"Failed loading sector map file {fpath}: {e}")

        # 2. Load historical sector intervals (DATA-007)
        history_csv = self.sector_dir / "nifty50_sector_history.csv"
        self._history = []
        if history_csv.exists():
            try:
                with open(history_csv, encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        eff_from = row["effective_from"].strip()
                        eff_to = row["effective_to"].strip() if row.get("effective_to") else None
                        sym = row["symbol"].strip().upper()
                        sec = row["sector"].strip()
                        ind = row.get("industry", "").strip()
                        src = row.get("source", "NSE Sector Classification").strip()
                        nts = row.get("notes", "").strip()
                        self._history.append(
                            HistoricalSectorRecord(
                                symbol=sym,
                                sector=sec,
                                industry=ind,
                                effective_from=eff_from,
                                effective_to=eff_to,
                                source=src,
                                notes=nts,
                            )
                        )
                logger.info(f"Loaded {len(self._history)} historical sector interval records from {history_csv.name}")
            except Exception as e:
                logger.error(f"Failed loading sector history CSV {history_csv}: {e}")

    def get_sector(self, symbol: str, as_of: Any = None, version: str | None = None) -> str:
        """
        DATA-007: Return the primary sector name for a symbol at a given historical timestamp `as_of`.
        If `as_of` is None, returns the active or specified version's sector.
        """
        sym_upper = symbol.upper()

        # Point-in-time historical resolution (strict: no lookahead fallback when as_of is provided)
        if as_of is not None:
            if self._history:
                target_date_str = self._normalize_date(as_of)
                for rec in self._history:
                    if rec.symbol == sym_upper:
                        if rec.effective_from <= target_date_str:
                            if rec.effective_to is None or target_date_str <= rec.effective_to:
                                return rec.sector
            return "Unknown"

        # Fallback to active/target snapshot
        target_version = version or self._active_version
        if target_version and target_version in self._snapshots:
            mapping = self._snapshots[target_version].mappings.get(sym_upper, {})
            if "sector" in mapping:
                return mapping["sector"]

        # Fallback to static mapping in config if present
        from config import SECTOR_MAP

        return SECTOR_MAP.get(sym_upper, "Unknown")

    def get_sector_as_of(self, symbol: str, as_of: Any) -> str:
        """Explicit convenience alias for get_sector(symbol, as_of=as_of) (DATA-007)."""
        return self.get_sector(symbol, as_of=as_of)

    def get_industry(self, symbol: str, as_of: Any = None, version: str | None = None) -> str:
        """
        DATA-007: Return granular industry description for a symbol at historical timestamp `as_of`.
        """
        sym_upper = symbol.upper()

        if as_of is not None:
            if self._history:
                target_date_str = self._normalize_date(as_of)
                for rec in self._history:
                    if rec.symbol == sym_upper:
                        if rec.effective_from <= target_date_str:
                            if rec.effective_to is None or target_date_str <= rec.effective_to:
                                return rec.industry
            return "Unknown"

        target_version = version or self._active_version
        if target_version and target_version in self._snapshots:
            mapping = self._snapshots[target_version].mappings.get(sym_upper, {})
            return mapping.get("industry", "Unknown")
        return "Unknown"

    def get_industry_as_of(self, symbol: str, as_of: Any) -> str:
        """Explicit convenience alias for get_industry(symbol, as_of=as_of) (DATA-007)."""
        return self.get_industry(symbol, as_of=as_of)

    def get_sector_map_as_of(self, as_of: Any) -> dict[str, str]:
        """
        DATA-007: Return a full {symbol: sector} dictionary valid at historical date `as_of`.
        """
        target_date_str = self._normalize_date(as_of)
        sector_map: dict[str, str] = {}

        if self._history:
            for rec in self._history:
                if rec.effective_from <= target_date_str:
                    if rec.effective_to is None or target_date_str <= rec.effective_to:
                        sector_map[rec.symbol] = rec.sector

        if not sector_map:
            from config import SECTOR_MAP

            return dict(SECTOR_MAP)

        return sector_map

    def get_symbols_for_sector(
        self, sector: str, as_of: Any = None, version: str | None = None
    ) -> list[str]:
        """Return all symbols belonging to a specified sector, optionally at historical timestamp `as_of`."""
        sec_lower = sector.lower()

        if as_of is not None:
            s_map = self.get_sector_map_as_of(as_of)
            return sorted([sym for sym, sec in s_map.items() if sec.lower() == sec_lower])

        target_version = version or self._active_version
        if not target_version or target_version not in self._snapshots:
            return []

        mappings = self._snapshots[target_version].mappings
        return sorted([sym for sym, info in mappings.items() if info.get("sector", "").lower() == sec_lower])

    def get_sector_history(self, symbol: str | None = None) -> list[HistoricalSectorRecord]:
        """Return historical sector interval records, optionally filtered by symbol."""
        if symbol:
            sym_upper = symbol.upper()
            return [rec for rec in self._history if rec.symbol == sym_upper]
        return list(self._history)

    def get_full_sector_dict(self, version: str | None = None) -> dict[str, str]:
        """Return a simple {symbol: sector} dictionary compatible with legacy SECTOR_MAP."""
        target_version = version or self._active_version
        if target_version and target_version in self._snapshots:
            return {
                sym: info.get("sector", "Unknown") for sym, info in self._snapshots[target_version].mappings.items()
            }
        from config import SECTOR_MAP

        return dict(SECTOR_MAP)


# Global default instance
sector_map_provider = SectorMapProvider()


def get_symbol_sector(symbol: str, as_of: Any = None) -> str:
    return sector_map_provider.get_sector(symbol, as_of=as_of)


if __name__ == "__main__":
    configure_logging(log_filename="sector_provider_selftest.log")
    print("\n=== SECTOR MAP PROVIDER SELF-TEST ===")
    provider = SectorMapProvider()
    print(f"RELIANCE Sector: {provider.get_sector('RELIANCE')} | Industry: {provider.get_industry('RELIANCE')}")
    print(f"HDFCBANK Sector: {provider.get_sector('HDFCBANK')}")
    banking_stocks = provider.get_symbols_for_sector("Banking")
    print(f"Banking stocks ({len(banking_stocks)}): {banking_stocks}")
    assert "HDFCBANK" in banking_stocks
    assert "ICICIBANK" in banking_stocks
    assert provider.get_sector("INFY") == "IT"

    # Historical point-in-time checks (DATA-007)
    print("\nTesting historical point-in-time sector resolution...")
    assert provider.get_sector("TATACONSUM", as_of="2020-05-01") == "Beverages"
    assert provider.get_sector("TATACONSUM", as_of="2022-05-01") == "FMCG"
    print(f"TATACONSUM 2020: {provider.get_sector('TATACONSUM', as_of='2020-05-01')} | 2022: {provider.get_sector('TATACONSUM', as_of='2022-05-01')}")

    assert provider.get_sector("SHRIRAMFIN", as_of="2021-06-01") == "AutoFinance"
    assert provider.get_sector("SHRIRAMFIN", as_of="2023-06-01") == "NBFC"
    print(f"SHRIRAMFIN 2021: {provider.get_sector('SHRIRAMFIN', as_of='2021-06-01')} | 2023: {provider.get_sector('SHRIRAMFIN', as_of='2023-06-01')}")

    assert provider.get_sector("HDFC", as_of="2022-01-01") == "NBFC"
    assert provider.get_sector("IOC", as_of="2021-01-01") == "Energy"
    assert provider.get_sector("GAIL", as_of="2020-06-01") == "Utilities"
    assert provider.get_sector("BEL", as_of="2024-10-01") == "CapitalGoods"
    assert provider.get_sector("TRENT", as_of="2024-10-01") == "Retail"

    s_map_2020 = provider.get_sector_map_as_of("2020-06-01")
    assert s_map_2020["TATACONSUM"] == "Beverages"
    assert s_map_2020["GAIL"] == "Utilities"

    print("STATUS: PASS")

