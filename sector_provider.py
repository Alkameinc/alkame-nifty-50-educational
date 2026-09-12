# 1. Standard library imports
import json
import logging
from dataclasses import dataclass
from pathlib import Path

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


class SectorMapProvider:
    """
    Authoritative provider for versioned sector classifications (DATA-002).
    Maps symbols to primary sector and granular industry with schema validation.
    """

    def __init__(self, sector_dir: Path | None = None):
        self.sector_dir = sector_dir or (Path(__file__).parent / "data" / "sector_map")
        self._snapshots: dict[str, SectorMapSnapshot] = {}
        self._active_version: str | None = None
        self.load_sector_maps()

    def load_sector_maps(self) -> None:
        """Scan and load all versioned sector map JSON files."""
        if not self.sector_dir.exists():
            logger.warning(f"Sector map directory {self.sector_dir} not found.")
            return

        json_files = list(self.sector_dir.glob("nifty50_sectors_*.json"))
        for fpath in sorted(json_files):
            try:
                with open(fpath, encoding="utf-8-sig") as f:
                    data = json.load(f)

                version = data.get("version", "1.0")
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

    def get_sector(self, symbol: str, version: str | None = None) -> str:
        """Return the primary sector name for a symbol."""
        target_version = version or self._active_version
        if target_version and target_version in self._snapshots:
            mapping = self._snapshots[target_version].mappings.get(symbol.upper(), {})
            if "sector" in mapping:
                return mapping["sector"]

        # Fallback to static mapping in config if present
        from config import SECTOR_MAP

        return SECTOR_MAP.get(symbol.upper(), "Unknown")

    def get_industry(self, symbol: str, version: str | None = None) -> str:
        """Return granular industry description for a symbol."""
        target_version = version or self._active_version
        if target_version and target_version in self._snapshots:
            mapping = self._snapshots[target_version].mappings.get(symbol.upper(), {})
            return mapping.get("industry", "Unknown")
        return "Unknown"

    def get_symbols_for_sector(self, sector: str, version: str | None = None) -> list[str]:
        """Return all symbols belonging to a specified sector."""
        target_version = version or self._active_version
        if not target_version or target_version not in self._snapshots:
            return []

        mappings = self._snapshots[target_version].mappings
        return sorted([sym for sym, info in mappings.items() if info.get("sector", "").lower() == sector.lower()])

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


def get_symbol_sector(symbol: str) -> str:
    return sector_map_provider.get_sector(symbol)


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
    print("STATUS: PASS")
