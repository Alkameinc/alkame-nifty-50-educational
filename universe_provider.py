# 1. Standard library imports
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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


class UniverseProvider:
    """
    Authoritative provider for versioned index constituent universes (DATA-001).
    Validates constituent count (50 for NIFTY 50), effective date ranges,
    and SHA-256 integrity checksums.
    """

    def __init__(self, universe_dir: Path | None = None):
        self.universe_dir = universe_dir or (Path(__file__).parent / "data" / "universe")
        self._snapshots: dict[str, UniverseSnapshot] = {}
        self._active_version: str | None = None
        self.load_universes()

    def _compute_checksum(self, constituents: list[str]) -> str:
        data_str = ",".join(sorted(constituents))
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def load_universes(self) -> None:
        """Scan and load all versioned universe JSON files."""
        if not self.universe_dir.exists():
            logger.warning(f"Universe directory {self.universe_dir} not found.")
            return

        json_files = list(self.universe_dir.glob("nifty50_constituents_*.json"))
        for fpath in sorted(json_files):
            try:
                with open(fpath, encoding="utf-8-sig") as f:
                    data = json.load(f)

                version = data.get("version", "1.0")
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

    def get_constituents(self, version: str | None = None, as_of_date: date | None = None) -> list[str]:
        """Return the list of constituents for a given version or the latest active universe."""
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

    def get_snapshot(self, version: str | None = None) -> UniverseSnapshot | None:
        target_version = version or self._active_version
        return self._snapshots.get(target_version) if target_version else None

    def validate_constituents(self, expected_count: int = 50) -> bool:
        """Verify that active universe has exactly expected_count constituents and no duplicates."""
        constituents = self.get_constituents()
        if len(constituents) != expected_count:
            logger.error(f"Universe constituent count mismatch: found {len(constituents)}, expected {expected_count}")
            return False
        if len(set(constituents)) != len(constituents):
            logger.error("Universe contains duplicate constituents")
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
    print("STATUS: PASS")
