# 1. Standard library imports
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party imports
import pandas as pd
import numpy as np

# 3. Local imports
from config import (
    GLOBAL_TICKERS,
    GLOBAL_RISK_ZSCORE_WARN_THRESHOLD,
    GLOBAL_RISK_ZSCORE_CRISIS_THRESHOLD,
    GLOBAL_RISK_CONFIDENCE_DOWNGRADE_ELEVATED,
    GLOBAL_RISK_CONFIDENCE_DOWNGRADE_CRISIS,
    GLOBAL_RISK_LOOKBACK_DAYS,
    CRUDE_SENSITIVE_SECTORS,
    DATA_DIR,
    ensure_directories,
    configure_logging,
)
from data_fetcher import DataFetcher
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
RISK_LEVEL_NORMAL = "NORMAL"
RISK_LEVEL_ELEVATED = "ELEVATED"
RISK_LEVEL_CRISIS = "CRISIS"

TOGGLE_STATE_PATH = DATA_DIR / "global_risk_toggle_state.json"

# Which sectors each driver is considered to meaningfully expose.
# VIX and Gold moves are treated as broad risk-off signals (affect everything);
# Crude is treated as sector-specific per your requirement that we distinguish
# "affects all stocks" vs "affects a sector" vs "affects one stock".
DRIVER_EXPOSED_SECTORS = {
    "VIX": ["ALL"],
    "DOLLAR_INDEX": ["ALL"],
    "GOLD": ["ALL"],
    "CRUDE_BRENT": CRUDE_SENSITIVE_SECTORS,
    "CRUDE_WTI": CRUDE_SENSITIVE_SECTORS,
}

NON_EXPOSED_SECTOR_DAMPENING = 0.5  # non-exposed sectors get half the downgrade strength


@dataclass
class GlobalRiskReading:
    timestamp: datetime
    composite_zscore: float
    risk_level: str
    dominant_driver: Optional[str]
    driver_details: Dict[str, float]
    banner_message: str


@dataclass
class ToggleState:
    enabled: bool
    reason: str
    level_at_activation: Optional[str]
    activated_at: Optional[str]
    updated_at: str


class GlobalRiskMonitor:
    """
    Always-on composite risk monitor. Computes a z-score of VIX/DXY/Gold/Crude
    moves every cycle and classifies risk level. Never changes prediction
    behavior by itself — only surfaces a banner. The actual risk-adjustment
    only applies once a human explicitly enables the toggle via set_toggle().
    """

    def __init__(self, data_fetcher: Optional[DataFetcher] = None):
        self.data_fetcher = data_fetcher or DataFetcher()
        ensure_directories()
        self._toggle_state = self._load_toggle_state()

    # -----------------------------------------------------------------
    # Toggle state persistence
    # -----------------------------------------------------------------
    def _load_toggle_state(self) -> ToggleState:
        try:
            if TOGGLE_STATE_PATH.exists():
                with open(TOGGLE_STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ToggleState(**data)
        except Exception as e:
            logger.error(f"Failed loading toggle state, defaulting to OFF: {e}")
        return ToggleState(
            enabled=False, reason="", level_at_activation=None,
            activated_at=None, updated_at=datetime.now().isoformat(),
        )

    def _save_toggle_state(self) -> None:
        f = None
        try:
            f = open(TOGGLE_STATE_PATH, "w", encoding="utf-8")
            json.dump(asdict(self._toggle_state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed saving toggle state: {e}")
        finally:
            if f is not None:
                f.close()

    def set_toggle(self, enabled: bool, reason: str = "", current_level: Optional[str] = None) -> ToggleState:
        """Human-gated override switch. Nothing in this system flips this automatically."""
        try:
            now_iso = datetime.now().isoformat()
            self._toggle_state = ToggleState(
                enabled=enabled,
                reason=reason,
                level_at_activation=current_level if enabled else self._toggle_state.level_at_activation,
                activated_at=now_iso if enabled else self._toggle_state.activated_at,
                updated_at=now_iso,
            )
            self._save_toggle_state()
            logger.info(f"Global risk toggle set to {enabled} by human. Reason: {reason or '(none given)'}")
        except Exception as e:
            logger.error(f"Failed setting toggle state: {e}")
        return self._toggle_state

    def get_toggle_state(self) -> ToggleState:
        return self._toggle_state

    # -----------------------------------------------------------------
    # Composite risk computation
    # -----------------------------------------------------------------
    def _pct_change_zscore(self, df: pd.DataFrame) -> Optional[float]:
        """Compute the z-score of the most recent bar's % change relative to
        its own rolling history — i.e. 'how unusual is today's move'."""
        try:
            if df is None or df.empty or len(df) < GLOBAL_RISK_LOOKBACK_DAYS:
                return None
            closes = df["Close"].dropna()
            pct_changes = closes.pct_change().dropna()
            if len(pct_changes) < GLOBAL_RISK_LOOKBACK_DAYS:
                return None
            recent_window = pct_changes.iloc[-GLOBAL_RISK_LOOKBACK_DAYS:]
            mean = recent_window.mean()
            std = recent_window.std()
            if std == 0 or np.isnan(std):
                return 0.0
            latest_change = pct_changes.iloc[-1]
            return float((latest_change - mean) / std)
        except Exception as e:
            logger.error(f"Failed computing pct-change z-score: {e}")
            return None

    def compute_composite_risk(self) -> GlobalRiskReading:
        """
        Fetch VIX/DXY/Gold/Crude data and compute a composite risk reading.
        Missing tickers are excluded from the composite rather than crashing
        the whole computation — degrades gracefully.
        """
        now = datetime.now()
        if hasattr(self, "_cached_reading") and hasattr(self, "_cached_time"):
            if (now - self._cached_time).total_seconds() < 300: # 5 minutes
                return self._cached_reading

        driver_zscores: Dict[str, float] = {}
        try:
            global_data = self.data_fetcher.fetch_global_tickers()
            for name in ["VIX", "DOLLAR_INDEX", "GOLD", "CRUDE_BRENT"]:
                df = global_data.get(name)
                z = self._pct_change_zscore(df)
                if z is not None:
                    driver_zscores[name] = z
                else:
                    logger.warning(f"Could not compute z-score for {name} — excluding from composite.")

            if not driver_zscores:
                logger.error("No global risk drivers available at all — returning NORMAL as a safe default.")
                health_registry.report("global_risk_monitor", ok=False, detail="No global risk drivers available")
                return GlobalRiskReading(
                    timestamp=datetime.now(), composite_zscore=0.0, risk_level=RISK_LEVEL_NORMAL,
                    dominant_driver=None, driver_details={},
                    banner_message="Global risk data unavailable — monitor could not compute a reading.",
                )

            composite = float(np.mean([abs(z) for z in driver_zscores.values()]))
            dominant_driver = max(driver_zscores, key=lambda k: abs(driver_zscores[k]))

            if composite >= GLOBAL_RISK_ZSCORE_CRISIS_THRESHOLD:
                level = RISK_LEVEL_CRISIS
            elif composite >= GLOBAL_RISK_ZSCORE_WARN_THRESHOLD:
                level = RISK_LEVEL_ELEVATED
            else:
                level = RISK_LEVEL_NORMAL

            banner = self._build_banner_message(level, dominant_driver, driver_zscores)
            health_registry.report("global_risk_monitor", ok=True)
            
            result = GlobalRiskReading(
                timestamp=datetime.now(), composite_zscore=composite, risk_level=level,
                dominant_driver=dominant_driver, driver_details=driver_zscores, banner_message=banner,
            )
            self._cached_reading = result
            self._cached_time = now
            return result

        except Exception as e:
            logger.error(f"Failed computing composite global risk: {e}")
            health_registry.report("global_risk_monitor", ok=False, detail="Failed computing composite risk", error=str(e))
            result = GlobalRiskReading(
                timestamp=datetime.now(), composite_zscore=0.0, risk_level=RISK_LEVEL_NORMAL,
                dominant_driver=None, driver_details={},
                banner_message=f"Global risk monitor error — defaulting to NORMAL. ({e})",
            )
            self._cached_reading = result
            self._cached_time = now
            return result

    @staticmethod
    def _build_banner_message(level: str, dominant_driver: Optional[str], driver_zscores: Dict[str, float]) -> str:
        if level == RISK_LEVEL_NORMAL:
            return "Global market conditions normal."
        driver_label = dominant_driver.replace("_", " ").title() if dominant_driver else "market conditions"
        z = driver_zscores.get(dominant_driver, 0.0) if dominant_driver else 0.0
        direction = "spiked" if z > 0 else "dropped sharply"
        if level == RISK_LEVEL_CRISIS:
            return (
                f"Global Risk Indicator: CRISIS — {driver_label} has {direction} well beyond normal range "
                f"(z-score {z:.2f}). Trading conditions may be significantly affected. "
                f"Enable risk-adjusted predictions?"
            )
        return (
            f"Global Risk Indicator: ELEVATED — {driver_label} has {direction} beyond normal range "
            f"(z-score {z:.2f}). Trading conditions may be affected. Enable risk-adjusted predictions?"
        )

    # -----------------------------------------------------------------
    # Confidence adjustment (only applied by predictor.py when toggle is ON)
    # -----------------------------------------------------------------
    def get_confidence_multiplier(self, sector: Optional[str], reading: GlobalRiskReading) -> float:
        """
        Returns the multiplier predictor.py should apply to a stock's confidence
        score, based on the CURRENT toggle state and the stock's sector exposure
        to whichever driver is dominant. Returns 1.0 (no change) if the toggle
        is off — the human must opt in before this has any effect.
        """
        try:
            if not self._toggle_state.enabled:
                return 1.0

            if reading.risk_level == RISK_LEVEL_CRISIS:
                base_multiplier = GLOBAL_RISK_CONFIDENCE_DOWNGRADE_CRISIS
            elif reading.risk_level == RISK_LEVEL_ELEVATED:
                base_multiplier = GLOBAL_RISK_CONFIDENCE_DOWNGRADE_ELEVATED
            else:
                return 1.0

            exposed_sectors = DRIVER_EXPOSED_SECTORS.get(reading.dominant_driver or "", ["ALL"])
            is_exposed = "ALL" in exposed_sectors or (sector is not None and sector in exposed_sectors)

            if is_exposed:
                return base_multiplier
            # Non-exposed sectors still get a milder downgrade — a global shock rarely
            # leaves anything fully untouched, but shouldn't be penalized as hard.
            return 1.0 - (1.0 - base_multiplier) * NON_EXPOSED_SECTOR_DAMPENING

        except Exception as e:
            logger.error(f"Failed computing confidence multiplier for sector={sector}: {e}")
            return 1.0  # fail safe: no adjustment rather than an unpredictable one


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="global_risk_monitor_selftest.log")
    logger.info("Running global_risk_monitor.py self-test...")

    try:
        monitor = GlobalRiskMonitor()
        print("\n=== GLOBAL RISK MONITOR SELF-TEST RESULT ===")

        reading = monitor.compute_composite_risk()
        print(f"Composite z-score: {reading.composite_zscore:.2f}")
        print(f"Risk level: {reading.risk_level}")
        print(f"Dominant driver: {reading.dominant_driver}")
        print(f"Banner: {reading.banner_message}")

        # Toggle OFF by default -> multiplier must be 1.0 regardless of risk level
        multiplier_off = monitor.get_confidence_multiplier("Energy", reading)
        print(f"Confidence multiplier with toggle OFF: {multiplier_off}")
        assert multiplier_off == 1.0, "Multiplier must be 1.0 (no effect) while toggle is off"

        # Turn toggle ON manually (simulating a human clicking 'enable' on the dashboard)
        monitor.set_toggle(True, reason="Self-test: simulated crude oil shock", current_level=reading.risk_level)
        state = monitor.get_toggle_state()
        print(f"Toggle state after manual enable: enabled={state.enabled}, reason='{state.reason}'")
        assert state.enabled is True

        # Force a synthetic CRISIS reading to test the multiplier logic deterministically
        # (we don't rely on live data actually being in a crisis state for this test)
        synthetic_reading = GlobalRiskReading(
            timestamp=datetime.now(), composite_zscore=3.0, risk_level=RISK_LEVEL_CRISIS,
            dominant_driver="CRUDE_BRENT", driver_details={"CRUDE_BRENT": 3.0},
            banner_message="synthetic test reading",
        )
        exposed_multiplier = monitor.get_confidence_multiplier("Energy", synthetic_reading)   # Energy IS crude-sensitive
        unexposed_multiplier = monitor.get_confidence_multiplier("IT", synthetic_reading)     # IT is NOT crude-sensitive
        print(f"Exposed sector (Energy) multiplier during synthetic crude CRISIS: {exposed_multiplier}")
        print(f"Unexposed sector (IT) multiplier during synthetic crude CRISIS: {unexposed_multiplier}")
        assert exposed_multiplier == GLOBAL_RISK_CONFIDENCE_DOWNGRADE_CRISIS
        assert unexposed_multiplier > exposed_multiplier, "Unexposed sector should be downgraded less than exposed sector"

        # Reset toggle back off so repeated test runs start clean
        monitor.set_toggle(False, reason="Self-test cleanup")

        print("STATUS: PASS")
        logger.info("global_risk_monitor.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"global_risk_monitor.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — assertion error: {ae}")
    except Exception as e:
        logger.error(f"global_risk_monitor.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
