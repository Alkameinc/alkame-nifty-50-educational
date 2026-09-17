# 1. Standard library imports
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import (
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
    HORIZON_TO_MA_PERIOD,
    HORIZON_TO_SR_METHOD,
    configure_logging,
)
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 5. Domain Exceptions (REF-002)
# ---------------------------------------------------------------------------
class ReferenceLevelError(Exception):
    """Base exception for ReferenceLevelEngine."""
    pass


class InsufficientDataError(ReferenceLevelError):
    """Raised when data has insufficient bars or missing price series."""
    pass


# ---------------------------------------------------------------------------
# 6. Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ReferenceLevels:
    """
    REF-001: Absolute reference levels.
    When status is UNAVAILABLE, band values must be None rather than manufactured fallbacks.
    """
    symbol: str
    as_of: datetime
    current_price: float
    status: str = "AVAILABLE"  # "AVAILABLE" | "UNAVAILABLE"
    moving_average_value: float | None = None
    moving_average_period: int | None = None
    support_band_low: float | None = None
    support_band_high: float | None = None
    resistance_band_low: float | None = None
    resistance_band_high: float | None = None
    user_avg_cost: float | None = None
    error: str | None = None

    def __post_init__(self):
        # DATA-004: Standardize internal timestamps on timezone-aware UTC
        if self.as_of is not None and getattr(self.as_of, "tzinfo", None) is None:
            self.as_of = self.as_of.replace(tzinfo=timezone.utc)

    @property
    def is_available(self) -> bool:
        return self.status == "AVAILABLE"


@dataclass
class ReferenceLevelDeltas:
    pct_from_current_price: float
    pct_from_moving_average: float
    pct_from_support_band: float
    pct_from_resistance_band: float
    pct_from_user_avg_cost: float | None
    status: str = "AVAILABLE"  # "AVAILABLE" | "UNAVAILABLE"


# ---------------------------------------------------------------------------
# 7. Classes and functions
# ---------------------------------------------------------------------------
class ReferenceLevelEngine:
    """
    Computes four price reference levels (Current Price, Moving Average,
    Support/Resistance Band, User Average Cost) and their distance (deltas)
    from current price for a given stock and horizon.
    """

    def __init__(self):
        pass

    def compute_swing_levels(
        self,
        df: pd.DataFrame,
        lookback_window: int,
        quantile_threshold: float = 0.10,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        Calculates swing highs and lows to define support and resistance bands over a lookback window.
        Returns: support_low, support_high, resistance_low, resistance_high.
        Under REF-001, returns (None, None, None, None) on failure instead of fabricating current_price fallbacks.
        """
        if df is None or df.empty or "High" not in df.columns or "Low" not in df.columns or len(df) < 5:
            return None, None, None, None

        if len(df) < lookback_window:
            lookback_window = len(df)

        recent_df = df.iloc[-lookback_window:]

        highs = recent_df["High"].sort_values(ascending=False).dropna().values
        lows = recent_df["Low"].sort_values(ascending=True).dropna().values

        if len(highs) == 0 or len(lows) == 0:
            # REF-001: Never manufacture current_price offsets on failure
            return None, None, None, None

        # Resistance band (e.g. top quantile_threshold of highs)
        n_res = max(1, int(len(highs) * quantile_threshold))
        res_band = highs[:n_res]
        resistance_high = float(np.max(res_band))
        resistance_low = float(np.min(res_band))

        # Support band (e.g. bottom quantile_threshold of lows)
        n_sup = max(1, int(len(lows) * quantile_threshold))
        sup_band = lows[:n_sup]
        support_low = float(np.min(sup_band))
        support_high = float(np.max(sup_band))

        # Ensure logical ordering
        if support_high > resistance_low:
            # REF-001: In compressed/inverting volatility, do not manufacture synthetic bands
            return None, None, None, None

        return support_low, support_high, resistance_low, resistance_high

    def compute_bollinger_bands(
        self, df: pd.DataFrame, period: int = BOLLINGER_PERIOD, std_dev: float = BOLLINGER_STD_DEV
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        Uses Bollinger bands to define support/resistance.
        Support band is lower band ± a small margin, resistance is upper band ± a small margin.
        Under REF-001, returns (None, None, None, None) on failure instead of fabricating current_price fallbacks.
        """
        if df is None or df.empty or "Close" not in df.columns or len(df) < period:
            return None, None, None, None

        sma = df["Close"].rolling(window=period).mean().iloc[-1]
        std = df["Close"].rolling(window=period).std().iloc[-1]

        if pd.isna(sma) or pd.isna(std) or std < 0:
            # REF-001: Never fabricate current_price offsets on failure
            return None, None, None, None

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        # Define a small 0.5% band around the bollinger lines
        return float(lower * 0.995), float(lower * 1.005), float(upper * 0.995), float(upper * 1.005)

    def get_reference_levels(
        self, symbol: str, stock_df: pd.DataFrame, horizon: str, user_avg_cost: float | None = None
    ) -> ReferenceLevels:
        """
        Computes the absolute values of the reference levels.
        REF-001: Returns status="UNAVAILABLE" and levels=None on failure.
        REF-002: Separates expected data issues from unexpected programming errors, logging structured errors.
        """
        # Validate inputs (expected data issues)
        if stock_df is None or stock_df.empty or "Close" not in stock_df.columns:
            msg = f"Insufficient or empty market data for symbol={symbol}"
            logger.warning(
                f"REF-002: Reference level calculation unavailable: {msg}",
                extra={
                    "symbol": symbol,
                    "timestamp": str(datetime.now(timezone.utc)),
                    "operation": "get_reference_levels",
                    "exception_type": "InsufficientDataError",
                    "detail": msg,
                },
            )
            health_registry.report("reference_level_engine", ok=False, detail=msg)
            return ReferenceLevels(
                symbol=symbol,
                as_of=datetime.now(timezone.utc),
                current_price=0.0,
                status="UNAVAILABLE",
                error=msg,
                user_avg_cost=user_avg_cost,
            )

        if len(stock_df) < 5:
            msg = f"Insufficient market data bars ({len(stock_df)} < 5) for symbol={symbol}"
            logger.warning(
                f"REF-002: Reference level calculation unavailable: {msg}",
                extra={
                    "symbol": symbol,
                    "timestamp": str(datetime.now(timezone.utc)),
                    "operation": "get_reference_levels",
                    "exception_type": "InsufficientDataError",
                    "detail": msg,
                },
            )
            health_registry.report("reference_level_engine", ok=False, detail=msg)
            try:
                cp = float(stock_df["Close"].iloc[-1])
            except Exception:
                cp = 0.0
            return ReferenceLevels(
                symbol=symbol,
                as_of=datetime.now(timezone.utc),
                current_price=cp,
                status="UNAVAILABLE",
                error=msg,
                user_avg_cost=user_avg_cost,
            )

        as_of = stock_df.index[-1]
        current_price = 0.0

        try:
            if hasattr(as_of, "to_pydatetime"):
                as_of = as_of.to_pydatetime()
            elif not isinstance(as_of, datetime):
                as_of = datetime.now(timezone.utc)
            if getattr(as_of, "tzinfo", None) is None:
                as_of = as_of.replace(tzinfo=timezone.utc)

            current_price = float(stock_df["Close"].iloc[-1])

            ma_period = HORIZON_TO_MA_PERIOD.get(horizon, 50)
            ma_series = stock_df["Close"].rolling(window=ma_period, min_periods=1).mean()
            ma_value = float(ma_series.iloc[-1]) if not ma_series.empty and not pd.isna(ma_series.iloc[-1]) else None

            sr_method = HORIZON_TO_SR_METHOD.get(horizon, "swing_levels")
            if sr_method == "bollinger":
                sl, sh, rl, rh = self.compute_bollinger_bands(stock_df)
            else:
                lookback_map = {"30D": 60, "3M": 120, "6M": 252, "1Y": 500}
                lookback = lookback_map.get(horizon, 120)
                sl, sh, rl, rh = self.compute_swing_levels(stock_df, lookback)

            if sl is None or sh is None or rl is None or rh is None or ma_value is None:
                msg = f"Failed to resolve complete reference levels for symbol={symbol}, horizon={horizon}"
                logger.warning(
                    f"REF-002: Reference levels incomplete: {msg}",
                    extra={
                        "symbol": symbol,
                        "timestamp": str(as_of),
                        "operation": "get_reference_levels",
                        "exception_type": "InsufficientDataError",
                        "detail": msg,
                    },
                )
                health_registry.report("reference_level_engine", ok=False, detail=msg)
                return ReferenceLevels(
                    symbol=symbol,
                    as_of=as_of,
                    current_price=current_price,
                    status="UNAVAILABLE",
                    error=msg,
                    user_avg_cost=user_avg_cost,
                )

            levels = ReferenceLevels(
                symbol=symbol,
                as_of=as_of,
                current_price=current_price,
                status="AVAILABLE",
                moving_average_value=ma_value,
                moving_average_period=ma_period,
                support_band_low=sl,
                support_band_high=sh,
                resistance_band_low=rl,
                resistance_band_high=rh,
                user_avg_cost=user_avg_cost,
            )
            health_registry.report("reference_level_engine", ok=True)
            return levels

        except (KeyError, ValueError, TypeError) as e:
            err_msg = f"Data validation error computing reference levels for {symbol}: {e}"
            logger.error(
                f"REF-002: {err_msg}",
                extra={
                    "symbol": symbol,
                    "timestamp": str(as_of),
                    "operation": "get_reference_levels",
                    "exception_type": type(e).__name__,
                    "detail": str(e),
                },
            )
            health_registry.report("reference_level_engine", ok=False, detail=err_msg, error=str(e))
            return ReferenceLevels(
                symbol=symbol,
                as_of=as_of if isinstance(as_of, datetime) else datetime.now(timezone.utc),
                current_price=current_price,
                status="UNAVAILABLE",
                error=err_msg,
                user_avg_cost=user_avg_cost,
            )
        except Exception as e:
            logger.exception(
                f"REF-002: Unexpected defect in get_reference_levels for {symbol}: {e}",
                extra={
                    "symbol": symbol,
                    "timestamp": str(as_of),
                    "operation": "get_reference_levels",
                    "exception_type": type(e).__name__,
                    "detail": str(e),
                },
            )
            health_registry.report("reference_level_engine", ok=False, detail="Unexpected engine defect", error=str(e))
            return ReferenceLevels(
                symbol=symbol,
                as_of=as_of if isinstance(as_of, datetime) else datetime.now(timezone.utc),
                current_price=current_price,
                status="UNAVAILABLE",
                error=f"Unexpected defect: {e}",
                user_avg_cost=user_avg_cost,
            )

    def compute_deltas(self, levels: ReferenceLevels) -> ReferenceLevelDeltas:
        """
        Converts absolute reference levels into percentage deltas from the current price.
        Under REF-001, if levels.is_available is False, returns deltas with 0.0 and status="UNAVAILABLE".
        """
        if not levels.is_available or levels.moving_average_value is None or levels.support_band_low is None:
            return ReferenceLevelDeltas(
                pct_from_current_price=0.0,
                pct_from_moving_average=0.0,
                pct_from_support_band=0.0,
                pct_from_resistance_band=0.0,
                pct_from_user_avg_cost=None,
                status="UNAVAILABLE",
            )

        cp = levels.current_price
        ma = levels.moving_average_value

        pct_from_ma = ((cp - ma) / ma) * 100.0 if ma > 0 else 0.0

        if cp < levels.support_band_low:
            pct_from_support = ((cp - levels.support_band_low) / levels.support_band_low) * 100.0
        elif levels.support_band_high is not None and cp > levels.support_band_high:
            pct_from_support = ((cp - levels.support_band_high) / levels.support_band_high) * 100.0
        else:
            pct_from_support = 0.0

        if levels.resistance_band_low is not None and cp < levels.resistance_band_low:
            pct_from_resistance = ((cp - levels.resistance_band_low) / levels.resistance_band_low) * 100.0
        elif levels.resistance_band_high is not None and cp > levels.resistance_band_high:
            pct_from_resistance = ((cp - levels.resistance_band_high) / levels.resistance_band_high) * 100.0
        else:
            pct_from_resistance = 0.0

        if levels.user_avg_cost is not None and levels.user_avg_cost > 0:
            pct_from_cost = ((cp - levels.user_avg_cost) / levels.user_avg_cost) * 100.0
        else:
            pct_from_cost = None

        return ReferenceLevelDeltas(
            pct_from_current_price=0.0,
            pct_from_moving_average=pct_from_ma,
            pct_from_support_band=pct_from_support,
            pct_from_resistance_band=pct_from_resistance,
            pct_from_user_avg_cost=pct_from_cost,
            status="AVAILABLE",
        )

    def validate_band_sensitivity(
        self,
        df: pd.DataFrame,
        lookbacks: list[int] | None = None,
        quantiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        REF-003: Evaluates support and resistance band stability and sensitivity
        across multiple lookback periods and quantile thresholds.
        Asserts the invariant: support_low <= support_high <= resistance_low <= resistance_high.
        """
        if lookbacks is None:
            lookbacks = [30, 60, 120, 252]
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.20]

        results = {}
        for lb in lookbacks:
            for q in quantiles:
                key = f"lb_{lb}_q_{int(q*100)}"
                sl, sh, rl, rh = self.compute_swing_levels(df, lookback_window=lb, quantile_threshold=q)
                valid = (
                    sl is not None
                    and sh is not None
                    and rl is not None
                    and rh is not None
                    and (sl <= sh)
                    and (sh <= rl)
                    and (rl <= rh)
                )
                spread_pct = ((rl - sh) / sh * 100.0) if valid and sh > 0 else 0.0
                results[key] = {
                    "lookback": lb,
                    "quantile": q,
                    "support_low": sl,
                    "support_high": sh,
                    "resistance_low": rl,
                    "resistance_high": rh,
                    "valid_ordering": valid,
                    "spread_pct": spread_pct,
                }
        return results


# ---------------------------------------------------------------------------
# 8. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="reference_level_selftest.log")
    logger.info("Running reference_level_engine.py self-test...")

    try:
        engine = ReferenceLevelEngine()

        # Synthetic data with an uptrend
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz="UTC")
        closes = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
        highs = closes + np.random.uniform(1, 5, 100)
        lows = closes - np.random.uniform(1, 5, 100)
        df = pd.DataFrame({"Close": closes, "High": highs, "Low": lows}, index=dates)

        print("\n=== REFERENCE LEVEL ENGINE SELF-TEST ===")

        # 1. Moving average check
        levels = engine.get_reference_levels("TEST", df, horizon="30D", user_avg_cost=150.0)
        assert levels.is_available is True
        deltas = engine.compute_deltas(levels)

        print(f"Current Price: {levels.current_price:.2f}")
        print(f"Moving Average ({levels.moving_average_period}): {levels.moving_average_value:.2f}")
        print(f"Pct from MA: {deltas.pct_from_moving_average:.2f}%")
        assert np.sign(levels.current_price - levels.moving_average_value) == np.sign(deltas.pct_from_moving_average)

        # 2. Support/Resistance check
        print(f"Support Band: {levels.support_band_low:.2f} - {levels.support_band_high:.2f}")
        print(f"Resistance Band: {levels.resistance_band_low:.2f} - {levels.resistance_band_high:.2f}")
        assert levels.support_band_low <= levels.support_band_high
        assert levels.resistance_band_low <= levels.resistance_band_high

        # 3. User average cost check
        print(f"User Avg Cost: {levels.user_avg_cost}")
        print(f"Pct from Cost: {deltas.pct_from_user_avg_cost}%")
        assert deltas.pct_from_user_avg_cost is not None

        # 4. No position check
        levels_no_pos = engine.get_reference_levels("TEST", df, horizon="30D", user_avg_cost=None)
        deltas_no_pos = engine.compute_deltas(levels_no_pos)
        assert deltas_no_pos.pct_from_user_avg_cost is None

        # 5. REF-001 Failure check (empty DataFrame must return UNAVAILABLE without fabricating numbers)
        empty_levels = engine.get_reference_levels("TEST", pd.DataFrame(), horizon="30D")
        assert empty_levels.is_available is False
        assert empty_levels.status == "UNAVAILABLE"
        assert empty_levels.support_band_low is None

        empty_deltas = engine.compute_deltas(empty_levels)
        assert empty_deltas.status == "UNAVAILABLE"

        # 6. Sensitivity check (REF-003)
        sens = engine.validate_band_sensitivity(df)
        assert len(sens) > 0
        assert all(s["valid_ordering"] for s in sens.values())

        print("STATUS: PASS")

    except Exception as e:
        logger.error(f"reference_level_engine.py self-test crashed: {e}")
        print(f"STATUS: FAIL - {e}")
