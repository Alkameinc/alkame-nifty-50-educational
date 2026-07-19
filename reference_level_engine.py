# 1. Standard library imports
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

# 2. Third-party imports
import pandas as pd
import numpy as np

# 3. Local imports
from config import (
    HORIZON_TO_MA_PERIOD,
    HORIZON_TO_SR_METHOD,
    configure_logging,
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
)
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ReferenceLevels:
    symbol: str
    as_of: datetime
    current_price: float
    moving_average_value: float
    moving_average_period: int
    support_band_low: float
    support_band_high: float
    resistance_band_low: float
    resistance_band_high: float
    user_avg_cost: Optional[float]

@dataclass
class ReferenceLevelDeltas:
    pct_from_current_price: float
    pct_from_moving_average: float
    pct_from_support_band: float
    pct_from_resistance_band: float
    pct_from_user_avg_cost: Optional[float]

# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class ReferenceLevelEngine:
    """
    Computes four price reference levels (Current Price, Moving Average,
    Support/Resistance Band, User Average Cost) and their distance (deltas)
    from current price for a given stock and horizon.
    """
    
    def __init__(self):
        pass

    def compute_swing_levels(self, df: pd.DataFrame, lookback_window: int) -> tuple[float, float, float, float]:
        """
        Calculates swing highs and lows to define support and resistance bands over a lookback window.
        Returns: support_low, support_high, resistance_low, resistance_high
        """
        if len(df) < lookback_window:
            lookback_window = len(df)
            
        recent_df = df.iloc[-lookback_window:]
        
        # A simple method: group highs and lows into bands. 
        # For resistance, we take the top quartile of highs.
        # For support, we take the bottom quartile of lows.
        highs = recent_df["High"].sort_values(ascending=False).dropna().values
        lows = recent_df["Low"].sort_values(ascending=True).dropna().values
        
        if len(highs) == 0 or len(lows) == 0:
            current_price = df["Close"].iloc[-1]
            return current_price * 0.99, current_price * 0.995, current_price * 1.005, current_price * 1.01
            
        # Resistance band (e.g. top 10% of highs)
        n_res = max(1, int(len(highs) * 0.10))
        res_band = highs[:n_res]
        resistance_high = np.max(res_band)
        resistance_low = np.min(res_band)
        
        # Support band (e.g. bottom 10% of lows)
        n_sup = max(1, int(len(lows) * 0.10))
        sup_band = lows[:n_sup]
        support_low = np.min(sup_band)
        support_high = np.max(sup_band)
        
        # Ensure logical ordering
        if support_high > resistance_low:
            # Fallback if bands cross
            current_price = df["Close"].iloc[-1]
            return current_price * 0.95, current_price * 0.96, current_price * 1.04, current_price * 1.05
            
        return support_low, support_high, resistance_low, resistance_high

    def compute_bollinger_bands(self, df: pd.DataFrame, period: int = BOLLINGER_PERIOD, std_dev: float = BOLLINGER_STD_DEV) -> tuple[float, float, float, float]:
        """
        Uses Bollinger bands to define support/resistance.
        Support band is lower band ± a small margin, resistance is upper band ± a small margin.
        """
        if len(df) < period:
            period = len(df)
        
        sma = df["Close"].rolling(window=period).mean().iloc[-1]
        std = df["Close"].rolling(window=period).std().iloc[-1]
        
        if pd.isna(sma) or pd.isna(std):
            current = df["Close"].iloc[-1]
            return current * 0.99, current * 0.995, current * 1.005, current * 1.01
            
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        # Define a small 0.5% band around the bollinger lines
        return lower * 0.995, lower * 1.005, upper * 0.995, upper * 1.005

    def get_reference_levels(self, symbol: str, stock_df: pd.DataFrame, horizon: str, user_avg_cost: Optional[float] = None) -> ReferenceLevels:
        """
        Computes the absolute values of the reference levels.
        """
        try:
            current_price = stock_df["Close"].iloc[-1]
            as_of = stock_df.index[-1]
            
            ma_period = HORIZON_TO_MA_PERIOD.get(horizon, 50)
            ma_value = stock_df["Close"].rolling(window=ma_period, min_periods=1).mean().iloc[-1]
            
            sr_method = HORIZON_TO_SR_METHOD.get(horizon, "swing_levels")
            if sr_method == "bollinger":
                sl, sh, rl, rh = self.compute_bollinger_bands(stock_df)
            else:
                # Use a lookback window suitable for the horizon
                # e.g., 30D -> 60 days, 1Y -> 252 days
                lookback_map = {"30D": 60, "3M": 120, "6M": 252, "1Y": 500}
                lookback = lookback_map.get(horizon, 120)
                sl, sh, rl, rh = self.compute_swing_levels(stock_df, lookback)
                
            levels = ReferenceLevels(
                symbol=symbol,
                as_of=as_of,
                current_price=current_price,
                moving_average_value=ma_value,
                moving_average_period=ma_period,
                support_band_low=sl,
                support_band_high=sh,
                resistance_band_low=rl,
                resistance_band_high=rh,
                user_avg_cost=user_avg_cost
            )
            health_registry.report("reference_level_engine", ok=True)
            return levels
            
        except Exception as e:
            logger.error(f"Failed to compute reference levels for {symbol} ({horizon}): {e}")
            health_registry.report("reference_level_engine", ok=False, detail="Failed to compute levels", error=str(e))
            current = stock_df["Close"].iloc[-1] if not stock_df.empty else 1.0
            return ReferenceLevels(
                symbol=symbol, as_of=datetime.now(), current_price=current,
                moving_average_value=current, moving_average_period=1,
                support_band_low=current, support_band_high=current,
                resistance_band_low=current, resistance_band_high=current,
                user_avg_cost=None
            )

    def compute_deltas(self, levels: ReferenceLevels) -> ReferenceLevelDeltas:
        """
        Converts absolute reference levels into percentage deltas from the current price.
        """
        cp = levels.current_price
        
        # pct_from_ma = (CP - MA) / MA * 100
        pct_from_ma = ((cp - levels.moving_average_value) / levels.moving_average_value) * 100.0 if levels.moving_average_value > 0 else 0.0
        
        # Distance from nearest support band edge. 
        # Negative means below support. Positive means above support.
        # Let's say if cp < support_band_low, we are below.
        if cp < levels.support_band_low:
            pct_from_support = ((cp - levels.support_band_low) / levels.support_band_low) * 100.0
        elif cp > levels.support_band_high:
            pct_from_support = ((cp - levels.support_band_high) / levels.support_band_high) * 100.0
        else:
            pct_from_support = 0.0  # inside the band
            
        # Resistance band
        if cp < levels.resistance_band_low:
            pct_from_resistance = ((cp - levels.resistance_band_low) / levels.resistance_band_low) * 100.0
        elif cp > levels.resistance_band_high:
            pct_from_resistance = ((cp - levels.resistance_band_high) / levels.resistance_band_high) * 100.0
        else:
            pct_from_resistance = 0.0  # inside the band
            
        if levels.user_avg_cost is not None and levels.user_avg_cost > 0:
            pct_from_cost = ((cp - levels.user_avg_cost) / levels.user_avg_cost) * 100.0
        else:
            pct_from_cost = None
            
        return ReferenceLevelDeltas(
            pct_from_current_price=0.0,
            pct_from_moving_average=pct_from_ma,
            pct_from_support_band=pct_from_support,
            pct_from_resistance_band=pct_from_resistance,
            pct_from_user_avg_cost=pct_from_cost
        )


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    configure_logging(log_filename="reference_level_selftest.log")
    logger.info("Running reference_level_engine.py self-test...")
    
    try:
        engine = ReferenceLevelEngine()
        
        # Synthetic data with an uptrend
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        closes = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
        highs = closes + np.random.uniform(1, 5, 100)
        lows = closes - np.random.uniform(1, 5, 100)
        df = pd.DataFrame({"Close": closes, "High": highs, "Low": lows}, index=dates)
        
        print("\n=== REFERENCE LEVEL ENGINE SELF-TEST ===")
        
        # 1. Moving average check
        levels = engine.get_reference_levels("TEST", df, horizon="30D", user_avg_cost=150.0)
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
        
        print("STATUS: PASS")
        
    except Exception as e:
        logger.error(f"reference_level_engine.py self-test crashed: {e}")
        print(f"STATUS: FAIL - {e}")
