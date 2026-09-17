# 1. Standard library imports
from dataclasses import dataclass
import logging
import re
from typing import Any

# 2. Third-party imports
import numpy as np
import pandas as pd

# Suppress pandas future warnings about downcasting
pd.set_option('future.no_silent_downcasting', True)

# 3. Local imports
from config import (
    ATR_EXPANSION_MULTIPLIER,
    ATR_PERIOD,
    BAR_INTERVAL,
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
    CORRELATION_BREAKDOWN_THRESHOLD,
    CORRELATION_LOOKBACK_BARS,
    GAP_THRESHOLD_PCT,
    HORIZON_INTRADAY,
    HORIZON_SCALP,
    LOW_LIQUIDITY_VOLUME_FLOOR,
    MA_FAST_PERIOD,
    MA_SLOW_PERIOD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    ORB_MINUTES,
    OUTPERFORMANCE_THRESHOLD_PCT,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    VOLUME_SPIKE_LOOKBACK_BARS,
    VOLUME_SPIKE_MULTIPLIER,
    configure_logging,
)
from reference_level_engine import ReferenceLevelDeltas
from magic_numbers_config import (
    RSI_NEUTRAL_VALUE,
    DEFAULT_MA_PERIOD,
    HORIZON_LOOKBACK_MAP,
    DEFAULT_LOOKBACK_PERIOD,
    RESISTANCE_HIGH_QUANTILE,
    RESISTANCE_LOW_QUANTILE,
    SUPPORT_HIGH_QUANTILE,
    RESISTANCE_BAND_LOWER_THRESHOLD,
    RESISTANCE_BAND_UPPER_THRESHOLD,
    TRADING_DAYS_PER_MONTH_APPROX,
    TRADING_DAYS_PER_QUARTER_APPROX,
    MUTATION_TEST_PRICE_DELTA,
    MUTATION_TEST_VOLUME_MULTIPLIER,
    DEFAULT_TEST_DAYS,
    DEFAULT_TEST_BARS_PER_DAY,
    LONG_TEST_DAYS,
    MINUTES_PER_HOUR,
)

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Suffix applied to any column that is safe to feed into model training —
# i.e. it has been shifted by one bar and cannot see the current bar's own
# not-yet-fully-realized outcome. Raw (unshifted) columns remain available
# for live dashboard alerting on the current bar, but must NEVER be passed
# to model_trainer.py.
ML_SAFE_SUFFIX = "_feat"


@dataclass
class AlignmentCoverageReport:
    """DATA-003 / FEAT-003: Timestamp alignment coverage between series."""

    source_count: int
    matched_count: int
    coverage_ratio: float
    status: str  # "HEALTHY" | "DEGRADED" | "UNAVAILABLE"
    threshold: float = 0.95


@dataclass(frozen=True)
class FeatureSpec:
    """
    FEAT-002: Causal feature contract specification.
    Guarantees that features consumed by models have strictly defined sources,
    maximum lookbacks, and lag shifts >= 1 to prevent quantitative lookahead bias.
    """

    name: str
    source: str  # "stock_ohlcv" | "index_ohlcv" | "reference_levels" | "portfolio"
    max_lookback: int
    shift: int  # Must be >= 1 for any feature used in ML training
    allowed_timestamp: str = "prior_bar_close"
    dtype: str = "float64"
    description: str = ""

    def __post_init__(self):
        if self.shift < 1:
            raise ValueError(
                f"FeatureSpec '{self.name}' violates causal contract: shift={self.shift} < 1. "
                "All ML features must have shift >= 1 to prevent lookahead bias."
            )


CANONICAL_FEATURE_CATALOG: dict[str, FeatureSpec] = {
    f"rsi{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"rsi{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=RSI_PERIOD,
        shift=1,
        description="Relative Strength Index (14-bar) shifted by 1 bar",
    ),
    f"macd_line{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"macd_line{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=MACD_SLOW,
        shift=1,
        description="MACD fast-slow EMA difference shifted by 1 bar",
    ),
    f"macd_signal{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"macd_signal{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=MACD_SLOW + MACD_SIGNAL,
        shift=1,
        description="MACD 9-bar signal EMA shifted by 1 bar",
    ),
    f"macd_histogram{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"macd_histogram{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=MACD_SLOW + MACD_SIGNAL,
        shift=1,
        description="MACD line minus signal histogram shifted by 1 bar",
    ),
    f"bb_upper{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"bb_upper{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=BOLLINGER_PERIOD,
        shift=1,
        description="Bollinger Upper Band (20-bar, 2 std) shifted by 1 bar",
    ),
    f"bb_middle{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"bb_middle{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=BOLLINGER_PERIOD,
        shift=1,
        description="Bollinger Middle SMA (20-bar) shifted by 1 bar",
    ),
    f"bb_lower{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"bb_lower{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=BOLLINGER_PERIOD,
        shift=1,
        description="Bollinger Lower Band (20-bar, 2 std) shifted by 1 bar",
    ),
    f"atr{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"atr{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=ATR_PERIOD,
        shift=1,
        description="Average True Range (14-bar) shifted by 1 bar",
    ),
    f"vwap{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"vwap{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=75,
        shift=1,
        description="Intraday Volume Weighted Average Price shifted by 1 bar",
    ),
    f"ema_fast{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"ema_fast{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=MA_FAST_PERIOD,
        shift=1,
        description="Fast Exponential Moving Average (9-bar) shifted by 1 bar",
    ),
    f"ema_slow{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"ema_slow{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=MA_SLOW_PERIOD,
        shift=1,
        description="Slow Exponential Moving Average (21-bar) shifted by 1 bar",
    ),
    f"orb_breakout{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"orb_breakout{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=75,
        shift=1,
        dtype="bool",
        description="Opening range breakout boolean indicator shifted by 1 bar",
    ),
    f"gap_pct{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"gap_pct{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=2,
        shift=1,
        description="Overnight/session gap percentage shifted by 1 bar",
    ),
    f"volume_ratio{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"volume_ratio{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=VOLUME_SPIKE_LOOKBACK_BARS,
        shift=1,
        description="Volume relative to 20-bar rolling average shifted by 1 bar",
    ),
    f"outperformance_pct{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"outperformance_pct{ML_SAFE_SUFFIX}",
        source="index_ohlcv",
        max_lookback=75,
        shift=1,
        description="Stock vs NIFTY index intraday return delta shifted by 1 bar",
    ),
    f"nifty_correlation{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"nifty_correlation{ML_SAFE_SUFFIX}",
        source="index_ohlcv",
        max_lookback=CORRELATION_LOOKBACK_BARS,
        shift=1,
        description="Rolling 50-bar correlation to NIFTY index shifted by 1 bar",
    ),
    f"pct_from_ma{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"pct_from_ma{ML_SAFE_SUFFIX}",
        source="reference_levels",
        max_lookback=500,
        shift=1,
        description="Percent distance from moving average shifted by 1 bar",
    ),
    f"pct_from_support_band{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"pct_from_support_band{ML_SAFE_SUFFIX}",
        source="reference_levels",
        max_lookback=500,
        shift=1,
        description="Percent distance from support band shifted by 1 bar",
    ),
    f"pct_from_resistance_band{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"pct_from_resistance_band{ML_SAFE_SUFFIX}",
        source="reference_levels",
        max_lookback=500,
        shift=1,
        description="Percent distance from resistance band shifted by 1 bar",
    ),
    f"pct_from_user_avg_cost{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"pct_from_user_avg_cost{ML_SAFE_SUFFIX}",
        source="portfolio",
        max_lookback=1,
        shift=1,
        description="Percent distance from user entry cost shifted by 1 bar",
    ),
    f"has_position{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"has_position{ML_SAFE_SUFFIX}",
        source="portfolio",
        max_lookback=1,
        shift=1,
        dtype="float64",
        description="Position active flag shifted by 1 bar",
    ),
    f"rolling_1m_return{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"rolling_1m_return{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=21,
        shift=1,
        description="Rolling 21-day return shifted by 1 bar",
    ),
    f"rolling_3m_return{ML_SAFE_SUFFIX}": FeatureSpec(
        name=f"rolling_3m_return{ML_SAFE_SUFFIX}",
        source="stock_ohlcv",
        max_lookback=63,
        shift=1,
        description="Rolling 63-day return shifted by 1 bar",
    ),
}


def validate_feature_contract(
    features: pd.DataFrame | list[str],
    catalog: dict[str, FeatureSpec] | None = None,
) -> list[str]:
    """
    FEAT-002: Validates that every feature in the provided list or DataFrame
    complies with a causal FeatureSpec contract.
    Returns a list of error strings (empty if all valid).
    """
    if catalog is None:
        catalog = CANONICAL_FEATURE_CATALOG

    if isinstance(features, pd.DataFrame):
        feature_cols = [c for c in features.columns if c.endswith(ML_SAFE_SUFFIX)]
        # Also check for any raw, unlagged feature leak
        raw_candidates = [
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "vwap",
            "ema_fast",
            "ema_slow",
            "orb_breakout",
            "gap_pct",
            "volume_ratio",
            "outperformance_pct",
            "nifty_correlation",
            "pct_from_ma",
            "pct_from_support_band",
            "pct_from_resistance_band",
            "pct_from_user_avg_cost",
            "has_position",
        ]
        raw_leaks = [c for c in features.columns if c in raw_candidates]
        if raw_leaks:
            return [f"Raw unshifted feature columns present in feature DataFrame: {raw_leaks}"]
    else:
        feature_cols = list(features)

    errors = []
    for col in feature_cols:
        if col not in catalog:
            errors.append(f"Feature '{col}' is not registered in CANONICAL_FEATURE_CATALOG.")
            continue
        spec = catalog[col]
        if spec.shift < 1:
            errors.append(f"Feature '{col}' has non-causal shift={spec.shift} (must be >= 1).")
    return errors


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
def _interval_to_minutes(interval: str) -> int:
    """Parse a yfinance-style interval string ('5m', '1m', '1h') into minutes."""
    match = re.match(r"^(\d+)([mh])$", interval.strip().lower())
    if not match:
        logger.error(f"Could not parse interval '{interval}', defaulting to 5 minutes.")
        return 5
    value, unit = int(match.group(1)), match.group(2)
    return value * MINUTES_PER_HOUR if unit == "h" else value


def _validate_ohlcv(df: pd.DataFrame, name: str = "df") -> bool:
    if df is None or df.empty:
        logger.error(f"{name} is None or empty — cannot engineer features.")
        return False
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"{name} is missing required columns {missing} — cannot engineer features.")
        return False
    return True


class FeatureEngineer:
    """
    Computes technical indicators and event flags from OHLCV bar data.

    For every indicator, two columns are produced:
      - the RAW value (usable for live/current-bar dashboard alerts)
      - a '_feat' suffixed value, shifted by one bar via .shift(1), which is
        the ONLY version model_trainer.py is allowed to consume. This is the
        concrete enforcement of the "no feature uses future data" hard rule.
    """

    def __init__(self, bar_interval: str = BAR_INTERVAL):
        self.bar_interval_minutes = _interval_to_minutes(bar_interval)
        self.orb_bar_count = max(1, ORB_MINUTES // self.bar_interval_minutes)

    # -----------------------------------------------------------------
    # Individual indicator computations
    # -----------------------------------------------------------------
    @staticmethod
    def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        try:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
            rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
            rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), RSI_NEUTRAL_VALUE)

            return rsi  # neutral RSI where undefined (e.g. no losses yet)
        except Exception as e:
            logger.error(f"Failed computing RSI: {e}")
            return pd.Series(np.nan, index=close.index)

    @staticmethod
    def compute_macd(
        close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        try:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Failed computing MACD: {e}")
            nan_series = pd.Series(np.nan, index=close.index)
            return nan_series, nan_series, nan_series

    @staticmethod
    def compute_bollinger_bands(
        close: pd.Series, period: int = BOLLINGER_PERIOD, std_dev: float = BOLLINGER_STD_DEV
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        try:
            middle = close.rolling(window=period, min_periods=period).mean()
            std = close.rolling(window=period, min_periods=period).std()
            upper = middle + std_dev * std
            lower = middle - std_dev * std
            return upper, middle, lower
        except Exception as e:
            logger.error(f"Failed computing Bollinger Bands: {e}")
            nan_series = pd.Series(np.nan, index=close.index)
            return nan_series, nan_series, nan_series

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        try:
            prev_close = df["Close"].shift(1)
            tr1 = df["High"] - df["Low"]
            tr2 = (df["High"] - prev_close).abs()
            tr3 = (df["Low"] - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=period).mean()
            return atr
        except Exception as e:
            logger.error(f"Failed computing ATR: {e}")
            return pd.Series(np.nan, index=df.index)

    @staticmethod
    def compute_vwap(df: pd.DataFrame) -> pd.Series:
        """Session VWAP, reset at the start of each calendar day."""
        try:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
            pv = typical_price * df["Volume"]
            day_key = df.index.date
            cum_pv = pv.groupby(day_key).cumsum()
            cum_vol = df["Volume"].groupby(day_key).cumsum().replace(0, np.nan)
            vwap = cum_pv / cum_vol
            return vwap
        except Exception as e:
            logger.error(f"Failed computing VWAP: {e}")
            return pd.Series(np.nan, index=df.index)

    @staticmethod
    def compute_moving_averages(
        close: pd.Series, fast: int = MA_FAST_PERIOD, slow: int = MA_SLOW_PERIOD
    ) -> tuple[pd.Series, pd.Series]:
        try:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            return ema_fast, ema_slow
        except Exception as e:
            logger.error(f"Failed computing moving averages: {e}")
            nan_series = pd.Series(np.nan, index=close.index)
            return nan_series, nan_series

    def compute_opening_range_breakout(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        For each calendar day, computes the opening-range high/low from the
        first `orb_bar_count` bars. Bars within the opening range have NaN ORB levels.
        After the opening range completes, flags +1 (breakout up), -1 (breakout
        down), or 0 for subsequent bars based on where Close sits relative to that range.
        """
        try:
            or_high = pd.Series(np.nan, index=df.index)
            or_low = pd.Series(np.nan, index=df.index)
            breakout = pd.Series(0, index=df.index)

            for day, day_df in df.groupby(df.index.date):
                if len(day_df) <= self.orb_bar_count:
                    continue
                opening_bars = day_df.iloc[: self.orb_bar_count]
                day_or_high = opening_bars["High"].max()
                day_or_low = opening_bars["Low"].min()

                post_orb_index = day_df.index[self.orb_bar_count :]
                or_high.loc[post_orb_index] = day_or_high
                or_low.loc[post_orb_index] = day_or_low

                post_orb_close = day_df.loc[post_orb_index, "Close"]
                day_breakout = np.where(
                    post_orb_close > day_or_high,
                    1,
                    np.where(post_orb_close < day_or_low, -1, 0),
                )
                breakout.loc[post_orb_index] = day_breakout

            return or_high, or_low, breakout
        except Exception as e:
            logger.error(f"Failed computing opening range breakout: {e}")
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series, pd.Series(0, index=df.index)

    @staticmethod
    def compute_gap(df: pd.DataFrame, threshold_pct: float = GAP_THRESHOLD_PCT) -> tuple[pd.Series, pd.Series]:
        """
        Computes the day's opening gap vs the previous day's last close,
        broadcast across every bar of that day (so the dashboard can show
        'today opened with a gap of X%' at any point during the session).
        """
        try:
            gap_pct = pd.Series(np.nan, index=df.index)
            gap_event = pd.Series(False, index=df.index)

            daily_groups = list(df.groupby(df.index.date))
            prev_day_last_close = None

            for day, day_df in daily_groups:
                if prev_day_last_close is not None and len(day_df) > 0:
                    day_open = day_df["Open"].iloc[0]
                    pct = ((day_open - prev_day_last_close) / prev_day_last_close) * 100.0
                    gap_pct.loc[day_df.index] = pct
                    gap_event.loc[day_df.index] = abs(pct) >= threshold_pct
                if len(day_df) > 0:
                    prev_day_last_close = day_df["Close"].iloc[-1]

            return gap_pct, gap_event
        except Exception as e:
            logger.error(f"Failed computing gap: {e}")
            return pd.Series(np.nan, index=df.index), pd.Series(False, index=df.index)

    @staticmethod
    def compute_volume_spike(
        volume: pd.Series,
        lookback: int = VOLUME_SPIKE_LOOKBACK_BARS,
        multiplier: float = VOLUME_SPIKE_MULTIPLIER,
    ) -> tuple[pd.Series, pd.Series]:
        try:
            # Baseline uses only PRIOR bars (shift(1) before rolling) so the
            # current bar's own volume is never part of its own baseline.
            rolling_avg = volume.shift(1).rolling(window=lookback, min_periods=lookback).mean()
            ratio = volume / rolling_avg.replace(0, np.nan)
            spike_event = ratio >= multiplier
            return ratio, spike_event.fillna(False)
        except Exception as e:
            logger.error(f"Failed computing volume spike: {e}")
            return pd.Series(np.nan, index=volume.index), pd.Series(False, index=volume.index)

    @staticmethod
    def compute_outperformance(
        stock_close: pd.Series, index_close: pd.Series, threshold_pct: float = OUTPERFORMANCE_THRESHOLD_PCT
    ) -> tuple[pd.Series, pd.Series]:
        """Stock's cumulative % change since day-open minus the index's, aligned by timestamp."""
        try:
            aligned_stock, aligned_index = stock_close.align(index_close, join="inner")
            if aligned_stock.empty:
                logger.warning("No overlapping timestamps between stock and index for outperformance calc.")
                return pd.Series(dtype=float), pd.Series(dtype=bool)

            day_key = aligned_stock.index.date
            stock_day_open = aligned_stock.groupby(day_key).transform("first")
            index_day_open = aligned_index.groupby(day_key).transform("first")

            stock_pct = (aligned_stock - stock_day_open) / stock_day_open * 100.0
            index_pct = (aligned_index - index_day_open) / index_day_open * 100.0
            outperformance = stock_pct - index_pct
            flag = outperformance.abs() >= threshold_pct
            return outperformance, flag
        except Exception as e:
            logger.error(f"Failed computing outperformance: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=bool)

    @staticmethod
    def compute_correlation_breakdown(
        stock_close: pd.Series,
        index_close: pd.Series,
        lookback: int = CORRELATION_LOOKBACK_BARS,
        threshold: float = CORRELATION_BREAKDOWN_THRESHOLD,
    ) -> tuple[pd.Series, pd.Series]:
        try:
            aligned_stock, aligned_index = stock_close.align(index_close, join="inner")
            if len(aligned_stock) < lookback:
                logger.warning("Not enough overlapping data for correlation breakdown calc.")
                return pd.Series(dtype=float), pd.Series(dtype=bool)

            stock_returns = aligned_stock.pct_change(fill_method=None)
            index_returns = aligned_index.pct_change(fill_method=None)
            rolling_corr = stock_returns.rolling(window=lookback, min_periods=lookback).corr(index_returns)
            breakdown_flag = rolling_corr < threshold
            return rolling_corr, breakdown_flag.fillna(False)
        except Exception as e:
            logger.error(f"Failed computing correlation breakdown: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=bool)

    @staticmethod
    def check_alignment(
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        min_ratio: float = 0.95,
    ) -> AlignmentCoverageReport:
        """DATA-003 / FEAT-003: Calculate stock/index timestamp alignment coverage."""
        if stock_df is None or stock_df.empty:
            return AlignmentCoverageReport(
                source_count=0, matched_count=0, coverage_ratio=0.0, status="UNAVAILABLE", threshold=min_ratio
            )
        if index_df is None or index_df.empty:
            return AlignmentCoverageReport(
                source_count=len(stock_df), matched_count=0, coverage_ratio=0.0, status="UNAVAILABLE", threshold=min_ratio
            )

        stock_idx = pd.to_datetime(stock_df.index, utc=True)
        index_idx = pd.to_datetime(index_df.index, utc=True)
        matched = len(stock_idx.intersection(index_idx))
        total = len(stock_idx)
        ratio = matched / total if total > 0 else 0.0

        if ratio >= min_ratio:
            status = "HEALTHY"
        elif ratio >= 0.50:
            status = "DEGRADED"
        else:
            status = "UNAVAILABLE"

        return AlignmentCoverageReport(
            source_count=total,
            matched_count=matched,
            coverage_ratio=ratio,
            status=status,
            threshold=min_ratio,
        )

    @staticmethod
    def validate_feature_contract(df_or_columns: pd.DataFrame | list[str]) -> list[str]:
        """FEAT-002: Delegate to module-level validate_feature_contract."""
        return validate_feature_contract(df_or_columns)

    # -----------------------------------------------------------------
    # Master orchestration
    # -----------------------------------------------------------------
    def engineer_features(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        reference_deltas: ReferenceLevelDeltas | None = None,
    ) -> pd.DataFrame | None:
        """
        Computes every technical feature for one stock's OHLCV DataFrame.
        Returns a new DataFrame (does not mutate the input) with:
          - raw indicator columns (safe for live/current-bar dashboard use only)
          - '_feat' suffixed, .shift(1)-lagged columns (the ONLY ones
            model_trainer.py may consume)
        Returns None if the input is invalid.
        """
        if not _validate_ohlcv(stock_df, "stock_df"):
            return None

        try:
            out = stock_df.copy()
            # Drop bars where all OHLCV are NaN or Close is missing
            if out["Close"].isna().any():
                out = out.dropna(subset=["Close"]).copy()
            close = out["Close"]
            volume = out["Volume"]

            out["rsi"] = self.compute_rsi(close)
            out["rsi_overbought"] = out["rsi"] >= RSI_OVERBOUGHT
            out["rsi_oversold"] = out["rsi"] <= RSI_OVERSOLD

            macd_line, signal_line, histogram = self.compute_macd(close)
            out["macd_line"], out["macd_signal"], out["macd_histogram"] = macd_line, signal_line, histogram
            out["macd_bullish_cross"] = (histogram > 0) & (histogram.shift(1) <= 0)
            out["macd_bearish_cross"] = (histogram < 0) & (histogram.shift(1) >= 0)

            bb_upper, bb_middle, bb_lower = self.compute_bollinger_bands(close)
            out["bb_upper"], out["bb_middle"], out["bb_lower"] = bb_upper, bb_middle, bb_lower
            out["bb_breakout_upper"] = close > bb_upper
            out["bb_breakout_lower"] = close < bb_lower

            out["atr"] = self.compute_atr(out)
            atr_rolling_avg = out["atr"].shift(1).rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()
            out["atr_expansion"] = out["atr"] >= (atr_rolling_avg * ATR_EXPANSION_MULTIPLIER)

            out["vwap"] = self.compute_vwap(out)
            out["above_vwap"] = close > out["vwap"]
            out["vwap_cross_up"] = out["above_vwap"] & (~out["above_vwap"].shift(1).fillna(False))
            out["vwap_cross_down"] = (~out["above_vwap"]) & (out["above_vwap"].shift(1).fillna(False))

            ema_fast, ema_slow = self.compute_moving_averages(close)
            out["ema_fast"], out["ema_slow"] = ema_fast, ema_slow
            ma_bullish = ema_fast > ema_slow
            out["ma_bullish_cross"] = ma_bullish & (~ma_bullish.shift(1).fillna(False))
            out["ma_bearish_cross"] = (~ma_bullish) & (ma_bullish.shift(1).fillna(False))

            or_high, or_low, orb_breakout = self.compute_opening_range_breakout(out)
            out["orb_high"], out["orb_low"], out["orb_breakout"] = or_high, or_low, orb_breakout

            gap_pct, gap_event = self.compute_gap(out)
            out["gap_pct"], out["gap_event"] = gap_pct, gap_event

            vol_ratio, vol_spike = self.compute_volume_spike(volume)
            out["volume_ratio"], out["volume_spike"] = vol_ratio, vol_spike
            rolling_vol_avg = (
                volume.shift(1)
                .rolling(window=VOLUME_SPIKE_LOOKBACK_BARS, min_periods=VOLUME_SPIKE_LOOKBACK_BARS)
                .mean()
            )
            out["low_liquidity"] = rolling_vol_avg < LOW_LIQUIDITY_VOLUME_FLOOR

            if index_df is not None and _validate_ohlcv(index_df, "index_df"):
                align_report = self.check_alignment(out, index_df)
                out.attrs["alignment_report"] = align_report
                out.attrs["alignment_status"] = align_report.status
                out.attrs["alignment_ratio"] = align_report.coverage_ratio
                out["alignment_ratio"] = align_report.coverage_ratio
                out["alignment_status"] = align_report.status

                if align_report.status in ("DEGRADED", "UNAVAILABLE"):
                    logger.warning(
                        f"DATA-003 / FEAT-003: Stock/index timestamp alignment {align_report.status}: "
                        f"{align_report.matched_count}/{align_report.source_count} bars matched "
                        f"({align_report.coverage_ratio*100:.1f}%, threshold={align_report.threshold*100:.1f}%)"
                    )

                if align_report.status == "UNAVAILABLE":
                    # FEAT-003: If alignment is unavailable (< 50%), refuse to compute spurious cross-series features
                    out["outperformance_pct"] = np.nan
                    out["outperformance_flag"] = False
                    out["nifty_correlation"] = np.nan
                    out["correlation_breakdown"] = False
                else:
                    outperf, outperf_flag = self.compute_outperformance(close, index_df["Close"])
                    out["outperformance_pct"] = outperf.reindex(out.index)
                    out["outperformance_flag"] = outperf_flag.reindex(out.index).fillna(False)

                    corr, corr_breakdown = self.compute_correlation_breakdown(close, index_df["Close"])
                    out["nifty_correlation"] = corr.reindex(out.index)
                    out["correlation_breakdown"] = corr_breakdown.reindex(out.index).fillna(False)
            else:
                logger.info("No index_df provided — skipping outperformance/correlation features.")

            if reference_deltas is not None:
                out["pct_from_ma"] = reference_deltas.pct_from_moving_average
                out["pct_from_support_band"] = reference_deltas.pct_from_support_band
                out["pct_from_resistance_band"] = reference_deltas.pct_from_resistance_band
                if reference_deltas.pct_from_user_avg_cost is not None:
                    out["pct_from_user_avg_cost"] = reference_deltas.pct_from_user_avg_cost
                    out["has_position"] = 1.0
                else:
                    out["pct_from_user_avg_cost"] = 0.0
                    out["has_position"] = 0.0
            else:
                out["pct_from_ma"] = 0.0
                out["pct_from_support_band"] = 0.0
                out["pct_from_resistance_band"] = 0.0
                out["pct_from_user_avg_cost"] = 0.0
                out["has_position"] = 0.0

            # ML-safe lagged versions: every numeric/boolean feature intended for
            # model_trainer.py gets an explicit '_feat' column shifted by exactly
            # one bar, so training never sees a bar's own not-yet-fully-realized value.
            ml_candidate_cols = [
                "rsi",
                "macd_line",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "atr",
                "vwap",
                "ema_fast",
                "ema_slow",
                "orb_breakout",
                "gap_pct",
                "volume_ratio",
                "outperformance_pct",
                "nifty_correlation",
                "pct_from_ma",
                "pct_from_support_band",
                "pct_from_resistance_band",
                "pct_from_user_avg_cost",
                "has_position",
            ]
            for col in ml_candidate_cols:
                if col in out.columns:
                    out[f"{col}{ML_SAFE_SUFFIX}"] = out[col].shift(1)

            return out

        except Exception as e:
            logger.error(f"Failed engineering features: {e}")
            return None

    def engineer_features_for_horizon(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        reference_deltas: ReferenceLevelDeltas | None = None,
        horizon: str = HORIZON_INTRADAY,
    ) -> pd.DataFrame | None:
        """
        Computes features tailored to a specific horizon.
        For daily and above, it drops intraday noise features (orb, vwap, gap)
        and computes macro momentum features (1M, 3M rolling returns).
        """
        out = self.engineer_features(stock_df, index_df, reference_deltas)
        if out is None:
            return None

        if reference_deltas is None:
            # Fix pipeline leakage: populate historical rolling S/R and MA features for the level model
            from config import BOLLINGER_PERIOD, BOLLINGER_STD_DEV, HORIZON_TO_MA_PERIOD, HORIZON_TO_SR_METHOD

            ma_period = HORIZON_TO_MA_PERIOD.get(horizon, DEFAULT_MA_PERIOD)
            rolling_ma = out["Close"].rolling(window=ma_period, min_periods=1).mean()
            out["pct_from_ma"] = ((out["Close"] - rolling_ma) / rolling_ma) * 100.0

            sr_method = HORIZON_TO_SR_METHOD.get(horizon, "bollinger")
            if sr_method == "bollinger":
                sma = out["Close"].rolling(window=BOLLINGER_PERIOD, min_periods=1).mean()
                std = out["Close"].rolling(window=BOLLINGER_PERIOD, min_periods=1).std().fillna(0)
                upper = sma + (BOLLINGER_STD_DEV * std)
                lower = sma - (BOLLINGER_STD_DEV * std)

                out["pct_from_support_band"] = np.where(
                    out["Close"] < lower * 0.995,
                    ((out["Close"] - lower * 0.995) / (lower * 0.995)) * 100.0,
                    np.where(
                        out["Close"] > lower * 1.005, ((out["Close"] - lower * 1.005) / (lower * 1.005)) * 100.0, 0.0
                    ),
                )
                out["pct_from_resistance_band"] = np.where(
                    out["Close"] < upper * 0.995,
                    ((out["Close"] - upper * 0.995) / (upper * 0.995)) * 100.0,
                    np.where(
                        out["Close"] > upper * 1.005, ((out["Close"] - upper * 1.005) / (upper * 1.005)) * 100.0, 0.0
                    ),
                )
            else:
                lookback = HORIZON_LOOKBACK_MAP.get(horizon, DEFAULT_LOOKBACK_PERIOD)

                res_high = out["High"].rolling(window=lookback, min_periods=1).quantile(RESISTANCE_HIGH_QUANTILE)
                res_low = out["High"].rolling(window=lookback, min_periods=1).quantile(RESISTANCE_LOW_QUANTILE)
                sup_high = out["Low"].rolling(window=lookback, min_periods=1).quantile(SUPPORT_HIGH_QUANTILE)
                sup_low = out["Low"].rolling(window=lookback, min_periods=1).min()

                out["pct_from_support_band"] = np.where(
                    out["Close"] < sup_low,
                    ((out["Close"] - sup_low) / sup_low) * 100.0,
                    np.where(out["Close"] > sup_high, ((out["Close"] - sup_high) / sup_high) * 100.0, 0.0),
                )
                out["pct_from_resistance_band"] = np.where(
                    out["Close"] < res_low,
                    ((out["Close"] - res_low) / res_low) * 100.0,
                    np.where(out["Close"] > res_high, ((out["Close"] - res_high) / res_high) * 100.0, 0.0),
                )

            # Re-shift the ML-safe columns to prevent lookahead
            for col in ["pct_from_ma", "pct_from_support_band", "pct_from_resistance_band"]:
                out[f"{col}{ML_SAFE_SUFFIX}"] = out[col].shift(1)

        if horizon in (HORIZON_INTRADAY, HORIZON_SCALP):
            return out

        # For non-intraday horizons (daily bars), exclude intraday-specific features
        intraday_cols = [
            "orb_breakout",
            "orb_breakout_feat",
            "orb_high",
            "orb_low",
            "vwap",
            "vwap_feat",
            "above_vwap",
            "vwap_cross_up",
            "vwap_cross_down",
            "gap_pct",
            "gap_pct_feat",
            "gap_event",
        ]
        out = out.drop(columns=[c for c in intraday_cols if c in out.columns])

        # Add long-horizon features
        # 1-month return (approx 21 trading days)
        out["rolling_1m_return"] = out["Close"].pct_change(periods=TRADING_DAYS_PER_MONTH_APPROX) * 100.0
        # 3-month return (approx 63 trading days)
        out["rolling_3m_return"] = out["Close"].pct_change(periods=TRADING_DAYS_PER_QUARTER_APPROX) * 100.0

        # Make them ML-safe
        out[f"rolling_1m_return{ML_SAFE_SUFFIX}"] = out["rolling_1m_return"].shift(1)
        out[f"rolling_3m_return{ML_SAFE_SUFFIX}"] = out["rolling_3m_return"].shift(1)

        return out

    def verify_feature_equivalence(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        timestamps: list[Any] | None = None,
        horizon: str = "INTRADAY",
        tolerance: float = 1e-9,
    ) -> "FeatureEquivalenceReport":
        return verify_feature_equivalence(
            stock_df=stock_df,
            index_df=index_df,
            timestamps=timestamps,
            horizon=horizon,
            tolerance=tolerance,
            engineer=self,
        )


@dataclass
class FeatureEquivalenceReport:
    """
    BACK-001: Verification report comparing batch (backtest) vs sliced (live) features.
    """
    is_equivalent: bool
    timestamps_checked: list[Any]
    features_checked: list[str]
    max_absolute_error: float
    mismatches: list[dict[str, Any]]
    details: str


def verify_feature_equivalence(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
    timestamps: list[Any] | None = None,
    horizon: str = "INTRADAY",
    tolerance: float = 1e-9,
    engineer: FeatureEngineer | None = None,
) -> FeatureEquivalenceReport:
    """
    BACK-001: Verifies bit-for-bit / high-precision numerical equivalence between
    batch backtest feature computation and sequential point-in-time live feature extraction.

    For each timestamp T in timestamps:
      1. Batch backtest feature vector:
         features_batch = engineer.engineer_features_for_horizon(stock_df, index_df, horizon).loc[[T]]
      2. Sliced live feature vector:
         features_live = engineer.engineer_features_for_horizon(stock_df.loc[:T], index_df.loc[:T] if index_df is not None else None, horizon).iloc[[-1:]]
      3. Assert schema match (columns, ordering, dtypes).
      4. Assert |features_batch[col] - features_live[col]| <= tolerance.
    """
    eng = engineer or FeatureEngineer()

    # Step 1: Compute full batch features (Backtest path)
    batch_df = eng.engineer_features_for_horizon(stock_df, index_df, horizon=horizon)
    if batch_df is None or batch_df.empty:
        return FeatureEquivalenceReport(
            is_equivalent=False,
            timestamps_checked=[],
            features_checked=[],
            max_absolute_error=float("inf"),
            mismatches=[{"error": "Batch feature engineering failed or returned empty dataframe"}],
            details="Batch feature engineering failed",
        )

    # Filter to ML-safe canonical features
    feature_cols = [c for c in batch_df.columns if c in CANONICAL_FEATURE_CATALOG or c.endswith(ML_SAFE_SUFFIX)]

    # Determine timestamps to check
    if timestamps is None or len(timestamps) == 0:
        warmup = min(60, len(stock_df) // 2)
        valid_indices = stock_df.index[warmup:]
        if len(valid_indices) <= 10:
            timestamps = list(valid_indices)
        else:
            step = len(valid_indices) // 10
            timestamps = [valid_indices[i * step] for i in range(10)]
            if valid_indices[-1] not in timestamps:
                timestamps.append(valid_indices[-1])

    max_err = 0.0
    mismatches: list[dict[str, Any]] = []

    for ts in timestamps:
        if ts not in stock_df.index:
            continue

        # Sliced data strictly up to ts (Replay/Live path)
        stock_slice = stock_df.loc[:ts]
        index_slice = index_df.loc[:ts] if index_df is not None else None

        live_df = eng.engineer_features_for_horizon(stock_slice, index_slice, horizon=horizon)
        if live_df is None or live_df.empty:
            mismatches.append({
                "timestamp": str(ts),
                "error": "Live sliced feature engineering returned empty",
            })
            continue

        # Schema match check:
        live_cols = [c for c in live_df.columns if c in CANONICAL_FEATURE_CATALOG or c.endswith(ML_SAFE_SUFFIX)]
        if set(feature_cols) != set(live_cols):
            missing = set(feature_cols) - set(live_cols)
            extra = set(live_cols) - set(feature_cols)
            mismatches.append({
                "timestamp": str(ts),
                "error": f"Schema mismatch: missing={missing}, extra={extra}",
            })
            continue

        batch_row = batch_df.loc[ts]
        live_row = live_df.iloc[-1]

        for col in feature_cols:
            b_val = batch_row[col]
            l_val = live_row[col]

            # Handle boolean types
            if isinstance(b_val, (bool, np.bool_)) or isinstance(l_val, (bool, np.bool_)):
                if bool(b_val) != bool(l_val):
                    mismatches.append({
                        "timestamp": str(ts),
                        "column": col,
                        "batch_val": b_val,
                        "live_val": l_val,
                        "diff": "boolean mismatch",
                    })
                continue

            # Handle NaN / None
            b_nan = pd.isna(b_val)
            l_nan = pd.isna(l_val)
            if b_nan and l_nan:
                continue
            if b_nan != l_nan:
                mismatches.append({
                    "timestamp": str(ts),
                    "column": col,
                    "batch_val": b_val,
                    "live_val": l_val,
                    "diff": "NaN disparity",
                })
                continue

            # Numerical comparison
            err = abs(float(b_val) - float(l_val))
            if err > max_err:
                max_err = err

            if err > tolerance:
                mismatches.append({
                    "timestamp": str(ts),
                    "column": col,
                    "batch_val": float(b_val),
                    "live_val": float(l_val),
                    "diff": err,
                })

    is_equiv = len(mismatches) == 0
    details = (
        f"Verified feature equivalence across {len(timestamps)} timestamps and {len(feature_cols)} features. "
        f"Max error: {max_err:.2e}, Mismatches: {len(mismatches)}."
    )

    return FeatureEquivalenceReport(
        is_equivalent=is_equiv,
        timestamps_checked=timestamps,
        features_checked=feature_cols,
        max_absolute_error=max_err,
        mismatches=mismatches,
        details=details,
    )


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="feature_engineer_selftest.log")
    logger.info("Running feature_engineer.py self-test...")

    def _build_synthetic_ohlcv(n_days: int = DEFAULT_TEST_DAYS, bars_per_day: int = DEFAULT_TEST_BARS_PER_DAY, seed: int = 42) -> pd.DataFrame:
        """Builds deterministic synthetic 5-min OHLCV bars across several
        trading days, purely offline — no network dependency for this test."""
        rng = np.random.default_rng(seed)
        rows = []
        timestamps = []
        price = 1000.0
        base_date = pd.Timestamp("2026-06-01 09:15:00")

        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                drift = rng.normal(0, 1.5)
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + rng.normal(0, 1.0))
                high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
                low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p

        df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS, index=pd.DatetimeIndex(timestamps))
        return df

    try:
        print("\n=== FEATURE ENGINEER SELF-TEST RESULT ===")
        engineer = FeatureEngineer()

        stock_df = _build_synthetic_ohlcv(n_days=5, bars_per_day=75, seed=42)
        index_df = _build_synthetic_ohlcv(n_days=5, bars_per_day=75, seed=7)
        # Force the index onto the same timestamps as the stock for a clean alignment test
        index_df.index = stock_df.index

        features = engineer.engineer_features(stock_df, index_df)
        basic_ok = features is not None and not features.empty
        print(
            f"Feature computation ran: {'OK' if basic_ok else 'FAILED'} — rows={0 if features is None else len(features)}"
        )

        expected_cols = [
            "rsi",
            "macd_line",
            "bb_upper",
            "atr",
            "vwap",
            "ema_fast",
            "orb_breakout",
            "gap_pct",
            "volume_ratio",
            "outperformance_pct",
            "nifty_correlation",
            "pct_from_ma",
            "pct_from_support_band",
            "pct_from_resistance_band",
            "pct_from_user_avg_cost",
            "has_position",
            "rsi_feat",
            "macd_line_feat",
            "vwap_feat",
            "pct_from_ma_feat",
            "pct_from_support_band_feat",
            "pct_from_resistance_band_feat",
            "pct_from_user_avg_cost_feat",
            "has_position_feat",
        ]
        assert features is not None
        missing_cols = [c for c in expected_cols if c not in features.columns]
        print(
            f"All expected columns present: {len(missing_cols) == 0}"
            + (f" (missing: {missing_cols})" if missing_cols else "")
        )

        # --- Critical test: NO LOOKAHEAD BIAS ---
        # Mutate only the LAST bar's Close/Volume drastically, recompute, and confirm
        # every '_feat' (ML-safe) column is UNCHANGED for all earlier rows. If any
        # earlier row's feature value changes because of a future bar's data,
        # that is a real lookahead bug.
        mutated_df = stock_df.copy()
        mutated_df.iloc[-1, mutated_df.columns.get_loc("Close")] += MUTATION_TEST_PRICE_DELTA
        mutated_df.iloc[-1, mutated_df.columns.get_loc("Volume")] *= MUTATION_TEST_VOLUME_MULTIPLIER

        mutated_features = engineer.engineer_features(mutated_df, index_df)
        assert mutated_features is not None

        feat_cols = [c for c in features.columns if c.endswith(ML_SAFE_SUFFIX)]
        no_lookahead = True
        for col in feat_cols:
            original_vals = features[col].iloc[:-1]
            mutated_vals = mutated_features[col].iloc[:-1]
            if not original_vals.equals(mutated_vals):
                # allow NaN==NaN mismatches to be treated as equal
                both_nan = original_vals.isna() & mutated_vals.isna()
                diffs = (original_vals != mutated_vals) & (~both_nan)
                if diffs.any():
                    no_lookahead = False
                    print(f"  LOOKAHEAD VIOLATION in column '{col}' at {diffs.sum()} row(s)!")

        print(f"No-lookahead check (mutating last bar doesn't change earlier '_feat' rows): {no_lookahead}")

        # Sanity: RSI should be within [0, 100]
        rsi_in_range = features["rsi"].dropna().between(0, 100).all()
        print(f"RSI within valid [0,100] range: {rsi_in_range}")

        # --- Test engineer_features_for_horizon ---
        # Generate enough data (at least 65 days) to test rolling returns
        stock_df_long = _build_synthetic_ohlcv(n_days=LONG_TEST_DAYS, bars_per_day=1, seed=42)
        index_df_long = _build_synthetic_ohlcv(n_days=LONG_TEST_DAYS, bars_per_day=1, seed=7)
        index_df_long.index = stock_df_long.index

        features_long = engineer.engineer_features_for_horizon(
            stock_df_long, index_df_long, horizon="30D"  # Any non-INTRADAY horizon
        )

        assert features_long is not None
        horizon_ok = not features_long.empty
        print(f"Multi-horizon feature computation ran: {'OK' if horizon_ok else 'FAILED'}")

        intraday_excluded = "vwap_feat" not in features_long.columns and "gap_pct_feat" not in features_long.columns
        print(f"Intraday features excluded for long horizon: {intraday_excluded}")

        rolling_present = "rolling_3m_return_feat" in features_long.columns
        print(f"Long-horizon momentum features present: {rolling_present}")

        overall_pass = (
            basic_ok
            and not missing_cols
            and no_lookahead
            and rsi_in_range
            and horizon_ok
            and intraday_excluded
            and rolling_present
        )
        print("STATUS: PASS" if overall_pass else "STATUS: FAIL — see details above")

        assert overall_pass, "One or more feature_engineer.py self-test checks failed"
        logger.info("feature_engineer.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"feature_engineer.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"feature_engineer.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
