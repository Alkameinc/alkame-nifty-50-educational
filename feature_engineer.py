# 1. Standard library imports
import logging
import re
from typing import Optional, Tuple

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import (
    BAR_INTERVAL,
    HORIZON_INTRADAY,
    ORB_MINUTES,
    GAP_THRESHOLD_PCT,
    VOLUME_SPIKE_MULTIPLIER,
    VOLUME_SPIKE_LOOKBACK_BARS,
    RSI_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
    ATR_PERIOD,
    ATR_EXPANSION_MULTIPLIER,
    MA_FAST_PERIOD,
    MA_SLOW_PERIOD,
    LOW_LIQUIDITY_VOLUME_FLOOR,
    OUTPERFORMANCE_THRESHOLD_PCT,
    CORRELATION_LOOKBACK_BARS,
    CORRELATION_BREAKDOWN_THRESHOLD,
    configure_logging,
)
from reference_level_engine import ReferenceLevelDeltas

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
    return value * 60 if unit == "h" else value


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
            return rsi.fillna(50.0)  # neutral RSI where undefined (e.g. no losses yet)
        except Exception as e:
            logger.error(f"Failed computing RSI: {e}")
            return pd.Series(np.nan, index=close.index)

    @staticmethod
    def compute_macd(
        close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
    ) -> Tuple[pd.Series, pd.Series]:
        try:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            return ema_fast, ema_slow
        except Exception as e:
            logger.error(f"Failed computing moving averages: {e}")
            nan_series = pd.Series(np.nan, index=close.index)
            return nan_series, nan_series

    def compute_opening_range_breakout(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        For each calendar day, computes the opening-range high/low from the
        first `orb_bar_count` bars, then flags +1 (breakout up), -1 (breakout
        down), or 0 for every bar in that day based on where Close sits
        relative to that range.
        """
        try:
            or_high = pd.Series(np.nan, index=df.index)
            or_low = pd.Series(np.nan, index=df.index)
            breakout = pd.Series(0, index=df.index)

            for day, day_df in df.groupby(df.index.date):
                if len(day_df) < self.orb_bar_count:
                    continue
                opening_bars = day_df.iloc[: self.orb_bar_count]
                day_or_high = opening_bars["High"].max()
                day_or_low = opening_bars["Low"].min()

                or_high.loc[day_df.index] = day_or_high
                or_low.loc[day_df.index] = day_or_low

                day_breakout = np.where(
                    day_df["Close"] > day_or_high, 1,
                    np.where(day_df["Close"] < day_or_low, -1, 0),
                )
                breakout.loc[day_df.index] = day_breakout

            return or_high, or_low, breakout
        except Exception as e:
            logger.error(f"Failed computing opening range breakout: {e}")
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series, pd.Series(0, index=df.index)

    @staticmethod
    def compute_gap(df: pd.DataFrame, threshold_pct: float = GAP_THRESHOLD_PCT) -> Tuple[pd.Series, pd.Series]:
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
    ) -> Tuple[pd.Series, pd.Series]:
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
    ) -> Tuple[pd.Series, pd.Series]:
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
    ) -> Tuple[pd.Series, pd.Series]:
        try:
            aligned_stock, aligned_index = stock_close.align(index_close, join="inner")
            if len(aligned_stock) < lookback:
                logger.warning("Not enough overlapping data for correlation breakdown calc.")
                return pd.Series(dtype=float), pd.Series(dtype=bool)

            stock_returns = aligned_stock.pct_change()
            index_returns = aligned_index.pct_change()
            rolling_corr = stock_returns.rolling(window=lookback, min_periods=lookback).corr(index_returns)
            breakdown_flag = rolling_corr < threshold
            return rolling_corr, breakdown_flag.fillna(False)
        except Exception as e:
            logger.error(f"Failed computing correlation breakdown: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=bool)

    # -----------------------------------------------------------------
    # Master orchestration
    # -----------------------------------------------------------------
    def engineer_features(
        self, stock_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None,
        reference_deltas: Optional[ReferenceLevelDeltas] = None
    ) -> Optional[pd.DataFrame]:
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
            rolling_vol_avg = volume.shift(1).rolling(
                window=VOLUME_SPIKE_LOOKBACK_BARS, min_periods=VOLUME_SPIKE_LOOKBACK_BARS
            ).mean()
            out["low_liquidity"] = rolling_vol_avg < LOW_LIQUIDITY_VOLUME_FLOOR

            if index_df is not None and _validate_ohlcv(index_df, "index_df"):
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
                "rsi", "macd_line", "macd_signal", "macd_histogram", "bb_upper", "bb_middle", "bb_lower",
                "atr", "vwap", "ema_fast", "ema_slow", "orb_breakout", "gap_pct", "volume_ratio",
                "outperformance_pct", "nifty_correlation",
                "pct_from_ma", "pct_from_support_band", "pct_from_resistance_band",
                "pct_from_user_avg_cost", "has_position",
            ]
            for col in ml_candidate_cols:
                if col in out.columns:
                    out[f"{col}{ML_SAFE_SUFFIX}"] = out[col].shift(1)

            return out

        except Exception as e:
            logger.error(f"Failed engineering features: {e}")
            return None

    def engineer_features_for_horizon(
        self, stock_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None,
        reference_deltas: Optional[ReferenceLevelDeltas] = None,
        horizon: str = HORIZON_INTRADAY
    ) -> Optional[pd.DataFrame]:
        """
        Computes features tailored to a specific horizon. 
        For daily and above, it drops intraday noise features (orb, vwap, gap)
        and computes macro momentum features (1M, 3M rolling returns).
        """
        out = self.engineer_features(stock_df, index_df, reference_deltas)
        if out is None:
            return None
            
        if horizon == HORIZON_INTRADAY:
            return out
            
        # For non-intraday horizons (daily bars), exclude intraday-specific features
        intraday_cols = [
            "orb_breakout", "orb_breakout_feat",
            "orb_high", "orb_low",
            "vwap", "vwap_feat",
            "above_vwap", "vwap_cross_up", "vwap_cross_down",
            "gap_pct", "gap_pct_feat", "gap_event"
        ]
        out.drop(columns=[c for c in intraday_cols if c in out.columns], inplace=True)
        
        # Add long-horizon features
        # 1-month return (approx 21 trading days)
        out["rolling_1m_return"] = out["Close"].pct_change(periods=21) * 100.0
        # 3-month return (approx 63 trading days)
        out["rolling_3m_return"] = out["Close"].pct_change(periods=63) * 100.0
        
        # Make them ML-safe
        out[f"rolling_1m_return{ML_SAFE_SUFFIX}"] = out["rolling_1m_return"].shift(1)
        out[f"rolling_3m_return{ML_SAFE_SUFFIX}"] = out["rolling_3m_return"].shift(1)
        
        return out


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="feature_engineer_selftest.log")
    logger.info("Running feature_engineer.py self-test...")

    def _build_synthetic_ohlcv(n_days: int = 5, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
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
        print(f"Feature computation ran: {'OK' if basic_ok else 'FAILED'} — rows={0 if features is None else len(features)}")

        expected_cols = [
            "rsi", "macd_line", "bb_upper", "atr", "vwap", "ema_fast", "orb_breakout",
            "gap_pct", "volume_ratio", "outperformance_pct", "nifty_correlation",
            "pct_from_ma", "pct_from_support_band", "pct_from_resistance_band",
            "pct_from_user_avg_cost", "has_position",
            "rsi_feat", "macd_line_feat", "vwap_feat",
            "pct_from_ma_feat", "pct_from_support_band_feat", "pct_from_resistance_band_feat",
            "pct_from_user_avg_cost_feat", "has_position_feat"
        ]
        missing_cols = [c for c in expected_cols if c not in features.columns]
        print(f"All expected columns present: {len(missing_cols) == 0}" +
              (f" (missing: {missing_cols})" if missing_cols else ""))

        # --- Critical test: NO LOOKAHEAD BIAS ---
        # Mutate only the LAST bar's Close/Volume drastically, recompute, and confirm
        # every '_feat' (ML-safe) column is UNCHANGED for all earlier rows. If any
        # earlier row's feature value changes because of a future bar's data,
        # that is a real lookahead bug.
        mutated_df = stock_df.copy()
        mutated_df.iloc[-1, mutated_df.columns.get_loc("Close")] += 500.0
        mutated_df.iloc[-1, mutated_df.columns.get_loc("Volume")] *= 20

        mutated_features = engineer.engineer_features(mutated_df, index_df)

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
        stock_df_long = _build_synthetic_ohlcv(n_days=70, bars_per_day=1, seed=42)
        index_df_long = _build_synthetic_ohlcv(n_days=70, bars_per_day=1, seed=7)
        index_df_long.index = stock_df_long.index
        
        features_long = engineer.engineer_features_for_horizon(
            stock_df_long, index_df_long, horizon="30D"  # Any non-INTRADAY horizon
        )
        
        horizon_ok = features_long is not None and not features_long.empty
        print(f"Multi-horizon feature computation ran: {'OK' if horizon_ok else 'FAILED'}")
        
        intraday_excluded = "vwap_feat" not in features_long.columns and "gap_pct_feat" not in features_long.columns
        print(f"Intraday features excluded for long horizon: {intraday_excluded}")
        
        rolling_present = "rolling_3m_return_feat" in features_long.columns
        print(f"Long-horizon momentum features present: {rolling_present}")

        overall_pass = basic_ok and not missing_cols and no_lookahead and rsi_in_range and horizon_ok and intraday_excluded and rolling_present
        print("STATUS: PASS" if overall_pass else "STATUS: FAIL — see details above")

        assert overall_pass, "One or more feature_engineer.py self-test checks failed"
        logger.info("feature_engineer.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"feature_engineer.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"feature_engineer.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
