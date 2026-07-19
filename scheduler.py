# 1. Standard library imports
import logging
import time as time_module
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

# 2. Third-party imports
import pandas as pd

# 3. Local imports
from config import (
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME,
    MARKET_TIMEZONE,
    SCHEDULER_INTERVAL_MINUTES,
    PREDICTION_HORIZON_BARS,
    PREDICTION_DEADBAND_PCT,
    BAR_INTERVAL,
    ALL_HORIZONS,
    HORIZON_INTRADAY,
    configure_logging,
)
from data_fetcher import DataFetcher
from predictor import Predictor, PredictionSignal, MultiHorizonSignal
from narrative_builder import NarrativeBuilder
from position_planner import PositionPlanner
from reference_level_engine import ReferenceLevelEngine
from event_classifier import EventClassifier
from history_manager import HistoryManager
from backtester import Backtester
from runtime_validator import EdgeCheckResult, CalibrationResult
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
def _interval_to_minutes(interval: str) -> int:
    """Small local helper — same parsing logic as feature_engineer.py's,
    duplicated rather than imported to avoid a circular/unnecessary dependency
    on a private helper for such a tiny piece of parsing."""
    import re
    match = re.match(r"^(\d+)([mh])$", interval.strip().lower())
    if not match:
        return 5
    value, unit = int(match.group(1)), match.group(2)
    return value * 60 if unit == "h" else value


BAR_INTERVAL_MINUTES = _interval_to_minutes(BAR_INTERVAL)


@dataclass
class LiveWorthinessSnapshot:
    edge_check_result: EdgeCheckResult
    calibration_result: CalibrationResult
    refreshed_at: datetime


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class Scheduler:
    """
    Orchestrates one full pipeline cycle: checks market hours, generates a
    signal per symbol (fetching fresh calibration data from real history),
    persists predictions and events, and resolves past predictions whose
    outcome horizon has elapsed. The expensive backtest-derived live/edge
    status is cached per symbol and only refreshed periodically, not on
    every single cycle.
    """

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        predictor: Optional[Predictor] = None,
        event_classifier: Optional[EventClassifier] = None,
        history_manager: Optional[HistoryManager] = None,
        backtester: Optional[Backtester] = None,
    ):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.predictor = predictor or Predictor(data_fetcher=self.data_fetcher)
        self.event_classifier = event_classifier or EventClassifier()
        self.history_manager = history_manager or HistoryManager()
        self.backtester = backtester or Backtester()
        self.reference_level_engine = ReferenceLevelEngine()
        self.position_planner = PositionPlanner()
        self.narrative_builder = NarrativeBuilder()
        self._live_worthiness_cache: Dict[str, LiveWorthinessSnapshot] = {}

    # -----------------------------------------------------------------
    # Market hours
    # -----------------------------------------------------------------
    @staticmethod
    def is_market_open(now: Optional[datetime] = None) -> bool:
        """Weekday (Mon-Fri) and within MARKET_OPEN_TIME-MARKET_CLOSE_TIME, IST.
        Does NOT yet account for exchange holidays — that would come from the
        macro calendar's festive/holiday entries in a future refinement."""
        try:
            tz = ZoneInfo(MARKET_TIMEZONE)
            now = now.astimezone(tz) if now is not None else datetime.now(tz)
            if now.weekday() >= 5:  # Saturday=5, Sunday=6
                return False
            current_time = now.time()
            return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME
        except Exception as e:
            logger.error(f"Failed checking market hours: {e}")
            return False  # fail safe: assume closed rather than risk acting when unsure

    # -----------------------------------------------------------------
    # Live-worthiness caching (backed by backtester, refreshed periodically)
    # -----------------------------------------------------------------
    def refresh_live_worthiness(self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame) -> LiveWorthinessSnapshot:
        backtest_result = self.backtester.run_backtest_for_symbol(symbol, stock_df, index_df)
        edge_result = EdgeCheckResult(
            status=backtest_result.edge_check_status, n_periods=backtest_result.n_test_predictions,
            strategy_cumulative_return_pct=backtest_result.strategy_cumulative_return_pct,
            baseline_cumulative_return_pct=backtest_result.baseline_cumulative_return_pct,
            alpha_pct=backtest_result.alpha_pct,
        )
        # Prefer REAL resolved history for calibration if enough exists; otherwise
        # fall back to the backtest's own calibration snapshot.
        real_calibration_df = self.history_manager.build_calibration_dataset(symbol)
        if len(real_calibration_df) >= self.predictor.runtime_validator.min_calibration_samples:
            calibration_result = self.predictor.runtime_validator.compute_calibration(real_calibration_df)
            logger.info(f"Using REAL resolved history for {symbol} calibration ({len(real_calibration_df)} samples).")
        else:
            calibration_result = CalibrationResult(
                status=backtest_result.calibration_status, n_samples=backtest_result.n_test_predictions,
                expected_calibration_error=backtest_result.calibration_ece,
                is_well_calibrated=(backtest_result.calibration_ece is not None
                                     and backtest_result.calibration_ece <= self.predictor.runtime_validator.ece_threshold),
                bins=[],
            )
            logger.info(f"Not enough real resolved history for {symbol} yet — using backtest-derived calibration snapshot.")

        snapshot = LiveWorthinessSnapshot(
            edge_check_result=edge_result, calibration_result=calibration_result, refreshed_at=datetime.now(),
        )
        self._live_worthiness_cache[symbol] = snapshot
        return snapshot

    def get_cached_live_worthiness(self, symbol: str) -> Optional[LiveWorthinessSnapshot]:
        return self._live_worthiness_cache.get(symbol)

    # -----------------------------------------------------------------
    # One cycle for one symbol
    # -----------------------------------------------------------------

    def run_cycle_stream_for_symbol(
        self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame,
        macro_events: Optional[List] = None, corporate_events: Optional[List[dict]] = None,
        news_articles: Optional[List[dict]] = None,
    ):
        try:
            snapshot = self.get_cached_live_worthiness(symbol)
            calibration_result = snapshot.calibration_result if snapshot else None
            edge_check_result = snapshot.edge_check_result if snapshot else None

            horizons_to_run = list(ALL_HORIZONS)

            stream = self.predictor.generate_multi_horizon_stream(
                symbol=symbol,
                horizons=horizons_to_run,
                stock_df=stock_df,
                index_df=index_df,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
                calibration_result=calibration_result,
                edge_check_result=edge_check_result
            )
            
            for sig in stream:
                # We yield each PredictionSignal as it finishes computing
                yield sig
                
        except Exception as e:
            logger.error(f"Stream cycle failed for {symbol}: {e}")
            
    def run_one_cycle_for_symbol(
        self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame,
        macro_events: Optional[List] = None, corporate_events: Optional[List[dict]] = None,
        news_articles: Optional[List[dict]] = None,
    ) -> Optional[MultiHorizonSignal]:
        try:
            snapshot = self.get_cached_live_worthiness(symbol)
            calibration_result = snapshot.calibration_result if snapshot else None
            edge_check_result = snapshot.edge_check_result if snapshot else None

            horizons_to_run = list(ALL_HORIZONS)

            # We need daily data for long horizons. Since run_one_cycle_for_symbol only takes stock_df, we rely on predictor to fetch it.
            multi_signal = self.predictor.generate_multi_horizon_signal(
                symbol=symbol,
                horizons=horizons_to_run,
                stock_df=stock_df,
                index_df=index_df,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
                calibration_result=calibration_result,
                edge_check_result=edge_check_result
            )

            narrative = self.narrative_builder.build_stock_narrative(multi_signal, None)
            cmp = stock_df["Close"].iloc[-1] if not stock_df.empty else 0.0
            levels = self.reference_level_engine.get_reference_levels(symbol, stock_df, multi_signal.primary_horizon)
            ma_lvl = levels.moving_average_value
            supp_lvl = levels.support_band_high
            plan = self.position_planner.generate_plan(
                multi_signal=multi_signal, current_price=cmp, 
                ma_level=ma_lvl, support_level=supp_lvl
            )
            
            dca_ladder = []
            if plan and plan.ladder:
                total_cap = plan.max_allowed_inr or 1.0
                for step in plan.ladder:
                    dca_ladder.append({
                        "type": step.reason,
                        "price": step.trigger_price,
                        "capital_allocation_pct": round((step.allocation_inr / total_cap) * 100, 1),
                        "shares": int(step.allocation_inr / step.trigger_price) if step.trigger_price else 0
                    })

            primary_sig = multi_signal.signals.get(multi_signal.primary_horizon)
            
            for hor, sig in multi_signal.signals.items():
                if hor == multi_signal.primary_horizon:
                    self.history_manager.save_prediction(sig, narrative=narrative, dca_ladder=dca_ladder)
                else:
                    self.history_manager.save_prediction(sig)

            if primary_sig and primary_sig.contributing_events:
                for event in primary_sig.contributing_events:
                    self.history_manager.save_event(event)

            return multi_signal

        except Exception as e:
            logger.error(f"Cycle failed for {symbol}: {e}")
            return None

    # -----------------------------------------------------------------
    # Outcome resolution — closes the loop that grows real calibration data
    # -----------------------------------------------------------------
    def resolve_pending_outcomes(self, symbol: str, stock_df: pd.DataFrame,
                                  horizon_bars: int = PREDICTION_HORIZON_BARS) -> int:
        """
        Finds unresolved predictions for `symbol` old enough that their
        outcome horizon has definitely elapsed, computes what actually
        happened from stock_df, and resolves them in history_manager.
        Returns the number of predictions resolved this call.
        """
        resolved_count = 0
        try:
            pending = self.history_manager.get_predictions(symbol=symbol, only_unresolved=True, limit=1000)
            if not pending or stock_df.empty:
                return 0

            horizon_minutes = horizon_bars * BAR_INTERVAL_MINUTES

            for record in pending:
                pred_time = pd.Timestamp(record.timestamp)
                if pred_time.tzinfo is not None and stock_df.index.tz is None:
                    pred_time = pred_time.tz_localize(None)
                elif pred_time.tzinfo is None and stock_df.index.tz is not None:
                    pred_time = pred_time.tz_localize(stock_df.index.tz)

                target_time = pred_time + pd.Timedelta(minutes=horizon_minutes)
                if stock_df.index.max() < target_time:
                    continue  # not enough time has passed yet to know the outcome

                # Find the closest available bar at/after prediction time, and at/after target time.
                bars_at_or_after_pred = stock_df.index[stock_df.index >= pred_time]
                bars_at_or_after_target = stock_df.index[stock_df.index >= target_time]
                if len(bars_at_or_after_pred) == 0 or len(bars_at_or_after_target) == 0:
                    continue

                start_close = stock_df.loc[bars_at_or_after_pred[0], "Close"]
                end_close = stock_df.loc[bars_at_or_after_target[0], "Close"]
                pct_move = (end_close - start_close) / start_close * 100.0

                if pct_move > PREDICTION_DEADBAND_PCT:
                    actual_class = "UP"
                elif pct_move < -PREDICTION_DEADBAND_PCT:
                    actual_class = "DOWN"
                else:
                    actual_class = "FLAT"

                if self.history_manager.resolve_outcome(record.id, actual_class):
                    resolved_count += 1

            if resolved_count:
                logger.info(f"Resolved {resolved_count} pending prediction(s) for {symbol}.")
            return resolved_count

        except Exception as e:
            logger.error(f"Failed resolving pending outcomes for {symbol}: {e}")
            return resolved_count

    # -----------------------------------------------------------------
    # Continuous loop (real deployment entry point)
    # -----------------------------------------------------------------
    def run_forever(self, symbol_data_provider, max_iterations: Optional[int] = None) -> None:
        """
        symbol_data_provider: a callable returning Dict[symbol -> (stock_df, index_df)]
        each time it's called, i.e. the actual live data refresh logic lives
        outside this file (in data_fetcher.py) and is injected here.
        max_iterations is provided purely for testability — production use
        leaves it as None and lets this run until the process is stopped.
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            health_registry.report("scheduler", ok=True, detail="Heartbeat")
            if not self.is_market_open():
                logger.info("Market closed — sleeping until next check.")
                time_module.sleep(60 if max_iterations is None else 0)
                iterations += 1
                continue

            try:
                symbol_data = symbol_data_provider()
                for symbol, (stock_df, index_df) in symbol_data.items():
                    self.resolve_pending_outcomes(symbol, stock_df)
                    self.run_one_cycle_for_symbol(symbol, stock_df, index_df)
            except Exception as e:
                logger.error(f"Error during scheduler cycle: {e}")
                health_registry.report("scheduler", ok=False, detail="Error during scheduler cycle", error=str(e))

            iterations += 1
            if max_iterations is None:
                time_module.sleep(SCHEDULER_INTERVAL_MINUTES * 60)


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import numpy as np
    from config import DB_DIR

    configure_logging(log_filename="scheduler_selftest.log")
    logger.info("Running scheduler.py self-test...")

    def _build_synthetic_ohlcv(n_days: int = 40, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows, timestamps = [], []
        price = 1000.0
        base_date = pd.Timestamp("2026-01-05 09:15:00")
        recent_closes = []
        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    trend = recent_closes[-1] - recent_closes[-10]
                    bias = 1.5 if trend < -6 else (-1.5 if trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = rng.normal(bias, 1.5)
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + rng.normal(bias * 0.5, 1.0))
                high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
                low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)
        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps))

    test_symbol = "SCHED_TEST"  # single test symbol allowed in the __main__ block only
    test_db_path = DB_DIR / "test_scheduler_selftest.sqlite3"
    if test_db_path.exists():
        os.remove(test_db_path)

    try:
        print("\n=== SCHEDULER SELF-TEST RESULT ===")

        # --- Test 1: is_market_open pure logic, no dependencies ---
        tz = ZoneInfo(MARKET_TIMEZONE)
        tuesday_10am = datetime(2026, 7, 14, 10, 0, tzinfo=tz)   # a Tuesday, well within market hours
        saturday_10am = datetime(2026, 7, 18, 10, 0, tzinfo=tz)  # a Saturday
        tuesday_8am = datetime(2026, 7, 14, 8, 0, tzinfo=tz)     # a Tuesday, before market open
        print(f"Tuesday 10am -> market open: {Scheduler.is_market_open(tuesday_10am)} (expect True)")
        print(f"Saturday 10am -> market open: {Scheduler.is_market_open(saturday_10am)} (expect False)")
        print(f"Tuesday 8am -> market open: {Scheduler.is_market_open(tuesday_8am)} (expect False)")
        assert Scheduler.is_market_open(tuesday_10am) is True
        assert Scheduler.is_market_open(saturday_10am) is False
        assert Scheduler.is_market_open(tuesday_8am) is False

        # --- Setup for cycle + resolution tests ---
        stock_df = _build_synthetic_ohlcv(seed=42)
        index_df = _build_synthetic_ohlcv(seed=99)
        index_df.index = stock_df.index
        # Shift timestamps so the last bar is "now" (bypasses the staleness check in this offline test)
        shift = pd.Timestamp.now() - stock_df.index[-1] - pd.Timedelta(minutes=1)
        stock_df.index = stock_df.index + shift
        index_df.index = index_df.index + shift

        history_manager = HistoryManager(db_path=test_db_path)
        scheduler = Scheduler(history_manager=history_manager)

        # --- Test 2: refresh_live_worthiness populates the cache ---
        snapshot = scheduler.refresh_live_worthiness(test_symbol, stock_df, index_df)
        cached = scheduler.get_cached_live_worthiness(test_symbol)
        print(f"Live-worthiness cached: {cached is not None}, edge_status={snapshot.edge_check_result.status}")
        assert cached is not None
        # --- Test 3: run_one_cycle_for_symbol produces AND persists a signal ---
        before_count = len(history_manager.get_predictions(test_symbol))
        signal = scheduler.run_one_cycle_for_symbol(test_symbol, stock_df, index_df, macro_events=[], corporate_events=[], news_articles=[])
        after_count = len(history_manager.get_predictions(test_symbol))
        print(f"Signal generated: action={signal.action if signal else None}")
        print(f"Predictions persisted: before={before_count}, after={after_count}")
        assert signal is not None
        assert after_count == before_count + 1

        # --- Test 4: resolve_pending_outcomes resolves a prediction whose horizon has elapsed ---
        old_signal = PredictionSignal(
            symbol=test_symbol, timestamp=stock_df.index[100], action="BUY", model_predicted_class="UP",
            raw_confidence=0.7, risk_adjusted_confidence=0.7, calibrated_confidence=None, agreement_fraction=0.6,
            downside_summary="d", upside_summary="u", reasoning=[],
        )
        old_pred_id = history_manager.save_prediction(old_signal)
        resolved_count = scheduler.resolve_pending_outcomes(test_symbol, stock_df)
        resolved_record = [r for r in history_manager.get_predictions(test_symbol) if r.id == old_pred_id][0]
        print(f"Old prediction resolved: {resolved_record.outcome_resolved}, actual_class={resolved_record.outcome_actual_class}")
        assert resolved_count >= 1
        assert resolved_record.outcome_resolved is True
        assert resolved_record.outcome_actual_class in ("UP", "DOWN", "FLAT")

        print("STATUS: PASS")
        logger.info("scheduler.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"scheduler.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"scheduler.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
    finally:
        if test_db_path.exists():
            os.remove(test_db_path)
# 1. Standard library imports
import logging
import time as time_module
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

# 2. Third-party imports
import pandas as pd

# 3. Local imports
from config import (
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME,
    MARKET_TIMEZONE,
    SCHEDULER_INTERVAL_MINUTES,
    PREDICTION_HORIZON_BARS,
    PREDICTION_DEADBAND_PCT,
    BAR_INTERVAL,
    configure_logging,
)
from data_fetcher import DataFetcher
from predictor import Predictor, PredictionSignal
from event_classifier import EventClassifier
from history_manager import HistoryManager
from backtester import Backtester
from runtime_validator import EdgeCheckResult, CalibrationResult

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
def _interval_to_minutes(interval: str) -> int:
    """Small local helper — same parsing logic as feature_engineer.py's,
    duplicated rather than imported to avoid a circular/unnecessary dependency
    on a private helper for such a tiny piece of parsing."""
    import re
    match = re.match(r"^(\d+)([mh])$", interval.strip().lower())
    if not match:
        return 5
    value, unit = int(match.group(1)), match.group(2)
    return value * 60 if unit == "h" else value


BAR_INTERVAL_MINUTES = _interval_to_minutes(BAR_INTERVAL)


@dataclass
class LiveWorthinessSnapshot:
    edge_check_result: EdgeCheckResult
    calibration_result: CalibrationResult
    refreshed_at: datetime


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class Scheduler:
    """
    Orchestrates one full pipeline cycle: checks market hours, generates a
    signal per symbol (fetching fresh calibration data from real history),
    persists predictions and events, and resolves past predictions whose
    outcome horizon has elapsed. The expensive backtest-derived live/edge
    status is cached per symbol and only refreshed periodically, not on
    every single cycle.
    """

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        predictor: Optional[Predictor] = None,
        event_classifier: Optional[EventClassifier] = None,
        history_manager: Optional[HistoryManager] = None,
        backtester: Optional[Backtester] = None,
    ):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.predictor = predictor or Predictor(data_fetcher=self.data_fetcher)
        self.event_classifier = event_classifier or EventClassifier()
        self.history_manager = history_manager or HistoryManager()
        self.backtester = backtester or Backtester()
        self._live_worthiness_cache: Dict[str, LiveWorthinessSnapshot] = {}

    # -----------------------------------------------------------------
    # Market hours
    # -----------------------------------------------------------------
    @staticmethod
    def is_market_open(now: Optional[datetime] = None) -> bool:
        """
        Check if NSE equity market is open.
        
        Rules:
        - Must be a weekday (Mon-Fri)
        - Must be within MARKET_OPEN_TIME – MARKET_CLOSE_TIME (IST)
        - Must NOT be an official NSE trading holiday for 2026
        """
        try:
            tz = ZoneInfo(MARKET_TIMEZONE)
            now = now.astimezone(tz) if now is not None else datetime.now(tz)

            # Weekend check
            if now.weekday() >= 5:  # Saturday=5, Sunday=6
                return False

            # Official NSE Equity Holidays 2026
            nse_holidays_2026 = {
                "2026-01-15",  # Municipal Corporation Election - Maharashtra
                "2026-01-26",  # Republic Day
                "2026-03-03",  # Holi
                "2026-03-26",  # Shri Ram Navami
                "2026-03-31",  # Shri Mahavir Jayanti
                "2026-04-03",  # Good Friday
                "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
                "2026-05-01",  # Maharashtra Day
                "2026-05-28",  # Bakri Id
                "2026-06-26",  # Muharram
                "2026-09-14",  # Ganesh Chaturthi
                "2026-10-02",  # Mahatma Gandhi Jayanti
                "2026-10-20",  # Dussehra
                "2026-11-10",  # Diwali-Balipratipada
                "2026-11-24",  # Guru Nanak Jayanti
                "2026-12-25",  # Christmas
            }

            today_str = now.strftime("%Y-%m-%d")
            if today_str in nse_holidays_2026:
                return False

            # Time window check
            current_time = now.time()
            return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME

        except Exception as e:
            logger.error(f"Failed checking market hours: {e}")
            return False  # fail safe

    # -----------------------------------------------------------------
    # Live-worthiness caching (backed by backtester, refreshed periodically)
    # -----------------------------------------------------------------
    def refresh_live_worthiness(self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame) -> LiveWorthinessSnapshot:
        backtest_result = self.backtester.run_backtest_for_symbol(symbol, stock_df, index_df)
        edge_result = EdgeCheckResult(
            status=backtest_result.edge_check_status, n_periods=backtest_result.n_test_predictions,
            strategy_cumulative_return_pct=backtest_result.strategy_cumulative_return_pct,
            baseline_cumulative_return_pct=backtest_result.baseline_cumulative_return_pct,
            alpha_pct=backtest_result.alpha_pct,
        )
        # Prefer REAL resolved history for calibration if enough exists; otherwise
        # fall back to the backtest's own calibration snapshot.
        real_calibration_df = self.history_manager.build_calibration_dataset(symbol)
        if len(real_calibration_df) >= self.predictor.runtime_validator.min_calibration_samples:
            calibration_result = self.predictor.runtime_validator.compute_calibration(real_calibration_df)
            logger.info(f"Using REAL resolved history for {symbol} calibration ({len(real_calibration_df)} samples).")
        else:
            calibration_result = CalibrationResult(
                status=backtest_result.calibration_status, n_samples=backtest_result.n_test_predictions,
                expected_calibration_error=backtest_result.calibration_ece,
                is_well_calibrated=(backtest_result.calibration_ece is not None
                                     and backtest_result.calibration_ece <= self.predictor.runtime_validator.ece_threshold),
                bins=[],
            )
            logger.info(f"Not enough real resolved history for {symbol} yet — using backtest-derived calibration snapshot.")

        snapshot = LiveWorthinessSnapshot(
            edge_check_result=edge_result, calibration_result=calibration_result, refreshed_at=datetime.now(),
        )
        self._live_worthiness_cache[symbol] = snapshot
        return snapshot

    def get_cached_live_worthiness(self, symbol: str) -> Optional[LiveWorthinessSnapshot]:
        return self._live_worthiness_cache.get(symbol)

    # -----------------------------------------------------------------
    # One cycle for one symbol
    # -----------------------------------------------------------------
    def run_one_cycle_for_symbol(
        self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame,
        macro_events: Optional[List] = None, corporate_events: Optional[List[dict]] = None,
        news_articles: Optional[List[dict]] = None,
    ) -> Optional[PredictionSignal]:
        try:
            snapshot = self.get_cached_live_worthiness(symbol)
            calibration_result = snapshot.calibration_result if snapshot else None
            edge_check_result = snapshot.edge_check_result if snapshot else None

            signal = self.predictor.generate_signal(
                symbol, stock_df, index_df, macro_events=macro_events, corporate_events=corporate_events,
                news_articles=news_articles, calibration_result=calibration_result, edge_check_result=edge_check_result,
            )

            self.history_manager.save_prediction(signal)
            for event in signal.contributing_events:
                self.history_manager.save_event(event)

            return signal

        except Exception as e:
            logger.error(f"Cycle failed for {symbol}: {e}")
            return None

    # -----------------------------------------------------------------
    # Outcome resolution — closes the loop that grows real calibration data
    # -----------------------------------------------------------------
    def resolve_pending_outcomes(self, symbol: str, stock_df: pd.DataFrame,
                                  horizon_bars: int = PREDICTION_HORIZON_BARS) -> int:
        """
        Finds unresolved predictions for `symbol` old enough that their
        outcome horizon has definitely elapsed, computes what actually
        happened from stock_df, and resolves them in history_manager.
        Returns the number of predictions resolved this call.
        """
        resolved_count = 0
        try:
            pending = self.history_manager.get_predictions(symbol=symbol, only_unresolved=True, limit=1000)
            if not pending or stock_df.empty:
                return 0

            horizon_minutes = horizon_bars * BAR_INTERVAL_MINUTES

            for record in pending:
                pred_time = pd.Timestamp(record.timestamp)
                if pred_time.tzinfo is not None and stock_df.index.tz is None:
                    pred_time = pred_time.tz_localize(None)
                elif pred_time.tzinfo is None and stock_df.index.tz is not None:
                    pred_time = pred_time.tz_localize(stock_df.index.tz)

                target_time = pred_time + pd.Timedelta(minutes=horizon_minutes)
                if stock_df.index.max() < target_time:
                    continue  # not enough time has passed yet to know the outcome

                # Find the closest available bar at/after prediction time, and at/after target time.
                bars_at_or_after_pred = stock_df.index[stock_df.index >= pred_time]
                bars_at_or_after_target = stock_df.index[stock_df.index >= target_time]
                if len(bars_at_or_after_pred) == 0 or len(bars_at_or_after_target) == 0:
                    continue

                start_close = stock_df.loc[bars_at_or_after_pred[0], "Close"]
                end_close = stock_df.loc[bars_at_or_after_target[0], "Close"]
                pct_move = (end_close - start_close) / start_close * 100.0

                if pct_move > PREDICTION_DEADBAND_PCT:
                    actual_class = "UP"
                elif pct_move < -PREDICTION_DEADBAND_PCT:
                    actual_class = "DOWN"
                else:
                    actual_class = "FLAT"

                if self.history_manager.resolve_outcome(record.id, actual_class):
                    resolved_count += 1

            if resolved_count:
                logger.info(f"Resolved {resolved_count} pending prediction(s) for {symbol}.")
            return resolved_count

        except Exception as e:
            logger.error(f"Failed resolving pending outcomes for {symbol}: {e}")
            return resolved_count

    # -----------------------------------------------------------------
    # Continuous loop (real deployment entry point)
    # -----------------------------------------------------------------
    def run_forever(self, symbol_data_provider, max_iterations: Optional[int] = None) -> None:
        """
        symbol_data_provider: a callable returning Dict[symbol -> (stock_df, index_df)]
        each time it's called, i.e. the actual live data refresh logic lives
        outside this file (in data_fetcher.py) and is injected here.
        max_iterations is provided purely for testability — production use
        leaves it as None and lets this run until the process is stopped.
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            if not self.is_market_open():
                logger.info("Market closed — sleeping until next check.")
                time_module.sleep(60 if max_iterations is None else 0)
                iterations += 1
                continue

            try:
                symbol_data = symbol_data_provider()
                for symbol, (stock_df, index_df) in symbol_data.items():
                    self.resolve_pending_outcomes(symbol, stock_df)
                    self.run_one_cycle_for_symbol(symbol, stock_df, index_df)
            except Exception as e:
                logger.error(f"Error during scheduler cycle: {e}")

            iterations += 1
            if max_iterations is None:
                time_module.sleep(SCHEDULER_INTERVAL_MINUTES * 60)


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import numpy as np
    from config import DB_DIR

    configure_logging(log_filename="scheduler_selftest.log")
    logger.info("Running scheduler.py self-test...")

    def _build_synthetic_ohlcv(n_days: int = 40, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows, timestamps = [], []
        price = 1000.0
        base_date = pd.Timestamp("2026-01-05 09:15:00")
        recent_closes = []
        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    trend = recent_closes[-1] - recent_closes[-10]
                    bias = 1.5 if trend < -6 else (-1.5 if trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = rng.normal(bias, 1.5)
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + rng.normal(bias * 0.5, 1.0))
                high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
                low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)
        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps))

    test_symbol = "SCHED_TEST"  # single test symbol allowed in the __main__ block only
    test_db_path = DB_DIR / "test_scheduler_selftest.sqlite3"
    if test_db_path.exists():
        os.remove(test_db_path)

    try:
        print("\n=== SCHEDULER SELF-TEST RESULT ===")

        # --- Test 1: is_market_open pure logic, no dependencies ---
        tz = ZoneInfo(MARKET_TIMEZONE)
        tuesday_10am = datetime(2026, 7, 14, 10, 0, tzinfo=tz)   # a Tuesday, well within market hours
        saturday_10am = datetime(2026, 7, 18, 10, 0, tzinfo=tz)  # a Saturday
        tuesday_8am = datetime(2026, 7, 14, 8, 0, tzinfo=tz)     # a Tuesday, before market open
        print(f"Tuesday 10am -> market open: {Scheduler.is_market_open(tuesday_10am)} (expect True)")
        print(f"Saturday 10am -> market open: {Scheduler.is_market_open(saturday_10am)} (expect False)")
        print(f"Tuesday 8am -> market open: {Scheduler.is_market_open(tuesday_8am)} (expect False)")
        assert Scheduler.is_market_open(tuesday_10am) is True
        assert Scheduler.is_market_open(saturday_10am) is False
        assert Scheduler.is_market_open(tuesday_8am) is False

        # --- Setup for cycle + resolution tests ---
        stock_df = _build_synthetic_ohlcv(seed=42)
        index_df = _build_synthetic_ohlcv(seed=99)
        index_df.index = stock_df.index
        # Shift timestamps so the last bar is "now" (bypasses the staleness check in this offline test)
        shift = pd.Timestamp.now() - stock_df.index[-1] - pd.Timedelta(minutes=1)
        stock_df.index = stock_df.index + shift
        index_df.index = index_df.index + shift

        history_manager = HistoryManager(db_path=test_db_path)
        scheduler = Scheduler(history_manager=history_manager)

        # --- Test 2: refresh_live_worthiness populates the cache ---
        snapshot = scheduler.refresh_live_worthiness(test_symbol, stock_df, index_df)
        cached = scheduler.get_cached_live_worthiness(test_symbol)
        print(f"Live-worthiness cached: {cached is not None}, edge_status={snapshot.edge_check_result.status}")
        assert cached is not None

        # --- Test 3: run_one_cycle_for_symbol produces AND persists a signal ---
        before_count = len(history_manager.get_predictions(test_symbol))
        signal = scheduler.run_one_cycle_for_symbol(test_symbol, stock_df, index_df, macro_events=[], corporate_events=[], news_articles=[])
        after_count = len(history_manager.get_predictions(test_symbol))
        print(f"Signal generated: action={signal.action if signal else None}")
        print(f"Predictions persisted: before={before_count}, after={after_count}")
        assert signal is not None
        assert after_count == before_count + 1

        # --- Test 4: resolve_pending_outcomes resolves a prediction whose horizon has elapsed ---
        old_signal = PredictionSignal(
            symbol=test_symbol, timestamp=stock_df.index[100], action="BUY", model_predicted_class="UP",
            raw_confidence=0.7, risk_adjusted_confidence=0.7, calibrated_confidence=None, agreement_fraction=0.6,
            downside_summary="d", upside_summary="u", reasoning=[],
        )
        old_pred_id = history_manager.save_prediction(old_signal)
        resolved_count = scheduler.resolve_pending_outcomes(test_symbol, stock_df)
        resolved_record = [r for r in history_manager.get_predictions(test_symbol) if r.id == old_pred_id][0]
        print(f"Old prediction resolved: {resolved_record.outcome_resolved}, actual_class={resolved_record.outcome_actual_class}")
        assert resolved_count >= 1
        assert resolved_record.outcome_resolved is True
        assert resolved_record.outcome_actual_class in ("UP", "DOWN", "FLAT")

        print("STATUS: PASS")
        logger.info("scheduler.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"scheduler.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"scheduler.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
    finally:
        if test_db_path.exists():
            os.remove(test_db_path)
