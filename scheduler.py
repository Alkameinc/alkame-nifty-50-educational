# 1. Standard library imports
import hashlib
import logging
import time as time_module
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Optional, cast
from zoneinfo import ZoneInfo

# 2. Third-party imports
import pandas as pd

from backtester import Backtester

# 3. Local imports
from config import (
    BAR_INTERVAL,
    MARKET_TIMEZONE,
    NSE_ANNOUNCEMENT_REFRESH_MINUTES,
    PREDICTION_DEADBAND_PCT,
    SCHEDULER_INTERVAL_MINUTES,
    configure_logging,
)
from corporate_events_fetcher import CorporateEventsFetcher
from data_fetcher import DataFetcher
from event_classifier import EventClassifier
from health_monitor import registry as health_registry
from history_manager import HistoryManager
from market_calendar import market_calendar
from model_trainer import ModelTrainer
from macro_calendar import MacroCalendar
from news_sentiment_fetcher import NewsSentimentFetcher
from model_lineage import compute_feature_schema_hash
from prediction_concurrency import (
    PredictionConcurrencyCoordinator,
    compute_prediction_key,
    normalize_timestamp_for_key,
)
from predictor import MultiHorizonSignal, PredictionContext, PredictionSignal, Predictor
from runtime_validator import CalibrationResult, EdgeCheckResult
from scheduler_circuit_breaker import (
    SCHEDULER_STATE_DEGRADED,
    SCHEDULER_STATE_FAILED,
    SCHEDULER_STATE_HALTED,
    SCHEDULER_STATE_HEALTHY,
    SCHEDULER_STATE_STARTING,
    SchedulerCircuitBreaker,
    SchedulerMetrics,
)

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
EVENT_CONTEXT_REFRESH_MINUTES = max(1, int(NSE_ANNOUNCEMENT_REFRESH_MINUTES))


@dataclass
class LiveWorthinessSnapshot:
    edge_check_result: EdgeCheckResult
    calibration_result: CalibrationResult
    refreshed_at: datetime


@dataclass
class EventContext:
    macro_events: list | None
    corporate_events: list[dict] | None
    news_articles: list[dict] | None

    macro_status: str
    corporate_status: str
    news_status: str

    status: str
    as_of: datetime
    refreshed_at: datetime
    errors: list[str]


@dataclass
class CycleResult:
    success: bool
    status: str  # "SUCCESS", "DATA_UNAVAILABLE", "PREDICTION_FAILED", "ERROR"
    symbol: str
    signal: Optional["MultiHorizonSignal"] = None
    error: str | None = None


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
        data_fetcher: DataFetcher | None = None,
        predictor: Predictor | None = None,
        event_classifier: EventClassifier | None = None,
        history_manager: HistoryManager | None = None,
        backtester: Backtester | None = None,
        corporate_events_fetcher: CorporateEventsFetcher | None = None,
        macro_calendar: MacroCalendar | None = None,
        news_sentiment_fetcher: NewsSentimentFetcher | None = None,
        circuit_breaker: SchedulerCircuitBreaker | None = None,
        concurrency_coordinator: PredictionConcurrencyCoordinator | None = None,
    ):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.predictor = predictor or Predictor(data_fetcher=self.data_fetcher)
        self.event_classifier = event_classifier or EventClassifier()
        self.history_manager = history_manager or HistoryManager()
        self.backtester = backtester or Backtester()

        self.corporate_events_fetcher = corporate_events_fetcher or CorporateEventsFetcher()
        self.macro_calendar = macro_calendar or MacroCalendar()
        self.news_sentiment_fetcher = news_sentiment_fetcher or NewsSentimentFetcher()
        self.circuit_breaker = circuit_breaker or SchedulerCircuitBreaker()
        self.concurrency_coordinator = (
            concurrency_coordinator or PredictionConcurrencyCoordinator(history_manager=self.history_manager)
        )

        self._live_worthiness_cache: dict[tuple[str, str], LiveWorthinessSnapshot] = {}
        self._event_context_cache: dict[str, EventContext] = {}

        # SCHED-003: Startup storage recovery & integrity audit
        self.storage_recovery_report = None
        try:
            if hasattr(self.history_manager, "run_recovery"):
                self.storage_recovery_report = self.history_manager.run_recovery()
            else:
                from storage_reliability import run_storage_recovery

                self.storage_recovery_report = run_storage_recovery(getattr(self.history_manager, "engine", None))
            logger.info(
                f"[SCHED-003] Storage recovery completed: WAL={self.storage_recovery_report.wal_checkpointed}, "
                f"integrity={self.storage_recovery_report.integrity_ok}, "
                f"predictions={self.storage_recovery_report.prediction_count} "
                f"(unresolved={self.storage_recovery_report.unresolved_prediction_count})"
            )
        except Exception as sr_err:
            logger.warning(f"[SCHED-003] Startup storage recovery warning: {sr_err}")

        # SCHED-001 / OPS-001: Register initial state in health registry
        try:
            health_registry.report(
                "scheduler",
                ok=True,
                detail=f"Scheduler initialized in {self.circuit_breaker.state} state.",
            )
        except Exception:
            pass

    def compute_cycle_key(
        self,
        symbol: str,
        stock_df: pd.DataFrame | None = None,
        timestamp: Any | None = None,
    ) -> str:
        """SCHED-002: Computes a deterministic idempotency key for this symbol's prediction cycle."""
        if timestamp is not None:
            bar_ts = timestamp
        elif stock_df is not None and not stock_df.empty:
            bar_ts = stock_df.index[-1]
        else:
            bar_ts = datetime.now(timezone.utc)

        model_ver = "v1.0"
        schema_hash = ""
        try:
            if hasattr(self.predictor, "ensemble_manager"):
                loaded = self.predictor.ensemble_manager.load_ensemble(symbol, horizon="INTRADAY")
                if loaded:
                    _, _, meta = loaded
                    model_ver = meta.get("model_version", "v1.0")
                    schema_hash = meta.get("feature_schema_hash", "")
        except Exception:
            pass

        if not schema_hash:
            if stock_df is not None and not stock_df.empty:
                feat_cols = [c for c in stock_df.columns if c.endswith("_feat")]
                schema_hash = compute_feature_schema_hash(feat_cols)
            else:
                schema_hash = compute_feature_schema_hash([])

        return compute_prediction_key(symbol, bar_ts, model_ver, schema_hash)

    # -----------------------------------------------------------------
    # Market hours
    # -----------------------------------------------------------------
    @staticmethod
    def is_market_open(now: datetime | None = None) -> bool:
        """Check if NSE equity market is open, delegating to the unified market_calendar."""
        from market_calendar import is_market_open as _is_open

        return _is_open(now)

    # -----------------------------------------------------------------
    # Live-worthiness caching (backed by backtester, refreshed periodically)
    # -----------------------------------------------------------------
    def refresh_live_worthiness(
        self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame, horizon: str = "INTRADAY"
    ) -> LiveWorthinessSnapshot:
        backtest_result = self.backtester.run_backtest_for_symbol(symbol, stock_df, index_df, horizon=horizon)
        edge_result = EdgeCheckResult(
            status=backtest_result.edge_check_status,
            n_periods=backtest_result.n_test_predictions,
            strategy_cumulative_return_pct=backtest_result.strategy_cumulative_return_pct,
            baseline_cumulative_return_pct=backtest_result.baseline_cumulative_return_pct,
            alpha_pct=backtest_result.alpha_pct,
        )

        # Get the current model version and feature version from the predictor's ensemble manager
        from ensemble_manager import EnsemblePrediction

        dummy_pred = EnsemblePrediction(
            predicted_class="FLAT", confidence=0.0, agreement_fraction=0.0, per_model_votes={}
        )
        current_model_ver = dummy_pred.model_version

        # Prefer REAL resolved history for calibration if enough exists; otherwise
        # fall back to the backtest's own calibration snapshot.
        real_calibration_df = self.history_manager.build_calibration_dataset(
            symbol, horizon=horizon, model_version=current_model_ver
        )
        if len(real_calibration_df) >= self.predictor.runtime_validator.min_calibration_samples:
            calibration_result = self.predictor.runtime_validator.compute_calibration(real_calibration_df)
            logger.info(
                f"Using REAL resolved history for {symbol} ({horizon}) calibration ({len(real_calibration_df)} samples)."
            )
        else:
            calibration_result = CalibrationResult(
                status=backtest_result.calibration_status,
                n_samples=backtest_result.n_test_predictions,
                expected_calibration_error=backtest_result.calibration_ece,
                is_well_calibrated=(
                    backtest_result.calibration_ece is not None
                    and backtest_result.calibration_ece <= self.predictor.runtime_validator.ece_threshold
                ),
                bins=[],
            )
            logger.info(
                f"Not enough real resolved history for {symbol} ({horizon}) yet — using backtest-derived calibration snapshot."
            )

        cache_key = (symbol, horizon)
        previous_snapshot = self._live_worthiness_cache.get(cache_key)

        # Do not replace usable evidence with a failed or unavailable refresh.
        # Keep the previous snapshot so a transient refresh failure cannot
        # silently destroy the last known valid calibration/edge evidence.
        refresh_usable = (
            backtest_result.success
            and calibration_result.status == "SUFFICIENT"
            and bool(calibration_result.bins)
            and edge_result.status == "EDGE_CONFIRMED"
        )
        if not refresh_usable and previous_snapshot is not None:
            logger.warning(
                f"Keeping previous live-worthiness snapshot for {symbol} ({horizon}); "
                f"refresh was unavailable or failed."
            )
            return previous_snapshot

        snapshot = LiveWorthinessSnapshot(
            edge_check_result=edge_result,
            calibration_result=calibration_result,
            refreshed_at=datetime.now(),
        )
        self._live_worthiness_cache[cache_key] = snapshot
        return snapshot

    def get_cached_live_worthiness(self, symbol: str, horizon: str = "INTRADAY") -> LiveWorthinessSnapshot | None:
        snapshot = self._live_worthiness_cache.get((symbol, horizon))
        if snapshot is None:
            return None

        from config import LIVE_WORTHINESS_REFRESH_HOURS

        now_val = pd.Timestamp.now()
        ref_at = snapshot.refreshed_at
        if hasattr(now_val, "to_pydatetime"):
            now_val = now_val.to_pydatetime()
        if ref_at.tzinfo is not None and now_val.tzinfo is None:
            now_val = now_val.replace(tzinfo=ref_at.tzinfo)
        elif ref_at.tzinfo is None and now_val.tzinfo is not None:
            now_val = now_val.replace(tzinfo=None)

        age = now_val - ref_at
        if age >= timedelta(hours=LIVE_WORTHINESS_REFRESH_HOURS):
            return None

        return snapshot

    # -----------------------------------------------------------------
    # Event context collection / filtering
    # -----------------------------------------------------------------
    @staticmethod
    def _coerce_event_datetime(value: object) -> datetime | None:
        if isinstance(value, pd.Timestamp):
            return cast(datetime, value.to_pydatetime())
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=ZoneInfo(MARKET_TIMEZONE))
        return value.astimezone(UTC)

    @classmethod
    def _filter_events_as_of(cls, context: EventContext, as_of: datetime) -> EventContext:
        as_of_utc = cls._as_utc(as_of)

        corporate_events: list[dict] | None = None
        if context.corporate_events is not None:
            corporate_events = []
            for event in context.corporate_events:
                raw = event.get("raw") or {}
                available_at = (
                    event.get("available_at")
                    or event.get("fetched_at")
                    or raw.get("available_at")
                    or raw.get("fetched_at")
                    or raw.get("published_at")
                )
                event_time = cls._coerce_event_datetime(available_at)
                if event_time is not None and cls._as_utc(event_time) <= as_of_utc:
                    corporate_events.append(event)

        news_articles: list[dict] | None = None
        if context.news_articles is not None:
            news_articles = []
            for article in context.news_articles:
                available_at = article.get("available_at") or article.get("published_at") or article.get("fetched_at")
                event_time = cls._coerce_event_datetime(available_at)
                if event_time is not None and cls._as_utc(event_time) <= as_of_utc:
                    news_articles.append(article)

        macro_events = context.macro_events

        statuses = [context.macro_status, context.corporate_status, context.news_status]
        filtered_statuses = [
            "NO_EVENTS" if status == "EVENTS_AVAILABLE" and not events else status
            for status, events in zip(
                statuses,
                [macro_events, corporate_events, news_articles],
                strict=False,
            )
        ]

        unavailable = sum(status == "EVENT_SOURCE_UNAVAILABLE" for status in filtered_statuses)
        if unavailable == 3:
            overall_status = "EVENT_SOURCE_UNAVAILABLE"
        elif unavailable > 0 or "EVENT_SOURCE_PARTIAL" in filtered_statuses:
            overall_status = "EVENT_SOURCE_PARTIAL"
        elif any(status == "EVENTS_AVAILABLE" for status in filtered_statuses):
            overall_status = "EVENTS_AVAILABLE"
        else:
            overall_status = "NO_EVENTS"

        return EventContext(
            macro_events=macro_events,
            corporate_events=corporate_events,
            news_articles=news_articles,
            macro_status=filtered_statuses[0],
            corporate_status=filtered_statuses[1],
            news_status=filtered_statuses[2],
            status=overall_status,
            as_of=as_of,
            refreshed_at=context.refreshed_at,
            errors=list(context.errors),
        )

    def get_event_context(
        self,
        symbol: str,
        as_of: datetime | None = None,
        force_refresh: bool = False,
    ) -> EventContext:
        """Collect event inputs once and expose explicit source availability state.

        ``[]`` means a source was successfully checked and had no eligible events;
        ``None`` means that source was unavailable.  Source timestamps are applied
        before the context is handed to Predictor so future events cannot leak into
        a signal generated at an earlier as-of time.
        """
        query_time = as_of or datetime.now()
        cached = self._event_context_cache.get(symbol)
        cache_fresh = (
            cached is not None
            and as_of is None
            and not force_refresh
            and datetime.now() - cached.refreshed_at < timedelta(minutes=EVENT_CONTEXT_REFRESH_MINUTES)
        )
        if cache_fresh and cached is not None:
            return self._filter_events_as_of(cached, query_time)

        errors: list[str] = []

        try:
            macro_events: list | None = self.macro_calendar.get_active_macro_events(query_time.date())
            macro_ok = getattr(self.macro_calendar, "_last_query_ok", True)
            if macro_ok is False:
                macro_events = None
                errors.append("Macro event source unavailable")
                macro_status = "EVENT_SOURCE_UNAVAILABLE"
            else:
                macro_status = "EVENTS_AVAILABLE" if macro_events else "NO_EVENTS"
        except Exception as e:
            macro_events = None
            macro_status = "EVENT_SOURCE_UNAVAILABLE"
            errors.append(f"Macro event source unavailable: {e}")

        try:
            corporate_events: list[dict] | None = self.corporate_events_fetcher.fetch_all_for_symbol(symbol)
            corporate_status = getattr(self.corporate_events_fetcher, "_last_fetch_status", None)
            if corporate_status is None:
                corporate_status = "EVENTS_AVAILABLE" if corporate_events else "NO_EVENTS"
            if corporate_status == "EVENT_SOURCE_UNAVAILABLE":
                corporate_events = None
            errors.extend(getattr(self.corporate_events_fetcher, "_last_fetch_errors", []))
        except Exception as e:
            corporate_events = None
            corporate_status = "EVENT_SOURCE_UNAVAILABLE"
            errors.append(f"Corporate event source unavailable: {e}")

        try:
            news_articles: list[dict] | None = self.news_sentiment_fetcher.get_news_for_symbol(symbol)
            news_status = getattr(self.news_sentiment_fetcher, "_last_fetch_status", None)
            if news_status is None:
                news_status = "EVENTS_AVAILABLE" if news_articles else "NO_EVENTS"
            if news_status in {"EVENT_SOURCE_UNAVAILABLE", "EVENT_SOURCE_PARTIAL"}:
                news_articles = None
            errors.extend(getattr(self.news_sentiment_fetcher, "_last_fetch_errors", []))
        except Exception as e:
            news_articles = None
            news_status = "EVENT_SOURCE_UNAVAILABLE"
            errors.append(f"News source unavailable: {e}")

        source_statuses = [macro_status, corporate_status, news_status]
        unavailable = sum(status == "EVENT_SOURCE_UNAVAILABLE" for status in source_statuses)
        if unavailable == 3:
            overall_status = "EVENT_SOURCE_UNAVAILABLE"
        elif unavailable > 0 or "EVENT_SOURCE_PARTIAL" in source_statuses:
            overall_status = "EVENT_SOURCE_PARTIAL"
        elif any(status == "EVENTS_AVAILABLE" for status in source_statuses):
            overall_status = "EVENTS_AVAILABLE"
        else:
            overall_status = "NO_EVENTS"

        effective_as_of = as_of or datetime.now()
        context = EventContext(
            macro_events=macro_events,
            corporate_events=corporate_events,
            news_articles=news_articles,
            macro_status=macro_status,
            corporate_status=corporate_status,
            news_status=news_status,
            status=overall_status,
            as_of=effective_as_of,
            refreshed_at=datetime.now(),
            errors=errors,
        )
        self._event_context_cache[symbol] = context
        return self._filter_events_as_of(context, effective_as_of)

    @staticmethod
    def _data_version(stock_df: pd.DataFrame, as_of: datetime) -> str:
        """Stable identity for the market-data snapshot used by a forecast."""
        if stock_df is None or stock_df.empty:
            return "UNKNOWN"
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in stock_df.columns]
        as_of_ts = pd.Timestamp(as_of)
        if hasattr(stock_df.index, "tz") and stock_df.index.tz is None and as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_localize(None)
        elif hasattr(stock_df.index, "tz") and stock_df.index.tz is not None and as_of_ts.tz is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        tail = stock_df.loc[stock_df.index <= as_of_ts, cols].tail(500)
        payload = tail.to_csv().encode("utf-8")
        return "data-" + hashlib.sha256(payload).hexdigest()[:24]

    # -----------------------------------------------------------------
    # Canonical Prediction Context Builder (API-002)
    # -----------------------------------------------------------------
    def build_prediction_context(
        self,
        symbol: str,
        stock_df: pd.DataFrame | None = None,
        index_df: pd.DataFrame | None = None,
        as_of: datetime | None = None,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
    ) -> PredictionContext:
        """API-002: Build canonical unified prediction context for symbol."""
        data_status = "DATA_UNAVAILABLE"
        if stock_df is None:
            from config import to_yfinance_ticker

            raw_stock = self.data_fetcher.fetch_ohlcv(to_yfinance_ticker(symbol), return_metadata=True)
            stock_df = getattr(raw_stock, "data", getattr(raw_stock, "df", raw_stock))
            if hasattr(raw_stock, "status") and hasattr(raw_stock.status, "value"):
                data_status = raw_stock.status.value
            elif stock_df is not None and not stock_df.empty:
                data_status = "LIVE"
        else:
            data_status = "LIVE" if not stock_df.empty else "DATA_UNAVAILABLE"

        if index_df is None:
            raw_index = self.data_fetcher.fetch_nifty_index()
            index_df = getattr(raw_index, "data", getattr(raw_index, "df", raw_index))

        # Collect event context if not explicitly passed
        event_status = "EVENTS_AVAILABLE"
        if macro_events is None and corporate_events is None and news_articles is None:
            try:
                event_context = self.get_event_context(symbol, as_of=as_of) if as_of is not None else self.get_event_context(symbol)
            except TypeError:
                event_context = self.get_event_context(symbol)
            macro_events = event_context.macro_events
            corporate_events = event_context.corporate_events
            news_articles = event_context.news_articles
            event_status = event_context.status

        from config import HORIZON_CONFIG

        horizons = list(HORIZON_CONFIG.keys())
        calib_results = {}
        edge_results = {}
        for h in horizons:
            snapshot = self.get_cached_live_worthiness(symbol, horizon=h)
            if snapshot:
                calib_results[h] = snapshot.calibration_result
                edge_results[h] = snapshot.edge_check_result

        ts = as_of or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if as_of is not None and as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        return PredictionContext(
            symbol=symbol,
            timestamp=ts,
            market_data=stock_df,
            index_data=index_df,
            macro_events=macro_events,
            corporate_events=corporate_events,
            news_events=news_articles,
            data_status=data_status,
            event_status=event_status,
            calibration_results=calib_results,
            edge_check_results=edge_results,
            as_of=as_of,
        )

    # -----------------------------------------------------------------
    # One cycle for one symbol
    # -----------------------------------------------------------------
    def run_one_cycle_for_symbol(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
        return_structured: bool = False,
    ) -> Optional["MultiHorizonSignal"] | CycleResult:
        if not self.circuit_breaker.can_execute():
            msg = (
                f"Scheduler cycle suppressed for {symbol}: circuit breaker is HALTED. "
                f"Reason: {self.circuit_breaker.metrics.trip_reason}"
            )
            logger.warning(msg)
            if return_structured:
                return CycleResult(
                    success=False,
                    status="CIRCUIT_BREAKER_HALTED",
                    symbol=symbol,
                    signal=None,
                    error=msg,
                )
            return None

        if stock_df is None or stock_df.empty or index_df is None or index_df.empty:
            logger.warning(f"Data unavailable for {symbol} in cycle run.")
            if return_structured:
                return CycleResult(
                    success=False,
                    status="DATA_UNAVAILABLE",
                    symbol=symbol,
                    signal=None,
                    error="Empty stock or index data",
                )
            return None

        cycle_key = self.compute_cycle_key(symbol, stock_df)

        def _execute_cycle() -> MultiHorizonSignal:
            # Auto-collect event context if omitted
            m_events = macro_events
            c_events = corporate_events
            n_articles = news_articles
            if m_events is None and c_events is None and n_articles is None:
                try:
                    event_context = self.get_event_context(symbol)
                    m_events = event_context.macro_events
                    c_events = event_context.corporate_events
                    n_articles = event_context.news_articles
                except Exception:
                    pass

            bar_ts = (
                stock_df.index[-1]
                if (stock_df is not None and not stock_df.empty and isinstance(stock_df.index[-1], (pd.Timestamp, datetime)))
                else None
            )

            # Build canonical prediction context
            context = self.build_prediction_context(
                symbol=symbol,
                stock_df=stock_df,
                index_df=index_df,
                as_of=bar_ts,
                macro_events=m_events,
                corporate_events=c_events,
                news_articles=n_articles,
            )

            from config import HORIZON_CONFIG

            horizons = list(HORIZON_CONFIG.keys())

            if hasattr(self.predictor, "predict_context"):
                multi_sig = self.predictor.predict_context(context, horizons=horizons)
            else:
                kwargs = {
                    "symbol": context.symbol,
                    "horizons": horizons,
                    "stock_df": context.market_data,
                    "index_df": context.index_data,
                    "macro_events": context.macro_events,
                    "corporate_events": context.corporate_events,
                    "news_articles": context.news_events,
                    "calibration_results": context.calibration_results,
                    "edge_check_results": context.edge_check_results,
                }
                import inspect
                sig_params = inspect.signature(self.predictor.generate_multi_horizon_signal).parameters
                if "as_of" in sig_params:
                    kwargs["as_of"] = context.as_of or context.timestamp
                multi_sig = self.predictor.generate_multi_horizon_signal(**kwargs)

            for h, sig in multi_sig.signals.items():
                data_version = self._data_version(stock_df, sig.timestamp)
                if hasattr(self.history_manager, "save_prediction_bundle"):
                    self.history_manager.save_prediction_bundle(
                        sig,
                        events=sig.contributing_events,
                        prediction_key=getattr(sig, "prediction_key", None),
                        data_version=data_version,
                        label_definition_version="direction-adaptive-deadband-v1",
                        entry_timestamp=str(sig.timestamp),
                        entry_price=getattr(sig, "entry_price", None),
                    )
                else:
                    self.history_manager.save_prediction(
                        sig,
                        data_version=data_version,
                        label_definition_version="direction-adaptive-deadband-v1",
                        entry_timestamp=str(sig.timestamp),
                        entry_price=getattr(sig, "entry_price", None),
                    )
                    for event in sig.contributing_events:
                        self.history_manager.save_event(event)

            return multi_sig

        try:
            multi_signal = self.concurrency_coordinator.execute_or_wait(cycle_key, _execute_cycle)
            if return_structured:
                return CycleResult(success=True, status="SUCCESS", symbol=symbol, signal=multi_signal)
            return multi_signal

        except Exception as e:
            logger.error(f"Cycle failed for {symbol}: {e}")
            if return_structured:
                return CycleResult(success=False, status="ERROR", symbol=symbol, signal=None, error=str(e))
            return None

    def run_cycle_stream_for_symbol(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
    ):
        try:
            context = self.build_prediction_context(
                symbol=symbol,
                stock_df=stock_df,
                index_df=index_df,
                as_of=None,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
            )

            from config import HORIZON_CONFIG

            horizons = list(HORIZON_CONFIG.keys())
            stream = self.predictor.predict_stream_context(context, horizons=horizons)
            yield from stream
        except Exception as e:
            logger.error(f"Stream cycle failed for {symbol}: {e}")
            yield PredictionSignal(
                symbol=symbol,
                timestamp=stock_df.index[-1] if not stock_df.empty else pd.Timestamp.now(),
                horizon="INTRADAY",
                action="HOLD",
                model_predicted_class="FLAT",
                model_version="v1.0",
                feature_version="v1.0",
                raw_confidence=0.0,
                risk_adjusted_confidence=0.0,
                calibrated_confidence=None,
                agreement_fraction=0.0,
                target_price=0.0,
                stop_loss=0.0,
                downside_summary=f"Stream error: {e}",
                upside_summary="",
                reasoning=[f"Stream failed: {e}"],
                contributing_events=[],
                global_risk_level="UNKNOWN",
                risk_toggle_enabled=False,
                is_safe_to_trade_live=False,
                data_stale=True,
                suppressed=True,
                suppression_reasons=[f"Stream failed: {e}"],
            )

    # -----------------------------------------------------------------
    # Outcome resolution — closes the loop that grows real calibration data
    # -----------------------------------------------------------------
    def resolve_pending_outcomes(self, symbol: str, stock_df: pd.DataFrame) -> int:
        """Resolve only when the recorded entry and N valid future observations exist.

        Resolution is observation-based, not wall-clock based: weekends, holidays and
        missing bars do not satisfy a horizon. Unresolved rows are processed oldest-first
        so a continuous stream of new predictions cannot starve old eligible rows.
        """
        resolved_count = 0
        try:
            if stock_df is None or stock_df.empty:
                return 0
            pending = self.history_manager.get_predictions(symbol=symbol, only_unresolved=True, limit=250)
            if not pending:
                return 0

            from config import HORIZON_CONFIG
            trainer = ModelTrainer()
            bars = stock_df.copy()
            bars = bars[~bars.index.duplicated(keep="last")].sort_index()
            if "Close" not in bars.columns:
                return 0

            def _aligned(ts: pd.Timestamp) -> pd.Timestamp:
                if ts.tzinfo is not None and bars.index.tz is None:
                    return ts.tz_localize(None)
                if ts.tzinfo is None and bars.index.tz is not None:
                    return ts.tz_localize(bars.index.tz)
                return ts

            def _valid_index_after(entry_ts: pd.Timestamp) -> list[pd.Timestamp]:
                result = []
                for ts in bars.index:
                    if ts <= entry_ts:
                        continue
                    ts_date = ts.date()
                    if market_calendar.is_trading_day(ts_date):
                        result.append(ts)
                return result

            for record in pending:
                h_config = HORIZON_CONFIG.get(record.horizon)
                if not h_config:
                    continue
                if record.entry_timestamp is None or record.entry_price is None:
                    # Legacy/malformed provenance is intentionally left unresolved.
                    continue

                try:
                    entry_ts = _aligned(pd.Timestamp(record.entry_timestamp))
                    entry_pos = bars.index.get_indexer([entry_ts])[0]
                except Exception:
                    entry_pos = -1
                if entry_pos < 0:
                    continue

                entry_close = float(bars.iloc[entry_pos]["Close"])
                if not pd.notna(entry_close):
                    continue
                if abs(entry_close - float(record.entry_price)) > max(1e-8, abs(entry_close) * 1e-6):
                    # Stored entry and supplied market history disagree: never grade a
                    # different question using a substituted close.
                    continue

                horizon_bars = int(h_config["horizon_bars"])
                future_index = _valid_index_after(entry_ts)
                if len(future_index) < horizon_bars:
                    continue
                target_ts = future_index[horizon_bars - 1]
                endpoint_close = float(bars.loc[target_ts, "Close"])
                if not pd.notna(endpoint_close):
                    continue

                # Use the same adaptive deadband definition as model training. The
                # threshold is evaluated at the recorded entry, so later volatility
                # cannot retroactively alter the grade.
                try:
                    thresholds = trainer.compute_adaptive_deadband(
                        bars, horizon_bars, float(h_config["deadband_pct_default"])
                    )
                    deadband = float(thresholds.loc[entry_ts])
                except Exception:
                    deadband = float(h_config["deadband_pct_default"])

                pct_move = (endpoint_close - entry_close) / entry_close * 100.0
                if pct_move > deadband:
                    actual_class = "UP"
                elif pct_move < -deadband:
                    actual_class = "DOWN"
                else:
                    actual_class = "FLAT"

                reason = (
                    f"{record.horizon}: entry={entry_ts.isoformat()} close={entry_close:.6f}; "
                    f"endpoint={target_ts.isoformat()} close={endpoint_close:.6f}; "
                    f"valid_future_bars={horizon_bars}; move={pct_move:.6f}%; deadband={deadband:.6f}%"
                )
                if self.history_manager.resolve_outcome(
                    record.id,
                    actual_class,
                    entry_price=entry_close,
                    endpoint_price=endpoint_close,
                    entry_timestamp=entry_ts.isoformat(),
                    target_timestamp=target_ts.isoformat(),
                    resolution_reason=reason,
                ):
                    resolved_count += 1

            if resolved_count:
                logger.info("Resolved %s pending prediction(s) for %s.", resolved_count, symbol)
            return resolved_count
        except Exception as e:
            logger.error("Failed resolving pending outcomes for %s: %s", symbol, e)
            return resolved_count

    # -----------------------------------------------------------------
    # Full Cycle Orchestration (SCHED-001, OPS-001)
    # -----------------------------------------------------------------
    def run_cycle(
        self,
        symbol_data_provider=None,
        symbols: list[str] | None = None,
        ignore_market_hours: bool = False,
    ) -> dict[str, Any]:
        """
        SCHED-001 & OPS-001: Orchestrates one full pipeline cycle with strict
        circuit breaker enforcement, operational timestamp tracking, and alertable
        health state reporting.
        """
        # 1. Circuit breaker gate
        if not self.circuit_breaker.can_execute():
            logger.warning(
                f"Skipping cycle execution: circuit breaker is HALTED ({self.circuit_breaker.metrics.trip_reason})"
            )
            return {
                "success": False,
                "status": "CIRCUIT_BREAKER_HALTED",
                "state": self.circuit_breaker.state,
                "error": self.circuit_breaker.metrics.trip_reason,
                "cycle_results": {},
            }

        # 2. Market hours check
        if not ignore_market_hours and not self.is_market_open():
            logger.info("Market closed — skipping cycle execution.")
            return {
                "success": True,
                "status": "MARKET_CLOSED",
                "state": self.circuit_breaker.state,
                "cycle_results": {},
            }

        try:
            # 3. Market data collection
            self.circuit_breaker.metrics.last_data_fetch = datetime.now(timezone.utc)
            if symbol_data_provider is not None:
                symbol_data = symbol_data_provider()
            else:
                from config import NIFTY50_SYMBOLS, to_yfinance_ticker

                target_symbols = symbols or NIFTY50_SYMBOLS
                symbol_data = {}
                index_raw = self.data_fetcher.fetch_nifty_index()
                index_df = index_raw if isinstance(index_raw, pd.DataFrame) else pd.DataFrame()
                for sym in target_symbols:
                    ticker = to_yfinance_ticker(sym)
                    stock_raw = self.data_fetcher.fetch_ohlcv(ticker)
                    stock_df = stock_raw if isinstance(stock_raw, pd.DataFrame) else pd.DataFrame()
                    symbol_data[sym] = (stock_df, index_df)

            if not symbol_data:
                raise ValueError("No market data returned by provider")

            # 4. Symbol processing
            cycle_results: dict[str, CycleResult] = {}
            for symbol, (stock_df, index_df) in symbol_data.items():
                self.resolve_pending_outcomes(symbol, stock_df)
                event_context = self.get_event_context(symbol)
                self.circuit_breaker.metrics.last_event_fetch = datetime.now(timezone.utc)

                res = self.run_one_cycle_for_symbol(
                    symbol,
                    stock_df,
                    index_df,
                    macro_events=event_context.macro_events,
                    corporate_events=event_context.corporate_events,
                    news_articles=event_context.news_articles,
                    return_structured=True,
                )
                if isinstance(res, CycleResult):
                    cycle_results[symbol] = res
                    if res.success:
                        self.circuit_breaker.metrics.last_prediction = datetime.now(timezone.utc)

            # Evaluate cycle success: at least one symbol succeeded and no circuit breaker halts
            successful_symbols = [s for s, r in cycle_results.items() if r.success]
            if not successful_symbols and cycle_results:
                raise RuntimeError(
                    f"All {len(cycle_results)} symbol cycle runs failed or had unavailable data."
                )

            self.circuit_breaker.record_success()
            try:
                health_registry.report(
                    "scheduler",
                    ok=True,
                    detail=f"Completed cycle for {len(successful_symbols)}/{len(cycle_results)} symbols. State: {self.circuit_breaker.state}",
                )
            except Exception:
                pass

            return {
                "success": True,
                "status": "SUCCESS",
                "state": self.circuit_breaker.state,
                "symbols_processed": len(cycle_results),
                "successful_symbols": len(successful_symbols),
                "cycle_results": cycle_results,
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Scheduler cycle execution failed: {error_msg}")
            self.circuit_breaker.record_failure(error_msg)
            try:
                health_registry.report(
                    "scheduler",
                    ok=(self.circuit_breaker.state in (SCHEDULER_STATE_HEALTHY, SCHEDULER_STATE_STARTING)),
                    detail=f"Scheduler cycle failed: {error_msg}. State: {self.circuit_breaker.state}",
                    error=error_msg,
                )
            except Exception:
                pass
            return {
                "success": False,
                "status": "FAILED",
                "state": self.circuit_breaker.state,
                "error": error_msg,
                "cycle_results": {},
            }

    def get_status(self) -> dict[str, Any]:
        """Returns the current state and operational metrics of the scheduler."""
        status = self.circuit_breaker.get_status()
        status["concurrency"] = self.concurrency_coordinator.get_metrics()
        if self.storage_recovery_report:
            from dataclasses import asdict

            status["storage"] = asdict(self.storage_recovery_report)
        else:
            status["storage"] = {"integrity_ok": True, "wal_checkpointed": True}
        return status

    def reset_circuit_breaker(self) -> dict[str, Any]:
        """Administratively resets the circuit breaker and clears failure counters."""
        self.circuit_breaker.reset()
        try:
            health_registry.report(
                "scheduler",
                ok=True,
                detail=f"Circuit breaker administratively reset. State: {self.circuit_breaker.state}",
            )
        except Exception:
            pass
        return self.get_status()

    # -----------------------------------------------------------------
    # Continuous loop (real deployment entry point)
    # -----------------------------------------------------------------
    def run_forever(self, symbol_data_provider, max_iterations: int | None = None) -> None:
        """
        symbol_data_provider: a callable returning Dict[symbol -> (stock_df, index_df)]
        each time it's called, i.e. the actual live data refresh logic lives
        outside this file (in data_fetcher.py) and is injected here.
        max_iterations is provided purely for testability — production use
        leaves it as None and lets this run until the process is stopped.
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            if not self.circuit_breaker.can_execute():
                logger.warning("Scheduler circuit breaker is HALTED — sleeping.")
                time_module.sleep(60 if max_iterations is None else 0)
                iterations += 1
                continue

            self.run_cycle(symbol_data_provider=symbol_data_provider)

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
        recent_closes: list[float] = []
        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    trend = recent_closes[-1] - recent_closes[-10]
                    bias = 1.5 if trend < -6 else (-1.5 if trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = float(rng.normal(bias, 1.5))
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + float(rng.normal(bias * 0.5, 1.0)))
                high_p = max(open_p, close_p) + abs(float(rng.normal(0, 0.5)))
                low_p = min(open_p, close_p) - abs(float(rng.normal(0, 0.5)))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)
        return pd.DataFrame(
            rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps)
        )

    test_symbol = "SCHED_TEST"  # single test symbol allowed in the __main__ block only
    test_db_path = DB_DIR / "test_scheduler_selftest.sqlite3"
    if test_db_path.exists():
        os.remove(test_db_path)

    try:
        print("\n=== SCHEDULER SELF-TEST RESULT ===")

        # --- Test 1: is_market_open pure logic, no dependencies ---
        tz = ZoneInfo(MARKET_TIMEZONE)
        tuesday_10am = datetime(2026, 7, 14, 10, 0, tzinfo=tz)  # a Tuesday, well within market hours
        saturday_10am = datetime(2026, 7, 18, 10, 0, tzinfo=tz)  # a Saturday
        tuesday_8am = datetime(2026, 7, 14, 8, 0, tzinfo=tz)  # a Tuesday, before market open
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
        signal = scheduler.run_one_cycle_for_symbol(
            test_symbol, stock_df, index_df, macro_events=[], corporate_events=[], news_articles=[]
        )
        after_count = len(history_manager.get_predictions(test_symbol))
        from predictor import MultiHorizonSignal

        action_str = (
            signal.signals["INTRADAY"].action
            if isinstance(signal, MultiHorizonSignal)
            and "INTRADAY" in signal.signals
            and signal.signals["INTRADAY"] is not None
            else None
        )
        print(f"Signal generated: action={action_str}")
        print(f"Predictions persisted: before={before_count}, after={after_count}")
        assert signal is not None
        assert after_count > before_count

        # --- Test 4: resolve_pending_outcomes resolves a prediction whose horizon has elapsed ---
        old_signal = PredictionSignal(
            symbol=test_symbol,
            timestamp=stock_df.index[100],
            horizon="INTRADAY",
            action="BUY",
            model_predicted_class="UP",
            model_version="UNKNOWN",
            feature_version="UNKNOWN",
            raw_confidence=0.7,
            risk_adjusted_confidence=0.7,
            calibrated_confidence=None,
            agreement_fraction=0.6,
            downside_summary="d",
            upside_summary="u",
            reasoning=[],
        )
        old_pred_id = history_manager.save_prediction(old_signal)
        resolved_count = scheduler.resolve_pending_outcomes(test_symbol, stock_df)
        resolved_record = [r for r in history_manager.get_predictions(test_symbol) if r.id == old_pred_id][0]
        print(
            f"Old prediction resolved: {resolved_record.outcome_resolved}, actual_class={resolved_record.outcome_actual_class}"
        )
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
