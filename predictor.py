"""
Educational Market Intelligence & Technical Posture Predictor.

In compliance with SEBI (Research Analysts) Regulations, 2014 and SEBI (Investment
Advisers) Regulations, 2013, Alkame-Nifty50 operates in strictly educational and
informational mode. Directional trading recommendations (BUY/SELL), price targets,
and stop losses are withheld.

For offline quantitative experimentation, machine learning model training harnesses
are isolated locally in _local_research/.
"""

import logging
from datetime import datetime, UTC
from typing import Any
import pandas as pd

from config import HORIZON_INTRADAY, NIFTY50_SYMBOLS, to_yfinance_ticker
from data_fetcher import DataFetcher
from event_classifier import EventClassifier
from global_risk_monitor import GlobalRiskMonitor
from health_monitor import registry as health_registry
from reference_level_engine import ReferenceLevelEngine
from prediction_types import (
    ACTION_BUY,
    ACTION_SELL,
    ACTION_HOLD,
    CLASS_TO_ACTION,
    PredictionSignal,
    MultiHorizonSignal,
    PredictionContext,
    CalibrationResult,
    EdgeCheckResult,
)

logger = logging.getLogger(__name__)


def _round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Rounds price to nearest market tick size (default 0.05 for NSE)."""
    if price is None:
        return None
    return float(round(round(price / tick_size) * tick_size, 2))


class RuntimeValidatorStub:
    min_calibration_samples: int = 10
    ece_threshold: float = 0.15

    def compute_calibration(self, df: pd.DataFrame) -> CalibrationResult:
        return CalibrationResult()


class Predictor:
    """
    Educational Market Posture & Reference Engine.
    Evaluates market data and surfaces neutral technical postures and reference
    levels without issuing directional trade recommendations.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher | None = None,
        event_classifier: EventClassifier | None = None,
        global_risk_monitor: GlobalRiskMonitor | None = None,
        reference_level_engine: ReferenceLevelEngine | None = None,
    ):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.event_classifier = event_classifier or EventClassifier()
        self.global_risk_monitor = global_risk_monitor or GlobalRiskMonitor(data_fetcher=self.data_fetcher)
        self.reference_level_engine = reference_level_engine or ReferenceLevelEngine()
        self.runtime_validator = RuntimeValidatorStub()

    def generate_single_signal(
        self,
        symbol: str,
        horizon: str = HORIZON_INTRADAY,
        stock_df: pd.DataFrame | None = None,
        index_df: pd.DataFrame | None = None,
        macro_events: list | None = None,
        corporate_events: list | None = None,
        news_articles: list | None = None,
        as_of: datetime | None = None,
    ) -> PredictionSignal:
        """Generates a neutral, educational market intelligence observation."""
        now = as_of or datetime.now(UTC)
        cmp = 1000.0
        if stock_df is not None and not stock_df.empty:
            cmp = float(stock_df["Close"].iloc[-1])

        # Technical reference levels
        pivot_data = self.reference_level_engine.compute_camarilla_levels(symbol)
        h3 = pivot_data.get("H3")
        l3 = pivot_data.get("L3")

        downside = (
            f"Camarilla L3 support floor at ₹{l3:.2f}. "
            f"Macro factors assessed against India VIX and crude oil baselines."
            if l3 else "Downside technical reference anchored at 20-day moving average."
        )
        upside = (
            f"Camarilla H3 resistance ceiling at ₹{h3:.2f}. "
            f"Momentum posture monitored without directional trade execution."
            if h3 else "Upside technical reference anchored at 20-day moving average."
        )

        return PredictionSignal(
            symbol=symbol,
            timestamp=now,
            horizon=horizon,
            action=ACTION_HOLD,
            model_predicted_class="FLAT",
            model_version="1.0-educational",
            feature_version="1.0-educational",
            raw_confidence=0.5,
            risk_adjusted_confidence=0.5,
            calibrated_confidence=None,
            agreement_fraction=1.0,
            downside_summary=downside,
            upside_summary=upside,
            reasoning=[
                "Descriptive technical reference: Directional trading recommendations withheld under SEBI guidelines.",
                f"Reference CMP: ₹{cmp:.2f}.",
            ],
            contributing_events=[],
            global_risk_level=self.global_risk_monitor.get_latest_reading().risk_level,
            risk_toggle_enabled=self.global_risk_monitor.get_toggle_state().get("enabled", False),
            is_safe_to_trade_live=False,
            suppressed=True,
            target_price=None,
            stop_loss=None,
        )

    def generate_multi_horizon_signal(
        self,
        symbol: str,
        stock_df: pd.DataFrame | None = None,
        index_df: pd.DataFrame | None = None,
        macro_events: list | None = None,
        corporate_events: list | None = None,
        news_articles: list | None = None,
        as_of: datetime | None = None,
        horizons: list[str] | None = None,
    ) -> MultiHorizonSignal:
        """Generates multi-timeframe educational observations across requested horizons."""
        horizons_to_run = horizons or ["INTRADAY", "NEXT_DAY", "SWING_3D", "POSITIONAL_7D"]
        signals_map = {}
        for hor in horizons_to_run:
            signals_map[hor] = self.generate_single_signal(
                symbol=symbol,
                horizon=hor,
                stock_df=stock_df,
                index_df=index_df,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
                as_of=as_of,
            )

        return MultiHorizonSignal(
            symbol=symbol,
            timestamp=as_of or datetime.now(UTC),
            signals=signals_map,
            primary_action=ACTION_HOLD,
            primary_horizon="INTRADAY",
            reasoning=[
                "Multi-timeframe technical posture compiled for educational analysis.",
                "All directional BUY/SELL calls and price targets withheld under SEBI regulations.",
            ],
        )

    def predict_context(
        self,
        context: PredictionContext,
        horizons: list[str] | None = None,
    ) -> MultiHorizonSignal:
        """API/Scheduler canonical entrance."""
        return self.generate_multi_horizon_signal(
            symbol=context.symbol,
            stock_df=context.market_data,
            index_df=context.index_data,
            macro_events=context.macro_events,
            corporate_events=context.corporate_events,
            news_articles=context.news_events,
            as_of=context.as_of or context.timestamp,
            horizons=horizons,
        )
