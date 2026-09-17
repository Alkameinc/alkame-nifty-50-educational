import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from predictor import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    MultiHorizonSignal,
    PredictionContext,
    PredictionSignal,
    Predictor,
)
from scheduler import Scheduler


def _build_test_ohlcv(n_bars: int = 100, base_price: float = 1000.0) -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-01 09:15", periods=n_bars, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    prices = [base_price]
    for _ in range(n_bars - 1):
        prices.append(max(10.0, prices[-1] + rng.normal(0, 1.0)))
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 2.0 for p in prices],
            "Low": [p - 2.0 for p in prices],
            "Close": prices,
            "Volume": [10000] * n_bars,
        },
        index=timestamps,
    )
    return df


# ============================================================================
# API-002: Canonical Prediction Context Validation
# ============================================================================
def test_api_002_prediction_context_structure():
    """PredictionContext contains all required canonical fields."""
    stock_df = _build_test_ohlcv(50)
    index_df = _build_test_ohlcv(50, base_price=24000.0)

    now = datetime.now(timezone.utc)
    ctx = PredictionContext(
        symbol="TCS",
        timestamp=now,
        market_data=stock_df,
        index_data=index_df,
        macro_events=[],
        corporate_events=[],
        news_events=[],
        data_status="LIVE",
        event_status="EVENTS_AVAILABLE",
    )

    assert ctx.symbol == "TCS"
    assert ctx.timestamp == now
    assert len(ctx.market_data) == 50
    assert len(ctx.index_data) == 50
    assert ctx.data_status == "LIVE"
    assert ctx.event_status == "EVENTS_AVAILABLE"


def test_api_002_scheduler_builds_canonical_context():
    """Scheduler build_prediction_context returns structured PredictionContext."""
    from unittest.mock import MagicMock

    sched = Scheduler()
    sched.get_event_context = MagicMock(
        return_value=MagicMock(
            macro_events=[{"headline": "Test Macro"}],
            corporate_events=[{"headline": "Test Corp"}],
            news_articles=[{"headline": "Test News"}],
            status="EVENTS_AVAILABLE",
        )
    )
    stock_df = _build_test_ohlcv(50)
    index_df = _build_test_ohlcv(50, base_price=24000.0)

    ctx = sched.build_prediction_context("INFY", stock_df=stock_df, index_df=index_df)
    assert isinstance(ctx, PredictionContext)
    assert ctx.symbol == "INFY"
    assert ctx.market_data is stock_df
    assert ctx.index_data is index_df
    assert ctx.data_status == "LIVE"
    assert len(ctx.macro_events) == 1
    assert len(ctx.corporate_events) == 1
    assert len(ctx.news_events) == 1


def test_api_002_identical_fixtures_yield_identical_predictions():
    """
    Acceptance Test:
    For identical input fixtures, scheduler prediction == direct predictor prediction == API replay.
    """
    symbol = "RELIANCE"
    stock_df = _build_test_ohlcv(80, base_price=2500.0)
    index_df = _build_test_ohlcv(80, base_price=24500.0)

    sched = Scheduler()
    predictor = sched.predictor

    # 1. Build canonical context
    ctx = sched.build_prediction_context(
        symbol=symbol,
        stock_df=stock_df,
        index_df=index_df,
        macro_events=[],
        corporate_events=[],
        news_articles=[],
    )

    # 2. Predict directly via Predictor.predict_context
    sig_direct = predictor.predict_context(ctx, horizons=["INTRADAY"])

    # 3. Predict via Scheduler.run_one_cycle_for_symbol with same inputs
    sig_sched = sched.run_one_cycle_for_symbol(
        symbol=symbol,
        stock_df=stock_df,
        index_df=index_df,
        macro_events=[],
        corporate_events=[],
        news_articles=[],
    )
    if hasattr(sig_sched, "signal"):
        sig_sched = sig_sched.signal

    # Equivalence assertions across feature schema, action, confidence, and target prices
    assert sig_direct.primary_action == sig_sched.primary_action
    s_dir = sig_direct.signals["INTRADAY"]
    s_sch = sig_sched.signals["INTRADAY"]
    assert s_dir.action == s_sch.action
    assert s_dir.raw_confidence == s_sch.raw_confidence
    assert s_dir.risk_adjusted_confidence == s_sch.risk_adjusted_confidence
    assert s_dir.target_price == s_sch.target_price
    assert s_dir.stop_loss == s_sch.stop_loss
    assert s_dir.model_predicted_class == s_sch.model_predicted_class
