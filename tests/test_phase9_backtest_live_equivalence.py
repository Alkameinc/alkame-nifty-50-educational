# 1. Standard library imports
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from backtester import Backtester, BacktestResult
from config import (
    ALL_HORIZONS,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    HORIZON_3D,
    HORIZON_30D,
)
from feature_engineer import (
    CANONICAL_FEATURE_CATALOG,
    FeatureEngineer,
    FeatureEquivalenceReport,
    verify_feature_equivalence,
)
from predictor import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    PredictionSignal,
    Predictor,
)
from runtime_validator import (
    CalibrationResult,
    EdgeCheckResult,
    STATUS_EDGE_CONFIRMED,
    STATUS_NO_EDGE,
    STATUS_SUFFICIENT,
)


def _build_synthetic_ohlcv(
    n_days: int = 15,
    bars_per_day: int = 75,
    base_price: float = 1000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Builds deterministic synthetic 5-min OHLCV bars across several days."""
    rng = np.random.default_rng(seed)
    rows = []
    timestamps = []
    price = base_price
    base_date = pd.Timestamp("2026-06-01 09:15:00", tz="UTC")

    for day in range(n_days):
        day_start = base_date + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=5 * bar)
            drift = rng.normal(0, 1.5)
            price = max(10.0, price + drift)
            open_p = price
            close_p = max(10.0, price + rng.normal(0, 1.0))
            high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
            low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
            vol = int(abs(rng.normal(50000, 15000)))
            rows.append([open_p, high_p, low_p, close_p, vol])
            timestamps.append(ts)
            price = close_p

    df = pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex(timestamps),
    )
    return df


class TestFeatureEquivalence(unittest.TestCase):
    """
    BACK-001: Prove backtest/live feature equivalence.
    Batch features == Sliced live features at timestamp T.
    """

    def setUp(self):
        self.engineer = FeatureEngineer()
        self.stock_df = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, base_price=2500.0, seed=101)
        self.index_df = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, base_price=24000.0, seed=202)
        self.index_df.index = self.stock_df.index

    def test_batch_vs_sliced_feature_values_identical_intraday(self):
        """Verify high-precision numerical parity between batch and sliced features."""
        report = verify_feature_equivalence(
            self.stock_df,
            self.index_df,
            horizon=HORIZON_INTRADAY,
            tolerance=1e-9,
            engineer=self.engineer,
        )
        self.assertTrue(report.is_equivalent, f"Mismatches found: {report.mismatches}")
        self.assertLess(report.max_absolute_error, 1e-9)
        self.assertEqual(len(report.mismatches), 0)
        self.assertGreater(len(report.timestamps_checked), 0)

    def test_feature_equivalence_across_all_horizons(self):
        """Verify feature equivalence across SWING and POSITIONAL horizons."""
        long_stock = _build_synthetic_ohlcv(n_days=40, bars_per_day=10, base_price=1500.0, seed=303)
        long_index = _build_synthetic_ohlcv(n_days=40, bars_per_day=10, base_price=23000.0, seed=404)
        long_index.index = long_stock.index

        for hor in [HORIZON_3D, HORIZON_30D]:
            report = verify_feature_equivalence(
                long_stock,
                long_index,
                horizon=hor,
                tolerance=1e-9,
                engineer=self.engineer,
            )
            self.assertTrue(report.is_equivalent, f"Failed for {hor}: {report.mismatches}")
            self.assertLess(report.max_absolute_error, 1e-9)
            self.assertEqual(len(report.mismatches), 0)

    def test_schema_consistency_and_type_preservation(self):
        """Assert exact column schema and ordering match between batch and sliced extraction."""
        t_target = self.stock_df.index[100]
        batch_df = self.engineer.engineer_features_for_horizon(self.stock_df, self.index_df, horizon=HORIZON_INTRADAY)
        live_df = self.engineer.engineer_features_for_horizon(
            self.stock_df.loc[:t_target], self.index_df.loc[:t_target], horizon=HORIZON_INTRADAY
        )

        batch_cols = list(batch_df.columns)
        live_cols = list(live_df.columns)
        self.assertEqual(batch_cols, live_cols, "Column ordering or names differ between batch and live.")

        for col in batch_cols:
            self.assertEqual(
                batch_df[col].dtype,
                live_df[col].dtype,
                f"Data type mismatch in column '{col}': {batch_df[col].dtype} vs {live_df[col].dtype}",
            )

    def test_no_lookahead_leakage_across_sessions(self):
        """Ensure boundary timestamps across session opens have zero lookahead contamination."""
        day_boundary_timestamps = [
            ts for i, ts in enumerate(self.stock_df.index) if i > 0 and ts.date() != self.stock_df.index[i - 1].date()
        ]
        self.assertGreater(len(day_boundary_timestamps), 0)

        report = verify_feature_equivalence(
            self.stock_df,
            self.index_df,
            timestamps=day_boundary_timestamps,
            horizon=HORIZON_INTRADAY,
            tolerance=1e-9,
            engineer=self.engineer,
        )
        self.assertTrue(report.is_equivalent, f"Boundary leakage found: {report.mismatches}")
        self.assertLess(report.max_absolute_error, 1e-9)


class TestSignalEquivalence(unittest.TestCase):
    """
    BACK-002: Prove backtest/live signal equivalence.
    Replay signal == Live signal at timestamp T across all dimensions.
    """

    def setUp(self):
        self.stock_df = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, base_price=2500.0, seed=505)
        self.index_df = _build_synthetic_ohlcv(n_days=10, bars_per_day=75, base_price=24000.0, seed=606)
        self.index_df.index = self.stock_df.index
        self.backtester = Backtester()

    def test_signal_equivalence_across_quantitative_dimensions(self):
        """Assert identical action, confidence, and class between live and replay paths."""
        res = self.backtester.verify_signal_equivalence(
            symbol="RELIANCE",
            stock_df=self.stock_df,
            index_df=self.index_df,
            horizon=HORIZON_INTRADAY,
            tolerance=1e-6,
        )
        self.assertTrue(res["is_equivalent"], f"Signal mismatches: {res['mismatches']}")
        self.assertEqual(len(res["mismatches"]), 0)
        self.assertIn("timestamp", res["allowed_nondeterministic_differences"][0])

    def test_event_availability_point_in_time_filtering(self):
        """Events occurring strictly after T must be excluded from backtest signal at T."""
        t_eval = self.stock_df.index[120]

        # 3 events: one past, one current, one future
        past_evt = {
            "timestamp": (t_eval - timedelta(hours=2)).isoformat(),
            "headline": "Past earnings beat",
            "event_type": "EARNINGS",
            "sentiment_score": 0.8,
            "impact_horizon": "INTRADAY",
            "affected_tickers": ["RELIANCE"],
        }
        current_evt = {
            "timestamp": t_eval.isoformat(),
            "headline": "Order win announced now",
            "event_type": "CONTRACT",
            "sentiment_score": 0.5,
            "impact_horizon": "INTRADAY",
            "affected_tickers": ["RELIANCE"],
        }
        future_evt = {
            "timestamp": (t_eval + timedelta(hours=3)).isoformat(),
            "headline": "Future regulatory fine",
            "event_type": "REGULATORY",
            "sentiment_score": -0.9,
            "impact_horizon": "INTRADAY",
            "affected_tickers": ["RELIANCE"],
        }

        all_events = [past_evt, current_evt, future_evt]

        # Replay signal at t_eval
        sig = self.backtester.replay_point_in_time_signal(
            symbol="RELIANCE",
            timestamp=t_eval,
            stock_df=self.stock_df,
            index_df=self.index_df,
            corporate_events=all_events,
            horizon=HORIZON_INTRADAY,
        )

        contributing_headlines = [e.headline_or_label for e in sig.contributing_events]
        self.assertNotIn(
            "Future regulatory fine",
            contributing_headlines,
            "Future event leaked into point-in-time replay signal!",
        )

    def test_safety_gate_equivalence_forced_hold(self):
        """When calibration or edge check fails, both live and backtest paths must force HOLD."""
        t_eval = self.stock_df.index[150]

        # Calibration with excessive ECE (miscalibrated)
        miscal_res = CalibrationResult(
            status=STATUS_SUFFICIENT,
            n_samples=100,
            expected_calibration_error=0.25,
            is_well_calibrated=False,
            reasons=["Expected Calibration Error 0.25 exceeds threshold."],
        )
        edge_fail_res = EdgeCheckResult(
            status=STATUS_NO_EDGE,
            n_periods=100,
            strategy_cumulative_return_pct=0.5,
            baseline_cumulative_return_pct=2.0,
            alpha_pct=-1.5,
        )

        sig_replay = self.backtester.replay_point_in_time_signal(
            symbol="RELIANCE",
            timestamp=t_eval,
            stock_df=self.stock_df,
            index_df=self.index_df,
            calibration_result=miscal_res,
            edge_check_result=edge_fail_res,
            horizon=HORIZON_INTRADAY,
        )

        sig_live = self.backtester.predictor.generate_signal(
            symbol="RELIANCE",
            stock_df=self.stock_df.loc[:t_eval],
            index_df=self.index_df.loc[:t_eval],
            calibration_result=miscal_res,
            edge_check_result=edge_fail_res,
            horizon=HORIZON_INTRADAY,
        )

        # Invariant: Action must be HOLD in both paths
        self.assertEqual(sig_replay.action, ACTION_HOLD)
        self.assertEqual(sig_live.action, ACTION_HOLD)
        self.assertIsNone(sig_replay.calibrated_confidence)
        self.assertIsNone(sig_live.calibrated_confidence)

    def test_run_signal_replay_backtest_execution(self):
        """Verify that run_signal_replay_backtest executes end-to-end and returns BacktestResult."""
        res = self.backtester.run_signal_replay_backtest(
            symbol="RELIANCE",
            stock_df=self.stock_df,
            index_df=self.index_df,
            horizon=HORIZON_INTRADAY,
            test_fraction=0.2,
        )
        self.assertIsInstance(res, BacktestResult)
        self.assertTrue(res.success)
        self.assertGreater(res.n_test_predictions, 0)
        self.assertIn(res.edge_check_status, ["EDGE_CONFIRMED", "NO_EDGE"])
        self.assertIn(res.calibration_status, ["SUFFICIENT", "INSUFFICIENT_DATA"])
