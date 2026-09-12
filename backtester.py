# 1. Standard library imports
import logging
from dataclasses import dataclass, field

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import (
    ALL_HORIZONS,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    SLIPPAGE_BPS,
    TRANSACTION_COST_BPS,
    configure_logging,
)
from ensemble_manager import EnsembleManager
from execution_simulator import ExecutionSimulator
from history_manager import HistoryManager
from runtime_validator import RuntimeValidator

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
CLASS_TO_DIRECTION = {"UP": 1, "DOWN": -1, "FLAT": 0}


@dataclass
class BacktestResult:
    symbol: str
    horizon: str
    n_test_predictions: int
    n_trades_taken: int
    strategy_cumulative_return_pct: float
    baseline_cumulative_return_pct: float
    alpha_pct: float
    edge_check_status: str
    calibration_status: str
    calibration_ece: float | None
    is_live_worthy: bool
    gate_reasons: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    cost_sensitivity: dict[str, float] = field(default_factory=dict)
    walk_forward_folds: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class Backtester:
    """
    Walks the ensemble forward through its own held-out test period (the same
    chronological split ensemble_manager.py already enforces), simulates one
    trade per prediction, applies slippage/transaction costs ONLY to bars
    where a trade actually happened, and compares cumulative strategy return
    against simply holding the NIFTY index over the same window. Feeds both
    a real edge-check and a real calibration dataset into runtime_validator.py.
    """

    def __init__(
        self,
        ensemble_manager: EnsembleManager | None = None,
        runtime_validator: RuntimeValidator | None = None,
        history_manager: HistoryManager | None = None,
        execution_simulator: ExecutionSimulator | None = None,
    ):
        self.ensemble_manager = ensemble_manager or EnsembleManager()
        self.runtime_validator = runtime_validator or RuntimeValidator()
        self.history_manager = history_manager or HistoryManager()
        self.execution_simulator = execution_simulator or ExecutionSimulator()

    @staticmethod
    def _forward_return_pct(close: pd.Series, horizon_bars: int) -> pd.Series:
        return (close.shift(-horizon_bars) - close) / close * 100.0

    def run_backtest_for_symbol(
        self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame, horizon: str = HORIZON_INTRADAY
    ) -> BacktestResult:
        try:
            train_result = self.ensemble_manager.train_ensemble_for_symbol(symbol, stock_df, index_df, horizon=horizon)
            if not train_result.success:
                return BacktestResult(
                    symbol=symbol,
                    horizon=horizon,
                    n_test_predictions=0,
                    n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0,
                    baseline_cumulative_return_pct=0.0,
                    alpha_pct=0.0,
                    edge_check_status="NO_EDGE",
                    calibration_status="INSUFFICIENT_DATA",
                    calibration_ece=None,
                    is_live_worthy=False,
                    success=False,
                    error=f"Ensemble training failed: {train_result.error}",
                )

            prepared = self.ensemble_manager.model_trainer.prepare_dataset(stock_df, index_df, horizon=horizon)
            if prepared is None:
                return BacktestResult(
                    symbol=symbol,
                    horizon=horizon,
                    n_test_predictions=0,
                    n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0,
                    baseline_cumulative_return_pct=0.0,
                    alpha_pct=0.0,
                    edge_check_status="NO_EDGE",
                    calibration_status="INSUFFICIENT_DATA",
                    calibration_ece=None,
                    is_live_worthy=False,
                    success=False,
                    error="Dataset preparation failed for backtest.",
                )
            X, y, _ = prepared
            horizon_bars = HORIZON_CONFIG.get(horizon, {}).get("horizon_bars", 0)
            _, X_test, _, y_test = self.ensemble_manager.model_trainer.time_based_split(X, y, purge_window=horizon_bars)

            if len(X_test) == 0:
                return BacktestResult(
                    symbol=symbol,
                    horizon=horizon,
                    n_test_predictions=0,
                    n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0,
                    baseline_cumulative_return_pct=0.0,
                    alpha_pct=0.0,
                    edge_check_status="NO_EDGE",
                    calibration_status="INSUFFICIENT_DATA",
                    calibration_ece=None,
                    is_live_worthy=False,
                    success=False,
                    error="Test set is empty — nothing to backtest.",
                )

            ensemble_predictions = self.ensemble_manager.predict(symbol, X_test, horizon=horizon)
            if ensemble_predictions is None:
                return BacktestResult(
                    symbol=symbol,
                    horizon=horizon,
                    n_test_predictions=0,
                    n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0,
                    baseline_cumulative_return_pct=0.0,
                    alpha_pct=0.0,
                    edge_check_status="NO_EDGE",
                    calibration_status="INSUFFICIENT_DATA",
                    calibration_ece=None,
                    is_live_worthy=False,
                    success=False,
                    error="Ensemble prediction failed on test set.",
                )

            stock_forward_return = self._forward_return_pct(stock_df["Close"], horizon_bars).reindex(X_test.index)
            index_forward_return = self._forward_return_pct(index_df["Close"], horizon_bars).reindex(X_test.index)

            total_cost_pct = (SLIPPAGE_BPS + TRANSACTION_COST_BPS) / 100.0

            per_trade_net_returns = []
            n_trades_taken = 0
            calibration_rows = []
            signals_series = pd.Series(0, index=stock_df.index)

            for i, ts in enumerate(X_test.index):
                pred = ensemble_predictions[i]
                direction = CLASS_TO_DIRECTION.get(pred.predicted_class, 0)
                raw_fwd_return = stock_forward_return.loc[ts]

                if ts in signals_series.index:
                    signals_series.loc[ts] = direction

                if direction == 0 or pd.isna(raw_fwd_return):
                    net_return = 0.0
                else:
                    n_trades_taken += 1
                    gross_return = direction * raw_fwd_return
                    net_return = gross_return - total_cost_pct

                per_trade_net_returns.append(net_return)

                actual_class = y_test.iloc[i]
                calibration_rows.append(
                    {
                        "confidence": pred.confidence,
                        "correct": bool(pred.predicted_class == actual_class),
                    }
                )

            strategy_returns = pd.Series(per_trade_net_returns, index=X_test.index)

            edge_result = self.runtime_validator.compute_edge_vs_baseline(
                strategy_returns,
                index_forward_return.fillna(0.0),
                slippage_bps=0,
                transaction_cost_bps=0,
            )

            calibration_df = pd.DataFrame(calibration_rows)
            calibration_result = self.runtime_validator.compute_calibration(calibration_df)
            gate = self.runtime_validator.validate_before_live(calibration_result, edge_result)

            # Realistic execution simulation on test period
            sim_report = self.execution_simulator.simulate(symbol, stock_df, signals_series, horizon=horizon)
            sensitivity_reports = self.execution_simulator.run_sensitivity_analysis(
                symbol, stock_df, signals_series, horizon=horizon
            )
            cost_sensitivity = {name: rep.cumulative_net_return_pct for name, rep in sensitivity_reports.items()}

            if self.history_manager:
                self.history_manager.save_backtest_result(
                    symbol,
                    horizon,
                    edge_result.strategy_cumulative_return_pct,
                    edge_result.baseline_cumulative_return_pct,
                    edge_result.alpha_pct,
                    edge_result.status,
                    calibration_result.status,
                    calibration_result.expected_calibration_error,
                    (gate.safe_to_show_calibrated_confidence and gate.safe_to_treat_as_live_edge),
                )

            return BacktestResult(
                symbol=symbol,
                horizon=horizon,
                n_test_predictions=len(X_test),
                n_trades_taken=n_trades_taken,
                strategy_cumulative_return_pct=edge_result.strategy_cumulative_return_pct,
                baseline_cumulative_return_pct=edge_result.baseline_cumulative_return_pct,
                alpha_pct=edge_result.alpha_pct,
                edge_check_status=edge_result.status,
                calibration_status=calibration_result.status,
                calibration_ece=calibration_result.expected_calibration_error,
                is_live_worthy=(gate.safe_to_show_calibrated_confidence and gate.safe_to_treat_as_live_edge),
                gate_reasons=gate.reasons,
                success=True,
                max_drawdown_pct=sim_report.max_drawdown_pct,
                win_rate_pct=sim_report.win_rate_pct,
                profit_factor=sim_report.profit_factor,
                sharpe_ratio=sim_report.sharpe_ratio,
                cost_sensitivity=cost_sensitivity,
            )

        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            return BacktestResult(
                symbol=symbol,
                horizon=horizon,
                n_test_predictions=0,
                n_trades_taken=0,
                strategy_cumulative_return_pct=0.0,
                baseline_cumulative_return_pct=0.0,
                alpha_pct=0.0,
                edge_check_status="NO_EDGE",
                calibration_status="INSUFFICIENT_DATA",
                calibration_ece=None,
                is_live_worthy=False,
                success=False,
                error=str(e),
            )

    def run_walk_forward_backtest(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        horizon: str = HORIZON_INTRADAY,
        n_splits: int = 3,
        purge_window: int | None = None,
        embargo_window: int = 1,
        mode: str = "expanding",
    ) -> BacktestResult:
        """
        Executes multi-fold Purged Walk-Forward Backtesting (QNT-001, QNT-002, QNT-003).
        Enforces purge windows prior to test intervals to eliminate forward label peeking,
        and embargo windows to prevent autocorrelation leakage.
        Simulates realistic execution across all folds and produces cross-scenario sensitivity.
        """
        try:
            prepared = self.ensemble_manager.model_trainer.prepare_dataset(stock_df, index_df, horizon=horizon)
            if prepared is None:
                return BacktestResult(
                    symbol=symbol,
                    horizon=horizon,
                    n_test_predictions=0,
                    n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0,
                    baseline_cumulative_return_pct=0.0,
                    alpha_pct=0.0,
                    edge_check_status="NO_EDGE",
                    calibration_status="INSUFFICIENT_DATA",
                    calibration_ece=None,
                    is_live_worthy=False,
                    success=False,
                    error="Dataset preparation failed for walk-forward backtest.",
                )
            X, y, feature_columns = prepared
            horizon_bars = HORIZON_CONFIG.get(horizon, {}).get("horizon_bars", 5)
            p_win = purge_window if purge_window is not None else horizon_bars

            stock_forward_return = self._forward_return_pct(stock_df["Close"], horizon_bars)
            index_forward_return = self._forward_return_pct(index_df["Close"], horizon_bars)

            fold_records = []
            all_calibration_rows = []
            all_per_trade_returns = []
            all_index_returns = []
            total_predictions = 0
            total_trades = 0

            signals_series = pd.Series(0, index=stock_df.index)

            for fold_idx, X_train, X_test, y_train, y_test in self.ensemble_manager.model_trainer.walk_forward_split(
                X,
                y,
                n_splits=n_splits,
                purge_window=p_win,
                embargo_window=embargo_window,
                mode=mode,
            ):
                if len(X_test) == 0 or len(X_train) == 0:
                    continue

                # Train models for this fold
                models = {}
                for m_name in ["gradient_boosting", "random_forest", "logistic_regression"]:
                    try:
                        m_instance = self.ensemble_manager._create_model(m_name)
                        m_instance.fit(X_train, y_train)
                        models[m_name] = m_instance
                    except Exception as me:
                        logger.warning(f"Failed training fold model {m_name}: {me}")

                if not models:
                    continue

                # Predict on fold test set
                fold_signals = []
                fold_confs = []
                for idx in range(len(X_test)):
                    x_row = X_test.iloc[[idx]]
                    preds = [m.predict(x_row)[0] for m in models.values()]
                    # Plurality vote
                    vote = max(set(preds), key=preds.count)
                    agreement = preds.count(vote) / len(preds)
                    fold_signals.append(vote)
                    fold_confs.append(agreement)

                    ts = X_test.index[idx]
                    dir_val = CLASS_TO_DIRECTION.get(vote, 0)
                    if ts in signals_series.index:
                        signals_series.loc[ts] = dir_val

                total_cost_pct = (SLIPPAGE_BPS + TRANSACTION_COST_BPS) / 100.0
                fold_trade_returns = []
                fold_trades_count = 0

                for i, ts in enumerate(X_test.index):
                    dir_val = CLASS_TO_DIRECTION.get(fold_signals[i], 0)
                    raw_ret = stock_forward_return.loc[ts] if ts in stock_forward_return.index else np.nan
                    if dir_val != 0 and not pd.isna(raw_ret):
                        fold_trades_count += 1
                        net_ret = dir_val * raw_ret - total_cost_pct
                    else:
                        net_ret = 0.0

                    fold_trade_returns.append(net_ret)
                    all_per_trade_returns.append(net_ret)
                    if ts in index_forward_return.index:
                        all_index_returns.append(index_forward_return.loc[ts])
                    else:
                        all_index_returns.append(0.0)

                    actual_class = y_test.iloc[i]
                    all_calibration_rows.append(
                        {
                            "confidence": fold_confs[i],
                            "correct": bool(fold_signals[i] == actual_class),
                        }
                    )

                total_predictions += len(X_test)
                total_trades += fold_trades_count

                fold_cum_strat = float(np.sum(fold_trade_returns))
                idx_slice = index_forward_return.reindex(X_test.index).fillna(0.0)
                fold_cum_idx = float(np.sum(idx_slice))
                fold_records.append(
                    {
                        "fold": fold_idx,
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                        "trades": fold_trades_count,
                        "strategy_return_pct": round(fold_cum_strat, 2),
                        "baseline_return_pct": round(fold_cum_idx, 2),
                        "alpha_pct": round(fold_cum_strat - fold_cum_idx, 2),
                    }
                )

            if not fold_records:
                return BacktestResult(
                    symbol=symbol,
                    horizon=horizon,
                    n_test_predictions=0,
                    n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0,
                    baseline_cumulative_return_pct=0.0,
                    alpha_pct=0.0,
                    edge_check_status="NO_EDGE",
                    calibration_status="INSUFFICIENT_DATA",
                    calibration_ece=None,
                    is_live_worthy=False,
                    success=False,
                    error="No valid folds completed.",
                )

            strategy_series = pd.Series(all_per_trade_returns)
            baseline_series = pd.Series(all_index_returns).fillna(0.0)

            edge_result = self.runtime_validator.compute_edge_vs_baseline(
                strategy_series, baseline_series, slippage_bps=0, transaction_cost_bps=0
            )

            calibration_df = pd.DataFrame(all_calibration_rows)
            calibration_result = self.runtime_validator.compute_calibration(calibration_df)
            gate = self.runtime_validator.validate_before_live(calibration_result, edge_result)

            # Realistic execution simulation and multi-tier sensitivity
            sim_report = self.execution_simulator.simulate(symbol, stock_df, signals_series, horizon=horizon)
            sensitivity_reports = self.execution_simulator.run_sensitivity_analysis(
                symbol, stock_df, signals_series, horizon=horizon
            )
            cost_sensitivity = {name: rep.cumulative_net_return_pct for name, rep in sensitivity_reports.items()}

            return BacktestResult(
                symbol=symbol,
                horizon=horizon,
                n_test_predictions=total_predictions,
                n_trades_taken=total_trades,
                strategy_cumulative_return_pct=edge_result.strategy_cumulative_return_pct,
                baseline_cumulative_return_pct=edge_result.baseline_cumulative_return_pct,
                alpha_pct=edge_result.alpha_pct,
                edge_check_status=edge_result.status,
                calibration_status=calibration_result.status,
                calibration_ece=calibration_result.expected_calibration_error,
                is_live_worthy=(gate.safe_to_show_calibrated_confidence and gate.safe_to_treat_as_live_edge),
                gate_reasons=gate.reasons,
                success=True,
                max_drawdown_pct=sim_report.max_drawdown_pct,
                win_rate_pct=sim_report.win_rate_pct,
                profit_factor=sim_report.profit_factor,
                sharpe_ratio=sim_report.sharpe_ratio,
                cost_sensitivity=cost_sensitivity,
                walk_forward_folds=fold_records,
            )

        except Exception as e:
            logger.error(f"Walk-forward backtest failed for {symbol}: {e}")
            return BacktestResult(
                symbol=symbol,
                horizon=horizon,
                n_test_predictions=0,
                n_trades_taken=0,
                strategy_cumulative_return_pct=0.0,
                baseline_cumulative_return_pct=0.0,
                alpha_pct=0.0,
                edge_check_status="NO_EDGE",
                calibration_status="INSUFFICIENT_DATA",
                calibration_ece=None,
                is_live_worthy=False,
                success=False,
                error=str(e),
            )

    def run_backtest_for_all_symbols(
        self,
        symbol_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    ) -> dict[str, dict[str, BacktestResult]]:
        """symbol_data maps symbol -> (stock_df, index_df). Runs the full
        per-symbol backtest for each and returns all results keyed by symbol and horizon."""
        results: dict[str, dict[str, BacktestResult]] = {}
        for symbol, (stock_df, index_df) in symbol_data.items():
            results[symbol] = {}
            for horizon in ALL_HORIZONS:
                results[symbol][horizon] = self.run_backtest_for_symbol(symbol, stock_df, index_df, horizon=horizon)

        n_live_worthy = sum(1 for sym_res in results.values() for r in sym_res.values() if r.is_live_worthy)
        logger.info(
            f"Backtested {len(results)} symbol(s) across {len(ALL_HORIZONS)} horizons; {n_live_worthy} total strategies currently live-worthy."
        )
        return results


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="backtester_selftest.log")
    logger.info("Running backtester.py self-test...")

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
        return pd.DataFrame(
            rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps)
        )

    test_symbol = "BACKTEST_SYNTH"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== BACKTESTER SELF-TEST RESULT ===")
        backtester = Backtester()

        stock_df = _build_synthetic_ohlcv(seed=42)
        index_df = _build_synthetic_ohlcv(seed=99)
        index_df.index = stock_df.index

        result = backtester.run_backtest_for_symbol(test_symbol, stock_df, index_df)

        print(f"Success: {result.success}")
        if not result.success:
            print(f"Error: {result.error}")
        print(f"Test predictions: {result.n_test_predictions}, Trades actually taken: {result.n_trades_taken}")
        print(f"Strategy cumulative return: {result.strategy_cumulative_return_pct:.2f}%")
        print(f"Baseline (NIFTY buy-hold) cumulative return: {result.baseline_cumulative_return_pct:.2f}%")
        print(f"Alpha: {result.alpha_pct:.2f}%, edge_check_status={result.edge_check_status}")
        print(f"Calibration status: {result.calibration_status}, ECE: {result.calibration_ece}")
        print(f"Is live-worthy: {result.is_live_worthy}")
        print(f"Gate reasons: {result.gate_reasons}")

        assert result.success, "Backtest pipeline should complete successfully on valid synthetic data"
        assert result.n_test_predictions > 0
        assert result.edge_check_status in ("EDGE_CONFIRMED", "NO_EDGE")
        assert result.calibration_status in ("SUFFICIENT", "INSUFFICIENT_DATA")
        # With ~580 test rows from a 40-day synthetic set, calibration data should clearly be sufficient
        assert result.calibration_status == "SUFFICIENT", "Expected enough test rows for calibration to be sufficient"
        # is_live_worthy must be perfectly consistent with the two underlying statuses — never a third, independent answer
        expected_live_worthy = (
            result.calibration_status == "SUFFICIENT" and result.edge_check_status == "EDGE_CONFIRMED"
        )
        # Note: calibration ALSO requires is_well_calibrated, which calibration_status alone doesn't capture,
        # so we only assert the weaker necessary condition here rather than full equivalence.
        if result.is_live_worthy:
            assert result.edge_check_status == "EDGE_CONFIRMED"
            assert result.calibration_status == "SUFFICIENT"

        print("STATUS: PASS")
        logger.info("backtester.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"backtester.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"backtester.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
