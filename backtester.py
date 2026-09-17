# 1. Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Any, cast

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
from runtime_validator import CalibrationResult, EdgeCheckResult, RuntimeValidator
from universe_provider import UniverseProvider, universe_provider as default_universe_provider

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
    n_universe_excluded_bars: int = 0
    enforce_universe: bool = False
    gross_cumulative_return_pct: float = 0.0
    total_transaction_costs_pct: float = 0.0
    total_slippage_pct: float = 0.0
    net_cumulative_return_pct: float = 0.0
    pnl_breakdown: dict[str, float] = field(default_factory=dict)
    stress_scenario_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def is_viable_after_friction(self) -> bool:
        """BACK-003: Strictly evaluates if net returns and alpha are positive after costs."""
        return self.net_cumulative_return_pct > 0.0 and self.alpha_pct > 0.0


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
        predictor: Any = None,
        universe_provider: UniverseProvider | None = None,
    ):
        self.ensemble_manager = ensemble_manager or EnsembleManager()
        self.runtime_validator = runtime_validator or RuntimeValidator()
        self.history_manager = history_manager or HistoryManager()
        self.execution_simulator = execution_simulator or ExecutionSimulator()
        self._predictor = predictor
        self.universe_provider = universe_provider or default_universe_provider

    @property
    def predictor(self):
        if self._predictor is None:
            from predictor import Predictor
            self._predictor = Predictor(
                ensemble_manager=self.ensemble_manager,
                runtime_validator=self.runtime_validator,
            )
        return self._predictor

    @staticmethod
    def _forward_return_pct(close: pd.Series, horizon_bars: int) -> pd.Series:
        return (close.shift(-horizon_bars) - close) / close * 100.0

    def run_backtest_for_symbol(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        horizon: str = HORIZON_INTRADAY,
        enforce_universe: bool = False,
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
                    enforce_universe=enforce_universe,
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
                    enforce_universe=enforce_universe,
                )

            X, y, _ = prepared
            horizon_bars = cast(int, HORIZON_CONFIG.get(horizon, {}).get("horizon_bars", 0))
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
                    enforce_universe=enforce_universe,
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
                    enforce_universe=enforce_universe,
                )

            stock_forward_return = self._forward_return_pct(stock_df["Close"], horizon_bars).reindex(X_test.index)
            index_forward_return = self._forward_return_pct(index_df["Close"], horizon_bars).reindex(X_test.index)

            total_cost_pct = (SLIPPAGE_BPS + TRANSACTION_COST_BPS) / 100.0

            per_trade_net_returns = []
            n_trades_taken = 0
            n_universe_excluded_bars = 0
            calibration_rows = []
            signals_series = pd.Series(0, index=stock_df.index)

            for i, ts in enumerate(X_test.index):
                pred = ensemble_predictions[i]
                raw_fwd_return = stock_forward_return.loc[ts]

                # BACK-004: Point-in-time universe membership verification
                is_eligible = True
                if enforce_universe:
                    is_eligible = self.universe_provider.is_member(symbol, ts)
                    if not is_eligible:
                        n_universe_excluded_bars += 1

                if is_eligible:
                    direction = CLASS_TO_DIRECTION.get(pred.predicted_class, 0)
                else:
                    direction = 0  # Prohibit trades when symbol was not an index member at timestamp ts

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

            # BACK-003: Mandatory Phase 12 Stress Test Scenarios
            stress_reports = self.execution_simulator.run_stress_test_analysis(
                symbol, stock_df, signals_series, horizon=horizon
            )
            stress_scenario_results = {}
            for s_name, s_rep in stress_reports.items():
                stress_scenario_results[s_name] = {
                    "gross_pnl_pct": s_rep.cumulative_gross_return_pct,
                    "transaction_costs_pct": s_rep.total_transaction_costs_paid_pct,
                    "slippage_pct": s_rep.total_slippage_paid_pct,
                    "total_costs_pct": s_rep.total_costs_paid_pct,
                    "net_pnl_pct": s_rep.cumulative_net_return_pct,
                    "win_rate_pct": s_rep.win_rate_pct,
                    "profit_factor": s_rep.profit_factor,
                    "max_drawdown_pct": s_rep.max_drawdown_pct,
                    "sharpe_ratio": s_rep.sharpe_ratio,
                    "is_viable": s_rep.is_viable,
                }

            # Top-level separated P&L metrics
            gross_pnl = sim_report.cumulative_gross_return_pct
            tx_costs = sim_report.total_transaction_costs_paid_pct
            slippage_pnl = sim_report.total_slippage_paid_pct
            net_pnl = sim_report.cumulative_net_return_pct

            pnl_breakdown = {
                "gross_pnl_pct": gross_pnl,
                "transaction_costs_pct": tx_costs,
                "slippage_pct": slippage_pnl,
                "net_pnl_pct": net_pnl,
            }

            # Viability Gate (BACK-003): Never report only gross performance when evaluating strategy viability
            is_live_worthy = bool(gate.safe_to_show_calibrated_confidence and gate.safe_to_treat_as_live_edge)
            gate_reasons = list(gate.reasons)
            if net_pnl <= 0.0 or sim_report.cumulative_net_return_pct <= 0.0:
                is_live_worthy = False
                gate_reasons.append(
                    "Strategy is non-viable after friction: net return is negative after slippage and transaction costs (gross-only profitability is non-viable)."
                )

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
                    is_live_worthy,
                )

            return BacktestResult(
                symbol=symbol,
                horizon=horizon,
                n_test_predictions=len(X_test),
                n_trades_taken=n_trades_taken,
                strategy_cumulative_return_pct=net_pnl,
                baseline_cumulative_return_pct=edge_result.baseline_cumulative_return_pct,
                alpha_pct=edge_result.alpha_pct,
                edge_check_status=edge_result.status,
                calibration_status=calibration_result.status,
                calibration_ece=calibration_result.expected_calibration_error,
                is_live_worthy=is_live_worthy,
                gate_reasons=gate_reasons,
                success=True,
                max_drawdown_pct=sim_report.max_drawdown_pct,
                win_rate_pct=sim_report.win_rate_pct,
                profit_factor=sim_report.profit_factor,
                sharpe_ratio=sim_report.sharpe_ratio,
                cost_sensitivity=cost_sensitivity,
                n_universe_excluded_bars=n_universe_excluded_bars,
                enforce_universe=enforce_universe,
                gross_cumulative_return_pct=gross_pnl,
                total_transaction_costs_pct=tx_costs,
                total_slippage_pct=slippage_pnl,
                net_cumulative_return_pct=net_pnl,
                pnl_breakdown=pnl_breakdown,
                stress_scenario_results=stress_scenario_results,
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
            horizon_bars = cast(int, HORIZON_CONFIG.get(horizon, {}).get("horizon_bars", 5))
            p_win = cast(int, purge_window if purge_window is not None else horizon_bars)

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
                base_estimators = self.ensemble_manager._build_base_estimators()
                for m_name in ["gradient_boosting", "random_forest", "logistic_regression"]:
                    try:
                        m_instance = base_estimators.get(m_name)
                        if m_instance is not None:
                            cast(Any, m_instance).fit(X_train, y_train)
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
                    preds = [cast(Any, m).predict(x_row)[0] for m in models.values()]
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

    def replay_point_in_time_signal(
        self,
        symbol: str,
        timestamp: Any,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
        calibration_result: CalibrationResult | None = None,
        edge_check_result: EdgeCheckResult | None = None,
        horizon: str = HORIZON_INTRADAY,
        enforce_universe: bool = False,
    ) -> Any:
        """
        BACK-002 & BACK-004: Replays the exact live signal generation logic at a historical timestamp T.
        Slices market data strictly up to T, filters events strictly up to T, and executes
        Predictor.generate_signal. If enforce_universe=True, verifies whether symbol was an active
        constituent of NIFTY 50 at timestamp T.
        """
        if timestamp not in stock_df.index:
            raise KeyError(f"Timestamp {timestamp} not found in stock_df index.")

        stock_slice = stock_df.loc[:timestamp]
        index_slice = index_df.loc[:timestamp] if index_df is not None and not index_df.empty else None

        # Point-in-time event filtering: exclude any event occurring strictly after timestamp
        ts_utc = pd.to_datetime(timestamp, utc=True)

        def _is_event_available(evt_dict: dict) -> bool:
            e_ts = evt_dict.get("timestamp") or evt_dict.get("date") or evt_dict.get("time")
            if e_ts is None:
                return True
            try:
                return pd.to_datetime(e_ts, utc=True) <= ts_utc
            except Exception:
                return True

        filtered_corp = [e for e in corporate_events if _is_event_available(e)] if corporate_events else None
        filtered_news = [e for e in news_articles if _is_event_available(e)] if news_articles else None
        filtered_macro = (
            [e for e in macro_events if (isinstance(e, dict) and _is_event_available(e))]
            if macro_events
            else macro_events
        )

        signal = self.predictor.generate_signal(
            symbol=symbol,
            stock_df=stock_slice,
            index_df=index_slice,
            macro_events=filtered_macro,
            corporate_events=filtered_corp,
            news_articles=filtered_news,
            calibration_result=calibration_result,
            edge_check_result=edge_check_result,
            horizon=horizon,
        )

        # BACK-004: Point-in-time universe check
        if enforce_universe and not self.universe_provider.is_member(symbol, timestamp):
            signal.action = "HOLD"
            signal.suppressed = True
            reason_str = f"SURVIVORSHIP_BIAS_GATE: Symbol {symbol} was not in NIFTY 50 universe at {timestamp}"
            if hasattr(signal, "suppression_reasons"):
                signal.suppression_reasons = list(signal.suppression_reasons or []) + [reason_str]
            if hasattr(signal, "reasoning"):
                signal.reasoning = list(signal.reasoning or []) + [reason_str]

        return signal

    def run_signal_replay_backtest(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
        horizon: str = HORIZON_INTRADAY,
        test_fraction: float = 0.2,
        enforce_universe: bool = False,
    ) -> BacktestResult:
        """
        BACK-002 & BACK-004: Executes a full-fidelity signal replay backtest where each bar in the
        test period receives a point-in-time signal from the complete live predictor pipeline
        (including risk adjustment, event sentiment, and runtime safety gates).
        """
        horizon_bars = cast(int, HORIZON_CONFIG.get(horizon, {}).get("horizon_bars", 1))
        split_idx = int(len(stock_df) * (1 - test_fraction))
        test_df = stock_df.iloc[split_idx:]
        if test_df.empty:
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
                error="Test set is empty",
                enforce_universe=enforce_universe,
            )

        stock_forward_return = self._forward_return_pct(stock_df["Close"], horizon_bars).reindex(test_df.index)
        index_forward_return = self._forward_return_pct(index_df["Close"], horizon_bars).reindex(test_df.index)
        total_cost_pct = (SLIPPAGE_BPS + TRANSACTION_COST_BPS) / 100.0

        per_trade_net_returns = []
        n_trades_taken = 0
        n_universe_excluded_bars = 0
        signals_series = pd.Series(0, index=stock_df.index)
        calibration_rows = []

        for ts in test_df.index:
            sig = self.replay_point_in_time_signal(
                symbol=symbol,
                timestamp=ts,
                stock_df=stock_df,
                index_df=index_df,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
                horizon=horizon,
                enforce_universe=enforce_universe,
            )

            if enforce_universe and not self.universe_provider.is_member(symbol, ts):
                n_universe_excluded_bars += 1

            direction = 1 if sig.action == "BUY" else (-1 if sig.action == "SELL" else 0)
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

            # Determine actual outcome class for calibration
            fwd_ret = stock_forward_return.loc[ts]
            actual_class = "FLAT"
            if not pd.isna(fwd_ret):
                deadband = HORIZON_CONFIG[horizon]["deadband_pct_default"]
                if fwd_ret > deadband:
                    actual_class = "UP"
                elif fwd_ret < -deadband:
                    actual_class = "DOWN"

            calibration_rows.append(
                {
                    "confidence": sig.risk_adjusted_confidence,
                    "correct": bool(sig.model_predicted_class == actual_class),
                }
            )

        strategy_returns = pd.Series(per_trade_net_returns, index=test_df.index)
        edge_result = self.runtime_validator.compute_edge_vs_baseline(
            strategy_returns,
            index_forward_return.fillna(0.0),
            slippage_bps=0,
            transaction_cost_bps=0,
        )

        cal_df = pd.DataFrame(calibration_rows)
        cal_res = self.runtime_validator.compute_calibration(cal_df)
        gate = self.runtime_validator.validate_before_live(cal_res, edge_result)

        sim_report = self.execution_simulator.simulate(symbol, stock_df, signals_series, horizon=horizon)
        sensitivity_reports = self.execution_simulator.run_sensitivity_analysis(
            symbol, stock_df, signals_series, horizon=horizon
        )
        cost_sensitivity = {name: rep.cumulative_net_return_pct for name, rep in sensitivity_reports.items()}
        stress_reports = self.execution_simulator.run_stress_test_analysis(
            symbol, stock_df, signals_series, horizon=horizon
        )
        stress_scenario_results = {
            s_name: {
                "gross_pnl_pct": s_rep.cumulative_gross_return_pct,
                "transaction_costs_pct": s_rep.total_transaction_costs_paid_pct,
                "slippage_pct": s_rep.total_slippage_paid_pct,
                "total_costs_pct": s_rep.total_costs_paid_pct,
                "net_pnl_pct": s_rep.cumulative_net_return_pct,
                "win_rate_pct": s_rep.win_rate_pct,
                "profit_factor": s_rep.profit_factor,
                "max_drawdown_pct": s_rep.max_drawdown_pct,
                "sharpe_ratio": s_rep.sharpe_ratio,
                "is_viable": s_rep.is_viable,
            }
            for s_name, s_rep in stress_reports.items()
        }

        gross_pnl = sim_report.cumulative_gross_return_pct
        tx_costs = sim_report.total_transaction_costs_paid_pct
        slippage_pnl = sim_report.total_slippage_paid_pct
        net_sim_pnl = sim_report.cumulative_net_return_pct
        pnl_breakdown = {
            "gross_pnl_pct": gross_pnl,
            "transaction_costs_pct": tx_costs,
            "slippage_pct": slippage_pnl,
            "net_pnl_pct": net_sim_pnl,
        }

        is_live_worthy = bool(gate.safe_to_show_calibrated_confidence and gate.safe_to_treat_as_live_edge)
        gate_reasons = list(gate.reasons)
        if is_live_worthy and (edge_result.strategy_cumulative_return_pct <= 0.0 or sim_report.cumulative_net_return_pct <= 0.0):
            is_live_worthy = False
            gate_reasons.append(
                "Strategy is non-viable after friction: net return is negative after slippage and transaction costs."
            )

        return BacktestResult(
            symbol=symbol,
            horizon=horizon,
            n_test_predictions=len(test_df),
            n_trades_taken=n_trades_taken,
            strategy_cumulative_return_pct=edge_result.strategy_cumulative_return_pct,
            baseline_cumulative_return_pct=edge_result.baseline_cumulative_return_pct,
            alpha_pct=edge_result.alpha_pct,
            edge_check_status=edge_result.status,
            calibration_status=cal_res.status,
            calibration_ece=cal_res.expected_calibration_error,
            is_live_worthy=is_live_worthy,
            gate_reasons=gate_reasons,
            success=True,
            max_drawdown_pct=sim_report.max_drawdown_pct,
            win_rate_pct=sim_report.win_rate_pct,
            profit_factor=sim_report.profit_factor,
            sharpe_ratio=sim_report.sharpe_ratio,
            cost_sensitivity=cost_sensitivity,
            n_universe_excluded_bars=n_universe_excluded_bars,
            enforce_universe=enforce_universe,
            gross_cumulative_return_pct=gross_pnl,
            total_transaction_costs_pct=tx_costs,
            total_slippage_pct=slippage_pnl,
            net_cumulative_return_pct=net_sim_pnl,
            pnl_breakdown=pnl_breakdown,
            stress_scenario_results=stress_scenario_results,
        )

    def verify_signal_equivalence(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        timestamps: list[Any] | None = None,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
        calibration_result: CalibrationResult | None = None,
        edge_check_result: EdgeCheckResult | None = None,
        horizon: str = HORIZON_INTRADAY,
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """
        BACK-002: Verifies that point-in-time replay signals match live predictor signals
        identically across all semantic, quantitative, event, and risk dimensions.
        """
        if timestamps is None or len(timestamps) == 0:
            warmup = min(60, len(stock_df) // 2)
            valid_indices = stock_df.index[warmup:]
            step = max(1, len(valid_indices) // 5)
            timestamps = [valid_indices[i * step] for i in range(min(5, len(valid_indices)))]
            if valid_indices[-1] not in timestamps:
                timestamps.append(valid_indices[-1])

        mismatches: list[dict[str, Any]] = []

        for ts in timestamps:
            # 1. Direct Live Pipeline at T (caller slices up to T)
            stock_slice = stock_df.loc[:ts]
            index_slice = index_df.loc[:ts] if index_df is not None and not index_df.empty else None

            # Filter events point-in-time
            ts_utc = pd.to_datetime(ts, utc=True)

            def _avail(e: dict) -> bool:
                t_val = e.get("timestamp") or e.get("date")
                return pd.to_datetime(t_val, utc=True) <= ts_utc if t_val else True

            f_corp = [e for e in corporate_events if _avail(e)] if corporate_events else None
            f_news = [e for e in news_articles if _avail(e)] if news_articles else None
            f_macro = (
                [e for e in macro_events if (isinstance(e, dict) and _avail(e))] if macro_events else macro_events
            )

            signal_live = self.predictor.generate_signal(
                symbol=symbol,
                stock_df=stock_slice,
                index_df=index_slice,
                macro_events=f_macro,
                corporate_events=f_corp,
                news_articles=f_news,
                calibration_result=calibration_result,
                edge_check_result=edge_check_result,
                horizon=horizon,
            )

            # 2. Backtest Point-in-Time Replay at T
            signal_replay = self.replay_point_in_time_signal(
                symbol=symbol,
                timestamp=ts,
                stock_df=stock_df,
                index_df=index_df,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
                calibration_result=calibration_result,
                edge_check_result=edge_check_result,
                horizon=horizon,
            )

            # Check action match
            if signal_live.action != signal_replay.action:
                mismatches.append({
                    "timestamp": str(ts),
                    "dimension": "action",
                    "live": signal_live.action,
                    "replay": signal_replay.action,
                })

            # Check model predicted class match
            if signal_live.model_predicted_class != signal_replay.model_predicted_class:
                mismatches.append({
                    "timestamp": str(ts),
                    "dimension": "model_predicted_class",
                    "live": signal_live.model_predicted_class,
                    "replay": signal_replay.model_predicted_class,
                })

            # Check raw confidence
            if abs(signal_live.raw_confidence - signal_replay.raw_confidence) > tolerance:
                mismatches.append({
                    "timestamp": str(ts),
                    "dimension": "raw_confidence",
                    "live": signal_live.raw_confidence,
                    "replay": signal_replay.raw_confidence,
                })

            # Check risk adjusted confidence
            if abs(signal_live.risk_adjusted_confidence - signal_replay.risk_adjusted_confidence) > tolerance:
                mismatches.append({
                    "timestamp": str(ts),
                    "dimension": "risk_adjusted_confidence",
                    "live": signal_live.risk_adjusted_confidence,
                    "replay": signal_replay.risk_adjusted_confidence,
                })

            # Check calibrated confidence
            l_cal = signal_live.calibrated_confidence
            r_cal = signal_replay.calibrated_confidence
            if (l_cal is None) != (r_cal is None):
                mismatches.append({
                    "timestamp": str(ts),
                    "dimension": "calibrated_confidence_presence",
                    "live": l_cal,
                    "replay": r_cal,
                })
            elif l_cal is not None and r_cal is not None:
                if abs(l_cal - r_cal) > tolerance:
                    mismatches.append({
                        "timestamp": str(ts),
                        "dimension": "calibrated_confidence_value",
                        "live": l_cal,
                        "replay": r_cal,
                    })

            # Check suppression match
            if signal_live.suppressed != signal_replay.suppressed:
                mismatches.append({
                    "timestamp": str(ts),
                    "dimension": "suppressed",
                    "live": signal_live.suppressed,
                    "replay": signal_replay.suppressed,
                })

            # Check contributing events match
            if len(signal_live.contributing_events) != len(signal_replay.contributing_events):
                mismatches.append({
                    "timestamp": str(ts),
                    "dimension": "contributing_events_count",
                    "live": len(signal_live.contributing_events),
                    "replay": len(signal_replay.contributing_events),
                })

        return {
            "is_equivalent": len(mismatches) == 0,
            "timestamps_checked": [str(t) for t in timestamps],
            "mismatches": mismatches,
            "allowed_nondeterministic_differences": [
                "timestamp: wall-clock execution time (now()) in live signal vs historical bar timestamp",
                "inference latency metrics",
            ],
            "details": f"Checked {len(timestamps)} bars; {len(mismatches)} mismatches.",
        }

    def run_universe_backtest(
        self,
        symbol_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        horizon: str = HORIZON_INTRADAY,
        enforce_universe: bool = True,
    ) -> dict[str, Any]:
        """
        BACK-004: Simulates a multi-symbol portfolio backtest across historical periods.
        If enforce_universe=True, checks eligible_universe(T) at each simulation timestamp T,
        prohibiting trades in symbols not in the index at T.
        """
        per_symbol_results: dict[str, BacktestResult] = {}
        total_predictions = 0
        total_trades = 0
        total_universe_excluded_bars = 0

        for symbol, (stock_df, index_df) in symbol_data.items():
            res = self.run_backtest_for_symbol(
                symbol=symbol,
                stock_df=stock_df,
                index_df=index_df,
                horizon=horizon,
                enforce_universe=enforce_universe,
            )
            per_symbol_results[symbol] = res
            total_predictions += res.n_test_predictions
            total_trades += res.n_trades_taken
            total_universe_excluded_bars += res.n_universe_excluded_bars

        strategy_returns = [r.strategy_cumulative_return_pct for r in per_symbol_results.values() if r.success]
        baseline_returns = [r.baseline_cumulative_return_pct for r in per_symbol_results.values() if r.success]
        alphas = [r.alpha_pct for r in per_symbol_results.values() if r.success]

        avg_strat = float(np.mean(strategy_returns)) if strategy_returns else 0.0
        avg_base = float(np.mean(baseline_returns)) if baseline_returns else 0.0
        avg_alpha = float(np.mean(alphas)) if alphas else 0.0

        return {
            "enforce_universe": enforce_universe,
            "horizon": horizon,
            "symbols_evaluated": list(symbol_data.keys()),
            "total_test_predictions": total_predictions,
            "total_trades_taken": total_trades,
            "total_universe_excluded_bars": total_universe_excluded_bars,
            "portfolio_avg_strategy_return_pct": round(avg_strat, 2),
            "portfolio_avg_baseline_return_pct": round(avg_base, 2),
            "portfolio_avg_alpha_pct": round(avg_alpha, 2),
            "per_symbol_results": per_symbol_results,
        }

    def quantify_survivorship_bias(
        self,
        symbol_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        horizon: str = HORIZON_INTRADAY,
    ) -> dict[str, Any]:
        """
        BACK-004: Quantifies survivorship bias by comparing backtest results with
        enforce_universe=False (static universe / survivorship-biased) versus
        enforce_universe=True (point-in-time dynamic universe / survivorship-bias-free).
        """
        biased_report = self.run_universe_backtest(
            symbol_data=symbol_data,
            horizon=horizon,
            enforce_universe=False,
        )
        bias_free_report = self.run_universe_backtest(
            symbol_data=symbol_data,
            horizon=horizon,
            enforce_universe=True,
        )

        strat_diff = (
            biased_report["portfolio_avg_strategy_return_pct"]
            - bias_free_report["portfolio_avg_strategy_return_pct"]
        )
        alpha_diff = biased_report["portfolio_avg_alpha_pct"] - bias_free_report["portfolio_avg_alpha_pct"]
        trades_diff = biased_report["total_trades_taken"] - bias_free_report["total_trades_taken"]

        return {
            "biased_run": biased_report,
            "bias_free_run": bias_free_report,
            "survivorship_bias_delta": {
                "strategy_return_inflation_pct": round(strat_diff, 2),
                "alpha_inflation_pct": round(alpha_diff, 2),
                "phantom_trades_count": trades_diff,
                "universe_excluded_bars": bias_free_report["total_universe_excluded_bars"],
            },
            "summary": (
                f"Survivorship bias inflated portfolio return by {strat_diff:+.2f}% and alpha by {alpha_diff:+.2f}% "
                f"via {trades_diff} phantom trades on {bias_free_report['total_universe_excluded_bars']} non-eligible constituent bars."
            ),
        }

    def run_stress_test(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        horizon: str = HORIZON_INTRADAY,
        scenarios: list[str] | None = None,
        enforce_universe: bool = False,
    ) -> dict[str, Any]:
        """
        BACK-003: Multi-scenario cost and slippage stress test.
        Runs backtest and execution simulation across specified or mandatory Phase 12 scenarios:
        BASE, +25% cost, +50% cost, +100% cost, HIGH_SLIPPAGE, LOW_LIQUIDITY.
        """
        res = self.run_backtest_for_symbol(
            symbol=symbol,
            stock_df=stock_df,
            index_df=index_df,
            horizon=horizon,
            enforce_universe=enforce_universe,
        )
        if not res.success:
            return {"success": False, "error": res.error, "scenarios": {}}

        scenarios_to_return = scenarios or list(res.stress_scenario_results.keys())
        filtered_results = {k: v for k, v in res.stress_scenario_results.items() if k in scenarios_to_return}

        return {
            "success": True,
            "symbol": symbol,
            "horizon": horizon,
            "n_trades_taken": res.n_trades_taken,
            "gross_cumulative_return_pct": res.gross_cumulative_return_pct,
            "total_transaction_costs_pct": res.total_transaction_costs_pct,
            "total_slippage_pct": res.total_slippage_pct,
            "net_cumulative_return_pct": res.net_cumulative_return_pct,
            "pnl_breakdown": res.pnl_breakdown,
            "is_viable_base": res.stress_scenario_results.get("BASE", {}).get("is_viable", False),
            "scenarios": filtered_results,
        }


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
