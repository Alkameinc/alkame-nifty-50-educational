import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import HORIZON_CONFIG, HORIZON_INTRADAY, configure_logging

# 4. Logger setup
logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class ExitReason(str, Enum):
    TARGET = "TARGET"
    STOP_LOSS = "STOP_LOSS"
    HORIZON_EXPIRY = "HORIZON_EXPIRY"
    GAP_EXIT = "GAP_EXIT"


@dataclass
class CostScenario:
    name: str
    slippage_bps: float
    brokerage_bps: float
    stt_bps: float  # Securities Transaction Tax round-trip
    exchange_charges_bps: float  # NSE turnover fee
    gst_pct: float  # 18% on brokerage + exchange
    stamp_duty_bps: float  # Stamp duty on buy leg
    sebi_turnover_bps: float = 0.01

    @property
    def total_cost_pct(self) -> float:
        """Total round-trip friction expressed as a percentage."""
        brok_and_exch = self.brokerage_bps + self.exchange_charges_bps
        gst_bps = brok_and_exch * (self.gst_pct / 100.0)
        statutory_bps = (
            self.stt_bps + self.exchange_charges_bps + gst_bps + self.stamp_duty_bps + self.sebi_turnover_bps
        )
        total_bps = self.slippage_bps * 2.0 + self.brokerage_bps + statutory_bps
        return total_bps / 100.0  # bps -> %


STANDARD_COST_SCENARIOS = {
    "OPTIMISTIC": CostScenario(
        name="OPTIMISTIC",
        slippage_bps=2.0,
        brokerage_bps=0.0,
        stt_bps=0.0,
        exchange_charges_bps=0.0,
        gst_pct=0.0,
        stamp_duty_bps=0.0,
    ),
    "BASE": CostScenario(
        name="BASE",
        slippage_bps=5.0,
        brokerage_bps=3.0,
        stt_bps=1.25,  # Intraday equity STT (0.025% on sell side)
        exchange_charges_bps=0.345,
        gst_pct=18.0,
        stamp_duty_bps=0.3,
    ),
    "PESSIMISTIC": CostScenario(
        name="PESSIMISTIC",
        slippage_bps=10.0,
        brokerage_bps=5.0,
        stt_bps=2.5,
        exchange_charges_bps=0.345,
        gst_pct=18.0,
        stamp_duty_bps=0.3,
    ),
    "STRESS": CostScenario(
        name="STRESS",
        slippage_bps=20.0,
        brokerage_bps=10.0,
        stt_bps=5.0,
        exchange_charges_bps=0.5,
        gst_pct=18.0,
        stamp_duty_bps=0.5,
    ),
}


def create_scaled_cost_scenario(
    base: CostScenario, cost_multiplier: float, name: str, slippage_bps: float | None = None
) -> CostScenario:
    """Scale brokerage and statutory transaction costs while keeping tax percentages intact."""
    slip = slippage_bps if slippage_bps is not None else base.slippage_bps
    return CostScenario(
        name=name,
        slippage_bps=slip,
        brokerage_bps=base.brokerage_bps * cost_multiplier,
        stt_bps=base.stt_bps * cost_multiplier,
        exchange_charges_bps=base.exchange_charges_bps * cost_multiplier,
        gst_pct=base.gst_pct,
        stamp_duty_bps=base.stamp_duty_bps * cost_multiplier,
        sebi_turnover_bps=base.sebi_turnover_bps * cost_multiplier,
    )


# BACK-003: Phase 12 Configurable Cost and Slippage Stress Scenarios
_BASE_REF = STANDARD_COST_SCENARIOS["BASE"]

PHASE12_COST_SCENARIOS: dict[str, CostScenario] = {
    "BASE": _BASE_REF,
    "+25% cost": create_scaled_cost_scenario(_BASE_REF, 1.25, "+25% cost"),
    "+50% cost": create_scaled_cost_scenario(_BASE_REF, 1.50, "+50% cost"),
    "+100% cost": create_scaled_cost_scenario(_BASE_REF, 2.00, "+100% cost"),
    "HIGH_SLIPPAGE": CostScenario(
        name="HIGH_SLIPPAGE",
        slippage_bps=25.0,  # 5x base slippage (50.0 bps round trip)
        brokerage_bps=_BASE_REF.brokerage_bps,
        stt_bps=_BASE_REF.stt_bps,
        exchange_charges_bps=_BASE_REF.exchange_charges_bps,
        gst_pct=_BASE_REF.gst_pct,
        stamp_duty_bps=_BASE_REF.stamp_duty_bps,
        sebi_turnover_bps=_BASE_REF.sebi_turnover_bps,
    ),
    "LOW_LIQUIDITY": CostScenario(
        name="LOW_LIQUIDITY",
        slippage_bps=35.0,  # 7x base slippage (70.0 bps round trip) + elevated friction
        brokerage_bps=_BASE_REF.brokerage_bps * 1.5,
        stt_bps=_BASE_REF.stt_bps,
        exchange_charges_bps=_BASE_REF.exchange_charges_bps * 1.5,
        gst_pct=_BASE_REF.gst_pct,
        stamp_duty_bps=_BASE_REF.stamp_duty_bps,
        sebi_turnover_bps=_BASE_REF.sebi_turnover_bps,
    ),
}

# Aliases for convenient programmatic access
PHASE12_COST_SCENARIOS["COST_PLUS_25"] = PHASE12_COST_SCENARIOS["+25% cost"]
PHASE12_COST_SCENARIOS["COST_PLUS_50"] = PHASE12_COST_SCENARIOS["+50% cost"]
PHASE12_COST_SCENARIOS["COST_PLUS_100"] = PHASE12_COST_SCENARIOS["+100% cost"]

PHASE12_MANDATORY_SCENARIOS = [
    "BASE",
    "+25% cost",
    "+50% cost",
    "+100% cost",
    "HIGH_SLIPPAGE",
    "LOW_LIQUIDITY",
]


@dataclass
class SimulatedTrade:
    symbol: str
    horizon: str
    direction: int  # 1 for LONG, -1 for SHORT
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return_pct: float
    net_return_pct: float
    slippage_cost_pct: float
    statutory_cost_pct: float
    total_cost_pct: float
    exit_reason: str
    held_bars: int


@dataclass
class SimulationReport:
    symbol: str
    horizon: str
    cost_scenario: str
    n_signals: int
    n_trades: int
    win_rate_pct: float
    profit_factor: float
    cumulative_gross_return_pct: float
    cumulative_net_return_pct: float
    total_costs_paid_pct: float
    total_slippage_paid_pct: float = 0.0
    total_transaction_costs_paid_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    trades: list[SimulatedTrade] = field(default_factory=list)

    @property
    def pnl_breakdown(self) -> dict[str, float]:
        """BACK-003: Explicit P&L decomposition separating gross, transaction costs, slippage, and net."""
        return {
            "gross_pnl_pct": self.cumulative_gross_return_pct,
            "transaction_costs_pct": self.total_transaction_costs_paid_pct,
            "slippage_pct": self.total_slippage_paid_pct,
            "net_pnl_pct": self.cumulative_net_return_pct,
        }

    @property
    def is_viable(self) -> bool:
        """
        BACK-003: Evaluate strategy viability after all friction.
        Never report only gross performance when evaluating strategy viability.
        """
        if self.n_trades == 0:
            return False
        return self.cumulative_net_return_pct > 0.0 and self.profit_factor > 1.0


class ExecutionSimulator:
    """
    Event-driven discrete bar execution simulator (QNT-002, QNT-003).
    Models realistic order fills with:
    - 1-bar execution delay: signal at close t is filled at open t+1 +/- slippage
    - Bar-by-bar high/low path checking for target vs stop loss hit resolution
    - Conservative priority: if both target and stop touched on the same bar,
      stop-loss is always assumed to have hit first
    - Indian statutory taxation and slippage scenarios (Optimistic, Base, Pessimistic, Stress)
    """

    def __init__(self, cost_scenario: CostScenario = STANDARD_COST_SCENARIOS["BASE"]):
        self.cost_scenario = cost_scenario

    def simulate(
        self,
        symbol: str,
        df: pd.DataFrame,
        signals: pd.Series,
        horizon: str = HORIZON_INTRADAY,
        stop_loss_pct: float | None = None,
        profit_target_pct: float | None = None,
        cost_scenario: CostScenario | None = None,
    ) -> SimulationReport:
        """
        Runs simulation across df bars using signals series (indexed by timestamp, values: 1=UP, -1=DOWN, 0=FLAT).
        """
        scenario = cost_scenario or self.cost_scenario
        horizon_cfg = HORIZON_CONFIG.get(horizon, HORIZON_CONFIG["INTRADAY"])
        horizon_bars = cast(int, horizon_cfg.get("horizon_bars", 5))

        # Default stop/target derived from horizon deadband if not explicitly given
        deadband = cast(float, horizon_cfg.get("deadband_pct_default", 0.5))
        sl_pct = stop_loss_pct if stop_loss_pct is not None else deadband * 1.5
        tp_pct = profit_target_pct if profit_target_pct is not None else deadband * 2.5

        trades: list[SimulatedTrade] = []
        n_signals = int((signals != 0).sum())
        slippage_frac = scenario.slippage_bps / 10000.0
        round_trip_cost_pct = scenario.total_cost_pct

        timestamps = df.index
        n_bars = len(df)
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values

        i = 0
        while i < n_bars - 1:
            ts = timestamps[i]
            if ts not in signals.index:
                i += 1
                continue

            sig = signals.loc[ts]
            if sig == 0 or pd.isna(sig):
                i += 1
                continue

            direction = 1 if sig > 0 else -1

            # 1-bar execution delay: entry happens at bar i+1 Open
            entry_idx = i + 1
            if entry_idx >= n_bars:
                break

            entry_time = timestamps[entry_idx]
            raw_entry = opens[entry_idx]

            # Entry slippage: Long pays more, Short receives less
            if direction == 1:
                entry_price = raw_entry * (1.0 + slippage_frac)
                stop_price = entry_price * (1.0 - sl_pct / 100.0)
                target_price = entry_price * (1.0 + tp_pct / 100.0)
            else:
                entry_price = raw_entry * (1.0 - slippage_frac)
                stop_price = entry_price * (1.0 + sl_pct / 100.0)
                target_price = entry_price * (1.0 - tp_pct / 100.0)

            # Traversal along intermediate bars up to horizon_bars
            max_exit_idx = min(n_bars - 1, entry_idx + horizon_bars)
            exit_idx = max_exit_idx
            exit_price = closes[max_exit_idx]
            exit_reason = ExitReason.HORIZON_EXPIRY

            for k in range(entry_idx, max_exit_idx + 1):
                b_high = highs[k]
                b_low = lows[k]

                if direction == 1:
                    stop_hit = b_low <= stop_price
                    target_hit = b_high >= target_price
                    if stop_hit and target_hit:
                        # Ambiguity resolution: conservative worst-case ordering
                        exit_price = stop_price * (1.0 - slippage_frac)
                        exit_reason = ExitReason.STOP_LOSS
                        exit_idx = k
                        break
                    elif stop_hit:
                        exit_price = stop_price * (1.0 - slippage_frac)
                        exit_reason = ExitReason.STOP_LOSS
                        exit_idx = k
                        break
                    elif target_hit:
                        exit_price = target_price * (1.0 - slippage_frac)
                        exit_reason = ExitReason.TARGET
                        exit_idx = k
                        break
                else:  # SHORT
                    stop_hit = b_high >= stop_price
                    target_hit = b_low <= target_price
                    if stop_hit and target_hit:
                        # Ambiguity resolution: conservative worst-case ordering
                        exit_price = stop_price * (1.0 + slippage_frac)
                        exit_reason = ExitReason.STOP_LOSS
                        exit_idx = k
                        break
                    elif stop_hit:
                        exit_price = stop_price * (1.0 + slippage_frac)
                        exit_reason = ExitReason.STOP_LOSS
                        exit_idx = k
                        break
                    elif target_hit:
                        exit_price = target_price * (1.0 + slippage_frac)
                        exit_reason = ExitReason.TARGET
                        exit_idx = k
                        break

            # If horizon expired, apply exit slippage at final bar close
            if exit_reason == ExitReason.HORIZON_EXPIRY:
                if direction == 1:
                    exit_price = closes[exit_idx] * (1.0 - slippage_frac)
                else:
                    exit_price = closes[exit_idx] * (1.0 + slippage_frac)

            exit_time = timestamps[exit_idx]

            # Return calculation
            if direction == 1:
                gross_ret = (exit_price - entry_price) / entry_price * 100.0
            else:
                gross_ret = (entry_price - exit_price) / entry_price * 100.0

            slippage_cost_pct = (scenario.slippage_bps * 2.0) / 100.0
            statutory_cost_pct = max(0.0, round_trip_cost_pct - slippage_cost_pct)
            net_ret = gross_ret - round_trip_cost_pct

            trade = SimulatedTrade(
                symbol=symbol,
                horizon=horizon,
                direction=direction,
                signal_time=ts,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=round(float(entry_price), 4),
                exit_price=round(float(exit_price), 4),
                gross_return_pct=round(float(gross_ret), 4),
                net_return_pct=round(float(net_ret), 4),
                slippage_cost_pct=round(float(slippage_cost_pct), 4),
                statutory_cost_pct=round(float(statutory_cost_pct), 4),
                total_cost_pct=round(float(round_trip_cost_pct), 4),
                exit_reason=exit_reason.value,
                held_bars=exit_idx - entry_idx,
            )
            trades.append(trade)

            # Advance to next bar after trade exit
            i = max(i + 1, exit_idx)

        # Compute summary metrics
        n_trades = len(trades)
        if n_trades == 0:
            return SimulationReport(
                symbol=symbol,
                horizon=horizon,
                cost_scenario=scenario.name,
                n_signals=n_signals,
                n_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                cumulative_gross_return_pct=0.0,
                cumulative_net_return_pct=0.0,
                total_costs_paid_pct=0.0,
                total_slippage_paid_pct=0.0,
                total_transaction_costs_paid_pct=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                trades=[],
            )

        net_returns = np.array([t.net_return_pct for t in trades])
        gross_returns = np.array([t.gross_return_pct for t in trades])
        costs = np.array([t.total_cost_pct for t in trades])
        slippage_costs = np.array([t.slippage_cost_pct for t in trades])
        statutory_costs = np.array([t.statutory_cost_pct for t in trades])

        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        win_rate = (len(wins) / n_trades) * 100.0

        gross_win_sum = float(np.sum(gross_returns[gross_returns > 0])) if np.any(gross_returns > 0) else 0.0
        gross_loss_sum = abs(float(np.sum(gross_returns[gross_returns < 0]))) if np.any(gross_returns < 0) else 0.0
        profit_factor = (
            round(gross_win_sum / gross_loss_sum, 3) if gross_loss_sum > 0 else (99.0 if gross_win_sum > 0 else 0.0)
        )

        cum_net = float(np.sum(net_returns))
        cum_gross = float(np.sum(gross_returns))
        total_costs = float(np.sum(costs))
        total_slippage = float(np.sum(slippage_costs))
        total_tx_costs = float(np.sum(statutory_costs))

        # Max Drawdown
        equity_curve = np.cumsum(net_returns)
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = peak - equity_curve
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Sharpe & Sortino (annualized using 252 trading days)
        mean_ret = float(np.mean(net_returns))
        std_ret = float(np.std(net_returns))
        sharpe = round((mean_ret / std_ret) * np.sqrt(252), 3) if std_ret > 0 else 0.0

        downside_std = float(np.std(losses)) if len(losses) > 1 else 0.0
        sortino = round((mean_ret / downside_std) * np.sqrt(252), 3) if downside_std > 0 else 0.0

        return SimulationReport(
            symbol=symbol,
            horizon=horizon,
            cost_scenario=scenario.name,
            n_signals=n_signals,
            n_trades=n_trades,
            win_rate_pct=round(win_rate, 2),
            profit_factor=profit_factor,
            cumulative_gross_return_pct=round(cum_gross, 2),
            cumulative_net_return_pct=round(cum_net, 2),
            total_costs_paid_pct=round(total_costs, 2),
            total_slippage_paid_pct=round(total_slippage, 2),
            total_transaction_costs_paid_pct=round(total_tx_costs, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            trades=trades,
        )

    def run_sensitivity_analysis(
        self,
        symbol: str,
        df: pd.DataFrame,
        signals: pd.Series,
        horizon: str = HORIZON_INTRADAY,
    ) -> dict[str, SimulationReport]:
        """
        Runs the simulation across all standard cost scenarios (OPTIMISTIC, BASE, PESSIMISTIC, STRESS)
        to evaluate strategy robustness and cost-decay profiles (QNT-003).
        """
        results = {}
        for name, scenario in STANDARD_COST_SCENARIOS.items():
            results[name] = self.simulate(
                symbol=symbol,
                df=df,
                signals=signals,
                horizon=horizon,
                cost_scenario=scenario,
            )
        return results

    def run_stress_test_analysis(
        self,
        symbol: str,
        df: pd.DataFrame,
        signals: pd.Series,
        horizon: str = HORIZON_INTRADAY,
        scenarios: list[str] | None = None,
    ) -> dict[str, SimulationReport]:
        """
        BACK-003: Runs the simulation across Phase 12 mandated stress scenarios:
        BASE, +25% cost, +50% cost, +100% cost, HIGH_SLIPPAGE, LOW_LIQUIDITY.
        """
        scenario_names = scenarios or PHASE12_MANDATORY_SCENARIOS
        results = {}
        for name in scenario_names:
            if name in PHASE12_COST_SCENARIOS:
                scenario = PHASE12_COST_SCENARIOS[name]
            elif name in STANDARD_COST_SCENARIOS:
                scenario = STANDARD_COST_SCENARIOS[name]
            else:
                raise ValueError(f"Unknown cost scenario: {name}")

            results[name] = self.simulate(
                symbol=symbol,
                df=df,
                signals=signals,
                horizon=horizon,
                cost_scenario=scenario,
            )
        return results


if __name__ == "__main__":
    configure_logging(log_filename="execution_simulator_selftest.log")
    print("ExecutionSimulator self-test...")

    # Build simple synthetic bars
    dates = pd.date_range("2026-01-01 09:15", periods=100, freq="5min")
    prices = 100.0 + np.cumsum(np.random.normal(0.05, 0.5, size=100))
    df = pd.DataFrame(
        {"Open": prices, "High": prices + 0.3, "Low": prices - 0.3, "Close": prices + 0.1, "Volume": 10000}, index=dates
    )

    signals = pd.Series(0, index=dates)
    signals.iloc[5] = 1  # Long
    signals.iloc[25] = -1  # Short
    signals.iloc[50] = 1  # Long

    sim = ExecutionSimulator()
    sensitivity = sim.run_sensitivity_analysis("TEST", df, signals)
    assert len(sensitivity) == 4, "run_sensitivity_analysis should retain exactly 4 tiers"
    for name, report in sensitivity.items():
        print(
            f"Scenario: {name} | Trades: {report.n_trades} | Net: {report.cumulative_net_return_pct:.2f}% | DD: {report.max_drawdown_pct:.2f}%"
        )

    # BACK-003: Verify Phase 12 Stress Test Analysis
    stress_reports = sim.run_stress_test_analysis("TEST", df, signals)
    assert len(stress_reports) == 6, f"Expected 6 Phase 12 stress scenarios, got {len(stress_reports)}"
    for s_name in PHASE12_MANDATORY_SCENARIOS:
        assert s_name in stress_reports
        rep = stress_reports[s_name]
        bd = rep.pnl_breakdown
        assert "gross_pnl_pct" in bd and "transaction_costs_pct" in bd and "slippage_pct" in bd and "net_pnl_pct" in bd
        # Invariant: gross - tx_cost - slippage == net (within rounding)
        diff = abs((bd["gross_pnl_pct"] - bd["transaction_costs_pct"] - bd["slippage_pct"]) - bd["net_pnl_pct"])
        assert diff <= 0.05, f"P&L decomposition failed for {s_name}: {bd}"
        print(
            f"Stress: {s_name:<14} | Gross: {bd['gross_pnl_pct']:>6.2f}% | TxCost: {bd['transaction_costs_pct']:>5.2f}% | "
            f"Slippage: {bd['slippage_pct']:>5.2f}% | Net: {bd['net_pnl_pct']:>6.2f}% | Viable: {rep.is_viable}"
        )

    print("ExecutionSimulator self-test PASSED")
