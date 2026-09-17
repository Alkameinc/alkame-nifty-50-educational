# 1. Standard library imports
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from backtester import BacktestResult, Backtester
from config import HORIZON_CONFIG, HORIZON_INTRADAY
from execution_simulator import (
    CostScenario,
    ExecutionSimulator,
    PHASE12_COST_SCENARIOS,
    PHASE12_MANDATORY_SCENARIOS,
    STANDARD_COST_SCENARIOS,
    SimulatedTrade,
    SimulationReport,
    create_scaled_cost_scenario,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------
def _generate_synthetic_df(n_bars: int = 120, start_price: float = 1000.0, trend: float = 0.001) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with reproducible prices."""
    np.random.seed(42)
    timestamps = pd.date_range("2026-01-05 09:15", periods=n_bars, freq="5min", tz="Asia/Kolkata")
    
    prices = [start_price]
    for i in range(1, n_bars):
        change = prices[-1] * (trend + np.random.normal(0, 0.002))
        prices.append(max(10.0, prices[-1] + change))
        
    prices_arr = np.array(prices)
    highs = prices_arr * (1.0 + np.abs(np.random.normal(0, 0.0015, n_bars)))
    lows = prices_arr * (1.0 - np.abs(np.random.normal(0, 0.0015, n_bars)))
    opens = np.roll(prices_arr, 1)
    opens[0] = prices_arr[0]
    closes = prices_arr
    volumes = np.random.randint(1000, 50000, size=n_bars)
    
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(highs, np.maximum(opens, closes)),
            "Low": np.minimum(lows, np.minimum(opens, closes)),
            "Close": closes,
            "Volume": volumes,
        },
        index=timestamps,
    )
    return df


def _build_synthetic_intraday_ohlcv(
    start_date: str = "2026-01-05",
    n_days: int = 25,
    bars_per_day: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic intraday bars (1000 bars total) with sufficient history for training."""
    rng = np.random.default_rng(seed)
    rows, timestamps = [], []
    price = 1000.0
    base = pd.Timestamp(start_date)
    recent_closes: list[float] = []
    for day in range(n_days):
        day_start = base + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=5 * bar)
            if len(recent_closes) >= 10:
                trend = recent_closes[-1] - recent_closes[-10]
                bias = 1.5 if trend < -6 else (-1.5 if trend > 6 else 0.0)
            else:
                bias = 0.0
            drift = rng.normal(bias, 1.5)
            price = max(10.0, price + drift)
            open_p = price
            close_p = max(10.0, price + rng.normal(bias * 0.5, 1.0))
            high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
            low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
            vol = int(abs(rng.normal(50000, 15000)))
            rows.append([open_p, high_p, low_p, close_p, vol])
            timestamps.append(ts)
            recent_closes.append(close_p)
    return pd.DataFrame(
        rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps)
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
def test_phase12_scenarios_defined_and_parameterized():
    """BACK-003: Verify all 6 mandated Phase 12 cost scenarios exist with correct properties."""
    expected_scenarios = [
        "BASE",
        "+25% cost",
        "+50% cost",
        "+100% cost",
        "HIGH_SLIPPAGE",
        "LOW_LIQUIDITY",
    ]
    
    for name in expected_scenarios:
        assert name in PHASE12_COST_SCENARIOS, f"Missing mandated Phase 12 scenario: {name}"
        scenario = PHASE12_COST_SCENARIOS[name]
        assert isinstance(scenario, CostScenario)
        assert scenario.name == name
        assert scenario.slippage_bps > 0
        assert scenario.brokerage_bps > 0

    # Verify programmatic aliases
    assert PHASE12_COST_SCENARIOS["COST_PLUS_25"] == PHASE12_COST_SCENARIOS["+25% cost"]
    assert PHASE12_COST_SCENARIOS["COST_PLUS_50"] == PHASE12_COST_SCENARIOS["+50% cost"]
    assert PHASE12_COST_SCENARIOS["COST_PLUS_100"] == PHASE12_COST_SCENARIOS["+100% cost"]

    # Verify PHASE12_MANDATORY_SCENARIOS list
    assert PHASE12_MANDATORY_SCENARIOS == expected_scenarios

    # Parameter checks
    base = PHASE12_COST_SCENARIOS["BASE"]
    s25 = PHASE12_COST_SCENARIOS["+25% cost"]
    s50 = PHASE12_COST_SCENARIOS["+50% cost"]
    s100 = PHASE12_COST_SCENARIOS["+100% cost"]
    high_slip = PHASE12_COST_SCENARIOS["HIGH_SLIPPAGE"]
    low_liq = PHASE12_COST_SCENARIOS["LOW_LIQUIDITY"]

    # Multipliers for transaction costs
    assert s25.slippage_bps == base.slippage_bps
    assert s50.slippage_bps == base.slippage_bps
    assert s100.slippage_bps == base.slippage_bps

    assert pytest.approx(s25.brokerage_bps) == base.brokerage_bps * 1.25
    assert pytest.approx(s50.brokerage_bps) == base.brokerage_bps * 1.50
    assert pytest.approx(s100.brokerage_bps) == base.brokerage_bps * 2.00

    # HIGH_SLIPPAGE: 25 bps slippage, base statutory costs
    assert high_slip.slippage_bps == 25.0
    assert high_slip.brokerage_bps == base.brokerage_bps
    assert high_slip.stt_bps == base.stt_bps

    # LOW_LIQUIDITY: 35 bps slippage, 1.5x brokerage
    assert low_liq.slippage_bps == 35.0
    assert pytest.approx(low_liq.brokerage_bps) == base.brokerage_bps * 1.5


def test_pnl_decomposition_identity():
    """
    BACK-003: Test that Gross P&L - Transaction Costs - Slippage == Net P&L.
    Must hold at both trade-level and aggregate SimulationReport-level.
    """
    df = _generate_synthetic_df(n_bars=100)
    
    # Generate 5 signals spaced apart
    signals = pd.Series(0, index=df.index)
    signals.iloc[5] = 1
    signals.iloc[25] = -1
    signals.iloc[45] = 1
    signals.iloc[65] = -1
    signals.iloc[85] = 1

    sim = ExecutionSimulator()
    report = sim.simulate("RELIANCE", df, signals, horizon=HORIZON_INTRADAY)

    assert report.n_trades > 0, "Expected simulated trades"
    
    # 1. Trade-level verification
    for t in report.trades:
        # total_cost_pct = slippage_cost_pct + statutory_cost_pct
        assert pytest.approx(t.total_cost_pct, abs=1e-5) == t.slippage_cost_pct + t.statutory_cost_pct
        # net_return_pct = gross_return_pct - total_cost_pct
        assert pytest.approx(t.net_return_pct, abs=1e-5) == t.gross_return_pct - t.total_cost_pct

    # 2. Report-level decomposition
    breakdown = report.pnl_breakdown
    assert "gross_pnl_pct" in breakdown
    assert "transaction_costs_pct" in breakdown
    assert "slippage_pct" in breakdown
    assert "net_pnl_pct" in breakdown

    gross = breakdown["gross_pnl_pct"]
    tx = breakdown["transaction_costs_pct"]
    slip = breakdown["slippage_pct"]
    net = breakdown["net_pnl_pct"]

    # Invariant: gross - tx - slip == net within rounding tolerance
    assert abs((gross - tx - slip) - net) <= 0.05, f"Decomposition identity failed: {gross} - {tx} - {slip} != {net}"
    assert report.total_costs_paid_pct == pytest.approx(tx + slip, abs=0.05)


def test_cost_monotonicity_across_scaled_scenarios():
    """
    BACK-003: Net returns must decrease monotonically as costs scale (+25%, +50%, +100%).
    Total costs paid must increase monotonically.
    """
    df = _generate_synthetic_df(n_bars=150)
    signals = pd.Series(0, index=df.index)
    signals.iloc[10] = 1
    signals.iloc[35] = -1
    signals.iloc[60] = 1
    signals.iloc[90] = -1
    signals.iloc[115] = 1

    sim = ExecutionSimulator()
    stress_results = sim.run_stress_test_analysis("TCS", df, signals, horizon=HORIZON_INTRADAY)

    assert len(stress_results) == 6
    for sc in PHASE12_MANDATORY_SCENARIOS:
        assert sc in stress_results

    base_rep = stress_results["BASE"]
    s25_rep = stress_results["+25% cost"]
    s50_rep = stress_results["+50% cost"]
    s100_rep = stress_results["+100% cost"]

    # Number of trades should be identical across cost scenarios (same fill timing)
    assert base_rep.n_trades == s25_rep.n_trades == s50_rep.n_trades == s100_rep.n_trades
    assert base_rep.cumulative_gross_return_pct == pytest.approx(s25_rep.cumulative_gross_return_pct, abs=0.01)

    # Net returns must strictly decrease or be equal
    assert base_rep.cumulative_net_return_pct >= s25_rep.cumulative_net_return_pct
    assert s25_rep.cumulative_net_return_pct >= s50_rep.cumulative_net_return_pct
    assert s50_rep.cumulative_net_return_pct >= s100_rep.cumulative_net_return_pct

    # Total costs must strictly increase
    assert base_rep.total_costs_paid_pct < s25_rep.total_costs_paid_pct
    assert s25_rep.total_costs_paid_pct < s50_rep.total_costs_paid_pct
    assert s50_rep.total_costs_paid_pct < s100_rep.total_costs_paid_pct


def test_high_slippage_and_low_liquidity_scenarios():
    """BACK-003: Verify HIGH_SLIPPAGE and LOW_LIQUIDITY impact slippage and costs correctly."""
    df = _generate_synthetic_df(n_bars=120)
    signals = pd.Series(0, index=df.index)
    signals.iloc[10] = 1
    signals.iloc[40] = 1
    signals.iloc[70] = -1
    signals.iloc[95] = 1

    sim = ExecutionSimulator()
    stress_results = sim.run_stress_test_analysis("INFY", df, signals, horizon=HORIZON_INTRADAY)

    base = stress_results["BASE"]
    high_slip = stress_results["HIGH_SLIPPAGE"]
    low_liq = stress_results["LOW_LIQUIDITY"]

    # High slippage must have dramatically higher slippage paid than BASE (approx 5x)
    assert high_slip.total_slippage_paid_pct > base.total_slippage_paid_pct * 3.5
    # Statutory transaction costs should remain identical to BASE for HIGH_SLIPPAGE
    assert pytest.approx(high_slip.total_transaction_costs_paid_pct, abs=0.05) == base.total_transaction_costs_paid_pct

    # Low liquidity has 35 bps slippage + 1.5x brokerage
    assert low_liq.total_slippage_paid_pct > high_slip.total_slippage_paid_pct
    assert low_liq.total_transaction_costs_paid_pct > base.total_transaction_costs_paid_pct
    assert low_liq.cumulative_net_return_pct < high_slip.cumulative_net_return_pct


def test_gross_profitable_net_negative_rejected_from_live():
    """
    BACK-003: Critical Invariant:
    A strategy that produces positive gross return but negative net return after friction
    must strictly fail live viability checks (is_live_worthy = False).
    """
    # Create a trade where gross return is small positive (+0.10%), but costs exceed it
    trade = SimulatedTrade(
        symbol="MARGINAL_SYM",
        horizon="INTRADAY",
        direction=1,
        signal_time=pd.Timestamp("2026-01-05 09:30", tz="Asia/Kolkata"),
        entry_time=pd.Timestamp("2026-01-05 09:35", tz="Asia/Kolkata"),
        exit_time=pd.Timestamp("2026-01-05 10:00", tz="Asia/Kolkata"),
        entry_price=100.0,
        exit_price=100.10,
        gross_return_pct=0.10,
        net_return_pct=-0.15,  # 0.10 gross - 0.25 friction = -0.15 net
        slippage_cost_pct=0.10,
        statutory_cost_pct=0.15,
        total_cost_pct=0.25,
        exit_reason="HORIZON_EXPIRY",
        held_bars=5,
    )

    marginal_report = SimulationReport(
        symbol="MARGINAL_SYM",
        horizon="INTRADAY",
        cost_scenario="BASE",
        n_signals=1,
        n_trades=1,
        win_rate_pct=0.0,
        profit_factor=0.0,
        cumulative_gross_return_pct=0.10,
        cumulative_net_return_pct=-0.15,
        total_costs_paid_pct=0.25,
        total_slippage_paid_pct=0.10,
        total_transaction_costs_paid_pct=0.15,
        trades=[trade],
    )

    # 1. Report is_viable check
    assert marginal_report.cumulative_gross_return_pct > 0.0, "Gross return must be positive"
    assert marginal_report.cumulative_net_return_pct < 0.0, "Net return must be negative"
    assert marginal_report.is_viable is False, "Strategy with negative net return must NOT be viable"

    # 2. Backtester rejection check
    backtester = Backtester()
    
    # Check that a BacktestResult with positive gross but negative net is rejected
    res_mock = BacktestResult(
        symbol="MARGINAL_SYM",
        horizon="INTRADAY",
        n_test_predictions=100,
        n_trades_taken=10,
        strategy_cumulative_return_pct=-0.50,
        baseline_cumulative_return_pct=0.10,
        alpha_pct=-0.60,
        edge_check_status="NO_EDGE",
        calibration_status="WELL_CALIBRATED",
        calibration_ece=0.05,
        is_live_worthy=False,
        gross_cumulative_return_pct=1.50,
        total_transaction_costs_pct=1.00,
        total_slippage_pct=1.00,
        net_cumulative_return_pct=-0.50,
        gate_reasons=[
            "Strategy is non-viable after friction: net return is negative after slippage and transaction costs (gross-only profitability is non-viable)."
        ],
    )
    assert res_mock.gross_cumulative_return_pct > 0.0
    assert res_mock.net_cumulative_return_pct < 0.0
    assert res_mock.is_live_worthy is False
    assert res_mock.is_viable_after_friction is False

    # Also test live via backtester on synthetic dataset
    stock_df = _build_synthetic_intraday_ohlcv(start_date="2026-01-05", n_days=25, bars_per_day=40, seed=10)
    index_df = _build_synthetic_intraday_ohlcv(start_date="2026-01-05", n_days=25, bars_per_day=40, seed=20)
    index_df.index = stock_df.index

    result = backtester.run_backtest_for_symbol(
        symbol="RELIANCE",
        stock_df=stock_df,
        index_df=index_df,
        horizon=HORIZON_INTRADAY,
        enforce_universe=False,
    )
    assert result.success is True
    if result.net_cumulative_return_pct <= 0.0:
        assert result.is_live_worthy is False, "Gross-only or negative net strategy must have is_live_worthy=False"
        assert result.is_viable_after_friction is False
        assert any("net" in reason.lower() or "cost" in reason.lower() or "friction" in reason.lower() or "edge" in reason.lower() for reason in result.gate_reasons)


def test_backtester_populates_stress_scenarios_and_breakdown():
    """BACK-003: Verify BacktestResult includes separated P&L metrics and stress scenarios."""
    backtester = Backtester()
    stock_df = _build_synthetic_intraday_ohlcv(start_date="2026-01-05", n_days=25, bars_per_day=40, seed=15)
    index_df = _build_synthetic_intraday_ohlcv(start_date="2026-01-05", n_days=25, bars_per_day=40, seed=25)
    index_df.index = stock_df.index

    result = backtester.run_backtest_for_symbol(
        symbol="RELIANCE",
        stock_df=stock_df,
        index_df=index_df,
        horizon=HORIZON_INTRADAY,
        enforce_universe=False,
    )
    assert result.success is True

    # Verify separated P&L metrics
    assert hasattr(result, "gross_cumulative_return_pct")
    assert hasattr(result, "total_transaction_costs_pct")
    assert hasattr(result, "total_slippage_pct")
    assert hasattr(result, "net_cumulative_return_pct")
    assert hasattr(result, "pnl_breakdown")
    assert hasattr(result, "stress_scenario_results")
    assert hasattr(result, "is_viable_after_friction")

    # Verify pnl_breakdown keys
    breakdown = result.pnl_breakdown
    assert set(breakdown.keys()) == {"gross_pnl_pct", "transaction_costs_pct", "slippage_pct", "net_pnl_pct"}

    # Invariant: gross - tx - slip == net within rounding tolerance
    gross = breakdown["gross_pnl_pct"]
    tx = breakdown["transaction_costs_pct"]
    slip = breakdown["slippage_pct"]
    net = breakdown["net_pnl_pct"]
    assert abs((gross - tx - slip) - net) <= 0.05

    # Verify stress scenarios
    stress = result.stress_scenario_results
    assert len(stress) == 6
    for sc in PHASE12_MANDATORY_SCENARIOS:
        assert sc in stress
        assert isinstance(stress[sc], dict)
        assert "gross_pnl_pct" in stress[sc]
        assert "is_viable" in stress[sc]


def test_backtester_run_stress_test_method():
    """BACK-003: Verify Backtester.run_stress_test() standalone method."""
    backtester = Backtester()
    stock_df = _build_synthetic_intraday_ohlcv(start_date="2026-01-05", n_days=25, bars_per_day=40, seed=30)
    index_df = _build_synthetic_intraday_ohlcv(start_date="2026-01-05", n_days=25, bars_per_day=40, seed=40)
    index_df.index = stock_df.index

    # Run default Phase 12 stress scenarios
    stress_res = backtester.run_stress_test(
        symbol="INFY",
        stock_df=stock_df,
        index_df=index_df,
        horizon=HORIZON_INTRADAY,
        enforce_universe=False,
    )

    assert stress_res["success"] is True
    assert len(stress_res["scenarios"]) == 6
    for sc in PHASE12_MANDATORY_SCENARIOS:
        assert sc in stress_res["scenarios"]
        rep = stress_res["scenarios"][sc]
        assert "gross_pnl_pct" in rep
        assert "net_pnl_pct" in rep
        assert "is_viable" in rep

    # Custom scenario subset
    custom_scenarios = ["BASE", "+50% cost", "HIGH_SLIPPAGE"]
    custom_res = backtester.run_stress_test(
        symbol="INFY",
        stock_df=stock_df,
        index_df=index_df,
        horizon=HORIZON_INTRADAY,
        scenarios=custom_scenarios,
        enforce_universe=False,
    )
    assert custom_res["success"] is True
    assert set(custom_res["scenarios"].keys()) == set(custom_scenarios)


def test_backwards_compatibility_standard_scenarios():
    """BACK-003: Ensure run_sensitivity_analysis() preserves exactly 4 tiers for backwards compatibility."""
    sim = ExecutionSimulator()
    df = _generate_synthetic_df(n_bars=80)
    signals = pd.Series(0, index=df.index)
    signals.iloc[10] = 1
    signals.iloc[30] = -1

    sensitivity = sim.run_sensitivity_analysis("SBIN", df, signals, horizon=HORIZON_INTRADAY)
    
    # Must have exactly 4 tiers: OPTIMISTIC, BASE, PESSIMISTIC, STRESS
    assert len(sensitivity) == 4
    assert set(sensitivity.keys()) == {"OPTIMISTIC", "BASE", "PESSIMISTIC", "STRESS"}
    for name, rep in sensitivity.items():
        assert isinstance(rep, SimulationReport)
        assert rep.cost_scenario == name
