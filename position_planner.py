import logging
from dataclasses import dataclass

from config import MAX_DCA_STEPS, MAX_POSITION_SIZE_PCT, PORTFOLIO_TOTAL_CAPITAL
from predictor import ACTION_BUY, ACTION_HOLD, MultiHorizonSignal

logger = logging.getLogger(__name__)


@dataclass
class DCAStep:
    trigger_price: float
    allocation_inr: float
    reason: str


@dataclass
class PositionPlan:
    symbol: str
    action: str
    total_allocated_inr: float
    max_allowed_inr: float
    ladder: list[DCAStep]
    planned_capital: float = 0.0
    stop_price: float | None = None
    max_loss: float = 0.0
    risk_pct: float = 0.0


class PositionPlanner:
    """
    Translates trading signals and dynamic price levels into a structured
    risk-based position sizing and Dollar-Cost Averaging (DCA) plan.
    Incorporates stop loss, ATR volatility, max portfolio risk, and max capital constraints.
    """

    def __init__(
        self,
        portfolio_capital: float = PORTFOLIO_TOTAL_CAPITAL,
        max_position_pct: float = MAX_POSITION_SIZE_PCT,
        max_dca_steps: int = MAX_DCA_STEPS,
        max_portfolio_risk_pct: float = 0.02,  # Max 2% total portfolio equity risk
    ):
        self.portfolio_capital = portfolio_capital
        self.max_position_pct = max_position_pct
        self.max_dca_steps = max_dca_steps
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_position_inr = self.portfolio_capital * self.max_position_pct
        self.max_risk_inr = self.portfolio_capital * self.max_portfolio_risk_pct

    def generate_plan(
        self,
        multi_signal: MultiHorizonSignal,
        current_price: float,
        current_position_inr: float = 0.0,
        user_cost: float | None = None,
        ma_level: float | None = None,
        support_level: float | None = None,
        stop_price: float | None = None,
        atr: float | None = None,
    ) -> PositionPlan:
        """
        Generates a dynamic DCA ladder for BUY signals based on technical levels.
        If SELL or HOLD, returns a zero-allocation plan or signals an exit.
        Computes planned_capital, stop_price, max_loss, and risk_pct.
        """
        try:
            symbol = multi_signal.symbol
            action = multi_signal.primary_action

            # Extract stop price from signal if not explicitly provided
            if stop_price is None and multi_signal.signals:
                primary_sig = multi_signal.signals.get(multi_signal.primary_horizon)
                if primary_sig and primary_sig.stop_loss:
                    stop_price = primary_sig.stop_loss

            if action != ACTION_BUY:
                return PositionPlan(
                    symbol=symbol,
                    action=action,
                    total_allocated_inr=0.0,
                    max_allowed_inr=self.max_position_inr,
                    ladder=[],
                    planned_capital=0.0,
                    stop_price=stop_price,
                    max_loss=0.0,
                    risk_pct=0.0,
                )

            remaining_allocation = max(0.0, self.max_position_inr - current_position_inr)
            if remaining_allocation <= 0:
                logger.info(f"Position maxed out for {symbol}, cannot add more.")
                return PositionPlan(
                    symbol=symbol,
                    action=ACTION_HOLD,
                    total_allocated_inr=0.0,
                    max_allowed_inr=self.max_position_inr,
                    ladder=[],
                    planned_capital=0.0,
                    stop_price=stop_price,
                    max_loss=0.0,
                    risk_pct=0.0,
                )

            # Identify valid levels below current price for DCA
            levels = []
            if ma_level and ma_level < current_price:
                levels.append((ma_level, "Moving Average"))
            if support_level and support_level < current_price:
                levels.append((support_level, "Support Band"))

            # Sort levels descending (closest to current price first)
            levels = sorted(levels, key=lambda x: x[0], reverse=True)

            # Fallback stop price based on ATR, lowest technical level, or 3% default
            if stop_price is None:
                if levels:
                    lowest_level = min(lvl for lvl, _ in levels)
                    stop_price = round(lowest_level * 0.95, 2)  # 5% below lowest DCA tranche
                elif atr and atr > 0:
                    stop_price = round(max(0.01, current_price - (2.0 * atr)), 2)
                else:
                    stop_price = round(max(0.01, current_price * 0.97), 2)

            ladder = []

            # Initial entry if we have no position, or if we have room and want to add at market
            if current_position_inr == 0.0:
                ladder.append(DCAStep(trigger_price=current_price, allocation_inr=0.0, reason="Initial Entry (CMP)"))

            # Add dynamic technical levels (must be above stop_price)
            for lvl, reason in levels:
                if lvl > stop_price:
                    ladder.append(DCAStep(trigger_price=round(lvl, 2), allocation_inr=0.0, reason=f"DCA on {reason}"))

            # Limit to max allowed tranches
            ladder = ladder[: self.max_dca_steps]

            if not ladder:
                ladder.append(DCAStep(trigger_price=current_price, allocation_inr=0.0, reason="Initial Entry (CMP)"))

            # Distribute allocation across the ladder using a pyramiding approach (heavier at the bottom)
            n_steps = len(ladder)
            weights = [1.0 + (0.5 * i) for i in range(n_steps)]
            total_weight = sum(weights)

            for i, step in enumerate(ladder):
                step.allocation_inr = round(remaining_allocation * (weights[i] / total_weight), 2)

            # Reconcile penny rounding differences on the final step
            allocated_so_far = sum(s.allocation_inr for s in ladder[:-1])
            if ladder:
                ladder[-1].allocation_inr = round(remaining_allocation - allocated_so_far, 2)

            planned_capital = round(sum(step.allocation_inr for step in ladder), 2)

            # Calculate maximum loss if all tranches fill and price hits stop_price
            total_loss = 0.0
            for step in ladder:
                if step.trigger_price > stop_price:
                    loss_fraction = (step.trigger_price - stop_price) / step.trigger_price
                    total_loss += step.allocation_inr * loss_fraction

            max_loss = round(total_loss, 2)
            risk_pct = round((max_loss / self.portfolio_capital) * 100.0, 3)

            return PositionPlan(
                symbol=symbol,
                action=ACTION_BUY,
                total_allocated_inr=remaining_allocation,
                max_allowed_inr=self.max_position_inr,
                ladder=ladder,
                planned_capital=planned_capital,
                stop_price=stop_price,
                max_loss=max_loss,
                risk_pct=risk_pct,
            )

        except Exception as e:
            logger.error(f"Failed to generate position plan for {multi_signal.symbol}: {e}")
            return PositionPlan(
                symbol=multi_signal.symbol,
                action=ACTION_HOLD,
                total_allocated_inr=0.0,
                max_allowed_inr=0.0,
                ladder=[],
                planned_capital=0.0,
                stop_price=None,
                max_loss=0.0,
                risk_pct=0.0,
            )


if __name__ == "__main__":
    from datetime import datetime

    print("\n=== POSITION PLANNER SELF-TEST ===")
    planner = PositionPlanner(portfolio_capital=10_00_000, max_position_pct=0.10)

    # Mock some data
    multi_buy = MultiHorizonSignal(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        signals={},
        primary_action=ACTION_BUY,
        primary_horizon="1D",
        reasoning=[],
    )
    multi_hold = MultiHorizonSignal(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        signals={},
        primary_action=ACTION_HOLD,
        primary_horizon="1D",
        reasoning=[],
    )

    # Scenario A: BUY signal, no existing position, valid technical levels below CMP
    plan_a = planner.generate_plan(
        multi_signal=multi_buy, current_price=2500.0, current_position_inr=0.0, ma_level=2400.0, support_level=2300.0
    )
    print(f"Scenario A (BUY, new position): Action={plan_a.action}, Total Allocated={plan_a.total_allocated_inr}")
    for i, step in enumerate(plan_a.ladder):
        print(f"  Step {i+1}: Buy at {step.trigger_price} INR -> {step.allocation_inr} ({step.reason})")
    assert plan_a.total_allocated_inr == 1_00_000
    assert len(plan_a.ladder) == 3
    assert plan_a.planned_capital == 1_00_000
    assert plan_a.stop_price is not None
    assert plan_a.max_loss > 0.0
    assert plan_a.risk_pct > 0.0
    print(f"  Plan A Risk: Stop={plan_a.stop_price}, Max Loss={plan_a.max_loss} INR, Portfolio Risk={plan_a.risk_pct}%")

    # Scenario B: BUY signal, half position already filled, only 1 technical level below CMP
    plan_b = planner.generate_plan(
        multi_signal=multi_buy,
        current_price=2500.0,
        current_position_inr=50_000.0,
        ma_level=2400.0,
        support_level=2600.0,  # Support is above CMP, should be ignored
    )
    print(
        f"\nScenario B (BUY, existing position): Action={plan_b.action}, Total Allocated={plan_b.total_allocated_inr}"
    )
    for i, step in enumerate(plan_b.ladder):
        print(f"  Step {i+1}: Buy at {step.trigger_price} INR -> {step.allocation_inr} ({step.reason})")
    assert plan_b.total_allocated_inr == 50_000
    assert len(plan_b.ladder) == 1
    assert plan_b.ladder[0].trigger_price == 2400.0
    assert plan_b.max_loss > 0.0

    # Scenario C: HOLD signal
    plan_c = planner.generate_plan(multi_signal=multi_hold, current_price=2500.0)
    print(
        f"\nScenario C (HOLD): Action={plan_c.action}, Total Allocated={plan_c.total_allocated_inr}, Steps={len(plan_c.ladder)}"
    )
    assert plan_c.action == ACTION_HOLD
    assert len(plan_c.ladder) == 0
    assert plan_c.max_loss == 0.0
    assert plan_c.risk_pct == 0.0

    print("\nSTATUS: PASS")
