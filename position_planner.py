import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from config import PORTFOLIO_TOTAL_CAPITAL, MAX_POSITION_SIZE_PCT, MAX_DCA_STEPS
from predictor import MultiHorizonSignal, ACTION_BUY, ACTION_SELL, ACTION_HOLD

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
    ladder: List[DCAStep]

class PositionPlanner:
    """
    Translates trading signals and dynamic price levels into a structured 
    position sizing and Dollar-Cost Averaging (DCA) plan.
    """
    
    def __init__(
        self,
        portfolio_capital: float = PORTFOLIO_TOTAL_CAPITAL,
        max_position_pct: float = MAX_POSITION_SIZE_PCT,
        max_dca_steps: int = MAX_DCA_STEPS
    ):
        self.portfolio_capital = portfolio_capital
        self.max_position_pct = max_position_pct
        self.max_dca_steps = max_dca_steps
        self.max_position_inr = self.portfolio_capital * self.max_position_pct

    def generate_plan(
        self, 
        multi_signal: MultiHorizonSignal, 
        current_price: float, 
        current_position_inr: float = 0.0,
        user_cost: Optional[float] = None,
        ma_level: Optional[float] = None,
        support_level: Optional[float] = None
    ) -> PositionPlan:
        """
        Generates a dynamic DCA ladder for BUY signals based on technical levels.
        If SELL or HOLD, returns a zero-allocation plan or signals an exit.
        """
        try:
            symbol = multi_signal.symbol
            action = multi_signal.primary_action

            if action != ACTION_BUY:
                # E.g. if action == SELL, user should exit existing position. 
                # This planner focuses on entry sizing.
                return PositionPlan(
                    symbol=symbol, action=action, total_allocated_inr=0.0, 
                    max_allowed_inr=self.max_position_inr, ladder=[]
                )

            remaining_allocation = max(0.0, self.max_position_inr - current_position_inr)
            if remaining_allocation <= 0:
                logger.info(f"Position maxed out for {symbol}, cannot add more.")
                return PositionPlan(
                    symbol=symbol, action=ACTION_HOLD, total_allocated_inr=0.0, 
                    max_allowed_inr=self.max_position_inr, ladder=[]
                )

            # Identify valid levels below current price for DCA
            levels = []
            if ma_level and ma_level < current_price:
                levels.append((ma_level, "Moving Average"))
            if support_level and support_level < current_price:
                levels.append((support_level, "Support Band"))
            
            # Sort levels descending (closest to current price first)
            levels = sorted(levels, key=lambda x: x[0], reverse=True)
            
            ladder = []
            
            # Initial entry if we have no position, or if we have room and want to add at market
            if current_position_inr == 0.0:
                ladder.append(DCAStep(trigger_price=current_price, allocation_inr=0.0, reason="Initial Entry (CMP)"))
            
            # Add dynamic technical levels
            for lvl, reason in levels:
                ladder.append(DCAStep(trigger_price=round(lvl, 2), allocation_inr=0.0, reason=f"DCA on {reason}"))
                
            # Limit to max allowed tranches
            ladder = ladder[:self.max_dca_steps]
            
            if not ladder:
                ladder.append(DCAStep(trigger_price=current_price, allocation_inr=0.0, reason="Initial Entry (CMP)"))
                
            # Distribute allocation across the ladder using a pyramiding approach (heavier at the bottom)
            n_steps = len(ladder)
            weights = [1.0 + (0.5 * i) for i in range(n_steps)]
            total_weight = sum(weights)
            
            for i, step in enumerate(ladder):
                step.allocation_inr = round(remaining_allocation * (weights[i] / total_weight), 2)

            return PositionPlan(
                symbol=symbol, action=ACTION_BUY, 
                total_allocated_inr=remaining_allocation, 
                max_allowed_inr=self.max_position_inr, 
                ladder=ladder
            )
            
        except Exception as e:
            logger.error(f"Failed to generate position plan for {multi_signal.symbol}: {e}")
            return PositionPlan(symbol=multi_signal.symbol, action=ACTION_HOLD, total_allocated_inr=0.0, max_allowed_inr=0.0, ladder=[])

if __name__ == "__main__":
    from datetime import datetime
    
    print("\n=== POSITION PLANNER SELF-TEST ===")
    planner = PositionPlanner(portfolio_capital=10_00_000, max_position_pct=0.10)
    
    # Mock some data
    multi_buy = MultiHorizonSignal(
        symbol="RELIANCE", timestamp=datetime.now(), signals={},
        primary_action=ACTION_BUY, primary_horizon="1D", reasoning=[]
    )
    multi_hold = MultiHorizonSignal(
        symbol="RELIANCE", timestamp=datetime.now(), signals={},
        primary_action=ACTION_HOLD, primary_horizon="1D", reasoning=[]
    )
    
    # Scenario A: BUY signal, no existing position, valid technical levels below CMP
    plan_a = planner.generate_plan(
        multi_signal=multi_buy, current_price=2500.0, current_position_inr=0.0,
        ma_level=2400.0, support_level=2300.0
    )
    print(f"Scenario A (BUY, new position): Action={plan_a.action}, Total Allocated={plan_a.total_allocated_inr}")
    for i, step in enumerate(plan_a.ladder):
        print(f"  Step {i+1}: Buy at {step.trigger_price} INR -> {step.allocation_inr} ({step.reason})")
    assert plan_a.total_allocated_inr == 1_00_000
    assert len(plan_a.ladder) == 3
    
    # Scenario B: BUY signal, half position already filled, only 1 technical level below CMP
    plan_b = planner.generate_plan(
        multi_signal=multi_buy, current_price=2500.0, current_position_inr=50_000.0,
        ma_level=2400.0, support_level=2600.0 # Support is above CMP, should be ignored
    )
    print(f"\nScenario B (BUY, existing position): Action={plan_b.action}, Total Allocated={plan_b.total_allocated_inr}")
    for i, step in enumerate(plan_b.ladder):
        print(f"  Step {i+1}: Buy at {step.trigger_price} INR -> {step.allocation_inr} ({step.reason})")
    assert plan_b.total_allocated_inr == 50_000
    assert len(plan_b.ladder) == 1
    assert plan_b.ladder[0].trigger_price == 2400.0
    
    # Scenario C: HOLD signal
    plan_c = planner.generate_plan(multi_signal=multi_hold, current_price=2500.0)
    print(f"\nScenario C (HOLD): Action={plan_c.action}, Total Allocated={plan_c.total_allocated_inr}, Steps={len(plan_c.ladder)}")
    assert plan_c.action == ACTION_HOLD
    assert len(plan_c.ladder) == 0
    
    print("\nSTATUS: PASS")
