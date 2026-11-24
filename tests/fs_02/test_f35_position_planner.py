import pytest
from position_planner import PositionPlanner, ACTION_BUY, ACTION_HOLD
from narrative_builder import NarrativeBuilder
from tests.fs_02.fixtures.prediction_signal import get_specimen_multi_signal

def test_f35_position_planner_normal_cap_calculation():
    # Capital = 10,000 INR, 10% cap = 1,000 INR
    # Held = 400 INR -> remaining symbol cap = 600 INR
    # Total cash = 700 INR, committed = 500 INR -> free cash = 200 INR
    # Added allocation = min(600, 200) = 200 INR
    planner = PositionPlanner(portfolio_capital=10_000.0, max_position_pct=0.10)
    multi = get_specimen_multi_signal()
    multi.primary_action = ACTION_BUY
    
    plan = planner.generate_plan(
        multi_signal=multi,
        current_price=100.0,
        current_position_inr=400.0,
        ma_level=98.0,
        support_level=97.0,
        account_cash=700.0,
        committed_cash=500.0
    )
    
    assert plan.action == ACTION_BUY
    assert plan.total_allocated_inr == 200.0
    assert plan.planned_capital == 200.0
    assert plan.total_allocated_inr <= 200.0

def test_f35_position_planner_missing_account_facts():
    planner = PositionPlanner(portfolio_capital=10_000.0, max_position_pct=0.10)
    multi = get_specimen_multi_signal()
    multi.primary_action = ACTION_BUY
    
    # Zero free account cash (account_cash=0.0)
    plan = planner.generate_plan(
        multi_signal=multi,
        current_price=100.0,
        current_position_inr=400.0,
        account_cash=0.0
    )
    
    # Zero free cash cannot authorize allocation
    assert plan.action == ACTION_HOLD
    assert plan.total_allocated_inr == 0.0
    assert plan.ladder == []

def test_f35_position_planner_position_at_cap():
    planner = PositionPlanner(portfolio_capital=10_000.0, max_position_pct=0.10)
    multi = get_specimen_multi_signal()
    multi.primary_action = ACTION_BUY
    
    # Position already at max cap 1,000 INR
    plan = planner.generate_plan(
        multi_signal=multi,
        current_price=100.0,
        current_position_inr=1000.0,
        account_cash=10_000.0,
        committed_cash=0.0
    )
    
    assert plan.action == ACTION_HOLD
    assert plan.total_allocated_inr == 0.0

def test_f35_narrative_builder_score_wording():
    builder = NarrativeBuilder()
    multi = get_specimen_multi_signal()
    multi.signals["INTRADAY"].suppressed = False
    
    narrative = builder.build_stock_narrative(multi)
    assert "80% confidence" not in narrative
    assert "raw model score 80.0%" in narrative or "calibrated probability: unavailable" in narrative
