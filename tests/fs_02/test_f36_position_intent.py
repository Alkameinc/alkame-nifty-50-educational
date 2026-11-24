import pytest
from predictor import PredictionSignal, CLASS_TO_ACTION, ACTION_SELL, ACTION_HOLD
from position_planner import PositionPlanner, PositionPlan
from tests.fs_02.fixtures.prediction_signal import get_specimen_multi_signal

def test_f36_down_forecast_long_holding():
    planner = PositionPlanner(portfolio_capital=10_00_000, max_position_pct=0.10)
    multi = get_specimen_multi_signal()
    multi.primary_action = ACTION_SELL
    
    # 100 shares held @ 2500 CMP = 250,000 INR position
    plan = planner.generate_plan(
        multi_signal=multi,
        current_price=2500.0,
        current_position_inr=250_000.0,
        account_cash=500_000.0
    )
    
    assert plan.action == ACTION_SELL
    assert plan.total_allocated_inr == 0.0  # SELL does not allocate new capital
    assert plan.ladder == []

def test_f36_down_forecast_zero_holding():
    planner = PositionPlanner(portfolio_capital=10_00_000, max_position_pct=0.10)
    multi = get_specimen_multi_signal()
    multi.primary_action = ACTION_SELL
    
    # Zero shares held
    plan = planner.generate_plan(
        multi_signal=multi,
        current_price=2500.0,
        current_position_inr=0.0,
        account_cash=500_000.0
    )
    
    assert plan.action == ACTION_SELL
    assert plan.total_allocated_inr == 0.0  # Zero position cannot allocate capital
    assert plan.ladder == []

def test_f36_class_to_action_mapping_integrity():
    assert CLASS_TO_ACTION["UP"] == "BUY"
    assert CLASS_TO_ACTION["DOWN"] == "SELL"
    assert CLASS_TO_ACTION["FLAT"] == "HOLD"
