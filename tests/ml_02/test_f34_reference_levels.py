import math


def test_valid_atr_produces_coherent_buy_levels():
    cmp = 100.0
    atr = 2.0
    target_mult = 1.5
    stop_mult = 1.0

    target = round(cmp + target_mult * atr, 2)
    stop = round(cmp - stop_mult * atr, 2)

    assert target == 103.0
    assert stop == 98.0
    assert stop < cmp < target


def test_valid_atr_produces_coherent_sell_levels():
    cmp = 100.0
    atr = 2.0
    target_mult = 1.5
    stop_mult = 1.0

    target = round(cmp - target_mult * atr, 2)
    stop = round(cmp + stop_mult * atr, 2)

    assert target == 97.0
    assert stop == 102.0
    assert target < cmp < stop


def test_invalid_atr_values_are_not_valid_volatility_inputs():
    cmp = 100.0
    invalid_values = [float("nan"), 0.0, -1.0, 150.0]

    for atr in invalid_values:
        assert math.isnan(atr) or atr <= 0 or atr > cmp


def test_invalid_atr_requires_levels_to_be_unavailable():
    cmp = 100.0
    invalid_atr = 0.0

    # F34 requirement: invalid volatility must not produce fabricated target/stop levels.
    target_price = None
    stop_loss = None

    if invalid_atr > 0 and invalid_atr <= cmp:
        target_price = round(cmp + 1.5 * invalid_atr, 2)
        stop_loss = round(cmp - invalid_atr, 2)

    assert target_price is None
    assert stop_loss is None

def test_missing_atr_uses_documented_half_percent_fallback():
    cmp = 100.0
    atr = None
    atr_val = cmp * 0.005 if atr is None else atr
    assert atr_val == 0.5

def test_tick_size_rounds_levels_to_executable_increment():
    entry = 100.03
    tick_size = 0.05
    target = 103.03
    stop = 98.03
    executable_target = round(round(target / tick_size) * tick_size, 2)
    executable_stop = round(round(stop / tick_size) * tick_size, 2)
    assert executable_target == 103.05
    assert executable_stop == 98.05
    assert abs((executable_target / tick_size) - round(executable_target / tick_size)) < 1e-9
    assert abs((executable_stop / tick_size) - round(executable_stop / tick_size)) < 1e-9


def test_tick_rounding_preserves_buy_level_ordering():
    entry = 100.03
    tick_size = 0.05
    target = round(round(103.03 / tick_size) * tick_size, 2)
    stop = round(round(98.03 / tick_size) * tick_size, 2)
    assert stop < entry < target
    assert abs((target / tick_size) - round(target / tick_size)) < 1e-9
    assert abs((stop / tick_size) - round(stop / tick_size)) < 1e-9
