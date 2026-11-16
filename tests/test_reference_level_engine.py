import pandas as pd

from reference_level_engine import ReferenceLevelEngine


def test_reference_level_engine():
    engine = ReferenceLevelEngine()
    df = pd.DataFrame(
        {
            "High": [100.0, 105.0, 110.0, 108.0, 109.0, 112.0],
            "Low": [90.0, 95.0, 100.0, 98.0, 99.0, 102.0],
            "Close": [95.0, 100.0, 105.0, 103.0, 104.0, 107.0],
        }
    )

    # Needs some minimum length for calculations, e.g. 20 for bollinger bands.
    # Let's pad it out
    df_long = pd.concat([df] * 10, ignore_index=True)

    res = engine.get_reference_levels(symbol="TCS", stock_df=df_long, horizon="1D", user_avg_cost=100.0)

    assert res.symbol == "TCS"

    # Deltas
    deltas = engine.compute_deltas(res)
    assert deltas is not None

    # Swing levels directly
    hh, hl, lh, ll = engine.compute_swing_levels(df_long, lookback_window=10)

    # Done
