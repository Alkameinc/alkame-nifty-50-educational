import numpy as np
import pandas as pd

from feature_engineer import FeatureEngineer


def test_feature_engineer():
    fe = FeatureEngineer()

    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "Open": np.random.rand(100) * 10 + 100,
            "High": np.random.rand(100) * 10 + 110,
            "Low": np.random.rand(100) * 10 + 90,
            "Close": np.random.rand(100) * 10 + 100,
            "Volume": np.random.rand(100) * 1000 + 1000,
        },
        index=dates,
    )

    # Static methods
    rsi = fe.compute_rsi(df["Close"])
    assert len(rsi) == 100

    macd, signal, hist = fe.compute_macd(df["Close"])
    assert len(macd) == 100

    pass
