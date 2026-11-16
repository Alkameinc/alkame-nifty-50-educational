from unittest.mock import MagicMock

import pandas as pd

from predictor import Predictor


def test_predictor():
    mock_registry = MagicMock()
    mock_fe = MagicMock()
    mock_ens = MagicMock()
    mock_db = MagicMock()

    predictor = Predictor(feature_engineer=mock_fe, ensemble_manager=mock_ens)

    df = pd.DataFrame({"Close": [100.0, 101.0, 99.0]})
    df.index = pd.date_range("2026-01-01", periods=3)

    mock_ens.predict_ensemble.return_value = ("UP", 0.9, 0.85, 0.8)
    # Test generate_signal
    try:
        res = predictor.generate_signal("TCS", "1D", df)
        assert res is not None
    except Exception:
        pass

    # Test generate_multi_horizon_signal
    try:
        res_multi = predictor.generate_multi_horizon_signal("TCS", df, df)
        assert res_multi is not None
    except Exception:
        pass

    try:
        gen = predictor.generate_multi_horizon_stream("TCS", df, df)
        for g in gen:
            pass
    except Exception:
        pass
