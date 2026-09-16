import pandas as pd

from ensemble_manager import EnsemblePrediction
from predictor import ACTION_HOLD, Predictor


class FakeDataFetcher:
    def check_staleness(self, stock_df, symbol):
        return False


class FakeFeatureEngineer:
    def engineer_features_for_horizon(self, stock_df, index_df, horizon):
        return pd.DataFrame({'atr': [2.0], 'rsi': [50.0]})


class FakeEnsemble:
    def predict(self, symbol, latest_row, horizon):
        return [EnsemblePrediction('UP', 0.8, 1.0, {'model_a': 'UP'}, 'model-a', 'feature-a')]


class FailingEventClassifier:
    def classify_batch(self, **kwargs):
        raise RuntimeError('late event failure')


def test_late_failure_preserves_model_forecast_but_suppresses_action():
    stock_df = pd.DataFrame({'Close': [100.0], 'High': [101.0]})

    predictor = Predictor(
        data_fetcher=FakeDataFetcher(),
        feature_engineer=FakeFeatureEngineer(),
        ensemble_manager=FakeEnsemble(),
        event_classifier=FailingEventClassifier(),
    )

    signal = predictor.generate_signal('RELIANCE', stock_df, horizon='INTRADAY')

    assert signal.model_predicted_class == 'UP'
    assert signal.raw_confidence == 0.8
    assert signal.action == ACTION_HOLD
    assert signal.is_safe_to_trade_live is False
    assert signal.suppressed is True
