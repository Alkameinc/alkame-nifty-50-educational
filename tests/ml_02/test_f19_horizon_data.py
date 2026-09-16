import pandas as pd

from config import HORIZON_CONFIG
from predictor import Predictor


def _bars(count, freq):
    index = pd.date_range(
        "2026-09-01 09:15",
        periods=count,
        freq=freq,
    )

    return pd.DataFrame(
        {
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.0] * count,
            "volume": [1000] * count,
        },
        index=index,
    )


def test_horizon_contract_uses_correct_bar_intervals():
    assert HORIZON_CONFIG["INTRADAY"]["bar_interval"] == "5m"
    assert HORIZON_CONFIG["INTRADAY"]["horizon_bars"] == 6

    assert HORIZON_CONFIG["3D"]["bar_interval"] == "1d"
    assert HORIZON_CONFIG["3D"]["horizon_bars"] == 3

    assert HORIZON_CONFIG["7D"]["bar_interval"] == "1d"
    assert HORIZON_CONFIG["7D"]["horizon_bars"] == 5


def test_daily_horizon_passes_daily_data_to_signal(monkeypatch):
    intraday = _bars(6, "5min")
    daily = _bars(3, "1D")

    class FakeDataFetcher:
        def __init__(self):
            self.calls = []

        def fetch_daily_ohlcv_incremental(self, ticker, full_period):
            self.calls.append((ticker, full_period))
            return daily.copy()

        def check_staleness(self, df):
            return False

    fetcher = FakeDataFetcher()
    predictor = Predictor(data_fetcher=fetcher)

    captured = {}

    def fake_generate_signal(**kwargs):
        captured["stock_df"] = kwargs["stock_df"]
        captured["index_df"] = kwargs["index_df"]
        captured["horizon"] = kwargs["horizon"]
        return object()

    monkeypatch.setattr(
        predictor,
        "generate_signal",
        fake_generate_signal,
    )

    signals = list(
        predictor.generate_multi_horizon_stream(
            symbol="RELIANCE",
            horizons=["3D"],
            stock_df=intraday,
            index_df=intraday,
            macro_events=[],
            corporate_events=[],
            news_articles=[],
            calibration_results={},
            edge_check_results={},
        )
    )

    assert len(signals) == 1
    assert captured["horizon"] == "3D"
    assert len(captured["stock_df"]) == 3
    assert len(captured["index_df"]) == 3
    assert captured["stock_df"] is not intraday
    assert captured["index_df"] is not intraday


def test_daily_horizon_does_not_fall_back_to_intraday_when_daily_data_unavailable(
    monkeypatch,
):
    intraday = _bars(6, "5min")

    class UnavailableDataFetcher:
        def fetch_daily_ohlcv_incremental(self, ticker, full_period):
            return pd.DataFrame()

    fetcher = UnavailableDataFetcher()
    predictor = Predictor(data_fetcher=fetcher)

    captured = {}

    def fake_generate_signal(**kwargs):
        captured["stock_df"] = kwargs["stock_df"]
        captured["index_df"] = kwargs["index_df"]
        captured["horizon"] = kwargs["horizon"]
        return object()

    monkeypatch.setattr(
        predictor,
        "generate_signal",
        fake_generate_signal,
    )

    signals = list(
        predictor.generate_multi_horizon_stream(
            symbol="RELIANCE",
            horizons=["3D"],
            stock_df=intraday,
            index_df=intraday,
            macro_events=[],
            corporate_events=[],
            news_articles=[],
            calibration_results={},
            edge_check_results={},
        )
    )

    assert len(signals) == 1
    assert captured["horizon"] == "3D"
    assert captured["stock_df"] is not intraday
    assert captured["index_df"] is not intraday


def test_corporate_events_are_forwarded_to_each_horizon(monkeypatch):
    intraday = _bars(6, "5min")

    corporate_events = [
        {
            "event_id": "fixture-123",
            "published_at": "2026-09-07T10:00:00+05:30",
            "headline": "Fixture corporate announcement",
            "sentiment_score": None,
            "impact_horizon": "7D",
        }
    ]

    predictor = Predictor()

    captured = []

    def fake_generate_signal(**kwargs):
        captured.append(kwargs["corporate_events"])
        return object()

    monkeypatch.setattr(
        predictor,
        "generate_signal",
        fake_generate_signal,
    )

    list(
        predictor.generate_multi_horizon_stream(
            symbol="RELIANCE",
            horizons=["INTRADAY", "3D"],
            stock_df=intraday,
            index_df=intraday,
            macro_events=[],
            corporate_events=corporate_events,
            news_articles=[],
            calibration_results={},
            edge_check_results={},
        )
    )

    assert len(captured) == 2
    assert captured[0] is corporate_events
    assert captured[1] is corporate_events