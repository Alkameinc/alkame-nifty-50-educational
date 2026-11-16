from unittest.mock import MagicMock, patch

import pandas as pd

from global_risk_monitor import GlobalRiskMonitor


@patch("global_risk_monitor.DataFetcher")
def test_global_risk_monitor(mock_fetcher_class):
    mock_fetcher = MagicMock()
    df = pd.DataFrame({"Close": [100.0, 101.0, 99.0, 95.0, 90.0]})
    mock_fetcher.fetch_global_tickers.return_value = {"^VIX": df, "^NSEI": df, "INR=X": df}
    mock_fetcher_class.return_value = mock_fetcher

    monitor = GlobalRiskMonitor()

    # Toggle state
    state = monitor.set_toggle(enabled=True, reason="Test", current_level="NORMAL")
    assert state.enabled is True

    loaded = monitor.get_toggle_state()
    assert loaded.enabled is True

    # Compute
    reading = monitor.compute_composite_risk()
    assert reading.composite_zscore is not None
    assert reading.risk_level in ["NORMAL", "ELEVATED", "CRISIS", "UNAVAILABLE"]

    # Multiplier
    mult = monitor.get_confidence_multiplier("IT", reading)
    assert isinstance(mult, float)
