import pytest
import pandas as pd
from datetime import datetime, UTC
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api import app
from insights_engine import MarketInsightsEngine, COMPLIANCE_DISCLAIMER
from api_schemas import MarketInsightsResponse, MarketBreadthResponse


@pytest.fixture
def mock_df():
    """Generates synthetic 30-bar daily OHLCV DataFrame for testing."""
    dates = pd.date_range(end=datetime.now(UTC), periods=30, freq="D")
    return pd.DataFrame({
        "Open": [2500.0 + i for i in range(30)],
        "High": [2520.0 + i for i in range(30)],
        "Low": [2490.0 + i for i in range(30)],
        "Close": [2510.0 + i for i in range(30)],
        "Volume": [100000 + i * 1000 for i in range(30)],
    }, index=dates)


def test_compute_technical_snapshot(mock_df):
    """Verify that TechnicalSnapshot computes correct factual metrics without errors."""
    engine = MarketInsightsEngine()
    snap = engine.compute_technical_snapshot("RELIANCE", mock_df)
    
    assert snap.symbol == "RELIANCE"
    assert 0 <= snap.rsi_14 <= 100
    assert snap.cmp > 2500.0
    assert snap.ema_20 is not None
    assert "H3" in snap.pivots_camarilla
    assert "L3" in snap.pivots_camarilla
    assert snap.volume_surge in (True, False)


def test_get_stock_insights_structure(mock_df):
    """Verify get_stock_insights returns structured market intelligence with disclaimer."""
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_stock_data.return_value = mock_df
    mock_fetcher.fetch_ohlcv.return_value = mock_df
    mock_fetcher.fetch_stock_fundamentals.return_value = {"pe_ratio": "22.5", "sector": "Energy"}
    
    engine = MarketInsightsEngine(data_fetcher=mock_fetcher)
    result = engine.get_stock_insights("RELIANCE")
    
    assert result.symbol == "RELIANCE"
    assert result.disclaimer == COMPLIANCE_DISCLAIMER
    assert result.technical.rsi_14 >= 0
    assert result.valuation["sector"] == "Energy"
    assert isinstance(result.corporate_events, list)
    assert isinstance(result.macro_context, dict)
    assert isinstance(result.news_sentiment, dict)


def test_api_insights_endpoints():
    """Verify FastAPI routes for /api/v1/insights and /api/v1/market-breadth."""
    client = TestClient(app)
    
    resp_ins = client.get("/api/v1/insights/RELIANCE")
    assert resp_ins.status_code == 200
    data = resp_ins.json()
    assert data["symbol"] == "RELIANCE"
    assert "technical" in data
    assert "pivots_camarilla" in data["technical"]
    assert "disclaimer" in data
    
    resp_brd = client.get("/api/v1/market-breadth")
    assert resp_brd.status_code == 200
    bdata = resp_brd.json()
    assert "total_constituents" in bdata
    assert "all_constituents" in bdata
    assert "disclaimer" in bdata
