import logging
import pandas as pd
from typing import List
from dataclasses import dataclass
from config import NIFTY50_SYMBOLS, HORIZON_INTRADAY, to_yfinance_ticker
from data_fetcher import DataFetcher
from predictor import Predictor, ACTION_BUY, ACTION_SELL
from feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

@dataclass
class ScalpSetup:
    symbol: str
    action: str
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    risk_reward_ratio: float
    reasoning: List[str]

class ScalpingEngine:
    """
    Dedicated engine for identifying high-probability intraday scalping 
    opportunities using ATR-based risk management.
    """
    def __init__(self, predictor: Predictor, data_fetcher: DataFetcher, feature_engineer: FeatureEngineer):
        self.predictor = predictor
        self.data_fetcher = data_fetcher
        self.feature_engineer = feature_engineer

    def find_opportunities(self, limit: int = 5) -> List[ScalpSetup]:
        """
        Scans NIFTY 50 for intraday scalping setups, calculating precise 
        ATR-based entry, target, and stop loss levels.
        """
        logger.info("Scanning for intraday scalping setups...")
        opportunities = []
        
        index_ticker = "^NSEI"
        index_df = self.data_fetcher.fetch_ohlcv(index_ticker, interval="5m")
        
        for symbol in NIFTY50_SYMBOLS:
            try:
                yf_ticker = to_yfinance_ticker(symbol)
                stock_df = self.data_fetcher.fetch_ohlcv(yf_ticker, interval="5m")
                
                if stock_df is None or stock_df.empty:
                    continue
                
                if self.data_fetcher.check_staleness(stock_df, yf_ticker):
                    continue
                
                sig = self.predictor.generate_signal(
                    symbol=symbol,
                    horizon=HORIZON_INTRADAY,
                    stock_df=stock_df,
                    index_df=index_df,
                    macro_events=[],
                    news_articles=[]
                )
                
                # We want actionable BUY or SELL setups
                if sig.action in [ACTION_BUY, ACTION_SELL] and sig.is_safe_to_trade_live:
                    score = sig.risk_adjusted_confidence * sig.agreement_fraction
                    if score < 0.4: # Minimum conviction threshold for a scalp
                        continue
                        
                    # Calculate ATR-based targets and stops
                    features = self.feature_engineer.engineer_features_for_horizon(
                        stock_df, index_df, HORIZON_INTRADAY
                    )
                    if features.empty:
                        continue
                        
                    latest = features.iloc[-1]
                    cmp = stock_df["Close"].iloc[-1]
                    atr = latest["atr"] if "atr" in latest and not pd.isna(latest["atr"]) else (cmp * 0.005)
                    
                    if sig.action == ACTION_BUY:
                        sl = cmp - (1.5 * atr)
                        target = cmp + (3.0 * atr)
                    else:
                        sl = cmp + (1.5 * atr)
                        target = cmp - (3.0 * atr)
                        
                    risk = abs(cmp - sl)
                    reward = abs(target - cmp)
                    rr_ratio = reward / risk if risk > 0 else 0
                    
                    opportunities.append(ScalpSetup(
                        symbol=symbol,
                        action=sig.action,
                        entry_price=round(cmp, 2),
                        target_price=round(target, 2),
                        stop_loss=round(sl, 2),
                        confidence=round(score * 100, 1),
                        risk_reward_ratio=round(rr_ratio, 2),
                        reasoning=sig.reasoning
                    ))
            except Exception as e:
                logger.error(f"Error finding scalping ops for {symbol}: {e}")
                
        # Sort by confidence
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        return opportunities[:limit]

if __name__ == "__main__":
    print("\n=== SCALPING ENGINE SELF-TEST ===")
    
    # Simple Mocking to verify the syntax and structure
    class MockDataFetcher:
        def fetch_ohlcv(self, *args, **kwargs):
            import pandas as pd
            return pd.DataFrame({"Close": [1000, 1001, 1002, 1003]}, index=pd.date_range("2026-07-01", periods=4))
        def check_staleness(self, *args):
            return False
            
    class MockPredictor:
        def generate_signal(self, symbol, **kwargs):
            from predictor import PredictionSignal
            return PredictionSignal(
                symbol=symbol, timestamp=pd.Timestamp.now(), horizon=HORIZON_INTRADAY,
                action=ACTION_BUY if symbol == "RELIANCE" else ACTION_SELL, 
                model_predicted_class="UP",
                raw_confidence=0.8, risk_adjusted_confidence=0.8, calibrated_confidence=None,
                agreement_fraction=0.8, downside_summary="", upside_summary="", reasoning=[],
                is_safe_to_trade_live=True
            )
            
    class MockFeatureEngineer:
        def engineer_features_for_horizon(self, *args, **kwargs):
            import pandas as pd
            return pd.DataFrame({"atr": [5, 5, 5, 5]})
            
    engine = ScalpingEngine(MockPredictor(), MockDataFetcher(), MockFeatureEngineer())
    results = engine.find_opportunities(limit=2)
    
    print(f"Found {len(results)} scalping setups.")
    for r in results:
        print(f"{r.symbol}: {r.action} at {r.entry_price} (Target: {r.target_price}, SL: {r.stop_loss})")
        
    assert len(results) > 0
    print("\nSTATUS: PASS")
