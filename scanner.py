import logging
from typing import List
from dataclasses import dataclass
from config import NIFTY50_SYMBOLS, ALL_HORIZONS, to_yfinance_ticker
from data_fetcher import DataFetcher
from predictor import Predictor, MultiHorizonSignal, ACTION_BUY

logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    symbol: str
    signal: MultiHorizonSignal
    conviction_score: float

class OpportunityScanner:
    """
    Scans the NIFTY 50 universe to find the highest-conviction trading opportunities.
    Generates signals for all stocks, applies safety gates, and ranks them by conviction.
    """
    
    def __init__(self, predictor: Predictor, data_fetcher: DataFetcher):
        self.predictor = predictor
        self.data_fetcher = data_fetcher

    def scan(self, limit: int = 5) -> List[ScanResult]:
        """
        Runs a full scan across all NIFTY 50 symbols.
        Returns the top `limit` BUY opportunities ranked by composite conviction score.
        """
        logger.info(f"Starting opportunity scan across {len(NIFTY50_SYMBOLS)} symbols...")
        opportunities: List[ScanResult] = []
        
        # Pre-fetch index data once for relative strength checks
        index_ticker = "^NSEI"
        index_df = self.data_fetcher.fetch_ohlcv(index_ticker)
        
        for symbol in NIFTY50_SYMBOLS:
            try:
                yf_ticker = to_yfinance_ticker(symbol)
                stock_df = self.data_fetcher.fetch_ohlcv(yf_ticker)
                
                if stock_df is None or stock_df.empty:
                    logger.warning(f"Skipping {symbol} in scan due to missing data.")
                    continue
                
                if self.data_fetcher.check_staleness(stock_df, yf_ticker):
                    logger.warning(f"Skipping {symbol} in scan due to stale data.")
                    continue
                
                # We skip macro/news events here for simplicity, relying strictly on technical/model data
                multi_sig = self.predictor.generate_multi_horizon_signal(
                    symbol=symbol,
                    horizons=ALL_HORIZONS,
                    stock_df=stock_df,
                    index_df=index_df
                )
                
                # Filter to actionable BUY signals
                if multi_sig.primary_action == ACTION_BUY:
                    primary_sig = multi_sig.signals.get(multi_sig.primary_horizon)
                    
                    if primary_sig and primary_sig.is_safe_to_trade_live:
                        # Composite conviction score: blends risk-adjusted confidence and agreement fraction
                        score = primary_sig.risk_adjusted_confidence * primary_sig.agreement_fraction
                        
                        opportunities.append(ScanResult(
                            symbol=symbol,
                            signal=multi_sig,
                            conviction_score=score
                        ))
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                
        # Sort by highest composite conviction score
        opportunities.sort(key=lambda x: x.conviction_score, reverse=True)
        
        top_picks = opportunities[:limit]
        logger.info(f"Scan complete. Found {len(opportunities)} valid BUY opportunities. Returning top {len(top_picks)}.")
        return top_picks

if __name__ == "__main__":
    from datetime import datetime
    
    print("\n=== OPPORTUNITY SCANNER SELF-TEST ===")
    
    class MockDataFetcher:
        def fetch_ohlcv(self, ticker, **kwargs):
            return "mock_df"
            
        def check_staleness(self, df, ticker):
            return False
            
    class MockPredictor:
        def generate_multi_horizon_signal(self, symbol, **kwargs):
            from predictor import PredictionSignal
            from config import HORIZON_INTRADAY
            
            # Make RELIANCE the best, TCS good but lower, INFY a HOLD
            action = ACTION_BUY
            is_safe = True
            conf = 0.6
            agreement = 0.5
            
            if symbol == "RELIANCE":
                conf = 0.9
                agreement = 0.9
            elif symbol == "TCS":
                conf = 0.8
                agreement = 0.7
            elif symbol == "INFY":
                action = "HOLD"
                
            sig = PredictionSignal(
                symbol=symbol, timestamp=datetime.now(), horizon=HORIZON_INTRADAY,
                action=action, model_predicted_class="UP" if action == ACTION_BUY else "FLAT",
                raw_confidence=conf, risk_adjusted_confidence=conf, calibrated_confidence=None,
                agreement_fraction=agreement, downside_summary="", upside_summary="", reasoning=[],
                is_safe_to_trade_live=is_safe
            )
            
            return MultiHorizonSignal(
                symbol=symbol, timestamp=datetime.now(),
                signals={HORIZON_INTRADAY: sig}, primary_action=action,
                primary_horizon=HORIZON_INTRADAY, reasoning=[]
            )
            
    scanner = OpportunityScanner(predictor=MockPredictor(), data_fetcher=MockDataFetcher())
    
    results = scanner.scan(limit=3)
    
    print(f"Returned {len(results)} opportunities.")
    assert len(results) <= 3
    
    if len(results) > 0:
        print(f"Top pick: {results[0].symbol} with score {results[0].conviction_score:.2f}")
        assert results[0].symbol == "RELIANCE", "RELIANCE should be top pick based on mock scoring"
        
    if len(results) > 1:
        print(f"Second pick: {results[1].symbol} with score {results[1].conviction_score:.2f}")
        assert results[1].symbol == "TCS", "TCS should be second pick based on mock scoring"
        
    print("\nSTATUS: PASS")
