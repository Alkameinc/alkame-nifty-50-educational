import logging
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from config import NIFTY50_SYMBOLS, ALL_HORIZONS, to_yfinance_ticker
from data_fetcher import DataFetcher
from predictor import Predictor, MultiHorizonSignal, ACTION_BUY

logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    symbol: str
    signal: MultiHorizonSignal
    conviction_score: float

@dataclass
class ScanRejection:
    symbol: str
    reason: str       # human-readable reason
    category: str     # machine-readable: "stale_data", "missing_data", "hold", "safety_gate", "data_error"

@dataclass
class ScanSummary:
    scanned: int = 0
    actionable: int = 0
    hold: int = 0
    stale_data: int = 0
    missing_data: int = 0
    safety_gate: int = 0
    data_error: int = 0
    rejections: List[ScanRejection] = field(default_factory=list)
    opportunities: List[ScanResult] = field(default_factory=list)

    def add_rejection(self, symbol: str, reason: str, category: str):
        self.rejections.append(ScanRejection(symbol=symbol, reason=reason, category=category))
        if category == "stale_data":
            self.stale_data += 1
        elif category == "missing_data":
            self.missing_data += 1
        elif category == "hold":
            self.hold += 1
        elif category == "safety_gate":
            self.safety_gate += 1
        elif category == "data_error":
            self.data_error += 1


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
        Use scan_with_summary() for full structured results including rejections.
        """
        summary = self.scan_with_summary(limit=limit)
        return summary.opportunities

    def scan_with_summary(self, limit: int = 5) -> ScanSummary:
        """
        Runs a full scan and returns a ScanSummary with both opportunities and rejection details.
        """
        logger.info(f"Starting opportunity scan across {len(NIFTY50_SYMBOLS)} symbols...")
        summary = ScanSummary(scanned=len(NIFTY50_SYMBOLS))
        
        # Pre-fetch index data once for relative strength checks
        index_ticker = "^NSEI"
        index_df = self.data_fetcher.fetch_ohlcv(index_ticker)
        
        for symbol in NIFTY50_SYMBOLS:
            try:
                yf_ticker = to_yfinance_ticker(symbol)
                stock_df = self.data_fetcher.fetch_ohlcv(yf_ticker)
                
                if stock_df is None or stock_df.empty:
                    summary.add_rejection(symbol, f"No data available for {symbol}", "missing_data")
                    continue
                
                if self.data_fetcher.check_staleness(stock_df, yf_ticker):
                    summary.add_rejection(symbol, f"Data for {symbol} is stale", "stale_data")
                    continue
                
                # We skip macro/news events here for simplicity, relying strictly on technical/model data
                multi_sig = self.predictor.generate_multi_horizon_signal(
                    symbol=symbol,
                    horizons=ALL_HORIZONS,
                    stock_df=stock_df,
                    index_df=index_df
                )
                
                # Filter to actionable BUY signals
                if multi_sig.primary_action != ACTION_BUY:
                    summary.add_rejection(symbol, f"{symbol} signal is {multi_sig.primary_action}, not BUY", "hold")
                    continue

                primary_sig = multi_sig.signals.get(multi_sig.primary_horizon)
                
                if not primary_sig or not primary_sig.is_safe_to_trade_live:
                    reason = f"{symbol} not safe to trade live"
                    if primary_sig and primary_sig.suppression_reasons:
                        reason += f": {'; '.join(primary_sig.suppression_reasons[:2])}"
                    summary.add_rejection(symbol, reason, "safety_gate")
                    continue

                # Composite conviction score: blends risk-adjusted confidence and agreement fraction
                score = primary_sig.risk_adjusted_confidence * primary_sig.agreement_fraction
                
                summary.opportunities.append(ScanResult(
                    symbol=symbol,
                    signal=multi_sig,
                    conviction_score=score
                ))
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                summary.add_rejection(symbol, f"Error scanning {symbol}: {e}", "data_error")
                
        # Sort by highest composite conviction score
        summary.opportunities.sort(key=lambda x: x.conviction_score, reverse=True)
        summary.opportunities = summary.opportunities[:limit]
        summary.actionable = len(summary.opportunities)
        
        logger.info(f"Scan complete: {summary.actionable} actionable, "
                     f"{summary.hold} hold, {summary.stale_data} stale, "
                     f"{summary.missing_data} missing, {summary.safety_gate} safety-gated, "
                     f"{summary.data_error} errors")
        return summary

if __name__ == "__main__":
    from datetime import datetime
    
    print("\n=== OPPORTUNITY SCANNER SELF-TEST ===")
    
    class MockDataFetcher:
        def fetch_ohlcv(self, ticker, **kwargs):
            import pandas as pd
            return pd.DataFrame({'Close': [100]})
            
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
                model_version="UNKNOWN", feature_version="UNKNOWN",
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
    
    # Test scan() backward compatibility
    results = scanner.scan(limit=3)
    
    print(f"Returned {len(results)} opportunities.")
    assert len(results) <= 3
    
    if len(results) > 0:
        print(f"Top pick: {results[0].symbol} with score {results[0].conviction_score:.2f}")
        assert results[0].symbol == "RELIANCE", "RELIANCE should be top pick based on mock scoring"
        
    if len(results) > 1:
        print(f"Second pick: {results[1].symbol} with score {results[1].conviction_score:.2f}")
        assert results[1].symbol == "TCS", "TCS should be second pick based on mock scoring"
    
    # Test scan_with_summary()
    summary = scanner.scan_with_summary(limit=3)
    print(f"\nScan summary: scanned={summary.scanned}, actionable={summary.actionable}, "
          f"hold={summary.hold}, stale={summary.stale_data}, "
          f"missing={summary.missing_data}, safety_gate={summary.safety_gate}, "
          f"errors={summary.data_error}")
    print(f"Rejections: {len(summary.rejections)}")
    assert summary.scanned > 0
    assert summary.hold > 0, "INFY should be a HOLD rejection"
    # Verify INFY was tracked as a hold rejection
    infy_rejections = [r for r in summary.rejections if r.symbol == "INFY"]
    assert len(infy_rejections) == 1 and infy_rejections[0].category == "hold", \
        f"INFY should be rejected as hold, got: {infy_rejections}"
    
    print("\nSTATUS: PASS")
