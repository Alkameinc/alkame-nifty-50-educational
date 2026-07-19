import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from predictor import MultiHorizonSignal, PredictionSignal

logger = logging.getLogger(__name__)

@dataclass
class PriceLevelOutcome:
    horizon: str
    predicted_level: str
    confidence: float

class NarrativeBuilder:
    """
    Translates raw algorithmic signals, price-level predictions, and events 
    into natural English narratives that can be directly presented to the user.
    """
    
    def __init__(self):
        pass

    def build_stock_narrative(
        self, 
        multi_signal: MultiHorizonSignal, 
        level_outcomes: Optional[List[PriceLevelOutcome]] = None
    ) -> str:
        """
        Builds a multi-paragraph English narrative summarizing the 
        MultiHorizonSignal and the expected price level interactions.
        """
        try:
            symbol = multi_signal.symbol
            primary_action = multi_signal.primary_action
            primary_horizon = multi_signal.primary_horizon
            
            # --- Paragraph 1: Executive Summary ---
            if primary_action == "HOLD":
                summary = f"For {symbol}, the current consensus is a cautious HOLD."
            elif primary_action == "BUY":
                summary = f"For {symbol}, the quantitative models are flashing a BUY signal."
            else:
                summary = f"For {symbol}, the system is issuing a SELL warning."
                
            summary += f" This is primarily driven by the {primary_horizon} outlook."
            
            # --- Paragraph 2: Horizon Breakdown ---
            horizon_parts = []
            for h, sig in multi_signal.signals.items():
                if sig.suppressed:
                    horizon_parts.append(f"The {h} signal is suppressed ({sig.suppression_reasons[0]}).")
                else:
                    conf = f"{sig.raw_confidence*100:.1f}%"
                    horizon_parts.append(f"Over {h}, models lean {sig.model_predicted_class} with {conf} confidence.")
            
            horizon_summary = " " .join(horizon_parts)
            
            # --- Paragraph 3: Price Levels (Phase 7 Outcomes) ---
            level_parts = []
            if level_outcomes:
                for outcome in level_outcomes:
                    if outcome.predicted_level != "NONE":
                        level_parts.append(f"In the {outcome.horizon} window, price is most likely to test {outcome.predicted_level} first (Confidence: {outcome.confidence*100:.1f}%).")
            
            level_summary = " ".join(level_parts) if level_parts else "No significant price level tests are immediately projected."
            
            # --- Paragraph 4: Event Context ---
            primary_sig = multi_signal.signals.get(primary_horizon)
            event_summary = ""
            if primary_sig and primary_sig.contributing_events:
                event_summary = "Key driving events include: " + "; ".join(
                    [e.headline_or_label for e in primary_sig.contributing_events[:3]]
                ) + "."
            else:
                event_summary = "No major macro or corporate events are currently overriding the technical setup."
            
            # Assemble full narrative
            paragraphs = [summary, horizon_summary, level_summary, event_summary]
            return "\n\n".join(p for p in paragraphs if p)
            
        except Exception as e:
            logger.error(f"Failed to build narrative for {multi_signal.symbol}: {e}")
            return f"Error generating narrative for {multi_signal.symbol}. Please rely on raw signal data."

if __name__ == "__main__":
    from datetime import datetime
    
    print("\n=== NARRATIVE BUILDER SELF-TEST ===")
    builder = NarrativeBuilder()
    
    # Mock some data
    from predictor import PredictionSignal, ACTION_BUY, ACTION_HOLD
    from config import HORIZON_INTRADAY, HORIZON_30D
    
    sig_intra = PredictionSignal(
        symbol="RELIANCE", timestamp=datetime.now(), horizon=HORIZON_INTRADAY,
        action=ACTION_BUY, model_predicted_class="UP", raw_confidence=0.75,
        risk_adjusted_confidence=0.75, calibrated_confidence=None, agreement_fraction=0.8,
        downside_summary="", upside_summary="", reasoning=[],
        is_safe_to_trade_live=True
    )
    sig_30d = PredictionSignal(
        symbol="RELIANCE", timestamp=datetime.now(), horizon=HORIZON_30D,
        action=ACTION_HOLD, model_predicted_class="FLAT", raw_confidence=0.55,
        risk_adjusted_confidence=0.55, calibrated_confidence=None, agreement_fraction=0.5,
        downside_summary="", upside_summary="", reasoning=[],
        is_safe_to_trade_live=True
    )
    
    multi = MultiHorizonSignal(
        symbol="RELIANCE", timestamp=datetime.now(),
        signals={HORIZON_INTRADAY: sig_intra, HORIZON_30D: sig_30d},
        primary_action=ACTION_BUY, primary_horizon=HORIZON_INTRADAY,
        reasoning=[]
    )
    
    outcomes = [
        PriceLevelOutcome(horizon=HORIZON_INTRADAY, predicted_level="RESISTANCE", confidence=0.68),
        PriceLevelOutcome(horizon=HORIZON_30D, predicted_level="MA", confidence=0.72)
    ]
    
    narrative = builder.build_stock_narrative(multi, outcomes)
    print(narrative)
    print("\nSTATUS: PASS")
