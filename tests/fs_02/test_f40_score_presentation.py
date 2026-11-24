import pytest
from predictor import PredictionSignal
from scalping import ScalpSetup
from api import humanize_reasoning
from narrative_builder import NarrativeBuilder
from tests.fs_02.fixtures.prediction_signal import get_specimen_signal, get_specimen_multi_signal

def test_f40_calibrated_vs_raw_score_distinction():
    sig = get_specimen_signal()
    sig.raw_confidence = 0.8
    sig.risk_adjusted_confidence = 0.6
    sig.agreement_fraction = 0.5
    sig.calibrated_confidence = 0.55
    sig.suppressed = False
    
    multi = get_specimen_multi_signal()
    multi.signals["INTRADAY"] = sig
    
    narrative = NarrativeBuilder().build_stock_narrative(multi)
    assert "calibrated probability 55.0%" in narrative
    assert "80.0% confidence" not in narrative

def test_f40_calibrated_none_vs_zero():
    sig_none = get_specimen_signal()
    sig_none.calibrated_confidence = None
    sig_none.suppressed = False
    
    sig_zero = get_specimen_signal()
    sig_zero.calibrated_confidence = 0.0
    sig_zero.suppressed = False
    
    multi_none = get_specimen_multi_signal()
    multi_none.signals["INTRADAY"] = sig_none
    
    multi_zero = get_specimen_multi_signal()
    multi_zero.signals["INTRADAY"] = sig_zero
    
    narrative_none = NarrativeBuilder().build_stock_narrative(multi_none)
    narrative_zero = NarrativeBuilder().build_stock_narrative(multi_zero)
    
    assert "calibrated probability: unavailable" in narrative_none
    assert "calibrated probability 0.0%" in narrative_zero

def test_f40_humanize_reasoning_strips_raw_score_leakage():
    reasons = [
        "Ensemble model lean: UP (raw model score 0.80, model agreement 75%).",
        "Global risk level: NORMAL"
    ]
    humanized = humanize_reasoning(reasons)
    
    assert humanized[0] == "The AI models are predicting an upward trend."
    assert "80%" not in humanized[0]
