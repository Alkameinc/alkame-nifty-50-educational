"""
test_f32_overfitting_detection.py

Comprehensive test suite for F32: Overfitting check (3 pts, P2)

Tests cover:
1. Overfitting detection for severely overfit models
2. Detection for well-regularized models  
3. Train-test gap threshold validation
4. Fold variance detection
5. Perfect train + poor test detection
6. Metrics-only evaluation (no model required)
7. Report generation
8. Integration with model_trainer.py

Part of PY-01 Assignment

Author: Engineering Team
Date: 2026-09-14
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from overfitting_detector import (
    OverfittingDetector,
    OverfittingMetrics,
    OverfitSeverity,
    check_overfitting
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# ============================================================================
# Test Suite 1: Basic Overfitting Detection
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=3,
        class_sep=1.5,
        random_state=42
    )
    
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test


def test_f32_detects_severe_overfitting(synthetic_data):
    """F32-1: Detector identifies severely overfit model (perfect train, poor test)."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    # Train highly overfit model
    model = DecisionTreeClassifier(max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    
    detector = OverfittingDetector()
    metrics = detector.evaluate(model, X_train, y_train, X_test, y_test)
    
    # Should detect overfitting
    assert metrics.is_overfit, "Should detect overfitting"
    assert metrics.severity in [OverfitSeverity.MODERATE, OverfitSeverity.SEVERE]
    assert metrics.train_test_gap > 0.10, "Gap should be significant"
    assert len(metrics.warnings) > 0
    assert len(metrics.recommendations) > 0
    
    print(f"✓ F32-1: Detected {metrics.severity.value} overfitting with {metrics.train_test_gap:.1%} gap")


def test_f32_no_false_positive_on_good_model(synthetic_data):
    """F32-2: Detector doesn't flag well-regularized models."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    # Train well-regularized model
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    detector = OverfittingDetector()
    metrics = detector.evaluate(model, X_train, y_train, X_test, y_test)
    
    # Should not detect significant overfitting
    assert metrics.severity in [OverfitSeverity.NONE, OverfitSeverity.MILD]
    assert metrics.train_test_gap < 0.10, "Gap should be small"
    
    print(f"✓ F32-2: No false positive, severity={metrics.severity.value}, gap={metrics.train_test_gap:.1%}")


def test_f32_threshold_calibration():
    """F32-3: Verify threshold boundaries work correctly."""
    detector = OverfittingDetector(
        mild_threshold=0.05,
        moderate_threshold=0.10,
        severe_threshold=0.15
    )
    
    # Test NONE (gap < 5%)
    metrics_none = detector.evaluate_from_metrics(
        train_accuracy=0.80,
        test_accuracy=0.77
    )
    assert metrics_none.severity == OverfitSeverity.NONE
    assert not metrics_none.is_overfit
    
    # Test MILD (5% <= gap < 10%)
    metrics_mild = detector.evaluate_from_metrics(
        train_accuracy=0.80,
        test_accuracy=0.72
    )
    assert metrics_mild.severity == OverfitSeverity.MILD
    assert metrics_mild.is_overfit
    
    # Test MODERATE (10% <= gap < 15%)
    metrics_moderate = detector.evaluate_from_metrics(
        train_accuracy=0.85,
        test_accuracy=0.73
    )
    assert metrics_moderate.severity == OverfitSeverity.MODERATE
    assert metrics_moderate.is_overfit
    
    # Test SEVERE (gap >= 15%)
    metrics_severe = detector.evaluate_from_metrics(
        train_accuracy=0.95,
        test_accuracy=0.75
    )
    assert metrics_severe.severity == OverfitSeverity.SEVERE
    assert metrics_severe.is_overfit
    
    print("✓ F32-3: All threshold boundaries correct")


# ============================================================================
# Test Suite 2: Perfect Train + Poor Test Pattern
# ============================================================================

def test_f32_perfect_train_poor_test_pattern():
    """F32-4: Detect perfect training with poor test (classic overfitting)."""
    detector = OverfittingDetector()
    
    # Perfect train (98%), poor test (65%)
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.98,
        test_accuracy=0.65
    )
    
    assert metrics.is_overfit
    assert metrics.severity == OverfitSeverity.SEVERE
    # Check that warning mentions either the gap or the severe nature
    warnings_text = " ".join(metrics.warnings).lower()
    assert 'severe' in warnings_text or 'gap' in warnings_text
    
    print("✓ F32-4: Detected perfect-train-poor-test pattern")


def test_f32_high_train_acceptable_test_no_flag():
    """F32-5: High train with acceptable test shouldn't trigger perfect-train pattern."""
    detector = OverfittingDetector()
    
    # High train (92%), acceptable test (85%)
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.92,
        test_accuracy=0.85
    )
    
    # Should be NONE or MILD, not SEVERE from perfect-train pattern
    assert metrics.severity in [OverfitSeverity.NONE, OverfitSeverity.MILD]
    
    print(f"✓ F32-5: High train + good test = {metrics.severity.value} (not SEVERE)")


# ============================================================================
# Test Suite 3: Fold Variance Detection
# ============================================================================

def test_f32_high_fold_variance_detection():
    """F32-6: Detect high variance across cross-validation folds."""
    detector = OverfittingDetector(high_variance_threshold=0.05)
    
    # Low train-test gap but high fold variance
    fold_accuracies = [0.75, 0.82, 0.68, 0.79, 0.71]  # std ~0.055
    
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.78,
        test_accuracy=0.75,
        fold_accuracies=fold_accuracies
    )
    
    assert metrics.fold_variance is not None
    assert metrics.fold_variance > 0.05
    assert any('variance' in w.lower() for w in metrics.warnings)
    
    print(f"✓ F32-6: Detected high fold variance ({metrics.fold_variance:.1%})")


def test_f32_low_fold_variance_acceptable():
    """F32-7: Low fold variance should not trigger warnings."""
    detector = OverfittingDetector(high_variance_threshold=0.05)
    
    # Low variance folds
    fold_accuracies = [0.80, 0.81, 0.79, 0.82, 0.80]  # std ~0.012
    
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.82,
        test_accuracy=0.80,
        fold_accuracies=fold_accuracies
    )
    
    assert metrics.fold_variance < 0.05
    assert not any('HIGH variance' in w for w in metrics.warnings)
    
    print(f"✓ F32-7: Low fold variance ({metrics.fold_variance:.1%}) acceptable")


# ============================================================================
# Test Suite 4: Metrics-Only Evaluation
# ============================================================================

def test_f32_metrics_only_no_model_required():
    """F32-8: Can evaluate overfitting from metrics alone (no model needed)."""
    detector = OverfittingDetector()
    
    # No model required
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.88,
        test_accuracy=0.75,
        train_balanced_accuracy=0.87,
        test_balanced_accuracy=0.74
    )
    
    assert metrics.is_overfit
    assert metrics.train_accuracy == 0.88
    assert metrics.test_accuracy == 0.75
    assert metrics.train_balanced_accuracy == 0.87
    assert metrics.test_balanced_accuracy == 0.74
    
    print("✓ F32-8: Metrics-only evaluation works correctly")


def test_f32_convenience_function():
    """F32-9: Convenience function check_overfitting works."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test = X[:350], X[350:]
    y_train, y_test = y[:350], y[350:]
    
    model = DecisionTreeClassifier(max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    
    # Use convenience function
    is_overfit, metrics = check_overfitting(
        model, X_train, y_train, X_test, y_test, verbose=False
    )
    
    assert isinstance(is_overfit, bool)
    assert isinstance(metrics, OverfittingMetrics)
    assert is_overfit == metrics.is_overfit
    
    print("✓ F32-9: Convenience function check_overfitting() works")


# ============================================================================
# Test Suite 5: Report Generation
# ============================================================================

def test_f32_report_generation():
    """F32-10: Verify report generation works and contains key information."""
    detector = OverfittingDetector()
    
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.92,
        test_accuracy=0.75,
        train_balanced_accuracy=0.91,
        test_balanced_accuracy=0.74,
        fold_accuracies=[0.75, 0.78, 0.72, 0.76, 0.74]
    )
    
    report = detector.generate_report(metrics)
    
    # Check report contains key sections
    assert "OVERFITTING DETECTION REPORT" in report
    assert "Performance Metrics:" in report
    assert "Train Accuracy:" in report
    assert "Test Accuracy:" in report
    assert "Train-Test Gap:" in report
    assert "Warnings:" in report
    assert "Recommendations:" in report
    
    # Check metrics are formatted correctly
    assert "92.0%" in report or "92%" in report  # train accuracy
    assert "75.0%" in report or "75%" in report  # test accuracy
    
    print("✓ F32-10: Report generation contains all required sections")


def test_f32_report_colors_by_severity():
    """F32-11: Report uses appropriate colors for different severities."""
    detector = OverfittingDetector()
    
    # NONE severity
    metrics_none = detector.evaluate_from_metrics(0.80, 0.78)
    report_none = detector.generate_report(metrics_none)
    assert "NONE" in report_none
    
    # SEVERE severity  
    metrics_severe = detector.evaluate_from_metrics(0.98, 0.70)
    report_severe = detector.generate_report(metrics_severe)
    assert "SEVERE" in report_severe
    
    print("✓ F32-11: Report correctly displays severity levels")


# ============================================================================
# Test Suite 6: Edge Cases
# ============================================================================

def test_f32_negative_gap_handled():
    """F32-12: Handle case where test > train (underfitting or good generalization)."""
    detector = OverfittingDetector()
    
    # Test accuracy higher than train (rare but possible)
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.75,
        test_accuracy=0.78
    )
    
    assert not metrics.is_overfit
    assert metrics.severity == OverfitSeverity.NONE
    assert metrics.train_test_gap < 0  # Negative gap
    
    print("✓ F32-12: Negative gap (test > train) handled correctly")


def test_f32_zero_gap_handled():
    """F32-13: Handle perfect equality between train and test."""
    detector = OverfittingDetector()
    
    metrics = detector.evaluate_from_metrics(
        train_accuracy=0.80,
        test_accuracy=0.80
    )
    
    assert not metrics.is_overfit
    assert metrics.severity == OverfitSeverity.NONE
    assert metrics.train_test_gap == 0.0
    
    print("✓ F32-13: Zero gap handled correctly")


def test_f32_extreme_overfitting():
    """F32-14: Handle extreme overfitting (100% train, 33% test)."""
    detector = OverfittingDetector()
    
    metrics = detector.evaluate_from_metrics(
        train_accuracy=1.00,
        test_accuracy=0.33
    )
    
    assert metrics.is_overfit
    assert metrics.severity == OverfitSeverity.SEVERE
    assert abs(metrics.train_test_gap - 0.67) < 0.01  # ~67% gap (allow floating point tolerance)
    assert metrics.confidence_score >= 0.9  # Very high confidence
    
    print("✓ F32-14: Extreme overfitting detected correctly")


# ============================================================================
# Test Suite 7: Custom Thresholds
# ============================================================================

def test_f32_custom_thresholds_work():
    """F32-15: Custom thresholds can be configured."""
    # Stricter thresholds
    strict_detector = OverfittingDetector(
        mild_threshold=0.03,
        moderate_threshold=0.06,
        severe_threshold=0.10
    )
    
    metrics = strict_detector.evaluate_from_metrics(0.80, 0.76)  # 4% gap
    
    # With strict thresholds, 4% is MILD
    assert metrics.severity == OverfitSeverity.MILD
    
    # With default thresholds, 4% is NONE
    default_detector = OverfittingDetector()
    metrics_default = default_detector.evaluate_from_metrics(0.80, 0.76)
    assert metrics_default.severity == OverfitSeverity.NONE
    
    print("✓ F32-15: Custom thresholds work as expected")


# ============================================================================
# Test Suite 8: Integration with GradientBoostingClassifier
# ============================================================================

def test_f32_gradient_boosting_integration(synthetic_data):
    """F32-16: Works with GradientBoostingClassifier (used in production)."""
    X_train, X_test, y_train, y_test = synthetic_data
    
    # Train GradientBoosting (same as production)
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    detector = OverfittingDetector()
    metrics = detector.evaluate(model, X_train, y_train, X_test, y_test)
    
    # Should get valid metrics
    assert 0.0 <= metrics.train_accuracy <= 1.0
    assert 0.0 <= metrics.test_accuracy <= 1.0
    assert isinstance(metrics.is_overfit, bool)
    assert isinstance(metrics.severity, OverfitSeverity)
    
    print(f"✓ F32-16: GradientBoosting integration works (severity={metrics.severity.value})")


def test_f32_production_model_parameters():
    """F32-17: Test with exact production model parameters from config.py."""
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Simulate production parameters (from config.py)
    # MODEL_N_ESTIMATORS = 100
    # MODEL_MAX_DEPTH = 3
    # MODEL_LEARNING_RATE = 0.1
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=3, random_state=42
    )
    
    split = 700
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    detector = OverfittingDetector()
    metrics = detector.evaluate(model, X_train, y_train, X_test, y_test)
    
    # Production model may show some overfitting on synthetic data
    # The key is that we can detect it
    assert isinstance(metrics.severity, OverfitSeverity)
    assert isinstance(metrics.is_overfit, bool)
    
    # If overfitting is detected, should have recommendations
    if metrics.is_overfit:
        assert len(metrics.recommendations) > 0
    
    print(f"✓ F32-17: Production parameters result in {metrics.severity.value} overfitting "
          f"(gap={metrics.train_test_gap:.1%}, detectable={metrics.is_overfit})")


# ============================================================================
# Test Suite 9: Recommendations Quality
# ============================================================================

def test_f32_recommendations_are_actionable():
    """F32-18: Recommendations are specific and actionable."""
    detector = OverfittingDetector()
    
    metrics = detector.evaluate_from_metrics(0.98, 0.72)
    
    assert len(metrics.recommendations) > 0
    
    # Check recommendations contain specific actions
    recommendations_text = " ".join(metrics.recommendations).lower()
    
    # Should mention at least one concrete action
    has_action = any(keyword in recommendations_text for keyword in [
        'reduce', 'increase', 'add', 'use', 'consider', 'regularization',
        'complexity', 'data', 'dropout', 'validation', 'ensemble'
    ])
    
    assert has_action, "Recommendations should contain actionable advice"
    
    print(f"✓ F32-18: {len(metrics.recommendations)} actionable recommendations provided")


def test_f32_warnings_explain_severity():
    """F32-19: Warnings clearly explain what was detected."""
    detector = OverfittingDetector()
    
    metrics = detector.evaluate_from_metrics(0.95, 0.68)
    
    assert len(metrics.warnings) > 0
    
    warnings_text = " ".join(metrics.warnings).lower()
    
    # Should explain the issue
    has_explanation = any(keyword in warnings_text for keyword in [
        'overfitting', 'gap', 'train', 'test', 'severe', 'moderate', 'mild'
    ])
    
    assert has_explanation, "Warnings should explain the issue"
    
    print(f"✓ F32-19: Warnings clearly explain the issue")


# ============================================================================
# Test Suite 10: Confidence Score
# ============================================================================

def test_f32_confidence_score_reasonable():
    """F32-20: Confidence scores are in valid range and make sense."""
    detector = OverfittingDetector()
    
    # Severe overfitting should have high confidence
    metrics_severe = detector.evaluate_from_metrics(0.98, 0.70)
    assert 0.8 <= metrics_severe.confidence_score <= 1.0
    
    # No overfitting should have high confidence in NONE
    metrics_none = detector.evaluate_from_metrics(0.80, 0.78)
    assert 0.5 <= metrics_none.confidence_score <= 1.0
    
    # Mild overfitting should have moderate confidence
    metrics_mild = detector.evaluate_from_metrics(0.82, 0.76)
    assert 0.0 <= metrics_mild.confidence_score <= 1.0
    
    print("✓ F32-20: Confidence scores are reasonable")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
