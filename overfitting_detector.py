"""
overfitting_detector.py

Detects overfitting in ML models by comparing train vs test performance,
analyzing variance across validation folds, and checking for suspicious patterns.

Part of PY-01 F32: Overfitting check (3 pts, P2)

Overfitting indicators:
1. Large train-test performance gap (train >> test)
2. Near-perfect training accuracy with poor test accuracy
3. High variance across cross-validation folds
4. Degrading performance on out-of-sample data over time

Author: Engineering Team
Date: 2026-09-14
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class OverfitSeverity(Enum):
    """Severity levels for overfitting detection."""
    NONE = "NONE"  # No overfitting detected
    MILD = "MILD"  # Minor overfitting, acceptable for production
    MODERATE = "MODERATE"  # Significant overfitting, needs attention
    SEVERE = "SEVERE"  # Serious overfitting, not suitable for production


@dataclass
class OverfittingMetrics:
    """
    Comprehensive overfitting detection metrics.
    """
    # Basic metrics
    train_accuracy: float
    test_accuracy: float
    train_test_gap: float  # train - test
    
    # Additional metrics
    train_balanced_accuracy: Optional[float] = None
    test_balanced_accuracy: Optional[float] = None
    train_f1_macro: Optional[float] = None
    test_f1_macro: Optional[float] = None
    
    # Overfitting indicators
    severity: OverfitSeverity = OverfitSeverity.NONE
    is_overfit: bool = False
    confidence_score: float = 0.0  # 0-1, confidence in overfitting assessment
    
    # Fold variance (if cross-validation used)
    fold_variance: Optional[float] = None
    fold_accuracies: Optional[List[float]] = None
    
    # Warnings and recommendations
    warnings: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []


class OverfittingDetector:
    """
    Detects overfitting in trained ML models.
    
    Thresholds (configurable):
    - MILD: train-test gap > 5%
    - MODERATE: train-test gap > 10%
    - SEVERE: train-test gap > 15% OR train > 95% with test < 70%
    
    Example:
        detector = OverfittingDetector()
        metrics = detector.evaluate(model, X_train, y_train, X_test, y_test)
        
        if metrics.is_overfit:
            print(f"Overfitting detected: {metrics.severity}")
            for warning in metrics.warnings:
                print(f"  - {warning}")
    """
    
    # Thresholds for overfitting detection
    MILD_THRESHOLD = 0.05  # 5% gap
    MODERATE_THRESHOLD = 0.10  # 10% gap
    SEVERE_THRESHOLD = 0.15  # 15% gap
    
    # Perfect training with poor test threshold
    PERFECT_TRAIN_THRESHOLD = 0.95  # 95% training accuracy
    POOR_TEST_THRESHOLD = 0.70  # 70% test accuracy
    
    # Fold variance threshold (for cross-validation)
    HIGH_VARIANCE_THRESHOLD = 0.05  # 5% std dev across folds
    
    def __init__(
        self,
        mild_threshold: float = MILD_THRESHOLD,
        moderate_threshold: float = MODERATE_THRESHOLD,
        severe_threshold: float = SEVERE_THRESHOLD,
        perfect_train_threshold: float = PERFECT_TRAIN_THRESHOLD,
        poor_test_threshold: float = POOR_TEST_THRESHOLD,
        high_variance_threshold: float = HIGH_VARIANCE_THRESHOLD
    ):
        """
        Initialize overfitting detector with custom thresholds.
        
        Args:
            mild_threshold: Train-test gap for mild overfitting (default 0.05)
            moderate_threshold: Train-test gap for moderate overfitting (default 0.10)
            severe_threshold: Train-test gap for severe overfitting (default 0.15)
            perfect_train_threshold: Training accuracy threshold for "perfect" (default 0.95)
            poor_test_threshold: Test accuracy threshold for "poor" (default 0.70)
            high_variance_threshold: Std dev threshold for high variance (default 0.05)
        """
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold
        self.perfect_train_threshold = perfect_train_threshold
        self.poor_test_threshold = poor_test_threshold
        self.high_variance_threshold = high_variance_threshold
    
    def evaluate(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fold_accuracies: Optional[List[float]] = None
    ) -> OverfittingMetrics:
        """
        Evaluate model for overfitting.
        
        Args:
            model: Trained sklearn model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            fold_accuracies: Optional list of cross-validation fold accuracies
        
        Returns:
            OverfittingMetrics with comprehensive overfitting assessment
        """
        # Compute basic metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        gap = train_acc - test_acc
        
        # Additional metrics
        train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        # Initialize result
        metrics = OverfittingMetrics(
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            train_test_gap=gap,
            train_balanced_accuracy=train_bal_acc,
            test_balanced_accuracy=test_bal_acc,
            train_f1_macro=train_f1,
            test_f1_macro=test_f1,
            fold_accuracies=fold_accuracies
        )
        
        # Detect overfitting
        self._assess_overfitting(metrics)
        
        return metrics
    
    def _assess_overfitting(self, metrics: OverfittingMetrics) -> None:
        """
        Assess overfitting severity and populate warnings/recommendations.
        
        Modifies metrics in-place.
        """
        gap = metrics.train_test_gap
        train_acc = metrics.train_accuracy
        test_acc = metrics.test_accuracy
        
        # Check for severe overfitting patterns
        if gap >= self.severe_threshold:
            metrics.severity = OverfitSeverity.SEVERE
            metrics.is_overfit = True
            metrics.confidence_score = min(1.0, gap / 0.20)  # Normalize to 0-1
            metrics.warnings.append(
                f"SEVERE overfitting: train-test gap = {gap:.1%} (threshold: {self.severe_threshold:.1%})"
            )
            metrics.recommendations.extend([
                "Reduce model complexity (lower max_depth, fewer estimators)",
                "Increase regularization",
                "Add more training data",
                "Use dropout or early stopping",
                "Consider simpler model architecture"
            ])
        
        # Check for perfect train + poor test pattern
        elif train_acc >= self.perfect_train_threshold and test_acc < self.poor_test_threshold:
            metrics.severity = OverfitSeverity.SEVERE
            metrics.is_overfit = True
            metrics.confidence_score = 0.9
            metrics.warnings.append(
                f"SEVERE overfitting: near-perfect train ({train_acc:.1%}) with poor test ({test_acc:.1%})"
            )
            metrics.recommendations.extend([
                "Model is memorizing training data",
                "Drastically reduce model complexity",
                "Use ensemble methods with diverse models",
                "Verify data quality and label correctness"
            ])
        
        # Moderate overfitting
        elif gap >= self.moderate_threshold:
            metrics.severity = OverfitSeverity.MODERATE
            metrics.is_overfit = True
            metrics.confidence_score = min(0.8, gap / 0.15)
            metrics.warnings.append(
                f"MODERATE overfitting: train-test gap = {gap:.1%} (threshold: {self.moderate_threshold:.1%})"
            )
            metrics.recommendations.extend([
                "Consider reducing model complexity",
                "Increase regularization strength",
                "Use cross-validation for hyperparameter tuning",
                "Monitor performance on fresh data"
            ])
        
        # Mild overfitting
        elif gap >= self.mild_threshold:
            metrics.severity = OverfitSeverity.MILD
            metrics.is_overfit = True
            metrics.confidence_score = min(0.6, gap / 0.10)
            metrics.warnings.append(
                f"MILD overfitting: train-test gap = {gap:.1%} (threshold: {self.mild_threshold:.1%})"
            )
            metrics.recommendations.append(
                "Monitor model performance on fresh data. "
                "Mild overfitting is acceptable if test performance is strong."
            )
        
        # No overfitting detected
        else:
            metrics.severity = OverfitSeverity.NONE
            metrics.is_overfit = False
            metrics.confidence_score = 1.0 - (gap / self.mild_threshold) if gap > 0 else 1.0
            metrics.warnings.append(
                f"No overfitting detected: train-test gap = {gap:.1%} (threshold: {self.mild_threshold:.1%})"
            )
        
        # Check fold variance if provided
        if metrics.fold_accuracies and len(metrics.fold_accuracies) > 1:
            fold_std = np.std(metrics.fold_accuracies)
            metrics.fold_variance = fold_std
            
            if fold_std >= self.high_variance_threshold:
                if metrics.severity == OverfitSeverity.NONE:
                    metrics.severity = OverfitSeverity.MILD
                    metrics.is_overfit = True
                
                metrics.warnings.append(
                    f"HIGH variance across folds: std = {fold_std:.1%} (threshold: {self.high_variance_threshold:.1%})"
                )
                metrics.recommendations.append(
                    "High fold variance suggests model instability. "
                    "Consider more data, simpler model, or ensemble methods."
                )
    
    def evaluate_from_metrics(
        self,
        train_accuracy: float,
        test_accuracy: float,
        train_balanced_accuracy: Optional[float] = None,
        test_balanced_accuracy: Optional[float] = None,
        fold_accuracies: Optional[List[float]] = None
    ) -> OverfittingMetrics:
        """
        Evaluate overfitting from pre-computed metrics (no model required).
        
        Useful for analyzing saved model metrics or historical results.
        
        Args:
            train_accuracy: Training accuracy
            test_accuracy: Test accuracy
            train_balanced_accuracy: Optional training balanced accuracy
            test_balanced_accuracy: Optional test balanced accuracy
            fold_accuracies: Optional list of fold accuracies
        
        Returns:
            OverfittingMetrics with overfitting assessment
        """
        gap = train_accuracy - test_accuracy
        
        metrics = OverfittingMetrics(
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            train_test_gap=gap,
            train_balanced_accuracy=train_balanced_accuracy,
            test_balanced_accuracy=test_balanced_accuracy,
            fold_accuracies=fold_accuracies
        )
        
        self._assess_overfitting(metrics)
        
        return metrics
    
    def generate_report(self, metrics: OverfittingMetrics) -> str:
        """
        Generate human-readable overfitting report.
        
        Args:
            metrics: OverfittingMetrics from evaluation
        
        Returns:
            Formatted report string
        """
        severity_color = {
            OverfitSeverity.NONE: GREEN,
            OverfitSeverity.MILD: YELLOW,
            OverfitSeverity.MODERATE: YELLOW,
            OverfitSeverity.SEVERE: RED
        }.get(metrics.severity, RESET)
        
        lines = [
            f"\n{'=' * 80}",
            f"{BOLD}OVERFITTING DETECTION REPORT{RESET}",
            f"{'=' * 80}\n",
            f"Severity: {severity_color}{BOLD}{metrics.severity.value}{RESET}",
            f"Overfitting Detected: {severity_color}{'YES' if metrics.is_overfit else 'NO'}{RESET}",
            f"Confidence: {metrics.confidence_score:.1%}\n",
            f"{BOLD}Performance Metrics:{RESET}",
            f"  Train Accuracy:        {metrics.train_accuracy:.1%}",
            f"  Test Accuracy:         {metrics.test_accuracy:.1%}",
            f"  Train-Test Gap:        {severity_color}{metrics.train_test_gap:+.1%}{RESET}",
        ]
        
        if metrics.train_balanced_accuracy is not None:
            lines.extend([
                f"\n{BOLD}Balanced Accuracy:{RESET}",
                f"  Train:                 {metrics.train_balanced_accuracy:.1%}",
                f"  Test:                  {metrics.test_balanced_accuracy:.1%}",
                f"  Gap:                   {severity_color}{metrics.train_balanced_accuracy - metrics.test_balanced_accuracy:+.1%}{RESET}"
            ])
        
        if metrics.train_f1_macro is not None:
            lines.extend([
                f"\n{BOLD}F1 Macro:{RESET}",
                f"  Train:                 {metrics.train_f1_macro:.1%}",
                f"  Test:                  {metrics.test_f1_macro:.1%}",
                f"  Gap:                   {severity_color}{metrics.train_f1_macro - metrics.test_f1_macro:+.1%}{RESET}"
            ])
        
        if metrics.fold_accuracies:
            fold_mean = np.mean(metrics.fold_accuracies)
            fold_std = np.std(metrics.fold_accuracies)
            fold_min = np.min(metrics.fold_accuracies)
            fold_max = np.max(metrics.fold_accuracies)
            
            lines.extend([
                f"\n{BOLD}Cross-Validation Folds:{RESET}",
                f"  Fold Count:            {len(metrics.fold_accuracies)}",
                f"  Mean Accuracy:         {fold_mean:.1%}",
                f"  Std Dev:               {fold_std:.1%}",
                f"  Min:                   {fold_min:.1%}",
                f"  Max:                   {fold_max:.1%}",
                f"  Range:                 {fold_max - fold_min:.1%}"
            ])
        
        if metrics.warnings:
            lines.extend([
                f"\n{BOLD}Warnings:{RESET}"
            ])
            for warning in metrics.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        if metrics.recommendations:
            lines.extend([
                f"\n{BOLD}Recommendations:{RESET}"
            ])
            for rec in metrics.recommendations:
                lines.append(f"  💡 {rec}")
        
        lines.extend([
            f"\n{'=' * 80}"
        ])
        
        return "\n".join(lines)


def check_overfitting(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Tuple[bool, OverfittingMetrics]:
    """
    Convenience function to quickly check if a model is overfit.
    
    Args:
        model: Trained sklearn model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        verbose: If True, print report (default: True)
    
    Returns:
        Tuple of (is_overfit: bool, metrics: OverfittingMetrics)
    
    Example:
        is_overfit, metrics = check_overfitting(model, X_train, y_train, X_test, y_test)
        if is_overfit:
            print(f"Model is overfit with severity: {metrics.severity}")
    """
    detector = OverfittingDetector()
    metrics = detector.evaluate(model, X_train, y_train, X_test, y_test)
    
    if verbose:
        report = detector.generate_report(metrics)
        print(report)
    
    return metrics.is_overfit, metrics


# =============================================================================
# Self-test
# =============================================================================

def _self_test():
    """Self-test to verify overfitting detection works correctly."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    print(f"\n{BOLD}Overfitting Detector Self-Test{RESET}")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=3,
        class_sep=1.5,  # More separable classes
        random_state=42
    )
    
    # Split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    detector = OverfittingDetector()
    
    # Test 1: Overfit model (high depth, no regularization)
    print(f"\n{BOLD}Test 1: Overfit Model (DecisionTree max_depth=None){RESET}")
    overfit_model = DecisionTreeClassifier(max_depth=None, random_state=42)
    overfit_model.fit(X_train, y_train)
    
    overfit_metrics = detector.evaluate(
        overfit_model, X_train, y_train, X_test, y_test
    )
    
    print(detector.generate_report(overfit_metrics))
    
    # Test 2: Well-regularized model (Logistic Regression)
    print(f"\n{BOLD}Test 2: Well-Regularized Model (LogisticRegression){RESET}")
    
    good_model = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization
        random_state=42
    )
    good_model.fit(X_train, y_train)
    
    good_metrics = detector.evaluate(
        good_model, X_train, y_train, X_test, y_test
    )
    
    print(detector.generate_report(good_metrics))
    
    # Check if overfitting is at most MILD
    is_acceptable = good_metrics.severity in [OverfitSeverity.NONE, OverfitSeverity.MILD]
    
    # Test 3: From metrics (no model required)
    print(f"\n{BOLD}Test 3: Evaluation from Metrics Only{RESET}")
    metrics_only = detector.evaluate_from_metrics(
        train_accuracy=0.98,
        test_accuracy=0.72,
        train_balanced_accuracy=0.97,
        test_balanced_accuracy=0.71
    )
    
    print(detector.generate_report(metrics_only))
    
    # Summary
    print(f"\n{BOLD}Self-Test Summary:{RESET}")
    print(f"  Test 1 (Overfit):      {'✓ PASS' if overfit_metrics.is_overfit else '✗ FAIL'}")
    print(f"  Test 2 (Regularized):  {'✓ PASS' if is_acceptable else '✗ FAIL (severity: ' + good_metrics.severity.value + ')'}")
    print(f"  Test 3 (Metrics-only): {'✓ PASS' if metrics_only.is_overfit else '✗ FAIL'}")
    print()
    
    all_pass = (
        overfit_metrics.is_overfit and
        is_acceptable and
        metrics_only.is_overfit
    )
    
    if all_pass:
        print(f"{GREEN}{BOLD}✓ ALL TESTS PASSED{RESET}\n")
        return 0
    else:
        print(f"{RED}{BOLD}✗ SOME TESTS FAILED{RESET}\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
