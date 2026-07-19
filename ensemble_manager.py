# 1. Standard library imports
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 2. Third-party imports
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 3. Local imports
from config import (
    MODELS_DIR,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    LABEL_CLASSES,
    MODEL_RANDOM_SEED,
    MODEL_N_ESTIMATORS,
    MODEL_MAX_DEPTH,
    MODEL_LEARNING_RATE,
    ENSEMBLE_MODEL_TYPES,
    ENSEMBLE_RF_N_ESTIMATORS,
    ENSEMBLE_RF_MAX_DEPTH,
    ENSEMBLE_LR_MAX_ITER,
    ensure_directories,
    configure_logging,
)
from model_trainer import ModelTrainer
from feature_engineer import ML_SAFE_SUFFIX
from health_monitor import registry as health_registry

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
ENSEMBLE_FILE_SUFFIX = "_ensemble.joblib"
ENSEMBLE_METADATA_SUFFIX = "_ensemble_metadata.json"


@dataclass
class EnsembleTrainingResult:
    symbol: str
    trained_at: str
    n_train_samples: int
    n_test_samples: int
    feature_columns: List[str]
    per_model_accuracy: Dict[str, float]
    ensemble_accuracy: float
    mean_agreement: float
    success: bool
    error: Optional[str] = None


@dataclass
class EnsemblePrediction:
    predicted_class: str
    confidence: float
    agreement_fraction: float
    per_model_votes: Dict[str, str]


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class EnsembleManager:
    """
    Trains multiple structurally different classifiers (gradient boosting,
    random forest, logistic regression) on the same dataset and time-based
    split as model_trainer.py, then combines them via soft voting (averaging
    predicted class probabilities). Also computes a per-row agreement score —
    the fraction of constituent models that individually agree with the
    ensemble's final call — which downstream calibration/confidence logic
    relies on.
    """

    def __init__(self, model_trainer: Optional[ModelTrainer] = None):
        self.model_trainer = model_trainer or ModelTrainer()
        ensure_directories()

    @staticmethod
    def _build_base_estimators() -> Dict[str, object]:
        """Constructs fresh, unfitted estimator instances for every model
        type listed in config.ENSEMBLE_MODEL_TYPES."""
        available = {
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=MODEL_N_ESTIMATORS, max_depth=MODEL_MAX_DEPTH,
                learning_rate=MODEL_LEARNING_RATE, random_state=MODEL_RANDOM_SEED,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=ENSEMBLE_RF_N_ESTIMATORS, max_depth=ENSEMBLE_RF_MAX_DEPTH,
                random_state=MODEL_RANDOM_SEED,
            ),
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=ENSEMBLE_LR_MAX_ITER, random_state=MODEL_RANDOM_SEED)),
            ]),
        }
        selected = {}
        for name in ENSEMBLE_MODEL_TYPES:
            if name in available:
                selected[name] = available[name]
            else:
                logger.error(f"Unknown ensemble model type '{name}' in config — skipping it.")
        return selected

    @staticmethod
    def _reindexed_proba(model, X: pd.DataFrame) -> np.ndarray:
        """
        Returns a probability matrix with columns in the EXACT order of
        LABEL_CLASSES, regardless of the order sklearn assigned internally
        or whether this particular model happened to see every class during
        training. Missing classes get a probability of 0 for that column.
        """
        raw_proba = model.predict_proba(X)
        model_classes = list(model.classes_)
        reindexed = np.zeros((X.shape[0], len(LABEL_CLASSES)))
        for i, cls in enumerate(LABEL_CLASSES):
            if cls in model_classes:
                reindexed[:, i] = raw_proba[:, model_classes.index(cls)]
        return reindexed

    def train_ensemble_for_symbol(
        self, symbol: str, stock_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None,
        horizon: str = HORIZON_INTRADAY
    ) -> EnsembleTrainingResult:
        try:
            prepared = self.model_trainer.prepare_dataset(stock_df, index_df, horizon=horizon)
            if prepared is None:
                return EnsembleTrainingResult(
                    symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=0,
                    n_test_samples=0, feature_columns=[], per_model_accuracy={}, ensemble_accuracy=0.0,
                    mean_agreement=0.0, success=False,
                    error="Dataset preparation failed or returned no usable rows.",
                )
            X, y, feature_columns = prepared

            min_training_samples = HORIZON_CONFIG[horizon]["min_training_samples"]
            if len(X) < min_training_samples:
                msg = f"Only {len(X)} usable samples for {symbol} ({horizon}), need >= {min_training_samples}."
                logger.error(msg)
                return EnsembleTrainingResult(
                    symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=len(X),
                    n_test_samples=0, feature_columns=feature_columns, per_model_accuracy={},
                    ensemble_accuracy=0.0, mean_agreement=0.0, success=False, error=msg,
                )

            X_train, X_test, y_train, y_test = self.model_trainer.time_based_split(X, y)
            if len(X_train) > 0 and len(X_test) > 0:
                assert X_train.index.max() <= X_test.index.min(), (
                    "Time-based split violated: a training row is timestamped after a test row!"
                )

            estimators = self._build_base_estimators()
            if not estimators:
                raise ValueError("No valid ensemble model types configured.")

            fitted_models: Dict[str, object] = {}
            per_model_accuracy: Dict[str, float] = {}
            proba_matrices: Dict[str, np.ndarray] = {}

            for name, estimator in estimators.items():
                estimator.fit(X_train, y_train)
                fitted_models[name] = estimator
                proba = self._reindexed_proba(estimator, X_test)
                proba_matrices[name] = proba
                individual_preds = [LABEL_CLASSES[i] for i in proba.argmax(axis=1)]
                per_model_accuracy[name] = accuracy_score(y_test, individual_preds)

            # Soft-vote: average probability matrices across all models
            avg_proba = np.mean(list(proba_matrices.values()), axis=0)
            ensemble_pred_idx = avg_proba.argmax(axis=1)
            ensemble_preds = [LABEL_CLASSES[i] for i in ensemble_pred_idx]
            ensemble_accuracy = accuracy_score(y_test, ensemble_preds)

            # Agreement: fraction of individual models whose own vote matches the ensemble's vote
            model_names = list(proba_matrices.keys())
            individual_pred_arrays = {
                name: proba_matrices[name].argmax(axis=1) for name in model_names
            }
            agreement_counts = np.zeros(len(ensemble_pred_idx))
            for name in model_names:
                agreement_counts += (individual_pred_arrays[name] == ensemble_pred_idx).astype(int)
            agreement_fraction_per_row = agreement_counts / len(model_names)
            mean_agreement = float(np.mean(agreement_fraction_per_row))

            self._save_ensemble(symbol, fitted_models, feature_columns, per_model_accuracy,
                                 ensemble_accuracy, mean_agreement, horizon=horizon)

            health_registry.report("ensemble_manager", ok=True, detail=f"Trained ensemble for {symbol}")
            return EnsembleTrainingResult(
                symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=len(X_train),
                n_test_samples=len(X_test), feature_columns=feature_columns,
                per_model_accuracy=per_model_accuracy, ensemble_accuracy=ensemble_accuracy,
                mean_agreement=mean_agreement, success=True,
            )

        except Exception as e:
            logger.error(f"Ensemble training pipeline failed for {symbol}: {e}")
            health_registry.report("ensemble_manager", ok=False, detail=f"Training failed for {symbol}", error=str(e))
            return EnsembleTrainingResult(
                symbol=symbol, trained_at=datetime.now().isoformat(), n_train_samples=0,
                n_test_samples=0, feature_columns=[], per_model_accuracy={}, ensemble_accuracy=0.0,
                mean_agreement=0.0, success=False, error=str(e),
            )

    def _ensemble_path(self, symbol: str, horizon: str) -> Path:
        return MODELS_DIR / f"{symbol}_{horizon}{ENSEMBLE_FILE_SUFFIX}"

    def _ensemble_metadata_path(self, symbol: str, horizon: str) -> Path:
        return MODELS_DIR / f"{symbol}_{horizon}{ENSEMBLE_METADATA_SUFFIX}"

    def _save_ensemble(
        self, symbol: str, fitted_models: Dict[str, object], feature_columns: List[str],
        per_model_accuracy: Dict[str, float], ensemble_accuracy: float, mean_agreement: float,
        horizon: str = HORIZON_INTRADAY
    ) -> None:
        try:
            ensure_directories()
            joblib.dump({"models": fitted_models, "classes": LABEL_CLASSES}, self._ensemble_path(symbol, horizon))
            metadata = {
                "symbol": symbol,
                "horizon": horizon,
                "trained_at": datetime.now().isoformat(),
                "feature_columns": feature_columns,
                "label_classes": LABEL_CLASSES,
                "per_model_accuracy": per_model_accuracy,
                "ensemble_accuracy": ensemble_accuracy,
                "mean_agreement": mean_agreement,
            }
            f = None
            try:
                f = open(self._ensemble_metadata_path(symbol, horizon), "w", encoding="utf-8")
                json.dump(metadata, f, indent=2)
            finally:
                if f is not None:
                    f.close()
            logger.info(f"Saved ensemble + metadata for {symbol} ({horizon}) to {MODELS_DIR}")
        except Exception as e:
            logger.error(f"Failed saving ensemble for {symbol}: {e}")
            raise

    def load_ensemble(self, symbol: str, horizon: str = HORIZON_INTRADAY) -> Optional[Tuple[Dict[str, object], List[str], Dict]]:
        ensemble_path = self._ensemble_path(symbol, horizon)
        metadata_path = self._ensemble_metadata_path(symbol, horizon)
        try:
            if not ensemble_path.exists() or not metadata_path.exists():
                logger.error(f"No saved ensemble found for {symbol} ({horizon}) at {ensemble_path}. Train it first.")
                return None
            bundle = joblib.load(ensemble_path)
            f = None
            try:
                f = open(metadata_path, "r", encoding="utf-8")
                metadata = json.load(f)
            finally:
                if f is not None:
                    f.close()
            return bundle["models"], bundle["classes"], metadata
        except Exception as e:
            logger.error(f"Failed loading ensemble for {symbol}: {e}")
            return None

    def predict(self, symbol: str, X: pd.DataFrame, horizon: str = HORIZON_INTRADAY) -> Optional[List[EnsemblePrediction]]:
        """Runs the saved ensemble on new feature rows (must already be the
        '_feat'-lagged columns matching what the ensemble was trained on)."""
        loaded = self.load_ensemble(symbol, horizon=horizon)
        if loaded is None:
            return None
        models, classes, metadata = loaded

        try:
            expected_cols = metadata["feature_columns"]
            missing = [c for c in expected_cols if c not in X.columns]
            if missing:
                logger.error(f"Cannot predict for {symbol}: missing expected feature columns {missing}")
                return None
            X_ordered = X[expected_cols]

            proba_matrices = {name: self._reindexed_proba(model, X_ordered) for name, model in models.items()}
            avg_proba = np.mean(list(proba_matrices.values()), axis=0)
            pred_idx = avg_proba.argmax(axis=1)

            individual_pred_arrays = {name: p.argmax(axis=1) for name, p in proba_matrices.items()}

            results = []
            for row_i in range(len(X_ordered)):
                predicted_class = LABEL_CLASSES[pred_idx[row_i]]
                confidence = float(avg_proba[row_i, pred_idx[row_i]])
                votes = {name: LABEL_CLASSES[individual_pred_arrays[name][row_i]] for name in models}
                agreement = sum(1 for v in votes.values() if v == predicted_class) / len(votes)
                results.append(EnsemblePrediction(
                    predicted_class=predicted_class, confidence=confidence,
                    agreement_fraction=agreement, per_model_votes=votes,
                ))
            health_registry.report("ensemble_manager", ok=True, detail=f"Predicted for {symbol}")
            return results

        except Exception as e:
            logger.error(f"Failed running ensemble prediction for {symbol}: {e}")
            health_registry.report("ensemble_manager", ok=False, detail=f"Prediction failed for {symbol}", error=str(e))
            return None


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="ensemble_manager_selftest.log")
    logger.info("Running ensemble_manager.py self-test...")

    def _build_synthetic_ohlcv(n_days: int = 40, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
        """Fully offline, fixed-seed synthetic OHLCV — same generator style
        used in feature_engineer.py / model_trainer.py self-tests."""
        rng = np.random.default_rng(seed)
        rows, timestamps = [], []
        price = 1000.0
        base_date = pd.Timestamp("2026-01-05 09:15:00")
        recent_closes = []

        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    recent_trend = recent_closes[-1] - recent_closes[-10]
                    bias = 1.5 if recent_trend < -6 else (-1.5 if recent_trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = rng.normal(bias, 1.5)
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + rng.normal(bias * 0.5, 1.0))
                high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
                low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)

        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                             index=pd.DatetimeIndex(timestamps))

    test_symbol = "SYNTHTEST_ENSEMBLE"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== ENSEMBLE MANAGER SELF-TEST RESULT ===")
        manager = EnsembleManager()

        stock_df = _build_synthetic_ohlcv(n_days=40, bars_per_day=75, seed=42)
        index_df = _build_synthetic_ohlcv(n_days=40, bars_per_day=75, seed=99)
        index_df.index = stock_df.index

        result = manager.train_ensemble_for_symbol(test_symbol, stock_df, index_df)

        print(f"Training success: {result.success}")
        if not result.success:
            print(f"Error: {result.error}")
        print(f"Train samples: {result.n_train_samples}, Test samples: {result.n_test_samples}")
        print(f"Per-model accuracy: { {k: round(v, 3) for k, v in result.per_model_accuracy.items()} }")
        print(f"Ensemble accuracy: {result.ensemble_accuracy:.3f}")
        print(f"Mean agreement fraction: {result.mean_agreement:.3f}")

        n_models_ok = len(result.per_model_accuracy) == len(ENSEMBLE_MODEL_TYPES)
        print(f"All {len(ENSEMBLE_MODEL_TYPES)} configured model types trained: {n_models_ok}")

        # Load + predict round trip on a few rows
        loaded = manager.load_ensemble(test_symbol)
        load_ok = loaded is not None
        print(f"Ensemble save/load round trip: {'OK' if load_ok else 'FAILED'}")

        predict_ok = False
        agreement_in_range = False
        if load_ok:
            prepared = manager.model_trainer.prepare_dataset(stock_df, index_df)
            if prepared is not None:
                X, y, _ = prepared
                _, X_test, _, _ = manager.model_trainer.time_based_split(X, y)
                sample_predictions = manager.predict(test_symbol, X_test.iloc[:10])
                predict_ok = sample_predictions is not None and len(sample_predictions) == 10
                if predict_ok:
                    agreement_in_range = all(0.0 <= p.agreement_fraction <= 1.0 for p in sample_predictions)
                    confidences_in_range = all(0.0 <= p.confidence <= 1.0 for p in sample_predictions)
                    print(f"Sample prediction agreement fractions valid [0,1]: {agreement_in_range}")
                    print(f"Sample prediction confidences valid [0,1]: {confidences_in_range}")
                    print(f"Example prediction: {sample_predictions[0]}")

        overall_pass = result.success and n_models_ok and load_ok and predict_ok and agreement_in_range
        print("STATUS: PASS" if overall_pass else "STATUS: FAIL — see details above")

        assert overall_pass, "One or more ensemble_manager.py self-test checks failed"
        logger.info("ensemble_manager.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"ensemble_manager.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"ensemble_manager.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
