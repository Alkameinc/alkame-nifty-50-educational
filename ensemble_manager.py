# 1. Standard library imports
import hashlib
import json
import logging
import platform
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast


def _get_git_commit_sha() -> str:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        )
        return sha
    except Exception:
        return "UNKNOWN"


# 2. Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    ENSEMBLE_LR_MAX_ITER,
    ENSEMBLE_MODEL_TYPES,
    ENSEMBLE_RF_MAX_DEPTH,
    ENSEMBLE_RF_N_ESTIMATORS,
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    LABEL_CLASSES,
    MODEL_LEARNING_RATE,
    MODEL_MAX_DEPTH,
    MODEL_N_ESTIMATORS,
    MODEL_RANDOM_SEED,
    MODELS_DIR,
    configure_logging,
    ensure_directories,
)

# 3. Local imports
import config
from database import SessionLocal
from health_monitor import registry as health_registry
from model_lineage import build_lineage_metadata
from model_trainer import ModelTrainer
from models import ModelRegistry

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
    feature_columns: list[str]
    per_model_accuracy: dict[str, float]
    ensemble_accuracy: float
    mean_agreement: float
    success: bool
    error: str | None = None


@dataclass
class EnsemblePrediction:
    predicted_class: str
    confidence: float
    agreement_fraction: float
    per_model_votes: dict[str, str]
    model_version: str = "UNKNOWN"
    feature_version: str = "UNKNOWN"


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

    def __init__(self, model_trainer: ModelTrainer | None = None):
        self.model_trainer = model_trainer or ModelTrainer()
        ensure_directories()

    @staticmethod
    def _build_base_estimators() -> dict[str, object]:
        """Constructs fresh, unfitted estimator instances for every model
        type listed in config.ENSEMBLE_MODEL_TYPES."""
        available = {
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=MODEL_N_ESTIMATORS,
                max_depth=MODEL_MAX_DEPTH,
                learning_rate=MODEL_LEARNING_RATE,
                random_state=MODEL_RANDOM_SEED,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=ENSEMBLE_RF_N_ESTIMATORS,
                max_depth=ENSEMBLE_RF_MAX_DEPTH,
                random_state=MODEL_RANDOM_SEED,
            ),
            "logistic_regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=ENSEMBLE_LR_MAX_ITER, random_state=MODEL_RANDOM_SEED)),
                ]
            ),
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
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        horizon: str = HORIZON_INTRADAY,
    ) -> EnsembleTrainingResult:
        try:
            prepared = self.model_trainer.prepare_dataset(stock_df, index_df, horizon=horizon)
            if prepared is None:
                return EnsembleTrainingResult(
                    symbol=symbol,
                    trained_at=datetime.now().isoformat(),
                    n_train_samples=0,
                    n_test_samples=0,
                    feature_columns=[],
                    per_model_accuracy={},
                    ensemble_accuracy=0.0,
                    mean_agreement=0.0,
                    success=False,
                    error="Dataset preparation failed or returned no usable rows.",
                )
            X, y, feature_columns = prepared

            min_training_samples = cast(int, HORIZON_CONFIG[horizon]["min_training_samples"])
            if len(X) < min_training_samples:
                msg = f"Only {len(X)} usable samples for {symbol} ({horizon}), need >= {min_training_samples}."
                logger.error(msg)
                return EnsembleTrainingResult(
                    symbol=symbol,
                    trained_at=datetime.now().isoformat(),
                    n_train_samples=len(X),
                    n_test_samples=0,
                    feature_columns=feature_columns,
                    per_model_accuracy={},
                    ensemble_accuracy=0.0,
                    mean_agreement=0.0,
                    success=False,
                    error=msg,
                )

            horizon_bars = cast(int, HORIZON_CONFIG.get(horizon, {}).get("horizon_bars", 0))
            X_train, X_test, y_train, y_test = self.model_trainer.time_based_split(X, y, purge_window=horizon_bars)
            if len(X_train) > 0 and len(X_test) > 0:
                assert (
                    X_train.index.max() < X_test.index.min()
                ), "Time-based split violated: a training row is timestamped at or after a test row!"

            estimators = self._build_base_estimators()
            if not estimators:
                raise ValueError("No valid ensemble model types configured.")

            fitted_models: dict[str, object] = {}
            per_model_accuracy: dict[str, float] = {}
            proba_matrices: dict[str, np.ndarray] = {}

            for name, estimator in estimators.items():
                cast(Any, estimator).fit(X_train, y_train)
                fitted_models[name] = estimator
                proba = self._reindexed_proba(estimator, X_test)
                proba_matrices[name] = proba
                individual_preds = [LABEL_CLASSES[i] for i in proba.argmax(axis=1)]
                per_model_accuracy[name] = accuracy_score(y_test, individual_preds)

            # Soft-vote: average probability matrices across all models
            avg_proba = np.mean(list(proba_matrices.values()), axis=0)
            avg_proba = avg_proba / avg_proba.sum(axis=1, keepdims=True)  # Renormalize
            ensemble_pred_idx = avg_proba.argmax(axis=1)
            ensemble_preds = [LABEL_CLASSES[i] for i in ensemble_pred_idx]
            ensemble_accuracy = accuracy_score(y_test, ensemble_preds)

            # Agreement: fraction of individual models whose own vote matches the ensemble's vote
            model_names = list(proba_matrices.keys())
            individual_pred_arrays = {name: proba_matrices[name].argmax(axis=1) for name in model_names}
            agreement_counts = np.zeros(len(ensemble_pred_idx))
            for name in model_names:
                agreement_counts += (individual_pred_arrays[name] == ensemble_pred_idx).astype(int)
            agreement_fraction_per_row = agreement_counts / len(model_names)
            mean_agreement = float(np.mean(agreement_fraction_per_row))

            data_start = str(X_train.index.min()) if len(X_train) > 0 else None
            data_end = str(X_train.index.max()) if len(X_train) > 0 else None

            self._save_ensemble(
                symbol=symbol,
                fitted_models=fitted_models,
                feature_columns=feature_columns,
                per_model_accuracy=per_model_accuracy,
                ensemble_accuracy=ensemble_accuracy,
                mean_agreement=mean_agreement,
                horizon=horizon,
                training_sample_count=len(X_train),
                training_data_start=data_start,
                training_data_end=data_end,
            )

            health_registry.report("ensemble_manager", ok=True, detail=f"Trained ensemble for {symbol}")
            return EnsembleTrainingResult(
                symbol=symbol,
                trained_at=datetime.now().isoformat(),
                n_train_samples=len(X_train),
                n_test_samples=len(X_test),
                feature_columns=feature_columns,
                per_model_accuracy=per_model_accuracy,
                ensemble_accuracy=ensemble_accuracy,
                mean_agreement=mean_agreement,
                success=True,
            )

        except Exception as e:
            logger.error(f"Ensemble training pipeline failed for {symbol}: {e}")
            health_registry.report("ensemble_manager", ok=False, detail=f"Training failed for {symbol}", error=str(e))
            return EnsembleTrainingResult(
                symbol=symbol,
                trained_at=datetime.now().isoformat(),
                n_train_samples=0,
                n_test_samples=0,
                feature_columns=[],
                per_model_accuracy={},
                ensemble_accuracy=0.0,
                mean_agreement=0.0,
                success=False,
                error=str(e),
            )

    def _versioned_dir(self, symbol: str, horizon: str, run_id: str) -> Path:
        return MODELS_DIR / symbol / horizon / run_id

    def _current_pointer_path(self, symbol: str, horizon: str) -> Path:
        return MODELS_DIR / symbol / horizon / "current.json"

    def _ensemble_path(self, symbol: str, horizon: str) -> Path:
        """Primary flat-file path with horizon suffix (current standard)."""
        return MODELS_DIR / f"{symbol}_{horizon}{ENSEMBLE_FILE_SUFFIX}"

    def _legacy_ensemble_path(self, symbol: str) -> Path:
        """Legacy flat-file path WITHOUT horizon suffix (pre-multi-horizon compatibility)."""
        return MODELS_DIR / f"{symbol}{ENSEMBLE_FILE_SUFFIX}"

    def _ensemble_metadata_path(self, symbol: str, horizon: str) -> Path:
        return MODELS_DIR / f"{symbol}_{horizon}{ENSEMBLE_METADATA_SUFFIX}"

    def _legacy_metadata_path(self, symbol: str) -> Path:
        """Legacy metadata path WITHOUT horizon suffix."""
        return MODELS_DIR / f"{symbol}{ENSEMBLE_METADATA_SUFFIX}"

    def _save_ensemble(
        self,
        symbol: str,
        fitted_models: dict[str, object],
        feature_columns: list[str],
        per_model_accuracy: dict[str, float],
        ensemble_accuracy: float,
        mean_agreement: float,
        horizon: str = HORIZON_INTRADAY,
        training_sample_count: int = 0,
        training_data_start: str | None = None,
        training_data_end: str | None = None,
    ) -> str:
        try:
            ensure_directories()
            schema_str = ",".join(sorted(feature_columns))
            schema_hash = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()
            model_id = str(uuid.uuid4())
            commit_sha = _get_git_commit_sha()
            now_iso = datetime.now().isoformat()
            ts_compact = datetime.now().strftime("%Y%m%dT%H%M%S")
            run_id = f"{ts_compact}_{commit_sha[:7]}_{model_id[:8]}"

            # 1. Immutable versioned directory (QNT-008)
            run_dir = self._versioned_dir(symbol, horizon, run_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            bundle = {"models": fitted_models, "classes": LABEL_CLASSES}
            joblib.dump(bundle, run_dir / "ensemble.joblib")
            artifact_sha256 = hashlib.sha256(open(run_dir / "ensemble.joblib", "rb").read()).hexdigest()

            import sklearn

            extra_meta = {
                "model_id": model_id,
                "run_id": run_id,
                "feature_version": config.FEATURE_VERSION,
                "training_seed": MODEL_RANDOM_SEED,
                "label_classes": LABEL_CLASSES,
                "per_model_accuracy": per_model_accuracy,
                "ensemble_accuracy": ensemble_accuracy,
                "mean_agreement": mean_agreement,
                "environment": {
                    "python_version": platform.python_version(),
                    "sklearn_version": sklearn.__version__,
                    "joblib_version": joblib.__version__,
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                },
            }

            metadata = build_lineage_metadata(
                symbol=symbol,
                horizon=horizon,
                feature_columns=feature_columns,
                artifact_path_or_hash=artifact_sha256,
                dataset_start=training_data_start,
                dataset_end=training_data_end,
                sample_count=training_sample_count,
                model_version=config.MODEL_VERSION,
                universe_version=config.UNIVERSE_VERSION,
                extra_metadata=extra_meta,
            )

            with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            schema_info = {
                "symbol": symbol,
                "horizon": horizon,
                "feature_columns": feature_columns,
                "feature_schema_hash": schema_hash,
                "artifact_sha256": artifact_sha256,
                "created_at": now_iso,
            }
            with open(run_dir / "schema.json", "w", encoding="utf-8") as f:
                json.dump(schema_info, f, indent=2)

            metrics_info = {
                "per_model_accuracy": per_model_accuracy,
                "ensemble_accuracy": ensemble_accuracy,
                "mean_agreement": mean_agreement,
                "trained_at": now_iso,
            }
            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics_info, f, indent=2)

            # 2. Update atomic current pointer
            pointer_path = self._current_pointer_path(symbol, horizon)
            pointer_data = {
                "symbol": symbol,
                "horizon": horizon,
                "current_run_id": run_id,
                "model_id": model_id,
                "updated_at": now_iso,
                "path": str(run_dir),
            }
            with open(pointer_path, "w", encoding="utf-8") as f:
                json.dump(pointer_data, f, indent=2)

            # 3. Synchronize to legacy flat paths for backward compatibility
            joblib.dump(bundle, self._ensemble_path(symbol, horizon))
            with open(self._ensemble_metadata_path(symbol, horizon), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # DB-003, REP-003: Add model lineage and artifact compatibility metadata to DB
            try:
                with SessionLocal() as db:
                    # Deactivate previous active models for this symbol and horizon
                    db.query(ModelRegistry).filter_by(symbol=symbol, horizon=horizon, is_active=True).update(
                        {"is_active": False}
                    )
                    db.commit()

                    registry_entry = ModelRegistry(
                        model_version=run_id,
                        symbol=symbol,
                        horizon=horizon,
                        created_at=now_iso,
                        git_commit=commit_sha,
                        python_version=platform.python_version(),
                        os_platform=platform.system(),
                        sklearn_version=sklearn.__version__,
                        pandas_version=pd.__version__,
                        joblib_version=joblib.__version__,
                        features_hash=schema_hash,
                        artifact_path=str(run_dir),
                        is_active=True,
                    )
                    db.add(registry_entry)
                    db.commit()
            except Exception as dbe:
                logger.error(f"Failed writing model registry to DB for {symbol}: {dbe}")

            logger.info(f"Saved immutable ensemble version {run_id} for {symbol} ({horizon}) at {run_dir}")

            return run_id
        except Exception as e:
            logger.error(f"Failed saving ensemble for {symbol}: {e}")
            raise

    def load_ensemble(
        self, symbol: str, horizon: str = HORIZON_INTRADAY
    ) -> tuple[dict[str, object], list[str], dict] | None:
        """
        Load ensemble models with 3-tier fallback strategy:
        1. Versioned immutable path (preferred, QNT-008 compliant)
        2. Horizon-suffixed legacy flat path (multi-horizon standard)
        3. Un-suffixed legacy flat path (pre-multi-horizon compatibility with warnings)
        
        Returns (models_dict, classes_list, metadata_dict) or None if no valid artifact found.
        """
        try:
            # TIER 1: Check versioned current pointer first (QNT-008)
            pointer_path = self._current_pointer_path(symbol, horizon)
            if pointer_path.exists():
                with open(pointer_path, encoding="utf-8") as pf:
                    pointer_data = json.load(pf)
                curr_run_id = pointer_data.get("current_run_id")
                run_dir = self._versioned_dir(symbol, horizon, curr_run_id)
                ensemble_file = run_dir / "ensemble.joblib"
                meta_file = run_dir / "metadata.json"
                if ensemble_file.exists() and meta_file.exists():
                    with open(meta_file, encoding="utf-8") as mf:
                        metadata = json.load(mf)
                    logger.info(
                        f"Loaded ensemble for {symbol} ({horizon}) from versioned path: {run_dir} "
                        f"[run_id={curr_run_id}]"
                    )
                    return bundle["models"], bundle["classes"], metadata

            # TIER 2: Fall back to horizon-suffixed legacy flat paths (current standard)
            ensemble_path = self._ensemble_path(symbol, horizon)
            metadata_path = self._ensemble_metadata_path(symbol, horizon)
            if ensemble_path.exists() and metadata_path.exists():
                bundle = joblib.load(ensemble_path)
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                
                # Validate horizon compatibility
                meta_horizon = metadata.get("horizon", "UNKNOWN")
                if meta_horizon != horizon and meta_horizon != "UNKNOWN":
                    logger.warning(
                        f"Horizon mismatch: requested {horizon}, metadata says {meta_horizon} "
                        f"for {symbol} at {ensemble_path}. Proceeding with caution."
                    )
                
                logger.info(
                    f"Loaded ensemble for {symbol} ({horizon}) from legacy flat path: {ensemble_path}"
                )
                return bundle["models"], bundle["classes"], metadata

            # TIER 3: Fall back to un-suffixed legacy flat path (pre-multi-horizon compatibility)
            legacy_ensemble = self._legacy_ensemble_path(symbol)
            legacy_metadata = self._legacy_metadata_path(symbol)
            
            if legacy_ensemble.exists():
                bundle = joblib.load(legacy_ensemble)
                
                # Try to load metadata if available
                metadata = {}
                if legacy_metadata.exists():
                    with open(legacy_metadata, encoding="utf-8") as f:
                        metadata = json.load(f)
                
                # Check if metadata specifies a horizon
                meta_horizon = metadata.get("horizon", "UNKNOWN")
                
                # ALWAYS warn about legacy unsuffixed artifacts (migration debt alert)
                if meta_horizon != "UNKNOWN" and meta_horizon != horizon:
                    # Horizon mismatch - highest severity warning
                    logger.warning(
                        f"LEGACY ARTIFACT COMPATIBILITY WARNING: {symbol} requested for horizon {horizon}, "
                        f"but legacy unsuffixed artifact {legacy_ensemble.name} has metadata horizon={meta_horizon}. "
                        f"Using anyway but predictions may be unreliable. Consider retraining."
                    )
                elif meta_horizon == "UNKNOWN":
                    # No horizon metadata - provenance unknown
                    logger.warning(
                        f"LEGACY ARTIFACT PROVENANCE UNKNOWN: {symbol} loaded from legacy unsuffixed {legacy_ensemble.name} "
                        f"with no horizon metadata. Assuming horizon={horizon} but this is unverified. "
                        f"Artifact may be from pre-multi-horizon era. Consider retraining for guaranteed compatibility."
                    )
                    # Inject inferred horizon into metadata for downstream consumers
                    metadata["horizon"] = horizon
                    metadata["provenance_warning"] = "INFERRED_FROM_REQUEST"
                else:
                    # Horizon matches but still legacy unsuffixed - informational warning
                    logger.warning(
                        f"LEGACY unsuffixed artifact loaded: {symbol} from {legacy_ensemble.name} "
                        f"(horizon={horizon} matches metadata). This artifact uses pre-multi-horizon naming. "
                        f"Consider migrating to versioned or horizon-suffixed naming for better provenance tracking."
                    )
                
                logger.info(
                    f"Loaded ensemble for {symbol} ({horizon}) from LEGACY unsuffixed path: {legacy_ensemble} "
                    f"[compatibility=UNVERIFIED, provenance={meta_horizon}]"
                )
                return bundle["models"], bundle["classes"], metadata

            # NO ARTIFACT FOUND
            logger.error(
                f"No saved ensemble found for {symbol} ({horizon}). Checked:\n"
                f"  1. Versioned: {pointer_path}\n"
                f"  2. Flat w/ horizon: {ensemble_path}\n"
                f"  3. Legacy flat: {legacy_ensemble}\n"
                f"Train the model first using train_all.py or backtester."
            )
            return None
            
        except Exception as e:
            logger.error(f"Failed loading ensemble for {symbol} ({horizon}): {e}")
            return None

    def rollback_ensemble(self, symbol: str, horizon: str, target_run_id: str) -> bool:
        """
        Rolls back the active model pointer to a previous immutable run version (QNT-008).
        """
        try:
            run_dir = self._versioned_dir(symbol, horizon, target_run_id)
            ensemble_file = run_dir / "ensemble.joblib"
            meta_file = run_dir / "metadata.json"
            if not (ensemble_file.exists() and meta_file.exists()):
                logger.error(f"Cannot rollback to non-existent version {target_run_id} for {symbol} ({horizon})")
                return False

            with open(meta_file, encoding="utf-8") as mf:
                metadata = json.load(mf)

            pointer_path = self._current_pointer_path(symbol, horizon)
            pointer_data = {
                "symbol": symbol,
                "horizon": horizon,
                "current_run_id": target_run_id,
                "model_id": metadata.get("model_id"),
                "updated_at": datetime.now().isoformat(),
                "path": str(run_dir),
                "rolled_back_at": datetime.now().isoformat(),
            }
            with open(pointer_path, "w", encoding="utf-8") as pf:
                json.dump(pointer_data, pf, indent=2)

            # Update DB registry
            try:
                with SessionLocal() as db:
                    db.query(ModelRegistry).filter_by(symbol=symbol, horizon=horizon, is_active=True).update(
                        {"is_active": False}
                    )
                    db.query(ModelRegistry).filter_by(model_version=target_run_id).update({"is_active": True})
                    db.commit()
            except Exception as dbe:
                logger.error(f"Failed updating model registry DB for rollback on {symbol}: {dbe}")

            # Sync legacy files
            bundle = joblib.load(ensemble_file)
            joblib.dump(bundle, self._ensemble_path(symbol, horizon))
            with open(self._ensemble_metadata_path(symbol, horizon), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Successfully rolled back {symbol} ({horizon}) to version {target_run_id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed for {symbol} ({horizon}) to {target_run_id}: {e}")
            return False

    def list_model_versions(self, symbol: str, horizon: str) -> list[dict]:
        """
        Lists all available immutable model versions for (symbol, horizon) (QNT-008).
        """
        horizon_dir = MODELS_DIR / symbol / horizon
        if not horizon_dir.exists():
            return []

        curr_run_id = None
        pointer_path = self._current_pointer_path(symbol, horizon)
        if pointer_path.exists():
            try:
                with open(pointer_path, encoding="utf-8") as pf:
                    curr_run_id = json.load(pf).get("current_run_id")
            except Exception:
                pass

        versions = []
        for d in horizon_dir.iterdir():
            if d.is_dir():
                meta_file = d / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, encoding="utf-8") as f:
                            meta = json.load(f)
                        meta["is_current"] = d.name == curr_run_id
                        versions.append(meta)
                    except Exception:
                        pass

        versions.sort(key=lambda x: x.get("trained_at", ""), reverse=True)
        return versions

    def inventory_legacy_artifacts(self, symbols: list[str] | None = None) -> dict[str, dict]:
        """
        Scans models/ directory for legacy artifacts and reports compatibility status.
        
        Returns dict mapping artifact_name -> {
            "path": Path,
            "symbol": str,
            "has_metadata": bool,
            "metadata_horizon": str | None,
            "filename_horizon": str | None,
            "compatibility_status": str,  # COMPATIBLE | AMBIGUOUS | LEGACY_UNSUFFIXED | UNKNOWN
            "recommendation": str,
            "metadata_snippet": dict | None
        }
        """
        from config import ALL_HORIZONS
        
        inventory = {}
        
        # Scan for .joblib files in models/ root (legacy flat structure)
        for joblib_file in MODELS_DIR.glob("*_ensemble.joblib"):
            artifact_name = joblib_file.name
            
            # Try to parse symbol from filename
            # Format: {SYMBOL}_ensemble.joblib OR {SYMBOL}_{HORIZON}_ensemble.joblib
            name_parts = joblib_file.stem.replace("_ensemble", "").split("_")
            
            # Check if last part before _ensemble is a known horizon
            potential_symbol_parts = []
            potential_horizon = None
            
            for i, part in enumerate(name_parts):
                if part in ALL_HORIZONS:
                    potential_horizon = part
                    potential_symbol_parts = name_parts[:i]
                    break
            
            if not potential_horizon:
                # No horizon in filename - this is legacy unsuffixed format
                potential_symbol_parts = name_parts
            
            symbol_name = "_".join(potential_symbol_parts) if potential_symbol_parts else "UNKNOWN"
            
            # If symbols filter provided, skip non-matching
            if symbols and symbol_name not in symbols:
                continue
            
            # Check for metadata
            meta_path = joblib_file.with_suffix("").parent / f"{joblib_file.stem}_metadata.json"
            has_metadata = meta_path.exists()
            metadata_horizon = None
            metadata = {}
            
            if has_metadata:
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        metadata = json.load(f)
                        metadata_horizon = metadata.get("horizon", None)
                except Exception as e:
                    logger.warning(f"Could not read metadata for {artifact_name}: {e}")
            
            # Determine compatibility
            if potential_horizon and metadata_horizon:
                # Both filename and metadata have horizon
                if potential_horizon == metadata_horizon:
                    status = "COMPATIBLE"
                    recommendation = f"Artifact is properly tagged for horizon={potential_horizon}"
                else:
                    status = "AMBIGUOUS"
                    recommendation = (
                        f"Filename suggests {potential_horizon}, metadata says {metadata_horizon}. "
                        "Retrain recommended."
                    )
            elif potential_horizon and not metadata_horizon:
                status = "AMBIGUOUS"
                recommendation = (
                    f"Filename suggests horizon={potential_horizon} but metadata missing/unclear. "
                    "Artifact provenance uncertain."
                )
            elif not potential_horizon and metadata_horizon:
                status = "LEGACY_UNSUFFIXED"
                recommendation = (
                    f"Legacy pre-multi-horizon artifact. Metadata indicates horizon={metadata_horizon}. "
                    f"Will work but consider migrating to standard naming: "
                    f"{symbol_name}_{metadata_horizon}_ensemble.joblib"
                )
            else:
                status = "UNKNOWN"
                recommendation = (
                    "No horizon information in filename or metadata. Cannot determine intended use. "
                    "Strongly recommend retraining with current codebase."
                )
            
            inventory[artifact_name] = {
                "path": joblib_file,
                "symbol": symbol_name,
                "has_metadata": has_metadata,
                "metadata_horizon": metadata_horizon,
                "filename_horizon": potential_horizon,
                "compatibility_status": status,
                "recommendation": recommendation,
                "metadata_snippet": {
                    "model_version": metadata.get("model_version"),
                    "trained_at": metadata.get("trained_at"),
                    "code_commit_sha": metadata.get("code_commit_sha"),
                } if metadata else None,
            }
        
        return inventory

    def predict(self, symbol: str, X: pd.DataFrame, horizon: str = HORIZON_INTRADAY) -> list[EnsemblePrediction] | None:
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
            avg_proba = avg_proba / avg_proba.sum(axis=1, keepdims=True)  # Renormalize
            pred_idx = avg_proba.argmax(axis=1)

            individual_pred_arrays = {name: p.argmax(axis=1) for name, p in proba_matrices.items()}

            results = []
            m_ver = metadata.get("model_version", "v1.0")
            f_ver = metadata.get("feature_version", "v1.0")
            for row_i in range(len(X_ordered)):
                predicted_class = LABEL_CLASSES[pred_idx[row_i]]
                confidence = float(avg_proba[row_i, pred_idx[row_i]])
                votes = {name: LABEL_CLASSES[individual_pred_arrays[name][row_i]] for name in models}
                agreement = sum(1 for v in votes.values() if v == predicted_class) / len(votes)
                results.append(
                    EnsemblePrediction(
                        predicted_class=predicted_class,
                        confidence=confidence,
                        agreement_fraction=agreement,
                        per_model_votes=votes,
                        model_version=m_ver,
                        feature_version=f_ver,
                    )
                )
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
        recent_closes: list[float] = []

        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    recent_trend = recent_closes[-1] - recent_closes[-10]
                    bias = 1.5 if recent_trend < -6 else (-1.5 if recent_trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = float(rng.normal(bias, 1.5))
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + float(rng.normal(bias * 0.5, 1.0)))
                high_p = max(open_p, close_p) + abs(float(rng.normal(0, 0.5)))
                low_p = min(open_p, close_p) - abs(float(rng.normal(0, 0.5)))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)

        return pd.DataFrame(
            rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps)
        )

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
                if sample_predictions is not None:
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
