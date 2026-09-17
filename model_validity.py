# 1. Standard library imports
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import (
    CALIBRATION_ECE_THRESHOLD,
    EDGE_CHECK_MIN_ALPHA_PCT,
    HORIZON_INTRADAY,
    MODELS_DIR,
)
from health_monitor import (
    ENGINE_HEALTH_FAILED,
    ENGINE_HEALTH_DEGRADED,
    ENGINE_HEALTH_HEALTHY,
    EngineHealthResult,
)
from runtime_validator import (
    RuntimeValidator,
    STATUS_EDGE_CONFIRMED,
    STATUS_INSUFFICIENT_DATA,
    STATUS_SUFFICIENT,
)

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
VALIDITY_UNTRAINED = "UNTRAINED"
VALIDITY_TRAINED = "TRAINED"
VALIDITY_CALIBRATED = "CALIBRATED"
VALIDITY_VALIDATED = "VALIDATED"
VALIDITY_LIVE_ELIGIBLE = "LIVE_ELIGIBLE"
VALIDITY_STALE = "STALE"
VALIDITY_INVALID = "INVALID"

ALL_VALIDITY_STATES = {
    VALIDITY_UNTRAINED,
    VALIDITY_TRAINED,
    VALIDITY_CALIBRATED,
    VALIDITY_VALIDATED,
    VALIDITY_LIVE_ELIGIBLE,
    VALIDITY_STALE,
    VALIDITY_INVALID,
}

CALIBRATION_STATUS_WELL_CALIBRATED = "WELL_CALIBRATED"
CALIBRATION_STATUS_MISCALIBRATED = "MISCALIBRATED"
CALIBRATION_STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
CALIBRATION_STATUS_ADVERSARIAL = "ADVERSARIAL_MISCALIBRATED"
CALIBRATION_STATUS_UNAVAILABLE = "UNAVAILABLE"

EDGE_STATUS_CONFIRMED = "EDGE_CONFIRMED"
EDGE_STATUS_NO_EDGE = "NO_EDGE"
EDGE_STATUS_UNAVAILABLE = "UNAVAILABLE"


@dataclass
class ModelValidityResult:
    """
    MODEL-004: Decoupled Model Validity representation.
    Tracks statistical viability, calibration, and edge separate from engine health.
    """
    symbol: str
    horizon: str
    validity_status: str
    is_live_eligible: bool
    model_version: str | None = None
    trained_at: str | None = None
    age_days: float | None = None
    calibration_status: str = CALIBRATION_STATUS_UNAVAILABLE
    edge_status: str = EDGE_STATUS_UNAVAILABLE
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_timestamp(val: Any) -> datetime | None:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        try:
            clean_str = val.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def evaluate_model_validity(
    symbol: str,
    horizon: str = HORIZON_INTRADAY,
    ensemble_manager: Any = None,
    runtime_validator: RuntimeValidator | None = None,
    calibration_df: pd.DataFrame | None = None,
    strategy_returns: pd.Series | None = None,
    baseline_returns: pd.Series | None = None,
    max_model_age_days: int = 30,
    max_calibration_age_days: int = 90,
) -> ModelValidityResult:
    """
    Evaluates whether the statistical prediction model for a specific ticker and horizon
    is valid and live-eligible according to MODEL-004.
    """
    reasons: list[str] = []
    rv = runtime_validator or RuntimeValidator()

    # 1. Locate Model or Ensemble Artifacts
    artifact_path: Path | None = None
    metadata_path: Path | None = None
    metadata: dict[str, Any] = {}

    if ensemble_manager is None:
        try:
            from ensemble_manager import EnsembleManager
            ensemble_manager = EnsembleManager()
        except Exception as em_err:
            logger.debug(f"Could not import/init EnsembleManager: {em_err}")

    # Check versioned ensemble pointer
    if ensemble_manager is not None and hasattr(ensemble_manager, "_current_pointer_path"):
        pointer_file = ensemble_manager._current_pointer_path(symbol, horizon)
        if pointer_file.exists():
            try:
                with open(pointer_file, encoding="utf-8") as pf:
                    p_data = json.load(pf)
                curr_run_id = p_data.get("current_run_id")
                run_dir = ensemble_manager._versioned_dir(symbol, horizon, curr_run_id)
                e_file = run_dir / "ensemble.joblib"
                m_file = run_dir / "metadata.json"
                if e_file.exists() and m_file.exists():
                    artifact_path = e_file
                    metadata_path = m_file
            except Exception as e:
                logger.warning(f"Error reading ensemble pointer for {symbol} ({horizon}): {e}")

    # Check flat ensemble paths
    if artifact_path is None and ensemble_manager is not None:
        legacy_ensemble = ensemble_manager._ensemble_path(symbol, horizon)
        legacy_meta = ensemble_manager._ensemble_metadata_path(symbol, horizon)
        if legacy_ensemble.exists() and legacy_meta.exists():
            artifact_path = legacy_ensemble
            metadata_path = legacy_meta

    # Check single model path via ModelTrainer
    if artifact_path is None:
        try:
            from model_trainer import ModelTrainer
            trainer = ModelTrainer()
            single_path = trainer._model_path(symbol, horizon, is_level=False)
            single_meta = trainer._metadata_path(symbol, horizon, is_level=False)
            if single_path.exists() and single_meta.exists():
                artifact_path = single_path
                metadata_path = single_meta
        except Exception as mt_err:
            logger.debug(f"Could not check ModelTrainer paths: {mt_err}")

    # If no artifact on disk -> UNTRAINED
    if artifact_path is None or metadata_path is None:
        return ModelValidityResult(
            symbol=symbol,
            horizon=horizon,
            validity_status=VALIDITY_UNTRAINED,
            is_live_eligible=False,
            model_version=None,
            trained_at=None,
            age_days=None,
            calibration_status=CALIBRATION_STATUS_UNAVAILABLE,
            edge_status=EDGE_STATUS_UNAVAILABLE,
            reasons=["No trained model or ensemble artifact exists on disk for this symbol and horizon."],
            metadata={},
        )

    # 2. Check File Readability and Metadata Integrity
    try:
        with open(metadata_path, encoding="utf-8") as mf:
            metadata = json.load(mf)
    except Exception as e:
        return ModelValidityResult(
            symbol=symbol,
            horizon=horizon,
            validity_status=VALIDITY_INVALID,
            is_live_eligible=False,
            reasons=[f"Metadata file exists but could not be parsed: {e}"],
            metadata={},
        )

    # 3. Check Checksum & Lineage Integrity (MODEL-001, MODEL-003)
    expected_sha = metadata.get("artifact_sha256")
    if expected_sha:
        try:
            actual_sha = hashlib.sha256(open(artifact_path, "rb").read()).hexdigest()
            if actual_sha != expected_sha:
                return ModelValidityResult(
                    symbol=symbol,
                    horizon=horizon,
                    validity_status=VALIDITY_INVALID,
                    is_live_eligible=False,
                    model_version=metadata.get("model_version"),
                    reasons=[f"Artifact checksum verification failed (expected {expected_sha[:8]}, got {actual_sha[:8]})."],
                    metadata=metadata,
                )
        except Exception as fe:
            return ModelValidityResult(
                symbol=symbol,
                horizon=horizon,
                validity_status=VALIDITY_INVALID,
                is_live_eligible=False,
                reasons=[f"Failed reading artifact file for integrity verification: {fe}"],
                metadata=metadata,
            )

    # Verify model_hash lineage integrity if single model
    if "model_hash" in metadata:
        from model_lineage import verify_lineage_integrity
        ok, lineage_reason = verify_lineage_integrity(metadata, artifact_path)
        if not ok:
            return ModelValidityResult(
                symbol=symbol,
                horizon=horizon,
                validity_status=VALIDITY_INVALID,
                is_live_eligible=False,
                model_version=metadata.get("model_version"),
                reasons=[f"Model lineage integrity failed: {lineage_reason}"],
                metadata=metadata,
            )

    # 4. Check Model Age / Freshness
    model_ver = metadata.get("model_version")
    trained_at_str = metadata.get("trained_at") or metadata.get("created_at")
    trained_dt = _parse_timestamp(trained_at_str)
    if trained_dt is None:
        try:
            trained_dt = datetime.fromtimestamp(os.path.getmtime(artifact_path), tz=timezone.utc)
            trained_at_str = trained_dt.isoformat()
        except Exception:
            trained_dt = datetime.now(timezone.utc)
            trained_at_str = trained_dt.isoformat()

    now_utc = datetime.now(timezone.utc)
    age_days = max(0.0, (now_utc - trained_dt).total_seconds() / 86400.0)

    is_stale = age_days > max_model_age_days
    if is_stale:
        reasons.append(
            f"Model artifact is stale (age={age_days:.1f} days, max allowable={max_model_age_days} days)."
        )

    # 5. Check Calibration
    calibration_status = CALIBRATION_STATUS_UNAVAILABLE
    is_well_calibrated = False

    if calibration_df is not None and not calibration_df.empty:
        cal_res = rv.compute_calibration(calibration_df)
        # Check adversarial miscalibration conditions
        adversarial_found = False
        for r in cal_res.reasons:
            if any(k in r for k in ["Zero resolution", "Inverted confidence", "Severe class imbalance"]):
                adversarial_found = True
                reasons.append(f"Adversarial miscalibration detected: {r}")

        if adversarial_found:
            return ModelValidityResult(
                symbol=symbol,
                horizon=horizon,
                validity_status=VALIDITY_INVALID,
                is_live_eligible=False,
                model_version=model_ver,
                trained_at=trained_at_str,
                age_days=age_days,
                calibration_status=CALIBRATION_STATUS_ADVERSARIAL,
                edge_status=EDGE_STATUS_UNAVAILABLE,
                reasons=reasons,
                metadata=metadata,
            )

        if cal_res.is_well_calibrated:
            is_well_calibrated = True
            calibration_status = CALIBRATION_STATUS_WELL_CALIBRATED
        elif cal_res.status == STATUS_INSUFFICIENT_DATA:
            calibration_status = CALIBRATION_STATUS_INSUFFICIENT_DATA
            reasons.append("Insufficient historical predictions to verify confidence calibration.")
        else:
            calibration_status = CALIBRATION_STATUS_MISCALIBRATED
            reasons.append(
                f"Model miscalibrated: ECE={cal_res.expected_calibration_error:.3f} > threshold={rv.ece_threshold:.3f}."
            )
    else:
        # Fallback to metadata metrics if calibration_df not supplied
        metrics = metadata.get("metrics") or metadata.get("validation_metrics") or {}
        ece = metrics.get("expected_calibration_error") or metadata.get("expected_calibration_error")
        if ece is not None:
            if float(ece) <= CALIBRATION_ECE_THRESHOLD:
                is_well_calibrated = True
                calibration_status = CALIBRATION_STATUS_WELL_CALIBRATED
            else:
                calibration_status = CALIBRATION_STATUS_MISCALIBRATED
                reasons.append(f"Recorded ECE={float(ece):.3f} exceeds threshold={CALIBRATION_ECE_THRESHOLD:.3f}.")
        else:
            calibration_status = CALIBRATION_STATUS_UNAVAILABLE
            reasons.append("No calibration evaluation data supplied or recorded in metadata.")

    # 6. Check Edge vs Baseline
    edge_status = EDGE_STATUS_UNAVAILABLE
    is_edge_confirmed = False

    if strategy_returns is not None and baseline_returns is not None:
        edge_res = rv.compute_edge_vs_baseline(strategy_returns, baseline_returns)
        if edge_res.status == STATUS_EDGE_CONFIRMED:
            is_edge_confirmed = True
            edge_status = EDGE_STATUS_CONFIRMED
        else:
            edge_status = EDGE_STATUS_NO_EDGE
            reasons.append(
                f"No confirmed alpha edge: strategy alpha={edge_res.alpha_pct:.3f}% <= min={rv.min_alpha_pct:.3f}%."
            )
    else:
        # Check metadata for edge records
        extra_meta = metadata.get("extra_metadata", {})
        meta_alpha = extra_meta.get("alpha_pct") or metadata.get("alpha_pct")
        meta_edge = extra_meta.get("edge_status") or metadata.get("edge_status")
        if meta_edge == STATUS_EDGE_CONFIRMED or (meta_alpha is not None and float(meta_alpha) > EDGE_CHECK_MIN_ALPHA_PCT):
            is_edge_confirmed = True
            edge_status = EDGE_STATUS_CONFIRMED
        elif meta_edge is not None:
            edge_status = meta_edge
            reasons.append(f"Recorded edge check status: {meta_edge}.")
        else:
            edge_status = EDGE_STATUS_UNAVAILABLE
            reasons.append("No baseline alpha edge data supplied or recorded in metadata.")

    # 7. Overall Validity Resolution
    if is_stale:
        validity_status = VALIDITY_STALE
        is_live_eligible = False
    elif is_well_calibrated and is_edge_confirmed:
        validity_status = VALIDITY_LIVE_ELIGIBLE
        is_live_eligible = True
        reasons.append("Model is well-calibrated and has confirmed alpha edge against baseline.")
    elif is_well_calibrated:
        validity_status = VALIDITY_CALIBRATED
        is_live_eligible = False
        reasons.append("Model is well-calibrated but lacks confirmed alpha edge.")
    elif is_edge_confirmed:
        validity_status = VALIDITY_VALIDATED
        is_live_eligible = False
        reasons.append("Model has confirmed alpha edge but lacks proven probability calibration.")
    else:
        validity_status = VALIDITY_TRAINED
        is_live_eligible = False
        reasons.append("Model artifact is trained but unverified for calibration and edge.")

    return ModelValidityResult(
        symbol=symbol,
        horizon=horizon,
        validity_status=validity_status,
        is_live_eligible=is_live_eligible,
        model_version=model_ver,
        trained_at=trained_at_str,
        age_days=round(age_days, 2) if age_days is not None else None,
        calibration_status=calibration_status,
        edge_status=edge_status,
        reasons=reasons,
        metadata=metadata,
    )


def can_serve_live_signal(
    engine_health: EngineHealthResult | str,
    model_validity: ModelValidityResult,
) -> tuple[bool, list[str]]:
    """
    Core Invariant (MODEL-004):
    Decouple Engine Health from Model Validity.
    A live trading signal can ONLY be served if:
      1. Engine Health is NOT FAILED.
      2. Model Validity is strictly LIVE_ELIGIBLE.
    A healthy engine does NOT imply a valid model!
    """
    reasons: list[str] = []
    engine_status = engine_health.status if hasattr(engine_health, "status") else str(engine_health)

    if engine_status == ENGINE_HEALTH_FAILED:
        reasons.append("Engine health is FAILED: infrastructure components are offline or failing.")

    if not model_validity.is_live_eligible:
        reasons.append(
            f"Model validity is '{model_validity.validity_status}' (must be '{VALIDITY_LIVE_ELIGIBLE}'). "
            f"Reasons: {'; '.join(model_validity.reasons)}"
        )

    can_serve = (engine_status != ENGINE_HEALTH_FAILED) and model_validity.is_live_eligible
    if can_serve:
        reasons.append("Both engine health and model validity are verified for live signal serving.")

    return can_serve, reasons
