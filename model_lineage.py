# 1. Standard library imports
import copy
import hashlib
import json
import logging
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# 2. Third-party imports
import joblib
import numpy as np
import pandas as pd
import sklearn

# 3. Local imports
import config

# 4. Logger setup
logger = logging.getLogger(__name__)

# Canonical 12-field list required by Phase 7 (MODEL-001)
REQUIRED_LINEAGE_FIELDS = [
    "model_version",
    "model_hash",
    "feature_schema_hash",
    "config_hash",
    "dataset_version",
    "dataset_start",
    "dataset_end",
    "universe_version",
    "training_commit",
    "python_version",
    "dependency_lock_hash",
    "training_timestamp",
]


@dataclass
class ModelLineageRecord:
    """
    MODEL-001: Canonical 12-field model artifact lineage record.
    Guarantees end-to-end reproducibility, traceability, and tamper-resistance.
    """

    model_version: str
    model_hash: str
    feature_schema_hash: str
    config_hash: str
    dataset_version: str
    dataset_start: str | None
    dataset_end: str | None
    universe_version: str
    training_commit: str
    python_version: str
    dependency_lock_hash: str
    training_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_git_commit_sha() -> str:
    """Retrieves current Git commit SHA or returns 'UNKNOWN' if not in a git repo."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
                cwd=str(config.PROJECT_ROOT),
            )
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return "UNKNOWN"


def get_lineage_config_dict() -> dict[str, Any]:
    """
    MODEL-002: Extracts configuration values across the 6 mandatory categories:
    1. horizons
    2. features
    3. thresholds
    4. model parameters
    5. risk settings
    6. calibration parameters
    """
    return copy.deepcopy({
        "horizons": {
            "all_horizons": list(config.ALL_HORIZONS),
            "horizon_config": config.HORIZON_CONFIG,
            "horizon_to_ma": config.HORIZON_TO_MA_PERIOD,
            "horizon_to_sr": config.HORIZON_TO_SR_METHOD,
            "event_impact_horizon": config.EVENT_IMPACT_HORIZON,
        },
        "features": {
            "orb_minutes": config.ORB_MINUTES,
            "gap_threshold_pct": config.GAP_THRESHOLD_PCT,
            "volume_spike_multiplier": config.VOLUME_SPIKE_MULTIPLIER,
            "volume_spike_lookback_bars": config.VOLUME_SPIKE_LOOKBACK_BARS,
            "rsi_period": config.RSI_PERIOD,
            "rsi_overbought": config.RSI_OVERBOUGHT,
            "rsi_oversold": config.RSI_OVERSOLD,
            "macd_fast": config.MACD_FAST,
            "macd_slow": config.MACD_SLOW,
            "macd_signal": config.MACD_SIGNAL,
            "bollinger_period": config.BOLLINGER_PERIOD,
            "bollinger_std_dev": config.BOLLINGER_STD_DEV,
            "atr_period": config.ATR_PERIOD,
            "atr_expansion_multiplier": config.ATR_EXPANSION_MULTIPLIER,
            "ma_fast_period": config.MA_FAST_PERIOD,
            "ma_slow_period": config.MA_SLOW_PERIOD,
            "low_liquidity_volume_floor": config.LOW_LIQUIDITY_VOLUME_FLOOR,
        },
        "thresholds": {
            "outperformance_threshold_pct": config.OUTPERFORMANCE_THRESHOLD_PCT,
            "correlation_lookback_bars": config.CORRELATION_LOOKBACK_BARS,
            "correlation_breakdown_threshold": config.CORRELATION_BREAKDOWN_THRESHOLD,
            "data_staleness_minutes": config.DATA_STALENESS_THRESHOLD_MINUTES,
            "data_staleness_trading_days": config.DATA_STALENESS_THRESHOLD_TRADING_DAYS,
        },
        "model_parameters": {
            "model_random_seed": config.MODEL_RANDOM_SEED,
            "model_n_estimators": config.MODEL_N_ESTIMATORS,
            "model_max_depth": config.MODEL_MAX_DEPTH,
            "model_learning_rate": config.MODEL_LEARNING_RATE,
            "min_training_samples": config.MIN_TRAINING_SAMPLES_PER_STOCK,
            "test_fraction": config.TIME_SERIES_SPLIT_TEST_FRACTION,
            "label_classes": list(config.LABEL_CLASSES),
            "ensemble_model_types": list(config.ENSEMBLE_MODEL_TYPES),
            "ensemble_rf_n_estimators": config.ENSEMBLE_RF_N_ESTIMATORS,
            "ensemble_rf_max_depth": config.ENSEMBLE_RF_MAX_DEPTH,
            "ensemble_lr_max_iter": config.ENSEMBLE_LR_MAX_ITER,
        },
        "risk_settings": {
            "global_risk_warn": config.GLOBAL_RISK_ZSCORE_WARN_THRESHOLD,
            "global_risk_crisis": config.GLOBAL_RISK_ZSCORE_CRISIS_THRESHOLD,
            "global_risk_downgrade_elevated": config.GLOBAL_RISK_CONFIDENCE_DOWNGRADE_ELEVATED,
            "global_risk_downgrade_crisis": config.GLOBAL_RISK_CONFIDENCE_DOWNGRADE_CRISIS,
            "global_risk_lookback_days": config.GLOBAL_RISK_LOOKBACK_DAYS,
            "portfolio_total_capital": config.PORTFOLIO_TOTAL_CAPITAL,
            "max_position_size_pct": config.MAX_POSITION_SIZE_PCT,
            "max_dca_steps": config.MAX_DCA_STEPS,
            "slippage_bps": config.SLIPPAGE_BPS,
            "transaction_cost_bps": config.TRANSACTION_COST_BPS,
        },
        "calibration_parameters": {
            "min_calibration_samples": config.MIN_CALIBRATION_SAMPLES,
            "calibration_n_bins": config.CALIBRATION_N_BINS,
            "calibration_ece_threshold": config.CALIBRATION_ECE_THRESHOLD,
            "edge_check_min_alpha_pct": config.EDGE_CHECK_MIN_ALPHA_PCT,
        },
    })


def compute_config_hash(custom_config: dict[str, Any] | None = None) -> str:
    """
    MODEL-002: Computes a deterministic SHA256 configuration hash.
    Any changes to horizons, features, thresholds, model parameters,
    risk settings, or calibration parameters will produce a distinct hash.
    """
    cfg = custom_config if custom_config is not None else get_lineage_config_dict()
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_feature_schema_hash(feature_columns: list[str]) -> str:
    """Computes a deterministic SHA256 hash for feature columns and order."""
    schema_str = ",".join(sorted(feature_columns))
    return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()


def compute_dependency_lock_hash() -> str:
    """Computes a deterministic hash of installed environment packages and lockfile."""
    env_info = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    pyproject_path = config.PROJECT_ROOT / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                env_info["pyproject_sha256"] = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

    serialized = json.dumps(env_info, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_artifact_hash(path_or_bytes: Path | bytes) -> str:
    """
    MODEL-003: Computes SHA256 checksum of the trained model binary on disk or in memory.
    """
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found at {p}")
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    elif isinstance(path_or_bytes, bytes):
        return hashlib.sha256(path_or_bytes).hexdigest()
    else:
        raise TypeError(f"Expected Path or bytes, got {type(path_or_bytes)}")


def compute_dataset_version(
    symbol: str,
    dataset_start: str | None,
    dataset_end: str | None,
    sample_count: int = 0,
) -> str:
    """Computes a deterministic dataset version identifier."""
    key = f"{symbol}:{dataset_start}:{dataset_end}:{sample_count}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def build_lineage_metadata(
    symbol: str,
    horizon: str,
    feature_columns: list[str],
    artifact_path_or_hash: Path | str,
    dataset_start: str | None = None,
    dataset_end: str | None = None,
    sample_count: int = 0,
    model_version: str = "v1.0",
    universe_version: str = "NIFTY50_v1",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    MODEL-001 & MODEL-003: Constructs the complete metadata dictionary
    recording all 12 required lineage fields and preserves domain-specific fields.
    """
    if isinstance(artifact_path_or_hash, Path):
        model_hash = compute_artifact_hash(artifact_path_or_hash)
    else:
        model_hash = str(artifact_path_or_hash)

    feature_schema_hash = compute_feature_schema_hash(feature_columns)
    config_hash = compute_config_hash()
    training_commit = get_git_commit_sha()
    python_version = platform.python_version()
    dependency_lock_hash = compute_dependency_lock_hash()
    training_timestamp = datetime.now().isoformat()
    dataset_version = compute_dataset_version(symbol, dataset_start, dataset_end, sample_count)

    lineage = ModelLineageRecord(
        model_version=model_version,
        model_hash=model_hash,
        feature_schema_hash=feature_schema_hash,
        config_hash=config_hash,
        dataset_version=dataset_version,
        dataset_start=dataset_start,
        dataset_end=dataset_end,
        universe_version=universe_version,
        training_commit=training_commit,
        python_version=python_version,
        dependency_lock_hash=dependency_lock_hash,
        training_timestamp=training_timestamp,
    )

    metadata = lineage.to_dict()

    # Add backwards-compatible and operational metadata
    metadata["symbol"] = symbol
    metadata["horizon"] = horizon
    metadata["feature_columns"] = feature_columns
    metadata["artifact_sha256"] = model_hash  # backward compatibility alias
    metadata["code_commit_sha"] = training_commit  # backward compatibility alias
    metadata["trained_at"] = training_timestamp  # backward compatibility alias
    metadata["training_sample_count"] = sample_count

    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in metadata:
                metadata[k] = v

    return metadata


def verify_lineage_integrity(metadata: dict[str, Any], model_path: Path) -> tuple[bool, str]:
    """
    MODEL-003: Verifies that metadata contains all 12 required fields and that
    the model binary on disk matches the recorded model_hash.
    """
    # 1. Verify required fields presence and non-emptiness
    missing_fields = []
    for field in REQUIRED_LINEAGE_FIELDS:
        if field not in metadata or metadata[field] is None or metadata[field] == "":
            missing_fields.append(field)

    if missing_fields:
        return False, f"Missing required lineage fields: {missing_fields}"

    # 2. Verify model artifact checksum on disk
    if not model_path.exists():
        return False, f"Model file not found on disk: {model_path}"

    expected_hash = metadata.get("model_hash") or metadata.get("artifact_sha256")
    actual_hash = compute_artifact_hash(model_path)

    if actual_hash != expected_hash:
        return (
            False,
            f"Model artifact integrity failure! Expected SHA256={expected_hash}, actual={actual_hash}",
        )

    return True, "Model artifact lineage and integrity verified"
