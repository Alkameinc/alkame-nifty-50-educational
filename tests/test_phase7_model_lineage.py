"""
tests/test_phase7_model_lineage.py
Phase 7 Model Lineage Verification Test Suite (MODEL-001 to MODEL-003).

Tests:
1. MODEL-001: Every trained model artifact (base model & ensemble) records the canonical 12 lineage fields:
   model_version, model_hash, feature_schema_hash, config_hash, dataset_version,
   dataset_start, dataset_end, universe_version, training_commit, python_version,
   dependency_lock_hash, training_timestamp.
2. MODEL-002: Configuration hash sensitivity across all 6 categories:
   horizons, features, thresholds, model parameters, risk settings, calibration parameters.
3. MODEL-003: Model hash verification on disk and tamper detection.
"""

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

import config
from ensemble_manager import EnsembleManager
from model_lineage import (
    REQUIRED_LINEAGE_FIELDS,
    build_lineage_metadata,
    compute_artifact_hash,
    compute_config_hash,
    compute_dataset_version,
    compute_dependency_lock_hash,
    compute_feature_schema_hash,
    get_git_commit_sha,
    get_lineage_config_dict,
    verify_lineage_integrity,
)
from model_trainer import ModelTrainer


# ============================================================================
# Helper: Synthetic Data Generator
# ============================================================================
def _build_synthetic_ohlcv(n_days: int = 15, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-01-01 09:15:00")
    timestamps = []
    for day in range(n_days):
        day_start = base_date + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            ts = day_start + pd.Timedelta(minutes=5 * bar)
            timestamps.append(ts)

    n_bars = len(timestamps)
    returns = rng.normal(0.0002, 0.005, size=n_bars)
    prices = 1000.0 * np.cumprod(1.0 + returns)

    highs = prices * (1.0 + rng.uniform(0.0001, 0.002, size=n_bars))
    lows = prices * (1.0 - rng.uniform(0.0001, 0.002, size=n_bars))
    opens = (highs + lows) / 2.0
    volumes = rng.integers(1000, 50000, size=n_bars)

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        },
        index=pd.DatetimeIndex(timestamps),
    )


# ============================================================================
# 1. MODEL-001 — 12-Field Provenance Record
# ============================================================================
class TestMODEL001ProvenanceRecord:
    """MODEL-001: Every trained model artifact must record the 12 canonical lineage fields."""

    def test_base_model_contains_all_12_required_lineage_fields(self, tmp_path):
        trainer = ModelTrainer()
        symbol = "TEST_P7_BASE"
        horizon = "INTRADAY"

        df_stock = _build_synthetic_ohlcv(n_days=12, bars_per_day=75, seed=101)
        df_index = _build_synthetic_ohlcv(n_days=12, bars_per_day=75, seed=102)
        df_index.index = df_stock.index

        res = trainer.train_for_symbol(symbol, df_stock, df_index, horizon=horizon)
        assert res.success is True

        meta_path = trainer._metadata_path(symbol, horizon)
        assert meta_path.exists(), f"Metadata path {meta_path} does not exist"

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # Invariant: all 12 required fields must exist, be non-empty, and be non-null
        for field in REQUIRED_LINEAGE_FIELDS:
            assert field in meta, f"Required lineage field '{field}' is missing from metadata"
            assert meta[field] is not None, f"Field '{field}' must not be null"
            assert str(meta[field]).strip() != "", f"Field '{field}' must not be empty"

        # Check types and consistency
        assert meta["universe_version"] == "NIFTY50_v1"
        assert meta["python_version"] == platform.python_version()
        assert len(meta["model_hash"]) == 64  # SHA256 hex length
        assert len(meta["feature_schema_hash"]) == 64
        assert len(meta["config_hash"]) == 64
        assert len(meta["dependency_lock_hash"]) == 64

        # Model hash must match the actual file binary on disk
        model_path = trainer._model_path(symbol, horizon)
        actual_model_hash = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
        assert meta["model_hash"] == actual_model_hash

        # Cleanup
        model_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    def test_ensemble_model_contains_all_12_required_lineage_fields(self, tmp_path):
        em = EnsembleManager()
        symbol = "TEST_P7_ENS"
        horizon = "INTRADAY"

        df_stock = _build_synthetic_ohlcv(n_days=12, bars_per_day=75, seed=201)
        df_index = _build_synthetic_ohlcv(n_days=12, bars_per_day=75, seed=202)
        df_index.index = df_stock.index

        res = em.train_ensemble_for_symbol(symbol, df_stock, df_index, horizon=horizon)
        assert res.success is True

        pointer_path = em._current_pointer_path(symbol, horizon)
        with open(pointer_path) as pf:
            curr_run_id = json.load(pf)["current_run_id"]

        run_dir = em._versioned_dir(symbol, horizon, curr_run_id)
        meta_path = run_dir / "metadata.json"
        assert meta_path.exists()

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        for field in REQUIRED_LINEAGE_FIELDS:
            assert field in meta, f"Required lineage field '{field}' is missing from ensemble metadata"
            assert meta[field] is not None, f"Field '{field}' must not be null"
            assert str(meta[field]).strip() != "", f"Field '{field}' must not be empty"

        assert meta["model_hash"] == meta["artifact_sha256"]
        assert meta["training_commit"] == meta["code_commit_sha"]

        # Cleanup
        shutil.rmtree(config.MODELS_DIR / symbol, ignore_errors=True)
        em._ensemble_path(symbol, horizon).unlink(missing_ok=True)
        em._ensemble_metadata_path(symbol, horizon).unlink(missing_ok=True)


# ============================================================================
# 2. MODEL-002 — Configuration Hash Sensitivity Across 6 Categories
# ============================================================================
class TestMODEL002ConfigHashSensitivity:
    """
    MODEL-002: Changes to:
    - horizons
    - features
    - thresholds
    - model parameters
    - risk settings
    - calibration parameters
    must change the configuration hash.
    """

    def test_config_hash_determinism(self):
        """Identical configurations must produce the exact same configuration hash."""
        h1 = compute_config_hash()
        h2 = compute_config_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_sensitivity_horizons_change(self):
        """Modifying horizons parameters alters config_hash."""
        base_hash = compute_config_hash()
        mutated_cfg = get_lineage_config_dict()
        mutated_cfg["horizons"]["horizon_config"]["INTRADAY"]["horizon_bars"] = 10
        mutated_hash = compute_config_hash(mutated_cfg)
        assert mutated_hash != base_hash

    def test_sensitivity_features_change(self):
        """Modifying technical feature parameters alters config_hash."""
        base_hash = compute_config_hash()
        mutated_cfg = get_lineage_config_dict()
        mutated_cfg["features"]["rsi_period"] = 21
        mutated_hash = compute_config_hash(mutated_cfg)
        assert mutated_hash != base_hash

    def test_sensitivity_thresholds_change(self):
        """Modifying relative / market-context thresholds alters config_hash."""
        base_hash = compute_config_hash()
        mutated_cfg = get_lineage_config_dict()
        mutated_cfg["thresholds"]["outperformance_threshold_pct"] = 2.5
        mutated_hash = compute_config_hash(mutated_cfg)
        assert mutated_hash != base_hash

    def test_sensitivity_model_parameters_change(self):
        """Modifying model hyperparameters alters config_hash."""
        base_hash = compute_config_hash()
        mutated_cfg = get_lineage_config_dict()
        mutated_cfg["model_parameters"]["model_learning_rate"] = 0.10
        mutated_hash = compute_config_hash(mutated_cfg)
        assert mutated_hash != base_hash

    def test_sensitivity_risk_settings_change(self):
        """Modifying portfolio or global risk settings alters config_hash."""
        base_hash = compute_config_hash()
        mutated_cfg = get_lineage_config_dict()
        mutated_cfg["risk_settings"]["max_position_size_pct"] = 0.05
        mutated_hash = compute_config_hash(mutated_cfg)
        assert mutated_hash != base_hash

    def test_sensitivity_calibration_parameters_change(self):
        """Modifying calibration binning or ECE thresholds alters config_hash."""
        base_hash = compute_config_hash()
        mutated_cfg = get_lineage_config_dict()
        mutated_cfg["calibration_parameters"]["calibration_ece_threshold"] = 0.05
        mutated_hash = compute_config_hash(mutated_cfg)
        assert mutated_hash != base_hash


# ============================================================================
# 3. MODEL-003 — Artifact Checksum & Tamper Detection
# ============================================================================
class TestMODEL003ArtifactHashAndTamperDetection:
    """MODEL-003: Store model hash and training metadata with tamper detection."""

    def test_model_hash_matches_disk_binary(self):
        dummy_content = b"fake-model-binary-data-for-lineage-test-123"
        hash1 = compute_artifact_hash(dummy_content)
        assert len(hash1) == 64
        # Deterministic
        assert hash1 == hashlib.sha256(dummy_content).hexdigest()

    def test_tamper_detection_in_load_model(self, tmp_path):
        """
        If a model binary file is tampered with or corrupted after saving,
        load_model must fail closed with ValueError (integrity failure).
        """
        trainer = ModelTrainer()
        symbol = "TEST_P7_TAMPER"
        horizon = "INTRADAY"

        df_stock = _build_synthetic_ohlcv(n_days=12, bars_per_day=75, seed=301)
        df_index = _build_synthetic_ohlcv(n_days=12, bars_per_day=75, seed=302)
        df_index.index = df_stock.index

        res = trainer.train_for_symbol(symbol, df_stock, df_index, horizon=horizon)
        assert res.success is True

        # Clean load should succeed
        loaded = trainer.load_model(symbol, horizon=horizon, verify_integrity=True)
        assert loaded is not None

        # Deliberately corrupt model binary on disk
        model_path = trainer._model_path(symbol, horizon)
        with open(model_path, "ab") as f:
            f.write(b"ROUGE_TAMPERED_BYTES")

        # Loading tampered model must fail closed
        with pytest.raises(ValueError, match="Artifact integrity failure"):
            trainer.load_model(symbol, horizon=horizon, verify_integrity=True)

        # Cleanup
        model_path.unlink(missing_ok=True)
        trainer._metadata_path(symbol, horizon).unlink(missing_ok=True)

    def test_dataset_versioning_sensitivity(self):
        """Different observation date ranges or sample counts generate distinct dataset versions."""
        v1 = compute_dataset_version("TCS", "2026-01-01", "2026-02-01", 1000)
        v2 = compute_dataset_version("TCS", "2026-01-01", "2026-02-15", 1500)
        v3 = compute_dataset_version("INFY", "2026-01-01", "2026-02-01", 1000)

        assert v1 != v2
        assert v1 != v3
        assert len(v1) == 16

    def test_verify_lineage_integrity_detects_missing_fields(self, tmp_path):
        """verify_lineage_integrity detects if required fields are missing."""
        dummy_file = tmp_path / "model.bin"
        dummy_file.write_bytes(b"dummy")
        dummy_hash = hashlib.sha256(b"dummy").hexdigest()

        complete_meta = {field: "val" for field in REQUIRED_LINEAGE_FIELDS}
        complete_meta["model_hash"] = dummy_hash

        ok, msg = verify_lineage_integrity(complete_meta, dummy_file)
        assert ok is True

        # Remove a required field
        del complete_meta["training_commit"]
        ok_missing, msg_missing = verify_lineage_integrity(complete_meta, dummy_file)
        assert ok_missing is False
        assert "training_commit" in msg_missing
