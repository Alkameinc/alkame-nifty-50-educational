"""
Test F22: Legacy artifact naming mismatch

Tests that ensemble_manager and data_fetcher can correctly identify, load, and warn
about legacy artifacts that lack horizon/interval suffixes.

Test Strategy:
- NORMAL: Properly suffixed artifacts load without warnings
- BOUNDARY: Legacy unsuffixed artifacts load with appropriate warnings
- FAILURE: Missing/corrupted artifacts are handled gracefully
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

# Configure minimal logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory for isolated testing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for isolated testing."""
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    return cache_dir


@pytest.fixture
def mock_ensemble_bundle():
    """Create a minimal mock ensemble bundle for testing."""
    from sklearn.linear_model import LogisticRegression

    mock_model = LogisticRegression(random_state=42)
    # Fit on dummy data so it's a valid model
    X_dummy = np.array([[1, 2], [3, 4], [5, 6]])
    y_dummy = np.array([0, 1, 2])
    mock_model.fit(X_dummy, y_dummy)

    return {
        "models": {"lr": mock_model},
        "classes": ["DOWN", "FLAT", "UP"],
    }


@pytest.fixture
def mock_metadata():
    """Create minimal mock metadata."""
    return {
        "model_id": "test-id-12345",
        "run_id": "20260914T120000_abcd123_xyz789",
        "symbol": "TESTSTOCK",
        "horizon": "INTRADAY",
        "model_version": "v1.1",
        "trained_at": "2026-09-14T12:00:00",
        "feature_columns": ["rsi_feat", "macd_feat"],
        "ensemble_accuracy": 0.75,
        "mean_agreement": 0.80,
    }


# ============================================================================
# ENSEMBLE MANAGER TESTS
# ============================================================================


def test_f22_normal_horizon_suffixed_model_loads(temp_models_dir, mock_ensemble_bundle, mock_metadata):
    """
    NORMAL: Model with proper horizon suffix loads successfully.
    
    Input: TESTSTOCK_INTRADAY_ensemble.joblib with matching metadata
    Expected: Loads without warnings, metadata horizon matches request
    """
    from ensemble_manager import EnsembleManager

    # Setup: Create properly named artifact
    symbol = "TESTSTOCK"
    horizon = "INTRADAY"
    
    artifact_path = temp_models_dir / f"{symbol}_{horizon}_ensemble.joblib"
    metadata_path = temp_models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
    
    joblib.dump(mock_ensemble_bundle, artifact_path)
    with open(metadata_path, "w") as f:
        json.dump(mock_metadata, f)
    
    # Test: Load with patched MODELS_DIR
    with patch("ensemble_manager.MODELS_DIR", temp_models_dir):
        manager = EnsembleManager()
        result = manager.load_ensemble(symbol, horizon)
    
    # Verify: Successfully loaded
    assert result is not None, "Should load horizon-suffixed artifact"
    models, classes, metadata = result
    assert "lr" in models, "Should contain mock model"
    assert classes == ["DOWN", "FLAT", "UP"], "Should have correct classes"
    assert metadata["horizon"] == horizon, "Metadata horizon should match"
    
    print(f"✓ NORMAL case passed: {artifact_path.name} loaded successfully")


def test_f22_boundary_legacy_unsuffixed_model_with_metadata(
    temp_models_dir, mock_ensemble_bundle, mock_metadata, caplog
):
    """
    BOUNDARY: Legacy unsuffixed model WITH metadata loads with warnings.
    
    Input: TESTSTOCK_ensemble.joblib (no horizon suffix) but metadata specifies horizon
    Expected: Loads successfully, logs LEGACY_UNSUFFIXED warning, uses metadata horizon
    """
    from ensemble_manager import EnsembleManager

    symbol = "TESTSTOCK"
    horizon = "INTRADAY"
    
    # Setup: Create legacy unsuffixed artifact
    legacy_artifact = temp_models_dir / f"{symbol}_ensemble.joblib"
    legacy_metadata = temp_models_dir / f"{symbol}_ensemble_metadata.json"
    
    joblib.dump(mock_ensemble_bundle, legacy_artifact)
    with open(legacy_metadata, "w") as f:
        json.dump(mock_metadata, f)
    
    # Test: Load with patched MODELS_DIR
    with patch("ensemble_manager.MODELS_DIR", temp_models_dir):
        with caplog.at_level(logging.WARNING):
            manager = EnsembleManager()
            result = manager.load_ensemble(symbol, horizon)
    
    # Verify: Loaded but with warning
    assert result is not None, "Should load legacy unsuffixed artifact"
    models, classes, metadata = result
    assert metadata["horizon"] == horizon, "Should have metadata horizon"
    
    # Check for appropriate warning
    assert any(
        "LEGACY" in record.message and "unsuffixed" in record.message.lower()
        for record in caplog.records
    ), "Should warn about legacy unsuffixed artifact"
    
    print(f"✓ BOUNDARY case passed: {legacy_artifact.name} loaded with legacy warning")


def test_f22_boundary_legacy_unsuffixed_without_metadata(
    temp_models_dir, mock_ensemble_bundle, caplog
):
    """
    BOUNDARY: Legacy unsuffixed model WITHOUT horizon metadata loads with provenance warning.
    
    Input: TESTSTOCK_ensemble.joblib with no metadata (or metadata lacking horizon)
    Expected: Loads, infers horizon from request, logs PROVENANCE UNKNOWN warning
    """
    from ensemble_manager import EnsembleManager

    symbol = "TESTSTOCK"
    horizon = "3D"
    
    # Setup: Create legacy artifact without metadata
    legacy_artifact = temp_models_dir / f"{symbol}_ensemble.joblib"
    joblib.dump(mock_ensemble_bundle, legacy_artifact)
    # No metadata file created
    
    # Test
    with patch("ensemble_manager.MODELS_DIR", temp_models_dir):
        with caplog.at_level(logging.WARNING):
            manager = EnsembleManager()
            result = manager.load_ensemble(symbol, horizon)
    
    # Verify
    assert result is not None, "Should load legacy artifact even without metadata"
    models, classes, metadata = result
    assert metadata["horizon"] == horizon, "Should infer horizon from request"
    assert metadata.get("provenance_warning") == "INFERRED_FROM_REQUEST"
    
    # Check for provenance warning
    assert any(
        "PROVENANCE UNKNOWN" in record.message
        for record in caplog.records
    ), "Should warn about unknown provenance"
    
    print(f"✓ BOUNDARY case passed: {legacy_artifact.name} loaded with provenance warning")


def test_f22_boundary_horizon_mismatch(temp_models_dir, mock_ensemble_bundle, caplog):
    """
    BOUNDARY: Legacy artifact with metadata specifying DIFFERENT horizon than requested.
    
    Input: Request 3D but metadata says INTRADAY
    Expected: Loads with compatibility warning
    """
    from ensemble_manager import EnsembleManager

    symbol = "TESTSTOCK"
    requested_horizon = "3D"
    metadata_horizon = "INTRADAY"
    
    # Setup: Legacy artifact with mismatched horizon
    legacy_artifact = temp_models_dir / f"{symbol}_ensemble.joblib"
    legacy_metadata = temp_models_dir / f"{symbol}_ensemble_metadata.json"
    
    joblib.dump(mock_ensemble_bundle, legacy_artifact)
    
    meta = {"horizon": metadata_horizon, "feature_columns": ["rsi_feat"], "ensemble_accuracy": 0.7}
    with open(legacy_metadata, "w") as f:
        json.dump(meta, f)
    
    # Test
    with patch("ensemble_manager.MODELS_DIR", temp_models_dir):
        with caplog.at_level(logging.WARNING):
            manager = EnsembleManager()
            result = manager.load_ensemble(symbol, requested_horizon)
    
    # Verify
    assert result is not None, "Should still load despite mismatch"
    
    # Check for compatibility warning
    assert any(
        "COMPATIBILITY WARNING" in record.message and metadata_horizon in record.message
        for record in caplog.records
    ), "Should warn about horizon mismatch"
    
    print(f"✓ BOUNDARY case passed: Horizon mismatch ({requested_horizon} vs {metadata_horizon}) detected")


def test_f22_failure_no_artifact_found(temp_models_dir, caplog):
    """
    FAILURE: No artifact exists at any fallback path.
    
    Input: Request non-existent symbol/horizon
    Expected: Returns None, logs error listing all checked paths
    """
    from ensemble_manager import EnsembleManager

    symbol = "NONEXISTENT"
    horizon = "INTRADAY"
    
    # Test: Try to load non-existent artifact
    with patch("ensemble_manager.MODELS_DIR", temp_models_dir):
        with caplog.at_level(logging.ERROR):
            manager = EnsembleManager()
            result = manager.load_ensemble(symbol, horizon)
    
    # Verify
    assert result is None, "Should return None for missing artifact"
    
    # Check error message lists all checked paths
    error_messages = " ".join(record.message for record in caplog.records if record.levelno == logging.ERROR)
    assert "Versioned:" in error_messages, "Should mention versioned path"
    assert "Flat w/ horizon:" in error_messages, "Should mention horizon-suffixed path"
    assert "Legacy flat:" in error_messages, "Should mention legacy unsuffixed path"
    
    print(f"✓ FAILURE case passed: Missing artifact handled gracefully")


def test_f22_inventory_legacy_artifacts(temp_models_dir, mock_ensemble_bundle):
    """
    Test inventory_legacy_artifacts() correctly identifies and categorizes artifacts.
    
    Creates mix of properly named, legacy, and ambiguous artifacts and verifies categorization.
    """
    from ensemble_manager import EnsembleManager

    # Setup: Create various artifact types
    artifacts = [
        # Properly named with horizon
        ("STOCK1_INTRADAY_ensemble.joblib", {"horizon": "INTRADAY"}),
        # Legacy unsuffixed with metadata
        ("STOCK2_ensemble.joblib", {"horizon": "3D"}),
        # Legacy unsuffixed without metadata
        ("STOCK3_ensemble.joblib", None),
        # Horizon in filename but mismatched metadata
        ("STOCK4_7D_ensemble.joblib", {"horizon": "INTRADAY"}),
    ]
    
    for filename, metadata in artifacts:
        path = temp_models_dir / filename
        joblib.dump(mock_ensemble_bundle, path)
        
        if metadata:
            meta_path = temp_models_dir / filename.replace("_ensemble.joblib", "_ensemble_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
    
    # Test: Run inventory
    with patch("ensemble_manager.MODELS_DIR", temp_models_dir):
        manager = EnsembleManager()
        inventory = manager.inventory_legacy_artifacts()
    
    # Verify: Check categorization
    assert len(inventory) == 4, "Should find all 4 artifacts"
    
    # STOCK1: Properly named, should be COMPATIBLE
    stock1_info = inventory.get("STOCK1_INTRADAY_ensemble.joblib")
    assert stock1_info is not None
    assert stock1_info["compatibility_status"] == "COMPATIBLE"
    assert stock1_info["filename_horizon"] == "INTRADAY"
    assert stock1_info["metadata_horizon"] == "INTRADAY"
    
    # STOCK2: Legacy unsuffixed but metadata specifies horizon
    stock2_info = inventory.get("STOCK2_ensemble.joblib")
    assert stock2_info is not None
    assert stock2_info["compatibility_status"] == "LEGACY_UNSUFFIXED"
    assert stock2_info["metadata_horizon"] == "3D"
    assert stock2_info["filename_horizon"] is None
    
    # STOCK3: No horizon anywhere
    stock3_info = inventory.get("STOCK3_ensemble.joblib")
    assert stock3_info is not None
    assert stock3_info["compatibility_status"] == "UNKNOWN"
    
    # STOCK4: Mismatch between filename and metadata
    stock4_info = inventory.get("STOCK4_7D_ensemble.joblib")
    assert stock4_info is not None
    assert stock4_info["compatibility_status"] == "AMBIGUOUS"
    assert stock4_info["filename_horizon"] == "7D"
    assert stock4_info["metadata_horizon"] == "INTRADAY"
    
    print(f"✓ Inventory test passed: Correctly categorized {len(inventory)} artifacts")


# ============================================================================
# DATA FETCHER CACHE TESTS
# ============================================================================


def test_f22_cache_normal_interval_suffixed(temp_cache_dir):
    """
    NORMAL: Cache file with interval suffix loads correctly.
    
    Input: RELIANCE_NS_5m.csv
    Expected: Loads without warnings
    """
    from data_fetcher import DataFetcher

    ticker = "RELIANCE.NS"
    interval = "5m"
    
    # Setup: Create properly suffixed cache
    cache_data = pd.DataFrame(
        {
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000, 1100],
        },
        index=pd.date_range("2026-09-14 09:15", periods=2, freq="5min", tz="Asia/Kolkata"),
    )
    
    cache_file = temp_cache_dir / "RELIANCE_NS_5m.csv"
    cache_data.to_csv(cache_file)
    
    # Test
    with patch("data_fetcher.CACHE_DIR", temp_cache_dir):
        fetcher = DataFetcher(cache_dir=temp_cache_dir)
        result = fetcher._load_cache(ticker, interval=interval)
    
    # Verify
    assert result is not None, "Should load interval-suffixed cache"
    assert len(result) == 2, "Should have 2 rows"
    assert "Close" in result.columns
    
    print(f"✓ NORMAL cache case passed: {cache_file.name} loaded successfully")


def test_f22_cache_boundary_legacy_unsuffixed(temp_cache_dir, caplog):
    """
    BOUNDARY: Legacy cache file without interval suffix loads with warning.
    
    Input: ICICIBANK_NS.csv (no interval suffix)
    Expected: Loads with legacy compatibility warning
    """
    from data_fetcher import DataFetcher

    ticker = "ICICIBANK.NS"
    interval = "5m"
    
    # Setup: Create legacy unsuffixed cache
    cache_data = pd.DataFrame(
        {
            "Open": [200, 201],
            "High": [202, 203],
            "Low": [199, 200],
            "Close": [201, 202],
            "Volume": [2000, 2100],
        },
        index=pd.date_range("2026-09-14 09:15", periods=2, freq="5min", tz="Asia/Kolkata"),
    )
    
    legacy_cache = temp_cache_dir / "ICICIBANK_NS.csv"  # No interval suffix
    cache_data.to_csv(legacy_cache)
    
    # Test
    with patch("data_fetcher.CACHE_DIR", temp_cache_dir):
        with caplog.at_level(logging.WARNING):
            fetcher = DataFetcher(cache_dir=temp_cache_dir)
            result = fetcher._load_cache(ticker, interval=interval)
    
    # Verify
    assert result is not None, "Should load legacy unsuffixed cache"
    assert len(result) == 2
    
    # Check for compatibility warning
    assert any(
        "LEGACY CACHE COMPATIBILITY" in record.message
        for record in caplog.records
    ), "Should warn about legacy unsuffixed cache"
    
    print(f"✓ BOUNDARY cache case passed: {legacy_cache.name} loaded with legacy warning")


def test_f22_cache_failure_missing(temp_cache_dir):
    """
    FAILURE: No cache file exists at any path.
    
    Input: Non-existent ticker
    Expected: Returns None
    """
    from data_fetcher import DataFetcher

    ticker = "NONEXISTENT.NS"
    interval = "1d"
    
    # Test: Try to load non-existent cache
    with patch("data_fetcher.CACHE_DIR", temp_cache_dir):
        fetcher = DataFetcher(cache_dir=temp_cache_dir)
        result = fetcher._load_cache(ticker, interval=interval)
    
    # Verify
    assert result is None, "Should return None for missing cache"
    
    print(f"✓ FAILURE cache case passed: Missing cache handled gracefully")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
