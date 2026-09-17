# 1. Standard library imports
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Third-party imports
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

# 3. Local imports
from api import app
import config
from health_monitor import (
    ENGINE_HEALTH_DEGRADED,
    ENGINE_HEALTH_FAILED,
    ENGINE_HEALTH_HEALTHY,
    EngineHealthResult,
    HealthRegistry,
)
from model_validity import (
    ALL_VALIDITY_STATES,
    CALIBRATION_STATUS_ADVERSARIAL,
    CALIBRATION_STATUS_INSUFFICIENT_DATA,
    CALIBRATION_STATUS_MISCALIBRATED,
    CALIBRATION_STATUS_WELL_CALIBRATED,
    EDGE_STATUS_CONFIRMED,
    EDGE_STATUS_NO_EDGE,
    ModelValidityResult,
    VALIDITY_CALIBRATED,
    VALIDITY_INVALID,
    VALIDITY_LIVE_ELIGIBLE,
    VALIDITY_STALE,
    VALIDITY_TRAINED,
    VALIDITY_UNTRAINED,
    VALIDITY_VALIDATED,
    can_serve_live_signal,
    evaluate_model_validity,
)
from runtime_validator import (
    CalibrationResult,
    EdgeCheckResult,
    RuntimeValidator,
    STATUS_EDGE_CONFIRMED,
    STATUS_NO_EDGE,
    STATUS_SUFFICIENT,
)


class TestEngineHealthChecks(unittest.TestCase):
    """
    Test the 6 engine infrastructure health checks:
    api, data, events, scheduler, storage, model_file_availability.
    """

    def setUp(self):
        self.registry = HealthRegistry()

    def test_all_six_checks_present(self):
        result = self.registry.get_engine_health()
        self.assertIsInstance(result, EngineHealthResult)
        expected_checks = {"api", "data", "events", "scheduler", "storage", "model_file_availability"}
        self.assertEqual(set(result.checks.keys()), expected_checks)
        self.assertIn(result.status, [ENGINE_HEALTH_HEALTHY, ENGINE_HEALTH_DEGRADED, ENGINE_HEALTH_FAILED])

    def test_engine_health_healthy_when_all_operational(self):
        self.registry.report("data_fetcher", ok=True)
        self.registry.report("scheduler", ok=True)
        self.registry.report("corporate_events_fetcher", ok=True)
        self.registry.report("news_sentiment_fetcher", ok=True)
        self.registry.report("event_classifier", ok=True)

        result = self.registry.get_engine_health()
        self.assertEqual(result.checks["data"], ENGINE_HEALTH_HEALTHY)
        self.assertEqual(result.checks["scheduler"], ENGINE_HEALTH_HEALTHY)
        self.assertEqual(result.checks["events"], ENGINE_HEALTH_HEALTHY)
        self.assertEqual(result.checks["storage"], ENGINE_HEALTH_HEALTHY)
        self.assertEqual(result.checks["model_file_availability"], ENGINE_HEALTH_HEALTHY)

    def test_engine_health_degraded_when_events_down(self):
        for _ in range(config.HEALTH_DOWN_THRESHOLD):
            self.registry.report("corporate_events_fetcher", ok=False, detail="Temporary rate limit")
        result = self.registry.get_engine_health()
        self.assertEqual(result.checks["events"], ENGINE_HEALTH_DEGRADED)
        self.assertEqual(result.status, ENGINE_HEALTH_DEGRADED)
        self.assertTrue(any("event" in r.lower() for r in result.reasons))

    def test_engine_health_failed_when_critical_component_fails(self):
        for _ in range(config.HEALTH_DOWN_THRESHOLD):
            self.registry.report("data_fetcher", ok=False, detail="Upstream exchange API down")
        result = self.registry.get_engine_health()
        self.assertEqual(result.checks["data"], ENGINE_HEALTH_FAILED)
        self.assertEqual(result.status, ENGINE_HEALTH_FAILED)
        self.assertTrue(any("market data provider offline" in r.lower() for r in result.reasons))


class TestModelValidityStates(unittest.TestCase):
    """
    Test the 7 model validity lifecycle states:
    UNTRAINED, INVALID, STALE, TRAINED, CALIBRATED, VALIDATED, LIVE_ELIGIBLE.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_untrained_state_when_no_artifact_on_disk(self):
        res = evaluate_model_validity(
            symbol="NONEXISTENT_TICKER",
            horizon="INTRADAY",
            max_model_age_days=30,
        )
        self.assertEqual(res.validity_status, VALIDITY_UNTRAINED)
        self.assertFalse(res.is_live_eligible)
        self.assertIn("No trained model or ensemble artifact exists", res.reasons[0])

    def test_invalid_state_on_checksum_mismatch(self):
        symbol = "RELIANCE"
        horizon = "INTRADAY"
        model_file = self.models_dir / f"{symbol}_{horizon}_ensemble.joblib"
        meta_file = self.models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
        model_file.write_bytes(b"corrupted_binary_data_here")

        meta = {
            "model_id": f"{symbol}_{horizon}_v1",
            "model_version": "v1.0",
            "artifact_sha256": "expected_valid_sha256_hash_that_does_not_match",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        mock_em = MagicMock()
        mock_em._current_pointer_path.return_value = Path("nonexistent_pointer")
        mock_em._ensemble_path.return_value = model_file
        mock_em._ensemble_metadata_path.return_value = meta_file

        res = evaluate_model_validity(
            symbol=symbol,
            horizon=horizon,
            ensemble_manager=mock_em,
        )
        self.assertEqual(res.validity_status, VALIDITY_INVALID)
        self.assertFalse(res.is_live_eligible)
        self.assertTrue(any("checksum verification failed" in r.lower() for r in res.reasons))

    def test_invalid_state_on_adversarial_miscalibration(self):
        symbol = "TCS"
        horizon = "INTRADAY"
        model_file = self.models_dir / f"{symbol}_{horizon}_ensemble.joblib"
        meta_file = self.models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
        content = b"valid_ensemble_payload"
        model_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()
        meta = {
            "model_id": f"{symbol}_{horizon}_v1",
            "model_version": "v1.0",
            "artifact_sha256": h,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        mock_em = MagicMock()
        mock_em._current_pointer_path.return_value = Path("nonexistent_pointer")
        mock_em._ensemble_path.return_value = model_file
        mock_em._ensemble_metadata_path.return_value = meta_file

        # Adversarial calibration data: constant confidence (zero resolution)
        adv_df = pd.DataFrame({
            "confidence": [0.60] * 50,
            "correct": [1, 0] * 25,
        })

        res = evaluate_model_validity(
            symbol=symbol,
            horizon=horizon,
            ensemble_manager=mock_em,
            calibration_df=adv_df,
        )
        self.assertEqual(res.validity_status, VALIDITY_INVALID)
        self.assertFalse(res.is_live_eligible)
        self.assertEqual(res.calibration_status, CALIBRATION_STATUS_ADVERSARIAL)
        self.assertTrue(any("adversarial miscalibration" in r.lower() for r in res.reasons))

    def test_stale_state_when_older_than_cadence(self):
        symbol = "INFY"
        horizon = "INTRADAY"
        model_file = self.models_dir / f"{symbol}_{horizon}_ensemble.joblib"
        meta_file = self.models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
        content = b"infy_payload"
        model_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()

        old_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        meta = {
            "model_id": f"{symbol}_{horizon}_v1",
            "model_version": "v1.0",
            "artifact_sha256": h,
            "trained_at": old_date,
            "expected_calibration_error": 0.05,
            "alpha_pct": 2.5,
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        mock_em = MagicMock()
        mock_em._current_pointer_path.return_value = Path("nonexistent_pointer")
        mock_em._ensemble_path.return_value = model_file
        mock_em._ensemble_metadata_path.return_value = meta_file

        res = evaluate_model_validity(
            symbol=symbol,
            horizon=horizon,
            ensemble_manager=mock_em,
            max_model_age_days=30,
        )
        self.assertEqual(res.validity_status, VALIDITY_STALE)
        self.assertFalse(res.is_live_eligible)
        self.assertGreater(res.age_days, 30.0)
        self.assertTrue(any("stale" in r.lower() for r in res.reasons))

    def test_trained_state_when_uncalibrated_and_unvalidated(self):
        symbol = "HDFCBANK"
        horizon = "INTRADAY"
        model_file = self.models_dir / f"{symbol}_{horizon}_ensemble.joblib"
        meta_file = self.models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
        content = b"hdfc_payload"
        model_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()

        fresh_date = datetime.now(timezone.utc).isoformat()
        meta = {
            "model_id": f"{symbol}_{horizon}_v1",
            "model_version": "v1.0",
            "artifact_sha256": h,
            "trained_at": fresh_date,
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        mock_em = MagicMock()
        mock_em._current_pointer_path.return_value = Path("nonexistent_pointer")
        mock_em._ensemble_path.return_value = model_file
        mock_em._ensemble_metadata_path.return_value = meta_file

        res = evaluate_model_validity(
            symbol=symbol,
            horizon=horizon,
            ensemble_manager=mock_em,
        )
        self.assertEqual(res.validity_status, VALIDITY_TRAINED)
        self.assertFalse(res.is_live_eligible)

    def test_calibrated_state_when_well_calibrated_without_edge(self):
        symbol = "SBIN"
        horizon = "INTRADAY"
        model_file = self.models_dir / f"{symbol}_{horizon}_ensemble.joblib"
        meta_file = self.models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
        content = b"sbin_payload"
        model_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()

        fresh_date = datetime.now(timezone.utc).isoformat()
        meta = {
            "model_id": f"{symbol}_{horizon}_v1",
            "model_version": "v1.0",
            "artifact_sha256": h,
            "trained_at": fresh_date,
            "expected_calibration_error": 0.04,  # <= 0.08 threshold
            "alpha_pct": -1.2,  # No edge!
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        mock_em = MagicMock()
        mock_em._current_pointer_path.return_value = Path("nonexistent_pointer")
        mock_em._ensemble_path.return_value = model_file
        mock_em._ensemble_metadata_path.return_value = meta_file

        res = evaluate_model_validity(
            symbol=symbol,
            horizon=horizon,
            ensemble_manager=mock_em,
        )
        self.assertEqual(res.validity_status, VALIDITY_CALIBRATED)
        self.assertFalse(res.is_live_eligible)
        self.assertEqual(res.calibration_status, CALIBRATION_STATUS_WELL_CALIBRATED)

    def test_validated_state_when_edge_confirmed_without_calibration(self):
        symbol = "ITC"
        horizon = "INTRADAY"
        model_file = self.models_dir / f"{symbol}_{horizon}_ensemble.joblib"
        meta_file = self.models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
        content = b"itc_payload"
        model_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()

        fresh_date = datetime.now(timezone.utc).isoformat()
        meta = {
            "model_id": f"{symbol}_{horizon}_v1",
            "model_version": "v1.0",
            "artifact_sha256": h,
            "trained_at": fresh_date,
            "expected_calibration_error": 0.18,  # Miscalibrated!
            "alpha_pct": 3.4,  # Strong edge!
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        mock_em = MagicMock()
        mock_em._current_pointer_path.return_value = Path("nonexistent_pointer")
        mock_em._ensemble_path.return_value = model_file
        mock_em._ensemble_metadata_path.return_value = meta_file

        res = evaluate_model_validity(
            symbol=symbol,
            horizon=horizon,
            ensemble_manager=mock_em,
        )
        self.assertEqual(res.validity_status, VALIDITY_VALIDATED)
        self.assertFalse(res.is_live_eligible)
        self.assertEqual(res.edge_status, EDGE_STATUS_CONFIRMED)

    def test_live_eligible_when_both_calibrated_and_edge_confirmed(self):
        symbol = "WIPRO"
        horizon = "INTRADAY"
        model_file = self.models_dir / f"{symbol}_{horizon}_ensemble.joblib"
        meta_file = self.models_dir / f"{symbol}_{horizon}_ensemble_metadata.json"
        content = b"wipro_payload"
        model_file.write_bytes(content)
        import hashlib
        h = hashlib.sha256(content).hexdigest()

        fresh_date = datetime.now(timezone.utc).isoformat()
        meta = {
            "model_id": f"{symbol}_{horizon}_v1",
            "model_version": "v1.0",
            "artifact_sha256": h,
            "trained_at": fresh_date,
            "expected_calibration_error": 0.04,  # Well-calibrated
            "alpha_pct": 2.8,  # Edge confirmed
        }
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        mock_em = MagicMock()
        mock_em._current_pointer_path.return_value = Path("nonexistent_pointer")
        mock_em._ensemble_path.return_value = model_file
        mock_em._ensemble_metadata_path.return_value = meta_file

        res = evaluate_model_validity(
            symbol=symbol,
            horizon=horizon,
            ensemble_manager=mock_em,
        )
        self.assertEqual(res.validity_status, VALIDITY_LIVE_ELIGIBLE)
        self.assertTrue(res.is_live_eligible)
        self.assertEqual(res.calibration_status, CALIBRATION_STATUS_WELL_CALIBRATED)
        self.assertEqual(res.edge_status, EDGE_STATUS_CONFIRMED)


class TestCoreInvariantDecoupledHealthAndValidity(unittest.TestCase):
    """
    CRITICAL INVARIANT (MODEL-004):
    A healthy engine must NOT imply a valid model!
    Only (Engine != FAILED and Model == LIVE_ELIGIBLE) allows live serving.
    """

    def test_healthy_engine_does_not_permit_untrained_model(self):
        engine_health = EngineHealthResult(
            status=ENGINE_HEALTH_HEALTHY,
            checks={"api": "HEALTHY", "data": "HEALTHY", "events": "HEALTHY", "scheduler": "HEALTHY", "storage": "HEALTHY", "model_file_availability": "HEALTHY"},
            components=[],
            summary="Engine 100% operational",
            reasons=[],
        )
        model_validity = ModelValidityResult(
            symbol="RELIANCE",
            horizon="INTRADAY",
            validity_status=VALIDITY_UNTRAINED,
            is_live_eligible=False,
            reasons=["No model exists on disk."],
        )

        can_serve, reasons = can_serve_live_signal(engine_health, model_validity)
        self.assertFalse(can_serve)
        self.assertTrue(any("must be 'LIVE_ELIGIBLE'" in r for r in reasons))

    def test_healthy_engine_does_not_permit_calibrated_only_model(self):
        engine_health = EngineHealthResult(
            status=ENGINE_HEALTH_HEALTHY,
            checks={"api": "HEALTHY", "data": "HEALTHY", "events": "HEALTHY", "scheduler": "HEALTHY", "storage": "HEALTHY", "model_file_availability": "HEALTHY"},
            components=[],
            summary="Engine 100% operational",
            reasons=[],
        )
        model_validity = ModelValidityResult(
            symbol="TCS",
            horizon="INTRADAY",
            validity_status=VALIDITY_CALIBRATED,
            is_live_eligible=False,
            reasons=["Lacks confirmed alpha edge."],
        )

        can_serve, reasons = can_serve_live_signal(engine_health, model_validity)
        self.assertFalse(can_serve)

    def test_healthy_engine_does_not_permit_stale_or_invalid_model(self):
        engine_health = EngineHealthResult(
            status=ENGINE_HEALTH_HEALTHY,
            checks={"api": "HEALTHY", "data": "HEALTHY", "events": "HEALTHY", "scheduler": "HEALTHY", "storage": "HEALTHY", "model_file_availability": "HEALTHY"},
            components=[],
            summary="Engine 100% operational",
            reasons=[],
        )
        stale_model = ModelValidityResult(
            symbol="INFY",
            horizon="INTRADAY",
            validity_status=VALIDITY_STALE,
            is_live_eligible=False,
            reasons=["Model is 45 days old."],
        )
        can_serve, _ = can_serve_live_signal(engine_health, stale_model)
        self.assertFalse(can_serve)

        invalid_model = ModelValidityResult(
            symbol="INFY",
            horizon="INTRADAY",
            validity_status=VALIDITY_INVALID,
            is_live_eligible=False,
            reasons=["Checksum mismatch."],
        )
        can_serve, _ = can_serve_live_signal(engine_health, invalid_model)
        self.assertFalse(can_serve)

    def test_failed_engine_blocks_even_live_eligible_model(self):
        engine_health = EngineHealthResult(
            status=ENGINE_HEALTH_FAILED,
            checks={"api": "HEALTHY", "data": "FAILED", "events": "HEALTHY", "scheduler": "HEALTHY", "storage": "HEALTHY", "model_file_availability": "HEALTHY"},
            components=[],
            summary="Data feed offline",
            reasons=["Market data provider offline."],
        )
        valid_model = ModelValidityResult(
            symbol="RELIANCE",
            horizon="INTRADAY",
            validity_status=VALIDITY_LIVE_ELIGIBLE,
            is_live_eligible=True,
            reasons=["Model well-calibrated with confirmed edge."],
        )

        can_serve, reasons = can_serve_live_signal(engine_health, valid_model)
        self.assertFalse(can_serve)
        self.assertTrue(any("Engine health is FAILED" in r for r in reasons))

    def test_live_eligible_model_serves_when_engine_healthy(self):
        engine_health = EngineHealthResult(
            status=ENGINE_HEALTH_HEALTHY,
            checks={"api": "HEALTHY", "data": "HEALTHY", "events": "HEALTHY", "scheduler": "HEALTHY", "storage": "HEALTHY", "model_file_availability": "HEALTHY"},
            components=[],
            summary="Operational",
            reasons=[],
        )
        valid_model = ModelValidityResult(
            symbol="RELIANCE",
            horizon="INTRADAY",
            validity_status=VALIDITY_LIVE_ELIGIBLE,
            is_live_eligible=True,
            reasons=["All checks passed."],
        )

        can_serve, reasons = can_serve_live_signal(engine_health, valid_model)
        self.assertTrue(can_serve)
        self.assertTrue(any("verified for live signal serving" in r for r in reasons))


class TestApiHealthAndValidityEndpoints(unittest.TestCase):
    """
    Test FastAPI endpoints:
      - GET /api/v1/health (includes engine_health and checks)
      - GET /api/v1/model-validity/{symbol}
    """

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint_schema_and_fields(self):
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("overall", data)
        self.assertIn("diagnostics", data)
        self.assertIn("engine_health", data)
        self.assertIn("checks", data)
        self.assertIn("api", data["checks"])
        self.assertIn("data", data["checks"])
        self.assertIn("events", data["checks"])
        self.assertIn("scheduler", data["checks"])
        self.assertIn("storage", data["checks"])
        self.assertIn("model_file_availability", data["checks"])

    def test_model_validity_endpoint(self):
        resp = self.client.get("/api/v1/model-validity/RELIANCE")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["symbol"], "RELIANCE")
        self.assertEqual(data["horizon"], "INTRADAY")
        self.assertIn(data["validity_status"], ALL_VALIDITY_STATES)
        self.assertIsInstance(data["is_live_eligible"], bool)
        self.assertIn("calibration_status", data)
        self.assertIn("edge_status", data)
        self.assertIsInstance(data["reasons"], list)

    def test_model_validity_endpoint_invalid_symbol(self):
        resp = self.client.get("/api/v1/model-validity/INVALID_TICKER_XYZ")
        self.assertEqual(resp.status_code, 404)

    @patch("api.scheduler")
    def test_signal_endpoint_forces_hold_when_model_not_live_eligible(self, mock_scheduler):
        from predictor import MultiHorizonSignal, PredictionSignal
        dummy_sig = PredictionSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(timezone.utc),
            horizon="INTRADAY",
            action="BUY",
            model_predicted_class="UP",
            model_version="v1.0",
            feature_version="v1.0",
            raw_confidence=0.85,
            risk_adjusted_confidence=0.82,
            calibrated_confidence=0.80,
            agreement_fraction=1.0,
            downside_summary="Downside risk defined",
            upside_summary="Upside target defined",
            reasoning=["Test model predicted UP"],
            contributing_events=[],
            target_price=2600.0,
            stop_loss=2400.0,
        )
        multi_sig = MultiHorizonSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(timezone.utc),
            signals={"INTRADAY": dummy_sig},
            primary_action="BUY",
            primary_horizon="INTRADAY",
            reasoning=["Model lean UP"],
        )

        mock_ctx = MagicMock()
        mock_ctx.market_data = pd.DataFrame({"Close": [2500.0]})
        mock_ctx.index_data = pd.DataFrame({"Close": [22000.0]})
        mock_scheduler.get_event_context.return_value = MagicMock(macro_events=[], corporate_events=[], news_articles=[])
        mock_scheduler.build_prediction_context.return_value = mock_ctx
        mock_scheduler.run_one_cycle_for_symbol.return_value = MagicMock(signal=multi_sig)

        with patch("api.evaluate_model_validity") as mock_val:
            mock_val.return_value = ModelValidityResult(
                symbol="RELIANCE",
                horizon="INTRADAY",
                validity_status=VALIDITY_UNTRAINED,
                is_live_eligible=False,
                reasons=["No model exists on disk."],
            )
            resp = self.client.get("/api/v1/signal/RELIANCE")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            intraday = data["signals"]["INTRADAY"]
            # Invariant: Action MUST be forced to HOLD
            self.assertEqual(intraday["action"], "HOLD")
            self.assertIsNone(intraday["confidence"])
            self.assertIn("Holding: Model for INTRADAY is UNTRAINED", intraday["verdict_text"])

    @patch("api.scheduler")
    def test_signal_endpoint_preserves_action_when_live_eligible(self, mock_scheduler):
        from predictor import MultiHorizonSignal, PredictionSignal
        dummy_sig = PredictionSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(timezone.utc),
            horizon="INTRADAY",
            action="BUY",
            model_predicted_class="UP",
            model_version="v1.0",
            feature_version="v1.0",
            raw_confidence=0.85,
            risk_adjusted_confidence=0.82,
            calibrated_confidence=0.80,
            agreement_fraction=1.0,
            downside_summary="Downside risk defined",
            upside_summary="Upside target defined",
            reasoning=["Strong momentum"],
            contributing_events=[],
            target_price=2600.0,
            stop_loss=2400.0,
        )
        multi_sig = MultiHorizonSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(timezone.utc),
            signals={"INTRADAY": dummy_sig},
            primary_action="BUY",
            primary_horizon="INTRADAY",
            reasoning=["Strong momentum"],
        )

        mock_ctx = MagicMock()
        mock_ctx.market_data = pd.DataFrame({"Close": [2500.0]})
        mock_ctx.index_data = pd.DataFrame({"Close": [22000.0]})
        mock_scheduler.get_event_context.return_value = MagicMock(macro_events=[], corporate_events=[], news_articles=[])
        mock_scheduler.build_prediction_context.return_value = mock_ctx
        mock_scheduler.run_one_cycle_for_symbol.return_value = MagicMock(signal=multi_sig)

        with patch("api.evaluate_model_validity") as mock_val:
            mock_val.return_value = ModelValidityResult(
                symbol="RELIANCE",
                horizon="INTRADAY",
                validity_status=VALIDITY_LIVE_ELIGIBLE,
                is_live_eligible=True,
                reasons=["Model verified."],
            )
            resp = self.client.get("/api/v1/signal/RELIANCE")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            intraday = data["signals"]["INTRADAY"]
            self.assertEqual(intraday["action"], "BUY")
            self.assertEqual(intraday["confidence"], 0.80)
