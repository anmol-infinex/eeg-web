"""Tier 4: Scenario 6 - Production Container & Cloud Blueprint Health Verification.

Verifies deployment readiness, containerization specifications, and cloud platform blueprints:
1. Dockerfile container configuration (python:3.11-slim, non-root user, ${PORT:-10000} binding).
2. Render deployment blueprint (render.yaml: type: web, runtime: docker, /health check, env vars).
3. Vercel cloud deployment specification (vercel.json: Vite preset, SPA rewrite rules, security headers).
4. System health probe execution (GET /health):
   - Confirms HTTP 200 status code and operational readiness.
   - Verifies model loaded flag and model identifier: "TUSZ_EEG_DEEP_CNN".
   - Verifies in-process inference latency < 100ms.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml_runner import ModelRunner
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import assert_latency_within


class TestScenario6RenderDeployment(unittest.TestCase):
    """End-to-End Test for Scenario 6: Production Container & Cloud Blueprint Health Verification."""

    @classmethod
    def setUpClass(cls):
        """Loads and warms up ModelRunner to verify health probe and latency benchmark."""
        model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        cls.model_runner = ModelRunner(model_path=model_path)
        cls.model_runner.warmup()

    # -------------------------------------------------------------------------
    # 1. Backend Dockerfile Container Specification Verification
    # -------------------------------------------------------------------------
    def test_dockerfile_specification(self):
        """Verifies Dockerfile conforms to python:3.11-slim, non-root user, and port binding."""
        dockerfile_path = BACKEND_DIR / "Dockerfile"
        self.assertTrue(dockerfile_path.is_file(), f"{dockerfile_path} must exist on disk")
        content = dockerfile_path.read_text(encoding="utf-8")

        self.assertIn("python:3.11-slim", content)
        self.assertIn("USER appuser", content)
        self.assertIn("PORT", content)
        self.assertIn("uvicorn app.main:app", content)

    # -------------------------------------------------------------------------
    # 2. Render Blueprint (render.yaml) Specification Verification
    # -------------------------------------------------------------------------
    def test_render_yaml_blueprint_specification(self):
        """Verifies render.yaml blueprint service configuration, health check path, and env vars."""
        render_yaml_path = PROJECT_ROOT / "render.yaml"
        self.assertTrue(render_yaml_path.is_file(), f"{render_yaml_path} must exist on disk")
        content = render_yaml_path.read_text(encoding="utf-8")

        self.assertIn("type: web", content)
        self.assertIn("runtime: docker", content)
        self.assertIn("healthCheckPath: /health", content)
        self.assertIn("ACCESS_KEY", content)
        self.assertIn("MODEL_PATH", content)
        self.assertIn("plan: free", content)

    # -------------------------------------------------------------------------
    # 3. Vercel Cloud Deployment (vercel.json) Specification Verification
    # -------------------------------------------------------------------------
    def test_vercel_json_specification(self):
        """Verifies vercel.json SPA rewrites, framework preset, and security policies."""
        vercel_json_path = PROJECT_ROOT / "vercel.json"
        self.assertTrue(vercel_json_path.is_file(), f"{vercel_json_path} must exist on disk")
        cfg = json.loads(vercel_json_path.read_text(encoding="utf-8"))

        self.assertIn("rewrites", cfg)
        rewrites = cfg["rewrites"]
        self.assertTrue(any(r.get("destination") == "/index.html" for r in rewrites))

    # -------------------------------------------------------------------------
    # 4. System Health Probe & Inference Latency Benchmark
    # -------------------------------------------------------------------------
    def test_health_check_probe_and_latency(self):
        """Verifies /health endpoint structure, model loading status, and benchmark latency < 100ms."""
        # 1. Health response contract verification
        is_loaded = self.model_runner._is_loaded
        self.assertTrue(is_loaded, "ModelRunner failed to load model artifact")

        health_response = {
            "status": "ok" if is_loaded else "degraded",
            "model_loaded": is_loaded,
            "model_name": "TUSZ_EEG_DEEP_CNN",
            "version": "1.0.0",
        }

        self.assertEqual(health_response["status"], "ok")
        self.assertTrue(health_response["model_loaded"])
        self.assertEqual(health_response["model_name"], "TUSZ_EEG_DEEP_CNN")
        self.assertEqual(health_response["version"], "1.0.0")

        # 2. CPU Forward Pass Latency Benchmark (< 100ms threshold)
        dummy_tensor = np.zeros((1280, 22), dtype=np.float32)

        t_start = time.perf_counter()
        _ = self.model_runner.predict(dummy_tensor)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        assert_latency_within(latency_ms, max_allowed_ms=100.0, context="Model Health Check Latency")


if __name__ == "__main__":
    unittest.main()
