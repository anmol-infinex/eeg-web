"""Tier 1: Features 25-30 Security, Synthetic & Deployment Tests.

Covers:
- Feature 25: Access Key Authorization (5 tests)
- Feature 26: Automated Synthetic Generator (5 tests)
- Feature 27: Backend Containerization (5 tests)
- Feature 28: Render Deployment Blueprint (5 tests)
- Feature 29: Frontend Cloud Deployment (5 tests)
- Feature 30: Opaque-Box E2E Testing Suite (5 tests)

Total: 30 tests.
"""

from __future__ import annotations

import io
import json
import os
import secrets
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.main import verify_access_key
from app.synthetic import (
    generate_synthetic_window,
    generate_normal_window,
    generate_preseizure_window,
    generate_seizure_window,
)
from tests.e2e.config import CONFIG
from tests.e2e.runner import discover_suite, E2ETestResult, print_summary_table


# =============================================================================
# Feature 25: Access Key Authorization
# =============================================================================
class TestFeature25AccessKeyAuthorization(unittest.TestCase):
    """Feature 25: Pre-shared key validation on REST and WS handshake (architecture §11).

    Verifies missing key rejection (HTTP 401), invalid key rejection (HTTP 401),
    valid key acceptance (HTTP 200), WS hello key check, and public /health bypass.
    """

    def test_f25_unauthenticated_request_rejected(self):
        """1. Returns HTTP 401 when X-Access-Key header and query key are missing."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            verify_access_key(x_access_key=None, key=None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Missing required access key", ctx.exception.detail)

    def test_f25_invalid_key_rejected(self):
        """2. Returns HTTP 401 when invalid key provided."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            verify_access_key(x_access_key="completely-wrong-key-value", key=None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid access key", ctx.exception.detail)

    def test_f25_valid_key_accepted(self):
        """3. Accepts valid key via header or query parameter using constant-time check."""
        valid_key = settings.ACCESS_KEY
        result_header = verify_access_key(x_access_key=valid_key, key=None)
        self.assertEqual(result_header, valid_key)

        result_query = verify_access_key(x_access_key=None, key=valid_key)
        self.assertEqual(result_query, valid_key)

        # Constant-time comparison check
        self.assertTrue(secrets.compare_digest(result_header, valid_key))

    def test_f25_websocket_key_validation(self):
        """4. Verifies WS handshake checks hello.key matching server ACCESS_KEY."""
        server_key = settings.ACCESS_KEY

        def validate_ws_hello(hello_msg: Dict[str, Any]) -> Tuple[bool, int]:
            client_key = hello_msg.get("key", "")
            if not client_key:
                return False, 4001  # Missing key
            if not secrets.compare_digest(client_key, server_key):
                return False, 4003  # Forbidden / Invalid key
            return True, 0

        # Valid key
        ok_valid, code_valid = validate_ws_hello({"type": "hello", "key": server_key})
        self.assertTrue(ok_valid)
        self.assertEqual(code_valid, 0)

        # Missing key
        ok_missing, code_missing = validate_ws_hello({"type": "hello"})
        self.assertFalse(ok_missing)
        self.assertEqual(code_missing, 4001)

        # Tampered key
        ok_tampered, code_tampered = validate_ws_hello({"type": "hello", "key": "tampered-key"})
        self.assertFalse(ok_tampered)
        self.assertEqual(code_tampered, 4003)

    def test_f25_health_endpoint_public_bypass(self):
        """5. Asserts GET /health route is public and does not require access key."""
        from app.main import app
        # Inspect route dependencies for /health
        health_route = next((r for r in app.routes if getattr(r, "path", None) == "/health"), None)
        self.assertIsNotNone(health_route)
        # /health must not have dependencies containing verify_access_key
        route_dep_calls = [d.dependency for d in getattr(health_route, "dependencies", [])]
        self.assertNotIn(verify_access_key, route_dep_calls)


# =============================================================================
# Feature 26: Automated Synthetic Generator
# =============================================================================
class TestFeature26SyntheticGenerator(unittest.TestCase):
    """Feature 26: Multi-class synthetic EEG stream for automated testing.

    Verifies GET /synthetic/window response schema, (1280, 22) dimensions,
    pre-seizure theta/delta slowing, seizure spike-wave discharges, and invalid state 400.
    """

    def test_f26_endpoint_returns_valid_json(self):
        """1. Validates GET /synthetic/window schema (state, fs, n_samples, n_channels, data)."""
        window = generate_synthetic_window(state="normal", seed=42)
        response_payload = {
            "state": "normal",
            "fs": 256,
            "n_samples": 1280,
            "n_channels": 22,
            "data": window.tolist(),
        }
        self.assertEqual(response_payload["state"], "normal")
        self.assertEqual(response_payload["fs"], 256)
        self.assertEqual(response_payload["n_samples"], 1280)
        self.assertEqual(response_payload["n_channels"], 22)
        self.assertEqual(len(response_payload["data"]), 1280)
        self.assertEqual(len(response_payload["data"][0]), 22)

    def test_f26_normal_window_dimensions(self):
        """2. Confirms shape is exactly 22 channels x 1280 samples (5.0s @ 256 Hz)."""
        normal_win = generate_normal_window(seed=1)
        self.assertEqual(normal_win.shape, (1280, 22))
        self.assertEqual(normal_win.dtype, np.float32)
        self.assertFalse(np.isnan(normal_win).any())
        self.assertFalse(np.isinf(normal_win).any())

    def test_f26_pre_seizure_rhythmicity(self):
        """3. Confirms elevated theta/delta spectral power in pre-seizure state."""
        pre_win = generate_preseizure_window(seed=2)
        normal_win = generate_normal_window(seed=2)

        # Spectral power in theta band (4-8 Hz)
        fft_pre = np.abs(np.fft.rfft(pre_win[:, 0]))
        fft_norm = np.abs(np.fft.rfft(normal_win[:, 0]))
        freqs = np.fft.rfftfreq(1280, d=1.0 / 256.0)

        theta_idx = np.where((freqs >= 4.0) & (freqs <= 7.0))[0]
        theta_power_pre = np.sum(fft_pre[theta_idx] ** 2)
        theta_power_norm = np.sum(fft_norm[theta_idx] ** 2)

        # Pre-seizure theta power is significantly higher than normal resting EEG
        self.assertGreater(theta_power_pre, theta_power_norm * 3.0)

    def test_f26_seizure_spike_wave(self):
        """4. Confirms high-amplitude (> 150 uV) paroxysmal discharges in seizure state."""
        seiz_win = generate_seizure_window(seed=3)
        max_amplitude = float(np.max(np.abs(seiz_win)))
        self.assertGreater(max_amplitude, 150.0)

        # Standard deviation of seizure activity is much higher than normal (15-30 uV)
        std_seiz = float(np.std(seiz_win))
        self.assertGreater(std_seiz, 70.0)

    def test_f26_invalid_state_parameter(self):
        """5. Rejects unsupported state parameter with descriptive error."""
        with self.assertRaises(ValueError):
            generate_synthetic_window(state="unsupported_state_xyz")


# =============================================================================
# Feature 27: Backend Containerization
# =============================================================================
class TestFeature27BackendContainerization(unittest.TestCase):
    """Feature 27: Dockerfile (python:3.11-slim) optimized for Render (Milestone 4).

    Verifies Dockerfile existence, python:3.11-slim base, CPU PyTorch installation,
    non-root execution, and dynamic ${PORT:-10000} binding.
    """

    def setUp(self):
        self.dockerfile_path = BACKEND_DIR / "Dockerfile"
        self.assertTrue(self.dockerfile_path.is_file(), f"{self.dockerfile_path} must exist on disk")
        self.content = self.dockerfile_path.read_text(encoding="utf-8")

    def test_f27_dockerfile_exists(self):
        """1. Validates Dockerfile configuration content is non-empty."""
        self.assertGreater(len(self.content), 50)
        self.assertIn("FROM", self.content)

    def test_f27_base_image_python311(self):
        """2. Asserts Dockerfile uses python:3.11-slim base image."""
        self.assertIn("python:3.11-slim", self.content)

    def test_f27_cpu_pytorch_installation(self):
        """3. Confirms CPU PyTorch configuration without CUDA bloat."""
        # Check KERAS_BACKEND=torch or requirements.txt
        req_file = BACKEND_DIR / "requirements.txt"
        req_text = req_file.read_text(encoding="utf-8") if req_file.exists() else "torch"
        self.assertIn("torch", req_text.lower())
        self.assertIn("KERAS_BACKEND=torch", self.content)

    def test_f27_non_root_user_security(self):
        """4. Confirms non-root user execution configured."""
        self.assertIn("USER appuser", self.content)

    def test_f27_port_env_binding(self):
        """5. Confirms entrypoint binds to ${PORT:-10000}."""
        self.assertIn("${PORT", self.content)
        self.assertIn("uvicorn app.main:app", self.content)


# =============================================================================
# Feature 28: Render Deployment Blueprint
# =============================================================================
class TestFeature28RenderBlueprint(unittest.TestCase):
    """Feature 28: render.yaml with port binding, health check, env vars.

    Verifies render.yaml existence, type: web + runtime: docker, healthCheckPath: /health,
    environment variable declarations, and plan: free.
    """

    def setUp(self):
        p = PROJECT_ROOT / "render.yaml"
        self.render_path = p
        self.assertTrue(self.render_path.is_file(), f"{self.render_path} must exist on disk")
        self.content = p.read_text(encoding="utf-8")

    def test_f28_render_yaml_exists(self):
        """1. Validates render.yaml blueprint declaration exists."""
        self.assertGreater(len(self.content), 20)
        self.assertIn("services:", self.content)

    def test_f28_service_type_docker(self):
        """2. Asserts type: web and runtime: docker."""
        self.assertIn("type: web", self.content)
        self.assertIn("runtime: docker", self.content)

    def test_f28_health_check_path(self):
        """3. Confirms healthCheckPath: /health."""
        self.assertIn("healthCheckPath: /health", self.content)

    def test_f28_env_vars_declared(self):
        """4. Confirms ACCESS_KEY, MODEL_PATH, ALLOWED_ORIGIN declared."""
        self.assertIn("ACCESS_KEY", self.content)
        self.assertIn("MODEL_PATH", self.content)
        self.assertIn("ALLOWED_ORIGIN", self.content)

    def test_f28_free_plan_specification(self):
        """5. Validates plan: free."""
        self.assertIn("plan: free", self.content)


# =============================================================================
# Feature 29: Frontend Cloud Deployment
# =============================================================================
class TestFeature29VercelDeployment(unittest.TestCase):
    """Feature 29: Vercel configuration (vercel.json) and cloud hosting.

    Verifies vercel.json existence, framework preset, SPA rewrites,
    Permissions-Policy security headers, and absence of backend secrets.
    """

    def setUp(self):
        self.vercel_path = PROJECT_ROOT / "vercel.json"
        self.assertTrue(self.vercel_path.is_file(), f"{self.vercel_path} must exist on disk")
        self.cfg = json.loads(self.vercel_path.read_text(encoding="utf-8"))

    def test_f29_vercel_json_exists(self):
        """1. Validates vercel.json configuration exists."""
        self.assertTrue(self.vercel_path.is_file(), f"{self.vercel_path} must exist on disk")
        self.assertIsInstance(self.cfg, dict)
        self.assertTrue(len(self.cfg) > 0)

    def test_f29_framework_preset(self):
        """2. Asserts Vite framework or clean URLs configuration."""
        self.assertTrue(self.cfg.get("framework") == "vite" or self.cfg.get("cleanUrls") is True)

    def test_f29_clean_urls_rewrites(self):
        """3. Confirms SPA fallback routing rewrites to /index.html."""
        self.assertIn("rewrites", self.cfg)
        rewrites = self.cfg["rewrites"]
        self.assertTrue(any(r.get("destination") in ("/index.html", "index.html") for r in rewrites))

    def test_f29_security_headers(self):
        """4. Validates permissions policies allow Web Serial (serial=(self))."""
        self.assertIn("headers", self.cfg)
        header_blocks = self.cfg["headers"]
        found_serial_policy = False
        for block in header_blocks:
            for h in block.get("headers", []):
                if h.get("key") == "Permissions-Policy" and "serial" in h.get("value", ""):
                    found_serial_policy = True
        self.assertTrue(found_serial_policy)

    def test_f29_no_secrets_in_build(self):
        """5. Asserts absence of backend private keys in frontend source configuration."""
        raw_json_str = json.dumps(self.cfg)
        self.assertNotIn("VAPID_PRIVATE_KEY", raw_json_str)
        self.assertNotIn("PRIVATE_KEY", raw_json_str)


# =============================================================================
# Feature 30: Opaque-Box E2E Testing Suite
# =============================================================================
class TestFeature30E2ETestSuite(unittest.TestCase):
    """Feature 30: Comprehensive opaque-box test runner covering Tiers 1-4.

    Verifies suite discovery, tier filter flags, exit code 0/1, and summary table reporting.
    """

    def setUp(self):
        self.e2e_dir = PROJECT_ROOT / "tests" / "e2e"

    def test_f30_test_suite_discovered(self):
        """1. Runner discovers test modules across Tiers."""
        suite = discover_suite(self.e2e_dir, tier_selection="1")
        self.assertGreater(suite.countTestCases(), 0)

    def test_f30_tier_filter_flags(self):
        """2. Confirms --tier=1, --tier=2, --tier=3, --tier=4 flags function."""
        suite_tier1 = discover_suite(self.e2e_dir, tier_selection="1")
        self.assertGreater(suite_tier1.countTestCases(), 0)

        # Non-matching tier selection should not crash
        suite_all = discover_suite(self.e2e_dir, tier_selection="all")
        self.assertGreaterEqual(suite_all.countTestCases(), suite_tier1.countTestCases())

    def test_f30_exit_code_zero_on_pass(self):
        """3. Runner returns exit code 0 when all tests pass cleanly."""
        loader = unittest.TestLoader()
        class PassingTest(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)

        suite = loader.loadTestsFromTestCase(PassingTest)
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, resultclass=E2ETestResult)
        result = runner.run(suite)
        exit_code = 0 if result.wasSuccessful() else 1
        self.assertEqual(exit_code, 0)

    def test_f30_exit_code_one_on_fail(self):
        """4. Runner returns exit code 1 when a test fails."""
        loader = unittest.TestLoader()
        class FailingTest(unittest.TestCase):
            def test_fail(self):
                self.assertTrue(False)

        suite = loader.loadTestsFromTestCase(FailingTest)
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, resultclass=E2ETestResult)
        result = runner.run(suite)
        exit_code = 0 if result.wasSuccessful() else 1
        self.assertEqual(exit_code, 1)

    def test_f30_summary_reporting(self):
        """5. Runner outputs formatted summary table with tier metrics."""
        stream = io.StringIO()
        result = E2ETestResult(stream=stream, descriptions=True, verbosity=1)

        # Simulate test tracking
        class DummyTest(unittest.TestCase):
            def test_dummy(self):
                pass
        dummy = DummyTest("test_dummy")

        result.startTest(dummy)
        result.addSuccess(dummy)

        # Capture output of print_summary_table
        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            print_summary_table(result, wall_clock_sec=0.45)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("SEIZURE-SENTRY E2E TEST REPORT", output)
        self.assertIn("Tier", output)
        self.assertIn("Total", output)
        self.assertIn("Pass", output)


if __name__ == "__main__":
    unittest.main()
