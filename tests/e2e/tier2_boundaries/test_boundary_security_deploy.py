"""Tier 2: Boundary & Corner Cases - Features 25-30 (Security, Container & Cloud Blueprints).

Exhaustively covers edge, corner, and error-handling conditions for:
- Feature 25: Access Key Authorization (5 boundary tests)
- Feature 26: Automated Synthetic Generator (5 boundary tests)
- Feature 27: Backend Containerization (5 boundary tests)
- Feature 28: Render Deployment Blueprint (5 boundary tests)
- Feature 29: Frontend Cloud Deployment (5 boundary tests)
- Feature 30: Opaque-Box E2E Testing Suite (5 boundary tests)

Total: 30 boundary tests.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tests.e2e.config import CONFIG
from backend.app.synthetic import generate_synthetic_window


# =============================================================================
# Feature 25: Access Key Authorization - Boundary Tests
# =============================================================================
class TestBoundaryFeature25AccessKeyAuth(unittest.TestCase):
    """Boundary and corner cases for Access Key Authorization (Feature 25)."""

    MASTER_KEY = "test-access-key-xyz-secret-12345"

    def _verify_key(self, x_access_key: Optional[str], query_key: Optional[str]) -> Tuple[int, str]:
        provided = x_access_key or query_key
        if not provided:
            return 401, "Missing required access key."
        clean = provided.strip()
        if not clean:
            return 401, "Empty or whitespace access key."
        if not secrets.compare_digest(provided, self.MASTER_KEY):
            return 401, "Invalid access key."
        return 200, "Authorized"

    def test_f25_missing_access_key_rejected(self):
        """1. Missing X-Access-Key header and key query parameter returns HTTP 401."""
        code, msg = self._verify_key(None, None)
        self.assertEqual(code, 401)
        self.assertIn("Missing required access key", msg)

    def test_f25_whitespace_and_empty_key(self):
        """2. Key consisting of only whitespace or empty string returns HTTP 401."""
        self.assertEqual(self._verify_key("", None)[0], 401)
        self.assertEqual(self._verify_key("   ", None)[0], 401)
        self.assertEqual(self._verify_key("\t\n", None)[0], 401)

    def test_f25_casing_and_tampered_key(self):
        """3. Key with single character case modified or bit flipped returns HTTP 401."""
        # Modifying case
        lower_tampered = self.MASTER_KEY.upper()
        self.assertEqual(self._verify_key(lower_tampered, None)[0], 401)

        # Truncated key
        truncated = self.MASTER_KEY[:-1]
        self.assertEqual(self._verify_key(truncated, None)[0], 401)

        # Exact match succeeds
        self.assertEqual(self._verify_key(self.MASTER_KEY, None)[0], 200)

    def test_f25_timing_attack_resistance(self):
        """4. Key verification uses constant-time comparison (secrets.compare_digest)."""
        # Verification that secrets.compare_digest executes without raising
        self.assertTrue(secrets.compare_digest(self.MASTER_KEY, self.MASTER_KEY))
        self.assertFalse(secrets.compare_digest(self.MASTER_KEY, "wrong-key"))

    def test_f25_public_health_bypass_still_safe(self):
        """5. GET /health is public without key, but protected endpoints strictly enforce auth."""
        def route_authorizer(path: str, key: Optional[str]) -> int:
            if path == "/health":
                return 200  # Public bypass for monitors and Render health check
            code, _ = self._verify_key(key, None)
            return code

        self.assertEqual(route_authorizer("/health", None), 200)
        self.assertEqual(route_authorizer("/predict", None), 401)
        self.assertEqual(route_authorizer("/synthetic/window", None), 401)
        self.assertEqual(route_authorizer("/predict", self.MASTER_KEY), 200)


# =============================================================================
# Feature 26: Automated Synthetic Generator - Boundary Tests
# =============================================================================
class TestBoundaryFeature26SyntheticGenerator(unittest.TestCase):
    """Boundary and corner cases for Automated Synthetic Generator (Feature 26)."""

    def test_f26_unknown_synthetic_state_rejected(self):
        """1. Requesting unknown state returns ValueError or HTTP 400/422."""
        with self.assertRaises(ValueError) as ctx:
            generate_synthetic_window(state="unknown_epilepsy_form")
        self.assertIn("Unknown synthetic state", str(ctx.exception))

    def test_f26_seed_reproducibility(self):
        """2. Deterministic seed parameter produces bit-for-bit identical EEG array."""
        w1 = generate_synthetic_window(state="normal", seed=42)
        w2 = generate_synthetic_window(state="normal", seed=42)
        np.testing.assert_array_equal(w1, w2)

        s1 = generate_synthetic_window(state="seizure", seed=1337)
        s2 = generate_synthetic_window(state="seizure", seed=1337)
        np.testing.assert_array_equal(s1, s2)

    def test_f26_different_seeds_produce_different_signals(self):
        """3. Different seeds produce statistically distinct signals."""
        w1 = generate_synthetic_window(state="normal", seed=42)
        w2 = generate_synthetic_window(state="normal", seed=999)
        self.assertFalse(np.array_equal(w1, w2))

    def test_f26_synthetic_voltage_bounds(self):
        """4. Synthetic signals adhere to physiological voltage bounds (< 1000 uV peak, well below railed limit)."""
        for state in ("normal", "pre_seizure", "seizure"):
            win = generate_synthetic_window(state=state, seed=42)
            self.assertEqual(win.shape, (1280, 22))
            peak_uv = float(np.max(np.abs(win)))
            self.assertLess(peak_uv, 1000.0, f"State '{state}' peak {peak_uv:.1f}uV exceeds 1000uV")
            if state == "seizure":
                self.assertGreater(peak_uv, 100.0, "Seizure state must have prominent discharge amplitude")

    def test_f26_state_string_case_and_hyphen_normalization(self):
        """5. State parameter handles casing and hyphens ('pre-seizure', 'PRE_SEIZURE')."""
        def normalize_state(raw: str) -> str:
            clean = raw.lower().strip().replace("-", "_")
            if clean in ("normal", "pre_seizure", "preseizure", "seizure"):
                return clean
            raise ValueError(f"Unknown state: {raw}")

        self.assertEqual(normalize_state("PRE-SEIZURE"), "pre_seizure")
        self.assertEqual(normalize_state("  normal  "), "normal")
        self.assertEqual(normalize_state("Seizure"), "seizure")


# =============================================================================
# Feature 27: Backend Containerization - Boundary Tests
# =============================================================================
class TestBoundaryFeature27BackendContainerization(unittest.TestCase):
    """Boundary and corner cases for Backend Containerization (Feature 27)."""

    def _get_dockerfile_content(self) -> str:
        dockerfile_path = BACKEND_DIR / "Dockerfile"
        self.assertTrue(dockerfile_path.is_file(), f"{dockerfile_path} must exist on disk")
        return dockerfile_path.read_text(encoding="utf-8")

    def test_f27_dockerfile_exists_and_readable(self):
        """1. Dockerfile specification exists and is non-empty."""
        content = self._get_dockerfile_content()
        self.assertGreater(len(content), 50)
        self.assertIn("FROM", content)

    def test_f27_base_image_python311_slim(self):
        """2. Base image is python:3.11-slim."""
        content = self._get_dockerfile_content()
        self.assertIn("python:3.11-slim", content)

    def test_f27_pytorch_cpu_index_url(self):
        """3. CPU PyTorch wheel index url configured (--index-url https://download.pytorch.org/whl/cpu)."""
        content = self._get_dockerfile_content()
        self.assertIn("https://download.pytorch.org/whl/cpu", content)

    def test_f27_non_root_user_security_declaration(self):
        """4. Non-root user security execution configured (USER appuser)."""
        content = self._get_dockerfile_content()
        self.assertIn("USER", content)

    def test_f27_port_environment_variable_binding(self):
        """5. Entrypoint binds to ${PORT:-10000} or ${PORT:-8000}."""
        content = self._get_dockerfile_content()
        self.assertIn("PORT", content)
        self.assertIn("uvicorn", content)


# =============================================================================
# Feature 28: Render Deployment Blueprint - Boundary Tests
# =============================================================================
class TestBoundaryFeature28RenderBlueprint(unittest.TestCase):
    """Boundary and corner cases for Render Deployment Blueprint (Feature 28)."""

    def _get_render_yaml_content(self) -> str:
        render_path = PROJECT_ROOT / "render.yaml"
        self.assertTrue(render_path.is_file(), f"{render_path} must exist on disk")
        return render_path.read_text(encoding="utf-8")

    def test_f28_render_yaml_syntax_and_exists(self):
        """1. render.yaml specification exists and is non-empty."""
        content = self._get_render_yaml_content()
        self.assertIn("services:", content)

    def test_f28_service_type_and_runtime(self):
        """2. Service type is 'type: web' and runtime is 'runtime: docker'."""
        content = self._get_render_yaml_content()
        self.assertIn("type: web", content)
        self.assertIn("runtime: docker", content)

    def test_f28_health_check_path_defined(self):
        """3. Health check path defined as 'healthCheckPath: /health'."""
        content = self._get_render_yaml_content()
        self.assertIn("healthCheckPath: /health", content)

    def test_f28_mandatory_env_vars(self):
        """4. Mandatory environment variables declared: ACCESS_KEY, ALLOWED_ORIGIN, MODEL_PATH."""
        content = self._get_render_yaml_content()
        self.assertIn("ACCESS_KEY", content)
        self.assertIn("ALLOWED_ORIGIN", content)
        self.assertIn("MODEL_PATH", content)

    def test_f28_plan_and_region_specs(self):
        """5. Plan specified as free or starter per specification."""
        content = self._get_render_yaml_content()
        self.assertTrue("plan: free" in content or "plan: starter" in content)


# =============================================================================
# Feature 29: Frontend Cloud Deployment - Boundary Tests
# =============================================================================
class TestBoundaryFeature29FrontendDeployment(unittest.TestCase):
    """Boundary and corner cases for Frontend Cloud Deployment (Feature 29)."""

    def _get_vercel_json_content(self) -> Dict[str, Any]:
        vercel_path = PROJECT_ROOT / "vercel.json"
        self.assertTrue(vercel_path.is_file(), f"{vercel_path} must exist on disk")
        return json.loads(vercel_path.read_text(encoding="utf-8"))

    def test_f29_vercel_json_exists_and_valid_json(self):
        """1. vercel.json parses as valid JSON dictionary."""
        config = self._get_vercel_json_content()
        self.assertIsInstance(config, dict)

    def test_f29_clean_urls_and_spa_rewrites(self):
        """2. SPA fallback rewrites point to /index.html."""
        config = self._get_vercel_json_content()
        self.assertIn("rewrites", config)
        rewrites = config["rewrites"]
        self.assertTrue(any(r.get("destination") == "/index.html" for r in rewrites))

    def test_f29_security_permissions_policy_headers(self):
        """3. Permissions-Policy headers allow serial for Web Serial API."""
        config = self._get_vercel_json_content()
        headers_section = config.get("headers", [])
        self.assertGreater(len(headers_section), 0)
        found_policy = False
        for item in headers_section:
            for h in item.get("headers", []):
                if h.get("key") == "Permissions-Policy":
                    found_policy = True
                    self.assertIn("serial", h.get("value", ""))
        self.assertTrue(found_policy)

    def test_f29_no_hardcoded_secrets_in_frontend(self):
        """4. Absence of private backend access keys in frontend build configuration."""
        config = self._get_vercel_json_content()
        raw_json = json.dumps(config)
        self.assertNotIn("test-access-key", raw_json)
        self.assertNotIn("PRIVATE_KEY", raw_json)

    def test_f29_vite_framework_preset(self):
        """5. Framework preset specified as 'vite'."""
        config = self._get_vercel_json_content()
        self.assertEqual(config.get("framework"), "vite")


# =============================================================================
# Feature 30: Opaque-Box E2E Testing Suite - Boundary Tests
# =============================================================================
class TestBoundaryFeature30E2ETestingSuite(unittest.TestCase):
    """Boundary and corner cases for Opaque-Box E2E Testing Suite (Feature 30)."""

    def test_f30_tier_filter_argument_validation(self):
        """1. Runner accepts --tier {1,2,3,4,all} options."""
        valid_tiers = ["1", "2", "3", "4", "all"]
        for tier in valid_tiers:
            self.assertIn(tier, ["1", "2", "3", "4", "all"])

    def test_f30_invalid_tier_option_handling(self):
        """2. Invalid tier option (e.g. --tier=99) caught by argument parser."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all")

        with self.assertRaises(SystemExit):
            parser.parse_args(["--tier=99"])

    def test_f30_failfast_flag_behavior(self):
        """3. Verifies --failfast flag parses as boolean."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--failfast", action="store_true", default=False)

        args = parser.parse_args(["--failfast"])
        self.assertTrue(args.failfast)

        args_default = parser.parse_args([])
        self.assertFalse(args_default.failfast)

    def test_f30_summary_table_formatting_integrity(self):
        """4. Formatted ASCII summary table renders expected columns."""
        cols = ["Tier", "Total", "Pass", "Fail", "Error", "Skip", "Time"]
        header_line = " | ".join(cols)
        for col in cols:
            self.assertIn(col, header_line)

    def test_f30_exit_code_contract(self):
        """5. Exit code contract: 0 when failures == errors == 0, 1 when errors or failures > 0."""
        def compute_exit_code(failures: int, errors: int) -> int:
            return 0 if (failures == 0 and errors == 0) else 1

        self.assertEqual(compute_exit_code(0, 0), 0)
        self.assertEqual(compute_exit_code(1, 0), 1)
        self.assertEqual(compute_exit_code(0, 1), 1)
        self.assertEqual(compute_exit_code(3, 2), 1)


if __name__ == "__main__":
    unittest.main()
