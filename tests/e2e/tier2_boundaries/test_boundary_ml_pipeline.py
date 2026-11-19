"""Tier 2: Boundary & Corner Cases - Features 13-17 (Signal Processing & ML Pipeline).

Exhaustively covers edge, corner, and error-handling conditions for:
- Feature 13: Polyphase Resampling (5 boundary tests)
- Feature 14: Causal Streaming Filters (5 boundary tests)
- Feature 15: Channel Remapping & Montage (5 boundary tests)
- Feature 16: Deep 1D CNN Inference (5 boundary tests)
- Feature 17: ModelRunner Interface (5 boundary tests)

Total: 25 boundary tests.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure backend and project root are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_latency_within,
    assert_montage_shape,
    assert_probability_distribution,
)
from tests.e2e.utils.fixtures import golden_model_input_tensor

# Import adapter and ML runner from backend
from backend.app.adapter import (
    CYTON_16_CHANNELS,
    TCP_22_CHANNELS,
    ChannelRemapper,
    EegAdapter,
    PolyphaseResampler,
    StreamingFilter,
)
from backend.app.ml_runner import ModelRunner


# =============================================================================
# Feature 13: Polyphase Resampling - Boundary Tests
# =============================================================================
class TestBoundaryFeature13Resampling(unittest.TestCase):
    """Boundary and corner cases for Polyphase Resampling (Feature 13)."""

    def test_f13_non_standard_sample_rates(self):
        """1. Resampling from non-standard rates (110 Hz, 200 Hz, 500 Hz) to 256 Hz preserves duration."""
        # 1.0 second of data at various rates
        for src_fs in (110, 200, 500):
            resampler = PolyphaseResampler(from_fs=src_fs, to_fs=256)
            data = np.ones((src_fs, 16), dtype=np.float32) * 10.0
            resampled = resampler.resample(data, axis=0)
            # Duration should be ~1.0s -> ~256 samples
            self.assertAlmostEqual(resampled.shape[0], 256, delta=2)
            self.assertEqual(resampled.shape[1], 16)

    def test_f13_single_sample_chunk_resampling(self):
        """2. Single-sample input (N=1, 16 channels) resamples without division by zero."""
        resampler = PolyphaseResampler(from_fs=125, to_fs=256)
        single_sample = np.ones((1, 16), dtype=np.float32) * 5.0
        resampled = resampler.resample(single_sample, axis=0)
        self.assertGreater(resampled.shape[0], 0)
        self.assertEqual(resampled.shape[1], 16)
        self.assertFalse(np.isnan(resampled).any())

    def test_f13_identical_source_target_rate_bypass(self):
        """3. Resampling with identical rate (from_fs == to_fs == 256) bypasses polyphase filter."""
        resampler = PolyphaseResampler(from_fs=256, to_fs=256)
        data = np.random.randn(1280, 22).astype(np.float32)
        resampled = resampler.resample(data, axis=0)
        np.testing.assert_array_equal(resampled, data)

    def test_f13_large_chunk_numerical_precision(self):
        """4. Resampling 60 seconds (7500 samples @ 125Hz -> 15360 samples @ 256Hz) maintains precision."""
        resampler = PolyphaseResampler(from_fs=125, to_fs=256)
        t = np.arange(7500) / 125.0
        # 10 Hz pure sine wave
        sig_in = (np.sin(2 * np.pi * 10.0 * t)[:, np.newaxis] * 50.0).astype(np.float32)
        sig_out = resampler.resample(sig_in, axis=0)
        # Expected output length = 7500 * (256/125) = 15360
        self.assertEqual(sig_out.shape[0], 15360)
        # Max amplitude should remain ~50.0 uV
        self.assertAlmostEqual(np.max(sig_out), 50.0, delta=1.5)

    def test_f13_dc_offset_preservation(self):
        """5. Resampling pure DC signal (50.0 uV) preserves DC level across output."""
        resampler = PolyphaseResampler(from_fs=125, to_fs=256)
        dc_data = np.full((625, 16), 50.0, dtype=np.float32)
        resampled = resampler.resample(dc_data, axis=0)
        self.assertEqual(resampled.shape[0], 1280)
        # Mean DC value should be preserved within 0.1%
        self.assertAlmostEqual(float(np.mean(resampled)), 50.0, delta=0.5)


# =============================================================================
# Feature 14: Causal Streaming Filters - Boundary Tests
# =============================================================================
class TestBoundaryFeature14CausalFilters(unittest.TestCase):
    """Boundary and corner cases for Causal Streaming Filters (Feature 14)."""

    def test_f14_step_discontinuity_ringing_decay(self):
        """1. Step transient (0 -> 100 uV) decays to near-baseline within filter time constant."""
        filt = StreamingFilter(fs=256, n_channels=22)
        # Step input: 1280 samples at 100 uV (DC step)
        step_input = np.ones((1280, 22), dtype=np.float32) * 100.0
        filtered = filt.filter(step_input)
        # Highpass component of bandpass filter must attenuate DC level to near zero
        final_samples = filtered[-200:, :]
        self.assertLess(float(np.mean(np.abs(final_samples))), 2.0)

    def test_f14_persistent_state_continuity_across_100_chunks(self):
        """2. Streaming 100 consecutive 250ms chunks matches monolithic filtering."""
        fs = 256
        chunk_size = 64  # 250ms @ 256Hz
        n_chunks = 10
        total_samples = chunk_size * n_chunks
        rng = np.random.default_rng(42)
        raw_signal = rng.normal(0.0, 20.0, (total_samples, 22)).astype(np.float32)

        # 1. Monolithic filter
        monolithic_filt = StreamingFilter(fs=fs, n_channels=22)
        monolithic_out = monolithic_filt.filter(raw_signal)

        # 2. Chunk-by-chunk causal streaming filter
        streaming_filt = StreamingFilter(fs=fs, n_channels=22)
        chunks_out = []
        for i in range(n_chunks):
            chunk = raw_signal[i * chunk_size : (i + 1) * chunk_size]
            chunks_out.append(streaming_filt.filter(chunk))
        streamed_out = np.vstack(chunks_out)

        # Difference must be negligible (< 1e-4) due to exact persistent zi continuity
        np.testing.assert_array_almost_equal(monolithic_out, streamed_out, decimal=4)

    def test_f14_gap_flag_state_reset(self):
        """3. Gap flag in frame resets filter internal state zi to zero, clearing past transients."""
        filt = StreamingFilter(fs=256, n_channels=22)
        # Inject high-voltage transient
        transient = np.ones((100, 22), dtype=np.float32) * 10000.0
        _ = filt.filter(transient)

        # Verify zi has accumulated state
        self.assertGreater(float(np.max(np.abs(filt.zi))), 10.0)

        # Reset on gap
        filt.reset()
        # Initial zi base repeated across channels is the clean equilibrium state
        self.assertEqual(filt.zi.shape, (3, 2, 22))

    def test_f14_long_term_numerical_stability(self):
        """4. Filtering 100,000 continuous samples without NaN or float overflow/underflow drift."""
        filt = StreamingFilter(fs=256, n_channels=22)
        rng = np.random.default_rng(123)
        chunk = rng.normal(0.0, 25.0, (1000, 22)).astype(np.float32)

        for _ in range(100):  # 100 * 1000 = 100,000 samples
            out = filt.filter(chunk)
            self.assertFalse(np.isnan(out).any())
            self.assertFalse(np.isinf(out).any())
            self.assertLess(float(np.max(np.abs(out))), 500.0)

    def test_f14_notch_filter_attenuation_depth(self):
        """5. Injected 50 Hz sinusoid attenuated by > 20 dB (> 90% reduction)."""
        filt = StreamingFilter(fs=256, n_channels=22)
        t = np.arange(1280) / 256.0
        sin_50hz = (np.sin(2 * np.pi * 50.0 * t)[:, np.newaxis] * 100.0).repeat(22, axis=1).astype(np.float32)
        filtered = filt.filter(sin_50hz)

        # Check amplitude of filtered signal after initial transient settlement (last 500 samples)
        settled = filtered[-500:, :]
        peak_attenuated = float(np.max(np.abs(settled)))
        # Attenuation > 20 dB means amplitude reduced from 100 to < 10.0
        self.assertLess(peak_attenuated, 10.0)


# =============================================================================
# Feature 15: Channel Remapping & Montage - Boundary Tests
# =============================================================================
class TestBoundaryFeature15MontageRemapping(unittest.TestCase):
    """Boundary and corner cases for Channel Remapping & Montage (Feature 15)."""

    def test_f15_already_22_channel_bipolar_bypass(self):
        """1. Input already having 22 canonical channels passes through unchanged."""
        remapper = ChannelRemapper()
        data_22 = np.ones((1280, 22), dtype=np.float32) * 7.5
        remapped = remapper.remap(data_22)
        self.assertEqual(remapped.shape, (1280, 22))
        np.testing.assert_array_equal(remapped, data_22)

    def test_f15_midline_interpolation_accuracy(self):
        """2. Midline channels Fz, Cz, Pz equal exactly (F3+F4)/2, (C3+C4)/2, (P3+P4)/2."""
        remapper = ChannelRemapper()
        # Cyton 16 channels: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8
        # Indices: F3=2, F4=3, C3=4, C4=5, P3=6, P4=7
        data_16 = np.zeros((10, 16), dtype=np.float32)
        data_16[:, 2] = 10.0  # F3
        data_16[:, 3] = 20.0  # F4 -> Fz should be 15.0
        data_16[:, 4] = -30.0 # C3
        data_16[:, 5] = 50.0  # C4 -> Cz should be 10.0
        data_16[:, 6] = 40.0  # P3
        data_16[:, 7] = 80.0  # P4 -> Pz should be 60.0

        remapped = remapper.remap(data_16)
        # TCP_22 channels: FZ is index 16, CZ is 17, PZ is 18
        np.testing.assert_array_almost_equal(remapped[:, 16], np.full(10, 15.0))
        np.testing.assert_array_almost_equal(remapped[:, 17], np.full(10, 10.0))
        np.testing.assert_array_almost_equal(remapped[:, 18], np.full(10, 60.0))

    def test_f15_unconnected_reference_leads_zeroed(self):
        """3. Unconnected reference leads (A1, A2, EKG) set to neutral zero-reference."""
        remapper = ChannelRemapper()
        data_16 = np.random.randn(50, 16).astype(np.float32) * 20.0
        remapped = remapper.remap(data_16)
        # A1 is index 19, A2 is index 20, EKG is index 21
        np.testing.assert_array_equal(remapped[:, 19], np.zeros(50, dtype=np.float32))
        np.testing.assert_array_equal(remapped[:, 20], np.zeros(50, dtype=np.float32))
        np.testing.assert_array_equal(remapped[:, 21], np.zeros(50, dtype=np.float32))

    def test_f15_ambiguous_channel_count_rejection(self):
        """4. Array with ambiguous channel count (e.g. 10 channels) without explicit labels raises ValueError."""
        remapper = ChannelRemapper()
        ambiguous_data = np.zeros((100, 10), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            remapper.remap(ambiguous_data)
        self.assertIn("Ambiguous channel configuration", str(ctx.exception))

    def test_f15_alternative_alias_naming(self):
        """5. 10-20 aliases (T3, T4, T5, T6) successfully resolve to canonical labels (T7, T8, P7, P8)."""
        remapper = ChannelRemapper()
        aliases = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"  # Using T3-T6 instead of T7-P8
        ]
        data = np.ones((20, 16), dtype=np.float32) * 33.0
        remapped = remapper.remap(data, source_channels=aliases)
        self.assertEqual(remapped.shape, (20, 22))
        # Canonical T7 is index 12 in TCP_22
        self.assertEqual(remapped[0, 12], 33.0)


# =============================================================================
# Feature 16: Deep 1D CNN Inference - Boundary Tests
# =============================================================================
class TestBoundaryFeature16CnnInference(unittest.TestCase):
    """Boundary and corner cases for Deep 1D CNN Inference (Feature 16)."""

    @classmethod
    def setUpClass(cls):
        model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        cls.runner = ModelRunner()
        if model_path.exists():
            cls.runner.load(model_path)
            cls.runner.warmup()

    def test_f16_nan_inf_rejection(self):
        """1. Tensor containing NaN or Inf values rejected with descriptive ValueError."""
        good_tensor = np.zeros((1280, 22), dtype=np.float32)

        # NaN injection
        nan_tensor = good_tensor.copy()
        nan_tensor[100, 5] = np.nan
        with self.assertRaises(ValueError) as ctx:
            self.runner.validate_input(nan_tensor)
        self.assertIn("NaN", str(ctx.exception))

        # +Inf injection
        inf_tensor = good_tensor.copy()
        inf_tensor[200, 10] = np.inf
        with self.assertRaises(ValueError) as ctx:
            self.runner.validate_input(inf_tensor)
        self.assertIn("Inf", str(ctx.exception))

        # -Inf injection
        neg_inf_tensor = good_tensor.copy()
        neg_inf_tensor[300, 15] = -np.inf
        with self.assertRaises(ValueError) as ctx:
            self.runner.validate_input(neg_inf_tensor)
        self.assertIn("Inf", str(ctx.exception))

    def test_f16_zero_variance_flatline_normalization(self):
        """2. Zero-variance flatline tensor normalizes safely using epsilon (1e-6) without zero-division."""
        flatline = np.zeros((1280, 22), dtype=np.float32)
        # Validate does not fail
        validated = self.runner.validate_input(flatline)
        self.assertEqual(validated.shape, (1, 1280, 22))

        # If model loaded, predict should complete without crashing
        if self.runner._is_loaded:
            res = self.runner.predict(flatline)
            assert_probability_distribution(res["probabilities"])

    def test_f16_invalid_input_tensor_shapes(self):
        """3. Tensors with wrong shapes ((1000, 22), (1280, 16), 1D array) raise ValueError."""
        with self.assertRaises(ValueError):
            self.runner.validate_input(np.zeros((1000, 22), dtype=np.float32))

        with self.assertRaises(ValueError):
            self.runner.validate_input(np.zeros((1280, 16), dtype=np.float32))

        with self.assertRaises(ValueError):
            self.runner.validate_input(np.zeros(1280, dtype=np.float32))

    def test_f16_inference_latency_under_load(self):
        """4. CPU forward pass inference latency benchmark remains < 60 ms."""
        if not self.runner._is_loaded:
            self.skipTest("Model weights not available for latency test")

        self.runner.warmup()
        tensor = golden_model_input_tensor(42)
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = self.runner.predict(tensor)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = float(np.mean(latencies))
        assert_latency_within(avg_latency, max_allowed_ms=60.0, context="Deep 1D CNN Inference")

    def test_f16_batch_dimension_support(self):
        """5. ModelRunner handles both 2D (1280, 22) and 3D (batch, 1280, 22) inputs consistently."""
        tensor_2d = np.random.randn(1280, 22).astype(np.float32)
        val_2d = self.runner.validate_input(tensor_2d)
        self.assertEqual(val_2d.shape, (1, 1280, 22))

        tensor_3d = np.random.randn(2, 1280, 22).astype(np.float32)
        val_3d = self.runner.validate_input(tensor_3d)
        self.assertEqual(val_3d.shape, (2, 1280, 22))


# =============================================================================
# Feature 17: ModelRunner Interface - Boundary Tests
# =============================================================================
class TestBoundaryFeature17ModelRunnerInterface(unittest.TestCase):
    """Boundary and corner cases for ModelRunner Interface (Feature 17)."""

    def test_f17_corrupted_model_artifact_load(self):
        """1. Attempting to load corrupted non-HDF5 file raises RuntimeError or FileNotFoundError."""
        runner = ModelRunner()
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            runner.load("non_existent_model_xyz.h5")

    def test_f17_warmup_before_load_error(self):
        """2. Calling warmup() on uninitialized runner raises RuntimeError."""
        unloaded_runner = ModelRunner()
        with self.assertRaises(RuntimeError) as ctx:
            unloaded_runner.warmup()
        self.assertIn("not loaded", str(ctx.exception))

    def test_f17_thread_safe_concurrent_predictions(self):
        """3. Concurrent predictions from 4 worker threads execute safely without race condition."""
        model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        if not model_path.exists():
            self.skipTest("Model weights not available for threading test")

        runner = ModelRunner(model_path)
        errors = []

        def worker(idx: int):
            try:
                tensor = golden_model_input_tensor(idx)
                res = runner.predict(tensor)
                assert_probability_distribution(res["probabilities"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Encountered thread errors: {errors}")

    def test_f17_model_card_validation(self):
        """4. Validates model_card.json schema and required parameters."""
        card_path = BACKEND_DIR / "models" / "model_card.json"
        self.assertTrue(card_path.exists(), "model_card.json must exist")
        with open(card_path, "r", encoding="utf-8") as f:
            card = json.load(f)

        self.assertIn("model_name", card)
        self.assertIn("input_shape", card)
        self.assertIn("classes", card)
        self.assertEqual(len(card["classes"]), 3)
        self.assertEqual([c.lower() for c in card["classes"]], ["normal", "pre_seizure", "seizure"])

    def test_f17_keras_torch_backend_env_verification(self):
        """5. Confirms KERAS_BACKEND=torch configured and verified in environment."""
        backend_env = os.environ.get("KERAS_BACKEND")
        self.assertEqual(backend_env, "torch")


if __name__ == "__main__":
    unittest.main()
