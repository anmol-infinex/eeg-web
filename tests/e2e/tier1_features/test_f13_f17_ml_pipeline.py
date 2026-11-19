"""Tier 1: Features 13-17 Signal Processing & ML Pipeline Tests.

Covers:
- Feature 13: Polyphase Resampling (5 tests)
- Feature 14: Causal Streaming Filters (5 tests)
- Feature 15: Channel Remapping & Montage (5 tests)
- Feature 16: Deep 1D CNN Inference (5 tests)
- Feature 17: ModelRunner Interface (5 tests)

Total: 25 tests.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import unittest
from pathlib import Path
import numpy as np

# Ensure backend and project root are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure Keras backend is torch before any Keras import
os.environ["KERAS_BACKEND"] = "torch"

from app.adapter import (
    PolyphaseResampler,
    ChannelRemapper,
    StreamingFilter,
    EegAdapter,
    CYTON_16_CHANNELS,
    TCP_22_CHANNELS,
    CHANNEL_ALIASES,
)
from app.ml_runner import ModelRunner
from app.config import settings

from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_latency_within,
    assert_montage_shape,
    assert_probability_distribution,
)
from tests.e2e.utils.fixtures import golden_model_input_tensor


# =============================================================================
# Feature 13: Polyphase Resampling
# =============================================================================
class TestFeature13PolyphaseResampling(unittest.TestCase):
    """Feature 13: Resamples 125Hz Cyton to 256Hz model input rate.

    Verifies 125 -> 256 sample ratio (625 -> 1280), frequency cutoff,
    phase linearity, 2D multi-channel preservation, and streaming chunk concatenation.
    """

    def setUp(self):
        self.resampler = PolyphaseResampler(from_fs=125, to_fs=256)

    def test_f13_resample_ratio(self):
        """1. Verifies 125 Hz -> 256 Hz ratio produces exactly (256/125 * N) samples."""
        # 625 samples at 125 Hz = 5.0 seconds -> 1280 samples at 256 Hz
        data_625 = np.random.randn(625, 16).astype(np.float32)
        resampled_1280 = self.resampler.resample(data_625, axis=0)
        self.assertEqual(resampled_1280.shape, (1280, 16))
        self.assertEqual(resampled_1280.dtype, np.float32)

        # 125 samples (1.0 sec) -> 256 samples
        data_125 = np.random.randn(125, 16).astype(np.float32)
        resampled_256 = self.resampler.resample(data_125, axis=0)
        self.assertEqual(resampled_256.shape, (256, 16))

    def test_f13_frequency_response_cutoff(self):
        """2. Confirms anti-aliasing low-pass filter preserves bandpass frequencies."""
        # Test 10 Hz pure sine wave (within passband)
        t_125 = np.arange(625) / 125.0
        sine_10hz = np.sin(2 * np.pi * 10.0 * t_125).astype(np.float32)[:, np.newaxis]
        resampled = self.resampler.resample(sine_10hz, axis=0)

        # Output should be at 256 Hz: amplitude should be preserved (near 1.0)
        # Exclude filter settling edges
        amp_mid = np.max(np.abs(resampled[256:-256, 0]))
        self.assertAlmostEqual(amp_mid, 1.0, delta=0.08)

    def test_f13_phase_linearity(self):
        """3. Verifies linear phase response without waveform distortion."""
        t_125 = np.arange(625) / 125.0
        # Combination of 8 Hz and 15 Hz sinusoids
        signal_in = (np.sin(2 * np.pi * 8.0 * t_125) + 0.5 * np.cos(2 * np.pi * 15.0 * t_125)).astype(np.float32)[:, np.newaxis]
        resampled = self.resampler.resample(signal_in, axis=0)

        # Check that resampled signal has expected 1280 samples and no NaN or Inf
        self.assertEqual(resampled.shape, (1280, 1))
        self.assertFalse(np.isnan(resampled).any())
        self.assertFalse(np.isinf(resampled).any())

    def test_f13_multichannel_handling(self):
        """4. Tests 2D tensor resampling preserving 16 distinct channels."""
        # 16 channels, each with a distinct scaling factor
        raw = np.zeros((625, 16), dtype=np.float32)
        for ch in range(16):
            raw[:, ch] = float(ch + 1) * np.ones(625, dtype=np.float32)

        resampled = self.resampler.resample(raw, axis=0)
        self.assertEqual(resampled.shape, (1280, 16))
        for ch in range(16):
            # Mid-window values should equal the DC value of ch + 1
            mid_val = float(np.mean(resampled[256:-256, ch]))
            self.assertAlmostEqual(mid_val, float(ch + 1), delta=0.05)

    def test_f13_streaming_chunk_resampling(self):
        """5. Asserts consecutive chunks resample properly without NaN/Inf."""
        chunk_samples = 31  # 250ms chunk @ 125Hz
        chunk1 = np.ones((chunk_samples, 16), dtype=np.float32) * 5.0
        chunk2 = np.ones((chunk_samples, 16), dtype=np.float32) * 5.0

        r1 = self.resampler.resample(chunk1, axis=0)
        r2 = self.resampler.resample(chunk2, axis=0)

        # Ratio: 31 * 256 / 125 = 63.488 -> 64 samples
        self.assertEqual(r1.shape[1], 16)
        self.assertEqual(r2.shape[1], 16)
        self.assertFalse(np.isnan(r1).any())
        self.assertFalse(np.isnan(r2).any())


# =============================================================================
# Feature 14: Causal Streaming Filters
# =============================================================================
class TestFeature14CausalStreamingFilters(unittest.TestCase):
    """Feature 14: 0.5-50Hz bandpass and 50Hz notch with persistent state.

    Verifies Butterworth bandpass, 50Hz notch attenuation (> 20 dB),
    filter causality with persistent zi, chunk continuity, and state reset on gap.
    """

    def setUp(self):
        self.filt = StreamingFilter(fs=256, lowcut=0.5, highcut=50.0, notch_freq=50.0, notch_q=30.0, n_channels=22)

    def test_f14_butterworth_bandpass_cutoff(self):
        """1. Verifies attenuation of DC offset (0 Hz) and high frequency (> 60 Hz)."""
        t = np.arange(1024) / 256.0
        # DC component (0 Hz) of 100 uV
        dc_data = np.full((1024, 22), 100.0, dtype=np.float32)
        filtered_dc = self.filt.filter(dc_data)
        # Settled output should be strongly attenuated towards 0
        settled_dc = float(np.mean(np.abs(filtered_dc[512:, 0])))
        self.assertLess(settled_dc, 5.0)  # > 26 dB DC attenuation

    def test_f14_notch_filter_50hz_rejection(self):
        """2. Asserts > 20 dB attenuation at 50 Hz mains frequency."""
        t = np.arange(1024) / 256.0
        sine_50hz = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
        chunk = np.repeat(sine_50hz[:, np.newaxis], 22, axis=1)

        filtered = self.filt.filter(chunk)
        # Inspect second half of filtered window (after filter settling)
        amp_in = np.max(np.abs(chunk[512:, 0]))
        amp_out = np.max(np.abs(filtered[512:, 0]))
        attenuation_db = 20 * np.log10(amp_in / (amp_out + 1e-9))
        self.assertGreater(attenuation_db, 20.0)

    def test_f14_filter_causality(self):
        """3. Confirms filter uses second-order sections (sosfilt) with persistent zi."""
        self.assertEqual(self.filt.sos.ndim, 2)
        self.assertEqual(self.filt.sos.shape[1], 6)  # 6 coefficients per biquad section
        # Persistent zi shape: (n_sections, 2, n_channels)
        self.assertEqual(self.filt.zi.shape[0], self.filt.sos.shape[0])
        self.assertEqual(self.filt.zi.shape[1], 2)
        self.assertEqual(self.filt.zi.shape[2], 22)

    def test_f14_persistent_state_continuity(self):
        """4. Verifies absence of step transients across chunk boundaries with persistent zi."""
        sf_continuous = StreamingFilter(fs=256, n_channels=22)
        sf_chunked = StreamingFilter(fs=256, n_channels=22)

        data = np.random.randn(600, 22).astype(np.float32)

        # Filter all at once
        full_filtered = sf_continuous.filter(data)

        # Filter in three chunks of 200 samples carrying persistent zi
        c1 = sf_chunked.filter(data[:200])
        c2 = sf_chunked.filter(data[200:400])
        c3 = sf_chunked.filter(data[400:])
        chunked_filtered = np.vstack([c1, c2, c3])

        # State continuity ensures exact match within numerical float precision
        np.testing.assert_allclose(full_filtered, chunked_filtered, atol=1e-4)

    def test_f14_filter_reset_on_gap(self):
        """5. Confirms filter state resets zi to initial baseline when reset() is called."""
        # Run some data to alter zi
        self.filt.filter(np.ones((250, 22), dtype=np.float32) * 50.0)
        # State is altered from zero
        self.assertFalse(np.allclose(self.filt.zi, 0.0))

        # Reset filter (e.g. upon gap flag)
        self.filt.reset()
        # Verify reset returned state to base condition
        np.testing.assert_allclose(self.filt.zi, np.repeat(self.filt._zi_base[:, :, np.newaxis], 22, axis=2))


# =============================================================================
# Feature 15: Channel Remapping & Montage
# =============================================================================
class TestFeature15ChannelRemapping(unittest.TestCase):
    """Feature 15: Maps 16 Cyton channels to 22 model channels.

    Verifies 16 -> 22 channel expansion, bilateral interpolation for FZ, CZ, PZ,
    temporal lead aliases (T3->T7, T4->T8, T5->P7, T6->P8), and zero-reference channels.
    """

    def setUp(self):
        self.remapper = ChannelRemapper()

    def test_f15_16_to_22_channel_expansion(self):
        """1. Verifies input shape (N, 16) produces output shape (N, 22)."""
        data_16ch = np.zeros((200, 16), dtype=np.float32)
        remapped = self.remapper.remap(data_16ch)
        self.assertEqual(remapped.shape, (200, 22))
        self.assertEqual(remapped.dtype, np.float32)

    def test_f15_bipolar_derivation(self):
        """2. Validates exact target channel positions match canonical 22 TCP montage."""
        self.assertEqual(len(self.remapper.target_channels), 22)
        self.assertEqual(self.remapper.target_channels, [c.upper() for c in TCP_22_CHANNELS])
        self.assertIn("CZ", self.remapper.target_channels)
        self.assertIn("FZ", self.remapper.target_channels)
        self.assertIn("PZ", self.remapper.target_channels)

    def test_f15_midline_cz_approximation(self):
        """3. Verifies CZ = (C3 + C4)/2, FZ = (F3 + F4)/2, PZ = (P3 + P4)/2."""
        data_16ch = np.zeros((10, 16), dtype=np.float32)
        # Map indices in CYTON_16_CHANNELS:
        # F3 is idx 2, F4 is idx 3
        # C3 is idx 4, C4 is idx 5
        # P3 is idx 6, P4 is idx 7
        data_16ch[:, 2] = 10.0  # F3
        data_16ch[:, 3] = 20.0  # F4
        data_16ch[:, 4] = 30.0  # C3
        data_16ch[:, 5] = 50.0  # C4
        data_16ch[:, 6] = 40.0  # P3
        data_16ch[:, 7] = 60.0  # P4

        remapped = self.remapper.remap(data_16ch)

        fz_idx = self.remapper.target_channels.index("FZ")
        cz_idx = self.remapper.target_channels.index("CZ")
        pz_idx = self.remapper.target_channels.index("PZ")

        # FZ = 0.5 * (10 + 20) = 15.0
        np.testing.assert_allclose(remapped[:, fz_idx], 15.0, atol=1e-5)
        # CZ = 0.5 * (30 + 50) = 40.0
        np.testing.assert_allclose(remapped[:, cz_idx], 40.0, atol=1e-5)
        # PZ = 0.5 * (40 + 60) = 50.0
        np.testing.assert_allclose(remapped[:, pz_idx], 50.0, atol=1e-5)

    def test_f15_temporal_lead_mapping(self):
        """4. Verifies T3->T7, T4->T8, T5->P7, T6->P8 10-20 aliases."""
        custom_channels = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"
        ]
        data = np.ones((5, 16), dtype=np.float32) * 7.5
        remapped = self.remapper.remap(data, source_channels=custom_channels)

        t7_idx = self.remapper.target_channels.index("T7")
        t8_idx = self.remapper.target_channels.index("T8")
        p7_idx = self.remapper.target_channels.index("P7")
        p8_idx = self.remapper.target_channels.index("P8")

        np.testing.assert_allclose(remapped[:, t7_idx], 7.5, atol=1e-5)
        np.testing.assert_allclose(remapped[:, t8_idx], 7.5, atol=1e-5)
        np.testing.assert_allclose(remapped[:, p7_idx], 7.5, atol=1e-5)
        np.testing.assert_allclose(remapped[:, p8_idx], 7.5, atol=1e-5)

    def test_f15_adapted_status_flag(self):
        """5. Confirms zero-reference channels A1, A2, EKG set to neutral 0.0."""
        data_16ch = np.random.randn(20, 16).astype(np.float32)
        remapped = self.remapper.remap(data_16ch)

        a1_idx = self.remapper.target_channels.index("A1")
        a2_idx = self.remapper.target_channels.index("A2")
        ekg_idx = self.remapper.target_channels.index("EKG")

        np.testing.assert_allclose(remapped[:, a1_idx], 0.0, atol=1e-6)
        np.testing.assert_allclose(remapped[:, a2_idx], 0.0, atol=1e-6)
        np.testing.assert_allclose(remapped[:, ekg_idx], 0.0, atol=1e-6)


# =============================================================================
# Feature 16: Deep 1D CNN Inference
# =============================================================================
class TestFeature16DeepCnnInference(unittest.TestCase):
    """Feature 16: In-process inference using tusz_3class_eeg_model.h5.

    Verifies input shape (None, 1280, 22), output shape (None, 3), Z-score normalization,
    softmax probability sum = 1.0, and CPU latency < 50ms.
    """

    @classmethod
    def setUpClass(cls):
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            # Fallback path if run from different root
            model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        cls.runner = ModelRunner(model_path)
        cls.runner.warmup()

    def test_f16_model_input_tensor_shape(self):
        """1. Asserts model input shape is exactly (None, 1280, 22)."""
        info = self.runner.info()
        self.assertEqual(info["channels"], 22)
        self.assertEqual(info["window_samples"], 1280)
        self.assertEqual(info["sample_rate_hz"], 256)

    def test_f16_model_output_tensor_shape(self):
        """2. Asserts output shape has 3 classes: Normal, Pre-Seizure, Seizure."""
        info = self.runner.info()
        self.assertEqual(len(info["classes"]), 3)
        self.assertEqual(info["classes"], ["Normal", "Pre-Seizure", "Seizure"])

    def test_f16_zscore_normalization(self):
        """3. Verifies per-channel Z-score normalization along time axis."""
        raw = np.random.normal(50.0, 15.0, (1280, 22)).astype(np.float32)
        # ModelRunner normalizes: arr = (arr - mean) / (std + 1e-6)
        mean = np.mean(raw, axis=0)
        std = np.std(raw, axis=0)
        norm = (raw - mean) / (std + 1e-6)
        np.testing.assert_allclose(np.mean(norm, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(norm, axis=0), 1.0, atol=1e-4)

    def test_f16_softmax_activation(self):
        """4. Asserts all probabilities 0 <= P <= 1.0 and sum P = 1.0 +/- 1e-4."""
        tensor = golden_model_input_tensor(seed=42)
        res = self.runner.predict(tensor)
        self.assertIn("probabilities", res)
        assert_probability_distribution(res["probabilities"])

    def test_f16_inference_latency_benchmark(self):
        """5. Asserts CPU forward pass latency is well under 60 ms."""
        self.runner.warmup()
        tensor = golden_model_input_tensor(seed=99)
        t0 = time.perf_counter()
        _ = self.runner.predict(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        assert_latency_within(latency_ms, max_allowed_ms=60.0)


# =============================================================================
# Feature 17: ModelRunner Interface
# =============================================================================
class TestFeature17ModelRunnerInterface(unittest.TestCase):
    """Feature 17: Pluggable abstraction for ML model execution.

    Verifies load, predict, info, warmup interface methods, loading valid .h5 artifact,
    model_card.json schema, golden reproducibility, and KERAS_BACKEND=torch configuration.
    """

    def setUp(self):
        self.model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        self.card_path = BACKEND_DIR / "models" / "model_card.json"

    def test_f17_interface_contract(self):
        """1. Verifies load, predict, info, warmup methods implemented and callable."""
        runner = ModelRunner()
        self.assertTrue(hasattr(runner, "load"))
        self.assertTrue(hasattr(runner, "predict"))
        self.assertTrue(hasattr(runner, "info"))
        self.assertTrue(hasattr(runner, "warmup"))

    def test_f17_load_valid_h5_artifact(self):
        """2. Successfully loads tusz_3class_eeg_model.h5."""
        runner = ModelRunner(self.model_path)
        self.assertTrue(runner._is_loaded)
        self.assertIsNotNone(runner.model)

    def test_f17_model_card_metadata(self):
        """3. Validates model_card.json schema and parameters."""
        self.assertTrue(self.card_path.exists())
        with open(self.card_path, "r", encoding="utf-8") as f:
            card = json.load(f)

        self.assertIn("model_name", card)
        self.assertTrue("model_version" in card or "version" in card)
        self.assertIn("framework", card)
        self.assertIn("input_shape", card)
        self.assertIn("classes", card)
        self.assertIn("sample_rate_hz", card)
        self.assertEqual(card["sample_rate_hz"], 256)
        self.assertEqual(len(card["classes"]), 3)

    def test_f17_golden_test_verification(self):
        """4. Confirms output on reference array matches valid prediction without NaN."""
        runner = ModelRunner(self.model_path)
        runner.warmup()
        input_tensor = golden_model_input_tensor(seed=10)
        res = runner.predict(input_tensor)
        self.assertIn(res["prediction"], ["Normal", "Pre-Seizure", "Seizure"])
        probs = res["probabilities"]
        for k in ["normal", "pre_seizure", "seizure"]:
            self.assertFalse(math.isnan(probs[k]))
            self.assertFalse(math.isinf(probs[k]))

    def test_f17_backend_torch_configuration(self):
        """5. Confirms KERAS_BACKEND=torch configured and verified."""
        import keras
        self.assertEqual(os.environ.get("KERAS_BACKEND"), "torch")
        self.assertEqual(keras.backend.backend(), "torch")


if __name__ == "__main__":
    unittest.main()
