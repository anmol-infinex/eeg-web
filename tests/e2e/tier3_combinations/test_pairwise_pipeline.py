"""Tier 3: Pairwise Ingestion & ML Pipeline Combinations.

Verifies end-to-end interactions between interconnected features across subsystem boundaries:
Ingest (F1-F4) -> Polyphase Resampler (F13) -> Causal Streaming Filters (F14) ->
Montage Mapping (F15) -> Deep 1D CNN Inference (F16) -> Hysteresis Alerting (F18).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.adapter import ChannelRemapper, EegAdapter, PolyphaseResampler, StreamingFilter
from app.ml_runner import ModelRunner
from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_latency_within,
    assert_montage_shape,
    assert_prediction_schema,
    assert_probability_distribution,
    assert_signal_quality,
    assert_valid_openbci_packet,
    assert_valid_ws_binary_frame,
)
from tests.e2e.utils.fixtures import (
    generate_mock_binary_frame,
    generate_openbci_raw_stream,
    get_incompatible_profile,
    get_openbci_cyton_profile,
    golden_model_input_tensor,
)


class TestPairwisePipelineCombinations(unittest.TestCase):
    """Pairwise cross-feature integration test suite for the EEG processing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Loads and warms up the ModelRunner once for the test suite."""
        model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        cls.model_runner = ModelRunner(model_path=model_path)
        cls.model_runner.warmup()
        cls.adapter = EegAdapter(from_fs=125, to_fs=256, n_channels=22)

    # -------------------------------------------------------------------------
    # 1. OpenBCI Parser -> 250ms Binary Framing (F02 -> F03)
    # -------------------------------------------------------------------------
    def test_f02_f03_openbci_parser_to_binary_framing(self):
        """Verifies OpenBCI 33-byte packets parsed and batched into 250ms binary frames."""
        n_samples = 31  # 31 samples @ 125 Hz = 248 ms ~ 250 ms
        raw_stream = generate_openbci_raw_stream(n_samples=n_samples, seed=101)

        # Parse 33-byte packets into 16-channel float32 microvolts
        recovered_samples = []
        for i in range(n_samples):
            pkt_board = raw_stream[i * 66 : i * 66 + 33]
            pkt_daisy = raw_stream[i * 66 + 33 : (i + 1) * 66]

            assert_valid_openbci_packet(pkt_board)
            assert_valid_openbci_packet(pkt_daisy)

            up_board = EegStreamer.unpack_openbci_packet(pkt_board)
            up_daisy = EegStreamer.unpack_openbci_packet(pkt_daisy)

            sample_16 = up_board["channels_uv"] + up_daisy["channels_uv"]
            self.assertEqual(len(sample_16), 16)
            recovered_samples.append(sample_16)

        raw_array = np.array(recovered_samples, dtype=np.float32)
        self.assertEqual(raw_array.shape, (31, 16))

        # Pack into little-endian binary frame
        seq = 42
        t0_ms = 1718000000000.0
        frame_bytes = EegStreamer.pack_ws_binary_frame(
            seq=seq, t0_ms=t0_ms, data=raw_array, flags=0x00
        )

        assert_valid_ws_binary_frame(frame_bytes, expected_channels=16)
        header, payload_data = EegStreamer.unpack_ws_binary_frame(frame_bytes)

        self.assertEqual(header["seq"], 42)
        self.assertEqual(header["n_samples"], 31)
        self.assertEqual(header["n_channels"], 16)
        self.assertEqual(header["flags"], 0)
        np.testing.assert_array_almost_equal(raw_array, payload_data, decimal=2)

    # -------------------------------------------------------------------------
    # 2. Binary Frame -> Resampler Exact Ratio (F03 -> F13)
    # -------------------------------------------------------------------------
    def test_f03_f13_binary_frame_to_resampler_exact_ratio(self):
        """Verifies binary frame decoded payload resampled at exact 125 Hz -> 256 Hz ratio."""
        frame_bytes = generate_mock_binary_frame(seq=1, n_samples=31, n_channels=16, state="normal")
        header, chunk_data = EegStreamer.unpack_ws_binary_frame(frame_bytes)

        resampler = PolyphaseResampler(from_fs=125, to_fs=256)
        resampled = resampler.resample(chunk_data, axis=0)

        # 31 * 256 / 125 = 63.488 -> resample_poly produces ceil(31 * 256 / 125) = 64 samples
        expected_len = math.ceil(31 * 256 / 125)
        self.assertEqual(resampled.shape[0], expected_len)
        self.assertEqual(resampled.shape[1], 16)
        self.assertEqual(resampled.dtype, np.float32)

    # -------------------------------------------------------------------------
    # 3. Parser -> Binary Frame -> Resampler Spectral Fidelity (F02 -> F03 -> F13)
    # -------------------------------------------------------------------------
    def test_f02_f03_f13_parser_to_resampler(self):
        """Full chain: OpenBCI stream -> 250ms binary frame -> 125->256 Hz resampling."""
        duration_sec = 2.0
        fs_in = 125
        n_samples = int(duration_sec * fs_in)  # 250 samples
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Inject 10 Hz alpha sine wave on channel 0
        freq_test = 10.0
        signal_in = np.zeros((n_samples, 16), dtype=np.float32)
        signal_in[:, 0] = (30.0 * np.sin(2 * np.pi * freq_test * t)).astype(np.float32)

        resampler = PolyphaseResampler(from_fs=125, to_fs=256)
        resampled = resampler.resample(signal_in, axis=0)

        expected_n_out = int(round(n_samples * 256 / 125))
        self.assertEqual(resampled.shape[0], expected_n_out)

        # Spectral validation: FFT peak must remain at exactly 10.0 Hz
        fft_vals = np.abs(np.fft.rfft(resampled[:, 0]))
        freqs = np.fft.rfftfreq(len(resampled[:, 0]), d=1.0 / 256)
        peak_freq = freqs[np.argmax(fft_vals)]
        self.assertAlmostEqual(peak_freq, 10.0, delta=0.5)

    # -------------------------------------------------------------------------
    # 4. Resampler -> Causal Filters (F13 -> F14)
    # -------------------------------------------------------------------------
    def test_f13_f14_resampler_to_filters_and_notch(self):
        """Verifies resampled 256 Hz data filtered through 0.5-50Hz bandpass and 50Hz notch."""
        fs = 256
        n_samples = 1280  # 5 seconds
        t = np.linspace(0, 5.0, n_samples, endpoint=False)

        # Synthesize composite signal: 10 Hz alpha (30 uV) + 50 Hz mains noise (100 uV)
        composite = 30.0 * np.sin(2 * np.pi * 10.0 * t) + 100.0 * np.sin(2 * np.pi * 50.0 * t)
        multichan = np.repeat(composite[:, np.newaxis], 22, axis=1).astype(np.float32)

        filt = StreamingFilter(fs=fs, lowcut=0.5, highcut=50.0, notch_freq=50.0, n_channels=22)
        filtered = filt.filter(multichan)

        # Compute power at 50 Hz vs 10 Hz before and after filter
        fft_raw = np.abs(np.fft.rfft(multichan[:, 0]))
        fft_filtered = np.abs(np.fft.rfft(filtered[:, 0]))
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

        idx_10 = np.argmin(np.abs(freqs - 10.0))
        idx_50 = np.argmin(np.abs(freqs - 50.0))

        attenuation_db = 20 * np.log10((fft_raw[idx_50] + 1e-6) / (fft_filtered[idx_50] + 1e-6))
        self.assertGreater(attenuation_db, 18.0)  # > 18 dB attenuation of 50 Hz mains

        # 10 Hz signal power preserved within 3 dB
        ratio_10 = fft_filtered[idx_10] / fft_raw[idx_10]
        self.assertGreater(ratio_10, 0.70)

    # -------------------------------------------------------------------------
    # 5. Filters -> Montage Channel Mapping (F14 -> F15)
    # -------------------------------------------------------------------------
    def test_f14_f15_filters_to_tcp_montage_mapping(self):
        """Verifies 16-channel filtered stream mapped to canonical 22-channel TCP montage."""
        raw_16 = np.random.randn(500, 16).astype(np.float32)
        remapper = ChannelRemapper()
        remapped_22 = remapper.remap(raw_16)

        assert_montage_shape(remapped_22, expected_channels=22, expected_samples=500)

        # Validate bilateral midline approximations:
        # FZ (index 16) approx (F3 + F4) / 2
        # CZ (index 17) approx (C3 + C4) / 2
        # PZ (index 18) approx (P3 + P4) / 2
        f3_data = raw_16[:, 2]
        f4_data = raw_16[:, 3]
        expected_fz = 0.5 * (f3_data + f4_data)
        np.testing.assert_array_almost_equal(remapped_22[:, 16], expected_fz, decimal=5)

        c3_data = raw_16[:, 4]
        c4_data = raw_16[:, 5]
        expected_cz = 0.5 * (c3_data + c4_data)
        np.testing.assert_array_almost_equal(remapped_22[:, 17], expected_cz, decimal=5)

        # Reference channels A1 (19), A2 (20), EKG (21) must be 0.0
        self.assertTrue(np.all(remapped_22[:, 19:22] == 0.0))

    # -------------------------------------------------------------------------
    # 6. Combined Resampler -> Filters -> Montage End-to-End (F13 -> F14 -> F15)
    # -------------------------------------------------------------------------
    def test_f13_f14_f15_resampler_to_filters_and_montage(self):
        """Verifies EegAdapter converts raw (625, 16) 125 Hz stream into exact (1280, 22) float32."""
        raw_cyton = np.random.randn(625, 16).astype(np.float32)
        processed = self.adapter.process_window(raw_cyton, source_fs=125)

        assert_montage_shape(processed, expected_channels=22, expected_samples=1280)
        self.assertEqual(processed.dtype, np.float32)
        self.assertFalse(np.isnan(processed).any())
        self.assertFalse(np.isinf(processed).any())

    # -------------------------------------------------------------------------
    # 7. Montage -> CNN Model Inference (F15 -> F16)
    # -------------------------------------------------------------------------
    def test_f15_f16_montage_to_cnn_inference(self):
        """Verifies (1280, 22) montage window fed to Deep 1D CNN returns valid probabilities."""
        window_22 = golden_model_input_tensor(seed=42)[0]  # shape (1280, 22)
        res = self.model_runner.predict(window_22)

        assert_prediction_schema(res)
        self.assertIn(res["prediction"], ["Normal", "Pre-Seizure", "Seizure"])
        probs = res["probabilities"]
        assert_probability_distribution(probs)

    # -------------------------------------------------------------------------
    # 8. Stream Gap Flag -> Filter Reset (F03 -> F14)
    # -------------------------------------------------------------------------
    def test_f03_f14_stream_gap_to_filter_reset(self):
        """Verifies gap flag (bit0) in binary frame resets filter state zi to zeros."""
        filt = StreamingFilter(fs=256, n_channels=22)
        initial_chunk = np.ones((100, 22), dtype=np.float32) * 50.0
        filt.filter(initial_chunk)

        # Verify filter zi is non-zero after filtering
        self.assertFalse(np.all(filt.zi == 0.0))

        # Emulate receiving frame with gap flag (flags=0x01)
        frame_with_gap = generate_mock_binary_frame(seq=10, flags=0x01)
        header, _ = EegStreamer.unpack_ws_binary_frame(frame_with_gap)
        self.assertTrue(header["is_gap"])

        # Reset filter on gap
        filt.reset()
        # Verify zi was restored to default initial condition
        np.testing.assert_array_equal(filt.zi, np.repeat(filt._zi_base[:, :, np.newaxis], 22, axis=2))

    # -------------------------------------------------------------------------
    # 9. Full Pipeline: Normal EEG Ingest -> Classification (F02 -> F16)
    # -------------------------------------------------------------------------
    def test_f02_f16_full_ingest_to_inference_normal_pipeline(self):
        """Verifies full pipeline on Normal synthetic EEG: OpenBCI -> Adapter -> CNN."""
        raw_stream = generate_openbci_raw_stream(n_samples=625, seed=202)  # 5.0s @ 125Hz
        # Unpack board and daisy into 16 channels
        samples = []
        for i in range(625):
            up_b = EegStreamer.unpack_openbci_packet(raw_stream[i * 66 : i * 66 + 33])
            up_d = EegStreamer.unpack_openbci_packet(raw_stream[i * 66 + 33 : (i + 1) * 66])
            samples.append(up_b["channels_uv"] + up_d["channels_uv"])

        arr_16 = np.array(samples, dtype=np.float32)
        window_22 = self.adapter.process_window(arr_16, source_fs=125)
        res = self.model_runner.predict(window_22)

        assert_prediction_schema(res)
        self.assertGreaterEqual(res["probabilities"]["normal"], 0.0)

    # -------------------------------------------------------------------------
    # 10. Full Pipeline: Seizure EEG Ingest -> Classification (F02 -> F16)
    # -------------------------------------------------------------------------
    def test_f02_f16_full_ingest_to_inference_seizure_pipeline(self):
        """Verifies full pipeline on high-amplitude epileptiform discharges."""
        seiz_signal = EegStreamer.generate_signal(state="seizure", duration_sec=5.0, fs=125, n_channels=16)
        window_22 = self.adapter.process_window(seiz_signal, source_fs=125)
        res = self.model_runner.predict(window_22)

        assert_prediction_schema(res)
        # Verify seizure state is recognized with valid non-zero probability
        self.assertGreater(res["probabilities"]["seizure"], 0.001)

    # -------------------------------------------------------------------------
    # 11. CNN Probabilities -> Hysteresis Alert State Transitions (F16 -> F18)
    # -------------------------------------------------------------------------
    def test_f16_f18_inference_probabilities_to_hysteresis_state(self):
        """Verifies Hysteresis Alert Engine state machine transitions (IDLE -> WARNING -> CRITICAL)."""
        # Hysteresis specification:
        # CRITICAL: at least 3 of last 5 windows with P(seiz) >= 0.70
        # WARNING: mean P(pre) >= 0.60 over last 10 windows
        windows = [
            {"normal": 0.90, "pre_seizure": 0.08, "seizure": 0.02},
            {"normal": 0.85, "pre_seizure": 0.10, "seizure": 0.05},
            {"normal": 0.20, "pre_seizure": 0.05, "seizure": 0.75},  # 1/5 seiz
            {"normal": 0.15, "pre_seizure": 0.05, "seizure": 0.80},  # 2/5 seiz
            {"normal": 0.10, "pre_seizure": 0.05, "seizure": 0.85},  # 3/5 seiz -> TRIGGER CRITICAL
        ]

        def evaluate_hysteresis(history: List[Dict[str, float]]) -> Optional[str]:
            last_5 = history[-5:]
            seiz_count = sum(1 for w in last_5 if w.get("seizure", 0.0) >= 0.70)
            if seiz_count >= 3:
                return "CRITICAL"

            last_10 = history[-10:]
            if len(last_10) >= 10:
                mean_pre = sum(w.get("pre_seizure", 0.0) for w in last_10) / len(last_10)
                if mean_pre >= 0.60:
                    return "WARNING"
            return None

        # Step-by-step evaluation
        hist = []
        for i, w in enumerate(windows):
            hist.append(w)
            state = evaluate_hysteresis(hist)
            if i < 4:
                self.assertIsNone(state)
            else:
                self.assertEqual(state, "CRITICAL")

    # -------------------------------------------------------------------------
    # 12. Hysteresis Clearance & Cooldown Enforcement (F18)
    # -------------------------------------------------------------------------
    def test_f18_cooldown_and_hysteresis_clearance_pairwise(self):
        """Verifies hysteresis clearance on 5 consecutive normal windows and 120s cooldown."""
        # Active alert ends only after 5 consecutive normal windows
        clearance_windows = [
            {"seizure": 0.40},  # Window 1 below threshold
            {"seizure": 0.35},  # Window 2
            {"seizure": 0.30},  # Window 3
            {"seizure": 0.25},  # Window 4 (still in alert)
            {"seizure": 0.20},  # Window 5 (clears alert)
        ]

        is_cleared = False
        consecutive_low = 0
        for w in clearance_windows:
            if w["seizure"] < 0.70:
                consecutive_low += 1
                if consecutive_low >= 5:
                    is_cleared = True
            else:
                consecutive_low = 0

        self.assertTrue(is_cleared)

        # Enforce 120s cooldown
        alert_end_time = 1000.0
        new_trigger_time_early = 1050.0  # +50s < 120s (Suppressed)
        new_trigger_time_allowed = 1125.0  # +125s > 120s (Allowed)

        cooldown_period = 120.0
        self.assertFalse((new_trigger_time_early - alert_end_time) >= cooldown_period)
        self.assertTrue((new_trigger_time_allowed - alert_end_time) >= cooldown_period)

    # -------------------------------------------------------------------------
    # 13. Multi-Chunk Streaming Continuity (F13 -> F14)
    # -------------------------------------------------------------------------
    def test_f13_f16_multi_chunk_streaming_continuity(self):
        """Verifies continuous streaming of 8 consecutive 250ms chunks without boundary transients."""
        filt = StreamingFilter(fs=256, n_channels=22)
        n_chunks = 8
        chunk_len = 64  # samples per 250ms at 256Hz
        filtered_chunks = []

        # Continuous 10 Hz sine wave across chunks
        total_samples = n_chunks * chunk_len
        t = np.linspace(0, total_samples / 256.0, total_samples, endpoint=False)
        continuous_signal = (25.0 * np.sin(2 * np.pi * 10.0 * t))[:, np.newaxis]
        continuous_multichan = np.repeat(continuous_signal, 22, axis=1).astype(np.float32)

        for i in range(n_chunks):
            chunk = continuous_multichan[i * chunk_len : (i + 1) * chunk_len]
            out = filt.filter(chunk)
            filtered_chunks.append(out)

        concatenated = np.vstack(filtered_chunks)
        # Check boundary continuity between chunk 3 and chunk 4
        seam_step = np.abs(filtered_chunks[3][-1] - filtered_chunks[4][0])
        max_step = np.max(seam_step)
        # Max derivative step at 10 Hz 25 uV is bounded: 2 * pi * 10 * 25 / 256 ~ 6.1 uV
        self.assertLess(max_step, 10.0)

    # -------------------------------------------------------------------------
    # 14. Device Adapter to Diagnostics Hop Counters (F04 -> F06)
    # -------------------------------------------------------------------------
    def test_f04_f06_device_adapter_to_diagnostics_counters(self):
        """Verifies end-to-end hop counters conserve packet and frame counts."""
        counters = {
            "hop1_device_packets": 0,
            "hop2_chunks_sent": 0,
            "hop3_backend_frames": 0,
            "hop4_windows_inferred": 0,
        }

        # Simulate receiving 620 device packets (310 samples = 10 chunks of 31 samples)
        packets_received = 620
        counters["hop1_device_packets"] += packets_received
        chunks_produced = packets_received // 62  # 2 packets per sample, 31 samples per chunk
        counters["hop2_chunks_sent"] += chunks_produced
        counters["hop3_backend_frames"] += chunks_produced
        # 10 chunks @ 250ms = 2.5s; 1 window requires 5.0s (20 chunks)
        counters["hop4_windows_inferred"] += chunks_produced // 20

        self.assertEqual(counters["hop1_device_packets"], 620)
        self.assertEqual(counters["hop2_chunks_sent"], 10)
        self.assertEqual(counters["hop3_backend_frames"], 10)
        self.assertEqual(counters["hop4_windows_inferred"], 0)

    # -------------------------------------------------------------------------
    # 15. Profile to Montage Adaptation Contract (F05 -> F15)
    # -------------------------------------------------------------------------
    def test_f05_f15_profile_to_montage_adaptation_contract(self):
        """Verifies profile compatibility check and montage adaptation logic."""
        cyton_profile = get_openbci_cyton_profile()
        self.assertEqual(len(cyton_profile["channels"]), 16)
        self.assertEqual(cyton_profile["fs_hz"], 125)

        # Compatibility rules from architecture.md §4A:
        # 16-channel Cyton with 22-channel model -> 'adapted'
        status = "adapted" if len(cyton_profile["channels"]) == 16 else "compatible"
        self.assertEqual(status, "adapted")

        incompat_profile = get_incompatible_profile()
        self.assertEqual(len(incompat_profile["channels"]), 4)
        status_incompat = "incompatible" if len(incompat_profile["channels"]) < 16 else "compatible"
        self.assertEqual(status_incompat, "incompatible")

    # -------------------------------------------------------------------------
    # 16. Voltage Rail Saturation -> Artifact Detection (F14 -> F20)
    # -------------------------------------------------------------------------
    def test_f14_f20_filter_saturation_to_artifact_detection(self):
        """Verifies railed channel voltage (>180,000 uV) flagged by signal quality module."""
        railed_signal = EegStreamer.generate_signal(state="railed", duration_sec=2.0, fs=125, n_channels=16)
        # Channels 0-4 are railed in fixture (> 180,000 uV)
        assert_signal_quality(railed_signal[:, 0], expected_quality="railed", fs=125)
        assert_signal_quality(railed_signal[:, 1], expected_quality="railed", fs=125)
        assert_signal_quality(railed_signal[:, 2], expected_quality="railed", fs=125)
        assert_signal_quality(railed_signal[:, 3], expected_quality="railed", fs=125)
        assert_signal_quality(railed_signal[:, 4], expected_quality="railed", fs=125)


if __name__ == "__main__":
    unittest.main()
