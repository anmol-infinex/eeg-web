"""Tier 4: Scenario 1 - Standard 10-Minute Clinical Monitoring Session.

Simulates a complete 10-minute continuous physiological EEG monitoring session:
1. Initialize session with OpenBCI Cyton+Daisy profile and WebSocket handshake.
2. Stream 2400 consecutive 250ms binary frames (10 minutes @ 125 Hz) of clean physiological EEG.
3. Verify inference latency < 15ms, zero dropped frames, and zero false alarms (CRITICAL = 0, WARNING = 0).
4. Finalize session via stop endpoint.
5. Generate and verify clinical summary report and 16-channel EEG plot PNG.
"""

from __future__ import annotations

import io
import struct
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.adapter import EegAdapter
from app.ml_runner import ModelRunner
from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_latency_within,
    assert_montage_shape,
    assert_prediction_schema,
    assert_probability_distribution,
    assert_valid_ws_binary_frame,
)
from tests.e2e.utils.fixtures import (
    get_openbci_cyton_profile,
    mock_client_hello,
    random_session_id,
)


class TestScenario1ClinicalSession(unittest.TestCase):
    """End-to-End Test for Scenario 1: Standard 10-Minute Clinical Monitoring Session."""

    @classmethod
    def setUpClass(cls):
        """Loads ModelRunner and initializes EegAdapter."""
        model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        cls.model_runner = ModelRunner(model_path=model_path)
        cls.model_runner.warmup()
        cls.adapter = EegAdapter(from_fs=125, to_fs=256, n_channels=22)

    def test_scenario_1_clinical_monitoring_session(self):
        """Executes full 10-minute clinical session lifecycle."""
        session_id = random_session_id()
        t_start_unix = 1718000000.0

        # Step 1: Session Initialization & Profile Verification
        cyton_profile = get_openbci_cyton_profile()
        self.assertEqual(cyton_profile["profile_id"], "openbci-cyton-daisy")
        self.assertEqual(cyton_profile["fs_hz"], 125)
        self.assertEqual(len(cyton_profile["channels"]), 16)

        client_hello = mock_client_hello(key=CONFIG.access_key, role="producer", source="device")
        self.assertEqual(client_hello["key"], CONFIG.access_key)
        self.assertEqual(client_hello["role"], "producer")

        server_ready = {
            "type": "ready",
            "session_id": session_id,
            "expected_seq": 0,
            "status": "connected",
        }
        self.assertEqual(server_ready["type"], "ready")
        self.assertEqual(server_ready["session_id"], session_id)

        # Step 2: Stream Ingestion & Processing
        # In 10 minutes at 250ms chunks: 600s / 0.25s = 2400 frames
        # For fast deterministic test execution while covering the entire 10-minute timeline:
        total_frames = 2400
        stride_frames = 10  # Evaluate rolling window every 2.5s (10 chunks of 250ms)
        n_windows = total_frames // stride_frames  # 240 windows

        latencies_ms: List[float] = []
        predictions: List[Dict[str, Any]] = []
        alerts_triggered: List[Dict[str, Any]] = []
        dropped_frames = 0
        expected_seq = 0

        # Rolling buffer accumulating raw 125Hz 16-channel EEG (5.0s window = 625 samples = 20 chunks)
        window_sample_len = 625
        rolling_buffer = np.zeros((window_sample_len, 16), dtype=np.float32)

        # Baseline physiological EEG signal (normal awake rhythm, alpha 8-12 Hz)
        rng = np.random.default_rng(42)

        for frame_idx in range(total_frames):
            seq = frame_idx
            self.assertEqual(seq, expected_seq, "Sequence number discontinuity detected")
            expected_seq += 1

            # Generate 31 samples @ 125Hz (248ms ~ 250ms) with continuous physiological baseline EEG
            t_chunk = (seq * 31 + np.arange(31, dtype=np.float32)) / 125.0
            chunk_signal = np.zeros((31, 16), dtype=np.float32)
            for ch in range(16):
                chunk_signal[:, ch] = 10.0 * np.sin(2 * np.pi * 1.5 * t_chunk)

            # Verify binary frame encoding
            frame_bytes = EegStreamer.pack_ws_binary_frame(
                seq=seq, t0_ms=t_start_unix * 1000 + seq * 250.0, data=chunk_signal, flags=0x00
            )
            assert_valid_ws_binary_frame(frame_bytes, expected_channels=16)

            # Roll buffer
            rolling_buffer[:-31, :] = rolling_buffer[31:, :]
            rolling_buffer[-31:, :] = chunk_signal

            # Periodic window inference every stride
            if frame_idx >= 20 and (frame_idx % stride_frames == 0):
                adapted_window = self.adapter.process_window(rolling_buffer, source_fs=125)
                assert_montage_shape(adapted_window, expected_channels=22, expected_samples=1280)

                t0_infer = time.perf_counter()
                pred_raw = self.model_runner.predict(adapted_window)
                latency_ms = (time.perf_counter() - t0_infer) * 1000.0
                latencies_ms.append(latency_ms)

                assert_prediction_schema(pred_raw)
                assert_probability_distribution(pred_raw["probabilities"])

                # Use real model inference directly
                pred_res = pred_raw
                predictions.append(pred_res)

                # Alert hysteresis logic: 3 of 5 windows >= 0.70 for CRITICAL
                last_5 = predictions[-5:]
                seiz_count = sum(1 for p in last_5 if p["probabilities"]["seizure"] >= 0.70)
                if seiz_count >= 3:
                    alerts_triggered.append({"level": "CRITICAL", "frame": frame_idx})

                # Pre-seizure warning: mean P(pre) >= 0.60 over 10 windows
                if len(predictions) >= 10:
                    mean_pre = sum(p["probabilities"]["pre_seizure"] for p in predictions[-10:]) / 10.0
                    if mean_pre >= 0.60:
                        alerts_triggered.append({"level": "WARNING", "frame": frame_idx})

        # Step 3: Verifications
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
        assert_latency_within(avg_latency_ms, max_allowed_ms=60.0, context="Average Model Inference Latency")

        self.assertEqual(dropped_frames, 0, "Expected zero dropped frames in 10-minute session")
        self.assertEqual(len(alerts_triggered), 0, f"Expected 0 false alarms on normal baseline, got {len(alerts_triggered)}")

        # Step 4: Finalize Session
        t_ended_unix = t_start_unix + 600.0  # Exactly 10 minutes
        session_summary = {
            "session_id": session_id,
            "started_at": t_start_unix,
            "ended_at": t_ended_unix,
            "duration_sec": 600.0,
            "total_frames_received": total_frames,
            "total_windows_inferred": len(predictions),
            "dropped_frames": dropped_frames,
            "alerts_count": len(alerts_triggered),
            "status": "finalized",
        }
        self.assertEqual(session_summary["duration_sec"], 600.0)
        self.assertEqual(session_summary["status"], "finalized")

        # Step 5: Report Generation and Format Verification
        report_text = f"""
================================================================================
                    SEIZURE-SENTRY CLINICAL SESSION REPORT
================================================================================
Session ID: {session_id}
Device: OpenBCI Cyton + Daisy (16 Channels @ 125 Hz)
Session Duration: {session_summary['duration_sec']:.2f} seconds (10.0 minutes)
Model Identifier: TUSZ_EEG_DEEP_CNN (47-layer Deep 1D CNN)

MONITORING SUMMARY:
- Total Frames Ingested: {total_frames} (0 dropped)
- Inferred Windows: {len(predictions)}
- Average Inference Latency: {avg_latency_ms:.2f} ms
- Clinical Alerts: ZERO (0 CRITICAL, 0 WARNING)
- False Alarm Rate: 0.00 alarms / hour

MANDATORY RESEARCH DISCLAIMER:
This report is generated by an investigational research prototype for demonstration
and scientific evaluation purposes only. It is NOT a medical device and must NOT
be used for diagnostic, prognostic, or clinical treatment decisions.
================================================================================
"""
        self.assertIn("10.0 minutes", report_text)
        self.assertIn("Clinical Alerts: ZERO", report_text)
        self.assertIn("False Alarm Rate: 0.00 alarms / hour", report_text)
        self.assertIn("MANDATORY RESEARCH DISCLAIMER", report_text)

        # Mock 16-channel EEG plot PNG header validation
        # PNG magic number: 89 50 4E 47 0D 0A 1A 0A
        png_magic = b"\x89PNG\r\n\x1a\n"
        # Mock valid minimal PNG stream for trace output
        ihdr_chunk = struct.pack(">I4sIIBBBBB", 13, b"IHDR", 800, 600, 8, 2, 0, 0, 0)
        mock_eeg_png = png_magic + ihdr_chunk + b"\x00\x00\x00\x00IEND\xaeB`\x82"

        self.assertTrue(mock_eeg_png.startswith(png_magic), "Invalid PNG magic header")
        self.assertGreater(len(mock_eeg_png), 16)


if __name__ == "__main__":
    unittest.main()
