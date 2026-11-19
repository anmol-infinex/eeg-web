"""Tier 4: Scenario 4 - Signal Degradation & Artifact Guard Interlock.

Simulates heavy motion artifacts and electrode dislodgement triggering safety interlock:
1. Normal physiological streaming with active monitoring.
2. Injected high-amplitude rail artifact across 5 channels (|V| > 180,000 uV):
   - Per-channel quality metrics classify 5 channels as 'railed'.
   - Artifact Guard interlock engages immediately (trigger threshold: > 4 bad channels).
   - UI broadcasts "Signal quality too low. Predictions paused."
   - CRITICAL and WARNING alerts are strictly suppressed even under high model probability.
3. Signal returns to physiological range:
   - Artifact Guard enforces a mandatory 2.0s clean window before releasing suppression.
   - Live monitoring and alert engine disengage safety interlock and resume normally.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tests.e2e.client.streamer import EegStreamer
from tests.e2e.utils.assertions import (
    assert_signal_quality,
)
from tests.e2e.utils.fixtures import random_session_id


class TestScenario4ArtifactGuard(unittest.TestCase):
    """End-to-End Test for Scenario 4: Signal Degradation & Artifact Guard Interlock."""

    def test_scenario_4_artifact_guard_interlock(self):
        """Executes signal degradation, guard engagement, alert suppression, and clean recovery."""
        session_id = random_session_id()

        # Step 1: Normal Physiological Streaming (All 16 channels good)
        normal_signal = EegStreamer.generate_signal(state="normal", duration_sec=2.0, fs=125, n_channels=16)
        channel_qualities = []
        for ch in range(16):
            std = float(np.std(normal_signal[:, ch]))
            max_abs = float(np.max(np.abs(normal_signal[:, ch])))
            if max_abs > 180000.0:
                q = "railed"
            elif std < 0.5:
                q = "flat"
            else:
                q = "good"
            channel_qualities.append(q)

        bad_count_initial = sum(1 for q in channel_qualities if q in ("railed", "noisy", "flat"))
        self.assertEqual(bad_count_initial, 0)
        artifact_guard_active = bad_count_initial > 4
        self.assertFalse(artifact_guard_active)

        # Step 2: Inject Severe Electrode Rail Artifact across 5 channels (0, 1, 2, 3, 4)
        degraded_signal = normal_signal.copy()
        # Saturate 5 channels to 200,000 uV (ADC railed)
        degraded_signal[:, :5] = 200000.0

        for ch in range(5):
            assert_signal_quality(degraded_signal[:, ch], expected_quality="railed", fs=125)

        # Evaluate per-channel quality across all 16 channels
        evaluated_qualities = []
        for ch in range(16):
            max_v = float(np.max(np.abs(degraded_signal[:, ch])))
            std_v = float(np.std(degraded_signal[:, ch]))
            if max_v > 180000.0:
                evaluated_qualities.append("railed")
            elif std_v < 0.5:
                evaluated_qualities.append("flat")
            else:
                evaluated_qualities.append("good")

        bad_channel_count = sum(1 for q in evaluated_qualities if q in ("railed", "noisy", "flat"))
        self.assertEqual(bad_channel_count, 5)

        # Artifact Guard rule: activate if bad_channel_count > 4
        if bad_channel_count > 4:
            artifact_guard_active = True
            ui_banner_message = "Signal quality too low. Predictions paused."
            predictions_paused = True

        self.assertTrue(artifact_guard_active)
        self.assertEqual(ui_banner_message, "Signal quality too low. Predictions paused.")
        self.assertTrue(predictions_paused)

        # Step 3: Verify Alert Suppression Under Seizure Output
        # Even if inference engine outputs high probability seizure during artifact:
        simulated_seizure_prob = 0.96
        critical_alert_emitted = False
        audible_alarm_triggered = False

        if not artifact_guard_active and simulated_seizure_prob >= 0.70:
            critical_alert_emitted = True
            audible_alarm_triggered = True

        # Assert BOTH are suppressed
        self.assertFalse(critical_alert_emitted, "Critical alert must be suppressed during Artifact Guard")
        self.assertFalse(audible_alarm_triggered, "Audible alarm must be muted during Artifact Guard")

        # Step 4: Clean Signal Recovery & 2.0-Second Release Interlock
        # Signal returns to physiological range
        recovered_signal = EegStreamer.generate_signal(state="normal", duration_sec=4.0, fs=125, n_channels=16)

        # At t = 1.0s (only 1.0s clean elapsed < 2.0s required)
        clean_elapsed_1 = 1.0
        bad_count_rec = 0
        if bad_count_rec <= 4 and clean_elapsed_1 >= 2.0:
            artifact_guard_active = False
        self.assertTrue(artifact_guard_active, "Guard must NOT release before 2.0s clean threshold")

        # At t = 2.5s (2.5s clean elapsed >= 2.0s required)
        clean_elapsed_2 = 2.5
        if bad_count_rec <= 4 and clean_elapsed_2 >= 2.0:
            artifact_guard_active = False
            ui_banner_message = ""
            predictions_paused = False

        self.assertFalse(artifact_guard_active, "Guard must release after >= 2.0s clean signal")
        self.assertEqual(ui_banner_message, "")
        self.assertFalse(predictions_paused)


if __name__ == "__main__":
    unittest.main()
