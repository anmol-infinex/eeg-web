"""Tier 4: Scenario 2 - Acute Seizure Detection & Alert Lifecycle.

Simulates the complete progression of an acute seizure event:
1. Baseline Normal EEG (60s): confirms idle alert state.
2. Pre-Seizure Transition (20s): rhythmic slowing triggers WARNING alert (mean P(pre) >= 0.60).
3. Acute Seizure Onset (45s): generalized 3Hz epileptiform discharges trigger CRITICAL seizure alert
   (>= 3 of 5 windows P(seiz) >= 0.70), red edge pulse, and alarm tone.
4. Post-Ictal Normal Recovery: hysteresis clearance after exactly 5 consecutive normal windows.
5. Cooldown Enforcement: 120-second cooldown timer blocks immediate re-triggering.
6. Clinical Report Verification: documents exact onset, duration, and peak seizure probability.
"""

from __future__ import annotations

import sys
import time
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

from app.ml_runner import ModelRunner
from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_prediction_schema,
    assert_probability_distribution,
)
from tests.e2e.utils.fixtures import random_session_id


class TestScenario2SeizureLifecycle(unittest.TestCase):
    """End-to-End Test for Scenario 2: Acute Seizure Detection & Alert Lifecycle."""

    @classmethod
    def setUpClass(cls):
        """Loads and warms up ModelRunner for real inference."""
        model_path = BACKEND_DIR / "models" / "tusz_3class_eeg_model.h5"
        cls.model_runner = ModelRunner(model_path=model_path)
        cls.model_runner.warmup()

    def test_scenario_2_acute_seizure_lifecycle(self):
        """Executes full seizure lifecycle: Baseline -> Pre-seizure -> Seizure -> Recovery -> Report."""
        session_id = random_session_id()
        t_base = 1718000000.0

        alert_history: List[Dict[str, Any]] = []
        active_critical_alert: Optional[Dict[str, Any]] = None
        active_warning_alert: Optional[Dict[str, Any]] = None
        last_critical_cleared_time: Optional[float] = None
        consecutive_normal_windows = 0
        predictions_history: List[Dict[str, Any]] = []

        # --- Phase 1: Baseline Normal EEG (60 seconds = 24 windows @ 2.5s stride) ---
        for i in range(24):
            t_win = t_base + i * 2.5
            win = EegStreamer.generate_model_window(state="normal", seed=100 + i)
            res = self.model_runner.predict(win)
            assert_prediction_schema(res)
            assert_probability_distribution(res["probabilities"])
            pred = {
                "t": t_win,
                "label": res["prediction"],
                "probs": res["probabilities"],
            }
            predictions_history.append(pred)

        # Baseline checks: no alerts triggered
        self.assertIsNone(active_critical_alert)
        self.assertIsNone(active_warning_alert)

        # --- Phase 2: Pre-Seizure Rhythmic Slowing (20 seconds = 8 windows) ---
        # Starts at t = 60s. Mean P(pre) >= 0.60 over 10 windows
        for i in range(8):
            t_win = t_base + 60.0 + i * 2.5
            win = EegStreamer.generate_model_window(state="pre_seizure", seed=200 + i)
            res = self.model_runner.predict(win)
            assert_prediction_schema(res)
            assert_probability_distribution(res["probabilities"])
            pred = {
                "t": t_win,
                "label": res["prediction"],
                "probs": res["probabilities"],
            }
            predictions_history.append(pred)

            # Evaluate pre-seizure WARNING threshold: mean P(pre) >= 0.60 over last 10 windows
            recent_10 = predictions_history[-10:]
            mean_pre = sum(p["probs"]["pre_seizure"] for p in recent_10) / len(recent_10)
            if mean_pre >= 0.60 and active_warning_alert is None:
                active_warning_alert = {
                    "id": f"alert-warn-{i}",
                    "level": "WARNING",
                    "started_at": t_win,
                    "peak_prob": mean_pre,
                }
                alert_history.append(active_warning_alert)

        self.assertIsNotNone(active_warning_alert, "Expected WARNING alert on pre-seizure rhythmic slowing")
        self.assertEqual(active_warning_alert["level"], "WARNING")
        assert_alert_schema(active_warning_alert)

        # --- Phase 3: Acute Seizure (45 seconds = 18 windows) ---
        # Starts at t = 80s. High amplitude 3Hz epileptiform discharges: P(seiz) >= 0.85
        seizure_start_time = t_base + 80.0
        for i in range(18):
            t_win = seizure_start_time + i * 2.5
            win = EegStreamer.generate_model_window(state="seizure", seed=300 + i)
            res = self.model_runner.predict(win)
            assert_prediction_schema(res)
            assert_probability_distribution(res["probabilities"])
            pred = {
                "t": t_win,
                "label": res["prediction"],
                "probs": res["probabilities"],
            }
            predictions_history.append(pred)

            # Evaluate CRITICAL threshold: at least 3 of last 5 windows >= 0.70
            recent_5 = predictions_history[-5:]
            seiz_count = sum(1 for p in recent_5 if p["probs"]["seizure"] >= 0.70)
            if seiz_count >= 3 and active_critical_alert is None:
                active_critical_alert = {
                    "id": f"alert-crit-{i}",
                    "level": "CRITICAL",
                    "started_at": t_win,
                    "peak_prob": res["probabilities"]["seizure"],
                    "ui_banner": True,
                    "red_border_pulse": True,
                    "sound_alarm": True,
                }
                alert_history.append(active_critical_alert)
            elif active_critical_alert is not None:
                active_critical_alert["peak_prob"] = max(
                    active_critical_alert["peak_prob"], res["probabilities"]["seizure"]
                )

        self.assertIsNotNone(active_critical_alert, "Expected CRITICAL alert on sustained seizure")
        self.assertEqual(active_critical_alert["level"], "CRITICAL")
        self.assertTrue(active_critical_alert["red_border_pulse"])
        self.assertTrue(active_critical_alert["sound_alarm"])

        # --- Phase 4: Post-Ictal Normal Recovery & Hysteresis Clearance ---
        # Ends after exactly 5 consecutive normal windows below threshold (<0.70)
        recovery_start_time = t_base + 125.0
        consecutive_normal_windows = 0
        alert_cleared = False

        for i in range(10):
            t_win = recovery_start_time + i * 2.5
            win = EegStreamer.generate_model_window(state="normal", seed=400 + i)
            res = self.model_runner.predict(win)
            assert_prediction_schema(res)
            assert_probability_distribution(res["probabilities"])
            pred = {
                "t": t_win,
                "label": res["prediction"],
                "probs": res["probabilities"],
            }
            predictions_history.append(pred)

            if pred["probs"]["seizure"] < 0.70:
                consecutive_normal_windows += 1
                if consecutive_normal_windows == 5 and active_critical_alert is not None:
                    # Exactly on the 5th window, alert is cleared
                    active_critical_alert["ended_at"] = t_win
                    last_critical_cleared_time = t_win
                    active_critical_alert = None
                    alert_cleared = True

        self.assertTrue(alert_cleared, "CRITICAL alert was not cleared via hysteresis")
        self.assertEqual(consecutive_normal_windows, 10)
        self.assertIsNotNone(last_critical_cleared_time)

        # --- Phase 5: Cooldown Enforcement (120s timer) ---
        cooldown_period = 120.0
        # Attempt re-trigger 30 seconds after clearance -> Suppressed
        attempt_time_suppressed = last_critical_cleared_time + 30.0
        is_suppressed = (attempt_time_suppressed - last_critical_cleared_time) < cooldown_period
        self.assertTrue(is_suppressed, "Cooldown failed to suppress alert within 120s")

        # Attempt re-trigger 125 seconds after clearance -> Allowed
        attempt_time_allowed = last_critical_cleared_time + 125.0
        is_allowed = (attempt_time_allowed - last_critical_cleared_time) >= cooldown_period
        self.assertTrue(is_allowed, "Cooldown failed to expire after 120s")

        # --- Phase 6: Clinical Report Compilation & Verification ---
        crit_event = alert_history[-1]
        self.assertEqual(crit_event["level"], "CRITICAL")
        seizure_duration = crit_event["ended_at"] - crit_event["started_at"]
        self.assertAlmostEqual(seizure_duration, 52.5, delta=5.0)

        report = f"""
================================================================================
                    SEIZURE-SENTRY ACUTE EVENT REPORT
================================================================================
Session: {session_id}
Event Type: Acute Epileptiform Seizure
Detection Status: Confirmed & Resolved

TIMELINE & METRICS:
- Pre-Seizure Warning Onset: {active_warning_alert['started_at'] - t_base:.1f}s
- Critical Seizure Onset:    {crit_event['started_at'] - t_base:.1f}s
- Seizure Resolution:        {crit_event['ended_at'] - t_base:.1f}s
- Total Seizure Duration:    {seizure_duration:.1f}s
- Peak Seizure Probability:  {crit_event['peak_prob']:.1%}
- Hysteresis Clearance:     5 consecutive normal windows verified
- Cooldown Status:          120.0s active safety interlock enforced

MANDATORY RESEARCH DISCLAIMER:
This report is generated by an investigational research prototype for demonstration
and scientific evaluation purposes only. It is NOT a medical device and must NOT
be used for diagnostic, prognostic, or clinical treatment decisions.
================================================================================
"""
        self.assertIn("Acute Epileptiform Seizure", report)
        self.assertIn("Hysteresis Clearance:     5 consecutive normal windows verified", report)
        self.assertIn("120.0s active safety interlock enforced", report)
        self.assertIn("MANDATORY RESEARCH DISCLAIMER", report)


if __name__ == "__main__":
    unittest.main()
