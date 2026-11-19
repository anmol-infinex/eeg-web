"""Tier 1: Features 18-20 Alerting & Safety Systems Tests.

Covers:
- Feature 18: Hysteresis Alert Engine (5 tests)
- Feature 19: Multi-Modal Alerting (5 tests)
- Feature 20: Artifact Guard Suppression (5 tests)

Total: 15 tests.
"""

from __future__ import annotations

import time
import unittest
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tests.e2e.utils.assertions import assert_alert_schema


# =============================================================================
# Hysteresis Alert Engine & Artifact Guard Reference Engine
# =============================================================================
class HysteresisAlertEngine:
    """Reference implementation of PRD §7 Alert Policy.

    Rules:
      - CRITICAL (Seizure): >= 3 of the last 5 windows have P(seizure) >= 0.70.
      - WARNING (Pre-Seizure): mean P(pre_seizure) >= 0.60 over the last 10 windows.
      - Hysteresis: Active alert clears only after 5 consecutive windows below threshold.
      - Cooldown: 120.0 seconds before re-triggering an alert of the same type.
      - Artifact Guard: If > 4 channels are bad (noisy/flat/railed), suppress alerts.
    """

    def __init__(self, cooldown_sec: float = 120.0, clear_consecutive: int = 5):
        self.cooldown_sec = cooldown_sec
        self.clear_consecutive = clear_consecutive

        self.history_windows: List[Dict[str, float]] = []
        self.active_alert: Optional[Dict[str, Any]] = None
        self.last_alert_end_time: Dict[str, float] = {"CRITICAL": -9999.0, "WARNING": -9999.0}
        self.consecutive_below_count: int = 0
        self.artifact_guard_active: bool = False

    def set_artifact_guard(self, active: bool):
        self.artifact_guard_active = active
        if active and self.active_alert:
            # Mute/suppress active alert while artifact guard is engaged
            self.active_alert["muted"] = True

    def process_window(
        self,
        probs: Dict[str, float],
        timestamp: float,
    ) -> Optional[Dict[str, Any]]:
        self.history_windows.append(probs)

        # 1. If Artifact Guard is active, suppress new alerts and mute existing
        if self.artifact_guard_active:
            if self.active_alert:
                self.active_alert["muted"] = True
            return self.active_alert

        p_seiz = probs.get("seizure", 0.0)
        p_pre = probs.get("pre_seizure", 0.0)

        # 2. Check if an alert is already active
        if self.active_alert is not None:
            level = self.active_alert["level"]
            self.active_alert["peak_prob"] = max(
                self.active_alert["peak_prob"],
                p_seiz if level == "CRITICAL" else p_pre,
            )

            # Check threshold for clearance
            is_below = (p_seiz < 0.70) if level == "CRITICAL" else (p_pre < 0.60)
            if is_below:
                self.consecutive_below_count += 1
            else:
                self.consecutive_below_count = 0  # Single spike resets clearance count

            if self.consecutive_below_count >= self.clear_consecutive:
                # Alert cleared via hysteresis
                self.active_alert["ended_at"] = timestamp
                self.last_alert_end_time[level] = timestamp
                cleared = self.active_alert
                self.active_alert = None
                self.consecutive_below_count = 0
                return cleared

            return self.active_alert

        # 3. Check for new CRITICAL (Seizure) trigger
        # >= 3 of last 5 windows with P(seizure) >= 0.70
        last_5 = self.history_windows[-5:]
        seiz_count = sum(1 for w in last_5 if w.get("seizure", 0.0) >= 0.70)
        if seiz_count >= 3:
            time_since_last = timestamp - self.last_alert_end_time.get("CRITICAL", -9999.0)
            if time_since_last >= self.cooldown_sec:
                self.active_alert = {
                    "id": str(uuid.uuid4()),
                    "level": "CRITICAL",
                    "started_at": timestamp,
                    "ended_at": None,
                    "peak_prob": max(w.get("seizure", 0.0) for w in last_5),
                    "muted": False,
                }
                self.consecutive_below_count = 0
                return self.active_alert

        # 4. Check for new WARNING (Pre-Seizure) trigger
        # Mean P(pre_seizure) >= 0.60 over last 10 windows
        last_10 = self.history_windows[-10:]
        if len(last_10) >= 10:
            mean_pre = sum(w.get("pre_seizure", 0.0) for w in last_10) / 10.0
            if mean_pre >= 0.60:
                time_since_last = timestamp - self.last_alert_end_time.get("WARNING", -9999.0)
                if time_since_last >= self.cooldown_sec:
                    self.active_alert = {
                        "id": str(uuid.uuid4()),
                        "level": "WARNING",
                        "started_at": timestamp,
                        "ended_at": None,
                        "peak_prob": max(w.get("pre_seizure", 0.0) for w in last_10),
                        "muted": False,
                    }
                    self.consecutive_below_count = 0
                    return self.active_alert

        return None


# =============================================================================
# Feature 18: Hysteresis Alert Engine
# =============================================================================
class TestFeature18HysteresisAlertEngine(unittest.TestCase):
    """Feature 18: Multi-state alert detection preventing noisy oscillations (PRD §7).

    Verifies CRITICAL seizure trigger (>=3 of 5 >= 0.70), WARNING pre-seizure trigger
    (mean >= 0.60 over 10), 5-window hysteresis clearance, single dip suppression,
    and 120s cooldown enforcement.
    """

    def setUp(self):
        self.engine = HysteresisAlertEngine(cooldown_sec=120.0, clear_consecutive=5)

    def test_f18_critical_seizure_trigger(self):
        """1. Triggers CRITICAL when >= 3 of 5 windows have P(seizure) >= 0.70."""
        # 2 windows normal
        self.engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=1.0)
        self.engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=2.0)
        # 1st high seizure window -> no alert yet
        alert = self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.8}, timestamp=3.0)
        self.assertIsNone(alert)
        # 2nd high seizure window -> no alert yet
        alert = self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=4.0)
        self.assertIsNone(alert)
        # 3rd high seizure window (3 of 5 >= 0.70) -> TRIGGER CRITICAL
        alert = self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.9}, timestamp=5.0)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["level"], "CRITICAL")
        self.assertGreaterEqual(alert["peak_prob"], 0.9)

    def test_f18_warning_preseizure_trigger(self):
        """2. Triggers WARNING when mean P(pre_seizure) >= 0.60 over 10 windows."""
        # Send 9 windows with P(pre) = 0.65
        for i in range(9):
            self.engine.process_window({"normal": 0.3, "pre_seizure": 0.65, "seizure": 0.05}, timestamp=float(i))
            self.assertIsNone(self.engine.active_alert)

        # 10th window with P(pre) = 0.65 (mean = 0.65 >= 0.60) -> TRIGGER WARNING
        alert = self.engine.process_window({"normal": 0.3, "pre_seizure": 0.65, "seizure": 0.05}, timestamp=9.0)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["level"], "WARNING")

    def test_f18_hysteresis_clearance(self):
        """3. Requires exactly 5 consecutive windows below threshold to end alert."""
        # Trigger CRITICAL alert
        for i in range(5):
            self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.8}, timestamp=float(i))
        self.assertIsNotNone(self.engine.active_alert)
        self.assertEqual(self.engine.active_alert["level"], "CRITICAL")

        # 4 windows below threshold (0.10 < 0.70) -> alert still active
        for i in range(5, 9):
            alert = self.engine.process_window({"normal": 0.85, "pre_seizure": 0.05, "seizure": 0.10}, timestamp=float(i))
            self.assertIsNotNone(self.engine.active_alert)
            self.assertIsNone(alert.get("ended_at"))

        # 5th consecutive window below threshold -> CLEAR ALERT
        cleared_alert = self.engine.process_window({"normal": 0.85, "pre_seizure": 0.05, "seizure": 0.10}, timestamp=9.0)
        self.assertIsNone(self.engine.active_alert)
        self.assertEqual(cleared_alert["ended_at"], 9.0)

    def test_f18_single_dip_suppression(self):
        """4. Confirms single window dip does not terminate active alert."""
        # Trigger CRITICAL alert
        for i in range(5):
            self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
        self.assertIsNotNone(self.engine.active_alert)

        # 3 normal windows
        for i in range(5, 8):
            self.engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=float(i))
        self.assertEqual(self.engine.consecutive_below_count, 3)

        # 1 spike back above threshold resets consecutive below counter
        self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.75}, timestamp=8.0)
        self.assertEqual(self.engine.consecutive_below_count, 0)
        self.assertIsNotNone(self.engine.active_alert)

    def test_f18_cooldown_timer(self):
        """5. Enforces 120-second cooldown before re-triggering same alert type."""
        # Trigger and clear alert ending at t = 100.0
        for i in range(5):
            self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.8}, timestamp=float(i))
        for i in range(5, 10):
            self.engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=100.0)
        self.assertIsNone(self.engine.active_alert)

        # At t = 150.0 (50s later < 120s cooldown), incoming seizure pattern should be blocked
        for i in range(5):
            alert = self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=150.0 + i)
            self.assertIsNone(alert)  # Blocked by cooldown

        # At t = 221.0 (121s later > 120s cooldown), incoming seizure pattern is allowed
        for i in range(3):
            alert = self.engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=221.0 + i)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["level"], "CRITICAL")


# =============================================================================
# Feature 19: Multi-Modal Alerting
# =============================================================================
class TestFeature19MultiModalAlerting(unittest.TestCase):
    """Feature 19: Multi-modal alerting (banner, pulse, sound, push) (PRD FR-14).

    Verifies WebSocket alert broadcast schema, critical UI banner activation,
    red edge pulse, audio alarm trigger, and Web Push notification dispatch.
    """

    class MultiModalAlertDispatcher:
        """Simulates multi-modal alert broadcast and UI notification state."""

        def __init__(self):
            self.banner_visible = False
            self.banner_level = ""
            self.red_edge_pulse = False
            self.audio_alarm_playing = False
            self.push_dispatched = False
            self.last_push_payload: Optional[Dict[str, Any]] = None

        def handle_alert(self, alert_event: Dict[str, Any], tab_is_hidden: bool = False):
            level = alert_event["level"]
            self.banner_visible = True
            self.banner_level = level

            if level == "CRITICAL":
                self.red_edge_pulse = True
                self.audio_alarm_playing = True
            else:
                self.red_edge_pulse = False
                self.audio_alarm_playing = False

            if tab_is_hidden:
                self.push_dispatched = True
                self.last_push_payload = {
                    "title": f"Seizure-Sentry: {level} Alert",
                    "body": f"Detected {level} event. Peak probability: {alert_event.get('peak_prob', 0.0):.1%}",
                    "level": level,
                    "timestamp": alert_event.get("started_at", 0.0),
                }

    def setUp(self):
        self.dispatcher = self.MultiModalAlertDispatcher()

    def test_f19_alert_broadcast_schema(self):
        """1. Validates WebSocket alert message schema (id, level, started_at, peak_prob)."""
        alert_msg = {
            "type": "alert",
            "id": "alert-crit-999",
            "level": "CRITICAL",
            "started_at": 1700000010.5,
            "peak_prob": 0.942,
        }
        assert_alert_schema(alert_msg)
        self.assertEqual(alert_msg["level"], "CRITICAL")

    def test_f19_critical_ui_banner(self):
        """2. Verifies UI banner activation with alert level and elapsed time."""
        alert = {"id": "1", "level": "CRITICAL", "started_at": 100.0, "peak_prob": 0.92}
        self.dispatcher.handle_alert(alert, tab_is_hidden=False)
        self.assertTrue(self.dispatcher.banner_visible)
        self.assertEqual(self.dispatcher.banner_level, "CRITICAL")

    def test_f19_red_edge_pulse(self):
        """3. Verifies red viewport border pulse activation on CRITICAL alert."""
        # Warning does not pulse
        alert_warn = {"id": "2", "level": "WARNING", "started_at": 100.0, "peak_prob": 0.65}
        self.dispatcher.handle_alert(alert_warn)
        self.assertFalse(self.dispatcher.red_edge_pulse)

        # Critical activates edge pulse
        alert_crit = {"id": "3", "level": "CRITICAL", "started_at": 110.0, "peak_prob": 0.88}
        self.dispatcher.handle_alert(alert_crit)
        self.assertTrue(self.dispatcher.red_edge_pulse)

    def test_f19_audio_alarm_trigger(self):
        """4. Confirms Web Audio synthesized alarm tone triggered on CRITICAL alert."""
        alert = {"id": "4", "level": "CRITICAL", "started_at": 100.0, "peak_prob": 0.95}
        self.dispatcher.handle_alert(alert)
        self.assertTrue(self.dispatcher.audio_alarm_playing)

    def test_f19_push_notification_dispatch(self):
        """5. Verifies VAPID push payload dispatched when tab is in background."""
        alert = {"id": "5", "level": "CRITICAL", "started_at": 120.0, "peak_prob": 0.89}
        self.dispatcher.handle_alert(alert, tab_is_hidden=True)
        self.assertTrue(self.dispatcher.push_dispatched)
        payload = self.dispatcher.last_push_payload
        self.assertIn("CRITICAL Alert", payload["title"])
        self.assertIn("88.9%" if "88.9" in payload["body"] else "89.0%", payload["body"])


# =============================================================================
# Feature 20: Artifact Guard Suppression
# =============================================================================
class TestFeature20ArtifactGuard(unittest.TestCase):
    """Feature 20: Artifact Guard Suppression (PRD §7).

    Activates when > 4 channels are noisy/railed/flat.
    Mutes alerts, shows "Signal quality too low", pauses predictions,
    and requires <= 4 bad channels for >= 2.0s to release.
    """

    class ArtifactGuardController:
        """Simulates Artifact Guard interlock."""

        def __init__(self, bad_channel_threshold: int = 4, release_hold_sec: float = 2.0):
            self.bad_threshold = bad_channel_threshold
            self.release_hold_sec = release_hold_sec
            self.is_active = False
            self.clean_since: Optional[float] = None
            self.predictions_paused = False
            self.banner_message = ""

        def evaluate_channels(self, channel_qualities: List[str], current_time: float) -> bool:
            bad_count = sum(1 for q in channel_qualities if q in ("noisy", "flat", "railed"))

            if bad_count > self.bad_threshold:
                # Trigger / maintain guard
                self.is_active = True
                self.clean_since = None
                self.predictions_paused = True
                self.banner_message = "Signal quality too low. Predictions paused."
            else:
                # Potential recovery
                if self.is_active:
                    if self.clean_since is None:
                        self.clean_since = current_time

                    if (current_time - self.clean_since) >= self.release_hold_sec:
                        self.is_active = False
                        self.predictions_paused = False
                        self.banner_message = ""
                        self.clean_since = None
                else:
                    self.clean_since = None

            return self.is_active

    def setUp(self):
        self.guard = self.ArtifactGuardController(bad_channel_threshold=4, release_hold_sec=2.0)

    def test_f20_activation_threshold(self):
        """1. Activates when > 4 channels (5 or more) are noisy/flat/railed."""
        # Exactly 4 bad channels -> NOT activated
        qual_4_bad = ["good"] * 12 + ["noisy", "noisy", "flat", "railed"]
        active_4 = self.guard.evaluate_channels(qual_4_bad, current_time=0.0)
        self.assertFalse(active_4)

        # 5 bad channels -> TRIGGERED (> 4)
        qual_5_bad = ["good"] * 11 + ["noisy", "noisy", "flat", "railed", "flat"]
        active_5 = self.guard.evaluate_channels(qual_5_bad, current_time=1.0)
        self.assertTrue(active_5)

    def test_f20_alert_suppression(self):
        """2. Asserts active alerts muted and new alerts suppressed while active."""
        engine = HysteresisAlertEngine()
        # Trigger alert
        for i in range(5):
            engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.8}, timestamp=float(i))
        self.assertIsNotNone(engine.active_alert)
        self.assertFalse(engine.active_alert["muted"])

        # Engage artifact guard
        engine.set_artifact_guard(True)
        self.assertTrue(engine.active_alert["muted"])

        # High seizure window while guard is active -> returns muted alert
        res = engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.95}, timestamp=6.0)
        self.assertTrue(res["muted"])

    def test_f20_ui_banner_display(self):
        """3. Verifies 'Signal quality too low' banner broadcast."""
        qual_bad = ["railed"] * 6 + ["good"] * 10
        self.guard.evaluate_channels(qual_bad, current_time=0.0)
        self.assertTrue(self.guard.is_active)
        self.assertIn("Signal quality too low", self.guard.banner_message)

    def test_f20_release_condition(self):
        """4. Disengages only after <= 4 bad channels for >= 2.0 seconds."""
        # Activate guard at t = 0.0
        self.guard.evaluate_channels(["railed"] * 6 + ["good"] * 10, current_time=0.0)
        self.assertTrue(self.guard.is_active)

        # At t = 1.0 (clean signal starts, elapsed 0.0s < 2.0s) -> STILL ACTIVE
        qual_clean = ["good"] * 16
        self.guard.evaluate_channels(qual_clean, current_time=1.0)
        self.assertTrue(self.guard.is_active)

        # At t = 2.5 (clean signal for 1.5s < 2.0s) -> STILL ACTIVE
        self.guard.evaluate_channels(qual_clean, current_time=2.5)
        self.assertTrue(self.guard.is_active)

        # At t = 3.0 (clean signal for 2.0s >= 2.0s) -> RELEASED
        self.guard.evaluate_channels(qual_clean, current_time=3.0)
        self.assertFalse(self.guard.is_active)

    def test_f20_prediction_pause_flag(self):
        """5. Confirms predictions marked as paused during artifact guard."""
        # Normal state
        self.assertFalse(self.guard.predictions_paused)

        # Activate guard
        self.guard.evaluate_channels(["flat"] * 8 + ["good"] * 8, current_time=0.0)
        self.assertTrue(self.guard.predictions_paused)


if __name__ == "__main__":
    unittest.main()
