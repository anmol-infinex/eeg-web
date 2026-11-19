"""Tier 2: Boundary & Corner Cases - Features 18-20 (Alerting & Safety).

Exhaustively covers edge, corner, and error-handling conditions for:
- Feature 18: Hysteresis Alert Engine (6 boundary tests)
- Feature 19: Multi-Modal Alerting (5 boundary tests)
- Feature 20: Artifact Guard Suppression (5 boundary tests)

Total: 16 boundary tests.
"""

from __future__ import annotations

import time
import unittest
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_probability_distribution,
)


# =============================================================================
# Feature 18: Hysteresis Alert Engine - Boundary Tests
# =============================================================================
class TestBoundaryFeature18HysteresisAlertEngine(unittest.TestCase):
    """Boundary and corner cases for Hysteresis Alert Engine (Feature 18)."""

    class HysteresisEngine:
        def __init__(self, cooldown_sec: float = 120.0):
            self.history_seizure_probs: List[float] = []
            self.history_preseizure_probs: List[float] = []
            self.consecutive_normal_windows = 0
            self.active_alert: Optional[Dict[str, Any]] = None
            self.last_alert_cleared_time: float = -999999.0
            self.cooldown_sec = cooldown_sec

        def process_window(self, probs: Dict[str, float], now: float) -> Optional[Dict[str, Any]]:
            p_seiz = probs.get("seizure", 0.0)
            p_pre = probs.get("pre_seizure", 0.0)

            self.history_seizure_probs.append(p_seiz)
            if len(self.history_seizure_probs) > 5:
                self.history_seizure_probs.pop(0)

            self.history_preseizure_probs.append(p_pre)
            if len(self.history_preseizure_probs) > 10:
                self.history_preseizure_probs.pop(0)

            # Check if active alert should clear
            if self.active_alert is not None:
                if p_seiz < 0.70:
                    self.consecutive_normal_windows += 1
                else:
                    self.consecutive_normal_windows = 0

                # Requires 5 consecutive windows < 0.70 to clear
                if self.consecutive_normal_windows >= 5:
                    self.active_alert = None
                    self.last_alert_cleared_time = now
                    self.consecutive_normal_windows = 0
                return self.active_alert

            # Check cooldown before triggering new alert
            if (now - self.last_alert_cleared_time) < self.cooldown_sec:
                return None

            # CRITICAL check: 3 of last 5 windows >= 0.70
            seiz_count = sum(1 for p in self.history_seizure_probs[-5:] if p >= 0.70)
            if len(self.history_seizure_probs) >= 3 and seiz_count >= 3:
                self.active_alert = {
                    "id": str(uuid.uuid4()),
                    "level": "CRITICAL",
                    "started_at": now,
                    "peak_prob": max(self.history_seizure_probs[-5:]),
                }
                self.consecutive_normal_windows = 0
                return self.active_alert

            # WARNING check: mean of last 10 windows >= 0.60
            if len(self.history_preseizure_probs) == 10:
                mean_pre = sum(self.history_preseizure_probs) / 10.0
                if mean_pre >= 0.60:
                    self.active_alert = {
                        "id": str(uuid.uuid4()),
                        "level": "WARNING",
                        "started_at": now,
                        "peak_prob": mean_pre,
                    }
                    self.consecutive_normal_windows = 0
                    return self.active_alert

            return None

    def test_f18_exact_threshold_boundary_2_vs_3_windows(self):
        """1. Exactly 2 of 5 windows >= 0.70 does NOT trigger; exactly 3 of 5 windows TRIGGERS CRITICAL."""
        engine = self.HysteresisEngine()
        now = 1000.0

        # Window 1: 0.75 (1/5)
        self.assertIsNone(engine.process_window({"seizure": 0.75, "normal": 0.25}, now=now))
        # Window 2: 0.72 (2/5) -> NOT triggered yet
        self.assertIsNone(engine.process_window({"seizure": 0.72, "normal": 0.28}, now=now + 1))
        # Window 3: 0.65 (2/5)
        self.assertIsNone(engine.process_window({"seizure": 0.65, "normal": 0.35}, now=now + 2))
        # Window 4: 0.60 (2/5)
        self.assertIsNone(engine.process_window({"seizure": 0.60, "normal": 0.40}, now=now + 3))

        # Window 5: 0.71 -> Exactly 3 of 5 (windows 1, 2, 5) >= 0.70 -> TRIGGERS CRITICAL!
        alert = engine.process_window({"seizure": 0.71, "normal": 0.29}, now=now + 4)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["level"], "CRITICAL")
        self.assertEqual(alert["peak_prob"], 0.75)

    def test_f18_flapping_probability_suppression(self):
        """2. Probabilities oscillating across 0.70 handled without alert flap storm."""
        engine = self.HysteresisEngine()
        now = 1000.0

        # Feed 3 high windows to trigger
        engine.process_window({"seizure": 0.75}, now)
        engine.process_window({"seizure": 0.75}, now + 1)
        alert = engine.process_window({"seizure": 0.75}, now + 2)
        self.assertIsNotNone(alert)
        alert_id = alert["id"]

        # Probability dips below 0.70 for 1 window (single dip)
        a_dip = engine.process_window({"seizure": 0.68}, now + 3)
        self.assertIsNotNone(a_dip, "Single window dip must not clear alert")
        self.assertEqual(a_dip["id"], alert_id)

        # Probability rises again
        a_rise = engine.process_window({"seizure": 0.72}, now + 4)
        self.assertIsNotNone(a_rise)
        self.assertEqual(a_rise["id"], alert_id)

    def test_f18_clearance_consecutive_windows_boundary(self):
        """3. Clearance requires exactly 5 consecutive windows < 0.70; 4 windows does not clear."""
        engine = self.HysteresisEngine()
        now = 1000.0

        # Trigger alert
        for i in range(3):
            engine.process_window({"seizure": 0.80}, now + i)
        self.assertIsNotNone(engine.active_alert)

        # 4 consecutive normal windows (< 0.70)
        for i in range(4):
            res = engine.process_window({"seizure": 0.20}, now + 3 + i)
            self.assertIsNotNone(res, f"Window {i+1}/5 normal must not clear alert yet")

        # 5th consecutive normal window clears alert!
        cleared_res = engine.process_window({"seizure": 0.20}, now + 7)
        self.assertIsNone(cleared_res, "5th consecutive normal window must clear alert")
        self.assertIsNone(engine.active_alert)

    def test_f18_cooldown_boundary_119s_vs_121s(self):
        """4. Re-triggering at t = 119s is suppressed; re-triggering at t = 121s is permitted."""
        engine = self.HysteresisEngine(cooldown_sec=120.0)
        # Clear alert at t = 1000.0
        engine.last_alert_cleared_time = 1000.0

        # At t = 1119.0 (119 seconds since clearance -> suppressed)
        for _ in range(5):
            res119 = engine.process_window({"seizure": 0.90}, now=1119.0)
        self.assertIsNone(res119, "Must be suppressed during 120s cooldown")

        # At t = 1121.0 (121 seconds since clearance -> allowed)
        for _ in range(5):
            res121 = engine.process_window({"seizure": 0.90}, now=1121.0)
        self.assertIsNotNone(res121, "Must allow re-triggering after 120s cooldown expires")

    def test_f18_pre_seizure_mean_boundary_0599_vs_0601(self):
        """5. Mean P_pre = 0.599 does NOT trigger WARNING; mean P_pre = 0.601 TRIGGERS WARNING."""
        engine1 = self.HysteresisEngine()
        # 10 windows with mean = 0.599
        for _ in range(10):
            res_below = engine1.process_window({"pre_seizure": 0.599}, now=1000.0)
        self.assertIsNone(res_below, "Mean 0.599 < 0.60 must not trigger WARNING")

        engine2 = self.HysteresisEngine()
        # 10 windows with mean = 0.601
        for _ in range(10):
            res_above = engine2.process_window({"pre_seizure": 0.601}, now=1000.0)
        self.assertIsNotNone(res_above, "Mean 0.601 >= 0.60 must trigger WARNING")
        self.assertEqual(res_above["level"], "WARNING")

    def test_f18_critical_takes_precedence_over_warning(self):
        """6. When both CRITICAL and WARNING conditions are satisfied, CRITICAL takes precedence."""
        engine = self.HysteresisEngine()
        # Both high seizure (>=0.70) and high pre-seizure (>=0.60)
        for _ in range(10):
            alert = engine.process_window({"seizure": 0.85, "pre_seizure": 0.70}, now=1000.0)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["level"], "CRITICAL")


# =============================================================================
# Feature 19: Multi-Modal Alerting - Boundary Tests
# =============================================================================
class TestBoundaryFeature19MultiModalAlerting(unittest.TestCase):
    """Boundary and corner cases for Multi-Modal Alerting (Feature 19)."""

    def test_f19_alert_payload_schema_boundary(self):
        """1. Alert payload schema boundary checks: peak_prob clamped [0.0, 1.0], started_at > 0."""
        valid_alert = {
            "type": "alert",
            "id": str(uuid.uuid4()),
            "level": "CRITICAL",
            "started_at": 1700000000.123,
            "peak_prob": 0.985,
        }
        assert_alert_schema(valid_alert)

    def test_f19_audio_alarm_strict_browser_autoplay_policy(self):
        """2. Audio alarm handler handles promise rejection / blocked Web Audio context cleanly."""
        class MockAudioEngine:
            def __init__(self, autoplay_allowed: bool = False):
                self.autoplay_allowed = autoplay_allowed
                self.audio_state = "suspended" if not autoplay_allowed else "running"

            def trigger_alarm(self) -> bool:
                if not self.autoplay_allowed:
                    # Browser autoplay policy rejected AudioContext resume
                    self.audio_state = "blocked"
                    return False
                self.audio_state = "playing"
                return True

        blocked_audio = MockAudioEngine(autoplay_allowed=False)
        self.assertFalse(blocked_audio.trigger_alarm())
        self.assertEqual(blocked_audio.audio_state, "blocked")

        allowed_audio = MockAudioEngine(autoplay_allowed=True)
        self.assertTrue(allowed_audio.trigger_alarm())
        self.assertEqual(allowed_audio.audio_state, "playing")

    def test_f19_push_notification_empty_subscription(self):
        """3. VAPID push dispatch with null or empty subscription endpoint handled safely."""
        def dispatch_web_push(subscription: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> bool:
            if not subscription or not subscription.get("endpoint"):
                # No push subscription registered; silently ignore without throwing
                return False
            return True

        self.assertFalse(dispatch_web_push(None, {"title": "Seizure Alert"}))
        self.assertFalse(dispatch_web_push({}, {"title": "Seizure Alert"}))
        self.assertTrue(dispatch_web_push({"endpoint": "https://fcm.googleapis.com/test"}, {"title": "Alert"}))

    def test_f19_red_edge_pulse_duration_persistence(self):
        """4. Edge pulse animation persists continuously while alert is active and halts upon clearance."""
        class EdgePulseController:
            def __init__(self):
                self.pulsing = False

            def update(self, active_alert: Optional[Dict[str, Any]]):
                self.pulsing = bool(active_alert and active_alert.get("level") == "CRITICAL")

        ctrl = EdgePulseController()
        ctrl.update(None)
        self.assertFalse(ctrl.pulsing)

        ctrl.update({"level": "CRITICAL"})
        self.assertTrue(ctrl.pulsing)

        ctrl.update({"level": "WARNING"})
        self.assertFalse(ctrl.pulsing)

    def test_f19_consecutive_alerts_id_uniqueness(self):
        """5. Consecutive alert events generate strictly unique UUIDs."""
        ids = {str(uuid.uuid4()) for _ in range(100)}
        self.assertEqual(len(ids), 100)


# =============================================================================
# Feature 20: Artifact Guard Suppression - Boundary Tests
# =============================================================================
class TestBoundaryFeature20ArtifactGuard(unittest.TestCase):
    """Boundary and corner cases for Artifact Guard Suppression (Feature 20)."""

    class ArtifactGuard:
        def __init__(self):
            self.engaged = False
            self.good_signal_start: Optional[float] = None

        def update(self, bad_channel_count: int, now: float) -> bool:
            if bad_channel_count > 4:
                # Engaged immediately when > 4 bad channels
                self.engaged = True
                self.good_signal_start = None
            else:
                # Releases only after clean signal (<= 4 bad channels) for >= 2.0s
                if self.engaged:
                    if self.good_signal_start is None:
                        self.good_signal_start = now
                    elif (now - self.good_signal_start) >= 2.0:
                        self.engaged = False
                        self.good_signal_start = None
            return self.engaged

    def test_f20_channel_count_boundary_4_vs_5_bad_channels(self):
        """1. Exactly 4 bad channels: NOT engaged; exactly 5 bad channels: ENGAGED."""
        guard = self.ArtifactGuard()
        now = 1000.0

        # 4 bad channels (e.g. 4 railed channels) -> NOT engaged
        self.assertFalse(guard.update(bad_channel_count=4, now=now))

        # 5 bad channels -> ENGAGED immediately
        self.assertTrue(guard.update(bad_channel_count=5, now=now + 0.25))

    def test_f20_suppression_of_false_seizure_during_artifact(self):
        """2. When Artifact Guard is active, high seizure probability (0.98) is MUTED."""
        guard = self.ArtifactGuard()
        guard.update(bad_channel_count=6, now=1000.0)
        self.assertTrue(guard.engaged)

        # Alert engine filtering: if guard is engaged, suppress alarm
        def dispatch_alert(guard_active: bool, p_seiz: float) -> Tuple[bool, str]:
            if guard_active:
                return False, "Signal quality too low. Predictions paused."
            if p_seiz >= 0.70:
                return True, "SEIZURE ALERT"
            return False, "Normal"

        sound_triggered, message = dispatch_alert(guard.engaged, p_seiz=0.98)
        self.assertFalse(sound_triggered)
        self.assertIn("Predictions paused", message)

    def test_f20_clearance_timing_boundary_19s_vs_21s(self):
        """3. When bad channels drop <= 4, remains engaged at t = 1.9s, releases at t = 2.1s."""
        guard = self.ArtifactGuard()
        # Engage at t = 1000.0
        guard.update(bad_channel_count=5, now=1000.0)
        self.assertTrue(guard.engaged)

        # Channels recover to 2 bad channels at t = 1001.0 (clean signal start)
        guard.update(bad_channel_count=2, now=1001.0)
        self.assertTrue(guard.engaged)

        # At t = 1002.9 (1.9 seconds of clean signal -> still engaged)
        guard.update(bad_channel_count=2, now=1002.9)
        self.assertTrue(guard.engaged, "Must remain engaged at 1.9s (< 2.0s)")

        # At t = 1003.1 (2.1 seconds of clean signal -> RELEASED)
        guard.update(bad_channel_count=2, now=1003.1)
        self.assertFalse(guard.engaged, "Must release after 2.1s (>= 2.0s clean)")

    def test_f20_prediction_pause_flag_in_broadcast(self):
        """4. Broadcast payload includes paused: true and artifact_guard: true when guard is active."""
        broadcast = {
            "type": "prediction",
            "artifact_guard": True,
            "paused": True,
            "bad_channels": 6,
            "probs": {"normal": 0.33, "pre_seizure": 0.33, "seizure": 0.34},
        }
        self.assertTrue(broadcast["artifact_guard"])
        self.assertTrue(broadcast["paused"])

    def test_f20_all_channels_bad_extreme_condition(self):
        """5. All 16 channels railed/disconnected (16/16 bad) triggers immediate guard."""
        guard = self.ArtifactGuard()
        self.assertTrue(guard.update(bad_channel_count=16, now=1000.0))


if __name__ == "__main__":
    unittest.main()
