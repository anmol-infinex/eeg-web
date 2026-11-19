"""
backend/app/alert_engine.py
HysteresisAlertEngine and ArtifactGuard safety interlock for clinical seizure monitoring.
"""
from __future__ import annotations

import time
import uuid
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

logger = logging.getLogger("seizure_sentry.alert_engine")


class ArtifactGuard:
    """
    Evaluates multi-channel EEG quality and mutes alerts during heavy artifact contamination.
    Trigger condition: > 4 channels (5 or more) are noisy, flat, or railed.
    Disengagement condition: <= 4 bad channels for >= 2.0 continuous seconds.
    """
    def __init__(
        self,
        bad_channel_threshold: int = 4,
        release_hold_sec: float = 2.0,
        flat_uv: float = 0.5,
        rail_uv: float = 2000.0,
        noise_uv: float = 300.0,
    ):
        self.bad_threshold = bad_channel_threshold
        self.release_hold_sec = release_hold_sec
        self.flat_uv = flat_uv
        self.rail_uv = rail_uv
        self.noise_uv = noise_uv

        self.is_active: bool = False
        self.predictions_paused: bool = False
        self.banner_message: str = ""
        self.clean_since: Optional[float] = None
        self.last_bad_timestamp: float = 0.0

    def classify_channel(self, signal_channel: np.ndarray, fs: int = 125) -> str:
        """Classify single channel array into 'good', 'flat', 'railed', or 'noisy'."""
        arr = np.asarray(signal_channel, dtype=np.float32)
        if len(arr) == 0:
            return "flat"
        max_abs = float(np.max(np.abs(arr)))
        std = float(np.std(arr))

        if max_abs > 180000.0 or max_abs > self.rail_uv or std > 1500.0:
            return "railed"
        if std < self.flat_uv:
            return "flat"

        # Check mains noise or high broadband RMS
        if std > self.noise_uv:
            return "noisy"

        if len(arr) >= 32 and fs > 0:
            fft_vals = np.abs(np.fft.rfft(arr, axis=0))
            freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
            idx_50 = np.argmin(np.abs(freqs - 50.0))
            mains_peak = float(np.max(fft_vals[max(0, idx_50 - 2) : idx_50 + 3]))
            total_power = float(np.sum(fft_vals))
            if mains_peak / (total_power + 1e-6) > 0.25:
                return "noisy"

        return "good"

    def evaluate_channels(self, channel_qualities: List[str], current_time: Optional[float] = None) -> bool:
        """
        Evaluate a list of quality strings (e.g. ['good', 'noisy', ...]).
        Returns True if Artifact Guard is active, False otherwise.
        """
        now = current_time if current_time is not None else time.time()
        bad_count = sum(1 for q in channel_qualities if q.lower().strip() in ("noisy", "flat", "railed"))

        if bad_count > self.bad_threshold:
            self.is_active = True
            self.clean_since = None
            self.predictions_paused = True
            self.banner_message = "Signal quality too low. Predictions paused."
            self.last_bad_timestamp = now
        else:
            if self.is_active:
                if self.clean_since is None:
                    self.clean_since = now
                if (now - self.clean_since) >= self.release_hold_sec:
                    self.is_active = False
                    self.predictions_paused = False
                    self.banner_message = ""
                    self.clean_since = None
            else:
                self.clean_since = None

        return self.is_active

    def evaluate(self, window_or_qualities: Union[np.ndarray, List[str]], timestamp: Optional[float] = None, fs: int = 256) -> Dict[str, Any]:
        """
        Evaluate a 2D numpy window (samples, channels) or a list of quality strings.
        Returns diagnostic dictionary.
        """
        now = timestamp if timestamp is not None else time.time()
        if isinstance(window_or_qualities, (list, tuple)):
            qualities = [str(q) for q in window_or_qualities]
        elif isinstance(window_or_qualities, np.ndarray):
            if window_or_qualities.ndim == 1:
                qualities = [self.classify_channel(window_or_qualities, fs=fs)]
            else:
                n_channels = window_or_qualities.shape[1]
                qualities = [self.classify_channel(window_or_qualities[:, ch], fs=fs) for ch in range(n_channels)]
        else:
            qualities = []

        is_active = self.evaluate_channels(qualities, current_time=now)
        bad_indices = [idx for idx, q in enumerate(qualities) if q.lower().strip() in ("noisy", "flat", "railed")]

        return {
            "artifact_active": is_active,
            "predictions_paused": self.predictions_paused,
            "bad_count": len(bad_indices),
            "bad_channels": bad_indices,
            "qualities": qualities,
            "banner_message": self.banner_message,
            "reason": self.banner_message if is_active else "OK",
        }


class HysteresisAlertEngine:
    """
    Multi-state alert detection engine preventing noisy oscillations (PRD §7).
    - CRITICAL (Seizure): >= 3 of the last 5 windows have P(seizure) >= 0.70.
    - WARNING (Pre-Seizure): mean P(pre_seizure) >= 0.60 over the last 10 windows.
    - Hysteresis clearance: exactly 5 consecutive windows below threshold.
    - Cooldown: 120.0 seconds before re-triggering an alert of the same type.
    - Artifact Guard: when active, suppresses new alerts and mutes active alerts.
    """
    def __init__(
        self,
        cooldown_sec: float = 120.0,
        clear_consecutive: int = 5,
        critical_threshold: float = 0.70,
        warning_threshold: float = 0.60,
    ):
        self.cooldown_sec = cooldown_sec
        self.clear_consecutive = clear_consecutive
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold

        self.current_state: str = "NORMAL"
        self.history_windows: List[Dict[str, float]] = []
        self.active_alert: Optional[Dict[str, Any]] = None
        self.last_alert_end_time: Dict[str, float] = {"CRITICAL": -9999.0, "WARNING": -9999.0}
        self.consecutive_below_count: int = 0
        self.artifact_guard_active: bool = False

    def set_artifact_guard(self, active: bool) -> None:
        """Update artifact guard status and mute active alert if active."""
        self.artifact_guard_active = active
        if active and self.active_alert is not None:
            self.active_alert["muted"] = True

    def process_window(
        self,
        probs: Dict[str, float],
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single prediction window.
        Returns the active alert dict if active/cleared, or None.
        (Matches PRD §7 and test_f18_f20_alerts contract).
        """
        now = timestamp if timestamp is not None else time.time()
        self.history_windows.append(probs)
        if len(self.history_windows) > 50:
            self.history_windows.pop(0)

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

            # Check for escalation: if WARNING is active and CRITICAL triggers
            if level == "WARNING":
                last_5 = self.history_windows[-5:]
                seiz_count = sum(1 for w in last_5 if w.get("seizure", 0.0) >= self.critical_threshold)
                if seiz_count >= 3:
                    time_since_crit = now - self.last_alert_end_time.get("CRITICAL", -9999.0)
                    if time_since_crit >= self.cooldown_sec:
                        # Escalate to CRITICAL
                        self.active_alert["ended_at"] = now
                        self.last_alert_end_time["WARNING"] = now
                        self.active_alert = {
                            "id": str(uuid.uuid4()),
                            "level": "CRITICAL",
                            "started_at": now,
                            "ended_at": None,
                            "peak_prob": max(w.get("seizure", 0.0) for w in last_5),
                            "muted": False,
                            "ui_banner": True,
                            "red_border_pulse": True,
                            "sound_alarm": True,
                        }
                        self.current_state = "CRITICAL"
                        self.consecutive_below_count = 0
                        return self.active_alert

            # Threshold check for clearance
            is_below = (p_seiz < self.critical_threshold) if level == "CRITICAL" else (p_pre < self.warning_threshold)
            if is_below:
                self.consecutive_below_count += 1
            else:
                self.consecutive_below_count = 0  # Single spike resets clearance count

            if self.consecutive_below_count >= self.clear_consecutive:
                # Alert cleared via hysteresis
                self.active_alert["ended_at"] = now
                self.last_alert_end_time[level] = now
                cleared = dict(self.active_alert)
                self.active_alert = None
                self.current_state = "NORMAL"
                self.consecutive_below_count = 0
                return cleared

            return self.active_alert

        # 3. Check for new CRITICAL (Seizure) trigger
        # >= 3 of last 5 windows with P(seizure) >= 0.70
        last_5 = self.history_windows[-5:]
        seiz_count = sum(1 for w in last_5 if w.get("seizure", 0.0) >= self.critical_threshold)
        if seiz_count >= 3:
            time_since_last = now - self.last_alert_end_time.get("CRITICAL", -9999.0)
            if time_since_last >= self.cooldown_sec:
                self.active_alert = {
                    "id": str(uuid.uuid4()),
                    "level": "CRITICAL",
                    "started_at": now,
                    "ended_at": None,
                    "peak_prob": max(w.get("seizure", 0.0) for w in last_5),
                    "muted": False,
                    "ui_banner": True,
                    "red_border_pulse": True,
                    "sound_alarm": True,
                }
                self.current_state = "CRITICAL"
                self.consecutive_below_count = 0
                return self.active_alert

        # 4. Check for new WARNING (Pre-Seizure) trigger
        # Mean P(pre_seizure) >= 0.60 over last 10 windows
        last_10 = self.history_windows[-10:]
        if len(last_10) >= 10:
            mean_pre = sum(w.get("pre_seizure", 0.0) for w in last_10) / 10.0
            if mean_pre >= (self.warning_threshold - 1e-6):
                time_since_last = now - self.last_alert_end_time.get("WARNING", -9999.0)
                if time_since_last >= self.cooldown_sec:
                    self.active_alert = {
                        "id": str(uuid.uuid4()),
                        "level": "WARNING",
                        "started_at": now,
                        "ended_at": None,
                        "peak_prob": max(w.get("pre_seizure", 0.0) for w in last_10),
                        "muted": False,
                        "ui_banner": True,
                        "red_border_pulse": False,
                        "sound_alarm": False,
                    }
                    self.current_state = "WARNING"
                    self.consecutive_below_count = 0
                    return self.active_alert

        return None

    def process_prediction(
        self,
        probs: Dict[str, float],
        timestamp: Optional[float] = None,
        artifact_suppressed: bool = False,
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Process a new window prediction and return state transition tuples.
        Returns:
            (current_state, new_alert_dict_or_None, cleared_alert_dict_or_None)
        """
        now = timestamp if timestamp is not None else time.time()
        was_alert = self.active_alert
        was_level = was_alert["level"] if was_alert else None

        if artifact_suppressed != self.artifact_guard_active:
            self.set_artifact_guard(artifact_suppressed)

        result_alert = self.process_window(probs, timestamp=now)

        new_alert = None
        cleared_alert = None

        if was_alert is None and self.active_alert is not None:
            new_alert = dict(self.active_alert)
        elif was_alert is not None and self.active_alert is None:
            cleared_alert = result_alert  # Contains ended_at
        elif was_alert is not None and self.active_alert is not None and was_level != self.active_alert["level"]:
            # Escalated
            cleared_alert = dict(was_alert)
            new_alert = dict(self.active_alert)

        return self.current_state, new_alert, cleared_alert
