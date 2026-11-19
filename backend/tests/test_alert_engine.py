"""
backend/tests/test_alert_engine.py
Unit tests for HysteresisAlertEngine and ArtifactGuard.
"""
import numpy as np
import pytest
from app.alert_engine import HysteresisAlertEngine, ArtifactGuard


def test_critical_seizure_trigger_3_of_5():
    """Verify CRITICAL alert triggers when >= 3 of last 5 windows have P(seizure) >= 0.70."""
    engine = HysteresisAlertEngine(cooldown_sec=120.0, clear_consecutive=5)
    # 2 normal windows
    engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=1.0)
    engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=2.0)
    # 1st and 2nd high seizure windows -> no alert yet
    assert engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.80}, timestamp=3.0) is None
    assert engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=4.0) is None
    # 3rd high seizure window (3 of 5) -> triggers CRITICAL
    alert = engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.90}, timestamp=5.0)
    assert alert is not None
    assert alert["level"] == "CRITICAL"
    assert alert["peak_prob"] >= 0.90
    assert alert["ended_at"] is None
    assert alert["muted"] is False


def test_critical_not_triggered_on_2_windows():
    """Verify 2 high seizure windows within 5 windows do NOT trigger alert."""
    engine = HysteresisAlertEngine()
    engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=1.0)
    engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.80}, timestamp=2.0)
    engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=3.0)
    engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=4.0)
    alert = engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=5.0)
    assert alert is None
    assert engine.active_alert is None


def test_warning_preseizure_trigger_mean_10():
    """Verify WARNING triggers when mean P(pre_seizure) >= 0.60 over 10 windows."""
    engine = HysteresisAlertEngine(cooldown_sec=120.0)
    for i in range(9):
        res = engine.process_window({"normal": 0.3, "pre_seizure": 0.65, "seizure": 0.05}, timestamp=float(i))
        assert res is None

    alert = engine.process_window({"normal": 0.3, "pre_seizure": 0.65, "seizure": 0.05}, timestamp=9.0)
    assert alert is not None
    assert alert["level"] == "WARNING"
    assert alert["peak_prob"] >= 0.65


def test_warning_preseizure_float_precision_exact_boundary():
    """Verify WARNING triggers on 10 windows of exact 0.60 despite IEEE-754 float precision."""
    engine = HysteresisAlertEngine(cooldown_sec=120.0)
    for i in range(9):
        assert engine.process_window({"normal": 0.35, "pre_seizure": 0.60, "seizure": 0.05}, timestamp=float(i)) is None

    alert = engine.process_window({"normal": 0.35, "pre_seizure": 0.60, "seizure": 0.05}, timestamp=9.0)
    assert alert is not None
    assert alert["level"] == "WARNING"
    assert engine.active_alert is not None


def test_hysteresis_clearance_exactly_5_normal():
    """Verify active alert ends only after strictly 5 consecutive normal windows."""
    engine = HysteresisAlertEngine(cooldown_sec=120.0, clear_consecutive=5)
    # Trigger CRITICAL
    for i in range(5):
        engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
    assert engine.active_alert is not None

    # 4 consecutive normal windows -> still active
    for i in range(5, 9):
        alert = engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=float(i))
        assert alert is not None
        assert alert["ended_at"] is None

    # 5th consecutive normal window -> CLEARED
    cleared = engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=9.0)
    assert cleared is not None
    assert cleared["ended_at"] == 9.0
    assert engine.active_alert is None


def test_single_dip_resets_clearance_count():
    """Verify a spike resets consecutive normal clearance counter."""
    engine = HysteresisAlertEngine(cooldown_sec=120.0, clear_consecutive=5)
    for i in range(5):
        engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
    assert engine.active_alert is not None

    # 3 normal windows
    for i in range(5, 8):
        engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=float(i))
    assert engine.consecutive_below_count == 3

    # Spike
    engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=8.0)
    assert engine.consecutive_below_count == 0
    assert engine.active_alert is not None


def test_cooldown_timer_120s():
    """Verify 120s cooldown blocks re-triggering of the same alert type."""
    engine = HysteresisAlertEngine(cooldown_sec=120.0, clear_consecutive=5)
    for i in range(5):
        engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
    for i in range(5, 10):
        engine.process_window({"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, timestamp=float(i))
    assert engine.active_alert is None
    # Cleared at t = 9.0

    # At t = 50.0 (elapsed 41s < 120s) -> 3 high seizure windows do NOT trigger alert
    for i in range(50, 55):
        alert = engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
        assert alert is None

    # At t = 135.0 (elapsed 126s > 120s) -> triggers new CRITICAL
    for i in range(135, 140):
        alert = engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
    assert alert is not None
    assert alert["level"] == "CRITICAL"


def test_warning_escalates_to_critical():
    """Verify WARNING alert immediately escalates to CRITICAL when seizure activity arrives."""
    engine = HysteresisAlertEngine(cooldown_sec=120.0)
    for i in range(10):
        engine.process_window({"normal": 0.3, "pre_seizure": 0.65, "seizure": 0.05}, timestamp=float(i))
    assert engine.active_alert is not None
    assert engine.active_alert["level"] == "WARNING"

    # Seizure arrives: 3 of 5 windows with P(seiz) >= 0.70
    for i in range(10, 13):
        res = engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
    assert res is not None
    assert res["level"] == "CRITICAL"


def test_artifact_guard_activation_and_hold():
    """Verify ArtifactGuard activates on >4 bad channels and requires 2.0s clean to disengage."""
    guard = ArtifactGuard(bad_channel_threshold=4, release_hold_sec=2.0)

    # 4 bad channels -> inactive
    assert guard.evaluate_channels(["good"] * 12 + ["noisy", "noisy", "flat", "railed"], current_time=0.0) is False
    assert guard.is_active is False

    # 5 bad channels -> active
    assert guard.evaluate_channels(["good"] * 11 + ["noisy", "noisy", "flat", "railed", "flat"], current_time=1.0) is True
    assert guard.is_active is True
    assert guard.predictions_paused is True
    assert "Signal quality too low" in guard.banner_message

    # Clean signal arrives at t = 2.0 (elapsed 0.0s < 2.0s) -> still active
    assert guard.evaluate_channels(["good"] * 16, current_time=2.0) is True
    assert guard.evaluate_channels(["good"] * 16, current_time=3.5) is True  # 1.5s < 2.0s

    # At t = 4.0 (elapsed 2.0s >= 2.0s) -> disengages
    assert guard.evaluate_channels(["good"] * 16, current_time=4.0) is False
    assert guard.is_active is False
    assert guard.predictions_paused is False


def test_artifact_guard_suppresses_alerts():
    """Verify ArtifactGuard mutes active alert and suppresses new alerts."""
    engine = HysteresisAlertEngine()
    for i in range(5):
        engine.process_window({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.85}, timestamp=float(i))
    assert engine.active_alert is not None
    assert engine.active_alert["muted"] is False

    # Engage artifact guard
    engine.set_artifact_guard(True)
    assert engine.active_alert["muted"] is True

    # High seizure window while guard is engaged -> alert remains muted
    res = engine.process_window({"normal": 0.05, "pre_seizure": 0.05, "seizure": 0.95}, timestamp=6.0)
    assert res is not None
    assert res["muted"] is True
