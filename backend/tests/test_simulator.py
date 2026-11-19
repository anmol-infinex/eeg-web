"""
backend/tests/test_simulator.py
Unit tests for ReplaySimulator, pacing intervals, speed multipliers, and preset seizure demo progression.
"""
import pytest
import numpy as np
from app.simulator import ReplaySimulator, PresetSeizureDemoRunner, RECORDINGS


def test_recordings_catalog():
    """Verify recordings catalog contains baseline and seizure demo recordings."""
    sim = ReplaySimulator()
    recs = sim.get_recordings()
    assert len(recs) >= 2
    rec_ids = [r["recording_id"] for r in recs]
    assert "rec-normal-baseline" in rec_ids
    assert "preset_seizure_demo" in rec_ids

    for r in recs:
        assert "filename" in r
        assert "duration_sec" in r
        assert "has_seizure" in r
        assert "sample_rate" in r
        assert "channels" in r


def test_start_replay_simulated_badge():
    """Verify replay output includes permanent SIMULATED DATA badge."""
    sim = ReplaySimulator()
    res = sim.start_replay("preset_seizure_demo", speed=1.0)
    assert res["status"] == "started"
    assert res["simulated"] is True
    assert res["badge"] == "SIMULATED DATA"
    assert "session_id" in res


def test_replay_speed_multiplier_and_pacing():
    """Verify speed multiplier adjusts chunk emission interval (interval = 0.25 / speed)."""
    sim = ReplaySimulator()

    # 1.0x speed
    res_1x = sim.start_replay("rec-normal-baseline", speed=1.0)
    int_1x = sim.get_chunk_interval(res_1x["session_id"])
    assert int_1x == pytest.approx(0.25, abs=1e-3)

    # 2.0x speed -> 0.125s interval
    res_2x = sim.start_replay("rec-normal-baseline", speed=2.0)
    int_2x = sim.get_chunk_interval(res_2x["session_id"])
    assert int_2x == pytest.approx(0.125, abs=1e-3)

    # Clamping test: speed 20.0 clamped to 10.0 -> 0.025s interval
    res_max = sim.start_replay("rec-normal-baseline", speed=20.0)
    assert res_max["speed"] == 10.0
    int_max = sim.get_chunk_interval(res_max["session_id"])
    assert int_max == pytest.approx(0.025, abs=1e-3)


def test_replay_invalid_recording():
    """Verify ValueError when invalid recording ID is provided."""
    sim = ReplaySimulator()
    with pytest.raises(ValueError, match="not found"):
        sim.start_replay("non-existent-recording-id")


def test_preset_seizure_demo_progression():
    """Verify clinical progression: Normal (0-15s) -> Pre-Seizure (15-25s) -> Seizure (25-60s) -> Recovery (60-75s)."""
    runner = PresetSeizureDemoRunner()

    # 0-15s: Normal
    probs, state = runner.get_prediction_at_time(5.0)
    assert state == "Normal"
    assert probs["normal"] > 0.80
    assert probs["seizure"] < 0.10

    # 15-25s: Pre-Seizure
    probs, state = runner.get_prediction_at_time(20.0)
    assert state == "Pre-Seizure"
    assert probs["pre_seizure"] >= 0.60

    # 25-60s: Acute Seizure
    probs, state = runner.get_prediction_at_time(45.0)
    assert state == "Seizure"
    assert probs["seizure"] >= 0.70

    # 60-75s: Recovery Normal
    probs, state = runner.get_prediction_at_time(68.0)
    assert state == "Normal"
    assert probs["normal"] > 0.80
    assert probs["seizure"] < 0.10


def test_build_demo_dataset():
    """Verify built demo dataset has 22 channels, valid float32 values, and no NaNs."""
    sim = ReplaySimulator()
    dataset = sim.build_demo_dataset()
    assert isinstance(dataset, np.ndarray)
    assert dataset.shape[1] == 22
    assert dataset.shape[0] > 0
    assert dataset.dtype == np.float32
    assert not np.isnan(dataset).any()
    assert not np.isinf(dataset).any()


def test_stop_session():
    """Verify stopping replay cleans up session state."""
    sim = ReplaySimulator()
    res = sim.start_replay("rec-normal-baseline")
    sess_id = res["session_id"]
    assert sim.active_sessions[sess_id]["is_running"] is True

    sim.stop_session(sess_id)
    assert sim.active_sessions[sess_id]["is_running"] is False
