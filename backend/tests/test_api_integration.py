"""
backend/tests/test_api_integration.py
Integration tests for REST endpoints and authentication.
Direct endpoint invocation avoids AnyIO TestClient socket restrictions in sandboxed environments.
"""
import pytest
import numpy as np
from fastapi import HTTPException

from app.main import (
    app,
    verify_access_key,
    predict_window,
    get_synthetic_window,
    create_new_session_endpoint,
    get_session_details_endpoint,
    stop_session_endpoint,
    delete_session_endpoint,
    get_clinical_report_endpoint,
    get_eeg_trace_png_endpoint,
    get_replay_recordings_endpoint,
    start_replay_endpoint,
    stop_replay_endpoint,
    start_demo_endpoint,
    PredictRequest,
    CreateSessionRequest,
    ReplayStartRequest,
)
from app.config import settings
from app.ml_runner import ModelRunner
from app.synthetic import generate_synthetic_window
from app.simulator import ReplaySimulator
from app.ws_gateway import WsConnectionManager
from app.db import init_db


@pytest.fixture(scope="module", autouse=True)
def setup_app_state():
    """Set up app state before testing."""
    init_db()
    runner = ModelRunner(settings.MODEL_PATH)
    runner.warmup()
    app.state.model_runner = runner
    app.state.ws_manager = WsConnectionManager()
    app.state.simulator = ReplaySimulator()
    yield
    app.state.model_runner = None


def test_predict_unauthorized():
    """Verify verify_access_key raises 401 when ACCESS_KEY is omitted."""
    with pytest.raises(HTTPException) as exc_info:
        verify_access_key(None, None)
    assert exc_info.value.status_code == 401
    assert "Missing required access key" in exc_info.value.detail


def test_predict_invalid_key():
    """Verify verify_access_key raises 401 when wrong ACCESS_KEY is provided."""
    with pytest.raises(HTTPException) as exc_info:
        verify_access_key("wrong-secret-key", None)
    assert exc_info.value.status_code == 401
    assert "Invalid access key" in exc_info.value.detail


def test_predict_valid_key():
    """Verify verify_access_key succeeds with valid key in header or query."""
    assert verify_access_key(settings.ACCESS_KEY, None) == settings.ACCESS_KEY
    assert verify_access_key(None, settings.ACCESS_KEY) == settings.ACCESS_KEY


def test_predict_success():
    """Verify predict_window succeeds with 256Hz 22ch synthetic window."""
    synthetic = generate_synthetic_window("normal", seed=42)
    req = PredictRequest(eeg_data=synthetic.tolist(), fs=256)
    res = predict_window(req)
    assert res["prediction"] in ["Normal", "Pre-Seizure", "Seizure"]
    assert "probabilities" in res
    probs = res["probabilities"]
    assert "normal" in probs
    assert "pre_seizure" in probs
    assert "seizure" in probs


def test_predict_with_16ch_adaptation():
    """Verify predict_window accepts 16ch 125Hz data and adapts automatically."""
    raw_16ch = [[0.0] * 16 for _ in range(625)]
    req = PredictRequest(eeg_data=raw_16ch, fs=125)
    res = predict_window(req)
    assert res["prediction"] in ["Normal", "Pre-Seizure", "Seizure"]


def test_predict_empty_batch_rejection():
    """Verify predict_window rejects empty batch with HTTP 400."""
    req = PredictRequest(eeg_data=[], fs=256)
    with pytest.raises(HTTPException) as exc_info:
        predict_window(req)
    assert exc_info.value.status_code == 400
    assert "Empty EEG data array" in exc_info.value.detail


def test_predict_nan_rejection():
    """Verify predict_window rejects NaN values with HTTP 400 without poisoning filter."""
    nan_data = [[float("nan")] * 22 for _ in range(1280)]
    req = PredictRequest(eeg_data=nan_data, fs=256)
    with pytest.raises(HTTPException) as exc_info:
        predict_window(req)
    assert exc_info.value.status_code == 400


def test_synthetic_window_generation():
    """Verify get_synthetic_window returns valid 1280x22 array."""
    res = get_synthetic_window(state="normal")
    assert res["state"] == "normal"
    assert res["fs"] == 256
    assert res["n_samples"] == 1280
    assert res["n_channels"] == 22
    assert len(res["data"]) == 1280
    assert len(res["data"][0]) == 22


def test_session_rest_endpoints():
    """Verify sessions CRUD endpoints."""
    # Create session
    create_req = CreateSessionRequest(source="device", sample_rate=256, notes="Test session")
    sess = create_new_session_endpoint(create_req)
    sess_id = sess["id"]
    assert sess["status"] == "active"

    # Get details
    details = get_session_details_endpoint(sess_id)
    assert details["id"] == sess_id

    # Stop session
    stopped = stop_session_endpoint(sess_id)
    assert stopped["status"] == "stopped"

    # Get report (once stopped)
    report_res = get_clinical_report_endpoint(sess_id)
    assert report_res.status_code == 200
    assert sess_id in report_res.body.decode("utf-8")

    # Get EEG PNG thumbnail
    png_res = get_eeg_trace_png_endpoint(sess_id)
    assert png_res.status_code == 200
    assert png_res.body.startswith(b"\x89PNG\r\n\x1a\n")

    # Delete session
    del_res = delete_session_endpoint(sess_id)
    assert del_res["status"] == "deleted"


def test_replay_and_demo_endpoints():
    """Verify replay and demo endpoints."""
    # List recordings
    recs = get_replay_recordings_endpoint()
    assert len(recs) >= 2

    # Start replay
    rep_req = ReplayStartRequest(recording_id="preset_seizure_demo", speed=2.0)
    rep_res = start_replay_endpoint(rep_req)
    assert rep_res["status"] == "started"
    assert rep_res["simulated"] is True
    assert rep_res["badge"] == "SIMULATED DATA"

    # Stop replay
    stop_res = stop_replay_endpoint()
    assert stop_res["status"] == "stopped"

    # Start demo
    demo_res = start_demo_endpoint()
    assert demo_res["status"] == "started"
    assert demo_res["badge"] == "SIMULATED DATA"
