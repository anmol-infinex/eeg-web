"""
backend/tests/test_ml_runner.py
Unit tests for ModelRunner class.
"""
import concurrent.futures
import pytest
import numpy as np
from app.ml_runner import ModelRunner
from app.config import settings

@pytest.fixture(scope="module")
def runner():
    r = ModelRunner(settings.MODEL_PATH)
    r.warmup()
    return r

def test_model_runner_loaded(runner):
    """Verify model loaded status and shapes."""
    assert runner._is_loaded is True
    assert runner._is_warm is True
    info = runner.info()
    assert info["model_name"] == "TUSZ_EEG_DEEP_CNN"
    assert info["channels"] == 22
    assert info["window_samples"] == 1280

def test_model_runner_predict_valid_shape_2d(runner):
    """Verify prediction on a valid (1280, 22) 2D array."""
    dummy = np.random.randn(1280, 22).astype(np.float32)
    res = runner.predict(dummy)
    assert "prediction" in res
    assert res["prediction"] in ["Normal", "Pre-Seizure", "Seizure"]
    assert "probabilities" in res
    probs = res["probabilities"]
    prob_sum = probs["normal"] + probs["pre_seizure"] + probs["seizure"]
    assert abs(prob_sum - 1.0) < 1e-4

def test_model_runner_predict_valid_shape_3d(runner):
    """Verify prediction on a valid (1, 1280, 22) 3D batch array."""
    dummy = np.random.randn(1, 1280, 22).astype(np.float32)
    res = runner.predict(dummy)
    assert res["prediction"] in ["Normal", "Pre-Seizure", "Seizure"]

def test_model_runner_invalid_dimensions(runner):
    """Verify ValueError on incorrect shape dimensions."""
    with pytest.raises(ValueError, match="Expected 2D input shape"):
        runner.predict(np.zeros((1000, 22), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected 2D input shape"):
        runner.predict(np.zeros((1280, 16), dtype=np.float32))

def test_model_runner_nan_rejection(runner):
    """Verify ValueError when input array contains NaN values."""
    arr = np.zeros((1280, 22), dtype=np.float32)
    arr[10, 5] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        runner.predict(arr)

def test_model_runner_inf_rejection(runner):
    """Verify ValueError when input array contains Inf values."""
    arr = np.zeros((1280, 22), dtype=np.float32)
    arr[20, 8] = np.inf
    with pytest.raises(ValueError, match="contains Inf"):
        runner.predict(arr)

def test_model_runner_thread_safety(runner):
    """Verify thread-safety by launching 10 concurrent inference calls simultaneously."""
    dummy = np.random.randn(1280, 22).astype(np.float32)

    def call_predict():
        return runner.predict(dummy)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(call_predict) for _ in range(10)]
        results = [f.result() for f in futures]

    assert len(results) == 10
    for r in results:
        assert r["prediction"] in ["Normal", "Pre-Seizure", "Seizure"]

def test_model_runner_zero_samples_batch(runner):
    """Verify ValueError when 3D batch contains 0 samples."""
    empty_batch = np.zeros((0, 1280, 22), dtype=np.float32)
    with pytest.raises(ValueError, match="Input batch contains 0 samples"):
        runner.predict(empty_batch)
