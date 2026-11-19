"""
backend/tests/test_synthetic.py
Unit tests for the synthetic EEG generator.
"""
import pytest
import numpy as np
from app.synthetic import generate_synthetic_window

def test_synthetic_normal_shape_and_range():
    """Verify normal synthetic window is (1280, 22) with 10-25 uV amplitude."""
    w = generate_synthetic_window("normal", seed=42)
    assert w.shape == (1280, 22)
    assert w.dtype == np.float32
    assert not np.isnan(w).any()
    rms = np.sqrt(np.mean(w**2))
    assert 5.0 <= rms <= 30.0

def test_synthetic_preseizure_shape_and_range():
    """Verify pre-seizure synthetic window is (1280, 22) with 30-70 uV amplitude."""
    w = generate_synthetic_window("pre_seizure", seed=42)
    assert w.shape == (1280, 22)
    assert w.dtype == np.float32
    rms = np.sqrt(np.mean(w**2))
    assert 20.0 <= rms <= 80.0

def test_synthetic_seizure_shape_and_range():
    """Verify seizure synthetic window is (1280, 22) with 150-300 uV amplitude."""
    w = generate_synthetic_window("seizure", seed=42)
    assert w.shape == (1280, 22)
    assert w.dtype == np.float32
    rms = np.sqrt(np.mean(w**2))
    assert 100.0 <= rms <= 400.0

def test_synthetic_seed_reproducibility():
    """Verify that same seed produces bit-for-bit identical arrays."""
    w1 = generate_synthetic_window("normal", seed=123)
    w2 = generate_synthetic_window("normal", seed=123)
    np.testing.assert_array_equal(w1, w2)

def test_synthetic_invalid_state():
    """Verify ValueError on unknown state."""
    with pytest.raises(ValueError, match="Unknown synthetic state"):
        generate_synthetic_window("invalid_state_name")

def test_synthetic_states_model_alignment():
    """Verify that generated synthetic windows align with ModelRunner predictions (>= 80% accuracy per class)."""
    import os
    from app.ml_runner import ModelRunner
    from app.config import settings

    model_path = settings.MODEL_PATH
    if not os.path.exists(model_path):
        pytest.skip("Model binary tusz_3class_eeg_model.h5 not found for alignment test")

    runner = ModelRunner(model_path)
    runner.warmup()

    test_states = [
        ("normal", "Normal"),
        ("pre_seizure", "Pre-Seizure"),
        ("seizure", "Seizure"),
    ]

    for state_arg, expected_label in test_states:
        preds = []
        for seed in range(20):
            window = generate_synthetic_window(state_arg, seed=seed)
            res = runner.predict(window)
            preds.append(res["prediction"])

        correct = preds.count(expected_label)
        accuracy = correct / len(preds)
        assert accuracy >= 0.80, (
            f"Synthetic state '{state_arg}' alignment below 80%: {correct}/20 correct ({accuracy * 100:.1f}%). "
            f"Predictions: {preds}"
        )
