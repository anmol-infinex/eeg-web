"""Specialized E2E Assertion Helpers for Seizure-Sentry Tests.

Provides reusable assertions for probability normalization, montage shapes,
latency thresholds, binary frame layouts, and signal quality metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Union

import numpy as np


def assert_probability_distribution(
    probs: Union[Dict[str, float], Sequence[float], np.ndarray],
    tol: float = 1e-4,
    context: str = "",
) -> None:
    """Asserts that a probability collection is valid and sums to 1.0 within tolerance.

    Accepts dict (e.g. {'normal': 0.7, 'pre_seizure': 0.2, 'seizure': 0.1}) or array/list.
    """
    if isinstance(probs, dict):
        values = list(probs.values())
        keys = list(probs.keys())
    else:
        values = list(probs)
        keys = [f"index_{i}" for i in range(len(values))]

    if not values:
        raise AssertionError(f"Empty probability distribution {context}")

    for k, v in zip(keys, values):
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            raise AssertionError(f"Probability {k}={val} is NaN or Inf {context}")
        if val < -tol or val > (1.0 + tol):
            raise AssertionError(f"Probability {k}={val} is outside [0.0, 1.0] {context}")

    total = sum(float(v) for v in values)
    if abs(total - 1.0) > tol:
        raise AssertionError(
            f"Probabilities sum to {total:.6f}, expected 1.0 +/- {tol} {context}. Values: {probs}"
        )


def assert_montage_shape(
    data: np.ndarray,
    expected_channels: int = 22,
    expected_samples: int = 1280,
    context: str = "",
) -> None:
    """Asserts that the EEG tensor matches expected sample and channel dimensions."""
    if not isinstance(data, np.ndarray):
        raise AssertionError(f"Expected numpy ndarray, got {type(data)} {context}")

    shape = data.shape
    # Support (samples, channels) or (batch, samples, channels)
    if data.ndim == 2:
        actual_samples, actual_channels = shape[0], shape[1]
    elif data.ndim == 3:
        actual_samples, actual_channels = shape[1], shape[2]
    else:
        raise AssertionError(f"Expected 2D or 3D tensor, got shape {shape} {context}")

    if actual_samples != expected_samples or actual_channels != expected_channels:
        raise AssertionError(
            f"Montage shape mismatch {context}: got ({actual_samples}, {actual_channels}), "
            f"expected ({expected_samples}, {expected_channels})"
        )


def assert_latency_within(
    latency_ms: float,
    max_allowed_ms: float,
    context: str = "",
) -> None:
    """Asserts that measured processing or network latency does not exceed threshold."""
    if latency_ms > max_allowed_ms:
        raise AssertionError(
            f"Latency threshold exceeded {context}: measured {latency_ms:.2f}ms > max {max_allowed_ms:.2f}ms"
        )


def assert_status_code(
    actual_code: int,
    expected_code: int,
    message: str = "",
) -> None:
    """Asserts that an HTTP or WebSocket status code matches expectation."""
    if actual_code != expected_code:
        detail = f" - {message}" if message else ""
        raise AssertionError(f"Expected status {expected_code}, got {actual_code}{detail}")


def assert_valid_openbci_packet(packet_bytes: bytes) -> None:
    """Asserts that a raw byte buffer conforms to the 33-byte OpenBCI packet specification."""
    if len(packet_bytes) != 33:
        raise AssertionError(f"Expected 33 bytes, got {len(packet_bytes)}")
    if packet_bytes[0] != 0xA0:
        raise AssertionError(f"Invalid header byte: expected 0xA0, got 0x{packet_bytes[0]:02x}")
    stop_byte = packet_bytes[32]
    if stop_byte not in (0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6):
        raise AssertionError(f"Invalid stop byte: got 0x{stop_byte:02x}")


def assert_valid_ws_binary_frame(
    frame_bytes: bytes,
    expected_channels: int = 16,
) -> None:
    """Asserts that a WebSocket binary frame has a valid 16-byte header and matching float32 payload."""
    if len(frame_bytes) < 16:
        raise AssertionError(f"Frame length {len(frame_bytes)} is shorter than 16-byte header")

    import struct

    seq, t0_ms, n_samples, n_channels, flags = struct.unpack("<IdHBB", frame_bytes[:16])

    if expected_channels is not None and n_channels != expected_channels:
        raise AssertionError(f"Expected {expected_channels} channels, got {n_channels}")

    expected_payload_bytes = n_samples * n_channels * 4
    actual_payload_bytes = len(frame_bytes) - 16
    if actual_payload_bytes != expected_payload_bytes:
        raise AssertionError(
            f"Payload length mismatch: expected {expected_payload_bytes} bytes "
            f"({n_samples}x{n_channels}x4), got {actual_payload_bytes}"
        )


def assert_signal_quality(
    signal: np.ndarray,
    expected_quality: str,
    fs: int = 125,
    context: str = "",
) -> None:
    """Evaluates per-channel signal metrics and asserts match with expected quality label.

    Categories:
      - 'flat': std < 0.5 uV
      - 'railed': max abs > 180,000 uV
      - 'noisy': 50Hz mains power component or broadband RMS > 300 uV
      - 'good': normal physiological EEG (std 10-60 uV, peak < 150 uV)
    """
    arr = np.asarray(signal)
    max_abs = float(np.max(np.abs(arr)))
    std = float(np.std(arr))

    # Spectral evaluation for 50Hz mains noise
    n = len(arr)
    fft_vals = np.abs(np.fft.rfft(arr, axis=0))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    idx_50 = np.argmin(np.abs(freqs - 50.0))
    mains_peak = float(np.max(fft_vals[max(0, idx_50 - 2) : idx_50 + 3]))
    total_power = float(np.sum(fft_vals))
    mains_ratio = mains_peak / (total_power + 1e-6)

    determined = "good"
    if max_abs > 180000.0:
        determined = "railed"
    elif std < 0.5:
        determined = "flat"
    elif mains_ratio > 0.25 or std > 300.0:
        determined = "noisy"

    exp_lower = expected_quality.lower().strip()
    if determined != exp_lower:
        raise AssertionError(
            f"Signal quality mismatch {context}: determined '{determined}', "
            f"expected '{exp_lower}' (std={std:.2f}uV, max_abs={max_abs:.2f}uV, mains_ratio={mains_ratio:.3f})"
        )


def assert_prediction_schema(pred_dict: Dict[str, Any]) -> None:
    """Validates the structure of a model prediction broadcast dictionary."""
    if not isinstance(pred_dict, dict):
        raise AssertionError(f"Expected prediction dict, got {type(pred_dict)}")

    # Verify probability field
    probs = pred_dict.get("probs") or pred_dict.get("probabilities")
    if probs is None:
        raise AssertionError(f"Missing 'probs' in prediction payload: {pred_dict}")

    for cl in ("normal", "pre_seizure", "seizure"):
        if cl not in probs:
            raise AssertionError(f"Missing '{cl}' probability class in {probs}")

    assert_probability_distribution(probs)


def assert_alert_schema(alert_dict: Dict[str, Any]) -> None:
    """Validates the structure of an alert broadcast dictionary."""
    if not isinstance(alert_dict, dict):
        raise AssertionError(f"Expected alert dict, got {type(alert_dict)}")

    for req_key in ("id", "level", "started_at", "peak_prob"):
        if req_key not in alert_dict:
            raise AssertionError(f"Missing required alert field '{req_key}' in {alert_dict}")

    level = alert_dict["level"]
    if level not in ("INFO", "WARNING", "CRITICAL"):
        raise AssertionError(f"Invalid alert level: '{level}' (expected INFO, WARNING, or CRITICAL)")
