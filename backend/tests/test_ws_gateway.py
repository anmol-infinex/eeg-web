"""
backend/tests/test_ws_gateway.py
Unit tests for WebSocket gateway framing, binary frame unpacker, isolated SessionState, and diagnostics.
"""
import struct
import numpy as np
import pytest
from app.ws_gateway import (
    unpack_binary_frame,
    BinaryHeader,
    SessionDiagnostics,
    SessionState,
    WsConnectionManager,
    ClientHello,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAX_FRAME_SIZE,
)


def test_binary_frame_packing_and_unpacking():
    """Verify standard 16-byte <IdHBB little-endian header and float32 payload unpacking."""
    seq = 42
    t0_ms = 1718000000123.45
    n_samples = 31
    n_channels = 16
    flags = 0x01  # Gap flag

    header_bytes = struct.pack(HEADER_FORMAT, seq, t0_ms, n_samples, n_channels, flags)
    assert len(header_bytes) == HEADER_SIZE

    raw_data = np.random.randn(n_samples, n_channels).astype(np.float32)
    payload_bytes = raw_data.tobytes()

    frame = header_bytes + payload_bytes
    header, payload = unpack_binary_frame(frame)

    assert header.seq == 42
    assert header.t0_ms == pytest.approx(t0_ms, rel=1e-6)
    assert header.n_samples == 31
    assert header.n_channels == 16
    assert header.flags == 0x01
    assert payload.shape == (31, 16)
    assert payload.dtype == np.float32
    np.testing.assert_allclose(payload, raw_data, atol=1e-5)


def test_binary_frame_truncated_header():
    """Verify rejection when byte buffer is shorter than 16 bytes."""
    with pytest.raises(ValueError, match="shorter than 16-byte header"):
        unpack_binary_frame(b"\x00" * 15)


def test_binary_frame_payload_mismatch():
    """Verify rejection when payload size does not match n_samples * n_channels * 4."""
    header_bytes = struct.pack(HEADER_FORMAT, 1, 1000.0, 31, 16, 0)
    truncated_payload = b"\x00" * 100  # Expected 31 * 16 * 4 = 1984 bytes

    with pytest.raises(ValueError, match="Payload size mismatch"):
        unpack_binary_frame(header_bytes + truncated_payload)


def test_binary_frame_nan_or_inf_rejection():
    """Verify rejection when payload contains NaN or Inf values."""
    # NaN payload
    nan_data = np.ones((10, 16), dtype=np.float32)
    nan_data[2, 3] = np.nan
    h_bytes = struct.pack(HEADER_FORMAT, 1, 1000.0, 10, 16, 0)
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        unpack_binary_frame(h_bytes + nan_data.tobytes())

    # Inf payload
    inf_data = np.ones((10, 16), dtype=np.float32)
    inf_data[5, 2] = np.inf
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        unpack_binary_frame(h_bytes + inf_data.tobytes())


def test_binary_frame_oversized_rejection():
    """Verify rejection when frame exceeds 64 KB."""
    # 65537 bytes
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        unpack_binary_frame(b"\x00" * (MAX_FRAME_SIZE + 1))


def test_session_diagnostics_counters():
    """Verify diagnostics tracks frames received, dropped frames, and health status."""
    diag = SessionDiagnostics()
    assert diag.health == "ok"
    assert diag.dropped_chunks == 0

    # Receive sequential frames 0, 1, 2
    for s in range(3):
        diag.record_received_frame(s, 1984, 0)
    assert diag.hop3_frames_received == 3
    assert diag.dropped_chunks == 0
    assert diag.to_dict()["health"] == "ok"

    # Gap: receive frame 6 (missing 3, 4, 5 -> gap = 3)
    diag.record_received_frame(6, 1984, 1)
    assert diag.dropped_chunks == 3
    assert diag.to_dict()["health"] == "degraded"
    assert diag.to_dict()["dropped_chunks"] == 3


def test_session_state_rolling_buffer():
    """Verify rolling buffer accumulates chunks and returns (1280, 22) window once full."""
    sess = SessionState(session_id="test-sess", fs=256)
    assert sess.get_latest_window() is None

    # Ingest 10 chunks of 64 samples (640 samples < 1280)
    for _ in range(10):
        chunk = np.ones((64, 22), dtype=np.float32)
        sess.append_resampled(chunk)
    assert sess.total_buffered_samples == 640
    assert sess.get_latest_window() is None

    # Ingest 10 more chunks (total 1280 samples)
    for _ in range(10):
        chunk = np.ones((64, 22), dtype=np.float32) * 2.0
        sess.append_resampled(chunk)
    assert sess.total_buffered_samples == 1280

    window = sess.get_latest_window()
    assert window is not None
    assert window.shape == (1280, 22)
    assert window.dtype == np.float32


def test_session_state_buffer_overflow_cap():
    """Verify rolling buffer caps memory usage at max_buffer_samples (15,360 samples = 60s)."""
    sess = SessionState(session_id="test-cap", fs=256)
    sess.max_buffer_samples = 500  # Smaller cap for unit testing

    for _ in range(10):
        sess.append_resampled(np.ones((100, 22), dtype=np.float32))

    # Total ingested 1000, cap is 500
    assert sess.total_buffered_samples <= 500
