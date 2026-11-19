"""Reusable Test Fixtures and Synthetic Generators for E2E Tests.

Provides deterministic test tensors, OpenBCI byte stream fixtures,
device profile specifications, session UUID generators, and handshake payloads.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG


def random_session_id() -> str:
    """Generates a random UUID4 string for test sessions."""
    return str(uuid.uuid4())


def random_access_key(length: int = 32) -> str:
    """Generates a secure random hexadecimal access key."""
    return secrets.token_hex(length // 2)


def get_openbci_cyton_profile() -> Dict[str, Any]:
    """Returns the standard 16-channel OpenBCI Cyton+Daisy device profile."""
    return {
        "profile_id": "openbci-cyton-daisy",
        "name": "OpenBCI Cyton + Daisy",
        "fs_hz": 125,
        "unit": "uV",
        "channels": list(CONFIG.cyton_channels),
        "channels_note": "16-channel scalp montage mapped to pins 1-16",
        "scale": {"gain": 24, "uv_per_count": CONFIG.openbci_uv_per_count},
    }


def get_incompatible_profile() -> Dict[str, Any]:
    """Returns an incompatible 4-channel device profile (e.g. Muse headband)."""
    return {
        "profile_id": "muse-headband-4ch",
        "name": "Muse 2 / S Consumer Headband",
        "fs_hz": 256,
        "unit": "uV",
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "scale": {"gain": 1, "uv_per_count": 0.488},
    }


def generate_openbci_raw_stream(
    n_samples: int = 31,
    seed: Optional[int] = 42,
) -> bytes:
    """Generates a stream of alternating board and daisy 33-byte OpenBCI packets.

    Total bytes returned: n_samples * 2 * 33 bytes.
    """
    rng = np.random.default_rng(seed)
    buffer = bytearray()
    for i in range(n_samples):
        # 16-channel physiological sample
        sample_16 = rng.normal(0.0, 20.0, 16).astype(np.float32)
        pkt_board, pkt_daisy = EegStreamer.pack_daisy_pair(
            sample_counter_pair_start=(i * 2) % 256,
            channels_16_uv=sample_16,
        )
        buffer.extend(pkt_board)
        buffer.extend(pkt_daisy)
    return bytes(buffer)


def generate_mock_binary_frame(
    seq: int = 0,
    n_samples: int = 31,
    n_channels: int = 16,
    state: str = "normal",
    flags: int = 0,
    seed: Optional[int] = 42,
) -> bytes:
    """Generates a single 16-byte header + float32 payload WebSocket binary frame."""
    data = EegStreamer.generate_signal(
        state=state,
        duration_sec=(n_samples / 125.0),
        fs=125,
        n_channels=n_channels,
        seed=seed,
    )
    if len(data) != n_samples:
        data = data[:n_samples, :n_channels]
    t0_ms = 1700000000000.0 + (seq * 250.0)
    return EegStreamer.pack_ws_binary_frame(seq=seq, t0_ms=t0_ms, data=data, flags=flags)


def golden_model_input_tensor(seed: int = 1337) -> np.ndarray:
    """Generates a deterministic (1, 1280, 22) float32 tensor for golden model inference verification."""
    rng = np.random.default_rng(seed)
    # Generate 1280 samples x 22 channels
    raw = rng.normal(0.0, 25.0, (1280, 22)).astype(np.float32)
    # Normalize per-channel: (x - mean) / std
    mean = np.mean(raw, axis=0, keepdims=True)
    std = np.std(raw, axis=0, keepdims=True) + 1e-6
    normalized = (raw - mean) / std
    # Add batch dimension: shape (1, 1280, 22)
    return np.expand_dims(normalized, axis=0).astype(np.float32)


def mock_client_hello(
    key: str = "test-access-key-xyz",
    role: str = "producer",
    source: str = "device",
) -> Dict[str, Any]:
    """Generates a standard client WebSocket hello handshake dictionary."""
    return {
        "type": "hello",
        "key": key,
        "role": role,
        "source": source,
        "profile_id": "openbci-cyton-daisy",
        "fs": CONFIG.cyton_fs_hz,
        "channels": list(CONFIG.cyton_channels),
    }


def mock_viewer_hello(token: str) -> Dict[str, Any]:
    """Generates a read-only viewer WebSocket hello handshake dictionary."""
    return {
        "type": "hello",
        "key": token,
        "role": "viewer",
        "source": "viewer",
    }
