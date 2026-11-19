"""E2E Test Configuration and Constants.

Provides centralized URLs, keys, channel maps, sample rates, and timeouts
for the Seizure-Sentry opaque-box E2E test suite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class E2EConfig:
    """Runtime configuration for E2E tests."""

    # Backend target URLs
    target_url: str = field(
        default_factory=lambda: os.getenv("E2E_TARGET_URL", "http://localhost:8000").rstrip("/")
    )
    ws_url: str = field(
        default_factory=lambda: os.getenv("E2E_WS_URL", "ws://localhost:8000").rstrip("/")
    )

    # Authentication
    access_key: str = field(
        default_factory=lambda: os.getenv("E2E_ACCESS_KEY", "test-access-key-xyz")
    )

    # Timeouts (seconds)
    request_timeout: float = 10.0
    ws_connect_timeout: float = 5.0
    ws_frame_timeout: float = 5.0

    # Hardware & Protocol specifications
    cyton_fs_hz: int = 125
    model_fs_hz: int = 256
    samples_per_250ms_chunk: int = 31  # 31 samples @ 125Hz = 248ms ~ 250ms
    model_window_samples: int = 1280    # 5.0 seconds @ 256Hz
    model_window_channels: int = 22

    # Cyton 16-channel layout
    cyton_channels: List[str] = field(
        default_factory=lambda: [
            "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8"
        ]
    )

    # Model 22-channel TCP bipolar montage layout
    tcp_montage_channels: List[str] = field(
        default_factory=lambda: [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "FZ-CZ", "CZ-PZ",
            "T7-FT9", "FT9-FT10", "FT10-T8", "ECG"
        ]
    )

    # OpenBCI hardware constants
    openbci_packet_size: int = 33
    openbci_start_byte: int = 0xA0
    openbci_stop_bytes: tuple = (0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6)
    openbci_gain: int = 24
    openbci_vref: float = 4.5
    # Scale factor: 4.5 / 24 / (2^23 - 1) * 1e6 uV = 0.02235174... uV/count
    openbci_uv_per_count: float = 0.02235174

    # WS Binary frame header layout: uint32 seq, float64 t0_ms, uint16 n_samples, uint8 n_channels, uint8 flags
    ws_binary_header_size: int = 16
    ws_binary_header_fmt: str = "<IdHBB"


# Global configuration instance for test modules
CONFIG = E2EConfig()


def set_config(
    target_url: str | None = None,
    ws_url: str | None = None,
    access_key: str | None = None,
) -> E2EConfig:
    """Update global configuration."""
    global CONFIG
    if target_url is not None:
        CONFIG.target_url = target_url.rstrip("/")
    if ws_url is not None:
        CONFIG.ws_url = ws_url.rstrip("/")
    if access_key is not None:
        CONFIG.access_key = access_key
    return CONFIG
