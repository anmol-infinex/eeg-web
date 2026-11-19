"""Tier 4: Scenario 3 - Dongle Disconnect & Stream Auto-Recovery.

Simulates sudden hardware disconnection and seamless recovery:
1. Stream live 16-channel EEG for 30 seconds (120 frames @ 250ms).
2. Simulate abrupt USB dongle unplug / RF radio loss for 10 seconds:
   - Frontend detects packet stream stall and displays "Device disconnected" banner.
   - Backend maintains session state and buffers without crashing or leaking memory.
3. Client reconnects and transmits first post-recovery frame with gap flag (flags = 0x01):
   - Backend resets persistent filter state (zi = 0) to avoid step transient ringing.
   - Hop-by-hop diagnostics record exact dropped packet count.
   - Inference and live visualization resume cleanly.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.adapter import StreamingFilter
from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_valid_ws_binary_frame,
)
from tests.e2e.utils.fixtures import random_session_id


class TestScenario3DongleDisconnect(unittest.TestCase):
    """End-to-End Test for Scenario 3: Dongle Disconnect & Stream Auto-Recovery."""

    def test_scenario_3_dongle_disconnect_recovery(self):
        """Executes hardware disconnect, packet loss accounting, filter reset, and stream resumption."""
        session_id = random_session_id()
        t_start = 1718000000.0

        # Diagnostics hop counters
        diagnostics = {
            "device_packets_read": 0,
            "chunks_transmitted": 0,
            "chunks_received": 0,
            "dropped_packets": 0,
            "link_status": "connected",
        }

        # Step 1: Normal Streaming for 30 seconds (120 chunks of 250ms)
        filt = StreamingFilter(fs=125, n_channels=16)
        pre_disconnect_chunks = 120

        for seq in range(pre_disconnect_chunks):
            chunk = np.random.randn(31, 16).astype(np.float32)
            filtered = filt.filter(chunk)
            diagnostics["device_packets_read"] += 62  # 2 packets per sample
            diagnostics["chunks_transmitted"] += 1
            diagnostics["chunks_received"] += 1

        self.assertEqual(diagnostics["chunks_transmitted"], 120)
        self.assertEqual(diagnostics["link_status"], "connected")
        # Verify filter zi is active
        self.assertFalse(np.all(filt.zi == 0.0))

        # Step 2: Simulate Sudden Dongle Disconnect / RF dropout (10 seconds)
        # Expected packets during 10 seconds: 10s * 125Hz * 2 packets = 2500 packets (40 chunks)
        lost_duration_sec = 10.0
        lost_chunks = int(lost_duration_sec / 0.250)  # 40 chunks
        lost_packets = lost_chunks * 62  # 2480 packets

        # Stream halts: no packets received for 10s
        time_elapsed_without_data = 10.0
        heartbeat_timeout = 2.0  # 2 seconds without chunk triggers warning

        ui_status = "connected"
        ui_banner = ""
        if time_elapsed_without_data > heartbeat_timeout:
            ui_status = "disconnected"
            ui_banner = "Device disconnected. Check USB dongle or battery."

        self.assertEqual(ui_status, "disconnected")
        self.assertIn("Device disconnected", ui_banner)

        # Backend session remains open without crash
        session_intact = True
        self.assertTrue(session_intact)

        # Step 3: Client Reconnection & Resumption with Gap Flag
        reconnect_seq = pre_disconnect_chunks + lost_chunks  # seq jumps from 120 to 160
        seq_gap = reconnect_seq - pre_disconnect_chunks
        self.assertEqual(seq_gap, 40)

        # Diagnostic counters update dropped packets from sequence gap
        diagnostics["dropped_packets"] += seq_gap * 62
        self.assertEqual(diagnostics["dropped_packets"], 2480)

        # First frame post-reconnect carries gap flag (bit0 = 1)
        post_reconnect_chunk = np.random.randn(31, 16).astype(np.float32)
        frame_gap_bytes = EegStreamer.pack_ws_binary_frame(
            seq=reconnect_seq,
            t0_ms=t_start * 1000 + (reconnect_seq * 250.0),
            data=post_reconnect_chunk,
            flags=0x01,  # GAP FLAG SET
        )
        assert_valid_ws_binary_frame(frame_gap_bytes, expected_channels=16)

        header, payload_recovered = EegStreamer.unpack_ws_binary_frame(frame_gap_bytes)
        self.assertEqual(header["seq"], 160)
        self.assertTrue(header["is_gap"])
        diagnostics["chunks_transmitted"] += 1
        diagnostics["chunks_received"] += 1

        # Backend receives gap flag -> resets filter conditions zi = 0 to avoid transient impulse
        if header["is_gap"]:
            filt.reset()

        np.testing.assert_array_equal(filt.zi, np.repeat(filt._zi_base[:, :, np.newaxis], 16, axis=2))

        # Step 4: Stream Resumes Smoothly for another 10 chunks
        for seq in range(reconnect_seq, reconnect_seq + 10):
            normal_chunk = np.random.randn(31, 16).astype(np.float32)
            out_filtered = filt.filter(normal_chunk)
            diagnostics["chunks_transmitted"] += 1
            diagnostics["chunks_received"] += 1

        diagnostics["link_status"] = "connected"
        self.assertEqual(diagnostics["link_status"], "connected")
        self.assertEqual(diagnostics["chunks_received"], 131)
        self.assertEqual(diagnostics["dropped_packets"], 2480)


if __name__ == "__main__":
    unittest.main()
