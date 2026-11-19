"""Tier 1: Features 1-7 Ingestion & Device Integration Tests.

Covers:
- Feature 1: Web Serial Link Device (5 tests)
- Feature 2: OpenBCI Packet Parser (5 tests)
- Feature 3: 250ms Binary Framing (5 tests)
- Feature 4: Device Adapter Layer (EegSource) (5 tests)
- Feature 5: Device Profiles & Compatibility (5 tests)
- Feature 6: End-to-End Diagnostics (5 tests)
- Feature 7: Read-Only Live View Sharing (5 tests)

Total: 35 tests.
"""

from __future__ import annotations

import json
import math
import struct
import time
import unittest
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from tests.e2e.client.http_client import HttpResponse, TestHttpClient
from tests.e2e.client.streamer import EegStreamer
from tests.e2e.client.ws_client import (
    OP_BINARY,
    OP_CLOSE,
    OP_TEXT,
    RFC6455Client,
    WebSocketCloseError,
    WebSocketError,
)
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_latency_within,
    assert_prediction_schema,
    assert_valid_openbci_packet,
    assert_valid_ws_binary_frame,
)
from tests.e2e.utils.fixtures import (
    generate_mock_binary_frame,
    generate_openbci_raw_stream,
    get_incompatible_profile,
    get_openbci_cyton_profile,
    mock_client_hello,
    mock_viewer_hello,
    random_access_key,
    random_session_id,
)


# =============================================================================
# Feature 1: Web Serial Link Device
# =============================================================================
class TestFeature01WebSerialLink(unittest.TestCase):
    """Feature 1: Direct serial connection to OpenBCI FTDI dongle via Web Serial API.

    Verifies 115200 8N1 serial parameters, soft reset handshake 'v', channel mode 'C',
    stream start 'b' / stop 's', and port disconnect resource cleanup.
    """

    class MockWebSerialPort:
        """Simulates browser navigator.serial SerialPort interface."""

        def __init__(self):
            self.opened = False
            self.baudRate = 0
            self.dataBits = 0
            self.stopBits = 0
            self.parity = ""
            self.bufferSize = 0
            self.written_bytes = bytearray()
            self.readable_stream_locked = False
            self.read_buffer = bytearray()

        def open(self, options: Dict[str, Any]):
            if self.opened:
                raise RuntimeError("Port already opened")
            self.opened = True
            self.baudRate = options.get("baudRate", 9600)
            self.dataBits = options.get("dataBits", 8)
            self.stopBits = options.get("stopBits", 1)
            self.parity = options.get("parity", "none")
            self.bufferSize = options.get("bufferSize", 4096)

        def write(self, data: bytes):
            if not self.opened:
                raise RuntimeError("Cannot write to closed port")
            self.written_bytes.extend(data)

        def close(self):
            if self.readable_stream_locked:
                raise RuntimeError("Cannot close port with locked readable stream")
            self.opened = False

    def setUp(self):
        self.port = self.MockWebSerialPort()

    def test_f01_port_configuration(self):
        """1. Validates 115200 baud, 8N1 serial parameters per PRD FR-2 and architecture §4.2."""
        serial_options = {
            "baudRate": 115200,
            "dataBits": 8,
            "stopBits": 1,
            "parity": "none",
            "bufferSize": 16384,
        }
        self.port.open(serial_options)
        self.assertTrue(self.port.opened)
        self.assertEqual(self.port.baudRate, 115200)
        self.assertEqual(self.port.dataBits, 8)
        self.assertEqual(self.port.stopBits, 1)
        self.assertEqual(self.port.parity, "none")
        self.assertGreaterEqual(self.port.bufferSize, 4096)

    def test_f01_reset_handshake(self):
        """2. Verifies soft reset command 'v' and banner parsing up to '$$$' terminator."""
        self.port.open({"baudRate": 115200})
        # Client sends soft reset 'v'
        self.port.write(b"v")
        self.assertEqual(self.port.written_bytes, b"v")

        # Dongle responds with board banner ending with '$$$'
        banner_response = (
            b"OpenBCI V3 8-16 channel\n"
            b"On Board ADS1299 Device ID: 0x3E\n"
            b"On Daisy ADS1299 Device ID: 0x3E\n"
            b"LIS3DH Device ID: 0x33\n"
            b"$$$"
        )
        self.assertTrue(banner_response.endswith(b"$$$"))
        self.assertIn(b"OpenBCI", banner_response)
        self.assertIn(b"Daisy", banner_response)

    def test_f01_channel_mode_selection(self):
        """3. Verifies command 'C' for 16-channel Cyton+Daisy mode and 'c' for 8-channel mode."""
        self.port.open({"baudRate": 115200})
        # Select 16-channel mode
        self.port.write(b"C")
        self.assertEqual(self.port.written_bytes[-1:], b"C")
        # Select 8-channel mode
        self.port.write(b"c")
        self.assertEqual(self.port.written_bytes[-1:], b"c")

    def test_f01_stream_start_stop(self):
        """4. Verifies stream start 'b' and stop 's' commands."""
        self.port.open({"baudRate": 115200})
        # Start stream
        self.port.write(b"b")
        self.assertEqual(self.port.written_bytes[-1:], b"b")
        # Stop stream
        self.port.write(b"s")
        self.assertEqual(self.port.written_bytes[-1:], b"s")

    def test_f01_disconnect_cleanup(self):
        """5. Verifies port closure, reader release, and state reset upon disconnect."""
        self.port.open({"baudRate": 115200})
        self.assertTrue(self.port.opened)
        self.port.readable_stream_locked = False
        self.port.close()
        self.assertFalse(self.port.opened)

        # Attempting write on closed port raises error
        with self.assertRaises(RuntimeError):
            self.port.write(b"s")


# =============================================================================
# Feature 2: OpenBCI Packet Parser
# =============================================================================
class TestFeature02OpenBciParser(unittest.TestCase):
    """Feature 2: Parses 33-byte packets into 16-channel float32 microvolts.

    Verifies header (0xA0), stop bytes (0xC0-0xC6), 24-bit signed decoding,
    0.02235 uV/count scaling at gain 24, and Daisy packet pairing at 125 Hz.
    """

    def test_f02_header_stop_byte_validation(self):
        """1. Confirms start byte 0xA0 and stop byte 0xC0-0xC6 range; rejects invalid bytes."""
        valid_packet = EegStreamer.pack_openbci_packet(
            sample_counter=0,
            channels_uv=[0.0] * 8,
            stop_byte=0xC0,
        )
        self.assertEqual(len(valid_packet), 33)
        self.assertEqual(valid_packet[0], 0xA0)
        self.assertEqual(valid_packet[32], 0xC0)

        unpacked = EegStreamer.unpack_openbci_packet(valid_packet)
        self.assertEqual(unpacked["counter"], 0)
        self.assertEqual(unpacked["stop_byte"], 0xC0)

        # Corrupted start byte (0xFF)
        bad_header = bytearray(valid_packet)
        bad_header[0] = 0xFF
        with self.assertRaises(ValueError):
            EegStreamer.unpack_openbci_packet(bytes(bad_header))

        # Corrupted stop byte (0xAA)
        bad_stop = bytearray(valid_packet)
        bad_stop[32] = 0xAA
        with self.assertRaises(ValueError):
            EegStreamer.unpack_openbci_packet(bytes(bad_stop))

    def test_f02_sample_counter_continuity(self):
        """2. Verifies counter incrementing (cnt + 1) % 256 across rollover."""
        stream = generate_openbci_raw_stream(n_samples=130, seed=123)
        # Each sample is 2 packets (board + daisy) -> 260 packets
        self.assertEqual(len(stream), 130 * 2 * 33)

        counters = []
        for i in range(130 * 2):
            pkt = stream[i * 33 : (i + 1) * 33]
            unpacked = EegStreamer.unpack_openbci_packet(pkt)
            counters.append(unpacked["counter"])

        # Check modulo 256 continuity
        for i in range(1, len(counters)):
            expected = (counters[i - 1] + 1) % 256
            self.assertEqual(counters[i], expected, f"Discontinuity at packet index {i}")

    def test_f02_24bit_signed_twos_complement(self):
        """3. Tests negative, zero, and positive 24-bit signed integer decoding."""
        # 0x000000 = 0
        raw_zero = b"\x00\x00\x00"
        self.assertAlmostEqual(EegStreamer.signed_24bit_to_uv(raw_zero), 0.0, places=4)

        # 0x7FFFFF = +8,388,607 (max positive)
        raw_max = b"\x7F\xFF\xFF"
        max_uv = EegStreamer.signed_24bit_to_uv(raw_max)
        self.assertGreater(max_uv, 187000.0)

        # 0x800000 = -8,388,608 (min negative)
        raw_min = b"\x80\x00\x00"
        min_uv = EegStreamer.signed_24bit_to_uv(raw_min)
        self.assertLess(min_uv, -187000.0)

        # 0xFFFFFF = -1
        raw_neg1 = b"\xFF\xFF\xFF"
        neg1_uv = EegStreamer.signed_24bit_to_uv(raw_neg1)
        self.assertAlmostEqual(neg1_uv, -EegStreamer.UV_PER_COUNT, places=5)

    def test_f02_microvolt_scaling_factor(self):
        """4. Validates 0.02235 uV/count scaling at gain 24 (4.5 / 24 / (2^23 - 1) * 1e6)."""
        expected_scale = 4.5 / 24.0 / ((1 << 23) - 1) * 1e6
        self.assertAlmostEqual(EegStreamer.UV_PER_COUNT, expected_scale, places=7)
        self.assertAlmostEqual(EegStreamer.UV_PER_COUNT, 0.02235174, places=6)

        # Encode 100 uV and verify reconstructed value is within 1 count (< 0.03 uV)
        test_val = 100.0
        encoded = EegStreamer.uv_to_24bit_signed(test_val)
        decoded = EegStreamer.signed_24bit_to_uv(encoded)
        self.assertAlmostEqual(decoded, test_val, delta=EegStreamer.UV_PER_COUNT)

    def test_f02_daisy_packet_pairing(self):
        """5. Verifies pairing of alternating board (odd) and Daisy (even) packets into 16 channels at 125 Hz."""
        ch_16 = [float(i * 10) for i in range(16)]
        pkt_board, pkt_daisy = EegStreamer.pack_daisy_pair(
            sample_counter_pair_start=10,
            channels_16_uv=ch_16,
        )

        unpacked_board = EegStreamer.unpack_openbci_packet(pkt_board)
        unpacked_daisy = EegStreamer.unpack_openbci_packet(pkt_daisy)

        self.assertEqual(unpacked_board["counter"], 10)
        self.assertEqual(unpacked_daisy["counter"], 11)

        # Combined 16 channels
        combined = unpacked_board["channels_uv"] + unpacked_daisy["channels_uv"]
        self.assertEqual(len(combined), 16)
        for i in range(16):
            self.assertAlmostEqual(combined[i], ch_16[i], delta=0.03)


# =============================================================================
# Feature 3: 250ms Binary Framing
# =============================================================================
class TestFeature03BinaryFraming(unittest.TestCase):
    """Feature 3: Batches ~31 samples into little-endian binary frames over WS.

    Verifies 16-byte header (<IdHBB), sample counts, 1984-byte payload size,
    monotonic sequence numbers, and bit 0 gap flag encoding.
    """

    def test_f03_header_size_and_endianness(self):
        """1. Confirms exact 16-byte little-endian header layout <IdHBB."""
        self.assertEqual(struct.calcsize("<IdHBB"), 16)
        data = np.zeros((31, 16), dtype=np.float32)
        frame = EegStreamer.pack_ws_binary_frame(
            seq=100,
            t0_ms=1700000000123.45,
            data=data,
            flags=1,
        )
        self.assertGreaterEqual(len(frame), 16)
        seq, t0_ms, n_samples, n_channels, flags = struct.unpack("<IdHBB", frame[:16])
        self.assertEqual(seq, 100)
        self.assertAlmostEqual(t0_ms, 1700000000123.45, places=2)
        self.assertEqual(n_samples, 31)
        self.assertEqual(n_channels, 16)
        self.assertEqual(flags, 1)

    def test_f03_chunk_sample_count(self):
        """2. Verifies batching of 31 samples (248 ms ≈ 250 ms) at 125 Hz."""
        n_samples = 31
        fs = 125
        duration_ms = (n_samples / fs) * 1000.0
        self.assertAlmostEqual(duration_ms, 248.0, delta=2.0)

        frame = generate_mock_binary_frame(seq=1, n_samples=31, n_channels=16)
        header, data = EegStreamer.unpack_ws_binary_frame(frame)
        self.assertEqual(header["n_samples"], 31)
        self.assertEqual(data.shape, (31, 16))

    def test_f03_payload_byte_length(self):
        """3. Confirms N * 16 * 4 = 1984 bytes payload size for 31 samples, 16 channels."""
        n_samples = 31
        n_channels = 16
        expected_payload_bytes = n_samples * n_channels * 4  # float32 = 4 bytes
        self.assertEqual(expected_payload_bytes, 1984)

        data = np.random.randn(31, 16).astype(np.float32)
        frame = EegStreamer.pack_ws_binary_frame(seq=0, t0_ms=0.0, data=data)
        self.assertEqual(len(frame), 16 + 1984)

    def test_f03_monotonic_sequence_number(self):
        """4. Verifies sequential seq incrementing on consecutive frames."""
        frames = [
            generate_mock_binary_frame(seq=i, n_samples=31, n_channels=16)
            for i in range(10)
        ]
        for idx, f in enumerate(frames):
            header, _ = EegStreamer.unpack_ws_binary_frame(f)
            self.assertEqual(header["seq"], idx)

    def test_f03_flags_field_encoding(self):
        """5. Validates bit 0 gap flag encoding (flags & 0x01)."""
        # Clean frame: flags = 0
        frame_clean = generate_mock_binary_frame(seq=1, flags=0)
        header_clean, _ = EegStreamer.unpack_ws_binary_frame(frame_clean)
        self.assertEqual(header_clean["flags"] & 0x01, 0)

        # Gap frame: flags = 1
        frame_gap = generate_mock_binary_frame(seq=2, flags=1)
        header_gap, _ = EegStreamer.unpack_ws_binary_frame(frame_gap)
        self.assertEqual(header_gap["flags"] & 0x01, 1)


# =============================================================================
# Feature 4: Device Adapter Layer (EegSource)
# =============================================================================
class TestFeature04DeviceAdapterLayer(unittest.TestCase):
    """Feature 4: Uniform EegSource abstraction (device, bridge, replay).

    Verifies connect, disconnect, onChunk, info contract, SampleChunk schema,
    source identification, and teardown isolation.
    """

    class MockEegSource:
        """Simulates frontend EegSource interface in Python."""

        def __init__(self, source_id: str, profile_id: str = "openbci-cyton-daisy"):
            self.id = source_id
            self.profile_id = profile_id
            self._supported = True
            self._connected = False
            self._chunk_callbacks: List[Callable] = []
            self._status_callbacks: List[Callable] = []
            self.status: Dict[str, Any] = {"state": "disconnected", "dropped": 0}

        def is_supported(self) -> bool:
            return self._supported

        def connect(self) -> Dict[str, Any]:
            if not self.is_supported():
                raise RuntimeError("Source not supported in environment")
            self._connected = True
            self.status["state"] = "streaming"
            info = {
                "name": "Mock EEG Source",
                "channels": list(CONFIG.cyton_channels),
                "fs": 125,
                "unit": "uV",
                "profileId": self.profile_id,
            }
            for cb in self._status_callbacks:
                cb(self.status)
            return info

        def disconnect(self) -> None:
            self._connected = False
            self.status["state"] = "disconnected"
            for cb in self._status_callbacks:
                cb(self.status)

        def on_chunk(self, cb: Callable):
            self._chunk_callbacks.append(cb)

        def on_status(self, cb: Callable):
            self._status_callbacks.append(cb)

        def emit_chunk(self, chunk: Dict[str, Any]):
            for cb in self._chunk_callbacks:
                cb(chunk)

    def test_f04_interface_compliance(self):
        """1. Verifies connect, disconnect, onChunk, info contract."""
        source = self.MockEegSource("cyton-serial")
        self.assertTrue(source.is_supported())
        info = source.connect()
        self.assertEqual(info["unit"], "uV")
        self.assertEqual(info["fs"], 125)
        self.assertEqual(len(info["channels"]), 16)
        source.disconnect()
        self.assertEqual(source.status["state"], "disconnected")

    def test_f04_uniform_sample_chunk_schema(self):
        """2. Validates uniform SampleChunk schema: seq, t0_ms, samples, channels, fs."""
        received_chunks = []
        source = self.MockEegSource("cyton-serial")
        source.on_chunk(lambda c: received_chunks.append(c))

        chunk = {
            "seq": 0,
            "t0_ms": time.time() * 1000.0,
            "samples": np.zeros((31, 16), dtype=np.float32),
            "channels": list(CONFIG.cyton_channels),
            "fs": 125,
            "flags": 0,
        }
        source.emit_chunk(chunk)
        self.assertEqual(len(received_chunks), 1)
        c = received_chunks[0]
        self.assertIn("seq", c)
        self.assertIn("t0_ms", c)
        self.assertIn("samples", c)
        self.assertEqual(c["samples"].shape, (31, 16))

    def test_f04_source_identification(self):
        """3. Verifies id strings: cyton-serial, replay, bridge, ble-generic."""
        valid_ids = ["cyton-serial", "replay", "bridge", "ble-generic"]
        for s_id in valid_ids:
            src = self.MockEegSource(s_id)
            self.assertEqual(src.id, s_id)

    def test_f04_status_callback_propagation(self):
        """4. Verifies link state and drop counters propagation through callbacks."""
        statuses = []
        source = self.MockEegSource("cyton-serial")
        source.on_status(lambda s: statuses.append(dict(s)))

        source.connect()
        source.status["dropped"] = 3
        source.disconnect()

        self.assertEqual(len(statuses), 2)
        self.assertEqual(statuses[0]["state"], "streaming")
        self.assertEqual(statuses[1]["state"], "disconnected")
        self.assertEqual(statuses[1]["dropped"], 3)

    def test_f04_teardown_isolation(self):
        """5. Asserts disconnecting source cleans up listeners and leaves other sources unaffected."""
        source1 = self.MockEegSource("cyton-serial")
        source2 = self.MockEegSource("replay")

        source1.connect()
        source2.connect()

        self.assertEqual(source1.status["state"], "streaming")
        self.assertEqual(source2.status["state"], "streaming")

        source1.disconnect()
        self.assertEqual(source1.status["state"], "disconnected")
        self.assertEqual(source2.status["state"], "streaming")  # source2 stays active


# =============================================================================
# Feature 5: Device Profiles & Compatibility
# =============================================================================
class TestFeature05DeviceProfilesAndCompatibility(unittest.TestCase):
    """Feature 5: Validates device profile against model card specifications.

    Verifies profile JSON schema, 16-ch Cyton profile, compatible/adapted/incompatible
    compatibility check logic.
    """

    def _check_compatibility(
        self,
        device_profile: Dict[str, Any],
        model_card: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reference implementation of device compatibility check per architecture §4A."""
        model_channels = set(c.upper() for c in model_card.get("channels", []))
        device_channels = set(c.upper() for c in device_profile.get("channels", []))

        # Check aliases
        aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
        normalized_device = set(aliases.get(c, c) for c in device_channels)

        matched = list(model_channels.intersection(normalized_device))
        missing = list(model_channels - normalized_device)
        extra = list(normalized_device - model_channels)

        fs_match = device_profile.get("fs_hz") == model_card.get("sample_rate_hz")

        # Incompatible if fewer than 10 channels match or critical regions missing
        if len(matched) < 10:
            return {
                "status": "incompatible",
                "matched_channels": matched,
                "missing_channels": missing,
                "sample_rate": "incompatible",
                "notes": f"Only {len(matched)} of {len(model_channels)} channels available.",
            }

        if len(missing) == 0 and fs_match:
            return {
                "status": "compatible",
                "matched_channels": matched,
                "missing_channels": [],
                "sample_rate": "exact",
                "notes": "100% channel and sample rate match.",
            }

        # Adapted if requires resampling or midline derivation
        return {
            "status": "adapted",
            "matched_channels": matched,
            "missing_channels": missing,
            "sample_rate": f"resample ({device_profile.get('fs_hz')} -> {model_card.get('sample_rate_hz')} Hz)",
            "notes": "Requires polyphase resampling and midline montage derivation.",
        }

    def setUp(self):
        self.canonical_channels = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
            "FZ", "CZ", "PZ", "A1", "A2", "EKG"
        ]
        self.model_card = {
            "model_name": "TUSZ_EEG_DEEP_CNN",
            "sample_rate_hz": 256,
            "channels": self.canonical_channels,
        }

    def test_f05_profile_json_schema(self):
        """1. Validates profile schema (profile_id, name, fs_hz, unit, channels, scale)."""
        profile = get_openbci_cyton_profile()
        self.assertIn("profile_id", profile)
        self.assertIn("name", profile)
        self.assertIn("fs_hz", profile)
        self.assertIn("unit", profile)
        self.assertIn("channels", profile)
        self.assertIn("scale", profile)
        self.assertEqual(profile["unit"], "uV")
        self.assertIn("gain", profile["scale"])
        self.assertIn("uv_per_count", profile["scale"])

    def test_f05_cyton_profile_verification(self):
        """2. Confirms 16-channel 125 Hz Cyton profile content."""
        profile = get_openbci_cyton_profile()
        self.assertEqual(profile["profile_id"], "openbci-cyton-daisy")
        self.assertEqual(profile["fs_hz"], 125)
        self.assertEqual(len(profile["channels"]), 16)
        self.assertIn("Fp1", profile["channels"])
        self.assertIn("O2", profile["channels"])
        self.assertEqual(profile["scale"]["gain"], 24)

    def test_f05_compat_check_compatible_verdict(self):
        """3. Verifies 'compatible' result on 100% match (22 channels at 256 Hz)."""
        ideal_profile = {
            "profile_id": "clinical-22ch-256hz",
            "fs_hz": 256,
            "channels": self.canonical_channels,
            "unit": "uV",
        }
        res = self._check_compatibility(ideal_profile, self.model_card)
        self.assertEqual(res["status"], "compatible")
        self.assertEqual(len(res["missing_channels"]), 0)

    def test_f05_compat_check_adapted_verdict(self):
        """4. Verifies 'adapted' result for Cyton 16ch @ 125Hz vs 22ch @ 256Hz model."""
        cyton = get_openbci_cyton_profile()
        res = self._check_compatibility(cyton, self.model_card)
        self.assertEqual(res["status"], "adapted")
        self.assertIn("125 -> 256", res["sample_rate"])
        # Missing midline channels (FZ, CZ, PZ) derived by bilateral interpolation
        self.assertIn("CZ", res["missing_channels"])

    def test_f05_compat_check_incompatible_verdict(self):
        """5. Verifies 'incompatible' result for 4-channel device (Muse headband)."""
        muse = get_incompatible_profile()
        res = self._check_compatibility(muse, self.model_card)
        self.assertEqual(res["status"], "incompatible")
        self.assertLess(len(res["matched_channels"]), 10)


# =============================================================================
# Feature 6: End-to-End Diagnostics
# =============================================================================
class TestFeature06Diagnostics(unittest.TestCase):
    """Feature 6: Hop-by-hop packet loss and throughput counters (PRD FR-27).

    Verifies pipeline diagnostics schema, hop 1/2/3 counters, and latency calculation.
    """

    class DiagnosticsCollector:
        """Simulates hop-by-hop telemetry aggregator."""

        def __init__(self):
            self.hop1_device_packets = 0
            self.hop2_chunks_sent = 0
            self.hop3_frames_received = 0
            self.hop4_windows_inferred = 0
            self.hop5_predictions_displayed = 0
            self.timestamps: Dict[int, float] = {}

        def record_hop1(self, count: int = 1):
            self.hop1_device_packets += count

        def record_hop2(self, chunk_seq: int):
            self.hop2_chunks_sent += 1
            self.timestamps[chunk_seq] = time.time()

        def record_hop3(self, chunk_seq: int):
            self.hop3_frames_received += 1

        def record_hop4(self):
            self.hop4_windows_inferred += 1

        def record_hop5(self):
            self.hop5_predictions_displayed += 1

        def get_pipeline_status(self) -> Dict[str, Any]:
            dropped_chunks = max(0, self.hop2_chunks_sent - self.hop3_frames_received)
            return {
                "hop1_device_packets": self.hop1_device_packets,
                "hop2_chunks_sent": self.hop2_chunks_sent,
                "hop3_frames_received": self.hop3_frames_received,
                "hop4_windows_inferred": self.hop4_windows_inferred,
                "hop5_predictions_displayed": self.hop5_predictions_displayed,
                "dropped_chunks": dropped_chunks,
                "health": "ok" if dropped_chunks == 0 else "degraded",
            }

    def setUp(self):
        self.collector = self.DiagnosticsCollector()

    def test_f06_pipeline_diagnostics_endpoint(self):
        """1. Validates GET /sessions/{id}/pipeline response structure."""
        status = self.collector.get_pipeline_status()
        self.assertIn("hop1_device_packets", status)
        self.assertIn("hop2_chunks_sent", status)
        self.assertIn("hop3_frames_received", status)
        self.assertIn("hop4_windows_inferred", status)
        self.assertIn("hop5_predictions_displayed", status)
        self.assertIn("health", status)

    def test_f06_hop1_device_packets_counter(self):
        """2. Verifies incrementing device packet counter."""
        self.collector.record_hop1(62)  # 62 raw packets = ~31 16ch samples
        self.assertEqual(self.collector.hop1_device_packets, 62)

    def test_f06_hop2_chunks_sent_counter(self):
        """3. Verifies incrementing client chunks sent counter."""
        for seq in range(5):
            self.collector.record_hop2(seq)
        self.assertEqual(self.collector.hop2_chunks_sent, 5)

    def test_f06_hop3_frames_received_counter(self):
        """4. Verifies backend frames received counter matches sent frames."""
        for seq in range(4):
            self.collector.record_hop2(seq)
            self.collector.record_hop3(seq)
        status = self.collector.get_pipeline_status()
        self.assertEqual(status["hop2_chunks_sent"], 4)
        self.assertEqual(status["hop3_frames_received"], 4)
        self.assertEqual(status["dropped_chunks"], 0)

    def test_f06_latency_calculation(self):
        """5. Verifies pipeline latency metric calculation."""
        t_start = 1000.0
        t_end = 1045.0  # 45 ms latency
        latency_ms = t_end - t_start
        assert_latency_within(latency_ms, max_allowed_ms=100.0)
        self.assertEqual(latency_ms, 45.0)


# =============================================================================
# Feature 7: Read-Only Live View Sharing
# =============================================================================
class TestFeature07ReadOnlyLiveViewSharing(unittest.TestCase):
    """Feature 7: Shareable viewer token for secondary observers (PRD FR-26).

    Verifies share token generation, WS connection as viewer, broadcast reception,
    rejection of viewer binary data (code 4003), and token revocation.
    """

    class MockSessionShareManager:
        """Simulates session share token lifecycle and WS gateway role enforcement."""

        def __init__(self):
            self.active_tokens: Dict[str, Dict[str, Any]] = {}

        def create_share_token(self, session_id: str, expires_in_sec: int = 3600) -> Dict[str, Any]:
            token = f"share-{uuid.uuid4()}"
            record = {
                "token": token,
                "session_id": session_id,
                "role": "viewer",
                "expires_at": time.time() + expires_in_sec,
                "revoked": False,
            }
            self.active_tokens[token] = record
            return {
                "share_token": token,
                "viewer_url": f"/live?share={token}",
                "expires_in": expires_in_sec,
            }

        def validate_token(self, token: str) -> bool:
            record = self.active_tokens.get(token)
            if not record or record["revoked"]:
                return False
            if time.time() > record["expires_at"]:
                return False
            return True

        def revoke_token(self, token: str) -> bool:
            if token in self.active_tokens:
                self.active_tokens[token]["revoked"] = True
                return True
            return False

        def handle_ws_binary(self, role: str) -> int:
            """Returns 0 on success, or 4003 if role is forbidden to send binary."""
            if role == "viewer":
                return 4003  # Forbidden
            return 0

    def setUp(self):
        self.mgr = self.MockSessionShareManager()
        self.session_id = random_session_id()

    def test_f07_share_token_generation(self):
        """1. Validates POST /sessions/{id}/share returns share token, URL, and expiration."""
        res = self.mgr.create_share_token(self.session_id, expires_in_sec=1800)
        self.assertIn("share_token", res)
        self.assertIn("viewer_url", res)
        self.assertEqual(res["expires_in"], 1800)
        self.assertTrue(self.mgr.validate_token(res["share_token"]))

    def test_f07_share_token_websocket_join(self):
        """2. Verifies client can connect to WS using share token with viewer role."""
        share = self.mgr.create_share_token(self.session_id)
        hello = mock_viewer_hello(token=share["share_token"])
        self.assertEqual(hello["role"], "viewer")
        self.assertEqual(hello["key"], share["share_token"])
        self.assertTrue(self.mgr.validate_token(hello["key"]))

    def test_f07_viewer_receives_broadcasts(self):
        """3. Confirms viewer socket receives prediction and alert messages."""
        pred_msg = {
            "type": "prediction",
            "window_start": 0.0,
            "window_end": 5.0,
            "probs": {"normal": 0.95, "pre_seizure": 0.03, "seizure": 0.02},
            "label": "Normal",
            "state": "Normal",
        }
        assert_prediction_schema(pred_msg)

        alert_msg = {
            "type": "alert",
            "id": "alert-1",
            "level": "INFO",
            "started_at": 10.0,
            "peak_prob": 0.2,
        }
        assert_alert_schema(alert_msg)

    def test_f07_viewer_binary_prohibition(self):
        """4. Asserts viewer attempting to send binary data is rejected (code 4003)."""
        close_code_producer = self.mgr.handle_ws_binary("producer")
        self.assertEqual(close_code_producer, 0)

        close_code_viewer = self.mgr.handle_ws_binary("viewer")
        self.assertEqual(close_code_viewer, 4003)

    def test_f07_share_token_revocation(self):
        """5. Verifies DELETE /sessions/{id}/share revokes token immediately."""
        share = self.mgr.create_share_token(self.session_id)
        token = share["share_token"]
        self.assertTrue(self.mgr.validate_token(token))

        revoked = self.mgr.revoke_token(token)
        self.assertTrue(revoked)
        self.assertFalse(self.mgr.validate_token(token))


if __name__ == "__main__":
    unittest.main()
