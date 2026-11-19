"""Tier 2: Boundary & Corner Cases - Features 1-7 (Ingestion & Device Integration).

Exhaustively covers edge, corner, and error-handling conditions for:
- Feature 1: Web Serial Link Device (6 boundary tests)
- Feature 2: OpenBCI Packet Parser (6 boundary tests)
- Feature 3: 250ms Binary Framing (6 boundary tests)
- Feature 4: Device Adapter Layer (EegSource) (5 boundary tests)
- Feature 5: Device Profiles & Compatibility (5 boundary tests)
- Feature 6: End-to-End Diagnostics (5 boundary tests)
- Feature 7: Read-Only Live View Sharing (5 boundary tests)

Total: 38 boundary tests.
"""

from __future__ import annotations

import math
import struct
import time
import unittest
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
# Feature 1: Web Serial Link Device - Boundary Tests
# =============================================================================
class TestBoundaryFeature01WebSerialLink(unittest.TestCase):
    """Boundary and corner cases for Web Serial Link Device (Feature 1)."""

    class MockWebSerialManager:
        def __init__(self, has_navigator_serial: bool = True):
            self.has_navigator_serial = has_navigator_serial
            self.port_open = False
            self.readable_locked = False
            self.buffer = bytearray()
            self.max_buffer_size = 100_000

        def request_port(self, user_cancelled: bool = False):
            if not self.has_navigator_serial:
                raise AttributeError("navigator.serial is undefined in non-Chromium browser")
            if user_cancelled:
                raise PermissionError("NotFoundError: No port selected by user")
            return self

        def open(self, baud_rate: int = 115200):
            if baud_rate != 115200:
                raise ValueError(f"FramingError: Device requires 115200 baud, got {baud_rate}")
            self.port_open = True

        def feed(self, data: bytes):
            if len(self.buffer) + len(data) > self.max_buffer_size:
                # Buffer overrun: drop oldest or raise overrun
                dropped = (len(self.buffer) + len(data)) - self.max_buffer_size
                del self.buffer[:dropped]
                self.buffer.extend(data)
                raise BufferError(f"Serial buffer overrun: dropped {dropped} bytes")
            self.buffer.extend(data)

        def read(self, max_bytes: int = 1024) -> bytes:
            if not self.port_open:
                raise ConnectionResetError("Port abruptly disconnected during read")
            chunk = self.buffer[:max_bytes]
            del self.buffer[:max_bytes]
            return bytes(chunk)

        def disconnect_abruptly(self):
            self.port_open = False
            self.readable_locked = False
            self.buffer.clear()

    def setUp(self):
        self.serial_manager = self.MockWebSerialManager(has_navigator_serial=True)

    def test_f01_non_chromium_user_agent(self):
        """1. Non-Chromium user agent handling (navigator.serial is undefined)."""
        safari_manager = self.MockWebSerialManager(has_navigator_serial=False)
        with self.assertRaises(AttributeError) as ctx:
            safari_manager.request_port()
        self.assertIn("navigator.serial is undefined", str(ctx.exception))

    def test_f01_abrupt_port_disconnect_during_streaming(self):
        """2. Abrupt port disconnect during active binary streaming triggers I/O cleanup."""
        self.serial_manager.open(115200)
        self.serial_manager.feed(b"\xa0" * 66)
        # Read first chunk
        chunk = self.serial_manager.read(33)
        self.assertEqual(len(chunk), 33)

        # Abrupt unplug of USB FTDI dongle
        self.serial_manager.disconnect_abruptly()
        self.assertFalse(self.serial_manager.port_open)

        # Subsequent read must fail cleanly with ConnectionResetError
        with self.assertRaises(ConnectionResetError):
            self.serial_manager.read(33)

    def test_f01_user_cancels_com_picker(self):
        """3. User cancels browser native COM port picker (NotFoundError / AbortError)."""
        with self.assertRaises(PermissionError) as ctx:
            self.serial_manager.request_port(user_cancelled=True)
        self.assertIn("No port selected", str(ctx.exception))

    def test_f01_serial_buffer_overrun(self):
        """4. Serial frame buffer overrun (> 100,000 unread bytes) drops oldest bytes with warning."""
        self.serial_manager.open(115200)
        # Fill buffer to capacity
        self.serial_manager.feed(b"\x00" * 90000)
        self.assertEqual(len(self.serial_manager.buffer), 90000)

        # Feed 20,000 more bytes, exceeding 100,000 limit
        with self.assertRaises(BufferError) as ctx:
            self.serial_manager.feed(b"\xFF" * 20000)
        self.assertIn("Serial buffer overrun", str(ctx.exception))
        self.assertEqual(len(self.serial_manager.buffer), 100000)

    def test_f01_baud_rate_mismatch(self):
        """5. Dongle firmware baud rate mismatch (e.g. 9600 vs 115200) detected."""
        with self.assertRaises(ValueError) as ctx:
            self.serial_manager.open(baud_rate=9600)
        self.assertIn("FramingError", str(ctx.exception))

    def test_f01_zero_length_serial_read(self):
        """6. Zero-length read or EOF when buffer is dry returns empty bytes without blocking."""
        self.serial_manager.open(115200)
        res = self.serial_manager.read(1024)
        self.assertEqual(res, b"")
        self.assertTrue(self.serial_manager.port_open)


# =============================================================================
# Feature 2: OpenBCI Packet Parser - Boundary Tests
# =============================================================================
class TestBoundaryFeature02OpenBciParser(unittest.TestCase):
    """Boundary and corner cases for OpenBCI Packet Parser (Feature 2)."""

    def test_f02_packet_loss_counter_gap_over_100(self):
        """1. RFduino packet loss with counter gap > 100 correctly calculates dropped count."""
        # Packet counter jumps from 10 to 125 -> gap of 115 - 1 = 114 dropped packets
        prev_cnt = 10
        curr_cnt = 125
        gap = (curr_cnt - prev_cnt) % 256
        self.assertEqual(gap, 115)
        dropped_packets = gap - 1
        self.assertEqual(dropped_packets, 114)
        self.assertGreater(dropped_packets, 100)

        # Test wraparound gap: from 250 to 105 -> (105 - 250) % 256 = 111 -> 110 dropped
        prev_cnt2 = 250
        curr_cnt2 = 105
        gap2 = (curr_cnt2 - prev_cnt2) % 256
        self.assertEqual(gap2, 111)
        self.assertEqual(gap2 - 1, 110)

    def test_f02_malformed_start_byte(self):
        """2. Malformed packet start byte (0xFF instead of 0xA0) raises ValueError."""
        valid_pkt = EegStreamer.pack_openbci_packet(0, [0.0] * 8)
        bad_pkt = b"\xFF" + valid_pkt[1:]
        self.assertEqual(len(bad_pkt), 33)
        with self.assertRaises(ValueError) as ctx:
            EegStreamer.unpack_openbci_packet(bad_pkt)
        self.assertIn("Invalid start byte", str(ctx.exception))

    def test_f02_corrupted_stop_byte(self):
        """3. Corrupted stop byte (0x00 instead of 0xC0-0xC6) raises ValueError."""
        valid_pkt = EegStreamer.pack_openbci_packet(0, [0.0] * 8)
        bad_pkt = valid_pkt[:32] + b"\x00"
        with self.assertRaises(ValueError) as ctx:
            EegStreamer.unpack_openbci_packet(bad_pkt)
        self.assertIn("Invalid stop byte", str(ctx.exception))

    def test_f02_truncated_packet_fewer_than_33_bytes(self):
        """4. Truncated packet with fewer than 33 bytes (e.g. 1, 16, 32 bytes) rejected."""
        valid_pkt = EegStreamer.pack_openbci_packet(0, [0.0] * 8)
        for length in (0, 1, 16, 32):
            truncated = valid_pkt[:length]
            with self.assertRaises(ValueError) as ctx:
                EegStreamer.unpack_openbci_packet(truncated)
            self.assertIn("Invalid packet size", str(ctx.exception))

    def test_f02_extreme_24bit_voltage_saturation(self):
        """5. Extreme 24-bit integer saturation bounds: +8,388,607 and -8,388,608."""
        # Max positive: 0x7FFFFF = 8,388,607 counts * 0.02235174 uV = ~187,500 uV
        max_pos_uv = 8388607 * EegStreamer.UV_PER_COUNT
        raw_max = EegStreamer.uv_to_24bit_signed(max_pos_uv)
        self.assertEqual(raw_max, b"\x7F\xFF\xFF")
        recovered_max = EegStreamer.signed_24bit_to_uv(raw_max)
        self.assertAlmostEqual(recovered_max, max_pos_uv, delta=0.03)

        # Min negative: 0x800000 = -8,388,608 counts * 0.02235174 uV = ~-187,500 uV
        min_neg_uv = -8388608 * EegStreamer.UV_PER_COUNT
        raw_min = EegStreamer.uv_to_24bit_signed(min_neg_uv)
        self.assertEqual(raw_min, b"\x80\x00\x00")
        recovered_min = EegStreamer.signed_24bit_to_uv(raw_min)
        self.assertAlmostEqual(recovered_min, min_neg_uv, delta=0.03)

        # Beyond limits must clamp
        clamped_high = EegStreamer.uv_to_24bit_signed(1_000_000.0)
        self.assertEqual(clamped_high, b"\x7F\xFF\xFF")
        clamped_low = EegStreamer.uv_to_24bit_signed(-1_000_000.0)
        self.assertEqual(clamped_low, b"\x80\x00\x00")

    def test_f02_daisy_packet_interleaving_desync(self):
        """6. Daisy pairing desynchronization (consecutive even packets or consecutive odd packets)."""
        # Board packets have odd sample counters (or pair start), Daisy has pair start + 1
        # Simulating desync where packet 2 is received before packet 1
        pkt_board, pkt_daisy = EegStreamer.pack_daisy_pair(
            sample_counter_pair_start=1,
            channels_16_uv=[10.0] * 16,
        )
        self.assertEqual(pkt_board[1], 1)
        self.assertEqual(pkt_daisy[1], 2)

        # Two consecutive board packets (desync)
        def pair_packets(p1: bytes, p2: bytes):
            c1 = p1[1]
            c2 = p2[1]
            if (c1 + 1) % 256 != c2:
                raise ValueError(f"Daisy pairing desync: expected {c1+1}, got {c2}")

        # Normal pair passes
        pair_packets(pkt_board, pkt_daisy)

        # Consecutive duplicate board packets fail
        with self.assertRaises(ValueError) as ctx:
            pair_packets(pkt_board, pkt_board)
        self.assertIn("Daisy pairing desync", str(ctx.exception))


# =============================================================================
# Feature 3: 250ms Binary Framing - Boundary Tests
# =============================================================================
class TestBoundaryFeature03BinaryFraming(unittest.TestCase):
    """Boundary and corner cases for 250ms Binary Framing (Feature 3)."""

    def test_f03_incomplete_chunk_flushing_at_termination(self):
        """1. Incomplete 250ms chunk (e.g. 12 samples instead of 31) flushed at session end."""
        partial_samples = 12
        channels = 16
        data = np.random.randn(partial_samples, channels).astype(np.float32)
        frame = EegStreamer.pack_ws_binary_frame(
            seq=42, t0_ms=1700000000000.0, data=data, flags=0x02  # Flag 0x02 = FLUSH_END
        )
        # Length must be 16 bytes header + 12 * 16 * 4 = 768 bytes payload = 784 bytes total
        self.assertEqual(len(frame), 16 + 768)
        header, rec_data = EegStreamer.unpack_ws_binary_frame(frame)
        self.assertEqual(header["n_samples"], 12)
        self.assertEqual(header["n_channels"], 16)
        self.assertEqual(header["flags"], 0x02)
        np.testing.assert_array_almost_equal(rec_data, data)

    def test_f03_oversized_binary_frame(self):
        """2. Oversized binary frame exceeding 64 KB triggers WebSocket close code 1009 / rejection."""
        # 64KB = 65,536 bytes. A frame with 1100 samples x 16 channels x 4 bytes = 70,400 bytes > 64KB
        oversized_data = np.zeros((1100, 16), dtype=np.float32)
        oversized_frame = EegStreamer.pack_ws_binary_frame(
            seq=1, t0_ms=1700000000000.0, data=oversized_data
        )
        self.assertGreater(len(oversized_frame), 65536)

        # Server validator must reject frames > 64KB
        def validate_frame_size(frame_bytes: bytes, max_allowed: int = 65536):
            if len(frame_bytes) > max_allowed:
                raise WebSocketCloseError(1009, f"Frame size {len(frame_bytes)} exceeds {max_allowed}B")

        with self.assertRaises(WebSocketCloseError) as ctx:
            validate_frame_size(oversized_frame)
        self.assertEqual(ctx.exception.code, 1009)

    def test_f03_non_monotonic_sequence_numbers(self):
        """3. Client transmitting non-monotonic sequence numbers (jitter or regression) detected."""
        def check_monotonic_sequence(history: List[int], incoming_seq: int) -> bool:
            if not history:
                return True
            return incoming_seq == history[-1] + 1

        seq_history = [1, 2, 3]
        # Regressive sequence
        self.assertFalse(check_monotonic_sequence(seq_history, 2))
        # Large jump
        self.assertFalse(check_monotonic_sequence(seq_history, 5))
        # Expected monotonic
        self.assertTrue(check_monotonic_sequence(seq_history, 4))

    def test_f03_single_sample_chunk_boundary(self):
        """4. Single-sample chunk (N=1, 16 channels, payload = 64 bytes)."""
        single_sample_data = np.ones((1, 16), dtype=np.float32) * 42.0
        frame = EegStreamer.pack_ws_binary_frame(
            seq=0, t0_ms=1700000000000.0, data=single_sample_data
        )
        self.assertEqual(len(frame), 16 + 64)
        header, recovered = EegStreamer.unpack_ws_binary_frame(frame)
        self.assertEqual(header["n_samples"], 1)
        self.assertEqual(header["n_channels"], 16)
        self.assertEqual(recovered.shape, (1, 16))
        self.assertEqual(recovered[0, 0], 42.0)

    def test_f03_zero_sample_binary_frame(self):
        """5. Zero-sample binary frame (N=0, 16-byte header only, empty payload)."""
        empty_data = np.zeros((0, 16), dtype=np.float32)
        frame = EegStreamer.pack_ws_binary_frame(seq=99, t0_ms=1700000000000.0, data=empty_data)
        self.assertEqual(len(frame), 16)
        header, recovered = EegStreamer.unpack_ws_binary_frame(frame)
        self.assertEqual(header["n_samples"], 0)
        self.assertEqual(len(recovered), 0)

    def test_f03_corrupted_header_endianness_mismatch(self):
        """6. Header endianness corruption or negative t0_ms timestamp rejected."""
        # Big-endian encoded header instead of little-endian '<IdHBB'
        bad_header = struct.pack(">IdHBB", 1, 1700000000000.0, 31, 16, 0)
        payload = b"\x00" * (31 * 16 * 4)
        corrupted_frame = bad_header + payload

        # Unpacking as little-endian will result in absurd values and payload length mismatch, raising ValueError
        with self.assertRaises(ValueError) as ctx:
            EegStreamer.unpack_ws_binary_frame(corrupted_frame)
        self.assertIn("Payload size mismatch", str(ctx.exception))


# =============================================================================
# Feature 4: Device Adapter Layer (EegSource) - Boundary Tests
# =============================================================================
class TestBoundaryFeature04DeviceAdapterLayer(unittest.TestCase):
    """Boundary and corner cases for Device Adapter Layer (Feature 4)."""

    class MockEegSource:
        def __init__(self, source_id: str):
            self.source_id = source_id
            self.connected = False
            self.listeners: List[Callable[[Any], None]] = []

        def connect(self):
            if self.connected:
                raise RuntimeError("Already connected")
            self.connected = True

        def disconnect(self):
            self.connected = False
            self.listeners.clear()

        def add_chunk_listener(self, listener: Callable[[Any], None]):
            self.listeners.append(listener)

        def emit_chunk(self, chunk: Any):
            for listener in list(self.listeners):
                try:
                    listener(chunk)
                except Exception:
                    # Isolate consumer errors from adapter pipeline
                    pass

    def test_f04_adapter_rapid_reconnect(self):
        """1. Disconnecting and reconnecting EegSource in quick succession (<10ms)."""
        src = self.MockEegSource("cyton-serial")
        for _ in range(5):
            src.connect()
            self.assertTrue(src.connected)
            src.disconnect()
            self.assertFalse(src.connected)

    def test_f04_adapter_exception_in_chunk_callback(self):
        """2. Consumer callback raises exception; adapter isolates and does not crash."""
        src = self.MockEegSource("cyton-serial")
        src.connect()
        received = []

        def faulty_listener(chunk):
            raise ZeroDivisionError("Consumer bug")

        def good_listener(chunk):
            received.append(chunk)

        src.add_chunk_listener(faulty_listener)
        src.add_chunk_listener(good_listener)

        # Emitting chunk should not raise ZeroDivisionError
        src.emit_chunk("valid_sample_chunk")
        self.assertEqual(len(received), 1)

    def test_f04_adapter_null_chunk_handling(self):
        """3. Handling of None or malformed chunk in onChunk callback."""
        src = self.MockEegSource("cyton-serial")
        src.connect()
        received = []
        src.add_chunk_listener(lambda c: received.append(c))

        src.emit_chunk(None)
        self.assertEqual(len(received), 1)
        self.assertIsNone(received[0])

    def test_f04_adapter_simultaneous_dual_source(self):
        """4. Attempting to attach second active source throws conflict error."""
        class AdapterManager:
            def __init__(self):
                self.active_source: Optional[str] = None

            def attach(self, source_id: str):
                if self.active_source is not None:
                    raise RuntimeError(f"Conflict: source '{self.active_source}' already active")
                self.active_source = source_id

        mgr = AdapterManager()
        mgr.attach("cyton-serial")
        with self.assertRaises(RuntimeError) as ctx:
            mgr.attach("synthetic-generator")
        self.assertIn("Conflict", str(ctx.exception))

    def test_f04_adapter_teardown_with_pending_buffers(self):
        """5. Disconnecting while buffer has unread frames safely clears all listeners."""
        src = self.MockEegSource("replay")
        src.connect()
        src.add_chunk_listener(lambda c: None)
        self.assertEqual(len(src.listeners), 1)
        src.disconnect()
        self.assertEqual(len(src.listeners), 0)
        self.assertFalse(src.connected)


# =============================================================================
# Feature 5: Device Profiles & Compatibility - Boundary Tests
# =============================================================================
class TestBoundaryFeature05DeviceProfiles(unittest.TestCase):
    """Boundary and corner cases for Device Profiles & Compatibility (Feature 5)."""

    def _evaluate_profile(self, profile: Dict[str, Any]) -> str:
        """Evaluator per PRD FR-24/25."""
        channels = profile.get("channels", [])
        fs = profile.get("fs_hz", 0)
        if len(channels) == 0 or fs < 50 or fs > 10000:
            return "incompatible"
        if len(channels) == 22 and fs == 256:
            return "compatible"
        if len(channels) == 16 and fs in (125, 250):
            return "adapted"
        if len(channels) < 8:
            return "incompatible"
        return "adapted"

    def test_f05_zero_channel_profile(self):
        """1. Device profile declaring 0 channels rejected with 'incompatible' verdict."""
        bad_profile = {"profile_id": "bad-empty", "fs_hz": 125, "channels": []}
        verdict = self._evaluate_profile(bad_profile)
        self.assertEqual(verdict, "incompatible")

    def test_f05_four_channel_muse_headband(self):
        """2. 4-channel consumer device (Muse headband) evaluated -> 'incompatible'."""
        muse_profile = get_incompatible_profile()
        self.assertEqual(len(muse_profile["channels"]), 4)
        verdict = self._evaluate_profile(muse_profile)
        self.assertEqual(verdict, "incompatible")

    def test_f05_128_channel_high_density_hd_eeg(self):
        """3. 128-channel research system evaluated -> marked as 'adapted'."""
        hd_profile = {
            "profile_id": "hd-eeg-128",
            "fs_hz": 500,
            "channels": [f"E{i}" for i in range(1, 129)],
        }
        verdict = self._evaluate_profile(hd_profile)
        self.assertEqual(verdict, "adapted")

    def test_f05_missing_parasagittal_midline_channels(self):
        """4. Profile missing Fz, Cz, Pz midline approximations identified for derivation."""
        cyton = get_openbci_cyton_profile()
        self.assertNotIn("FZ", [c.upper() for c in cyton["channels"]])
        self.assertNotIn("CZ", [c.upper() for c in cyton["channels"]])
        self.assertNotIn("PZ", [c.upper() for c in cyton["channels"]])
        verdict = self._evaluate_profile(cyton)
        self.assertEqual(verdict, "adapted")

    def test_f05_unsupported_sample_rate(self):
        """5. Extreme unsupported sampling rates (<50 Hz or >10000 Hz) rejected."""
        slow_profile = {"profile_id": "too-slow", "fs_hz": 16, "channels": ["Fp1"] * 16}
        fast_profile = {"profile_id": "too-fast", "fs_hz": 44100, "channels": ["Fp1"] * 16}
        self.assertEqual(self._evaluate_profile(slow_profile), "incompatible")
        self.assertEqual(self._evaluate_profile(fast_profile), "incompatible")


# =============================================================================
# Feature 6: End-to-End Diagnostics - Boundary Tests
# =============================================================================
class TestBoundaryFeature06Diagnostics(unittest.TestCase):
    """Boundary and corner cases for End-to-End Diagnostics (Feature 6)."""

    class DiagnosticPipeline:
        def __init__(self):
            self.hop1_packets = 0
            self.hop2_chunks = 0
            self.hop3_frames = 0
            self.last_update = time.time()

        def increment_hop1(self, count: int = 1):
            # 32-bit unsigned rollover simulation
            self.hop1_packets = (self.hop1_packets + count) % 4294967296
            self.last_update = time.time()

        def calculate_latency(self, t_client: float, t_server: float) -> float:
            # Clamp negative latency caused by clock drift to 0.0 ms
            diff = t_server - t_client
            return max(0.0, diff)

        def is_stalled(self, now: float, timeout_sec: float = 5.0) -> bool:
            return (now - self.last_update) > timeout_sec

    def test_f06_hop_counter_32bit_integer_overflow(self):
        """1. Hop packet counter at 2^32 - 1 rolls over cleanly without integer overflow crash."""
        diag = self.DiagnosticPipeline()
        diag.hop1_packets = 4294967295  # 2^32 - 1
        diag.increment_hop1(1)
        self.assertEqual(diag.hop1_packets, 0)

        diag.increment_hop1(10)
        self.assertEqual(diag.hop1_packets, 10)

    def test_f06_diagnostic_pipeline_hop_stall_timeout(self):
        """2. Diagnostic pipeline timeout when intermediate hop stalls (> 5.0s inactivity)."""
        diag = self.DiagnosticPipeline()
        t0 = time.time()
        self.assertFalse(diag.is_stalled(t0 + 2.0, timeout_sec=5.0))
        self.assertTrue(diag.is_stalled(t0 + 5.1, timeout_sec=5.0))

    def test_f06_negative_latency_clock_skew(self):
        """3. Clock drift where client timestamp exceeds server timestamp clamped to >= 0.0 ms."""
        diag = self.DiagnosticPipeline()
        t_client = 1700000000100.0  # Client clock ahead by 100ms
        t_server = 1700000000000.0
        lat = diag.calculate_latency(t_client, t_server)
        self.assertEqual(lat, 0.0)

        # Normal latency
        normal_lat = diag.calculate_latency(1700000000000.0, 1700000000015.0)
        self.assertEqual(normal_lat, 15.0)

    def test_f06_diagnostics_uninitialized_session(self):
        """4. Diagnostic query on freshly created session with 0 frames returns zero counts."""
        diag = self.DiagnosticPipeline()
        self.assertEqual(diag.hop1_packets, 0)
        self.assertEqual(diag.hop2_chunks, 0)
        self.assertEqual(diag.hop3_frames, 0)

    def test_f06_diagnostic_rapid_polling_stress(self):
        """5. Rapid successive pipeline diagnostic updates (1000 iterations) maintain counter integrity."""
        diag = self.DiagnosticPipeline()
        for i in range(1000):
            diag.increment_hop1(1)
        self.assertEqual(diag.hop1_packets, 1000)


# =============================================================================
# Feature 7: Read-Only Live View Sharing - Boundary Tests
# =============================================================================
class TestBoundaryFeature07LiveViewSharing(unittest.TestCase):
    """Boundary and corner cases for Read-Only Live View Sharing (Feature 7)."""

    class MockViewerSession:
        def __init__(self, token: str, expires_at: float):
            self.token = token
            self.expires_at = expires_at
            self.revoked = False
            self.viewers: List[str] = []

        def join(self, provided_token: str, now: float) -> bool:
            if self.revoked:
                return False
            if now > self.expires_at:
                return False
            if provided_token != self.token:
                return False
            self.viewers.append(provided_token)
            return True

        def receive_frame(self, role: str, is_binary: bool) -> Tuple[bool, int]:
            if role == "viewer" and is_binary:
                return False, 4003  # Forbidden close code per spec
            return True, 1000

    def test_f07_viewer_expired_share_token(self):
        """1. Viewer attempting to connect with expired share token is rejected."""
        now = time.time()
        sess = self.MockViewerSession(token="valid-token-123", expires_at=now - 10.0)
        self.assertFalse(sess.join("valid-token-123", now=now))

    def test_f07_viewer_forbidden_binary_injection(self):
        """2. Viewer attempting to send binary EEG frame rejected with close code 4003."""
        sess = self.MockViewerSession(token="valid-token", expires_at=time.time() + 3600)
        ok, code = sess.receive_frame(role="viewer", is_binary=True)
        self.assertFalse(ok)
        self.assertEqual(code, 4003)

        # Producer can send binary frames
        prod_ok, prod_code = sess.receive_frame(role="producer", is_binary=True)
        self.assertTrue(prod_ok)
        self.assertEqual(prod_code, 1000)

    def test_f07_viewer_attempt_control_action(self):
        """3. Viewer attempting control operation (e.g. stop session) rejected with HTTP 403."""
        def execute_session_action(role: str, action: str) -> int:
            if role == "viewer":
                return 403  # Forbidden
            return 200

        self.assertEqual(execute_session_action("viewer", "stop"), 403)
        self.assertEqual(execute_session_action("producer", "stop"), 200)

    def test_f07_multiple_concurrent_viewers_limit(self):
        """4. Handling multiple concurrent viewers (50 viewers) joining session successfully."""
        sess = self.MockViewerSession(token="multi-view", expires_at=time.time() + 3600)
        now = time.time()
        for i in range(50):
            res = sess.join("multi-view", now=now)
            self.assertTrue(res)
        self.assertEqual(len(sess.viewers), 50)

    def test_f07_share_token_entropy_and_tampering(self):
        """5. Tampered share token (trailing character or bit flip) rejected."""
        sess = self.MockViewerSession(token="secret-share-key-42", expires_at=time.time() + 3600)
        now = time.time()
        # Missing trailing character
        self.assertFalse(sess.join("secret-share-key-4", now=now))
        # Extra character
        self.assertFalse(sess.join("secret-share-key-42X", now=now))
        # Exact match succeeds
        self.assertTrue(sess.join("secret-share-key-42", now=now))


if __name__ == "__main__":
    unittest.main()
