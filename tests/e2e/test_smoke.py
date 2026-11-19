"""Self-Testing Smoke Suite for Seizure-Sentry E2E Test Infrastructure.

Validates the test harness itself: config, HTTP client, WebSocket RFC 6455 framing,
OpenBCI packet parser, WS binary streamer, synthetic EEG generators, assertions,
and fixtures without external dependencies.
"""

from __future__ import annotations

import socket
import struct
import threading
import unittest
import numpy as np

from tests.e2e.client.http_client import HttpResponse, TestHttpClient
from tests.e2e.client.streamer import EegStreamer
from tests.e2e.client.ws_client import (
    OP_BINARY,
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    RFC6455Client,
    WebSocketCloseError,
    WebSocketError,
)
from tests.e2e.config import CONFIG, E2EConfig, set_config
from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_latency_within,
    assert_montage_shape,
    assert_prediction_schema,
    assert_probability_distribution,
    assert_signal_quality,
    assert_status_code,
    assert_valid_openbci_packet,
    assert_valid_ws_binary_frame,
)
from tests.e2e.utils.fixtures import (
    generate_mock_binary_frame,
    generate_openbci_raw_stream,
    get_incompatible_profile,
    get_openbci_cyton_profile,
    golden_model_input_tensor,
    mock_client_hello,
    mock_viewer_hello,
    random_access_key,
    random_session_id,
)


class TestConfigInfrastructure(unittest.TestCase):
    """Verifies configuration management and environment defaults."""

    def test_default_config_values(self):
        self.assertIsNotNone(CONFIG.target_url)
        self.assertIsNotNone(CONFIG.ws_url)
        self.assertEqual(CONFIG.cyton_fs_hz, 125)
        self.assertEqual(CONFIG.model_fs_hz, 256)
        self.assertEqual(CONFIG.model_window_samples, 1280)
        self.assertEqual(CONFIG.model_window_channels, 22)
        self.assertEqual(len(CONFIG.cyton_channels), 16)
        self.assertEqual(len(CONFIG.tcp_montage_channels), 22)
        self.assertEqual(CONFIG.openbci_packet_size, 33)
        self.assertEqual(CONFIG.openbci_start_byte, 0xA0)
        self.assertEqual(CONFIG.ws_binary_header_size, 16)

    def test_set_config_updates(self):
        orig_url = CONFIG.target_url
        try:
            set_config(target_url="http://127.0.0.1:9999/")
            self.assertEqual(CONFIG.target_url, "http://127.0.0.1:9999")
        finally:
            set_config(target_url=orig_url)


class TestHttpClientInfrastructure(unittest.TestCase):
    """Verifies HttpResponse tuple unpacking, properties, and TestHttpClient setup."""

    def test_http_response_unpacking(self):
        resp = HttpResponse(
            status_code=200,
            json_data={"status": "ok", "model": "TUSZ"},
            content=b'{"status": "ok", "model": "TUSZ"}',
            headers={"content-type": "application/json"},
            url="http://localhost:8000/health",
        )
        self.assertTrue(resp.ok)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")
        self.assertIn("status", resp.text)

        # Verify tuple unpacking (status_code, json_data, raw_bytes)
        code, body, raw = resp
        self.assertEqual(code, 200)
        self.assertEqual(body["model"], "TUSZ")
        self.assertEqual(raw, b'{"status": "ok", "model": "TUSZ"}')

    def test_http_client_initialization_and_auth_headers(self):
        client = TestHttpClient(base_url="http://testserver:8000", access_key="my-secret-key")
        self.assertEqual(client.base_url, "http://testserver:8000")
        self.assertEqual(client.access_key, "my-secret-key")


class MockSocket:
    """In-memory mock socket for RFC 6455 client unit testing without OS networking."""

    def __init__(self, incoming: bytes = b""):
        self.incoming = bytearray(incoming)
        self.sent = bytearray()
        self.closed = False

    def sendall(self, data: bytes):
        if self.closed:
            raise BrokenPipeError("Socket is closed")
        self.sent.extend(data)

    def recv(self, bufsize: int) -> bytes:
        if self.closed or not self.incoming:
            return b""
        chunk = self.incoming[:bufsize]
        del self.incoming[:bufsize]
        return bytes(chunk)

    def settimeout(self, timeout):
        pass

    def close(self):
        self.closed = True


class TestRFC6455ClientFraming(unittest.TestCase):
    """Verifies RFC 6455 handshake key calculation, frame masking, and frame round-trips."""

    def test_rfc6455_accept_header_calculation(self):
        # RFC 6455 §1.3 Official Test Vector:
        # Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ== -> s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        expected_accept = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        actual_accept = RFC6455Client.compute_accept_header(key)
        self.assertEqual(actual_accept, expected_accept)

    def test_mask_application_and_inversion(self):
        payload = b"Hello, OpenBCI WebSocket Frame!"
        mask_key = b"\x12\x34\x56\x78"
        masked = RFC6455Client.apply_mask(payload, mask_key)
        self.assertNotEqual(masked, payload)
        unmasked = RFC6455Client.apply_mask(masked, mask_key)
        self.assertEqual(unmasked, payload)

    def test_text_frame_encoding_and_decoding(self):
        text = '{"type": "ready", "session_id": "test-uuid-123"}'
        payload = text.encode("utf-8")
        mask_key = b"\xaa\xbb\xcc\xdd"

        # Masked client frame
        frame_bytes = RFC6455Client.encode_frame(OP_TEXT, payload, mask_key=mask_key)
        opcode, decoded_payload, consumed, fin = RFC6455Client.decode_frame(frame_bytes)

        self.assertEqual(opcode, OP_TEXT)
        self.assertEqual(decoded_payload.decode("utf-8"), text)
        self.assertEqual(consumed, len(frame_bytes))
        self.assertTrue(fin)

    def test_binary_frame_extended_lengths(self):
        # Test 16-bit extended length (126 <= len <= 65535)
        medium_payload = bytes([i % 256 for i in range(1984)])  # 31 * 16 * 4 bytes
        frame_medium = RFC6455Client.encode_frame(OP_BINARY, medium_payload)
        opcode, decoded, consumed, fin = RFC6455Client.decode_frame(frame_medium)
        self.assertEqual(opcode, OP_BINARY)
        self.assertEqual(decoded, medium_payload)
        self.assertEqual(consumed, len(frame_medium))

        # Test 64-bit extended length (> 65535)
        large_payload = b"X" * 70000
        frame_large = RFC6455Client.encode_frame(OP_BINARY, large_payload)
        opcode, decoded_large, consumed_large, _ = RFC6455Client.decode_frame(frame_large)
        self.assertEqual(opcode, OP_BINARY)
        self.assertEqual(len(decoded_large), 70000)

    def test_control_frames(self):
        # Ping
        ping_frame = RFC6455Client.encode_frame(OP_PING, b"heartbeat")
        op, data, _, _ = RFC6455Client.decode_frame(ping_frame)
        self.assertEqual(op, OP_PING)
        self.assertEqual(data, b"heartbeat")

        # Pong
        pong_frame = RFC6455Client.encode_frame(OP_PONG, b"heartbeat")
        op, data, _, _ = RFC6455Client.decode_frame(pong_frame)
        self.assertEqual(op, OP_PONG)

        # Close with status code 1000
        close_payload = struct.pack("!H", 1000) + b"Normal closure"
        close_frame = RFC6455Client.encode_frame(OP_CLOSE, close_payload)
        op, data, _, _ = RFC6455Client.decode_frame(close_frame)
        self.assertEqual(op, OP_CLOSE)
        code = struct.unpack("!H", data[:2])[0]
        self.assertEqual(code, 1000)

    def test_mock_socket_client_send_and_receive(self):
        """Tests client send/receive, masking, and ping-pong over mock socket."""
        # Prepare incoming server response: a Ping frame followed by a Text JSON frame
        ping_payload = b"server-ping-123"
        ping_frame = RFC6455Client.encode_frame(OP_PING, ping_payload, mask_key=None)

        json_msg = '{"type": "ready", "session_id": "sess-xyz"}'
        text_frame = RFC6455Client.encode_frame(OP_TEXT, json_msg.encode("utf-8"), mask_key=None)

        mock_sock = MockSocket(incoming=ping_frame + text_frame)

        client = RFC6455Client("ws://localhost:8000/ws/session/sess-xyz")
        client.sock = mock_sock
        client.closed = False

        # Client sends masked text message
        client.send_text('{"type": "hello", "role": "producer"}')
        self.assertTrue(len(mock_sock.sent) > 0)
        # Verify sent frame was masked
        op, sent_payload, _, _ = RFC6455Client.decode_frame(bytes(mock_sock.sent))
        self.assertEqual(op, OP_TEXT)
        self.assertEqual(sent_payload.decode("utf-8"), '{"type": "hello", "role": "producer"}')
        mock_sock.sent.clear()

        # Client receives message: ping should be answered with pong, returning the text frame
        received_obj = client.recv_json()
        self.assertEqual(received_obj["type"], "ready")
        self.assertEqual(received_obj["session_id"], "sess-xyz")

        # Verify client sent a Pong in response to the Ping
        self.assertTrue(len(mock_sock.sent) > 0)
        pong_op, pong_data, _, _ = RFC6455Client.decode_frame(bytes(mock_sock.sent))
        self.assertEqual(pong_op, OP_PONG)
        self.assertEqual(pong_data, ping_payload)

        # Close client
        client.close(1000, "Normal closure")
        self.assertTrue(client.closed)


class TestEegStreamerInfrastructure(unittest.TestCase):
    """Verifies OpenBCI 33-byte packet packing/unpacking, WS binary framing, and synthetic signals."""

    def test_openbci_packet_pack_unpack(self):
        sample_counter = 42
        channels_uv = [10.5, -25.2, 0.0, 100.0, -150.5, 33.3, -4.2, 50.0]
        aux_values = [100, -200, 300]

        packet = EegStreamer.pack_openbci_packet(
            sample_counter=sample_counter,
            channels_uv=channels_uv,
            aux_values=aux_values,
        )

        self.assertEqual(len(packet), 33)
        self.assertEqual(packet[0], 0xA0)
        self.assertEqual(packet[1], 42)
        self.assertEqual(packet[32], 0xC0)

        unpacked = EegStreamer.unpack_openbci_packet(packet)
        self.assertEqual(unpacked["counter"], 42)
        self.assertEqual(unpacked["aux"], [100, -200, 300])

        for orig, recovered in zip(channels_uv, unpacked["channels_uv"]):
            # Quantization error must be <= 0.02235 uV
            self.assertAlmostEqual(orig, recovered, delta=0.03)

    def test_openbci_daisy_packet_pairing(self):
        ch16 = [float(i * 10) for i in range(16)]
        pkt1, pkt2 = EegStreamer.pack_daisy_pair(
            sample_counter_pair_start=1,
            channels_16_uv=ch16,
        )
        self.assertEqual(len(pkt1), 33)
        self.assertEqual(len(pkt2), 33)
        self.assertEqual(pkt1[1], 1)
        self.assertEqual(pkt2[1], 2)

        up1 = EegStreamer.unpack_openbci_packet(pkt1)
        up2 = EegStreamer.unpack_openbci_packet(pkt2)
        recovered_16 = up1["channels_uv"] + up2["channels_uv"]
        self.assertEqual(len(recovered_16), 16)
        for orig, rec in zip(ch16, recovered_16):
            self.assertAlmostEqual(orig, rec, delta=0.03)

    def test_ws_binary_frame_pack_unpack(self):
        seq = 105
        t0_ms = 1718000000123.5
        n_samples = 31
        n_channels = 16
        raw_data = np.random.randn(n_samples, n_channels).astype(np.float32)

        frame = EegStreamer.pack_ws_binary_frame(
            seq=seq, t0_ms=t0_ms, data=raw_data, flags=0x01
        )
        self.assertEqual(len(frame), 16 + (31 * 16 * 4))

        header, recovered_data = EegStreamer.unpack_ws_binary_frame(frame)
        self.assertEqual(header["seq"], 105)
        self.assertAlmostEqual(header["t0_ms"], t0_ms, places=2)
        self.assertEqual(header["n_samples"], 31)
        self.assertEqual(header["n_channels"], 16)
        self.assertEqual(header["flags"], 1)
        self.assertTrue(header["is_gap"])

        np.testing.assert_array_almost_equal(raw_data, recovered_data)

    def test_synthetic_signal_states(self):
        # 1. Normal state
        norm = EegStreamer.generate_signal(state="normal", duration_sec=2.0, fs=125, n_channels=16)
        self.assertEqual(norm.shape, (250, 16))
        self.assertTrue(10.0 < np.std(norm) < 60.0)

        # 2. Pre-seizure state
        pre = EegStreamer.generate_signal(state="pre_seizure", duration_sec=2.0, fs=125, n_channels=16)
        self.assertEqual(pre.shape, (250, 16))
        self.assertTrue(np.std(pre) > 15.0)

        # 3. Seizure state (high amplitude epileptiform discharges)
        seiz = EegStreamer.generate_signal(state="seizure", duration_sec=2.0, fs=125, n_channels=16)
        self.assertEqual(seiz.shape, (250, 16))
        self.assertTrue(np.std(seiz) > 50.0)
        self.assertTrue(np.max(np.abs(seiz)) > 100.0)

        # 4. Flatline
        flat = EegStreamer.generate_signal(state="flat", duration_sec=2.0, fs=125, n_channels=16)
        self.assertTrue(np.std(flat) < 0.5)

        # 5. Railed
        railed = EegStreamer.generate_signal(state="railed", duration_sec=2.0, fs=125, n_channels=16)
        self.assertTrue(np.max(np.abs(railed)) > 180000.0)

        # 6. Model window (1280, 22)
        window = EegStreamer.generate_model_window(state="normal")
        self.assertEqual(window.shape, (1280, 22))


class TestAssertionsAndFixtures(unittest.TestCase):
    """Verifies assertion helpers and reusable test fixtures."""

    def test_assert_probability_distribution(self):
        valid_probs = {"normal": 0.85, "pre_seizure": 0.10, "seizure": 0.05}
        assert_probability_distribution(valid_probs)

        invalid_probs = {"normal": 0.85, "pre_seizure": 0.10, "seizure": 0.20}
        with self.assertRaises(AssertionError):
            assert_probability_distribution(invalid_probs)

    def test_assert_montage_shape(self):
        good = np.zeros((1280, 22), dtype=np.float32)
        assert_montage_shape(good, expected_channels=22, expected_samples=1280)

        bad = np.zeros((1000, 16), dtype=np.float32)
        with self.assertRaises(AssertionError):
            assert_montage_shape(bad, expected_channels=22, expected_samples=1280)

    def test_assert_latency_within(self):
        assert_latency_within(12.5, max_allowed_ms=30.0)
        with self.assertRaises(AssertionError):
            assert_latency_within(45.0, max_allowed_ms=30.0)

    def test_signal_quality_evaluator(self):
        good_sig = EegStreamer.generate_signal("normal", duration_sec=2.0)
        assert_signal_quality(good_sig, "good")

        flat_sig = EegStreamer.generate_signal("flat", duration_sec=2.0)
        assert_signal_quality(flat_sig, "flat")

        railed_sig = EegStreamer.generate_signal("railed", duration_sec=2.0)
        assert_signal_quality(railed_sig, "railed")

    def test_fixtures_generation(self):
        sid = random_session_id()
        self.assertEqual(len(sid), 36)

        key = random_access_key(32)
        self.assertEqual(len(key), 32)

        tensor = golden_model_input_tensor(1337)
        self.assertEqual(tensor.shape, (1, 1280, 22))

        profile = get_openbci_cyton_profile()
        self.assertEqual(profile["profile_id"], "openbci-cyton-daisy")
        self.assertEqual(len(profile["channels"]), 16)

        stream = generate_openbci_raw_stream(n_samples=5)
        self.assertEqual(len(stream), 5 * 2 * 33)

        frame_bytes = generate_mock_binary_frame(seq=1)
        assert_valid_ws_binary_frame(frame_bytes, expected_channels=16)


if __name__ == "__main__":
    unittest.main()
