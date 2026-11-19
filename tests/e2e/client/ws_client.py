"""Pure-Python RFC 6455 WebSocket Client.

Zero third-party dependencies: built strictly using standard library `socket`,
`struct`, `base64`, `hashlib`, `ssl`, and `urllib.parse`.
Handles handshake, frame masking/unmasking, control frames (ping/pong/close),
and text/binary message framing for Seizure-Sentry E2E tests.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import urllib.parse
from typing import Any, Dict, Optional, Tuple, Union

# RFC 6455 Constants
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


class WebSocketError(Exception):
    """Base exception for WebSocket protocol and communication errors."""


class WebSocketCloseError(WebSocketError):
    """Raised when the WebSocket connection is closed by the server."""

    def __init__(self, code: int, reason: str = ""):
        super().__init__(f"WebSocket closed with code {code}: {reason}")
        self.code = code
        self.reason = reason


class RFC6455Client:
    """Synchronous RFC 6455 WebSocket client with zero external dependencies."""

    def __init__(
        self,
        url: str,
        access_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
    ):
        self.url = url
        self.access_key = access_key
        self.custom_headers = headers or {}
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.closed: bool = True
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self._read_buffer = bytearray()

    @staticmethod
    def compute_accept_header(sec_key: str) -> str:
        """Computes the expected Sec-WebSocket-Accept response header."""
        sha1 = hashlib.sha1((sec_key + WS_GUID).encode("ascii")).digest()
        return base64.b64encode(sha1).decode("ascii")

    @staticmethod
    def apply_mask(payload: bytes, mask_key: bytes) -> bytes:
        """Applies or removes the 4-byte XOR mask on a payload."""
        if not mask_key or len(mask_key) != 4:
            raise ValueError("Mask key must be exactly 4 bytes")
        # Fast bytearray masking
        res = bytearray(payload)
        for i in range(len(res)):
            res[i] ^= mask_key[i % 4]
        return bytes(res)

    @classmethod
    def encode_frame(
        cls,
        opcode: int,
        payload: bytes,
        mask_key: Optional[bytes] = None,
        fin: bool = True,
    ) -> bytes:
        """Encodes an RFC 6455 WebSocket frame."""
        b0 = (0x80 if fin else 0) | (opcode & 0x0F)
        length = len(payload)
        is_masked = mask_key is not None
        mask_bit = 0x80 if is_masked else 0

        if length <= 125:
            header = struct.pack("!BB", b0, mask_bit | length)
        elif length <= 0xFFFF:
            header = struct.pack("!BBH", b0, mask_bit | 126, length)
        else:
            header = struct.pack("!BBQ", b0, mask_bit | 127, length)

        if is_masked:
            if len(mask_key) != 4:
                raise ValueError("Mask key must be exactly 4 bytes")
            masked_payload = cls.apply_mask(payload, mask_key)
            return header + mask_key + masked_payload
        return header + payload

    @classmethod
    def decode_frame(cls, data: bytes) -> Tuple[int, bytes, int, bool]:
        """Decodes an RFC 6455 frame from raw bytes.

        Returns (opcode, payload, total_bytes_consumed, fin_flag).
        Raises ValueError if buffer does not contain a complete frame.
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for WebSocket header")

        b0, b1 = struct.unpack("!BB", data[:2])
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        is_masked = bool(b1 & 0x80)
        payload_len = b1 & 0x7F
        offset = 2

        if payload_len == 126:
            if len(data) < 4:
                raise ValueError("Insufficient data for 16-bit payload length")
            payload_len = struct.unpack("!H", data[2:4])[0]
            offset = 4
        elif payload_len == 127:
            if len(data) < 10:
                raise ValueError("Insufficient data for 64-bit payload length")
            payload_len = struct.unpack("!Q", data[2:10])[0]
            offset = 10

        mask_key = None
        if is_masked:
            if len(data) < offset + 4:
                raise ValueError("Insufficient data for masking key")
            mask_key = data[offset : offset + 4]
            offset += 4

        total_frame_len = offset + payload_len
        if len(data) < total_frame_len:
            raise ValueError(
                f"Incomplete frame payload: have {len(data) - offset}, need {payload_len}"
            )

        raw_payload = data[offset:total_frame_len]
        if is_masked and mask_key:
            payload = cls.apply_mask(raw_payload, mask_key)
        else:
            payload = raw_payload

        return opcode, payload, total_frame_len, fin

    def connect(self, timeout: Optional[float] = None) -> RFC6455Client:
        """Establishes TCP connection and performs RFC 6455 upgrade handshake."""
        eff_timeout = timeout if timeout is not None else self.timeout
        parsed = urllib.parse.urlparse(self.url)
        scheme = parsed.scheme.lower()
        if scheme not in ("ws", "wss"):
            raise ValueError(f"Unsupported WebSocket scheme: {scheme} (must be ws or wss)")

        use_ssl = scheme == "wss"
        default_port = 443 if use_ssl else 80
        host = parsed.hostname or "localhost"
        port = parsed.port or default_port
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        # 1. Connect TCP socket
        raw_sock = socket.create_connection((host, port), timeout=eff_timeout)
        if use_ssl:
            ctx = ssl.create_default_context()
            self.sock = ctx.wrap_socket(raw_sock, server_hostname=host)
        else:
            self.sock = raw_sock

        self.sock.settimeout(eff_timeout)

        # 2. Build HTTP 101 Handshake request
        nonce = os.urandom(16)
        sec_key = base64.b64encode(nonce).decode("ascii")

        port_str = f":{port}" if port not in (80, 443) else ""
        headers = {
            "Host": f"{host}{port_str}",
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": sec_key,
            "Sec-WebSocket-Version": "13",
            "User-Agent": "SeizureSentry-RFC6455Client/1.0",
        }
        if self.access_key:
            headers["X-Access-Key"] = self.access_key
        headers.update(self.custom_headers)

        req_lines = [f"GET {path} HTTP/1.1"]
        for k, v in headers.items():
            req_lines.append(f"{k}: {v}")
        req_lines.extend(["", ""])
        handshake_data = "\r\n".join(req_lines).encode("ascii")

        self.sock.sendall(handshake_data)

        # 3. Read HTTP 101 Response
        resp_buf = bytearray()
        while b"\r\n\r\n" not in resp_buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WebSocketError("Server closed connection during handshake")
            resp_buf.extend(chunk)

        header_bytes, remainder = resp_buf.split(b"\r\n\r\n", 1)
        self._read_buffer.extend(remainder)

        header_lines = header_bytes.decode("utf-8", errors="replace").split("\r\n")
        status_line = header_lines[0]
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or parts[1] != "101":
            raise WebSocketError(f"WebSocket upgrade rejected by server: {status_line}")

        resp_headers = {}
        for line in header_lines[1:]:
            if ":" in line:
                hk, hv = line.split(":", 1)
                resp_headers[hk.strip().lower()] = hv.strip()

        # Validate Sec-WebSocket-Accept
        expected_accept = self.compute_accept_header(sec_key)
        actual_accept = resp_headers.get("sec-websocket-accept", "")
        if actual_accept != expected_accept:
            raise WebSocketError(
                f"Sec-WebSocket-Accept mismatch: expected {expected_accept}, got {actual_accept}"
            )

        self.closed = False
        return self

    def _read_exact(self, num_bytes: int, timeout: Optional[float] = None) -> bytes:
        """Reads exactly num_bytes from buffer and socket."""
        if timeout is not None and self.sock:
            self.sock.settimeout(timeout)

        while len(self._read_buffer) < num_bytes:
            if not self.sock:
                raise WebSocketError("Socket is not connected")
            chunk = self.sock.recv(max(4096, num_bytes - len(self._read_buffer)))
            if not chunk:
                raise ConnectionResetError("Connection closed while awaiting data")
            self._read_buffer.extend(chunk)

        result = bytes(self._read_buffer[:num_bytes])
        del self._read_buffer[:num_bytes]
        return result

    def send_frame(
        self, opcode: int, payload: bytes, mask: bool = True, fin: bool = True
    ) -> None:
        """Sends an RFC 6455 frame. Client-to-server frames are masked by default."""
        if self.closed or not self.sock:
            raise WebSocketError("Cannot send on closed WebSocket")
        mask_key = os.urandom(4) if mask else None
        frame_bytes = self.encode_frame(opcode, payload, mask_key=mask_key, fin=fin)
        self.sock.sendall(frame_bytes)

    def send_text(self, text: str) -> None:
        """Sends a text message (UTF-8)."""
        self.send_frame(OP_TEXT, text.encode("utf-8"))

    def send_json(self, data: Any) -> None:
        """Serializes and sends an object as a JSON text message."""
        self.send_text(json.dumps(data))

    def send_binary(self, data: bytes) -> None:
        """Sends a binary frame."""
        self.send_frame(OP_BINARY, data)

    def send_ping(self, payload: bytes = b"") -> None:
        """Sends a Ping control frame."""
        self.send_frame(OP_PING, payload)

    def send_pong(self, payload: bytes = b"") -> None:
        """Sends a Pong control frame."""
        self.send_frame(OP_PONG, payload)

    def recv_frame(self, timeout: Optional[float] = None) -> Tuple[int, bytes]:
        """Receives the next complete message (reassembling fragments if needed).

        Automatically responds to Ping with Pong.
        """
        message_fragments = bytearray()
        message_opcode = None

        while True:
            # 1. Read first 2 bytes of frame header
            b0, b1 = struct.unpack("!BB", self._read_exact(2, timeout=timeout))
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            is_masked = bool(b1 & 0x80)
            payload_len = b1 & 0x7F

            # 2. Extended length
            if payload_len == 126:
                payload_len = struct.unpack("!H", self._read_exact(2, timeout=timeout))[0]
            elif payload_len == 127:
                payload_len = struct.unpack("!Q", self._read_exact(8, timeout=timeout))[0]

            # 3. Mask key if present
            mask_key = self._read_exact(4, timeout=timeout) if is_masked else None

            # 4. Read payload
            raw_payload = self._read_exact(payload_len, timeout=timeout)
            payload = self.apply_mask(raw_payload, mask_key) if is_masked and mask_key else raw_payload

            # 5. Control frames
            if opcode == OP_CLOSE:
                code = 1000
                reason = ""
                if len(payload) >= 2:
                    code = struct.unpack("!H", payload[:2])[0]
                    reason = payload[2:].decode("utf-8", errors="replace")
                self.close_code = code
                self.close_reason = reason
                self.closed = True
                try:
                    # Echo close frame if we haven't sent one
                    self.send_frame(OP_CLOSE, payload)
                except Exception:
                    pass
                if self.sock:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                raise WebSocketCloseError(code, reason)

            if opcode == OP_PING:
                self.send_pong(payload)
                continue  # Ping handled, await next frame

            if opcode == OP_PONG:
                continue  # Pong received, await next frame

            # 6. Data frames
            if opcode != OP_CONTINUATION:
                message_opcode = opcode

            message_fragments.extend(payload)

            if fin:
                if message_opcode is None:
                    raise WebSocketError("Received continuation frame without initial opcode")
                return message_opcode, bytes(message_fragments)

    def recv_text(self, timeout: Optional[float] = None) -> str:
        """Receives a text message and decodes it as UTF-8."""
        opcode, data = self.recv_frame(timeout=timeout)
        if opcode != OP_TEXT:
            raise WebSocketError(f"Expected text frame (0x1), received opcode 0x{opcode:x}")
        return data.decode("utf-8", errors="replace")

    def recv_json(self, timeout: Optional[float] = None) -> Any:
        """Receives a text frame and parses it as JSON."""
        text = self.recv_text(timeout=timeout)
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"Received WebSocket message is not valid JSON: {text[:200]}") from e

    def recv_binary(self, timeout: Optional[float] = None) -> bytes:
        """Receives a binary frame and returns raw bytes."""
        opcode, data = self.recv_frame(timeout=timeout)
        if opcode != OP_BINARY:
            raise WebSocketError(f"Expected binary frame (0x2), received opcode 0x{opcode:x}")
        return data

    def close(self, code: int = 1000, reason: str = "") -> None:
        """Gracefully initiates close handshake and closes underlying socket."""
        if self.closed:
            return
        self.closed = True
        try:
            payload = struct.pack("!H", code) + reason.encode("utf-8")
            self.send_frame(OP_CLOSE, payload)
        except Exception:
            pass
        finally:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

    def __enter__(self) -> RFC6455Client:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
