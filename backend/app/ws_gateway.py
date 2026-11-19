"""
backend/app/ws_gateway.py
WebSocket streaming gateway: connection management, binary frame ingestion,
isolated per-session DSP/inference pipelines, and fan-out broadcasting.
"""
from __future__ import annotations

import time
import uuid
import struct
import logging
import asyncio
from typing import Dict, Set, Optional, Tuple, List, Any
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.config import settings
from app.adapter import EegAdapter
from app.alert_engine import HysteresisAlertEngine, ArtifactGuard
from app.db import record_prediction, record_alert, update_alert_end

logger = logging.getLogger("seizure_sentry.ws_gateway")

# Binary frame constants
HEADER_FORMAT = "<IdHBB"
HEADER_SIZE = 16  # 4 (uint32) + 8 (float64) + 2 (uint16) + 1 (uint8) + 1 (uint8)
MAX_FRAME_SIZE = 65536  # 64 KB limit

# Close codes
WS_CLOSE_NORMAL = 1000
WS_CLOSE_POLICY_VIOLATION = 1008
WS_CLOSE_MESSAGE_TOO_BIG = 1009
WS_CLOSE_AUTH_REQUIRED = 4001
WS_CLOSE_FORBIDDEN = 4003


class ClientHello(BaseModel):
    type: str = "hello"
    key: Optional[str] = None
    role: str = "producer"  # "producer" | "viewer"
    source: Optional[str] = "device"  # "device" | "replay" | "bridge" | "synthetic"
    profile_id: Optional[str] = "openbci-cyton-daisy"
    fs: Optional[int] = 125
    channels: Optional[List[str]] = None


class BinaryHeader:
    __slots__ = ("seq", "t0_ms", "n_samples", "n_channels", "flags")

    def __init__(self, seq: int, t0_ms: float, n_samples: int, n_channels: int, flags: int):
        self.seq = seq
        self.t0_ms = t0_ms
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.flags = flags


def unpack_binary_frame(data: bytes) -> Tuple[BinaryHeader, np.ndarray]:
    """
    Unpack 16-byte little-endian header (<IdHBB) and float32 microvolts payload.
    Validates byte length and rejects NaN / Inf values.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Frame length {len(data)} shorter than 16-byte header")

    if len(data) > MAX_FRAME_SIZE:
        raise ValueError(f"Frame length {len(data)} exceeds maximum allowed {MAX_FRAME_SIZE} bytes")

    seq, t0_ms, n_samples, n_channels, flags = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    expected_bytes = HEADER_SIZE + n_samples * n_channels * 4

    if len(data) != expected_bytes:
        raise ValueError(
            f"Payload size mismatch: expected {expected_bytes} bytes "
            f"({n_samples}x{n_channels}x4), got {len(data)} bytes"
        )

    payload = np.frombuffer(data[HEADER_SIZE:], dtype=np.float32).reshape(n_samples, n_channels)

    if np.isnan(payload).any() or np.isinf(payload).any():
        raise ValueError("Payload contains NaN or Inf values")

    return BinaryHeader(seq, t0_ms, n_samples, n_channels, flags), payload


class SessionDiagnostics:
    """End-to-End Hop Diagnostics Counters (PRD FR-27)."""
    def __init__(self):
        self.hop1_device_packets: int = 0
        self.hop2_chunks_sent: int = 0
        self.hop3_frames_received: int = 0
        self.hop4_windows_inferred: int = 0
        self.hop5_predictions_displayed: int = 0
        self.dropped_chunks: int = 0
        self.last_seq: Optional[int] = None
        self.last_latency_ms: float = 0.0

    @property
    def health(self) -> str:
        return "ok" if self.dropped_chunks == 0 else "degraded"

    def record_received_frame(self, seq: int, byte_count: int, flags: int) -> None:
        self.hop2_chunks_sent += 1
        self.hop3_frames_received += 1
        self.hop1_device_packets += 1  # 1 frame ~ chunk

        if self.last_seq is not None:
            gap = seq - (self.last_seq + 1)
            if gap > 0:
                self.dropped_chunks += gap
        self.last_seq = seq

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop1_device_packets": self.hop1_device_packets,
            "hop2_chunks_sent": self.hop2_chunks_sent,
            "hop3_frames_received": self.hop3_frames_received,
            "hop4_windows_inferred": self.hop4_windows_inferred,
            "hop5_predictions_displayed": self.hop5_predictions_displayed,
            "dropped_chunks": self.dropped_chunks,
            "health": "ok" if self.dropped_chunks == 0 else "degraded",
            "last_latency_ms": round(self.last_latency_ms, 2),
        }


class SessionState:
    """
    Isolated session pipeline encapsulating dedicated EegAdapter,
    rolling resampled buffer, ArtifactGuard, HysteresisAlertEngine, and diagnostics.
    """
    def __init__(
        self,
        session_id: str,
        source: str = "device",
        fs: int = 125,
        channels: Optional[List[str]] = None,
        is_simulated: bool = False,
    ):
        self.session_id: str = session_id
        self.source: str = source
        self.source_fs: int = fs
        self.channels: Optional[List[str]] = channels
        self.is_simulated: bool = is_simulated

        # Dedicated, isolated DSP pipeline per session (Prevents Reviewer 2 state poisoning)
        self.adapter: EegAdapter = EegAdapter(from_fs=fs, to_fs=256, n_channels=22)
        self.alert_engine: HysteresisAlertEngine = HysteresisAlertEngine()
        self.artifact_guard: ArtifactGuard = ArtifactGuard()
        self.diagnostics: SessionDiagnostics = SessionDiagnostics()

        # Connection sets
        self.producer_ws: Optional[WebSocket] = None
        self.viewers: Set[WebSocket] = set()
        self.viewer_tokens: Dict[WebSocket, str] = {}

        # Rolling inference buffer holding resampled 256Hz, 22-channel float32 samples
        self.buffer: List[np.ndarray] = []
        self.total_buffered_samples: int = 0
        self.samples_since_inference: int = 0
        self.window_size: int = 1280  # 5.0 seconds at 256 Hz
        self.stride_size: int = 256   # 1.0 second stride at 256 Hz
        self.max_buffer_samples: int = 15360  # 60s memory cap

    def append_resampled(self, chunk_256: np.ndarray) -> None:
        self.buffer.append(chunk_256)
        n = chunk_256.shape[0]
        self.total_buffered_samples += n
        self.samples_since_inference += n

        # Trim old buffer chunks if exceeding 60s cap
        if self.total_buffered_samples > self.max_buffer_samples:
            overflow = self.total_buffered_samples - self.max_buffer_samples
            while self.buffer and overflow > 0:
                first = self.buffer[0]
                if first.shape[0] <= overflow:
                    overflow -= first.shape[0]
                    self.total_buffered_samples -= first.shape[0]
                    self.buffer.pop(0)
                else:
                    self.buffer[0] = first[overflow:]
                    self.total_buffered_samples -= overflow
                    overflow = 0

    def get_latest_window(self) -> Optional[np.ndarray]:
        if self.total_buffered_samples < self.window_size:
            return None
        needed = self.window_size
        chunks_to_stitch = []
        for ch in reversed(self.buffer):
            if needed <= 0:
                break
            if ch.shape[0] <= needed:
                chunks_to_stitch.append(ch)
                needed -= ch.shape[0]
            else:
                chunks_to_stitch.append(ch[-needed:])
                needed = 0
        chunks_to_stitch.reverse()
        return np.vstack(chunks_to_stitch)


class WsConnectionManager:
    """
    Manages active streaming sessions, WebSockets, binary ingestion, and broadcasts.
    """
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(
        self,
        session_id: str,
        hello: ClientHello,
        is_simulated: bool = False,
    ) -> SessionState:
        async with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionState(
                    session_id=session_id,
                    source=hello.source or "device",
                    fs=hello.fs or 125,
                    channels=hello.channels,
                    is_simulated=is_simulated,
                )
            return self.sessions[session_id]

    async def register_connection(
        self,
        ws: WebSocket,
        session: SessionState,
        role: str,
        token: Optional[str] = None,
    ) -> None:
        async with self._lock:
            if role == "producer":
                if session.producer_ws is not None and session.producer_ws != ws:
                    try:
                        await session.producer_ws.close(code=WS_CLOSE_POLICY_VIOLATION)
                    except Exception:
                        pass
                session.producer_ws = ws
            else:
                session.viewers.add(ws)
                if token:
                    session.viewer_tokens[ws] = token

    async def unregister_connection(self, ws: WebSocket, session_id: str) -> None:
        async with self._lock:
            if session_id not in self.sessions:
                return
            session = self.sessions[session_id]
            if session.producer_ws == ws:
                session.producer_ws = None
                # Broadcast producer disconnected status to remaining viewers
                await self.broadcast(session_id, {
                    "type": "status",
                    "session_id": session_id,
                    "link_status": "disconnected",
                    "state": "producer_disconnected",
                    "timestamp": time.time(),
                })
            session.viewers.discard(ws)
            session.viewer_tokens.pop(ws, None)

            # Cleanup session if completely empty
            if session.producer_ws is None and len(session.viewers) == 0:
                del self.sessions[session_id]

    async def broadcast(self, session_id: str, message: dict, skip_ws: Optional[WebSocket] = None) -> None:
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]
        recipients = list(session.viewers)
        if session.producer_ws is not None:
            recipients.append(session.producer_ws)

        dead_sockets = []
        for ws in recipients:
            if ws == skip_ws:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.append(ws)

        for dead in dead_sockets:
            session.viewers.discard(dead)
            session.viewer_tokens.pop(dead, None)
            if session.producer_ws == dead:
                session.producer_ws = None

    async def disconnect_viewers(self, session_id: str, token: Optional[str] = None) -> None:
        """Disconnect viewers when token revoked."""
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]
        to_close = []
        for ws, t in list(session.viewer_tokens.items()):
            if token is None or t == token:
                to_close.append(ws)

        # If token was not tracked per-socket, disconnect all viewers
        if token is None and not to_close:
            to_close = list(session.viewers)

        for ws in to_close:
            try:
                await ws.close(code=WS_CLOSE_POLICY_VIOLATION)
            except Exception:
                pass
            session.viewers.discard(ws)
            session.viewer_tokens.pop(ws, None)

    async def run_session_inference(
        self,
        session: SessionState,
        window: np.ndarray,
        t0_ms: float,
        model_runner: Any,
    ) -> None:
        """
        Evaluate Artifact Guard, run ModelRunner inference, update AlertEngine,
        persist records to SQLite, and broadcast prediction/alert JSON.
        """
        t_start = time.perf_counter()
        now = time.time()

        # Step 1: Artifact Guard evaluation
        guard_res = session.artifact_guard.evaluate(window, timestamp=now, fs=256)
        artifact_active = guard_res["artifact_active"]

        if artifact_active:
            # Broadcast artifact status banner
            await self.broadcast(session.session_id, {
                "type": "status",
                "session_id": session.session_id,
                "artifact_guard_active": True,
                "predictions_paused": True,
                "banner_message": guard_res["banner_message"],
                "quality": guard_res["qualities"],
                "timestamp": now,
            })
            # When artifact guard is engaged, mute alerts and do not trigger new alerts
            session.alert_engine.set_artifact_guard(True)
            return

        session.alert_engine.set_artifact_guard(False)

        # Step 2: In-process ML inference
        if model_runner is None or not getattr(model_runner, "_is_loaded", False):
            return

        try:
            # Inference runs synchronously under ModelRunner's internal lock
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, model_runner.predict, window)
        except Exception as e:
            logger.error(f"Inference error in session {session.session_id}: {e}")
            return

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        session.diagnostics.last_latency_ms = elapsed_ms
        session.diagnostics.hop4_windows_inferred += 1

        probs = result.get("probabilities", {"normal": 0.0, "pre_seizure": 0.0, "seizure": 0.0})
        label = result.get("prediction", "Normal")

        # Step 3: Hysteresis Alert Engine state machine
        current_state, new_alert, cleared_alert = session.alert_engine.process_prediction(
            probs, timestamp=now, artifact_suppressed=False
        )

        window_start = max(0.0, (t0_ms / 1000.0) - 5.0)
        window_end = t0_ms / 1000.0

        # Step 4: Persist to SQLite
        try:
            record_prediction(
                session_id=session.session_id,
                t_start=window_start,
                t_end=window_end,
                probs=probs,
                label=label,
                state=current_state,
            )
            if new_alert:
                record_alert(
                    alert_id=new_alert["id"],
                    session_id=session.session_id,
                    level=new_alert["level"],
                    started_at=new_alert["started_at"],
                    peak_prob=new_alert["peak_prob"],
                    muted=new_alert.get("muted", False),
                )
            if cleared_alert:
                update_alert_end(
                    alert_id=cleared_alert["id"],
                    ended_at=cleared_alert["ended_at"] or now,
                    peak_prob=cleared_alert.get("peak_prob"),
                )
        except Exception as e:
            logger.error(f"Database write error for session {session.session_id}: {e}")

        # Step 5: Broadcast Alerts (if any state transition)
        if new_alert:
            await self.broadcast(session.session_id, {
                "type": "alert",
                "id": new_alert["id"],
                "session_id": session.session_id,
                "level": new_alert["level"],
                "started_at": new_alert["started_at"],
                "ended_at": None,
                "peak_prob": round(new_alert["peak_prob"], 4),
                "muted": new_alert.get("muted", False),
                "ui_banner": new_alert.get("ui_banner", True),
                "red_border_pulse": new_alert.get("red_border_pulse", False),
                "sound_alarm": new_alert.get("sound_alarm", False),
                "simulated": session.is_simulated,
            })

        if cleared_alert:
            await self.broadcast(session.session_id, {
                "type": "alert",
                "id": cleared_alert["id"],
                "session_id": session.session_id,
                "level": cleared_alert["level"],
                "started_at": cleared_alert["started_at"],
                "ended_at": cleared_alert["ended_at"],
                "peak_prob": round(cleared_alert["peak_prob"], 4),
                "muted": False,
                "ui_banner": False,
                "red_border_pulse": False,
                "sound_alarm": False,
                "simulated": session.is_simulated,
            })

        # Step 6: Broadcast Prediction
        pred_msg = {
            "type": "prediction",
            "session_id": session.session_id,
            "window_start": window_start,
            "window_end": window_end,
            "probs": probs,
            "probabilities": probs,
            "label": label,
            "state": label,  # normal/pre-seizure/seizure state label
            "simulated": session.is_simulated,
        }
        await self.broadcast(session.session_id, pred_msg)
        session.diagnostics.hop5_predictions_displayed += 1
