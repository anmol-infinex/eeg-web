"""
backend/app/simulator.py
Replay Simulator and Preset Seizure Demo Streamer.
Paces 250ms chunks at real-time speeds with permanent 'SIMULATED DATA' badge.
"""
from __future__ import annotations

import time
import uuid
import struct
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from app.synthetic import generate_normal_window, generate_preseizure_window, generate_seizure_window
from app.ws_gateway import HEADER_FORMAT, HEADER_SIZE

logger = logging.getLogger("seizure_sentry.simulator")

RECORDINGS = [
    {
        "recording_id": "rec-normal-baseline",
        "filename": "normal_resting_10min.edf",
        "duration_sec": 600.0,
        "has_seizure": False,
        "sample_rate": 125,
        "channels": 16,
    },
    {
        "recording_id": "preset_seizure_demo",
        "filename": "labelled_tusz_seizure_sample.edf",
        "duration_sec": 75.0,
        "has_seizure": True,
        "sample_rate": 125,
        "channels": 16,
    },
]


class PresetSeizureDemoRunner:
    """Clinical progression model for preset_seizure_demo."""

    @staticmethod
    def get_prediction_at_time(t_sec: float) -> Tuple[Dict[str, float], str]:
        """
        Timeline:
        0 - 15s: Normal baseline (P(normal) > 0.80)
        15 - 25s: Pre-seizure rhythmic theta buildup (P(pre) >= 0.60)
        25 - 60s: Generalized 3Hz paroxysmal seizure discharges (P(seiz) >= 0.70)
        60 - 75s: Post-ictal return to normal baseline
        """
        if t_sec < 15.0:
            probs = {"normal": 0.92, "pre_seizure": 0.05, "seizure": 0.03}
            state = "Normal"
        elif t_sec < 25.0:
            probs = {"normal": 0.25, "pre_seizure": 0.68, "seizure": 0.07}
            state = "Pre-Seizure"
        elif t_sec < 60.0:
            probs = {"normal": 0.04, "pre_seizure": 0.08, "seizure": 0.88}
            state = "Seizure"
        else:
            probs = {"normal": 0.88, "pre_seizure": 0.08, "seizure": 0.04}
            state = "Normal"
        return probs, state


class ReplaySimulator:
    """
    Simulates real-time hardware stream by generating 250ms chunks paced at 4 frames/sec
    scaled by speed multiplier (clamped [0.1, 10.0]).
    """
    RECORDINGS = RECORDINGS

    def __init__(self, speed: float = 1.0):
        self.speed = max(0.1, min(10.0, speed))
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self.demo_active: bool = False

    def get_recordings(self) -> List[Dict[str, Any]]:
        return list(self.RECORDINGS)

    def get_chunk_interval(self, session_id: str) -> float:
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]["base_chunk_interval"]
        return 0.25 / self.speed

    def build_demo_dataset(self) -> np.ndarray:
        """
        Build 75-second 256Hz 22-channel dataset:
        00-15s: Normal (3 x 5s windows)
        15-25s: Pre-seizure (2 x 5s windows)
        25-60s: Seizure (7 x 5s windows)
        60-75s: Normal recovery (3 x 5s windows)
        """
        normal_part = np.vstack([generate_normal_window(seed=10 + i) for i in range(3)])
        pre_part = np.vstack([generate_preseizure_window(seed=20 + i) for i in range(2)])
        seiz_part = np.vstack([generate_seizure_window(seed=30 + i) for i in range(7)])
        rec_part = np.vstack([generate_normal_window(seed=40 + i) for i in range(3)])
        return np.vstack([normal_part, pre_part, seiz_part, rec_part]).astype(np.float32)

    def start_replay(
        self,
        recording_id: str,
        speed: float = 1.0,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start or prepare replay state."""
        clamped_speed = max(0.1, min(10.0, float(speed)))
        rec = next((r for r in self.RECORDINGS if r["recording_id"] == recording_id), None)
        if not rec:
            raise ValueError(f"Recording '{recording_id}' not found")

        sess_id = session_id or str(uuid.uuid4())
        interval = 0.25 / clamped_speed

        state = {
            "session_id": sess_id,
            "recording_id": recording_id,
            "filename": rec["filename"],
            "speed": clamped_speed,
            "is_simulated": True,
            "badge": "SIMULATED DATA",
            "started_at": time.time(),
            "base_chunk_interval": interval,
            "is_running": True,
        }
        self.active_sessions[sess_id] = state

        return {
            "session_id": sess_id,
            "status": "started",
            "recording": rec["filename"],
            "speed": clamped_speed,
            "simulated": True,
            "badge": "SIMULATED DATA",
        }

    async def stream_task(
        self,
        session_id: str,
        ws_manager: Any,
        model_runner: Any,
        dataset: Optional[np.ndarray] = None,
    ) -> None:
        """Paced streaming loop executing inside asyncio event loop."""
        if dataset is None:
            dataset = self.build_demo_dataset()

        sess_info = self.active_sessions.get(session_id, {})
        speed = sess_info.get("speed", self.speed)
        chunk_interval = 0.25 / speed
        chunk_size = 64  # 250ms @ 256Hz
        total_samples = dataset.shape[0]
        seq = 0

        logger.info(f"Replay task starting for session {session_id} at {speed}x pace")

        try:
            for offset in range(0, total_samples, chunk_size):
                if session_id not in self.active_sessions or not self.active_sessions[session_id].get("is_running", True):
                    break

                t_start = time.perf_counter()
                chunk = dataset[offset : offset + chunk_size]
                if chunk.shape[0] < chunk_size:
                    pad = np.zeros((chunk_size - chunk.shape[0], chunk.shape[1]), dtype=np.float32)
                    chunk = np.vstack([chunk, pad])

                t0_ms = time.time() * 1000.0
                flags = 0x80  # Bit 7 = Simulated Data

                if session_id in ws_manager.sessions:
                    session = ws_manager.sessions[session_id]
                    session.diagnostics.record_received_frame(seq, len(chunk) * 4, flags)

                    # Append resampled/filtered chunk
                    session.append_resampled(chunk)

                    # Trigger rolling window inference
                    if (
                        session.total_buffered_samples >= session.window_size
                        and session.samples_since_inference >= session.stride_size
                    ):
                        window = session.get_latest_window()
                        if window is not None:
                            session.samples_since_inference = 0
                            asyncio.create_task(
                                ws_manager.run_session_inference(session, window, t0_ms, model_runner)
                            )

                seq += 1
                elapsed = time.perf_counter() - t_start
                sleep_sec = max(0.0, chunk_interval - elapsed)
                await asyncio.sleep(sleep_sec)

        except asyncio.CancelledError:
            logger.info(f"Replay task cancelled for {session_id}")
        finally:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["is_running"] = False
            self.demo_active = False

    def stop_session(self, session_id: str) -> bool:
        """Stop replay task for session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["is_running"] = False
        task = self._running_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
            return True
        return False
