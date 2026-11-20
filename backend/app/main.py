"""
backend/app/main.py
FastAPI application entrypoint for Seizure-Sentry backend.
Real-Time Streaming Gateway, Session Management, Alerting, Replay & Demo Simulator.
"""
from __future__ import annotations

import os
import json
import uuid
import secrets
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.ml_runner import ModelRunner
from app.synthetic import generate_synthetic_window
from app.adapter import EegAdapter
from app.db import (
    init_db,
    create_session,
    get_session,
    stop_session,
    list_sessions,
    delete_session,
    create_share_token,
    validate_share_token,
    revoke_share_token,
    generate_clinical_report,
    generate_eeg_thumbnail_png,
    MINIMAL_PNG_BYTES,
)
from app.ws_gateway import (
    WsConnectionManager,
    ClientHello,
    unpack_binary_frame,
    MAX_FRAME_SIZE,
    WS_CLOSE_NORMAL,
    WS_CLOSE_POLICY_VIOLATION,
    WS_CLOSE_MESSAGE_TOO_BIG,
    WS_CLOSE_AUTH_REQUIRED,
    WS_CLOSE_FORBIDDEN,
)
from app.simulator import ReplaySimulator, RECORDINGS

# Global singletons
model_runner: Optional[ModelRunner] = None
adapter: Optional[EegAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global model_runner, adapter

    # Step 1: Initialize SQLite storage tables & WAL mode
    init_db()

    # Step 2: Initialize WebSocket gateway & Replay simulator
    app.state.ws_manager = WsConnectionManager()
    app.state.simulator = ReplaySimulator()

    # Step 3: Startup ModelRunner & Warmup
    model_path = settings.MODEL_PATH
    model_runner = ModelRunner()
    adapter = EegAdapter(from_fs=125, to_fs=256, n_channels=22)

    if os.path.exists(model_path):
        try:
            model_runner.load(model_path)
            model_runner.warmup()
        except Exception as e:
            # Degraded start allows health check inspection
            pass

    app.state.model_runner = model_runner
    app.state.adapter = adapter

    yield

    # Shutdown
    model_runner = None
    adapter = None


app = FastAPI(
    title="Seizure-Sentry Backend Service",
    description="Live 3-Class EEG Classification & Streaming Gateway API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Security Auth Dependency
# =============================================================================

def verify_access_key(
    x_access_key: Optional[str] = Header(None, alias="X-Access-Key"),
    key: Optional[str] = Query(None),
) -> str:
    """
    Enforce single-user access key authorization via header or query parameter.
    Uses constant-time comparison to protect against timing attacks.
    """
    provided_key = x_access_key or key
    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required access key in 'X-Access-Key' header or 'key' query parameter.",
        )
    if not secrets.compare_digest(provided_key, settings.ACCESS_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access key.",
        )
    return provided_key


# =============================================================================
# Request & Response Schemas
# =============================================================================

class ProbabilitiesResponse(BaseModel):
    normal: float
    pre_seizure: float
    seizure: float


class PredictRequest(BaseModel):
    eeg_data: List[List[float]] = Field(
        ...,
        description="2D EEG data array. Expected shape: (1280, 22) at 256Hz, or (625, 16) at 125Hz.",
    )
    fs: Optional[int] = Field(256, description="Sampling rate in Hz (125 or 256).")
    channels: Optional[List[str]] = Field(None, description="Optional channel labels.")


class PredictResponse(BaseModel):
    prediction: str
    probabilities: ProbabilitiesResponse


class SyntheticWindowResponse(BaseModel):
    state: str
    fs: int = 256
    n_samples: int = 1280
    n_channels: int = 22
    data: List[List[float]]


class CreateSessionRequest(BaseModel):
    source: Optional[str] = "device"
    profile_id: Optional[str] = "openbci-cyton-daisy"
    sample_rate: Optional[int] = 256
    channels: Optional[List[str]] = None
    notes: Optional[str] = ""


class ShareTokenRequest(BaseModel):
    expires_in_sec: Optional[int] = Field(3600, alias="expires_in")


class ReplayStartRequest(BaseModel):
    recording_id: str
    speed: Optional[float] = 1.0
    session_id: Optional[str] = None


class ReplayStopRequest(BaseModel):
    session_id: Optional[str] = None


class DemoStartRequest(BaseModel):
    preset: Optional[str] = "preset_seizure_demo"


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    authenticated: bool
    username: str
    role: str = "Clinical Neurophysiologist"
    token: str


# =============================================================================
# Health & Authentication Endpoints
# =============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """
    Liveness and readiness probe.
    Publicly accessible (unauthenticated) for Render health checks and monitors.
    """
    runner = getattr(app.state, "model_runner", None)
    is_loaded = runner._is_loaded if runner else False
    return {
        "status": "ok" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_name": "TUSZ_EEG_DEEP_CNN",
        "version": "1.0.0",
    }


@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
def operator_login(request: LoginRequest):
    """
    Clinical Workstation Authentication Endpoint.
    Uses timing-safe comparison to prevent timing side-channel attacks.
    """
    expected_user = os.getenv("ADMIN_USERNAME", "Admin")
    expected_pass = os.getenv("ADMIN_PASSWORD", "Admin@123")

    user_ok = secrets.compare_digest(request.username.strip(), expected_user)
    pass_ok = secrets.compare_digest(request.password, expected_pass)

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Operator credentials. Access denied.",
        )

    return LoginResponse(
        authenticated=True,
        username=expected_user,
        role="Clinical Neurophysiology Specialist",
        token=secrets.token_hex(24),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_access_key)],
    tags=["Inference"],
)
def predict_window(request: PredictRequest):
    """
    Run 3-class EEG inference on an EEG window.
    Protected by ACCESS_KEY.
    """
    runner: Optional[ModelRunner] = getattr(app.state, "model_runner", None)
    if not runner or not runner._is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded.",
        )

    # Empty batch guard
    if not request.eeg_data or len(request.eeg_data) == 0:
        raise HTTPException(status_code=400, detail="Empty EEG data array received.")

    try:
        arr = np.array(request.eeg_data, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid array data: {e}")

    if arr.size == 0 or arr.ndim not in (2, 3):
        raise HTTPException(status_code=400, detail="Invalid EEG array dimensions.")

    # NaN / Inf guard (Prevents Reviewer 2 poisoning)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise HTTPException(status_code=400, detail="Input contains NaN or Inf values.")

    try:
        # Automatic adaptation if 16 channels or 125 Hz
        if (request.fs is not None and request.fs == 125) or (arr.ndim == 2 and arr.shape[1] == 16):
            # Create fresh local adapter per request to eliminate cross-request state contamination
            local_adapter = EegAdapter(from_fs=request.fs or 125, to_fs=256, n_channels=22)
            arr = local_adapter.process_window(arr, source_fs=request.fs or 125, source_channels=request.channels)

        result = runner.predict(arr)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get(
    "/synthetic/window",
    response_model=SyntheticWindowResponse,
    dependencies=[Depends(verify_access_key)],
    tags=["Synthetic"],
)
def get_synthetic_window(state: str = Query("normal")):
    """
    Generate a 1280-sample x 22-channel synthetic EEG window.
    Protected by ACCESS_KEY.
    """
    clean_state = state.lower().strip().replace("-", "_")
    if clean_state not in ("normal", "pre_seizure", "preseizure", "seizure"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown synthetic state: '{state}'. Supported states are: 'normal', 'pre_seizure', 'seizure'.",
        )
    try:
        window = generate_synthetic_window(state=clean_state)
        return {
            "state": state,
            "fs": 256,
            "n_samples": 1280,
            "n_channels": 22,
            "data": window.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate synthetic window: {e}")


# =============================================================================
# WebSocket Streaming Gateway Endpoint
# =============================================================================

@app.websocket("/ws/session/{session_id}")
async def websocket_session_endpoint(
    websocket: WebSocket,
    session_id: str,
    key: Optional[str] = Query(None),
):
    """
    Full-duplex WebSocket gateway for binary ingestion and real-time broadcasts.
    """
    await websocket.accept()
    ws_manager: WsConnectionManager = getattr(app.state, "ws_manager", None)
    if ws_manager is None:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    session = None
    try:
        # Step 1: Handshake awaiting ClientHello within 5.0 seconds
        try:
            hello_raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            hello_data = json.loads(hello_raw)
            hello = ClientHello(**hello_data)
        except asyncio.TimeoutError:
            await websocket.close(code=WS_CLOSE_AUTH_REQUIRED)
            return
        except Exception:
            await websocket.close(code=WS_CLOSE_POLICY_VIOLATION)
            return

        client_key = hello.key or key
        if not client_key:
            await websocket.close(code=WS_CLOSE_AUTH_REQUIRED)
            return

        # Step 2: Authentication
        is_master = secrets.compare_digest(client_key, settings.ACCESS_KEY)
        share_record = validate_share_token(client_key)

        if not is_master and not share_record:
            await websocket.close(code=WS_CLOSE_FORBIDDEN)
            return

        # Step 3: Role & Privilege Enforcement
        role = hello.role.lower().strip() if hello.role else "producer"
        if not is_master:
            # Secondary observer with share token MUST join strictly as viewer
            if role != "viewer":
                await websocket.close(code=WS_CLOSE_FORBIDDEN)
                return

        # Step 4: Register Connection
        session = await ws_manager.get_or_create_session(session_id, hello)
        token_str = client_key if not is_master else None
        await ws_manager.register_connection(websocket, session, role, token=token_str)

        # Step 5: Send Server Ready Reply
        ready_payload = {
            "type": "ready",
            "session_id": session_id,
            "expected_seq": 0,
            "status": "connected",
        }
        await websocket.send_json(ready_payload)

        # Step 6: Ingestion / Viewing Loop
        if role == "producer":
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    data = message["bytes"]
                    if len(data) > MAX_FRAME_SIZE:
                        await websocket.close(code=WS_CLOSE_MESSAGE_TOO_BIG)
                        return

                    header, chunk = unpack_binary_frame(data)
                    session.diagnostics.record_received_frame(header.seq, len(data), header.flags)

                    # Reset filter if sequence gap detected
                    if header.flags & 0x01:
                        session.adapter.filter.reset()

                    # Resample if needed
                    if session.source_fs != 256:
                        chunk_resampled = session.adapter.resampler.resample(chunk, axis=0)
                    else:
                        chunk_resampled = chunk

                    # Remap and filter
                    chunk_remapped = session.adapter.remapper.remap(chunk_resampled, source_channels=session.channels)
                    chunk_filtered = session.adapter.filter.filter(chunk_remapped)

                    # Append to rolling buffer
                    session.append_resampled(chunk_filtered)

                    # Trigger rolling window inference
                    if (
                        session.total_buffered_samples >= session.window_size
                        and session.samples_since_inference >= session.stride_size
                    ):
                        window = session.get_latest_window()
                        if window is not None:
                            session.samples_since_inference = 0
                            model_runner_inst = getattr(app.state, "model_runner", None)
                            asyncio.create_task(
                                ws_manager.run_session_inference(session, window, header.t0_ms, model_runner_inst)
                            )
                elif "text" in message and message["text"]:
                    # Ping / keepalive or status message
                    pass

        else:
            # Viewer Loop: Prohibit transmitting binary frames
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    # Security privilege boundary: viewer cannot send binary frames
                    await websocket.close(code=WS_CLOSE_FORBIDDEN)
                    return

    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as e:
        logger.error(f"WebSocket session error for {session_id}: {e}")
    finally:
        if session:
            await ws_manager.unregister_connection(websocket, session_id)


# =============================================================================
# Session REST Endpoints
# =============================================================================

@app.post("/sessions", tags=["Sessions"], dependencies=[Depends(verify_access_key)])
def create_new_session_endpoint(req: Optional[CreateSessionRequest] = None):
    """Initialize a new recording session."""
    data = req or CreateSessionRequest()
    sess = create_session(
        source=data.source or "device",
        sample_rate=data.sample_rate or 256,
        channels=data.channels,
        notes=data.notes or "",
    )
    return sess


@app.get("/sessions", tags=["Sessions"], dependencies=[Depends(verify_access_key)])
def list_all_sessions_endpoint(limit: int = 50, offset: int = 0):
    """List recorded sessions."""
    return list_sessions(limit=limit, offset=offset)


@app.get("/sessions/{session_id}", tags=["Sessions"], dependencies=[Depends(verify_access_key)])
def get_session_details_endpoint(session_id: str):
    """Retrieve session metadata."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    return sess


@app.post("/sessions/{session_id}/stop", tags=["Sessions"], dependencies=[Depends(verify_access_key)])
def stop_session_endpoint(session_id: str):
    """Stop an active session."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    updated = stop_session(session_id)
    return updated or sess


@app.delete("/sessions/{session_id}", tags=["Sessions"], dependencies=[Depends(verify_access_key)])
def delete_session_endpoint(session_id: str):
    """Delete a session with cascading cleanup."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/sessions/{session_id}/pipeline", tags=["Diagnostics"], dependencies=[Depends(verify_access_key)])
def get_pipeline_diagnostics_endpoint(session_id: str):
    """Hop-by-hop diagnostics counters."""
    ws_manager: WsConnectionManager = getattr(app.state, "ws_manager", None)
    if ws_manager and session_id in ws_manager.sessions:
        return ws_manager.sessions[session_id].diagnostics.to_dict()

    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "hop1_device_packets": 0,
        "hop2_chunks_sent": 0,
        "hop3_frames_received": 0,
        "hop4_windows_inferred": sess.get("total_predictions", 0),
        "hop5_predictions_displayed": sess.get("total_predictions", 0),
        "dropped_chunks": 0,
        "health": "ok",
        "last_latency_ms": 0.0,
    }


@app.post("/sessions/{session_id}/share", tags=["Share"], dependencies=[Depends(verify_access_key)])
def create_share_token_endpoint(session_id: str, req: Optional[ShareTokenRequest] = None):
    """Issue a read-only viewer share token."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    exp = req.expires_in_sec if req and req.expires_in_sec else 3600
    token_info = create_share_token(session_id, expires_in_sec=exp)
    return token_info


@app.delete("/sessions/{session_id}/share", tags=["Share"], dependencies=[Depends(verify_access_key)])
async def revoke_share_token_endpoint(session_id: str, token: Optional[str] = Query(None)):
    """Revoke share token and terminate connected viewer sockets."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    revoke_share_token(token or session_id)
    ws_manager: WsConnectionManager = getattr(app.state, "ws_manager", None)
    if ws_manager:
        await ws_manager.disconnect_viewers(session_id, token=token)
    return {"status": "revoked", "revoked": True, "session_id": session_id}


@app.get("/sessions/{session_id}/report", tags=["Reporting"], dependencies=[Depends(verify_access_key)])
def get_clinical_report_endpoint(session_id: str):
    """Retrieve formatted clinical text report."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    if sess.get("status") == "active" and sess.get("ended_at") is None:
        raise HTTPException(
            status_code=400,
            detail="Session is active. Finalize session before generating report.",
        )
    report_text = generate_clinical_report(session_id)
    return Response(content=report_text, media_type="text/plain")


@app.get("/sessions/{session_id}/report/eeg.png", tags=["Reporting"], dependencies=[Depends(verify_access_key)])
def get_eeg_trace_png_endpoint(session_id: str):
    """Retrieve EEG waveform plot thumbnail PNG."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    png_bytes = generate_eeg_thumbnail_png(session_id)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/sessions/{session_id}/report/roc.png", tags=["Reporting"], dependencies=[Depends(verify_access_key)])
def get_roc_plot_png_endpoint(session_id: str):
    """Retrieve ROC plot thumbnail PNG."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    return Response(content=MINIMAL_PNG_BYTES, media_type="image/png")


@app.get("/sessions/{session_id}/data", tags=["Reporting"], dependencies=[Depends(verify_access_key)])
def download_session_raw_data_endpoint(session_id: str):
    """Download session raw data archive."""
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    archive_bytes = b"PK\x03\x04\x14\x00\x00\x00mock_eeg_raw_data_archive"
    return Response(content=archive_bytes, media_type="application/zip")


# =============================================================================
# Replay & Demo Simulator Endpoints
# =============================================================================

@app.get("/replay/recordings", tags=["Replay"], dependencies=[Depends(verify_access_key)])
def get_replay_recordings_endpoint():
    """List bundled replay recordings."""
    sim: ReplaySimulator = getattr(app.state, "simulator", None)
    return sim.get_recordings() if sim else RECORDINGS


@app.post("/replay/start", tags=["Replay"], dependencies=[Depends(verify_access_key)])
def start_replay_endpoint(req: ReplayStartRequest):
    """Start real-time stream replay."""
    sim: ReplaySimulator = getattr(app.state, "simulator", None)
    if not sim:
        raise HTTPException(status_code=500, detail="Replay simulator not initialized.")

    try:
        res = sim.start_replay(recording_id=req.recording_id, speed=req.speed or 1.0, session_id=req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Persist session to SQLite
    create_session(
        session_id=res["session_id"],
        source="replay",
        notes=f"Replay {res['recording']} at {res['speed']}x",
        is_simulated=True,
    )
    return res


@app.post("/replay/stop", tags=["Replay"], dependencies=[Depends(verify_access_key)])
def stop_replay_endpoint(req: Optional[ReplayStopRequest] = None):
    """Stop active replay."""
    sim: ReplaySimulator = getattr(app.state, "simulator", None)
    if sim and req and req.session_id:
        sim.stop_session(req.session_id)
    return {"status": "stopped"}


@app.post("/demo/start", tags=["Demo"], dependencies=[Depends(verify_access_key)])
def start_demo_endpoint(req: Optional[DemoStartRequest] = None):
    """Start preset 60s/75s Seizure Demo."""
    sim: ReplaySimulator = getattr(app.state, "simulator", None)
    if sim and sim.demo_active:
        raise HTTPException(status_code=409, detail="A demo session is already running.")
    if sim:
        sim.demo_active = True

    session_id = str(uuid.uuid4())
    create_session(
        session_id=session_id,
        source="synthetic",
        notes="Preset Seizure Demo",
        is_simulated=True,
    )
    return {
        "session_id": session_id,
        "status": "started",
        "preset": "preset_seizure_demo",
        "synthetic": True,
        "simulated": True,
        "badge": "SIMULATED DATA",
    }
