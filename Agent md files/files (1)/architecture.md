# Seizure-Sentry: Architecture (Prototype v0.2, single-user)

## 1. System at a glance

```
 EEG cap (16 ch)
     |
 OpenBCI Cyton + Daisy  --RFduino radio-->  USB dongle (FTDI COM port)
                                                  |  Web Serial, 115200 baud
                                                  v
 +------------------------ Browser (Chrome/Edge, HTTPS) ------------------------+
 | Link Device | Cyton parser | ring buffers | EEG canvas | quality | Alert UI   |
 +------------------------------------+------------------------------------------+
                                      | WSS: 250 ms binary chunks  ^ JSON predictions/alerts
                                      v                            |
 +------------------------- Backend (FastAPI, Docker) ---------------------------+
 | WS gateway -> ring buffer -> Adapter (channel map + resample) -> Preprocess    |
 |   -> Windowing -> ModelRunner (ONNX/torch/tf) -> Smoothing -> Alert engine     |
 |   -> Recorder (SQLite + files) -> Report generator -> Web Push (VAPID)         |
 +--------------------------------------------------------------------------------+
        ^ Replay/Simulator feeds the SAME gateway from EDF files
```

Key principle: **live device and Replay use the same ingest path**, so everything downstream is tested without hardware.

## 2. Tech stack (decided)

| Layer | Choice | Why |
|---|---|---|
| Frontend | Extend the existing NeuroTrace app. If its stack is React + TypeScript + Vite, keep it; otherwise match whatever it uses. Tailwind for styling. | Reuse deployed work |
| EEG rendering | Custom HTML Canvas renderer (ring buffer, requestAnimationFrame) | 16 ch scrolling at 30+ fps without chart-lib overhead |
| Device link | Web Serial API in a dedicated Web Worker where possible | Keeps UI smooth |
| Backend | Python 3.11, **FastAPI** + Uvicorn, native WebSockets | Same language as ML |
| Inference | **ONNX Runtime (CPU)** if model exports cleanly; else the original framework behind `ModelRunner` | Light, fast, swappable |
| Signal processing | NumPy + SciPy (SOS filters with persistent state, `resample_poly`) | Streaming-safe |
| Storage | **SQLite** (metadata, predictions, alerts) + raw data files on disk (`.npy` or compressed binary) | Single-user, zero ops |
| Push | Web Push (VAPID) via `pywebpush` + service worker | No vendor, no SMS |
| Reports | matplotlib (PNG), plain-text report | Matches required output |
| Hosting | Backend in Docker on **Render** (automatic HTTPS/WSS, deploy from GitHub); frontend on **Vercel**; code on **GitHub**. EC2 + Caddy is the alternative. | WSS and Web Serial/Push need a secure context. See integrations.md |
| Demo fallback | Cloudflare Tunnel or similar to expose a laptop-hosted backend over HTTPS | Quick demos |

If the instance RAM is tight (1 GB), use CPU-only ONNX Runtime and add swap. Never ship GPU/CUDA builds.

## 3. Repository layout

```
seizure-sentry/
  docs/                      PRD.md architecture.md design.md phases.md
  frontend/                  (NeuroTrace app extended)
    src/
      app/ or routes/        live, replay, review, sessions, settings
      lib/cyton/             serial.ts handshake.ts parser.ts scale.ts
      lib/stream/            ringBuffer.ts wsClient.ts quality.ts
      lib/alerts/            sound.ts notify.ts push.ts
      components/            EegCanvas StatePanel ProbBars Timeline AlertBanner DeviceModal
      workers/               cytonWorker.ts
    public/sw.js             service worker for Web Push
  backend/
    app/
      main.py  config.py  security.py
      ws/gateway.py          ingest + viewer socket
      pipeline/              adapter.py preprocess.py windowing.py smoothing.py
      ml/                    runner.py (ModelRunner) onnx_runner.py model_card.json
      alerts/engine.py
      push/webpush.py
      recording/             store.py report.py
      replay/simulator.py    reads EDF, emits frames in real time
      api/                   sessions.py reports.py settings.py health.py review.py
    models/                  (owner's trained model + notebook)
    samples/                 (sample EDFs incl. a labelled seizure)
    tests/
  deploy/                    Dockerfile docker-compose.yml Caddyfile
```

## 4. Link Device: Cyton + Daisy over Web Serial

### 4.1 Connection model
The Cyton's RFduino radio pairs with the **OpenBCI USB dongle**, which enumerates as an FTDI virtual COM port. The browser opens that port with Web Serial. The browser does **not** use Bluetooth GATT for this board. The user may need the FTDI VCP driver on Windows.

### 4.2 Handshake (`lib/cyton/handshake.ts`)
1. `navigator.serial.requestPort()` then open at **115200**, 8N1.
2. Send `v` (soft reset). Read until the `$$$` terminator and parse the banner (board info, Daisy detected).
3. Ensure 16-channel mode (command `C`; `c` returns to 8). Confirm by observing alternating board/Daisy packets.
4. Defaults for channel settings are fine (gain 24).
5. Send `b` to start streaming, `s` to stop.
6. Open the WSS connection and begin forwarding chunks.

> **Verify before finalising `parser.ts`:** confirm the commands, packet layout and Daisy odd/even pairing against the official OpenBCI "Cyton Data Format" and "Cyton SDK" docs, and cross-check with BrainFlow's Cyton-Daisy parser. Log raw packets during bring-up.

### 4.3 Packet parsing
- 33-byte packets: start byte `0xA0`, sample counter, 8 x 24-bit signed big-endian channel values, 3 x 16-bit accelerometer values, stop byte `0xC0` to `0xC6`.
- With Daisy, consecutive packets alternate between channels 1-8 (board) and 9-16 (Daisy). Pair them into one 16-channel sample at **125 Hz**.
- Scale to µV: `count x 4.5 / gain / (2^23 - 1) x 1e6`. At gain 24 this is about **0.02235 µV/count**.
- Resync after corruption by hunting for `0xA0 ... 0xCx` at 33-byte spacing. Detect drops from counter gaps and flag them.

### 4.4 Chunking
Accumulate **250 ms** (about 31 samples x 16 ch) and send one binary frame. Never send per-sample.

### 4.5 Client-side signal quality
Per channel over the last 2 s: **flat** (std below about 0.5 µV), **railed** (near full scale), **noisy** (excess 50 Hz or broadband RMS), else **good**. The backend recomputes it independently.

### 4.6 Fallback
If Web Serial parsing proves unreliable, build a small Python bridge (BrainFlow, Cyton-Daisy board) that sends the same binary frames to the gateway. Build only if needed.

## 4A. Device adapter layer (any EEG cap)

The goal: **adding a new EEG device never touches the ML, alerts, storage or UI code.**

```
 Cyton (Web Serial) --+
 BLE headset (WebBT) -+--> EegSource adapter --> standard stream --> backend --> model --> dashboard
 Local bridge (WSS) --+     (frontend)           (float32 µV,         (adapter.py maps channels
 EDF replay (server) -+                           labelled channels)    + resamples to the model)
```

**Frontend interface (`lib/sources/EegSource.ts`):**
```ts
interface EegSource {
  id: string;                    // "cyton-serial", "ble-generic", "bridge", "replay"
  isSupported(): boolean;        // e.g. navigator.serial exists
  connect(): Promise<DeviceInfo>;// native picker / handshake
  disconnect(): Promise<void>;
  onChunk(cb: (c: SampleChunk) => void): void; // ~250 ms, µV, sample-major
  onStatus(cb: (s: SourceStatus) => void): void;
}
interface DeviceInfo { name: string; channels: string[]; fs: number; unit: "uV"; profileId: string }
```
Folder layout: `lib/sources/{EegSource.ts, cytonSerial.ts, webBluetooth.ts, bridge.ts, index.ts}`; the Cyton code from section 4 lives behind `cytonSerial.ts`.

**Device profiles (`backend/app/devices/profiles/*.json`):**
```json
{
  "profile_id": "openbci-cyton-daisy",
  "name": "OpenBCI Cyton + Daisy",
  "fs_hz": 125,
  "unit": "uV",
  "channels": ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8"],
  "channels_note": "Actual labels depend on how the cap is wired. The owner confirms the electrode-to-pin mapping in Phase 3.",
  "scale": {"gain": 24, "uv_per_count": 0.02235}
}
```
The channel list above is an **illustrative default**, not a fact: the real mapping is set by which cap positions are plugged into which Cyton/Daisy pins. Phase 3 requires the owner to confirm it.

**Compatibility check (`backend/app/devices/compat.py`):** inputs `device_profile` and `model_card`; output:
- `status`: `compatible` | `adapted` | `incompatible`
- `matched_channels`, `missing_channels` (relative to the model), `extra_channels`
- `sample_rate`: needs `none` | `resample (125 -> N Hz)`
- `notes`: plain-language reasons shown in the UI
Rules: all model channels present and same rate = compatible. Resampling and/or channel reorder/selection = adapted. Missing model channels = incompatible by default; the owner may explicitly allow a documented fallback (e.g. substituting a nearby channel) in `channel_map.json`, and the UI then shows "adapted (approximation)".

**Local bridge (`tools/bridge/`, Phase 7):** a small Python app using BrainFlow to read any BrainFlow-supported board, convert to the standard frame, and open `wss://<backend>/ws/session/{id}` with `source:"bridge"`. The website (viewer) shows the same live session. This is the route for headsets that browsers cannot talk to directly.

**Roles on the session WebSocket:** `producer` (device, bridge or replay sends frames), `viewer` (receives status, EEG, predictions, alerts). A viewer using a **share token** is read-only.

## 5. WebSocket protocol

Single endpoint: `wss://<host>/ws/session/{session_id}`. Roles are fixed by one message at connect.

**Client to server (first text message):**
`{"type":"hello","key":"<access key or share token>","role":"producer|viewer","source":"device|bridge|replay","profile_id":"openbci-cyton-daisy","fs":125,"channels":["...labels..."]}`
Share tokens may only join as `viewer`.
Server replies `{"type":"ready","expected_seq":0}` or an error and closes.

**Client to server (binary), little-endian:**

| Offset | Type | Field |
|---|---|---|
| 0 | u32 | `seq` |
| 4 | f64 | `t0_unix_ms` of first sample |
| 12 | u16 | `n_samples` |
| 14 | u8 | `n_channels` (16) |
| 15 | u8 | `flags` (bit0: gap before this frame) |
| 16 | f32[n_samples x n_channels] | µV, sample-major |

**Server to client (JSON):**
- `prediction`: `{"window_start","window_end","probs":{"normal","pre_seizure","seizure"},"label","state","model_version"}`
- `status`: link state, per-channel quality, risk level, dropped frames
- `alert`: `{"id","level","started_at","ended_at?","peak_prob"}`
- `ack`: `{"seq"}` every N frames; the client resends unacked frames after a reconnect.

In replay mode the **server** reads the EDF and emits frames itself (and also streams them to the browser for display), so the browser only needs the display socket. Same message formats.

Limits: max frame size, max frames/s, max connection time without ack. Reject anything malformed.

## 6. Backend pipeline

1. **Gateway** validates sequence numbers, marks gaps, appends to a ring buffer (about 60 s) and the recorder.
2. **Adapter** (`adapter.py`): reorders/selects channels via `channel_map.json`, resamples 125 Hz to the model's rate, and logs a warning when expected channels are missing.
3. **Preprocess**: causal band-pass + 50 Hz notch with persistent filter state, per `model_card.json`. **It must be identical to training preprocessing.**
4. **Windowing**: every `stride_sec`, take the last `window_sec`, normalise as specified, and run inference in a worker thread.
5. **Smoothing + Alert engine** (PRD section 7): state machine with hysteresis, cooldown and the artifact guard.
6. **Fan-out**: predictions, status and alerts to the browser; on a new alert, Web Push.
7. **Backpressure:** if inference lags, drop the oldest queued windows (never block ingest) and count them.

## 7. ML service contract

`ModelRunner` interface: `load(path)`, `predict(window: ndarray[C,T]) -> probs[3]`, `info() -> model_card`.

`model_card.json` (values below are **placeholders**; Phase 0 fills in the real ones from the trained model):
```json
{
  "model_version": "TBD",
  "framework": "TBD",
  "sample_rate_hz": "TBD",
  "channels": ["TBD"],
  "window_sec": "TBD",
  "stride_sec": "TBD",
  "preprocess": {"bandpass_hz": "TBD", "notch_hz": 50, "normalize": "TBD"},
  "classes": ["normal", "pre_seizure", "seizure"],
  "input_shape": "TBD",
  "pre_seizure_definition": "TBD (owner's labelling rule)"
}
```
**Golden test:** run one sample EDF through the offline training/inference code and through the backend pipeline; probabilities must match within a tolerance. This is the proof that deployment did not change the model.

## 8. REST API (v1, all require the access key header)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness and model loaded |
| POST | `/sessions` | create session `{source}` |
| POST | `/sessions/{id}/stop` | end and finalise recording |
| GET | `/sessions` / `/sessions/{id}` | list / details with predictions and alerts |
| DELETE | `/sessions/{id}` | delete data |
| POST | `/replay/start` | `{recording, speed}` starts server-side replay |
| GET | `/replay/recordings` | bundled samples |
| GET | `/sessions/{id}/report` | text report; `/report/eeg.png`, `/report/roc.png`, `/report/confusion.png`, `/data` |
| POST | `/push/subscribe` | store the single Web Push subscription |
| GET | `/devices/profiles` | list supported device profiles |
| POST | `/devices/compatibility` | `{profile_id}` returns compatibility result against the loaded model |
| POST | `/sessions/{id}/share` | create a read-only share token `{expires_in}`; `DELETE` revokes |
| GET | `/sessions/{id}/pipeline` | per-hop counters for the data-path diagnostics (FR-27) |
| GET/PUT | `/settings` | thresholds and options |
| POST | `/review/predict` | offline EDF upload prediction (existing Review module) |

## 9. Data model (SQLite)

- `sessions(id, source, started_at, ended_at, model_version, sample_rate, notes)`
- `predictions(id, session_id, t_start, t_end, p_normal, p_pre, p_seizure, label, state)`
- `alerts(id, session_id, level, started_at, ended_at, peak_prob)`
- `settings(key, value)`
- `push_subscription(endpoint, p256dh, auth)` (single row)
- Raw EEG stored per session in `data/sessions/<id>/chunks/*.bin` (float32, compressed).

## 10. Alerts and push

Engine emits `alert` events; the browser shows banner, pulse and alarm. If the tab is hidden, the backend sends Web Push to the stored subscription via the service worker. The alarm sound requires one prior user interaction (browser autoplay rule); the Link Device click satisfies this.

## 11. Security (single-user, still done properly)

- HTTPS/WSS only; Caddy handles certificates.
- One **access key** (long random string in backend env, entered once in Settings and kept in memory/sessionStorage). Required on every REST call and in the WS `hello`. Constant-time comparison, rate limit on failures.
- Strict CORS allow-list (the frontend origin only).
- Validate every message (types, sizes, channel count, monotonic seq); cap frame size and rate.
- No secrets in the frontend bundle. VAPID private key and access key only in backend env.
- No health data in logs. Recordings on encrypted disk volume if available. Delete endpoint really deletes.
- Pin and audit dependencies; run containers as non-root; only ports 80/443 open.

## 12. Deployment

**Primary: Render + Vercel + GitHub** (see integrations.md for setup, `render.yaml`, CI and secrets). Render terminates TLS, so WSS works without Caddy or a custom domain. The compose/Caddy setup below is the **alternative** (EC2).

- `docker-compose`: `backend` + `caddy`. Caddy terminates TLS and proxies `/ws` and `/api`.
- Frontend deploys to Vercel with `VITE_API_URL` (or the framework's equivalent) pointing at the backend domain.
- Config via env: `ACCESS_KEY`, `VAPID_PUBLIC/PRIVATE`, `ALLOWED_ORIGIN`, `MODEL_PATH`, `DATA_DIR`.
- Backups are not required for the prototype; data is disposable.

## 13. Testing strategy

- **Unit:** Cyton packet parser (golden byte fixtures, resync, drops), scaling, adapter, filters, alert state machine.
- **Golden ML test:** offline vs backend predictions (section 7).
- **Integration:** Replay a labelled seizure EDF end-to-end and assert a CRITICAL alert and a correct report.
- **Hardware bring-up checklist:** link, 16 channels visible, blink test (frontal channels spike), jaw clench artifact, 50 Hz check, unplug/replug recovery.
- **Soak:** 1 hour replay, memory flat.
- **Security:** wrong key rejected, oversized/malformed frames rejected, CORS from another origin blocked.
