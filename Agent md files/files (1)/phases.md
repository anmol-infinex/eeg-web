# Seizure-Sentry: Build Phases (Prototype v0.2)

## Working agreements for the agent

- Read PRD.md, architecture.md, design.md and integrations.md first. Build **one phase at a time**; stop at the end of each phase and show how to verify it.
- **Never invent model facts** (channels, sample rate, window size, normalisation, class order). Read them from the model/training code. If you cannot, ask the owner.
- Never commit secrets. Use `.env` and `.env.example`.
- Keep commits small. Add tests with each feature. Do not skip the acceptance criteria.
- If a phase uncovers a conflict with the docs, update the docs and tell the owner what changed.
- Single-user: **do not** add accounts, roles or a patients table.

---

## Phase 0: Foundation and model contract

**Tasks**
- Create the monorepo layout (architecture.md section 3). Inspect the existing NeuroTrace frontend and keep its stack.
- Inspect the trained model in `backend/models/` and the training notebook/script. Produce `model_card.json` with the real: framework, input shape, channel list and order, sample rate, window/stride, preprocessing (filters, normalisation), class order, and the Pre-Seizure definition.
- Write `MODEL_CONTRACT.md` listing **mismatches vs hardware** (channel count/names vs Cyton's 16 at 125 Hz, sample rate). Propose the adapter plan and **ask the owner** before padding or retraining.
- Try ONNX export; compare outputs with the original model on sample data.
- Backend skeleton: FastAPI, `/health`, config, access-key check, Dockerfile.
- Set up GitHub, CI, Render and Vercel as described in **integrations.md** (private repo, GitHub MCP, `render.yaml`, Vercel Git integration). Deploy the empty skeleton early so deployment problems appear now, not at the end.

**Deliverables:** repo scaffold, `model_card.json`, `MODEL_CONTRACT.md`, working `/health`.
**Acceptance:** model loads in the backend and returns probabilities for a test array; contract document reviewed by the owner.

## Phase 1: Inference pipeline + Replay (no hardware)

**Tasks**
- `adapter.py` (channel map + resample), `preprocess.py` (streaming filters with state), `windowing.py`, `ModelRunner` + ONNX runner, smoothing.
- WebSocket gateway with the binary frame protocol (architecture.md section 5), validation, limits, and the `producer`/`viewer` roles.
- Device profiles + `compat.py` compatibility check (architecture.md section 4A) with unit tests, including an incompatible example (e.g. a 4-channel profile).
- `simulator.py`: stream an EDF at real-time speed through the gateway.
- **Golden test:** offline vs backend probabilities on a sample file.

**Deliverables:** backend that accepts frames and emits predictions; replay endpoint.
**Acceptance:** golden test passes within tolerance; replay of a labelled seizure yields rising P(seizure).

## Phase 2: Frontend shell + live visualization (still simulated)

**Tasks**
- Apply design tokens and app shell/nav. Keep Review module working.
- `EegCanvas` (16 ch ring buffer, 30+ fps), time-window and scale controls, channel list.
- WebSocket client, `StateCard`, `ProbBars`, `Timeline`, stats bar.
- Replay screen with the SIMULATED ribbon and "Seizure demo" button.

**Deliverables:** full Live UI driven by Replay.
**Acceptance:** replay shows smooth waveforms and live predictions; no memory growth over 30 minutes; works on Chrome and Edge.

## Phase 3: Link Device (Cyton + Daisy over Web Serial)

**Tasks**
- Verify packet format and Daisy pairing against OpenBCI docs and captured raw bytes. Write `parser.ts` with golden byte fixtures.
- Implement the Cyton code **behind the `EegSource` interface** (`lib/sources/cytonSerial.ts`), so later devices plug in without changes elsewhere.
- `serial.ts` + `handshake.ts`, resync, drop counting, scale to µV, 250 ms chunking, optional worker.
- Ask the owner to confirm the **electrode-to-pin mapping** (which cap position is on which Cyton/Daisy input) and write it into the device profile.
- Link Device modal (stepper, log, signal check tiles), signal quality module, disconnect/reconnect handling.
- Device status pill and error messages.

**Deliverables:** real 16-channel EEG on screen.
**Hardware checklist (owner + agent):** all 16 channels visible, blink spikes on Fp1/Fp2, jaw-clench artifact, 50 Hz visible when electrodes are poor, unplug/replug recovery.
**Acceptance:** Link Device to first live trace under 2 minutes; parser unit tests pass; sample counter shows no unexplained drops over 10 minutes.

## Phase 3B: End-to-end data-path proof (main requirement)

This phase proves the core promise: **cap -> website -> ML model -> live dashboard**.

**Tasks**
- Pipeline diagnostics page with live counters per hop: device packets, chunks sent, frames received by backend, windows inferred, predictions received by the UI (FR-27).
- Device Link modal runs the **compatibility check** and shows its result (design.md section 5).
- Live inference on real device data (not only replay), with channel mapping and resampling from the device profile.
- Write `docs/DATA_PATH_TEST.md`: a manual test script with expected numbers (e.g. 125 Hz x 16 ch in = windows per minute out).

**Acceptance:** with the cap worn, the owner sees live traces, rising counters at every hop, and model probabilities updating on the dashboard within the expected latency; unplugging the dongle shows the exact failing hop; replay and device produce the same message formats end to end.

## Phase 4: Live inference + alerts

**Tasks**
- Stream live chunks to the backend; show live predictions on real data.
- Alert engine (PRD section 7) with hysteresis, cooldown, artifact guard.
- Alert UI: banner, pulse, alarm, acknowledge/mute; browser Notification; Web Push (VAPID, service worker, subscription endpoint).
- Alert history.

**Deliverables:** end-to-end alerting for both device and replay.
**Acceptance:** replaying a labelled seizure raises a CRITICAL alert within the expected delay; push arrives with the tab in the background; poor signal suppresses alerts; unit tests for the alert state machine.

## Phase 5: Sessions, reports, settings

**Tasks**
- Recorder (SQLite + raw chunks), Sessions screen, delete.
- Report generator: text report, EEG PNG, raw data; ROC and confusion matrix for labelled replays/Review.
- Settings screen (key, URL, thresholds, filters, push, sound) and connection test.

**Deliverables:** downloadable reports in the format listed in PRD FR-19.
**Acceptance:** 10-minute live session records, reopens and exports a correct report; deleting a session removes its files.

## Phase 6: Deploy, harden, demo

**Tasks**
- Production deploy on **Render** (backend) and **Vercel** (frontend) per integrations.md; env config; CORS allow-list; handle Render cold start in the UI. EC2 + Caddy only as a fallback.
- Security pass (architecture.md section 11): wrong-key rejection, malformed/oversized frames, rate limits, non-root container, only 80/443 open.
- Soak test (1 hour), latency measurement, error-handling polish.
- Write `README.md` (setup, hardware steps, driver note, troubleshooting) and a **demo script**: (1) Link Device, (2) live trace + blink test, (3) Seizure demo replay with alert and push, (4) report download. Include a fallback plan if the venue has no network (local backend + tunnel).

**Deliverables:** public HTTPS deployment + README + demo script.
**Acceptance:** a fresh laptop with the dongle can run the complete demo from the README; security checks pass; 1-hour soak shows stable memory.

---

## Phase 7: More devices (after the prototype works)

**Tasks**
- Local bridge app (`tools/bridge/`, BrainFlow) that pushes the standard stream to the backend; pairing by session code.
- Web Bluetooth adapter for BLE headsets (one decoder + profile per headset, chosen by the owner).
- Optional LSL input via the bridge.
- Share live view (FR-26) if not already done.
- Each new device: profile JSON, adapter, compatibility test, and a documented known limit.

**Acceptance:** a second, different EEG device streams through the same pipeline with **no changes** to ML, alerts, storage or UI code, and the compatibility check gives a correct verdict for it.

---

## Definition of done (prototype)

- **Main path proven:** EEG cap -> website -> ML model -> live dashboard, with the data-path diagnostics page showing every hop.
- Live EEG from the Cyton+Daisy in the browser through Web Serial, through the `EegSource` adapter interface.
- Compatibility check shown for every device before monitoring.
- Existing model runs on the stream; predictions, probabilities and timeline update live.
- Alerts work (banner, sound, browser/Web Push).
- Replay/Seizure demo works end-to-end and is clearly labelled SIMULATED.
- Sessions can be recorded, exported and deleted.
- Deployed on Render + Vercel over HTTPS/WSS with the access key enforced and CI passing on GitHub.
- Disclaimer visible in the UI and reports.

## Parking lot (explicitly later)

Accounts and roles, Android app, SMS, custom STM32 hardware, retraining on 16 channels, per-patient personalisation, multi-patient dashboards, regulatory work, AWS S3 persistence (see integrations.md section 7).
