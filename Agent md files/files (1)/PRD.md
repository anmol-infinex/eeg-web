# Seizure-Sentry: Product Requirements Document (Prototype v0.2)

> Agent read order: **PRD.md → architecture.md → design.md → integrations.md → phases.md**. Build one phase at a time. Do not invent requirements or model facts. Anything unknown goes into "Open Items" (section 9) and Phase 0 resolves it.

## 1. Overview

Seizure-Sentry is a web platform that streams live EEG from an **OpenBCI Cyton + Daisy (16-channel)** headset into the browser, runs the owner's **already-trained ML model** on that stream, and shows the output (Normal / Pre-Seizure / Seizure) in a real-time dark, sci-fi, clinical-style dashboard with alerts.

```
EEG cap -> Cyton+Daisy board -> RFduino radio -> USB dongle -> Browser (Web Serial)
        -> Backend (preprocess + ML inference) -> Live dashboard + alerts
```

It evolves the existing **NeuroTrace EEG Clinical Review Console** (https://neurotrace-dashboard-eight.vercel.app). NeuroTrace's EDF upload/review features are kept as the "Review" module.

This is a **single-user research prototype / demo**. It is not a medical device.

## 2. Goals

**Main goal (everything else supports this):** an EEG cap connects to the website, its **live data flows into the ML model, and the model's output is visible live on the dashboard**. The system is built **device-agnostic**: any EEG source that provides raw EEG and has an adapter can be plugged in (section 4A).

1. **Link Device** from the website in a few clicks and see live 16-channel EEG.
2. Run the existing trained model on the live stream and display predictions continuously.
3. Alert on sustained seizure / pre-seizure output (on-screen, sound, browser push).
4. **Demo/Replay mode** that streams a recorded EEG file through the identical pipeline (hardware-free demos and tests).
5. Record sessions and export a downloadable report.

## 3. Non-goals

- No accounts, login, roles or patients database. One shared access key protects the backend.
- No Android/iOS app. Website only.
- No SMS or email. Alerts are dashboard + sound + browser/Web Push only.
- No custom hardware or STM32 firmware. OpenBCI Cyton is the acquisition device.
- No model retraining in this project. We deploy the existing model.
- No clinical claims or regulatory work.

## 4. Hardware (locked)

- OpenBCI **Cyton + Daisy**: 16 EEG channels, **125 Hz per channel** in 16-channel mode.
- Professional electrode cap.
- Board radio is an **RFduino** low-power Bluetooth link to the **OpenBCI USB dongle**. The browser never talks Bluetooth to the board. The dongle appears as a serial (COM) port and the browser reads it with the **Web Serial API**.
- Host: Windows laptop with **Chrome or Edge** (Web Serial is Chromium-only).

## 4A. Device-agnostic ingestion ("any EEG cap")

"Any EEG cap" is made true by an **adapter layer**, not by magic. Every source is converted to one standard stream (float32 µV, known channel labels, known sample rate), and everything after that point is identical.

| Source | How it connects | Status in prototype |
|---|---|---|
| **OpenBCI Cyton + Daisy** (16 ch) | USB dongle, Web Serial | **Build first (Phases 3-4)** |
| **Simulator / EDF replay** | Server-side file replay | Phase 1 |
| **Local bridge** (Python, BrainFlow) | Runs on the PC, pushes the same standard stream to the backend over WSS; covers many boards BrainFlow supports (Cyton, Ganglion, and other supported headsets) | Phase 7 |
| **Web Bluetooth headsets** (BLE GATT) | Browser connects directly; needs one decoder per headset | Phase 7 (per device) |
| **LSL stream** (Lab Streaming Layer) | Via the local bridge | Phase 7 (optional) |

Hard limits, to be shown honestly in the product:
- The device must expose **raw EEG samples** (not only band-power or "focus/calm" scores).
- Closed or licensed protocols may not be supported.
- **The model decides what is usable.** A model trained on 16-18 channels in a standard montage cannot be run meaningfully on a 4-channel headset. The app runs a **compatibility check** (channels, positions, sample rate) and says *Compatible*, *Compatible with adaptation*, or *Not compatible* with the reason.

## 5. User stories

| ID | Story |
|----|-------|
| US-1 | I click **Link Device**, choose the dongle's port, and see "Connected, streaming 16 ch". |
| US-2 | I see 16 EEG channels scrolling live with per-channel signal-quality badges. |
| US-3 | I see the model's state, class probabilities, confidence and a timeline update live. |
| US-4 | On a sustained seizure/pre-seizure output I get a banner, alarm sound and push notification. |
| US-5 | I start **Demo/Replay** (e.g. a seizure segment) and it flows through the same pipeline, clearly labelled SIMULATED. |
| US-6 | I stop a session and download a report plus raw data. |
| US-7 | I can still use **Review** (upload an EDF, inspect offline). |
| US-8 | When I connect any supported EEG device, the app tells me whether it is compatible with the model before I start monitoring. |
| US-9 | I can open a **read-only live view** of a running session on another screen (e.g. a second laptop or a tablet) using a share link. |

## 6. Functional requirements

**Device linking**
- FR-1 Use `navigator.serial`; show a clear "use Chrome/Edge" message elsewhere.
- FR-2 Open the dongle at 115200 baud 8N1, reset, detect Daisy, set 16-channel mode, start streaming (details in architecture.md section 4).
- FR-3 Parse the OpenBCI 33-byte packets into µV samples for 16 channels.
- FR-4 Handle unplug/disconnect, resync after corrupt bytes, count dropped packets from the sample counter, one-click reconnect.
- FR-5 Clean stop: send stop command, release the port.

**Live visualization**
- FR-6 Render 16 stacked channels at ≥30 fps; selectable window (5/10/20 s) and amplitude scale.
- FR-7 Per-channel quality badge: good / noisy / flat / railed.
- FR-8 Show sample rate, packets/s, dropped packets, link status.

**Device adapters (main path)**
- FR-22 Define an `EegSource` interface: `connect()`, `disconnect()`, `onChunk(samples)`, `info()` (name, channel labels, sample rate, units). Every source implements it; nothing downstream knows which device is connected.
- FR-23 Standard stream: float32 µV, sample-major, labelled channels, fixed sample rate, 250 ms chunks (architecture.md section 5 frame format).
- FR-24 `device_profiles/*.json` describe each supported device (channel names/positions, sample rate, scaling). Adding a device = adapter + profile, no changes to ML, alerts, UI or storage.
- FR-25 **Compatibility check** before monitoring: compare the device profile with `model_card.json` and display the result (design.md section 5). Block only when the result is *Not compatible*; warn otherwise.
- FR-26 **Share live view:** create a read-only link (random token, expiry, revocable) that opens the live monitor in view-only mode on another device. No accounts.
- FR-27 **End-to-end data-path proof:** a diagnostics page that shows each hop with a live counter (device packets, frames sent, frames received by backend, windows inferred, predictions displayed) so a broken link is found in seconds.

**Inference**
- FR-9 Browser streams 250 ms sample chunks to the backend over WebSocket; backend buffers, preprocesses and runs the model per window.
- FR-10 Backend returns per window: time range, class probabilities, label, smoothed state.
- FR-11 Model sits behind a `ModelRunner` interface so it can be swapped without touching the rest of the app.
- FR-12 A `channel_map.json` + resampler adapts Cyton's 16 channels at 125 Hz to whatever the model expects (see R-1, R-2).

**Alerts** (policy in section 7)
- FR-13 Alert engine with hysteresis and cooldown; levels INFO / WARNING (pre-seizure) / CRITICAL (seizure).
- FR-14 On alert: screen-edge pulse + banner, mutable alarm sound, browser Notification, and Web Push when the tab is in background.
- FR-15 Alert history with timestamps and supporting probabilities.

**Demo / Replay**
- FR-16 Replay a bundled or uploaded EDF at real-time speed through the same WebSocket pipeline with a permanent **SIMULATED DATA** badge.
- FR-17 One-click "Seizure demo" that replays a known labelled seizure segment.

**Sessions and reports**
- FR-18 Record raw samples, predictions and alerts per session (SQLite + files on disk).
- FR-19 Export: text prediction report (final prediction, average class probabilities, per-class window counts, sustained seizure time ranges), 16-channel EEG graph PNG, raw data file. For labelled replays also ROC curve and confusion matrix (PNG + text).
- FR-20 Sessions list to reopen, review and delete past recordings.

**Settings**
- FR-21 Alert thresholds, sound, push enable, backend URL and key, display defaults, filters (50 Hz notch, band-pass).

## 7. Alert policy (defaults, configurable)

- **Seizure (CRITICAL):** at least 3 of the last 5 windows with P(seizure) ≥ 0.70.
- **Pre-seizure (WARNING):** mean P(pre_seizure) ≥ 0.60 over the last 10 windows.
- **Hysteresis:** an event ends after 5 consecutive windows below threshold.
- **Cooldown:** 120 s before a new event of the same type.
- **Artifact guard:** if more than 4 channels are bad quality, suppress alerts and show "Signal quality too low".
- Track **false alarms per hour** as a headline metric.

## 8. Non-functional requirements

- **Latency:** chunk in → prediction out under 1 s (excluding the model's window length).
- **Stability:** 1 hour continuous streaming with bounded memory (ring buffers).
- **Security:** HTTPS/WSS only, shared access key on every request and WebSocket, strict CORS, input validation and size limits, rate limiting, no secrets in the frontend bundle, no PHI in logs.
- **Privacy:** recordings live only on the backend host; "Delete session" removes them.
- **Compatibility:** Chrome/Edge for Link Device; other modern browsers get Replay and Review.
- **Accessibility:** state is never conveyed by colour alone.
- **Observability:** structured logs, `/health`, diagnostics panel.

## 9. Risks and open items (Phase 0 must resolve)

| ID | Item | Handling |
|----|------|----------|
| R-1 | **Channel mismatch:** model was trained on a TUSZ-style 18-channel montage; Cyton+Daisy gives 16. | Read the model's true input spec. Build the adapter with `channel_map.json`. If channels are missing, log and display a warning and document accuracy impact. **Ask the owner before any padding/retraining decision.** |
| R-2 | **Sample rate:** hardware is 125 Hz; model likely 250/256 Hz. | Resample in the adapter; verify with a golden test. |
| R-3 | **Domain shift:** hospital EEG vs OpenBCI headset signals differ; live accuracy may drop. | Be honest in UI and report; Replay stays the reliable demo path. |
| R-4 | A healthy wearer will show "Normal"; a real seizure cannot be produced. | Use Replay/Seizure demo, always labelled SIMULATED. |
| R-5 | Web Serial and Web Push need HTTPS; an HTTPS page cannot call insecure `ws://`. | Host the backend on **Render** (automatic HTTPS/WSS). See integrations.md. |
| R-9 | Render free tier spins down when idle and has non-persistent disk. | Wake `/health` before demos; show a "starting server" state; upgrade or use S3 if recordings must persist. |
| R-10 | "Any EEG cap" is limited by raw-data access and by the model's expected montage. | Adapter + profile per device; compatibility check; state limits in the UI and README. |
| R-6 | Daisy packet ordering and which packets carry accelerometer data. | Verify against OpenBCI docs and real captured packets in Phase 3. |
| R-7 | Model framework/format/size unknown. | Phase 0 inspects; prefer ONNX if outputs match within tolerance. |
| R-8 | "Pre-Seizure" class definition comes from the owner's own labelling. | Record the definition in `model_card.json`. |

**Inputs needed from the owner:** (1) model file(s) + training notebook/script in `backend/models/`; (2) sample EDFs including a labelled seizure in `backend/samples/`; (3) the NeuroTrace frontend repo.

## 10. Success metrics

- Link Device to first live trace in under 2 minutes.
- 16 channels rendered at ≥30 fps on a mid-range laptop.
- Replaying a labelled seizure produces a CRITICAL alert through the live pipeline.
- Backend predictions on a recording match the offline model output (golden test).
- A 10-minute live session records and exports a report without errors.

## 11. Disclaimer (UI footer and every report)

"Seizure-Sentry is a research prototype for educational and demonstration purposes. It is not a medical device and must not be used for diagnosis, treatment, or medication decisions."
