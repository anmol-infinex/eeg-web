# Seizure-Sentry: UI/UX Design (v0.3)

**Direction:** a calm, professional clinical-software interface, in the family of patient-monitoring and lab software (think Grafana, Linear, and hospital EEG viewers), not a sci-fi dashboard. **Light and dark themes**, both first-class. It extends the existing NeuroTrace console; reuse its tokens where they exist.

## 0. Anti-"AI-generated" rules (hard rules for the agent)

The UI must not look like a template. Follow all of these:

1. **No neon, no glow, no gradients** (no gradient text, gradient buttons, glassmorphism, blurred blobs, or animated backgrounds).
2. **One accent colour** (blue). Green/amber/red are used **only** for clinical state and alerts, never as decoration.
3. **No emoji anywhere** in the UI or reports. One icon family only (Lucide, 16/20 px, 1.5 px stroke). Icons support labels; they never replace them.
4. **No generic "feature card grid with icon circles"** and no marketing slogans inside the app. Copy is plain and specific ("Dongle not found. Check that it is plugged in and the board is switched on.").
5. **Sentence case** everywhere ("Link device", not "LINK DEVICE"). No ALL-CAPS except tiny table headers and the SIMULATED badge.
6. **Real data density.** Tables, units, timestamps, tabular numerals. Whitespace is consistent (4 px base grid), not excessive.
7. **Flat surfaces:** 1 px borders, 6 px radius, no drop shadows except popovers/modals (one subtle shadow).
8. **Restrained motion:** 120-200 ms fades/slides only. No bouncing, no pulsing except the critical-alert edge (and not at all under `prefers-reduced-motion`).
9. **Meaningful empty states:** say what to do next. No lorem ipsum, no placeholder charts with fake data.
10. **Consistency check:** every screen uses the same page header, same table style, same button set, same form controls. If a component is new, add it to the component list below.

## 1. Typography

- UI: **IBM Plex Sans** (400/500/600). Numbers, timestamps, channel labels: **IBM Plex Mono** or Plex Sans with `font-variant-numeric: tabular-nums`.
- Scale: 12 (meta) / 13 (table, dense UI) / 14 (body) / 16 (section title) / 20 (page title) / 28 (large state value). Line-height 1.4-1.5.
- Page titles 20/600, sentence case. No decorative display fonts.

## 2. Colour tokens (CSS variables, two themes)

Implement as CSS variables on `:root[data-theme="light"]` and `:root[data-theme="dark"]`. Default follows `prefers-color-scheme`; the user can override in the header toggle (Light / Dark / System) and the choice is remembered.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#F5F6F8` | `#0F1216` | app background |
| `--surface` | `#FFFFFF` | `#171B21` | cards, panels |
| `--surface-2` | `#EEF0F3` | `#1E232B` | table header, hover, inset |
| `--border` | `#D9DDE3` | `#2A313B` | dividers, outlines |
| `--text` | `#14181F` | `#E8ECF1` | primary text |
| `--text-muted` | `#5B6573` | `#98A2B1` | secondary text |
| `--accent` | `#1F5FBF` | `#5B9CFF` | links, primary buttons, focus ring, selected nav |
| `--accent-contrast` | `#FFFFFF` | `#0B1220` | text on accent |
| `--ok` | `#1B7F4D` | `#3DBE82` | Normal state |
| `--warn` | `#B26A00` | `#E8A33D` | Pre-seizure warning |
| `--crit` | `#B42318` | `#F0625A` | Seizure / critical |
| `--sim` | `#5B4B8A` | `#A596D6` | SIMULATED data badge |
| `--eeg-bg` | `#FFFFFF` | `#0B0E12` | EEG canvas background |
| `--eeg-trace` | `#1B2A41` | `#9FD3C7` | EEG traces |
| `--eeg-grid` | `#E4E7EC` | `#1F252D` | EEG grid lines |

Notes:
- **Light-mode EEG canvas looks like a clinical paper recording** (white, fine grey grid, dark traces). Dark-mode canvas is near-black with soft teal-grey traces. Both must keep traces clearly legible.
- State chips use a tinted background (8-12% of the state colour) + 1 px border + text in the state colour. All states also have a text label and an icon.
- Verify contrast ≥ 4.5:1 for text and ≥ 3:1 for UI borders/icons in **both** themes.

## 3. App structure (professional sections)

Left navigation (collapsible, 220 px), top bar, content area, footer.

| Section | Purpose |
|---|---|
| **Overview** | Short product description, system status (backend, model, device), quick actions (Link device, Start replay), latest session summary, supported devices summary, data-handling note |
| **Live monitor** | The main screen (section 4) |
| **Devices** | Connect/manage EEG sources: supported device list, connection status, per-device profile, compatibility check against the model |
| **Recordings** | Table of sessions with search/sort; open, download, delete |
| **Reports** | Generate/download reports for a recording (text, EEG graph, ROC, confusion matrix) |
| **Alerts** | Alert history with filters (level, date), acknowledgement state |
| **Review** | Existing NeuroTrace EDF upload and offline analysis |
| **Model** | Model card: name/version, expected channels and sample rate, window/stride, class definitions, preprocessing, known limitations |
| **Settings** | Appearance (theme), alert thresholds, notifications, filters, backend URL and access key, diagnostics |
| **Help** | Setup guides (dongle driver, electrode placement tips), troubleshooting, about, disclaimer |

Top bar (left to right): section breadcrumb, **connection status chip** (Disconnected / Connecting / Streaming / Simulated), backend status, theme toggle, notification bell (alert count).
Footer (always): "Research prototype. Not a medical device." + version.

## 4. Live monitor layout

```
+----------------------------------------------------------------------------------+
| Live monitor                              Device: OpenBCI Cyton+Daisy  [Streaming]|
+------------+------------------------------------------------+--------------------+
| Channels   |  EEG  (10 s window)       [5s|10s|20s] [50 µV/div] [Filters v]      |
| Fp1  good  |  Fp1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                       |
| Fp2  good  |  Fp2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                       |
| F3   noisy |  ...                                                                   |
| ...        |  O2  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                       |
| (16 rows)  +------------------------------------------------+--------------------+
|            |  Model output                                  | Current state      |
|            |  Timeline (one cell per window) + alert marks  | Normal             |
|            |                                                | P(normal)   0.92   |
|            |                                                | P(pre)      0.06   |
|            |                                                | P(seizure)  0.02   |
+------------+------------------------------------------------+--------------------+
| 125 Hz  |  31 packets/s  |  dropped 0  |  latency 420 ms  |  00:12:41   [Stop]    |
+----------------------------------------------------------------------------------+
```

- The **EEG canvas is the largest element.** Everything else is secondary.
- **Channel list:** channel label, quality status (text + small marker). Clicking a channel highlights its trace.
- **Current state panel:** state name in 28/600 with a state chip, the three probabilities as a simple three-row table with thin horizontal bars (neutral bar colour; the leading class uses its state colour), model name/version in muted text. Not a gauge or radial chart.
- **Timeline:** compact strip, one cell per prediction window, coloured by state, with alert markers. Hover shows time and probabilities; click jumps (when paused or reviewing).
- **Stats bar:** monospace/tabular numbers, muted labels.
- Primary button area: **Link device** when disconnected, **Stop session** when streaming.

## 5. Key flows

**Link device (modal, 4 steps, plain language)**
1. Select source: list of supported devices/adapters (Cyton via USB dongle, Bluetooth headset, Local bridge, Simulator). Unsupported browser features are disabled with a short reason.
2. Connect: browser's native port/device picker, then a status log ("Dongle opened", "Board detected", "Daisy detected", "16 channels at 125 Hz").
3. Compatibility check: a small table comparing **device vs model** (channel count, channel names coverage, sample rate) with a clear result: *Compatible*, *Compatible with adaptation (resampling, channel mapping)*, or *Not compatible* with the reason. Never hide a mismatch.
4. Signal check: 16 tiles (channel name, mini trace, quality). **Start monitoring** is always available, warnings are shown, not blocked.

**Alerts**
- Warning (pre-seizure): amber banner under the top bar, short chime, push notification.
- Critical (seizure): red banner, repeating alarm until acknowledged or muted, 2 px red inset border on the app frame, push notification. Banner shows time, peak probability and **Acknowledge** / **Mute**.
- Poor signal: neutral grey banner "Signal quality too low. Predictions paused." (alerts suppressed per PRD).

**Replay / Simulated**
- A fixed violet "Simulated data" badge in the top bar, on the EEG canvas corner, and in every exported report and image.

## 6. States

| Situation | UI |
|---|---|
| No device | Empty canvas with grid only and a short instruction + **Link device** button |
| Model warming up | "Collecting first window (3 s)" in the state panel with a thin progress bar |
| Device lost | Amber banner "Device disconnected. Reconnect to continue." with **Reconnect**; data so far is kept |
| Backend unreachable | Banner "Cannot reach the analysis server" with **Retry**; EEG keeps drawing locally; predictions marked "paused" |
| Cold start (hosted backend waking) | "Starting analysis server, this can take up to a minute." with spinner and automatic retry |
| Unsupported browser | Link device disabled, explanation shown; Replay and Review still work |

## 7. Components (reuse across all sections)

PageHeader, NavItem, StatusChip, StateChip, Button (primary/secondary/ghost/destructive), IconButton, Select, Toggle, Tabs, Table (sortable, dense), Modal, Stepper, Banner, Toast, EmptyState, Tooltip, KeyValueList, EegCanvas, ChannelList, ProbabilityRows, PredictionTimeline, StatsBar, CompatibilityTable, ReportCard.

Buttons: one primary per view. Destructive actions confirm in a modal. Focus ring: 2 px `--accent`, always visible on keyboard focus.

## 8. Accessibility and safety copy

- State is never colour-only (label + icon). Keyboard: Space pause/resume view, M mute, A acknowledge, 1/2/3 time window. Respect `prefers-reduced-motion`.
- Alarm sound only after a user gesture, volume-limited, mutable.
- Wording: "model output", "detected pattern". Never "diagnosis"; never suggest medication changes.
- Persistent disclaimer in footer and in every report.

## 9. Reports

1. Header: session id, date/time, source (Device or Simulated), device name, model version.
2. Summary: final prediction, average class probabilities, per-class window counts, sustained seizure time ranges.
3. 16-channel EEG graph (PNG; light theme for printing regardless of app theme).
4. If labels exist: ROC curve and confusion matrix (PNG + text).
5. Disclaimer.

## 10. Responsive behaviour

Desktop first (1280 px+). At tablet widths the channel list collapses to a drawer. On phones, Live monitor shows state panel, alert banner and a compact 4-channel preview; full 16-channel view requires a larger screen. Link device is desktop only (Web Serial).
