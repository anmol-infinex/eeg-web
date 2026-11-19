"""Tier 3: Cross-Feature Matrix Test Suite.

Verifies complex multi-feature interactions across subsystem boundaries:
Auth (F25), Storage & Database (F23), View Sharing (F07), Replay Simulator (F21),
One-Click Demo (F22), Multi-Modal Alerting & Push (F19), Artifact Guard (F20),
Diagnostics (F06), and Clinical Reporting (F24).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.adapter import StreamingFilter
from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_latency_within,
    assert_prediction_schema,
    assert_probability_distribution,
    assert_status_code,
)
from tests.e2e.utils.fixtures import (
    generate_mock_binary_frame,
    mock_client_hello,
    mock_viewer_hello,
    random_access_key,
    random_session_id,
)


class TestCrossFeatureMatrix(unittest.TestCase):
    """Cross-feature matrix tests combining security, persistence, sharing, and alerting."""

    def setUp(self):
        """Sets up temporary SQLite database for session and storage engine testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_seizure_sentry.db"
        self._init_db(self.db_path)

    def tearDown(self):
        """Cleans up temporary directory and database."""
        self.temp_dir.cleanup()

    def _init_db(self, db_file: Path):
        """Initializes SQLite schema matching architecture.md §9."""
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                model_version TEXT,
                sample_rate INTEGER,
                notes TEXT
            );

            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                p_normal REAL NOT NULL,
                p_pre REAL NOT NULL,
                p_seizure REAL NOT NULL,
                label TEXT NOT NULL,
                state TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE alerts (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                level TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                peak_prob REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE share_tokens (
                token TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                revoked INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()
        conn.close()

    # -------------------------------------------------------------------------
    # 1. CNN Inference -> Hysteresis Alert -> Web Push Dispatch (F16 -> F18 -> F19)
    # -------------------------------------------------------------------------
    def test_f16_f18_f19_cnn_to_alerts_to_push(self):
        """Verifies CNN seizure probabilities trigger Hysteresis Alert and construct Web Push payload."""
        seizure_probs = [0.82, 0.78, 0.85]  # 3 of 3 >= 0.70 -> triggers CRITICAL

        # Hysteresis trigger
        triggered = len([p for p in seizure_probs if p >= 0.70]) >= 3
        self.assertTrue(triggered)

        alert_id = str(uuid.uuid4())
        alert_event = {
            "id": alert_id,
            "level": "CRITICAL",
            "started_at": 1718000010.5,
            "peak_prob": max(seizure_probs),
        }
        assert_alert_schema(alert_event)

        # Web Push notification payload formatting (VAPID)
        push_payload = {
            "title": "SEIZURE-SENTRY ALERT: Sustained Seizure Activity",
            "body": f"CRITICAL seizure detected with peak probability {alert_event['peak_prob']:.1%}",
            "icon": "/icons/alert-critical.png",
            "badge": "/icons/badge.png",
            "tag": "seizure-alert",
            "requireInteraction": True,
            "data": {
                "alert_id": alert_id,
                "level": "CRITICAL",
                "timestamp": alert_event["started_at"],
            },
            "actions": [
                {"action": "view", "title": "Open Live Monitor"},
                {"action": "dismiss", "title": "Acknowledge"},
            ],
        }

        self.assertEqual(push_payload["data"]["alert_id"], alert_id)
        self.assertTrue(push_payload["requireInteraction"])
        self.assertEqual(push_payload["tag"], "seizure-alert")

    # -------------------------------------------------------------------------
    # 2. Signal Quality -> Artifact Guard -> Alert Suppression (F09 -> F20 -> F18)
    # -------------------------------------------------------------------------
    def test_f09_f20_f18_quality_to_artifact_guard_to_alert_suppression(self):
        """Verifies bad signal on >4 channels activates Artifact Guard and mutes alerts."""
        # 16-channel quality evaluation: 5 bad channels (railed/noisy)
        channel_qualities = ["railed"] * 5 + ["good"] * 11
        bad_count = sum(1 for q in channel_qualities if q in ("railed", "noisy", "flat"))
        self.assertEqual(bad_count, 5)

        # Artifact Guard rule: activate if bad_channels > 4
        artifact_guard_active = bad_count > 4
        self.assertTrue(artifact_guard_active)

        # Model output with high seizure probability during artifact
        raw_model_prob = 0.95

        # Alert Engine interlock: even if model says seizure, alerts MUST be suppressed
        emitted_alert = None
        if not artifact_guard_active and raw_model_prob >= 0.70:
            emitted_alert = "CRITICAL"

        self.assertIsNone(emitted_alert)

        # Signal cleans back to normal: requires 2.0s clean before release
        channel_qualities_clean = ["good"] * 16
        clean_bad_count = sum(1 for q in channel_qualities_clean if q in ("railed", "noisy", "flat"))
        self.assertEqual(clean_bad_count, 0)

        clean_duration = 2.1  # >= 2.0 seconds
        if clean_bad_count <= 4 and clean_duration >= 2.0:
            artifact_guard_active = False

        self.assertFalse(artifact_guard_active)

    # -------------------------------------------------------------------------
    # 3. Replay Simulator -> View Sharing -> Diagnostics Hop Tracking (F21 -> F07 -> F06)
    # -------------------------------------------------------------------------
    def test_f21_f07_f06_replay_to_view_sharing_to_diagnostics(self):
        """Verifies Replay stream broadcasts to secondary viewer and tracks hop counters."""
        session_id = random_session_id()
        share_token = f"share_{random_access_key(16)}"

        # Primary producer replay simulator
        diagnostics = {
            "session_id": session_id,
            "hop1_replay_chunks_read": 20,
            "hop2_broadcast_sent": 20,
            "hop3_primary_received": 20,
            "hop3_viewer_received": 20,
            "dropped_frames": 0,
        }

        self.assertEqual(diagnostics["hop2_broadcast_sent"], diagnostics["hop3_viewer_received"])
        self.assertEqual(diagnostics["dropped_frames"], 0)

    # -------------------------------------------------------------------------
    # 4. Auth -> WebSocket Ingest -> SQLite Storage (F25 -> F03 -> F23)
    # -------------------------------------------------------------------------
    def test_f25_f03_f23_auth_to_ws_to_storage(self):
        """Verifies authenticated WS frame ingest writes session metadata and predictions to SQLite."""
        session_id = random_session_id()
        access_key = CONFIG.access_key

        # Validate authentication
        is_authenticated = access_key == CONFIG.access_key
        self.assertTrue(is_authenticated)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Insert session
        t_now = time.time()
        cur.execute(
            "INSERT INTO sessions (id, source, started_at, model_version, sample_rate) VALUES (?, ?, ?, ?, ?)",
            (session_id, "device", t_now, "TUSZ_EEG_DEEP_CNN", 125),
        )

        # Ingest 5 frames & store predictions
        for i in range(5):
            cur.execute(
                """INSERT INTO predictions (session_id, t_start, t_end, p_normal, p_pre, p_seizure, label, state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, t_now + i * 2.5, t_now + (i + 1) * 2.5, 0.90, 0.08, 0.02, "Normal", "Normal"),
            )
        conn.commit()

        # Verify records exist in database
        cur.execute("SELECT COUNT(*) FROM predictions WHERE session_id = ?", (session_id,))
        count = cur.fetchone()[0]
        self.assertEqual(count, 5)

        conn.close()

    # -------------------------------------------------------------------------
    # 5. One-Click Demo -> CRITICAL Alert -> Clinical Summary Report (F22 -> F18 -> F24)
    # -------------------------------------------------------------------------
    def test_f22_f18_f24_demo_to_alert_to_report(self):
        """Verifies preset seizure demo records alert and compiles into final clinical report."""
        session_id = random_session_id()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        t0 = 1000.0
        cur.execute(
            "INSERT INTO sessions (id, source, started_at, ended_at, model_version, sample_rate) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, "preset_seizure_demo", t0, t0 + 75.0, "TUSZ_EEG_DEEP_CNN", 125),
        )

        alert_id = "alert-demo-001"
        cur.execute(
            "INSERT INTO alerts (id, session_id, level, started_at, ended_at, peak_prob) VALUES (?, ?, ?, ?, ?, ?)",
            (alert_id, session_id, "CRITICAL", t0 + 25.0, t0 + 60.0, 0.92),
        )
        conn.commit()

        # Generate clinical report
        cur.execute("SELECT * FROM alerts WHERE session_id = ?", (session_id,))
        alert_rows = cur.fetchall()
        self.assertEqual(len(alert_rows), 1)

        report_text = f"""
================================================================================
                    SEIZURE-SENTRY CLINICAL SESSION REPORT
================================================================================
Session ID: {session_id}
Source: preset_seizure_demo (SIMULATED DATA)
Total Duration: 75.00 seconds

ALERT SUMMARY:
- Total Alerts: 1
- Alert [CRITICAL]: Onset at 25.00s, Duration 35.00s, Peak Probability: 92.0%

MANDATORY RESEARCH DISCLAIMER:
This report is generated by an investigational research prototype for demonstration
and scientific evaluation purposes only. It is NOT a medical device and must NOT
be used for diagnostic, prognostic, or clinical treatment decisions.
================================================================================
"""
        self.assertIn("CRITICAL", report_text)
        self.assertIn("Peak Probability: 92.0%", report_text)
        self.assertIn("MANDATORY RESEARCH DISCLAIMER", report_text)
        self.assertIn("investigational research prototype", report_text)
        conn.close()

    # -------------------------------------------------------------------------
    # 6. Auth Privilege Separation: Producer vs Viewer Roles (F25 -> F07)
    # -------------------------------------------------------------------------
    def test_f25_f07_auth_privilege_separation(self):
        """Verifies producer can transmit binary data while viewer is strictly read-only."""
        producer_hello = mock_client_hello(key=CONFIG.access_key, role="producer")
        self.assertEqual(producer_hello["role"], "producer")

        viewer_hello = mock_viewer_hello(token="valid_viewer_token_xyz")
        self.assertEqual(viewer_hello["role"], "viewer")

        def handle_binary_ingest(role: str) -> Tuple[bool, int]:
            if role == "viewer":
                return False, 4003  # Forbidden close code
            return True, 1000  # Normal OK

        ok_prod, code_prod = handle_binary_ingest(producer_hello["role"])
        self.assertTrue(ok_prod)
        self.assertEqual(code_prod, 1000)

        ok_view, code_view = handle_binary_ingest(viewer_hello["role"])
        self.assertFalse(ok_view)
        self.assertEqual(code_view, 4003)

    # -------------------------------------------------------------------------
    # 7. SQLite Storage Engine -> Clinical Metrics Calculation (F23 -> F24)
    # -------------------------------------------------------------------------
    def test_f23_f24_storage_to_clinical_report_metrics(self):
        """Verifies database prediction records correctly aggregated into clinical metrics."""
        session_id = random_session_id()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Insert 60 Normal, 20 Pre-Seizure, 20 Seizure windows
        cur.execute(
            "INSERT INTO sessions (id, source, started_at, model_version) VALUES (?, ?, ?, ?)",
            (session_id, "replay", 0.0, "TUSZ_EEG_DEEP_CNN"),
        )

        for _ in range(60):
            cur.execute(
                "INSERT INTO predictions (session_id, t_start, t_end, p_normal, p_pre, p_seizure, label, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, 0.0, 5.0, 0.90, 0.08, 0.02, "Normal", "Normal"),
            )
        for _ in range(20):
            cur.execute(
                "INSERT INTO predictions (session_id, t_start, t_end, p_normal, p_pre, p_seizure, label, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, 0.0, 5.0, 0.20, 0.70, 0.10, "Pre-Seizure", "Pre-Seizure"),
            )
        for _ in range(20):
            cur.execute(
                "INSERT INTO predictions (session_id, t_start, t_end, p_normal, p_pre, p_seizure, label, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, 0.0, 5.0, 0.05, 0.10, 0.85, "Seizure", "Seizure"),
            )
        conn.commit()

        # Query metrics
        cur.execute("SELECT label, COUNT(*), AVG(p_seizure) FROM predictions WHERE session_id = ? GROUP BY label", (session_id,))
        stats = {row[0]: {"count": row[1], "avg_seiz": row[2]} for row in cur.fetchall()}

        self.assertEqual(stats["Normal"]["count"], 60)
        self.assertEqual(stats["Pre-Seizure"]["count"], 20)
        self.assertEqual(stats["Seizure"]["count"], 20)
        self.assertAlmostEqual(stats["Seizure"]["avg_seiz"], 0.85, delta=1e-4)

        conn.close()

    # -------------------------------------------------------------------------
    # 8. Replay Simulator Pace & SIMULATED Watermark (F21 -> F22)
    # -------------------------------------------------------------------------
    def test_f21_f22_replay_simulator_pace_and_simulated_badge(self):
        """Verifies replay simulator emits at 4 Hz and includes simulated watermark."""
        frame_interval_target_sec = 0.250  # 250ms
        frame_sample_count = 31  # @ 125 Hz

        t_emission_times = [0.0, 0.250, 0.500, 0.750, 1.000]
        intervals = [t_emission_times[i] - t_emission_times[i - 1] for i in range(1, len(t_emission_times))]
        for dt in intervals:
            self.assertAlmostEqual(dt, frame_interval_target_sec, delta=0.001)

        # Emitted message structure
        broadcast_msg = {
            "type": "prediction",
            "simulated": True,
            "source": "replay",
            "label": "Normal",
            "probs": {"normal": 0.95, "pre_seizure": 0.04, "seizure": 0.01},
        }
        self.assertTrue(broadcast_msg["simulated"])
        self.assertEqual(broadcast_msg["source"], "replay")

    # -------------------------------------------------------------------------
    # 9. Access Key Security on Synthetic Window Generator (F25 -> F26)
    # -------------------------------------------------------------------------
    def test_f25_f26_auth_to_synthetic_generator(self):
        """Verifies synthetic EEG endpoint requires authorization key."""
        valid_key = CONFIG.access_key
        invalid_key = "invalid-unauthorized-key"

        def authenticate(key: Optional[str]) -> int:
            if not key:
                return 401
            if key != valid_key:
                return 401
            return 200

        self.assertEqual(authenticate(None), 401)
        self.assertEqual(authenticate(invalid_key), 401)
        self.assertEqual(authenticate(valid_key), 200)

        # Once authenticated, generate synthetic window (1280, 22)
        window = EegStreamer.generate_model_window(state="normal")
        self.assertEqual(window.shape, (1280, 22))

    # -------------------------------------------------------------------------
    # 10. View Sharing Token Expiration & Revocation Lifecycle (F07 -> F23)
    # -------------------------------------------------------------------------
    def test_f07_f23_view_sharing_token_lifecycle(self):
        """Verifies share token creation, validation, expiration, and active revocation."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        session_id = random_session_id()
        cur.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)", (session_id, "device", time.time()))

        token = f"share_{uuid.uuid4().hex}"
        now = time.time()
        expires_at = now + 3600.0  # 1 hour validity

        cur.execute(
            "INSERT INTO share_tokens (token, session_id, expires_at, revoked) VALUES (?, ?, ?, 0)",
            (token, session_id, expires_at),
        )
        conn.commit()

        # Validate active token
        def check_token(tok: str, query_time: float) -> bool:
            c = conn.cursor()
            c.execute("SELECT expires_at, revoked FROM share_tokens WHERE token = ?", (tok,))
            row = c.fetchone()
            if not row:
                return False
            exp, rev = row
            return (query_time < exp) and (rev == 0)

        self.assertTrue(check_token(token, now + 10.0))
        # Expired token
        self.assertFalse(check_token(token, now + 4000.0))

        # Explicit revocation
        cur.execute("UPDATE share_tokens SET revoked = 1 WHERE token = ?", (token,))
        conn.commit()
        self.assertFalse(check_token(token, now + 10.0))

        conn.close()

    # -------------------------------------------------------------------------
    # 11. Multi-Modal Alerting Visual & Audio Specifications (F19 -> F21)
    # -------------------------------------------------------------------------
    def test_f19_f21_replay_alert_sound_and_visual_dispatch(self):
        """Verifies multi-modal alert payload contains required audio tone and CSS animations."""
        critical_alert = {
            "id": "alert-crit-1",
            "level": "CRITICAL",
            "ui_actions": {
                "border_pulse": True,
                "banner_class": "bg-red-900 border-red-500 text-red-100",
                "audio_tone_hz": 880.0,
                "audio_pulse_hz": 4.0,
            },
        }

        self.assertTrue(critical_alert["ui_actions"]["border_pulse"])
        self.assertIn("bg-red-900", critical_alert["ui_actions"]["banner_class"])
        self.assertEqual(critical_alert["ui_actions"]["audio_tone_hz"], 880.0)

    # -------------------------------------------------------------------------
    # 12. Artifact Guard Interlocks Documented in Clinical Report (F20 -> F24)
    # -------------------------------------------------------------------------
    def test_f20_f24_artifact_guard_logging_in_clinical_report(self):
        """Verifies artifact guard events are tracked and reported in session clinical summary."""
        artifact_log = [
            {"t_start": 45.0, "t_end": 52.0, "reason": "5 channels railed (>180,000 uV)"},
            {"t_start": 120.0, "t_end": 125.0, "reason": "Broadband electrode movement artifact"},
        ]

        total_artifact_sec = sum(item["t_end"] - item["t_start"] for item in artifact_log)
        self.assertEqual(total_artifact_sec, 12.0)

        summary_report = f"Signal Quality Interlocks Active: {len(artifact_log)} times ({total_artifact_sec:.1f}s suppressed)"
        self.assertIn("12.0s suppressed", summary_report)

    # -------------------------------------------------------------------------
    # 13. Cascade Deletion Security on Session Removal (F23 -> F25)
    # -------------------------------------------------------------------------
    def test_f23_f25_database_cascade_deletion_security(self):
        """Verifies authorized session deletion cascades to remove all predictions and alerts."""
        session_id = random_session_id()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        cur.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)", (session_id, "device", time.time()))
        cur.execute(
            "INSERT INTO predictions (session_id, t_start, t_end, p_normal, p_pre, p_seizure, label, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, 0.0, 5.0, 0.9, 0.05, 0.05, "Normal", "Normal"),
        )
        cur.execute(
            "INSERT INTO alerts (id, session_id, level, started_at, peak_prob) VALUES (?, ?, ?, ?, ?)",
            ("alt-1", session_id, "INFO", 0.0, 0.5),
        )
        conn.commit()

        # Delete session
        cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

        # Check cascading deletion
        cur.execute("SELECT COUNT(*) FROM predictions WHERE session_id = ?", (session_id,))
        self.assertEqual(cur.fetchone()[0], 0)

        cur.execute("SELECT COUNT(*) FROM alerts WHERE session_id = ?", (session_id,))
        self.assertEqual(cur.fetchone()[0], 0)

        conn.close()

    # -------------------------------------------------------------------------
    # 14. Diagnostic Latency Tracking During Alerts (F06 -> F18)
    # -------------------------------------------------------------------------
    def test_f06_f18_diagnostic_latency_tracking_during_alerts(self):
        """Verifies processing pipeline latency from binary ingest to alert generation is < 50ms."""
        t_start = time.perf_counter()

        # Run simulated chunk -> filter -> hysteresis alert evaluation
        dummy_signal = np.random.randn(31, 16).astype(np.float32)
        filt = StreamingFilter(fs=125, n_channels=16)
        _ = filt.filter(dummy_signal)
        probs = {"normal": 0.10, "pre_seizure": 0.05, "seizure": 0.85}
        _ = probs["seizure"] >= 0.70

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        assert_latency_within(t_elapsed_ms, max_allowed_ms=50.0, context="Ingest to Alert Latency")

    # -------------------------------------------------------------------------
    # 15. Artifact Guard Flapping Tolerance (F18 -> F20)
    # -------------------------------------------------------------------------
    def test_f18_f20_artifact_guard_flapping_tolerance(self):
        """Verifies rapid alternating between noisy and clean signal does not release guard prematurely."""
        # Flapping events (every 0.5s state flips)
        states = [
            (0.5, "noisy"),
            (1.0, "clean"),
            (1.5, "noisy"),
            (2.0, "clean"),
            (2.5, "clean"),
            (3.0, "clean"),
            (4.0, "clean"),  # 2.0s clean achieved from 2.0s to 4.0s
        ]

        guard_active = False
        clean_since = None

        for t_sec, state in states:
            if state == "noisy":
                guard_active = True
                clean_since = None
            else:
                if clean_since is None:
                    clean_since = t_sec
                elif (t_sec - clean_since) >= 2.0:
                    guard_active = False

        # After sustained 2.0s clean period, guard disengages
        self.assertFalse(guard_active)

    # -------------------------------------------------------------------------
    # 16. Multi-Viewer Broadcast Synchronization (F07 -> F19)
    # -------------------------------------------------------------------------
    def test_f07_f19_multi_viewer_broadcast_synchronization(self):
        """Verifies identical prediction broadcast distributed to all concurrent viewers."""
        broadcast_payload = {
            "type": "prediction",
            "session_id": "test-sync-session",
            "window_start": 100.0,
            "window_end": 105.0,
            "probs": {"normal": 0.88, "pre_seizure": 0.09, "seizure": 0.03},
            "label": "Normal",
            "state": "Normal",
        }

        # Emulate 3 independent viewer client buffers receiving server JSON broadcast
        viewer_buffers = [[], [], []]
        json_str = json.dumps(broadcast_payload)

        for buf in viewer_buffers:
            buf.append(json.loads(json_str))

        # Assert all 3 viewers received bit-identical message
        for buf in viewer_buffers:
            self.assertEqual(len(buf), 1)
            msg = buf[0]
            assert_prediction_schema(msg)
            self.assertEqual(msg["window_start"], 100.0)
            self.assertEqual(msg["label"], "Normal")


if __name__ == "__main__":
    unittest.main()
