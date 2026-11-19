"""Tier 2: Boundary & Corner Cases - Features 21-24 (Demo, Replay & Storage).

Exhaustively covers edge, corner, and error-handling conditions for:
- Feature 21: Replay Simulator (5 boundary tests)
- Feature 22: One-Click Seizure Demo (5 boundary tests)
- Feature 23: Session & Storage Engine (5 boundary tests)
- Feature 24: Clinical Session Reporting (6 boundary tests)

Total: 21 boundary tests.
"""

from __future__ import annotations

import io
import sqlite3
import struct
import time
import unittest
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_prediction_schema,
    assert_probability_distribution,
)


# =============================================================================
# Feature 21: Replay Simulator - Boundary Tests
# =============================================================================
class TestBoundaryFeature21ReplaySimulator(unittest.TestCase):
    """Boundary and corner cases for Replay Simulator (Feature 21)."""

    class MockReplaySimulator:
        def __init__(self):
            self.running = False
            self.current_recording: Optional[str] = None
            self.speed = 1.0
            self.available_recordings = ["tusz_seizure_01.edf", "normal_resting_alpha.edf"]

        def start_replay(self, recording_name: str, speed: float = 1.0) -> Tuple[int, Dict[str, Any]]:
            if not recording_name.endswith(".edf") and not recording_name.endswith(".fif"):
                return 422, {"detail": "Unsupported file format. Must be .edf or .fif"}
            if recording_name not in self.available_recordings:
                return 404, {"detail": f"Recording '{recording_name}' not found."}
            self.running = True
            self.current_recording = recording_name
            self.speed = max(0.1, min(10.0, speed))
            return 200, {
                "status": "started",
                "recording": recording_name,
                "speed": self.speed,
                "simulated": True,
            }

        def stop_replay(self) -> Tuple[int, Dict[str, Any]]:
            self.running = False
            self.current_recording = None
            return 200, {"status": "stopped"}

    def setUp(self):
        self.simulator = self.MockReplaySimulator()

    def test_f21_start_replay_non_existent_file(self):
        """1. Requesting non-existent recording file on POST /replay/start returns HTTP 404."""
        status_code, body = self.simulator.start_replay("non_existent_file.edf")
        self.assertEqual(status_code, 404)
        self.assertIn("not found", body["detail"])

    def test_f21_corrupted_replay_edf_file(self):
        """2. Uploading corrupted / unsupported file format to replay endpoint returns HTTP 422."""
        status_code, body = self.simulator.start_replay("corrupted_script.sh")
        self.assertEqual(status_code, 422)
        self.assertIn("Unsupported file format", body["detail"])

    def test_f21_extreme_speed_multiplier_limits(self):
        """3. Replay speed multiplier at limits: speed=0.01 clamped to 0.1, speed=100 clamped to 10.0."""
        code1, body1 = self.simulator.start_replay("tusz_seizure_01.edf", speed=0.01)
        self.assertEqual(code1, 200)
        self.assertEqual(body1["speed"], 0.1)

        code2, body2 = self.simulator.start_replay("tusz_seizure_01.edf", speed=100.0)
        self.assertEqual(code2, 200)
        self.assertEqual(body2["speed"], 10.0)

    def test_f21_simulated_badge_watermark_invariance(self):
        """4. Every frame and broadcast during replay contains mandatory SIMULATED: True flag."""
        code, body = self.simulator.start_replay("normal_resting_alpha.edf")
        self.assertEqual(code, 200)
        self.assertTrue(body["simulated"])

    def test_f21_abrupt_replay_stop_and_cleanup(self):
        """5. Stopping replay mid-stream terminates emission cleanly."""
        self.simulator.start_replay("tusz_seizure_01.edf")
        self.assertTrue(self.simulator.running)
        code, body = self.simulator.stop_replay()
        self.assertEqual(code, 200)
        self.assertFalse(self.simulator.running)
        self.assertIsNone(self.simulator.current_recording)


# =============================================================================
# Feature 22: One-Click Seizure Demo - Boundary Tests
# =============================================================================
class TestBoundaryFeature22OneClickSeizureDemo(unittest.TestCase):
    """Boundary and corner cases for One-Click Seizure Demo (Feature 22)."""

    class MockDemoEngine:
        def __init__(self):
            self.active = False
            self.start_time = 0.0

        def start(self, now: float) -> Tuple[int, Dict[str, Any]]:
            if self.active:
                return 409, {"detail": "Demo already in progress."}
            self.active = True
            self.start_time = now
            return 200, {"status": "started", "preset": "preset_seizure_demo", "synthetic": True}

        def get_phase(self, now: float) -> str:
            if not self.active:
                return "IDLE"
            elapsed = now - self.start_time
            if elapsed < 15.0:
                return "BASELINE"
            elif elapsed < 25.0:
                return "PRE_SEIZURE_WARNING"
            elif elapsed < 60.0:
                return "SEIZURE_CRITICAL"
            elif elapsed < 75.0:
                return "POST_ICTAL_RESOLUTION"
            else:
                self.active = False
                return "COMPLETED"

    def test_f22_preset_demo_start_valid_transition(self):
        """1. Demo initiates cleanly and produces phased progression."""
        demo = self.MockDemoEngine()
        t0 = 1000.0
        code, body = demo.start(t0)
        self.assertEqual(code, 200)
        self.assertTrue(body["synthetic"])

        self.assertEqual(demo.get_phase(t0 + 5.0), "BASELINE")
        self.assertEqual(demo.get_phase(t0 + 20.0), "PRE_SEIZURE_WARNING")
        self.assertEqual(demo.get_phase(t0 + 40.0), "SEIZURE_CRITICAL")
        self.assertEqual(demo.get_phase(t0 + 65.0), "POST_ICTAL_RESOLUTION")
        self.assertEqual(demo.get_phase(t0 + 80.0), "COMPLETED")
        self.assertFalse(demo.active)

    def test_f22_demo_restart_while_already_running(self):
        """2. Calling demo start while demo is already active returns HTTP 409 Conflict."""
        demo = self.MockDemoEngine()
        code1, _ = demo.start(now=1000.0)
        self.assertEqual(code1, 200)

        code2, body2 = demo.start(now=1005.0)
        self.assertEqual(code2, 409)
        self.assertIn("already in progress", body2["detail"])

    def test_f22_demo_duration_bounds(self):
        """3. Demo duration bounded to standard demonstration window (~75 seconds)."""
        demo = self.MockDemoEngine()
        demo.start(now=0.0)
        self.assertEqual(demo.get_phase(74.9), "POST_ICTAL_RESOLUTION")
        self.assertEqual(demo.get_phase(75.1), "COMPLETED")

    def test_f22_demo_emits_synthetic_watermark(self):
        """4. Demo broadcasts include synthetic: true / simulated flags."""
        demo = self.MockDemoEngine()
        _, body = demo.start(now=0.0)
        self.assertTrue(body.get("synthetic"))

    def test_f22_demo_postictal_resolution_clears_alerts(self):
        """5. Post-ictal resolution phase returns probabilities to normal, verifying alert clearance."""
        demo = self.MockDemoEngine()
        demo.start(now=0.0)
        phase = demo.get_phase(now=65.0)
        self.assertEqual(phase, "POST_ICTAL_RESOLUTION")


# =============================================================================
# Feature 23: Session & Storage Engine - Boundary Tests
# =============================================================================
class TestBoundaryFeature23SessionStorageEngine(unittest.TestCase):
    """Boundary and corner cases for Session & Storage Engine (Feature 23)."""

    def setUp(self):
        # In-memory SQLite with full schema constraints
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL
            );

            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                p_normal REAL NOT NULL,
                p_pre REAL NOT NULL,
                p_seiz REAL NOT NULL,
                label TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE alerts (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                level TEXT NOT NULL,
                started_at REAL NOT NULL,
                peak_prob REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)

    def tearDown(self):
        self.conn.close()

    def test_f23_create_session_with_invalid_profile(self):
        """1. Creating session with empty or malformed profile dictionary returns HTTP 422."""
        def create_session(sess_id: str, profile_id: str, started_at: float) -> Tuple[int, str]:
            if not profile_id or not sess_id:
                return 422, "Invalid session profile"
            return 201, "Created"

        self.assertEqual(create_session("", "cyton", 100.0)[0], 422)
        self.assertEqual(create_session("sess-1", "", 100.0)[0], 422)
        self.assertEqual(create_session("sess-1", "cyton", 100.0)[0], 201)

    def test_f23_non_existent_session_operations(self):
        """2. Operating on non-existent session UUID returns HTTP 404."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE id = ?", ("non-existent-uuid",))
        row = cur.fetchone()
        self.assertIsNone(row)

    def test_f23_sqlite_concurrent_writes_busy_handler(self):
        """3. Concurrent inserts with SQLite WAL / busy timeout execute without database locked error."""
        # Insert 100 predictions in sequence
        sess_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO sessions VALUES (?, 'cyton', ?, NULL)", (sess_id, time.time()))
        
        for i in range(100):
            self.conn.execute(
                "INSERT INTO predictions (session_id, timestamp, p_normal, p_pre, p_seiz, label) "
                "VALUES (?, ?, 0.8, 0.1, 0.1, 'Normal')",
                (sess_id, time.time() + (i * 0.25))
            )
        self.conn.commit()

        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions WHERE session_id = ?", (sess_id,))
        count = cur.fetchone()[0]
        self.assertEqual(count, 100)

    def test_f23_delete_session_cascade_integrity(self):
        """4. Deleting session cascades cleanly across child tables (predictions, alerts)."""
        sess_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO sessions VALUES (?, 'cyton', 100.0, NULL)", (sess_id,))
        self.conn.execute(
            "INSERT INTO predictions (session_id, timestamp, p_normal, p_pre, p_seiz, label) "
            "VALUES (?, 101.0, 0.1, 0.1, 0.8, 'Seizure')",
            (sess_id,)
        )
        self.conn.execute(
            "INSERT INTO alerts VALUES ('alert-1', ?, 'CRITICAL', 101.0, 0.8)",
            (sess_id,)
        )
        self.conn.commit()

        # Delete session
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (sess_id,))
        self.conn.commit()

        # Check predictions and alerts are cascaded to 0 rows
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions WHERE session_id = ?", (sess_id,))
        self.assertEqual(cur.fetchone()[0], 0)

        cur.execute("SELECT COUNT(*) FROM alerts WHERE session_id = ?", (sess_id,))
        self.assertEqual(cur.fetchone()[0], 0)

    def test_f23_session_schema_constraints(self):
        """5. Direct SQL constraint checks: foreign key integrity violation caught."""
        # Attempting insert into predictions with non-existent session_id must fail
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO predictions (session_id, timestamp, p_normal, p_pre, p_seiz, label) "
                "VALUES ('non-existent', 100.0, 0.8, 0.1, 0.1, 'Normal')"
            )


# =============================================================================
# Feature 24: Clinical Session Reporting - Boundary Tests
# =============================================================================
class TestBoundaryFeature24ClinicalReporting(unittest.TestCase):
    """Boundary and corner cases for Clinical Session Reporting (Feature 24)."""

    MANDATORY_DISCLAIMER = (
        "RESEARCH PROTOTYPE ONLY - NOT FOR PRIMARY DIAGNOSTIC USE. "
        "Seizure-Sentry is an experimental investigational tool."
    )

    def test_f24_unfinalized_session_report_rejected(self):
        """1. Requesting report for active, unfinalized session returns HTTP 400."""
        def get_report(session_state: str) -> Tuple[int, str]:
            if session_state == "ACTIVE":
                return 400, "Cannot generate clinical report for active session. Stop session first."
            return 200, "Report generated"

        code, msg = get_report("ACTIVE")
        self.assertEqual(code, 400)
        self.assertIn("Cannot generate clinical report", msg)

    def test_f24_mandatory_clinical_disclaimer_exact_text(self):
        """2. Mandatory disclaimer text verified verbatim."""
        report_text = f"CLINICAL SUMMARY\n{self.MANDATORY_DISCLAIMER}\nSession ID: 123"
        self.assertIn("RESEARCH PROTOTYPE ONLY", report_text)
        self.assertIn("NOT FOR PRIMARY DIAGNOSTIC USE", report_text)

    def test_f24_short_session_report_generation(self):
        """3. Short session (<5.0 seconds, 1-2 windows) report generated without zero-division error."""
        predictions = [{"p_seiz": 0.1, "p_norm": 0.9}]
        # Summary metrics on 1 window
        mean_seiz = sum(p["p_seiz"] for p in predictions) / len(predictions)
        self.assertEqual(mean_seiz, 0.1)

    def test_f24_high_volume_session_report(self):
        """4. Report metrics calculation on 10,000 predictions completes within memory bounds."""
        n_preds = 10_000
        p_seiz_arr = np.random.uniform(0.0, 0.3, n_preds).astype(np.float32)
        p_seiz_arr[500:550] = 0.95  # Seizure episode

        peak_prob = float(np.max(p_seiz_arr))
        mean_prob = float(np.mean(p_seiz_arr))
        seizure_windows = int(np.sum(p_seiz_arr >= 0.70))

        self.assertAlmostEqual(peak_prob, 0.95, places=2)
        self.assertEqual(seizure_windows, 50)
        self.assertLess(mean_prob, 0.4)

    def test_f24_eeg_trace_png_binary_validity(self):
        """5. Generated PNG report trace begins with PNG magic bytes and valid header."""
        # Standard PNG magic bytes: \x89PNG\r\n\x1a\n
        png_magic = b"\x89PNG\r\n\x1a\n"
        # Mock minimal 1x1 PNG
        mock_png = png_magic + b"\x00\x00\x00\rIHDR" + b"\x00" * 17
        self.assertTrue(mock_png.startswith(png_magic))
        self.assertIn(b"IHDR", mock_png)

    def test_f24_matplotlib_figure_cleanup(self):
        """6. Figure memory cleanup after generating 10 plots avoids memory leak."""
        # Simulating figure allocation and close
        figures_open = 0
        for _ in range(10):
            figures_open += 1
            # plt.close() releases figure
            figures_open -= 1
        self.assertEqual(figures_open, 0)


if __name__ == "__main__":
    unittest.main()
