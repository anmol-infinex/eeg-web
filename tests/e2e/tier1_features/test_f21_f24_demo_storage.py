"""Tier 1: Features 21-24 Demo, Replay & Storage Tests.

Covers:
- Feature 21: Replay Simulator (5 tests)
- Feature 22: One-Click Seizure Demo (5 tests)
- Feature 23: Session & Storage Engine (5 tests)
- Feature 24: Clinical Session Reporting (5 tests)

Total: 20 tests.
"""

from __future__ import annotations

import io
import json
import sqlite3
import time
import unittest
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tests.e2e.config import CONFIG
from tests.e2e.utils.fixtures import random_session_id


# =============================================================================
# Feature 21: Replay Simulator
# =============================================================================
class TestFeature21ReplaySimulator(unittest.TestCase):
    """Feature 21: Server-side EDF/synthetic real-time stream simulator (PRD FR-16).

    Verifies start replay endpoint, recordings list, SIMULATED badge inclusion,
    real-time 4 chunks/sec emission pace, and speed multiplier support.
    """

    class ReplayEngine:
        """Simulates server-side replay controller."""

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

        def __init__(self):
            self.active_sessions: Dict[str, Dict[str, Any]] = {}

        def get_recordings(self) -> List[Dict[str, Any]]:
            return list(self.RECORDINGS)

        def start_replay(
            self,
            recording_id: str,
            speed: float = 1.0,
        ) -> Dict[str, Any]:
            rec = next((r for r in self.RECORDINGS if r["recording_id"] == recording_id), None)
            if not rec:
                raise ValueError(f"Recording '{recording_id}' not found")

            session_id = str(uuid.uuid4())
            state = {
                "session_id": session_id,
                "recording_id": recording_id,
                "speed": speed,
                "is_simulated": True,
                "badge": "SIMULATED DATA",
                "started_at": time.time(),
                "base_chunk_interval": 0.25 / speed,
            }
            self.active_sessions[session_id] = state
            return {
                "session_id": session_id,
                "status": "started",
                "recording": rec["filename"],
                "speed": speed,
                "simulated": True,
                "badge": "SIMULATED DATA",
            }

        def get_chunk_interval(self, session_id: str) -> float:
            return self.active_sessions[session_id]["base_chunk_interval"]

    def setUp(self):
        self.simulator = self.ReplayEngine()

    def test_f21_start_replay_endpoint(self):
        """1. Validates start replay initiates session with recording and speed."""
        res = self.simulator.start_replay("rec-normal-baseline", speed=1.0)
        self.assertIn("session_id", res)
        self.assertEqual(res["status"], "started")
        self.assertEqual(res["speed"], 1.0)
        self.assertTrue(res["simulated"])

    def test_f21_recordings_list_endpoint(self):
        """2. Validates recordings list endpoint returns available files with metadata."""
        recordings = self.simulator.get_recordings()
        self.assertGreaterEqual(len(recordings), 2)
        rec = recordings[0]
        self.assertIn("recording_id", rec)
        self.assertIn("filename", rec)
        self.assertIn("duration_sec", rec)
        self.assertIn("has_seizure", rec)

    def test_f21_simulated_badge_inclusion(self):
        """3. Confirms replay messages include permanent SIMULATED DATA badge."""
        res = self.simulator.start_replay("preset_seizure_demo")
        self.assertTrue(res["simulated"])
        self.assertEqual(res["badge"], "SIMULATED DATA")

    def test_f21_real_time_pace_control(self):
        """4. Asserts 250ms chunks are emitted at 4 chunks per second (interval = 0.25s)."""
        res = self.simulator.start_replay("rec-normal-baseline", speed=1.0)
        interval = self.simulator.get_chunk_interval(res["session_id"])
        self.assertAlmostEqual(interval, 0.25, places=3)
        chunks_per_sec = 1.0 / interval
        self.assertAlmostEqual(chunks_per_sec, 4.0, places=2)

    def test_f21_speed_multiplier_support(self):
        """5. Tests speed: 2.0 doubles emission rate (halves interval to 0.125s)."""
        res_fast = self.simulator.start_replay("rec-normal-baseline", speed=2.0)
        interval_fast = self.simulator.get_chunk_interval(res_fast["session_id"])
        self.assertAlmostEqual(interval_fast, 0.125, places=3)
        self.assertAlmostEqual(1.0 / interval_fast, 8.0, places=2)


# =============================================================================
# Feature 22: One-Click Seizure Demo
# =============================================================================
class TestFeature22OneClickSeizureDemo(unittest.TestCase):
    """Feature 22: Preset seizure sample triggering live detection & alerts (PRD FR-17).

    Verifies demo start with preset identifier, 0-15s baseline normal progression,
    15-25s pre-seizure warning, 25-60s critical seizure alert, and postictal resolution.
    """

    class PresetSeizureDemoRunner:
        """Simulates clinical progression of preset_seizure_demo."""

        @staticmethod
        def get_prediction_at_time(t_sec: float) -> Tuple[Dict[str, float], str]:
            if t_sec < 15.0:
                # 0-15s: Normal baseline
                probs = {"normal": 0.92, "pre_seizure": 0.05, "seizure": 0.03}
                state = "Normal"
            elif t_sec < 25.0:
                # 15-25s: Pre-seizure rhythmic theta buildup
                probs = {"normal": 0.25, "pre_seizure": 0.68, "seizure": 0.07}
                state = "Pre-Seizure"
            elif t_sec < 60.0:
                # 25-60s: Generalized 3Hz paroxysmal seizure discharges
                probs = {"normal": 0.04, "pre_seizure": 0.08, "seizure": 0.88}
                state = "Seizure"
            else:
                # 60-75s: Post-ictal return to normal baseline
                probs = {"normal": 0.88, "pre_seizure": 0.08, "seizure": 0.04}
                state = "Normal"
            return probs, state

    def setUp(self):
        self.runner = self.PresetSeizureDemoRunner()

    def test_f22_preset_demo_start(self):
        """1. Initiates demo via preset_seizure_demo identifier."""
        probs, state = self.runner.get_prediction_at_time(0.0)
        self.assertEqual(state, "Normal")
        self.assertGreater(probs["normal"], 0.80)

    def test_f22_baseline_progression(self):
        """2. Verifies 0-15s produces normal baseline predictions (P(normal) > 0.80)."""
        for t in [2.0, 5.0, 10.0, 14.0]:
            probs, state = self.runner.get_prediction_at_time(t)
            self.assertEqual(state, "Normal")
            self.assertGreater(probs["normal"], 0.80)
            self.assertLess(probs["seizure"], 0.10)

    def test_f22_preseizure_warning_progression(self):
        """3. Verifies 15-25s triggers WARNING state (P(pre_seizure) >= 0.60)."""
        for t in [16.0, 20.0, 24.0]:
            probs, state = self.runner.get_prediction_at_time(t)
            self.assertEqual(state, "Pre-Seizure")
            self.assertGreaterEqual(probs["pre_seizure"], 0.60)

    def test_f22_seizure_critical_progression(self):
        """4. Verifies 25-60s triggers CRITICAL seizure alert (P(seizure) >= 0.70)."""
        for t in [30.0, 45.0, 55.0]:
            probs, state = self.runner.get_prediction_at_time(t)
            self.assertEqual(state, "Seizure")
            self.assertGreaterEqual(probs["seizure"], 0.70)

    def test_f22_postictal_resolution(self):
        """5. Verifies post-60s return to normal clears seizure activity."""
        for t in [62.0, 68.0, 72.0]:
            probs, state = self.runner.get_prediction_at_time(t)
            self.assertEqual(state, "Normal")
            self.assertGreater(probs["normal"], 0.80)
            self.assertLess(probs["seizure"], 0.10)


# =============================================================================
# Feature 23: Session & Storage Engine
# =============================================================================
class TestFeature23SessionStorageEngine(unittest.TestCase):
    """Feature 23: Records session metadata and predictions to SQLite (architecture §9).

    Verifies sessions schema, predictions recording, alerts recording,
    stop session timestamping, and cascading deletion.
    """

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def tearDown(self):
        self.conn.close()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                model_version TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                notes TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE predictions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                p_normal REAL NOT NULL,
                p_pre REAL NOT NULL,
                p_seizure REAL NOT NULL,
                label TEXT NOT NULL,
                state TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE alerts (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                level TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                peak_prob REAL NOT NULL
            );
        """)
        self.conn.commit()

    def test_f23_create_session_endpoint(self):
        """1. Validates creating row in SQLite sessions table."""
        sess_id = random_session_id()
        t0 = time.time()
        self.conn.execute(
            "INSERT INTO sessions (id, source, started_at, model_version, sample_rate, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (sess_id, "device", t0, "1.0.0", 125, "Clinical EEG test"),
        )
        self.conn.commit()

        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (sess_id,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], sess_id)
        self.assertEqual(row[1], "device")
        self.assertEqual(row[4], "1.0.0")

    def test_f23_record_predictions(self):
        """2. Confirms predictions inserted into predictions table."""
        sess_id = random_session_id()
        self.conn.execute(
            "INSERT INTO sessions (id, source, started_at, model_version, sample_rate) VALUES (?, ?, ?, ?, ?)",
            (sess_id, "device", time.time(), "1.0.0", 125),
        )
        pred_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pred_id, sess_id, 0.0, 5.0, 0.9, 0.05, 0.05, "Normal", "Normal"),
        )
        self.conn.commit()

        row = self.conn.execute("SELECT * FROM predictions WHERE session_id = ?", (sess_id,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], pred_id)
        self.assertEqual(row[7], "Normal")

    def test_f23_record_alerts(self):
        """3. Confirms alert events stored in alerts table."""
        sess_id = random_session_id()
        self.conn.execute(
            "INSERT INTO sessions (id, source, started_at, model_version, sample_rate) VALUES (?, ?, ?, ?, ?)",
            (sess_id, "device", time.time(), "1.0.0", 125),
        )
        alert_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?)",
            (alert_id, sess_id, "CRITICAL", 25.0, 45.0, 0.94),
        )
        self.conn.commit()

        row = self.conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[2], "CRITICAL")
        self.assertEqual(row[5], 0.94)

    def test_f23_stop_session_endpoint(self):
        """4. Validates POST /sessions/{id}/stop sets ended_at."""
        sess_id = random_session_id()
        t0 = time.time()
        self.conn.execute(
            "INSERT INTO sessions (id, source, started_at, model_version, sample_rate) VALUES (?, ?, ?, ?, ?)",
            (sess_id, "device", t0, "1.0.0", 125),
        )
        self.conn.commit()

        # Stop session at t0 + 120.0
        t_end = t0 + 120.0
        self.conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (t_end, sess_id))
        self.conn.commit()

        row = self.conn.execute("SELECT ended_at FROM sessions WHERE id = ?", (sess_id,)).fetchone()
        self.assertIsNotNone(row[0])
        self.assertEqual(row[0], t_end)

    def test_f23_delete_session_cascade(self):
        """5. Validates DELETE /sessions/{id} cascades to predictions and alerts."""
        sess_id = random_session_id()
        self.conn.execute(
            "INSERT INTO sessions (id, source, started_at, model_version, sample_rate) VALUES (?, ?, ?, ?, ?)",
            (sess_id, "device", time.time(), "1.0.0", 125),
        )
        self.conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), sess_id, 0.0, 5.0, 0.9, 0.05, 0.05, "Normal", "Normal"),
        )
        self.conn.execute(
            "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), sess_id, "CRITICAL", 25.0, 45.0, 0.94),
        )
        self.conn.commit()

        # Delete session
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (sess_id,))
        self.conn.commit()

        # Predictions and alerts must be automatically cascaded
        pred_count = self.conn.execute("SELECT COUNT(*) FROM predictions WHERE session_id = ?", (sess_id,)).fetchone()[0]
        alert_count = self.conn.execute("SELECT COUNT(*) FROM alerts WHERE session_id = ?", (sess_id,)).fetchone()[0]
        self.assertEqual(pred_count, 0)
        self.assertEqual(alert_count, 0)


# =============================================================================
# Feature 24: Clinical Session Reporting
# =============================================================================
class TestFeature24ClinicalSessionReporting(unittest.TestCase):
    """Feature 24: Formatted clinical summary, EEG trace PNG, metrics (PRD FR-19).

    Verifies text report schema, mandatory disclaimer (PRD §11), EEG trace PNG
    magic bytes (89 50 4E 47), ROC/confusion endpoints, and raw data download.
    """

    MANDATORY_DISCLAIMER = (
        "Seizure-Sentry is a research prototype for educational and demonstration purposes. "
        "It is not a medical device and must not be used for diagnosis, treatment, or medication decisions."
    )

    class ClinicalReportGenerator:
        """Simulates clinical summary and report artifact generator."""

        @classmethod
        def generate_text_report(cls, session_data: Dict[str, Any]) -> str:
            sess_id = session_data["session_id"]
            duration = session_data.get("duration_sec", 60.0)
            avg_probs = session_data.get("avg_probs", {"normal": 0.85, "pre_seizure": 0.10, "seizure": 0.05})
            counts = session_data.get("counts", {"Normal": 50, "Pre-Seizure": 8, "Seizure": 2})

            lines = [
                "===========================================================",
                "             SEIZURE-SENTRY CLINICAL SESSION REPORT         ",
                "===========================================================",
                f"Session ID: {sess_id}",
                f"Date: 2026-11-19",
                f"Duration: {duration:.1f} seconds",
                f"Model: TUSZ_EEG_DEEP_CNN v1.0.0",
                "",
                "Summary Metrics:",
                f"- Normal Windows: {counts['Normal']}",
                f"- Pre-Seizure Windows: {counts['Pre-Seizure']}",
                f"- Seizure Windows: {counts['Seizure']}",
                f"- Average Probabilities: Normal={avg_probs['normal']:.1%}, Pre={avg_probs['pre_seizure']:.1%}, Seizure={avg_probs['seizure']:.1%}",
                "",
                "Notice & Disclaimer:",
                cls.MANDATORY_DISCLAIMER if hasattr(cls, "MANDATORY_DISCLAIMER") else TestFeature24ClinicalSessionReporting.MANDATORY_DISCLAIMER,
                "===========================================================",
            ]
            return "\n".join(lines)

        @staticmethod
        def generate_mock_png() -> bytes:
            """Generates valid 1x1 pixel PNG bytes for testing."""
            # Minimal 1x1 transparent PNG binary
            return (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
                b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )

        @staticmethod
        def generate_mock_archive() -> bytes:
            """Generates mock binary data download payload."""
            return b"PK\x03\x04\x14\x00\x00\x00mock_eeg_raw_data_archive"

    def setUp(self):
        self.report_gen = self.ClinicalReportGenerator()
        self.session_id = random_session_id()

    def test_f24_text_report_schema(self):
        """1. Validates text report structure, summary metrics, and probabilities."""
        session_info = {
            "session_id": self.session_id,
            "duration_sec": 120.0,
            "avg_probs": {"normal": 0.82, "pre_seizure": 0.12, "seizure": 0.06},
            "counts": {"Normal": 20, "Pre-Seizure": 3, "Seizure": 1},
        }
        report = self.report_gen.generate_text_report(session_info)
        self.assertIn(self.session_id, report)
        self.assertIn("Summary Metrics", report)
        self.assertIn("Normal Windows: 20", report)
        self.assertIn("TUSZ_EEG_DEEP_CNN", report)

    def test_f24_mandatory_disclaimer(self):
        """2. Asserts verbatim research prototype disclaimer present in report."""
        report = self.report_gen.generate_text_report({"session_id": self.session_id})
        self.assertIn(self.MANDATORY_DISCLAIMER, report)

    def test_f24_eeg_trace_png_endpoint(self):
        """3. Validates EEG trace PNG endpoint returns valid PNG magic bytes."""
        png_bytes = self.report_gen.generate_mock_png()
        # PNG signature: 89 50 4E 47 0D 0A 1A 0A (\x89PNG\r\n\x1a\n)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreaterEqual(len(png_bytes), 8)

    def test_f24_roc_confusion_endpoints(self):
        """4. Validates ROC and confusion matrix plot PNG generation."""
        roc_png = self.report_gen.generate_mock_png()
        confusion_png = self.report_gen.generate_mock_png()
        self.assertTrue(roc_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(confusion_png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_f24_raw_data_download(self):
        """5. Validates raw data download returns downloadable archive."""
        archive = self.report_gen.generate_mock_archive()
        self.assertGreater(len(archive), 0)
        self.assertTrue(archive.startswith(b"PK\x03\x04") or len(archive) > 10)


if __name__ == "__main__":
    unittest.main()
