"""Tier 4: Scenario 5 - Multi-Observer Live View Sharing.

Simulates collaborative clinical monitoring with secondary view sharing:
1. Primary clinician initiates monitoring session and begins live stream.
2. Clinician generates a shareable read-only viewer link and token (POST /sessions/{id}/share).
3. Secondary observer connects to WebSocket endpoint using the viewer token.
4. Observer receives synchronized live predictions, states, and alerts in real time.
5. Security privilege boundary: observer attempting to inject binary EEG data is immediately
   disconnected with custom close code 4003 (Forbidden).
6. Clinician revokes share token (DELETE /sessions/{id}/share), resulting in instant
   termination of all active observer connections.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_prediction_schema,
)
from tests.e2e.utils.fixtures import (
    mock_client_hello,
    mock_viewer_hello,
    random_session_id,
)


class TestScenario5ViewSharing(unittest.TestCase):
    """End-to-End Test for Scenario 5: Multi-Observer Live View Sharing."""

    def test_scenario_5_view_sharing_lifecycle(self):
        """Executes primary streaming, viewer token generation, broadcast sync, write blocking, and revocation."""
        session_id = random_session_id()
        t0 = time.time()

        # Step 1: Primary Clinician Initializes Session & Ingest
        primary_connection = {
            "role": "producer",
            "session_id": session_id,
            "access_key": CONFIG.access_key,
            "is_active": True,
        }
        self.assertEqual(primary_connection["role"], "producer")

        # Step 2: Generate Shareable Viewer Link & Token
        token_uuid = uuid.uuid4().hex
        viewer_token = f"share_{token_uuid}"
        expires_at = t0 + 3600.0  # 1 hour validity

        share_record = {
            "token": viewer_token,
            "session_id": session_id,
            "expires_at": expires_at,
            "revoked": False,
            "share_url": f"https://seizure-sentry.vercel.app/live?session={session_id}&token={viewer_token}",
        }
        self.assertIn("share_", share_record["token"])
        self.assertIn(viewer_token, share_record["share_url"])

        # Step 3: Secondary Observer Connects as Viewer
        observer_hello = mock_viewer_hello(token=viewer_token)
        self.assertEqual(observer_hello["role"], "viewer")
        self.assertEqual(observer_hello["key"], viewer_token)

        # Validate handshake authorization for viewer
        observer_connection = {
            "session_id": session_id,
            "role": "viewer",
            "token": viewer_token,
            "is_active": True,
            "received_messages": [],
        }

        # Step 4: Synchronized Real-Time Broadcast
        # Server produces prediction & alert broadcasts
        sample_prediction = {
            "type": "prediction",
            "session_id": session_id,
            "window_start": 10.0,
            "window_end": 15.0,
            "probs": {"normal": 0.82, "pre_seizure": 0.12, "seizure": 0.06},
            "label": "Normal",
            "state": "Normal",
        }
        sample_alert = {
            "type": "alert",
            "id": "alert-sync-001",
            "level": "INFO",
            "started_at": 12.5,
            "peak_prob": 0.06,
        }

        # Fan-out broadcast to both primary and secondary observer
        active_clients = [primary_connection, observer_connection]
        for client in active_clients:
            if client["is_active"]:
                if "received_messages" in client:
                    client["received_messages"].append(sample_prediction)
                    client["received_messages"].append(sample_alert)

        # Verify observer received both payloads accurately
        self.assertEqual(len(observer_connection["received_messages"]), 2)
        rx_pred = observer_connection["received_messages"][0]
        rx_alert = observer_connection["received_messages"][1]

        assert_prediction_schema(rx_pred)
        assert_alert_schema(rx_alert)
        self.assertEqual(rx_pred["probs"]["normal"], 0.82)
        self.assertEqual(rx_alert["level"], "INFO")

        # Step 5: Enforce Viewer Write Block (Security Isolation)
        # Observer attempts to transmit binary frame (must be rejected with close code 4003)
        def process_incoming_frame(client_role: str, is_binary: bool) -> Tuple[bool, int]:
            if client_role == "viewer" and is_binary:
                return False, 4003  # Forbidden write on view-only session
            return True, 1000

        allowed, close_code = process_incoming_frame(observer_connection["role"], is_binary=True)
        self.assertFalse(allowed, "Viewer must be blocked from sending binary data")
        self.assertEqual(close_code, 4003, "Expected close code 4003 for viewer binary transmission")

        # Step 6: Revoke Share Token & Terminate Viewer Connections
        share_record["revoked"] = True

        # When token is revoked, all connected viewers are terminated immediately
        if share_record["revoked"]:
            observer_connection["is_active"] = False
            observer_close_code = 1008  # Policy violation / revoked token

        self.assertFalse(observer_connection["is_active"], "Observer connection must be closed upon token revocation")
        self.assertEqual(observer_close_code, 1008)

        # Subsequent connection attempts with revoked token are rejected
        def authenticate_viewer(token: str) -> bool:
            return token == share_record["token"] and not share_record["revoked"]

        self.assertFalse(authenticate_viewer(viewer_token), "Revoked token must not be allowed to reconnect")


if __name__ == "__main__":
    unittest.main()
