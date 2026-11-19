"""
backend/tests/test_db.py
Unit tests for SQLite database storage, WAL mode, CRUD operations, cascading deletes, and clinical report.
"""
import time
import uuid
import pytest
from app.db import (
    init_db,
    get_db,
    create_session,
    get_session,
    stop_session,
    list_sessions,
    delete_session,
    record_prediction,
    get_session_predictions,
    record_alert,
    update_alert_end,
    get_session_alerts,
    create_share_token,
    validate_share_token,
    revoke_share_token,
    generate_clinical_report,
    generate_eeg_thumbnail_png,
    MANDATORY_DISCLAIMER,
)


@pytest.fixture
def test_db_path(tmp_path):
    """Fixture providing a temporary SQLite database path."""
    db_file = tmp_path / "test_sentry.db"
    init_db(db_path=db_file)
    return db_file


def test_schema_and_foreign_keys(test_db_path):
    """Verify tables exist and foreign keys are enforced."""
    with get_db(test_db_path) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "sessions" in tables
        assert "predictions" in tables
        assert "alerts" in tables
        assert "share_tokens" in tables

        fk_on = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk_on == 1


def test_session_lifecycle(test_db_path):
    """Verify session creation, retrieval, listing, stopping, and deletion."""
    sess_id = f"test-sess-{uuid.uuid4().hex[:8]}"
    created = create_session(
        session_id=sess_id,
        source="device",
        sample_rate=125,
        channels=["FP1", "FP2"],
        notes="Clinical trial unit test",
        db_path=test_db_path,
    )
    assert created["id"] == sess_id
    assert created["status"] == "active"

    # Get session
    retrieved = get_session(sess_id, db_path=test_db_path)
    assert retrieved is not None
    assert retrieved["id"] == sess_id
    assert retrieved["sample_rate"] == 125
    assert retrieved["channels"] == ["FP1", "FP2"]

    # List sessions
    all_sess = list_sessions(db_path=test_db_path)
    assert len(all_sess) >= 1
    assert any(s["id"] == sess_id for s in all_sess)

    # Stop session
    stopped = stop_session(sess_id, db_path=test_db_path)
    assert stopped is not None
    assert stopped["status"] == "stopped"
    assert stopped["ended_at"] is not None

    # Delete session
    assert delete_session(sess_id, db_path=test_db_path) is True
    assert get_session(sess_id, db_path=test_db_path) is None


def test_predictions_and_alerts_crud(test_db_path):
    """Verify inserting predictions and recording/updating alerts."""
    sess_id = f"test-sess-{uuid.uuid4().hex[:8]}"
    create_session(session_id=sess_id, db_path=test_db_path)

    # Insert prediction
    p_id = record_prediction(
        session_id=sess_id,
        t_start=0.0,
        t_end=5.0,
        probs={"normal": 0.85, "pre_seizure": 0.10, "seizure": 0.05},
        label="Normal",
        state="NORMAL",
        db_path=test_db_path,
    )
    assert p_id is not None

    preds = get_session_predictions(sess_id, db_path=test_db_path)
    assert len(preds) == 1
    assert preds[0]["label"] == "Normal"
    assert preds[0]["p_normal"] == pytest.approx(0.85)

    # Record alert
    alert_id = str(uuid.uuid4())
    record_alert(
        alert_id=alert_id,
        session_id=sess_id,
        level="CRITICAL",
        started_at=10.0,
        peak_prob=0.88,
        db_path=test_db_path,
    )

    alerts = get_session_alerts(sess_id, db_path=test_db_path)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "CRITICAL"
    assert alerts[0]["ended_at"] is None

    # Update alert end
    update_alert_end(alert_id, ended_at=25.0, peak_prob=0.94, db_path=test_db_path)
    updated_alerts = get_session_alerts(sess_id, db_path=test_db_path)
    assert updated_alerts[0]["ended_at"] == 25.0
    assert updated_alerts[0]["peak_prob"] == pytest.approx(0.94)


def test_cascade_delete(test_db_path):
    """Verify deleting a session cascades to predictions, alerts, and tokens."""
    sess_id = f"cascade-{uuid.uuid4().hex[:8]}"
    create_session(session_id=sess_id, db_path=test_db_path)

    record_prediction(
        session_id=sess_id,
        t_start=0.0,
        t_end=5.0,
        probs={"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05},
        label="Normal",
        state="NORMAL",
        db_path=test_db_path,
    )
    record_alert(
        alert_id=str(uuid.uuid4()),
        session_id=sess_id,
        level="WARNING",
        started_at=5.0,
        peak_prob=0.65,
        db_path=test_db_path,
    )
    create_share_token(session_id=sess_id, db_path=test_db_path)

    # Delete session
    delete_session(sess_id, db_path=test_db_path)

    with get_db(test_db_path) as conn:
        p_cnt = conn.execute("SELECT COUNT(*) FROM predictions WHERE session_id = ?", (sess_id,)).fetchone()[0]
        a_cnt = conn.execute("SELECT COUNT(*) FROM alerts WHERE session_id = ?", (sess_id,)).fetchone()[0]
        t_cnt = conn.execute("SELECT COUNT(*) FROM share_tokens WHERE session_id = ?", (sess_id,)).fetchone()[0]
        assert p_cnt == 0
        assert a_cnt == 0
        assert t_cnt == 0


def test_share_token_lifecycle(test_db_path):
    """Verify share token generation, expiration, and revocation."""
    sess_id = f"share-{uuid.uuid4().hex[:8]}"
    create_session(session_id=sess_id, db_path=test_db_path)

    # Valid token for 3600s
    res = create_share_token(session_id=sess_id, expires_in_sec=3600, db_path=test_db_path)
    token = res["share_token"]
    assert "share_" in token
    assert validate_share_token(token, db_path=test_db_path) is not None

    # Expired token (-10s)
    exp_res = create_share_token(session_id=sess_id, expires_in_sec=-10, db_path=test_db_path)
    assert validate_share_token(exp_res["share_token"], db_path=test_db_path) is None

    # Revoke valid token
    assert revoke_share_token(token, db_path=test_db_path) is True
    assert validate_share_token(token, db_path=test_db_path) is None


def test_clinical_report_generator(test_db_path):
    """Verify formatted clinical report output with mandatory disclaimer."""
    sess_id = f"report-{uuid.uuid4().hex[:8]}"
    create_session(session_id=sess_id, db_path=test_db_path)
    record_prediction(
        session_id=sess_id,
        t_start=0.0,
        t_end=5.0,
        probs={"normal": 0.88, "pre_seizure": 0.08, "seizure": 0.04},
        label="Normal",
        state="NORMAL",
        db_path=test_db_path,
    )
    stop_session(sess_id, db_path=test_db_path)

    report = generate_clinical_report(sess_id, db_path=test_db_path)
    assert sess_id in report
    assert "Summary Metrics" in report
    assert "Normal Windows: 1" in report
    assert MANDATORY_DISCLAIMER in report


def test_eeg_thumbnail_png_magic_bytes():
    """Verify valid PNG magic bytes returned for thumbnail."""
    png_bytes = generate_eeg_thumbnail_png()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) >= 8
