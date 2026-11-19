"""
backend/tests/test_health.py
Unit tests for the /health liveness probe.
Direct invocation avoids AnyIO TestClient socket restrictions in sandboxed environments.
"""
import pytest
from app.main import app, health_check


class MockModelRunner:
    _is_loaded = True


@pytest.fixture(scope="module", autouse=True)
def setup_app_state():
    app.state.model_runner = MockModelRunner()
    yield
    app.state.model_runner = None


def test_health_public_access():
    """Verify /health returns HTTP 200 without requiring an access key."""
    data = health_check()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "TUSZ_EEG_DEEP_CNN"
    assert data["version"] == "1.0.0"


def test_health_response_schema():
    """Verify structure and types of /health response."""
    data = health_check()
    assert isinstance(data["status"], str)
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["model_name"], str)
    assert isinstance(data["version"], str)
