"""Utility helpers, custom assertions, and test fixtures for E2E testing."""

from tests.e2e.utils.assertions import (
    assert_alert_schema,
    assert_latency_within,
    assert_montage_shape,
    assert_prediction_schema,
    assert_probability_distribution,
    assert_signal_quality,
    assert_status_code,
    assert_valid_openbci_packet,
    assert_valid_ws_binary_frame,
)
from tests.e2e.utils.fixtures import (
    generate_mock_binary_frame,
    generate_openbci_raw_stream,
    get_incompatible_profile,
    get_openbci_cyton_profile,
    golden_model_input_tensor,
    mock_client_hello,
    mock_viewer_hello,
    random_access_key,
    random_session_id,
)

__all__ = [
    "assert_probability_distribution",
    "assert_montage_shape",
    "assert_latency_within",
    "assert_status_code",
    "assert_valid_openbci_packet",
    "assert_valid_ws_binary_frame",
    "assert_signal_quality",
    "assert_prediction_schema",
    "assert_alert_schema",
    "random_session_id",
    "random_access_key",
    "get_openbci_cyton_profile",
    "get_incompatible_profile",
    "generate_openbci_raw_stream",
    "generate_mock_binary_frame",
    "golden_model_input_tensor",
    "mock_client_hello",
    "mock_viewer_hello",
]
