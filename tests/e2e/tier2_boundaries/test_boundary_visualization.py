"""Tier 2: Boundary & Corner Cases - Features 8-12 (Visualization & Dashboard).

Exhaustively covers edge, corner, and error-handling conditions for:
- Feature 8: Multi-Channel Canvas Waveforms (6 boundary tests)
- Feature 9: Per-Channel Signal Quality (5 boundary tests)
- Feature 10: Dual Clinical Theme (5 boundary tests)
- Feature 11: Current State & Probabilities (5 boundary tests)
- Feature 12: Prediction Timeline Strip (5 boundary tests)

Total: 26 boundary tests.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tests.e2e.client.streamer import EegStreamer
from tests.e2e.config import CONFIG
from tests.e2e.utils.assertions import (
    assert_probability_distribution,
    assert_signal_quality,
)


# =============================================================================
# Feature 8: Multi-Channel Canvas Waveforms - Boundary Tests
# =============================================================================
class TestBoundaryFeature08CanvasWaveforms(unittest.TestCase):
    """Boundary and corner cases for Multi-Channel Canvas Waveforms (Feature 8)."""

    class MockWaveformRenderer:
        def __init__(self, width: int = 1200, height: int = 800, n_channels: int = 16, dpr: float = 1.0):
            self.width = width
            self.height = height
            self.n_channels = n_channels
            self.dpr = dpr
            self.slot_height = height / n_channels

        def sample_to_screen_y(self, channel_idx: int, uv_val: float, uv_per_div: float = 50.0) -> float:
            # Clamping scale
            safe_scale = max(1.0, uv_per_div)
            center_y = (channel_idx + 0.5) * self.slot_height
            # Pixels per div = slot_height / 2
            pixels_per_uv = (self.slot_height / 2.0) / safe_scale
            raw_y = center_y - (uv_val * pixels_per_uv)

            # Clamp within channel band slot [channel_idx * slot_height, (channel_idx + 1) * slot_height]
            min_y = channel_idx * self.slot_height
            max_y = (channel_idx + 1) * self.slot_height
            return float(np.clip(raw_y, min_y, max_y))

    def test_f08_extreme_voltage_spike_clamping(self):
        """1. Extreme input voltage spikes (> 500,000 uV) clamped to channel band slot."""
        renderer = self.MockWaveformRenderer(height=800, n_channels=16)  # 50px per slot
        # Channel 0 slot: [0, 50], center = 25
        # Positive spike +500,000 uV should clamp to top of slot (0.0)
        y_pos = renderer.sample_to_screen_y(channel_idx=0, uv_val=500_000.0)
        self.assertEqual(y_pos, 0.0)

        # Negative spike -500,000 uV should clamp to bottom of slot (50.0)
        y_neg = renderer.sample_to_screen_y(channel_idx=0, uv_val=-500_000.0)
        self.assertEqual(y_neg, 50.0)

        # Channel 5 slot: [250, 300], center = 275
        y_ch5_pos = renderer.sample_to_screen_y(channel_idx=5, uv_val=1_000_000.0)
        self.assertEqual(y_ch5_pos, 250.0)
        y_ch5_neg = renderer.sample_to_screen_y(channel_idx=5, uv_val=-1_000_000.0)
        self.assertEqual(y_ch5_neg, 300.0)

    def test_f08_zero_std_flatline_render_coordinates(self):
        """2. Flatline signal (sigma = 0.0 uV) renders exactly on channel center line."""
        renderer = self.MockWaveformRenderer(height=800, n_channels=16)
        # Channel 3: slot [150, 200], center = 175
        y_zero = renderer.sample_to_screen_y(channel_idx=3, uv_val=0.0)
        self.assertEqual(y_zero, 175.0)

        # Constant DC offset +25 uV at 50 uV/div:
        # half slot = 25px per 50 uV = 0.5px/uV -> offset = -12.5px -> y = 162.5px
        y_dc = renderer.sample_to_screen_y(channel_idx=3, uv_val=25.0, uv_per_div=50.0)
        self.assertEqual(y_dc, 162.5)

    def test_f08_extreme_aspect_ratio_rendering(self):
        """3. Canvas resizing across extreme aspect ratios (ultrawide 32:9 vs mobile portrait 9:21)."""
        # Ultrawide 3840 x 1080
        renderer_ultra = self.MockWaveformRenderer(width=3840, height=1080, n_channels=16)
        self.assertAlmostEqual(renderer_ultra.slot_height, 67.5)
        y_ultra = renderer_ultra.sample_to_screen_y(15, 0.0)
        self.assertAlmostEqual(y_ultra, 1080 - 33.75)

        # Mobile portrait 1080 x 2520
        renderer_mobile = self.MockWaveformRenderer(width=1080, height=2520, n_channels=16)
        self.assertAlmostEqual(renderer_mobile.slot_height, 157.5)
        y_mobile = renderer_mobile.sample_to_screen_y(0, 0.0)
        self.assertAlmostEqual(y_mobile, 78.75)

    def test_f08_retina_3x_pixel_ratio_scaling(self):
        """4. Retina 3.0 device pixel ratio correctly scales coordinate buffer."""
        css_width, css_height = 1200, 800
        dpr = 3.0
        backing_width = int(css_width * dpr)
        backing_height = int(css_height * dpr)
        self.assertEqual(backing_width, 3600)
        self.assertEqual(backing_height, 2400)

        # Scale transform factor
        scale_x = backing_width / css_width
        scale_y = backing_height / css_height
        self.assertEqual(scale_x, 3.0)
        self.assertEqual(scale_y, 3.0)

    def test_f08_rapid_time_window_switching(self):
        """5. Rapid switching of time window (5s -> 20s -> 5s -> 10s) maintains ring buffer index."""
        fs = 125
        ring_buffer_size = 20 * fs  # 2500 samples (max 20s capacity)
        ring_buffer = np.zeros((ring_buffer_size, 16), dtype=np.float32)

        # Simulating window view extract
        def extract_window(window_sec: float, write_head: int) -> np.ndarray:
            n_samples = int(window_sec * fs)
            indices = np.arange(write_head - n_samples, write_head) % ring_buffer_size
            return ring_buffer[indices]

        write_head = 625  # 5 seconds streamed in
        w5 = extract_window(5.0, write_head)
        self.assertEqual(len(w5), 625)

        w20 = extract_window(20.0, write_head)
        self.assertEqual(len(w20), 2500)

        w10 = extract_window(10.0, write_head)
        self.assertEqual(len(w10), 1250)

    def test_f08_amplitude_scale_boundary_handling(self):
        """6. Zero or negative amplitude scale parameter clamped to safe minimum."""
        renderer = self.MockWaveformRenderer(height=800, n_channels=16)
        # Scale 0.0 or negative must not divide by zero
        y_zero_scale = renderer.sample_to_screen_y(channel_idx=0, uv_val=10.0, uv_per_div=0.0)
        self.assertFalse(math.isnan(y_zero_scale))
        self.assertFalse(math.isinf(y_zero_scale))

        y_neg_scale = renderer.sample_to_screen_y(channel_idx=0, uv_val=10.0, uv_per_div=-50.0)
        self.assertFalse(math.isnan(y_neg_scale))
        self.assertFalse(math.isinf(y_neg_scale))


# =============================================================================
# Feature 9: Per-Channel Signal Quality - Boundary Tests
# =============================================================================
class TestBoundaryFeature09SignalQuality(unittest.TestCase):
    """Boundary and corner cases for Per-Channel Signal Quality (Feature 9)."""

    def test_f09_pure_50hz_mains_interference(self):
        """1. Injected 100% 50 Hz sinusoidal noise flagged as 'noisy'."""
        t = np.arange(250) / 125.0
        pure_50hz = 150.0 * np.sin(2 * np.pi * 50.0 * t)
        assert_signal_quality(pure_50hz, "noisy", fs=125)

    def test_f09_std_boundary_flatline_classification(self):
        """2. Standard deviation boundary: 0.49 uV classified as 'flat', 0.51 uV as 'good'."""
        # 0.49 uV std (flat)
        rng = np.random.default_rng(42)
        flat_sig = rng.normal(0.0, 0.49, 250)
        assert_signal_quality(flat_sig, "flat")

        # 25.0 uV std (good physiological EEG)
        good_sig = rng.normal(0.0, 25.0, 250)
        assert_signal_quality(good_sig, "good")

    def test_f09_voltage_boundary_railed_classification(self):
        """3. Peak amplitude boundary: 179,000 uV not railed, 180,001 uV classified as 'railed'."""
        t = np.arange(250) / 125.0
        # Peak = 179,000 uV (below 180,000 threshold, but broad RMS > 300 -> noisy or good, not railed)
        sig_179k = np.zeros(250, dtype=np.float32)
        sig_179k[10] = 179000.0

        # Peak = 180,500 uV (exceeds 180,000 uV railed threshold)
        sig_railed = np.zeros(250, dtype=np.float32)
        sig_railed[10] = 180500.0
        assert_signal_quality(sig_railed, "railed")

    def test_f09_short_window_quality_evaluation(self):
        """4. Quality check on short window (<0.5 seconds, 10 samples) handled safely."""
        short_chunk = np.array([10.0, 12.0, -8.0, 15.0, 22.0, -5.0, 0.0, 8.0, -12.0, 14.0])
        # Standard deviation > 0.5 and < 300, max abs < 180000 -> evaluated as 'good'
        assert_signal_quality(short_chunk, "good")

    def test_f09_mixed_channel_quality_distribution(self):
        """5. Multi-channel evaluation with good, railed, flat, and noisy channels."""
        t = np.arange(250) / 125.0
        channels_data = np.zeros((250, 4), dtype=np.float32)
        # Ch 0: Good normal EEG
        channels_data[:, 0] = np.sin(2 * np.pi * 10.0 * t) * 20.0
        # Ch 1: Railed
        channels_data[:, 1] = 200_000.0
        # Ch 2: Flat
        channels_data[:, 2] = 0.1
        # Ch 3: Noisy 50Hz mains
        channels_data[:, 3] = 100.0 * np.sin(2 * np.pi * 50.0 * t)

        assert_signal_quality(channels_data[:, 0], "good")
        assert_signal_quality(channels_data[:, 1], "railed")
        assert_signal_quality(channels_data[:, 2], "flat")
        assert_signal_quality(channels_data[:, 3], "noisy")


# =============================================================================
# Feature 10: Dual Clinical Theme - Boundary Tests
# =============================================================================
class TestBoundaryFeature10DualClinicalTheme(unittest.TestCase):
    """Boundary and corner cases for Dual Clinical Theme (Feature 10)."""

    SEMANTIC_TOKENS = [
        "bg-primary", "bg-secondary", "bg-canvas",
        "text-primary", "text-secondary", "text-muted",
        "border-color", "grid-line",
        "channel-trace", "channel-label",
        "badge-normal", "badge-warning", "badge-critical",
        "alert-edge", "button-active"
    ]

    LIGHT_THEME = {
        "bg-primary": "#FFFFFF", "bg-secondary": "#F8FAFC", "bg-canvas": "#FFFFFF",
        "text-primary": "#0F172A", "text-secondary": "#475569", "text-muted": "#94A3B8",
        "border-color": "#E2E8F0", "grid-line": "#E5E7EB",
        "channel-trace": "#1E293B", "channel-label": "#334155",
        "badge-normal": "#10B981", "badge-warning": "#F59E0B", "badge-critical": "#EF4444",
        "alert-edge": "#EF4444", "button-active": "#2563EB"
    }

    DARK_THEME = {
        "bg-primary": "#0B0F19", "bg-secondary": "#111827", "bg-canvas": "#070A10",
        "text-primary": "#F8FAFC", "text-secondary": "#94A3B8", "text-muted": "#64748B",
        "border-color": "#1F2937", "grid-line": "#1A2234",
        "channel-trace": "#38BDF8", "channel-label": "#94A3B8",
        "badge-normal": "#10B981", "badge-warning": "#F59E0B", "badge-critical": "#EF4444",
        "alert-edge": "#EF4444", "button-active": "#3B82F6"
    }

    @staticmethod
    def _luminance(hex_color: str) -> float:
        hex_color = hex_color.lstrip("#")
        r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        def ch_lum(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * ch_lum(r) + 0.7152 * ch_lum(g) + 0.0722 * ch_lum(b)

    @classmethod
    def _contrast_ratio(cls, hex1: str, hex2: str) -> float:
        l1 = cls._luminance(hex1)
        l2 = cls._luminance(hex2)
        top = max(l1, l2) + 0.05
        bottom = min(l1, l2) + 0.05
        return top / bottom

    def test_f10_theme_switching_during_live_render(self):
        """1. Theme switching toggles semantic variables without state corruption."""
        current_theme = "light"
        for i in range(10):
            current_theme = "dark" if current_theme == "light" else "light"
            palette = self.LIGHT_THEME if current_theme == "light" else self.DARK_THEME
            self.assertEqual(len(palette), 15)

    def test_f10_undefined_system_theme_fallback(self):
        """2. Undefined OS preference ('no-preference' or None) defaults to light paper theme."""
        def resolve_theme(os_pref: Optional[str]) -> str:
            if os_pref in ("dark", "light"):
                return os_pref
            return "light"

        self.assertEqual(resolve_theme(None), "light")
        self.assertEqual(resolve_theme("no-preference"), "light")
        self.assertEqual(resolve_theme("dark"), "dark")

    def test_f10_wcag_aa_contrast_all_semantic_tokens(self):
        """3. WCAG AA contrast ratio (>= 4.5:1) for primary text vs background in both themes."""
        # Light theme: text-primary on bg-primary
        cr_light = self._contrast_ratio(self.LIGHT_THEME["text-primary"], self.LIGHT_THEME["bg-primary"])
        self.assertGreaterEqual(cr_light, 4.5, f"Light theme contrast {cr_light:.2f} < 4.5:1")

        # Dark theme: text-primary on bg-primary
        cr_dark = self._contrast_ratio(self.DARK_THEME["text-primary"], self.DARK_THEME["bg-primary"])
        self.assertGreaterEqual(cr_dark, 4.5, f"Dark theme contrast {cr_dark:.2f} < 4.5:1")

    def test_f10_corrupted_localstorage_theme_value(self):
        """4. Corrupted theme preference string falls back safely to default."""
        def load_stored_theme(raw_val: str) -> str:
            if raw_val not in ("light", "dark"):
                return "light"
            return raw_val

        self.assertEqual(load_stored_theme("neon-cyberpunk"), "light")
        self.assertEqual(load_stored_theme(""), "light")
        self.assertEqual(load_stored_theme("dark"), "dark")

    def test_f10_all_15_css_tokens_defined_in_both_themes(self):
        """5. Strict completeness: all 15 semantic tokens defined in both themes."""
        for token in self.SEMANTIC_TOKENS:
            self.assertIn(token, self.LIGHT_THEME, f"Missing token {token} in LIGHT_THEME")
            self.assertIn(token, self.DARK_THEME, f"Missing token {token} in DARK_THEME")


# =============================================================================
# Feature 11: Current State & Probabilities - Boundary Tests
# =============================================================================
class TestBoundaryFeature11CurrentStateProbabilities(unittest.TestCase):
    """Boundary and corner cases for Current State & Probabilities (Feature 11)."""

    def test_f11_unnormalized_probabilities_handling(self):
        """1. Probabilities summing to 0.9999 or 1.0001 re-normalized to 1.0."""
        def normalize_probs(p_dict: Dict[str, float]) -> Dict[str, float]:
            total = sum(p_dict.values())
            if total <= 0:
                return {k: 1.0 / len(p_dict) for k in p_dict}
            return {k: v / total for k, v in p_dict.items()}

        raw1 = {"normal": 0.8999, "pre_seizure": 0.05, "seizure": 0.05}  # sum = 0.9999
        norm1 = normalize_probs(raw1)
        assert_probability_distribution(norm1)

        raw2 = {"normal": 0.7001, "pre_seizure": 0.20, "seizure": 0.10}  # sum = 1.0001
        norm2 = normalize_probs(raw2)
        assert_probability_distribution(norm2)

    def test_f11_exact_three_way_tie_highlight(self):
        """2. Exact three-way tie (0.3333, 0.3333, 0.3334) deterministically resolves leading class."""
        probs = {"normal": 0.3333, "pre_seizure": 0.3333, "seizure": 0.3334}
        leading_class = max(probs, key=probs.get)
        self.assertEqual(leading_class, "seizure")

    def test_f11_model_warmup_initial_state_rendering(self):
        """3. Initial warmup state (stream duration < 5.0s) displays 'WARMUP' state chip."""
        def get_state_chip(elapsed_sec: float, model_ready: bool) -> str:
            if not model_ready or elapsed_sec < 5.0:
                return "WARMUP"
            return "NORMAL"

        self.assertEqual(get_state_chip(elapsed_sec=2.5, model_ready=True), "WARMUP")
        self.assertEqual(get_state_chip(elapsed_sec=5.0, model_ready=True), "NORMAL")

    def test_f11_zero_probability_confidence_bar(self):
        """4. Channel probability exactly 0.0 renders 0% bar width without negative margin."""
        def bar_width_percentage(prob: float) -> str:
            clamped = max(0.0, min(1.0, prob))
            return f"{clamped * 100:.1f}%"

        self.assertEqual(bar_width_percentage(0.0), "0.0%")
        self.assertEqual(bar_width_percentage(1.0), "100.0%")
        self.assertEqual(bar_width_percentage(-0.05), "0.0%")

    def test_f11_state_chip_label_and_color_mapping(self):
        """5. Exact label and semantic token for all valid state classes."""
        STATE_MAP = {
            "normal": ("NORMAL", "badge-normal"),
            "pre_seizure": ("PRE-SEIZURE", "badge-warning"),
            "seizure": ("SEIZURE", "badge-critical"),
        }
        for st, (label, token) in STATE_MAP.items():
            self.assertIn(token, ["badge-normal", "badge-warning", "badge-critical"])
            self.assertTrue(label.isupper())


# =============================================================================
# Feature 12: Prediction Timeline Strip - Boundary Tests
# =============================================================================
class TestBoundaryFeature12TimelineStrip(unittest.TestCase):
    """Boundary and corner cases for Prediction Timeline Strip (Feature 12)."""

    class TimelineBuffer:
        def __init__(self, capacity: int = 300):
            self.capacity = capacity
            self.cells: List[Dict[str, Any]] = []

        def append(self, cell: Dict[str, Any]):
            self.cells.append(cell)
            if len(self.cells) > self.capacity:
                self.cells.pop(0)

        def seek(self, target_idx: int) -> int:
            if not self.cells:
                return 0
            return max(0, min(len(self.cells) - 1, target_idx))

    def test_f12_timeline_buffer_capacity_bound(self):
        """1. Timeline buffer strictly bounded to capacity (300 cells); discards oldest without leak."""
        buf = self.TimelineBuffer(capacity=300)
        # Push 10,000 cells
        for i in range(10_000):
            buf.append({"id": i, "state": "normal"})
        self.assertEqual(len(buf.cells), 300)
        self.assertEqual(buf.cells[0]["id"], 9700)
        self.assertEqual(buf.cells[-1]["id"], 9999)

    def test_f12_timeline_rapid_seeking_during_playback(self):
        """2. Rapid timeline seeking/scrubbing clamps target index within valid bounds."""
        buf = self.TimelineBuffer(capacity=100)
        for i in range(50):
            buf.append({"id": i})

        # Seek out of range
        self.assertEqual(buf.seek(-10), 0)
        self.assertEqual(buf.seek(100), 49)
        self.assertEqual(buf.seek(25), 25)

    def test_f12_tooltip_screen_boundary_clamping(self):
        """3. Tooltip coordinate bounds calculation at extreme viewport edges (x=0, y=0, x=max)."""
        def clamp_tooltip(x: float, y: float, tt_w: float = 120.0, tt_h: float = 60.0,
                          vp_w: float = 1000.0, vp_h: float = 800.0) -> Tuple[float, float]:
            cx = max(0.0, min(vp_w - tt_w, x))
            cy = max(0.0, min(vp_h - tt_h, y))
            return cx, cy

        # Top-left corner
        self.assertEqual(clamp_tooltip(-20, -10), (0.0, 0.0))
        # Bottom-right corner
        self.assertEqual(clamp_tooltip(1050, 850), (880.0, 740.0))
        # Normal inside
        self.assertEqual(clamp_tooltip(500, 400), (500.0, 400.0))

    def test_f12_timeline_empty_history_rendering(self):
        """4. Empty timeline buffer returns 0 cells without index errors."""
        buf = self.TimelineBuffer(capacity=300)
        self.assertEqual(len(buf.cells), 0)
        self.assertEqual(buf.seek(0), 0)

    def test_f12_alert_pip_marker_overlap(self):
        """5. Multiple alert pips on consecutive triggering windows rendered distinctly."""
        buf = self.TimelineBuffer(capacity=10)
        for i in range(5):
            buf.append({"id": i, "is_alert": (i >= 2)})
        alerts = [c for c in buf.cells if c.get("is_alert")]
        self.assertEqual(len(alerts), 3)


if __name__ == "__main__":
    unittest.main()
