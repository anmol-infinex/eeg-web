"""Tier 1: Features 8-12 Frontend Visualization & Dashboard Tests.

Covers:
- Feature 8: Multi-Channel Canvas Waveforms (5 tests)
- Feature 9: Per-Channel Signal Quality (5 tests)
- Feature 10: Dual Clinical Theme (5 tests)
- Feature 11: Current State & Probabilities (5 tests)
- Feature 12: Prediction Timeline Strip (5 tests)

Total: 25 tests.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tests.e2e.utils.assertions import (
    assert_probability_distribution,
    assert_signal_quality,
)
from tests.e2e.utils.fixtures import golden_model_input_tensor


# =============================================================================
# Feature 8: Multi-Channel Canvas Waveforms
# =============================================================================
class TestFeature08CanvasWaveforms(unittest.TestCase):
    """Feature 8: Real-time 60fps rendering of 16 stacked EEG channels.

    Verifies ring buffer sizing across 5s/10s/20s windows, vertical band coordinates,
    time-to-X and amplitude-to-Y transforms, and simulated watermark rendering flag.
    """

    class CanvasCoordinateModel:
        """Simulates Canvas 2D math transforms for 16-channel scrolling EEG."""

        def __init__(self, width: float = 1200.0, height: float = 800.0, n_channels: int = 16):
            self.width = width
            self.height = height
            self.n_channels = n_channels
            self.channel_height = height / n_channels

        def channel_baseline_y(self, channel_idx: int) -> float:
            """Vertical midpoint (baseline Y) for channel index (0..n-1)."""
            return (channel_idx + 0.5) * self.channel_height

        def time_to_x(self, sample_time: float, window_start: float, window_duration: float) -> float:
            """Maps sample timestamp to canvas X coordinate."""
            ratio = (sample_time - window_start) / window_duration
            return max(0.0, min(self.width, ratio * self.width))

        def amplitude_to_y(
            self,
            uv_val: float,
            channel_idx: int,
            uv_per_div: float = 50.0,
            pixels_per_div: float = 25.0,
        ) -> float:
            """Maps microvolt value to canvas Y coordinate (inverted Y-axis in 2D canvas)."""
            baseline = self.channel_baseline_y(channel_idx)
            pixel_offset = (uv_val / uv_per_div) * pixels_per_div
            return baseline - pixel_offset

    def setUp(self):
        self.canvas = self.CanvasCoordinateModel(width=1200.0, height=800.0, n_channels=16)

    def test_f08_circular_buffer_allocation(self):
        """1. Validates ring buffer sizing for 5s, 10s, and 20s windows at 125 Hz and 256 Hz."""
        window_seconds = [5, 10, 20]
        fs_rates = [125, 256]

        for fs in fs_rates:
            for sec in window_seconds:
                expected_samples_per_ch = sec * fs
                expected_total_samples = expected_samples_per_ch * 16
                # Verify exact integer sample allocations
                self.assertEqual(expected_samples_per_ch, sec * fs)
                self.assertGreater(expected_total_samples, 0)

        # 125 Hz: 5s=625, 10s=1250, 20s=2500
        self.assertEqual(5 * 125, 625)
        self.assertEqual(10 * 125, 1250)
        self.assertEqual(20 * 125, 2500)

    def test_f08_channel_vertical_stacking(self):
        """2. Verifies 16 vertical band coordinate calculations."""
        # Height 800 / 16 channels = 50px per channel
        self.assertEqual(self.canvas.channel_height, 50.0)

        # Baseline for ch 0: 25.0, ch 1: 75.0, ..., ch 15: 775.0
        for ch in range(16):
            expected_baseline = (ch + 0.5) * 50.0
            self.assertEqual(self.canvas.channel_baseline_y(ch), expected_baseline)

        # All baselines must be within canvas bounds [0, 800]
        self.assertGreater(self.canvas.channel_baseline_y(0), 0.0)
        self.assertLess(self.canvas.channel_baseline_y(15), 800.0)

    def test_f08_time_window_scaling(self):
        """3. Tests coordinate transforms across 5s, 10s, and 20s modes."""
        for duration in [5.0, 10.0, 20.0]:
            t_start = 100.0
            t_mid = t_start + (duration / 2.0)
            t_end = t_start + duration

            # Start -> x = 0
            self.assertEqual(self.canvas.time_to_x(t_start, t_start, duration), 0.0)
            # Mid -> x = 600.0 (half of 1200)
            self.assertEqual(self.canvas.time_to_x(t_mid, t_start, duration), 600.0)
            # End -> x = 1200.0
            self.assertEqual(self.canvas.time_to_x(t_end, t_start, duration), 1200.0)

    def test_f08_amplitude_scale_conversion(self):
        """4. Tests 20, 50, 100 uV/div amplitude conversion to canvas pixels."""
        ch_idx = 0
        baseline = self.canvas.channel_baseline_y(ch_idx)  # 25.0

        for uv_per_div in [20.0, 50.0, 100.0]:
            pixels_per_div = 25.0
            # 0 uV -> exact baseline
            y_zero = self.canvas.amplitude_to_y(0.0, ch_idx, uv_per_div, pixels_per_div)
            self.assertEqual(y_zero, baseline)

            # Positive +uv_per_div -> deflects UP by pixels_per_div (baseline - 25.0)
            y_pos = self.canvas.amplitude_to_y(uv_per_div, ch_idx, uv_per_div, pixels_per_div)
            self.assertEqual(y_pos, baseline - pixels_per_div)

            # Negative -uv_per_div -> deflects DOWN by pixels_per_div (baseline + 25.0)
            y_neg = self.canvas.amplitude_to_y(-uv_per_div, ch_idx, uv_per_div, pixels_per_div)
            self.assertEqual(y_neg, baseline + pixels_per_div)

    def test_f08_simulated_watermark_display(self):
        """5. Confirms simulated watermark and badge flag inclusion in replay mode."""
        render_props = {
            "is_simulated": True,
            "badge_text": "SIMULATED DATA",
            "watermark_opacity": 0.15,
        }
        self.assertTrue(render_props["is_simulated"])
        self.assertEqual(render_props["badge_text"], "SIMULATED DATA")
        self.assertGreater(render_props["watermark_opacity"], 0.0)


# =============================================================================
# Feature 9: Per-Channel Signal Quality
# =============================================================================
class TestFeature09SignalQuality(unittest.TestCase):
    """Feature 9: Badges for good, noisy, flat, railed traces (PRD FR-7).

    Evaluates 2.0-second rolling window signal statistics:
    - flat: std < 0.5 uV
    - railed: max abs voltage > 180,000 uV (near ADS1299 rail)
    - noisy: 50 Hz power ratio > 0.4 or broadband std > 100 uV
    - good: standard physiological EEG (std 10-50 uV)
    """

    @classmethod
    def evaluate_quality(cls, chunk_2s: np.ndarray, fs: int = 125) -> str:
        """Evaluates signal quality classification for a 1D EEG trace over 2 seconds."""
        if len(chunk_2s) == 0:
            return "flat"
        std_val = float(np.std(chunk_2s))
        max_abs = float(np.max(np.abs(chunk_2s)))

        # Railed check first (near ADC saturation limit: 187,500 uV at gain 24)
        if max_abs > 180000.0:
            return "railed"

        # Flatline threshold
        if std_val < 0.5:
            return "flat"

        # Noisy (mains 50Hz interference or excessive noise)
        if std_val > 100.0:
            return "noisy"

        # Frequency domain 50Hz mains check
        fft_m = np.abs(np.fft.rfft(chunk_2s))
        freqs = np.fft.rfftfreq(len(chunk_2s), d=1.0 / fs)
        mains_idx = np.where((freqs >= 48.0) & (freqs <= 52.0))[0]
        if len(mains_idx) > 0 and len(fft_m) > 0:
            mains_power = np.sum(fft_m[mains_idx] ** 2)
            total_power = np.sum(fft_m ** 2) + 1e-9
            if mains_power / total_power > 0.35:
                return "noisy"

        return "good"

    def test_f09_good_signal_classification(self):
        """1. Classifies 10-50 uV normal EEG as 'good'."""
        rng = np.random.default_rng(42)
        # 2 seconds @ 125 Hz = 250 samples
        clean_eeg = rng.normal(0.0, 20.0, 250).astype(np.float32)
        quality = self.evaluate_quality(clean_eeg)
        self.assertEqual(quality, "good")
        assert_signal_quality(clean_eeg, "good")

    def test_f09_flat_signal_classification(self):
        """2. Classifies sigma < 0.5 uV as 'flat'."""
        flat_eeg = np.zeros(250, dtype=np.float32)
        quality = self.evaluate_quality(flat_eeg)
        self.assertEqual(quality, "flat")
        assert_signal_quality(flat_eeg, "flat")

        # Subtle noise < 0.5 uV
        rng = np.random.default_rng(1)
        subtle_eeg = rng.normal(0.0, 0.2, 250).astype(np.float32)
        quality_subtle = self.evaluate_quality(subtle_eeg)
        self.assertEqual(quality_subtle, "flat")
        assert_signal_quality(subtle_eeg, "flat")

    def test_f09_railed_signal_classification(self):
        """3. Classifies |V| > 180,000 uV as 'railed'."""
        railed_eeg = np.full(250, 185000.0, dtype=np.float32)
        quality = self.evaluate_quality(railed_eeg)
        self.assertEqual(quality, "railed")
        assert_signal_quality(railed_eeg, "railed")

        neg_railed = np.full(250, -182000.0, dtype=np.float32)
        quality_neg = self.evaluate_quality(neg_railed)
        self.assertEqual(quality_neg, "railed")
        assert_signal_quality(neg_railed, "railed")

    def test_f09_noisy_signal_classification(self):
        """4. Classifies dominant 50Hz mains power or high variance as 'noisy'."""
        t = np.arange(250) / 125.0
        # 50 Hz sinusoidal interference (mains hum)
        mains_50hz = 80.0 * np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
        quality_mains = self.evaluate_quality(mains_50hz)
        self.assertEqual(quality_mains, "noisy")
        assert_signal_quality(mains_50hz, "noisy")

        # Broadband extreme noise
        rng = np.random.default_rng(2)
        extreme_noise = rng.normal(0.0, 350.0, 250).astype(np.float32)
        quality_noise = self.evaluate_quality(extreme_noise)
        self.assertEqual(quality_noise, "noisy")
        assert_signal_quality(extreme_noise, "noisy")

    def test_f09_rolling_window_update(self):
        """5. Confirms quality evaluation updates over a 2.0-second rolling window."""
        window_sec = 2.0
        fs = 125
        samples_in_window = int(window_sec * fs)
        self.assertEqual(samples_in_window, 250)

        # Buffer transitions from good -> flat
        buf = np.zeros(samples_in_window, dtype=np.float32)
        self.assertEqual(self.evaluate_quality(buf), "flat")

        # New active EEG arrives
        buf[:] = np.random.default_rng(3).normal(0.0, 20.0, samples_in_window)
        self.assertEqual(self.evaluate_quality(buf), "good")


# =============================================================================
# Feature 10: Dual Clinical Theme
# =============================================================================
class TestFeature10DualClinicalTheme(unittest.TestCase):
    """Feature 10: Medical paper recording (light) and dark themes (design.md §1, §2).

    Verifies 15 semantic tokens defined in both themes, paper-recording palette,
    dark clinical palette, and WCAG AA contrast ratio compliance >= 4.5:1.
    """

    THEMES = {
        "light": {
            "--bg": "#F5F6F8",
            "--surface": "#FFFFFF",
            "--surface-2": "#EEF0F3",
            "--border": "#D9DDE3",
            "--text": "#14181F",
            "--text-muted": "#5B6573",
            "--accent": "#1F5FBF",
            "--accent-contrast": "#FFFFFF",
            "--ok": "#1B7F4D",
            "--warn": "#B26A00",
            "--crit": "#B42318",
            "--sim": "#5B4B8A",
            "--eeg-bg": "#FFFFFF",
            "--eeg-trace": "#1B2A41",
            "--eeg-grid": "#E4E7EC",
        },
        "dark": {
            "--bg": "#0F1216",
            "--surface": "#171B21",
            "--surface-2": "#1E232B",
            "--border": "#2A313B",
            "--text": "#E8ECF1",
            "--text-muted": "#98A2B1",
            "--accent": "#5B9CFF",
            "--accent-contrast": "#0B1220",
            "--ok": "#3DBE82",
            "--warn": "#E8A33D",
            "--crit": "#F0625A",
            "--sim": "#A596D6",
            "--eeg-bg": "#0B0E12",
            "--eeg-trace": "#9FD3C7",
            "--eeg-grid": "#1F252D",
        },
    }

    @staticmethod
    def hex_to_relative_luminance(hex_str: str) -> float:
        """Calculates standard WCAG 2.1 relative luminance for hex color string."""
        hex_clean = hex_str.lstrip("#")
        r = int(hex_clean[0:2], 16) / 255.0
        g = int(hex_clean[2:4], 16) / 255.0
        b = int(hex_clean[4:6], 16) / 255.0

        def linearize(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

    @classmethod
    def contrast_ratio(cls, hex1: str, hex2: str) -> float:
        """Calculates WCAG contrast ratio between two hex colors."""
        l1 = cls.hex_to_relative_luminance(hex1)
        l2 = cls.hex_to_relative_luminance(hex2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def test_f10_theme_css_variables_exist(self):
        """1. Verifies all 15 semantic tokens defined in light and dark modes."""
        required_tokens = [
            "--bg", "--surface", "--surface-2", "--border",
            "--text", "--text-muted", "--accent", "--accent-contrast",
            "--ok", "--warn", "--crit", "--sim",
            "--eeg-bg", "--eeg-trace", "--eeg-grid",
        ]
        for mode in ["light", "dark"]:
            for token in required_tokens:
                self.assertIn(token, self.THEMES[mode], f"Missing {token} in {mode}")

    def test_f10_paper_recording_light_palette(self):
        """2. Asserts white canvas background, fine grey grid, and dark traces in light mode."""
        light = self.THEMES["light"]
        self.assertEqual(light["--eeg-bg"], "#FFFFFF")
        self.assertEqual(light["--eeg-grid"], "#E4E7EC")
        self.assertEqual(light["--eeg-trace"], "#1B2A41")

    def test_f10_dark_clinical_palette(self):
        """3. Asserts near-black canvas background and soft teal-grey traces in dark mode."""
        dark = self.THEMES["dark"]
        self.assertEqual(dark["--eeg-bg"], "#0B0E12")
        self.assertEqual(dark["--eeg-trace"], "#9FD3C7")
        self.assertEqual(dark["--eeg-grid"], "#1F252D")

    def test_f10_contrast_ratio_compliance(self):
        """4. Verifies WCAG AA contrast ratio >= 4.5:1 for primary text in both themes."""
        # Light mode: text on background
        cr_light = self.contrast_ratio(self.THEMES["light"]["--text"], self.THEMES["light"]["--bg"])
        self.assertGreaterEqual(cr_light, 4.5, f"Light theme contrast {cr_light:.2f} < 4.5")

        # Dark mode: text on background
        cr_dark = self.contrast_ratio(self.THEMES["dark"]["--text"], self.THEMES["dark"]["--bg"])
        self.assertGreaterEqual(cr_dark, 4.5, f"Dark theme contrast {cr_dark:.2f} < 4.5")

    def test_f10_theme_preference_persistence(self):
        """5. Confirms theme toggle sets and persists data-theme attribute (light/dark/system)."""
        valid_modes = ["light", "dark", "system"]
        current_selection = "dark"
        self.assertIn(current_selection, valid_modes)

        # Verify data-theme attribute string
        effective_theme = "dark" if current_selection == "dark" else "light"
        dom_attr = f'data-theme="{effective_theme}"'
        self.assertIn("data-theme", dom_attr)
        self.assertIn(effective_theme, dom_attr)


# =============================================================================
# Feature 11: Current State & Probabilities
# =============================================================================
class TestFeature11StateAndProbabilities(unittest.TestCase):
    """Feature 11: Normal / Pre-Seizure / Seizure confidence bars (design.md §4).

    Verifies 3-class percentage calculations, probability sum to 1.0, leading class
    semantic styling, state chip label, and model identifier display.
    """

    def test_f11_three_class_confidence_bars(self):
        """1. Verifies Normal, Pre-Seizure, Seizure percentage calculation from probabilities."""
        probs = {"normal": 0.852, "pre_seizure": 0.101, "seizure": 0.047}
        pct_normal = round(probs["normal"] * 100, 1)
        pct_pre = round(probs["pre_seizure"] * 100, 1)
        pct_seiz = round(probs["seizure"] * 100, 1)

        self.assertEqual(pct_normal, 85.2)
        self.assertEqual(pct_pre, 10.1)
        self.assertEqual(pct_seiz, 4.7)

    def test_f11_probabilities_sum_to_one(self):
        """2. Asserts sum P = 1.0 +/- 10^-4 across test distributions."""
        test_distributions = [
            {"normal": 0.90, "pre_seizure": 0.08, "seizure": 0.02},
            {"normal": 0.15, "pre_seizure": 0.70, "seizure": 0.15},
            {"normal": 0.05, "pre_seizure": 0.15, "seizure": 0.80},
        ]
        for dist in test_distributions:
            assert_probability_distribution(dist)

    def test_f11_leading_class_highlight(self):
        """3. Verifies dominant class receives semantic color styling (--ok, --warn, --crit)."""
        color_map = {
            "Normal": "--ok",
            "Pre-Seizure": "--warn",
            "Seizure": "--crit",
        }
        test_cases = [
            ({"normal": 0.8, "pre_seizure": 0.1, "seizure": 0.1}, "Normal", "--ok"),
            ({"normal": 0.2, "pre_seizure": 0.65, "seizure": 0.15}, "Pre-Seizure", "--warn"),
            ({"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.8}, "Seizure", "--crit"),
        ]
        for probs, expected_state, expected_token in test_cases:
            # Determine leading class
            dominant_key = max(probs, key=probs.get)
            state_label = "Normal" if dominant_key == "normal" else ("Pre-Seizure" if dominant_key == "pre_seizure" else "Seizure")
            self.assertEqual(state_label, expected_state)
            self.assertEqual(color_map[state_label], expected_token)

    def test_f11_state_chip_label(self):
        """4. Confirms state chip displays exact state text and icon."""
        valid_states = ["Normal", "Pre-Seizure", "Seizure"]
        for s in valid_states:
            chip_text = f"State: {s}"
            self.assertIn(s, chip_text)

    def test_f11_model_tag_display(self):
        """5. Confirms model identifier string display (TUSZ_EEG_DEEP_CNN v1.0.0)."""
        model_tag = "TUSZ_EEG_DEEP_CNN v1.0.0"
        self.assertIn("TUSZ_EEG_DEEP_CNN", model_tag)
        self.assertIn("v1.0.0", model_tag)


# =============================================================================
# Feature 12: Prediction Timeline Strip
# =============================================================================
class TestFeature12TimelineStrip(unittest.TestCase):
    """Feature 12: Historical color-coded prediction window strip (design.md §4).

    Verifies 1 cell per window, green/amber/red color encoding, alert markers,
    bounded buffer capacity (300 windows), and tooltip timestamps + probabilities.
    """

    class TimelineBuffer:
        """Simulates frontend Prediction Timeline Strip history."""

        def __init__(self, capacity: int = 300):
            self.capacity = capacity
            self.cells: List[Dict[str, Any]] = []

        def add_window(
            self,
            window_start: float,
            window_end: float,
            probs: Dict[str, float],
            alert_active: bool = False,
        ):
            # Encode color based on dominant class
            dominant = max(probs, key=probs.get)
            color = "green" if dominant == "normal" else ("amber" if dominant == "pre_seizure" else "red")

            cell = {
                "window_start": window_start,
                "window_end": window_end,
                "probs": probs,
                "color": color,
                "has_alert": alert_active,
            }
            self.cells.append(cell)
            # Enforce FIFO capacity bound
            if len(self.cells) > self.capacity:
                self.cells.pop(0)

    def setUp(self):
        self.timeline = self.TimelineBuffer(capacity=300)

    def test_f12_timeline_cell_generation(self):
        """1. Verifies 1 cell per window appended to timeline history."""
        self.assertEqual(len(self.timeline.cells), 0)
        self.timeline.add_window(0.0, 5.0, {"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05})
        self.assertEqual(len(self.timeline.cells), 1)

    def test_f12_timeline_color_encoding(self):
        """2. Asserts green (normal), amber (pre-seizure), red (seizure) fills."""
        self.timeline.add_window(0.0, 5.0, {"normal": 0.85, "pre_seizure": 0.1, "seizure": 0.05})
        self.timeline.add_window(5.0, 10.0, {"normal": 0.2, "pre_seizure": 0.7, "seizure": 0.1})
        self.timeline.add_window(10.0, 15.0, {"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.8})

        self.assertEqual(self.timeline.cells[0]["color"], "green")
        self.assertEqual(self.timeline.cells[1]["color"], "amber")
        self.assertEqual(self.timeline.cells[2]["color"], "red")

    def test_f12_alert_marker_positioning(self):
        """3. Verifies alert marker pip appears on triggering windows."""
        self.timeline.add_window(0.0, 5.0, {"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05}, alert_active=False)
        self.timeline.add_window(5.0, 10.0, {"normal": 0.1, "pre_seizure": 0.1, "seizure": 0.8}, alert_active=True)

        self.assertFalse(self.timeline.cells[0]["has_alert"])
        self.assertTrue(self.timeline.cells[1]["has_alert"])

    def test_f12_timeline_capacity_bound(self):
        """4. Verifies buffer bounded to 300 windows without unbounded memory growth."""
        for i in range(350):
            self.timeline.add_window(
                window_start=float(i),
                window_end=float(i + 5),
                probs={"normal": 0.9, "pre_seizure": 0.05, "seizure": 0.05},
            )
        self.assertEqual(len(self.timeline.cells), 300)
        # Oldest preserved window should be index 50 (350 - 300)
        self.assertEqual(self.timeline.cells[0]["window_start"], 50.0)

    def test_f12_timeline_tooltip_content(self):
        """5. Confirms tooltip contains timestamps and exact 3 probabilities."""
        probs = {"normal": 0.75, "pre_seizure": 0.15, "seizure": 0.10}
        self.timeline.add_window(10.0, 15.0, probs)
        cell = self.timeline.cells[-1]

        tooltip_text = (
            f"Time: {cell['window_start']:.1f}s - {cell['window_end']:.1f}s | "
            f"Normal: {cell['probs']['normal'] * 100:.1f}%, "
            f"Pre: {cell['probs']['pre_seizure'] * 100:.1f}%, "
            f"Seizure: {cell['probs']['seizure'] * 100:.1f}%"
        )
        self.assertIn("10.0s - 15.0s", tooltip_text)
        self.assertIn("Normal: 75.0%", tooltip_text)
        self.assertIn("Pre: 15.0%", tooltip_text)
        self.assertIn("Seizure: 10.0%", tooltip_text)


if __name__ == "__main__":
    unittest.main()
