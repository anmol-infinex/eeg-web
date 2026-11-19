"""
backend/app/adapter.py
Signal processing adapter: polyphase resampling, channel remapping, and streaming SOS filtering.
"""
from typing import List, Optional, Tuple
import logging
import numpy as np
import scipy.signal as signal

logger = logging.getLogger("seizure_sentry.adapter")

# Canonical 22 TUSZ TCP Montage Channels
TCP_22_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "FZ", "CZ", "PZ", "A1", "A2", "EKG"
]

# Standard OpenBCI Cyton + Daisy 16 Channel Montage
CYTON_16_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8"
]

# Aliases for 10-20 electrode naming conventions
CHANNEL_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

class PolyphaseResampler:
    """
    Polyphase resampler using scipy.signal.resample_poly.
    Resamples from 125 Hz to 256 Hz (up=256, down=125).
    """
    def __init__(self, from_fs: int = 125, to_fs: int = 256):
        self.from_fs = from_fs
        self.to_fs = to_fs
        # gcd(256, 125) = 1
        self.up = to_fs
        self.down = from_fs

    def resample(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Resample input data along specified time axis.
        A 625-sample array at 125 Hz resamples to exactly 1280 samples at 256 Hz.
        """
        if self.from_fs == self.to_fs:
            return data.astype(np.float32, copy=False)
        
        resampled = signal.resample_poly(data, self.up, self.down, axis=axis)
        return resampled.astype(np.float32)

class ChannelRemapper:
    """
    Remaps source channels (e.g. 16 Cyton) to the 22 canonical TUSZ TCP montage channels.
    - Exact matches copied directly.
    - Midline channels (FZ, CZ, PZ) interpolated from bilateral pairs.
    - Unconnected/reference channels (A1, A2, EKG) set to neutral zero-reference.
    """
    def __init__(self, target_channels: Optional[List[str]] = None):
        self.target_channels = [c.upper() for c in (target_channels or TCP_22_CHANNELS)]

    def remap(self, data: np.ndarray, source_channels: Optional[List[str]] = None) -> np.ndarray:
        """
        data: (N_samples, N_source_channels)
        source_channels: list of string labels for each column in data.
        Returns: (N_samples, 22) float32 array matching self.target_channels.
        """
        n_samples = data.shape[0]
        n_targets = len(self.target_channels)
        output = np.zeros((n_samples, n_targets), dtype=np.float32)

        if source_channels is None:
            if data.shape[1] == 16:
                source_channels = CYTON_16_CHANNELS
            elif data.shape[1] == 22:
                return data.astype(np.float32)
            else:
                raise ValueError(f"Ambiguous channel configuration for input with {data.shape[1]} channels.")

        # Normalize source channel names
        clean_sources = [CHANNEL_ALIASES.get(ch.upper(), ch.upper()) for ch in source_channels]
        source_map = {ch: idx for idx, ch in enumerate(clean_sources)}

        for target_idx, target_ch in enumerate(self.target_channels):
            if target_ch in source_map:
                output[:, target_idx] = data[:, source_map[target_ch]]
            elif target_ch == "FZ" and "F3" in source_map and "F4" in source_map:
                # Bilateral interpolation for Frontal midline
                output[:, target_idx] = 0.5 * (data[:, source_map["F3"]] + data[:, source_map["F4"]])
            elif target_ch == "CZ" and "C3" in source_map and "C4" in source_map:
                # Bilateral interpolation for Central midline
                output[:, target_idx] = 0.5 * (data[:, source_map["C3"]] + data[:, source_map["C4"]])
            elif target_ch == "PZ" and "P3" in source_map and "P4" in source_map:
                # Bilateral interpolation for Parietal midline
                output[:, target_idx] = 0.5 * (data[:, source_map["P3"]] + data[:, source_map["P4"]])
            else:
                # Neutral zero-reference for reference/EKG channels
                output[:, target_idx] = 0.0

        return output

class StreamingFilter:
    """
    Causal streaming filter with persistent state across chunks.
    Combines:
    - 2nd-order Butterworth bandpass (0.5 - 50.0 Hz)
    - 50 Hz IIR notch filter (Q=30.0)
    Implemented using Second-Order Sections (SOS) for numerical stability.
    """
    def __init__(
        self,
        fs: int = 256,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        notch_freq: float = 50.0,
        notch_q: float = 30.0,
        n_channels: int = 22,
    ):
        self.fs = fs
        self.n_channels = n_channels

        # Bandpass SOS
        sos_bp = signal.butter(2, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
        
        # Notch SOS
        b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs=fs)
        sos_notch = signal.tf2sos(b_notch, a_notch)

        # Combined SOS matrix (3 sections, 6 coefficients each)
        self.sos = np.vstack([sos_bp, sos_notch])
        
        # Initialize persistent filter state zi to zeros: shape (n_sections, 2, n_channels)
        # Zero-initialization prevents DC baseline offset transient on AC EEG signals.
        self._zi_base = np.zeros((self.sos.shape[0], 2), dtype=np.float32)
        self.zi = np.zeros((self.sos.shape[0], 2, self.n_channels), dtype=np.float32)

    def filter(self, chunk: np.ndarray) -> np.ndarray:
        """
        Filter an incoming chunk of shape (N, n_channels) updating persistent state self.zi.
        Ensures zero chunk-boundary transient distortion.
        """
        if chunk.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {chunk.shape[1]}")

        filtered_chunk, self.zi = signal.sosfilt(self.sos, chunk, zi=self.zi, axis=0)
        return filtered_chunk.astype(np.float32)

    def reset(self) -> None:
        """Reset internal filter state to zeros."""
        self.zi = np.zeros((self.sos.shape[0], 2, self.n_channels), dtype=np.float32)

class EegAdapter:
    """
    End-to-end EEG adapter coordinating resampling, remapping, and filtering.
    """
    def __init__(self, from_fs: int = 125, to_fs: int = 256, n_channels: int = 22):
        self.resampler = PolyphaseResampler(from_fs=from_fs, to_fs=to_fs)
        self.remapper = ChannelRemapper()
        self.filter = StreamingFilter(fs=to_fs, n_channels=n_channels)

    def process_window(
        self,
        raw_data: np.ndarray,
        source_fs: int = 125,
        source_channels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Process a full EEG window into the required (1280, 22) float32 format.
        """
        # Step 1: Resample if source sample rate differs from 256 Hz
        if source_fs != 256:
            resampled = self.resampler.resample(raw_data, axis=0)
        else:
            resampled = raw_data

        # Step 2: Remap channels to canonical 22 TCP montage
        remapped = self.remapper.remap(resampled, source_channels=source_channels)

        # Step 3: Apply causal streaming bandpass + notch filter
        filtered = self.filter.filter(remapped)

        # Ensure exact 1280 samples length
        if filtered.shape[0] > 1280:
            filtered = filtered[-1280:, :]
        elif filtered.shape[0] < 1280:
            pad = np.zeros((1280 - filtered.shape[0], 22), dtype=np.float32)
            filtered = np.vstack([pad, filtered])

        return filtered.astype(np.float32)
