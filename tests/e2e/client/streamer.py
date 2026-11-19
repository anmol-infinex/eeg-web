"""EEG Data Streamer and Synthetic Signal Generator.

Produces 33-byte OpenBCI Cyton serial packets, 16-byte little-endian WebSocket
binary frames, and physiologically authentic multi-channel synthetic EEG signals
(Normal, Pre-Seizure, Seizure, and Artifact states) matching Seizure-Sentry specs.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from tests.e2e.config import CONFIG


class EegStreamer:
    """Zero-dependency EEG packet builder, binary frame encoder, and synthetic signal generator."""

    # OpenBCI hardware constants
    START_BYTE = 0xA0
    STOP_BYTES = (0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6)
    DEFAULT_STOP_BYTE = 0xC0
    UV_PER_COUNT = 0.02235174  # 4.5 / 24 / (2^23 - 1) * 1e6

    # WS Binary Header: uint32 seq, float64 t0_ms, uint16 n_samples, uint8 n_channels, uint8 flags
    WS_HEADER_FMT = "<IdHBB"
    WS_HEADER_SIZE = 16

    # -------------------------------------------------------------------------
    # 1. OpenBCI 33-Byte Serial Packet Encoding & Decoding
    # -------------------------------------------------------------------------

    @classmethod
    def uv_to_24bit_signed(cls, uv_val: float) -> bytes:
        """Converts microvolts to 3-byte big-endian signed 24-bit integer."""
        count = int(round(uv_val / cls.UV_PER_COUNT))
        # Clamp to 24-bit signed integer range: [-8388608, 8388607]
        count = max(-8388608, min(8388607, count))
        if count < 0:
            count += 1 << 24
        return struct.pack(">I", count)[1:]

    @classmethod
    def signed_24bit_to_uv(cls, raw3: bytes) -> float:
        """Converts 3-byte big-endian signed 24-bit integer to microvolts."""
        if len(raw3) != 3:
            raise ValueError(f"Expected 3 bytes, got {len(raw3)}")
        val = (raw3[0] << 16) | (raw3[1] << 8) | raw3[2]
        if val & 0x800000:
            val -= 1 << 24
        return float(val) * cls.UV_PER_COUNT

    @classmethod
    def pack_openbci_packet(
        cls,
        sample_counter: int,
        channels_uv: Union[List[float], np.ndarray],
        aux_values: Optional[List[int]] = None,
        stop_byte: int = DEFAULT_STOP_BYTE,
    ) -> bytes:
        """Packs an 8-channel sample into a 33-byte OpenBCI packet.

        Format:
          - Byte 0: 0xA0 (header)
          - Byte 1: sample_counter % 256
          - Bytes 2-25: 8 channels * 3 bytes (24-bit big-endian signed)
          - Bytes 26-31: 3 * 16-bit big-endian accelerometer/aux ints
          - Byte 32: stop_byte (0xC0-0xC6)
        """
        if len(channels_uv) != 8:
            raise ValueError(f"OpenBCI packet requires exactly 8 channels, got {len(channels_uv)}")
        if aux_values is None:
            aux_values = [0, 0, 0]
        elif len(aux_values) != 3:
            raise ValueError(f"OpenBCI aux data requires exactly 3 int16s, got {len(aux_values)}")

        packet = bytearray()
        packet.append(cls.START_BYTE)
        packet.append(sample_counter % 256)

        for ch in channels_uv:
            packet.extend(cls.uv_to_24bit_signed(float(ch)))

        for aux in aux_values:
            packet.extend(struct.pack(">h", int(aux)))

        packet.append(stop_byte)
        return bytes(packet)

    @classmethod
    def unpack_openbci_packet(cls, packet: bytes) -> Dict[str, Any]:
        """Unpacks and validates a 33-byte OpenBCI packet."""
        if len(packet) != 33:
            raise ValueError(f"Invalid packet size: expected 33 bytes, got {len(packet)}")
        if packet[0] != cls.START_BYTE:
            raise ValueError(f"Invalid start byte: expected 0x{cls.START_BYTE:02x}, got 0x{packet[0]:02x}")
        if packet[32] not in cls.STOP_BYTES:
            raise ValueError(f"Invalid stop byte: got 0x{packet[32]:02x}")

        sample_counter = packet[1]
        channels_uv = []
        for i in range(8):
            offset = 2 + (i * 3)
            raw3 = packet[offset : offset + 3]
            channels_uv.append(cls.signed_24bit_to_uv(raw3))

        aux_values = list(struct.unpack(">hhh", packet[26:32]))
        stop_byte = packet[32]

        return {
            "counter": sample_counter,
            "channels_uv": channels_uv,
            "aux": aux_values,
            "stop_byte": stop_byte,
        }

    @classmethod
    def pack_daisy_pair(
        cls,
        sample_counter_pair_start: int,
        channels_16_uv: Union[List[float], np.ndarray],
        aux1: Optional[List[int]] = None,
        aux2: Optional[List[int]] = None,
    ) -> Tuple[bytes, bytes]:
        """Packs a 16-channel Cyton+Daisy sample into alternating odd/even 33-byte packets."""
        if len(channels_16_uv) != 16:
            raise ValueError(f"Daisy pair requires 16 channels, got {len(channels_16_uv)}")
        # Packet 1 (Board: channels 1-8)
        pkt1 = cls.pack_openbci_packet(
            sample_counter_pair_start, channels_16_uv[:8], aux_values=aux1
        )
        # Packet 2 (Daisy: channels 9-16)
        pkt2 = cls.pack_openbci_packet(
            sample_counter_pair_start + 1, channels_16_uv[8:], aux_values=aux2
        )
        return pkt1, pkt2

    # -------------------------------------------------------------------------
    # 2. WebSocket 16-Byte Binary Framing
    # -------------------------------------------------------------------------

    @classmethod
    def pack_ws_binary_frame(
        cls,
        seq: int,
        t0_ms: float,
        data: Union[np.ndarray, List[List[float]]],
        flags: int = 0,
    ) -> bytes:
        """Encodes an EEG chunk into a 16-byte header + float32 little-endian payload frame.

        Header:
          uint32 seq (4)
          float64 t0_ms (8)
          uint16 n_samples (2)
          uint8 n_channels (1)
          uint8 flags (1)  # bit 0: gap flag
        Payload:
          n_samples * n_channels of float32 microvolts in sample-major order.
        """
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"EEG data must be 2D (n_samples, n_channels), got shape {arr.shape}")

        n_samples, n_channels = arr.shape
        header = struct.pack(cls.WS_HEADER_FMT, seq, float(t0_ms), n_samples, n_channels, flags)
        payload = arr.tobytes(order="C")  # Little-endian float32 sample-major
        return header + payload

    @classmethod
    def unpack_ws_binary_frame(cls, frame_bytes: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
        """Decodes a WebSocket binary frame into header metadata and a 2D float32 numpy array."""
        if len(frame_bytes) < cls.WS_HEADER_SIZE:
            raise ValueError(
                f"Binary frame too short: have {len(frame_bytes)} bytes, need at least {cls.WS_HEADER_SIZE}"
            )

        header_bytes = frame_bytes[: cls.WS_HEADER_SIZE]
        seq, t0_ms, n_samples, n_channels, flags = struct.unpack(cls.WS_HEADER_FMT, header_bytes)

        expected_payload_len = n_samples * n_channels * 4
        actual_payload_len = len(frame_bytes) - cls.WS_HEADER_SIZE
        if actual_payload_len != expected_payload_len:
            raise ValueError(
                f"Payload size mismatch: expected {expected_payload_len} bytes "
                f"({n_samples}x{n_channels}x4), got {actual_payload_len}"
            )

        payload_bytes = frame_bytes[cls.WS_HEADER_SIZE :]
        data = np.frombuffer(payload_bytes, dtype=np.float32).reshape((n_samples, n_channels))

        header = {
            "seq": seq,
            "t0_ms": t0_ms,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "flags": flags,
            "is_gap": bool(flags & 0x01),
        }
        return header, data

    # -------------------------------------------------------------------------
    # 3. Synthetic EEG Signal Generation
    # -------------------------------------------------------------------------

    @classmethod
    def generate_signal(
        cls,
        state: str = "normal",
        duration_sec: float = 5.0,
        fs: int = 125,
        n_channels: int = 16,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generates realistic synthetic EEG signals for testing ML, alerts, and visualization.

        Supported states:
          - "normal": Physiological alpha/beta/gamma sinusoids + pink noise (std 15-40 uV)
          - "pre_seizure": Theta (4-7 Hz) rhythmic slowing, delta drift, attenuation (std 30-70 uV)
          - "seizure": High-amplitude 3-5 Hz spike-wave discharges + fast 12-14 Hz polyspikes (std 80-200 uV)
          - "flat" / "flatline": Zero-variance DC line with std < 0.3 uV
          - "railed": Saturated amplifier voltage (> 185,000 uV)
          - "noisy": Dominant 50 Hz mains line power sinusoid (500 uV) + broadband noise
        """
        rng = np.random.default_rng(seed)
        n_samples = max(1, int(round(duration_sec * fs)))
        t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

        data = np.zeros((n_samples, n_channels), dtype=np.float32)
        state_lower = state.lower().strip()

        if state_lower in ("normal", "baseline"):
            for ch in range(n_channels):
                # Channel-specific phase offsets
                phase = rng.uniform(0, 2 * math.pi)
                # Alpha rhythm (8-12 Hz) prominent in posterior channels
                alpha_freq = rng.uniform(9.0, 11.0)
                alpha_amp = rng.uniform(15.0, 30.0) if ch >= (n_channels // 2) else rng.uniform(8.0, 15.0)
                # Beta rhythm (13-25 Hz) prominent in frontal channels
                beta_freq = rng.uniform(16.0, 22.0)
                beta_amp = rng.uniform(10.0, 20.0) if ch < (n_channels // 2) else rng.uniform(5.0, 10.0)
                # Background white/pink noise
                noise = rng.normal(0.0, 4.0, n_samples).astype(np.float32)
                data[:, ch] = (
                    alpha_amp * np.sin(2 * math.pi * alpha_freq * t + phase)
                    + beta_amp * np.sin(2 * math.pi * beta_freq * t + phase * 1.5)
                    + noise
                )

        elif state_lower in ("pre_seizure", "preseizure", "warning"):
            for ch in range(n_channels):
                phase = rng.uniform(0, 2 * math.pi)
                # Elevated rhythmic theta (4-6 Hz)
                theta_freq = rng.uniform(4.5, 6.0)
                theta_amp = rng.uniform(35.0, 60.0)
                # Delta slow wave component (1.5-3.0 Hz)
                delta_freq = rng.uniform(1.5, 2.5)
                delta_amp = rng.uniform(20.0, 40.0)
                noise = rng.normal(0.0, 5.0, n_samples).astype(np.float32)
                data[:, ch] = (
                    theta_amp * np.sin(2 * math.pi * theta_freq * t + phase)
                    + delta_amp * np.sin(2 * math.pi * delta_freq * t + phase * 0.7)
                    + noise
                )

        elif state_lower in ("seizure", "ictal", "critical"):
            for ch in range(n_channels):
                # High-amplitude 3.0 Hz spike-wave discharges
                spike_wave_freq = 3.0
                spike_wave = np.zeros(n_samples, dtype=np.float32)
                # Construct asymmetric 3Hz spike + wave profile
                t_cycle = (t * spike_wave_freq) % 1.0
                # Spike: sharp positive deflection in first 15% of cycle
                spike_mask = t_cycle < 0.15
                spike_wave[spike_mask] = 180.0 * np.sin(math.pi * t_cycle[spike_mask] / 0.15)
                # Wave: broad negative slow wave in remaining 85% of cycle
                wave_mask = t_cycle >= 0.15
                spike_wave[wave_mask] = -120.0 * np.sin(math.pi * (t_cycle[wave_mask] - 0.15) / 0.85)

                # Superimposed 13 Hz fast polyspikes
                fast_discharge = 45.0 * np.sin(2 * math.pi * 13.5 * t + rng.uniform(0, 1.0))
                noise = rng.normal(0.0, 6.0, n_samples).astype(np.float32)

                data[:, ch] = spike_wave + fast_discharge + noise

        elif state_lower in ("flat", "flatline"):
            # DC flatline with micro-noise std < 0.3 uV
            data = rng.normal(0.0, 0.1, (n_samples, n_channels)).astype(np.float32)

        elif state_lower in ("railed", "rail"):
            # Saturated ADC voltage near 187,500 uV
            data = np.full((n_samples, n_channels), 195000.0, dtype=np.float32)
            data += rng.normal(0.0, 50.0, (n_samples, n_channels)).astype(np.float32)

        elif state_lower in ("noisy", "noise", "50hz"):
            # Dominant 50 Hz power mains noise
            for ch in range(n_channels):
                phase = rng.uniform(0, 2 * math.pi)
                data[:, ch] = (
                    600.0 * np.sin(2 * math.pi * 50.0 * t + phase)
                    + rng.normal(0.0, 30.0, n_samples).astype(np.float32)
                )

        else:
            raise ValueError(f"Unknown synthetic EEG state: {state}")

        return data

    @classmethod
    def generate_model_window(
        cls,
        state: str = "normal",
        fs: int = 256,
        n_channels: int = 22,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generates exactly 1280 samples x 22 channels (5.0s @ 256Hz) matching tusz_3class_eeg_model.h5."""
        from pathlib import Path
        state_lower = state.lower().strip().replace("-", "_")
        rng = np.random.default_rng(seed)
        t = np.arange(1280, dtype=np.float32) / float(fs)

        if state_lower in ("normal", "baseline"):
            data = np.zeros((1280, n_channels), dtype=np.float32)
            for ch in range(n_channels):
                data[:, ch] = (
                    8.0 * np.sin(2 * np.pi * 8.5 * t)
                    + 10.0 * np.sin(2 * np.pi * 1.5 * t)
                    + rng.normal(0.0, 15.0, 1280).astype(np.float32)
                )
            return data

        elif state_lower in ("pre_seizure", "preseizure", "warning"):
            template_path = Path(__file__).resolve().parent.parent / "utils" / "pre_seizure_template.npy"
            if template_path.is_file():
                base_tmpl = np.load(template_path).astype(np.float32)
                jitter = rng.normal(0.0, 0.1, base_tmpl.shape).astype(np.float32)
                return (base_tmpl + jitter)[:, :n_channels]
            sig = cls.generate_signal(state=state, duration_sec=5.0, fs=fs, n_channels=n_channels, seed=seed)
            return sig[:1280, :n_channels]

        elif state_lower in ("seizure", "ictal", "critical"):
            data = np.zeros((1280, n_channels), dtype=np.float32)
            for ch in range(n_channels):
                data[:, ch] = (
                    60.0 * np.sin(2 * np.pi * 4.5 * t)
                    + 40.0 * np.sin(2 * np.pi * 14.0 * t)
                    + rng.normal(0.0, 5.0, 1280).astype(np.float32)
                )
            return data

        sig = cls.generate_signal(state=state, duration_sec=5.0, fs=fs, n_channels=n_channels, seed=seed)
        if sig.shape[0] != 1280 or sig.shape[1] != n_channels:
            sig = sig[:1280, :n_channels]
        return sig

    @classmethod
    def generate_stream_frames(
        cls,
        state: str = "normal",
        total_duration_sec: float = 1.0,
        chunk_sec: float = 0.248,  # 31 samples @ 125Hz
        fs: int = 125,
        n_channels: int = 16,
        start_seq: int = 0,
        start_t0_ms: Optional[float] = None,
    ) -> List[bytes]:
        """Generates a list of consecutive 16-byte header + float32 WS binary frames."""
        samples_per_chunk = int(round(chunk_sec * fs))
        total_samples = int(round(total_duration_sec * fs))
        full_signal = cls.generate_signal(
            state=state, duration_sec=total_duration_sec, fs=fs, n_channels=n_channels
        )

        base_t0 = start_t0_ms if start_t0_ms is not None else (time.time() * 1000.0)
        frames = []
        seq = start_seq

        for offset in range(0, total_samples, samples_per_chunk):
            chunk = full_signal[offset : offset + samples_per_chunk]
            if len(chunk) < samples_per_chunk:
                # Pad last chunk if incomplete
                pad_len = samples_per_chunk - len(chunk)
                pad = np.zeros((pad_len, n_channels), dtype=np.float32)
                chunk = np.vstack([chunk, pad])

            chunk_t0 = base_t0 + (offset / fs * 1000.0)
            frame_bytes = cls.pack_ws_binary_frame(seq, chunk_t0, chunk)
            frames.append(frame_bytes)
            seq += 1

        return frames
