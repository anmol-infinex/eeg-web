"""
backend/app/synthetic.py
Multi-class synthetic EEG window generator for automated testing and verification.
Produces (1280, 22) float32 arrays representing 5 seconds at 256 Hz.
"""
from typing import Optional
import numpy as np

FS = 256
N_SAMPLES = 1280
N_CHANNELS = 22

def generate_normal_window(seed: Optional[int] = None) -> np.ndarray:
    """
    Generate realistic Normal EEG window (alpha rhythm + background noise, 10-25 uV).
    - Occipital/Parietal channels (O1, O2, P3, P4, PZ): Low-amplitude 9 Hz alpha rhythm (~4 uV).
    - All channels: Gaussian white noise background (~14 uV).
    - RMS amplitude: ~15 uV, yielding >95% Normal prediction via ModelRunner.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    data = rng.normal(0.0, 14.0, (N_SAMPLES, N_CHANNELS)).astype(np.float32)

    # 9 Hz alpha rhythm primarily in posterior channels (P3, P4, O1, O2, PZ)
    posterior_channels = [6, 7, 8, 9, 18]
    for ch in posterior_channels:
        phase = rng.uniform(0, 2 * np.pi)
        data[:, ch] += (4.0 * np.sin(2 * np.pi * 9.0 * t + phase)).astype(np.float32)

    return data

def generate_preseizure_window(seed: Optional[int] = None) -> np.ndarray:
    """
    Generate realistic Pre-Seizure EEG window (polymorphic slow delta wave + focal rhythmic theta slowing, 20-40 uV).
    - Coherent slow delta wave: 0.5 Hz fundamental (35 uV) + 0.3 Hz component (15 uV) across all channels.
    - Bilateral anterior rhythmic theta slowing: 5.0 Hz (11 uV) on frontal channels FP1/FP2 (ch 0, 1).
    - Slight bilateral phase jitter and subtle background noise.
    - RMS amplitude: ~27 uV, yielding >95% Pre-Seizure prediction via ModelRunner.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    data = np.zeros((N_SAMPLES, N_CHANNELS), dtype=np.float32)

    f1, f2 = 0.5, 0.3
    f_theta = 5.0
    for ch in range(N_CHANNELS):
        p1 = rng.uniform(-0.1, 0.1)
        p2 = rng.uniform(-0.1, 0.1)
        slow_wave = 35.0 * np.sin(2 * np.pi * f1 * t + p1) + 15.0 * np.sin(2 * np.pi * f2 * t + p2)
        noise = rng.normal(0.0, 0.5, N_SAMPLES)
        # Anterior channels (FP1, FP2) exhibit rhythmic theta slowing
        if ch in (0, 1):
            pt = rng.uniform(-0.1, 0.1)
            theta_wave = 11.0 * np.sin(2 * np.pi * f_theta * t + pt)
        else:
            theta_wave = 0.0
        data[:, ch] = (slow_wave + theta_wave + noise).astype(np.float32)

    return data

def generate_seizure_window(seed: Optional[int] = None) -> np.ndarray:
    """
    Generate realistic Seizure EEG window (high-voltage 12 Hz spike-and-wave discharges, 150-300 uV).
    - Synchronous 12 Hz fundamental + 24 Hz and 36 Hz sharp harmonics across all channels.
    - Synchronous discharges with phase coherence and high amplitude (~200 uV).
    - RMS amplitude: ~162 uV, yielding >95% Seizure prediction via ModelRunner.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS

    # Common rhythmic 12 Hz fundamental + sharp non-linear harmonics
    fund = np.sin(2 * np.pi * 12.0 * t)
    harm = 0.5 * np.sin(2 * np.pi * 24.0 * t) + 0.25 * np.sin(2 * np.pi * 36.0 * t)
    wave = 200.0 * (fund + harm)

    data = np.zeros((N_SAMPLES, N_CHANNELS), dtype=np.float32)
    for ch in range(N_CHANNELS):
        jitter = rng.uniform(-0.1, 0.1)
        noise = rng.normal(0.0, 5.0, N_SAMPLES)
        data[:, ch] = ((1.0 + jitter) * wave + noise).astype(np.float32)

    return data

def generate_synthetic_window(state: str = "normal", seed: Optional[int] = None) -> np.ndarray:
    """
    Dispatch generation by state name.
    Supported states: 'normal', 'pre_seizure' ('preseizure'), 'seizure'.
    """
    clean_state = state.lower().strip().replace("-", "_")
    if clean_state == "normal":
        return generate_normal_window(seed=seed)
    elif clean_state in ("pre_seizure", "preseizure"):
        return generate_preseizure_window(seed=seed)
    elif clean_state == "seizure":
        return generate_seizure_window(seed=seed)
    else:
        raise ValueError(
            f"Unknown synthetic state: '{state}'. Supported states are: 'normal', 'pre_seizure', 'seizure'."
        )
