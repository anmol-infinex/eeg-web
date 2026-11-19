"""
backend/tests/test_adapter.py
Unit tests for PolyphaseResampler, ChannelRemapper, and StreamingFilter.
"""
import numpy as np
from app.adapter import PolyphaseResampler, ChannelRemapper, StreamingFilter, EegAdapter

def test_polyphase_resampler_exact_ratio():
    """Verify that 625 samples at 125 Hz resample to exactly 1280 samples at 256 Hz."""
    resampler = PolyphaseResampler(from_fs=125, to_fs=256)
    data_125hz = np.random.randn(625, 16).astype(np.float32)
    resampled = resampler.resample(data_125hz, axis=0)
    assert resampled.shape == (1280, 16)
    assert resampled.dtype == np.float32

def test_channel_remapper_16_to_22():
    """Verify that 16 Cyton channels map to 22 TCP montage channels."""
    remapper = ChannelRemapper()
    data_16ch = np.ones((100, 16), dtype=np.float32)
    # Set F3 = 2.0, F4 = 4.0
    data_16ch[:, 2] = 2.0  # F3
    data_16ch[:, 3] = 4.0  # F4

    remapped = remapper.remap(data_16ch)
    assert remapped.shape == (100, 22)
    
    # Target index 16 is FZ: should be 0.5 * (2.0 + 4.0) = 3.0
    np.testing.assert_allclose(remapped[:, 16], 3.0, atol=1e-5)
    
    # Target index 19 (A1) and 21 (EKG) should be zero-filled
    np.testing.assert_allclose(remapped[:, 19], 0.0, atol=1e-5)
    np.testing.assert_allclose(remapped[:, 21], 0.0, atol=1e-5)

def test_streaming_filter_notch_attenuation():
    """Verify that 50 Hz line noise is attenuated by at least 20 dB."""
    sf = StreamingFilter(fs=256, notch_freq=50.0, notch_q=30.0, n_channels=22)
    t = np.arange(1024) / 256.0
    sine_50hz = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
    chunk = np.repeat(sine_50hz[:, np.newaxis], 22, axis=1)

    filtered = sf.filter(chunk)
    # Allow settling; inspect second half of window
    amp_in = np.max(np.abs(chunk[512:, 0]))
    amp_out = np.max(np.abs(filtered[512:, 0]))
    attenuation_db = 20 * np.log10(amp_in / (amp_out + 1e-9))
    assert attenuation_db > 20.0

def test_streaming_filter_chunk_continuity():
    """
    Verify that filtering data in two chunks with persistent zi matches
    filtering the entire array at once (within numerical tolerance).
    """
    sf1 = StreamingFilter(fs=256, n_channels=22)
    sf2 = StreamingFilter(fs=256, n_channels=22)
    
    full_data = np.random.randn(500, 22).astype(np.float32)
    
    # Continuous filtering
    full_filtered = sf1.filter(full_data)

    # Chunked filtering (2 x 250 samples)
    chunk1 = full_data[:250]
    chunk2 = full_data[250:]
    filtered_chunk1 = sf2.filter(chunk1)
    filtered_chunk2 = sf2.filter(chunk2)
    chunked_filtered = np.vstack([filtered_chunk1, filtered_chunk2])

    np.testing.assert_allclose(full_filtered, chunked_filtered, atol=1e-4)

def test_eeg_adapter_end_to_end():
    """Verify full adapter transforms (625, 16) at 125 Hz into (1280, 22) at 256 Hz."""
    ad = EegAdapter(from_fs=125, to_fs=256, n_channels=22)
    raw_125hz = np.random.randn(625, 16).astype(np.float32)
    out = ad.process_window(raw_125hz, source_fs=125)
    assert out.shape == (1280, 22)
    assert out.dtype == np.float32
