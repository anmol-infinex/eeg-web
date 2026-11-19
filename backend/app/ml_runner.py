"""
backend/app/ml_runner.py
In-process Keras 3 / PyTorch CPU ML inference runner.
"""
import os
# Ensure Keras uses PyTorch backend before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import threading
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import numpy as np
import torch
import keras

logger = logging.getLogger("seizure_sentry.ml_runner")

CLASS_NAMES = ["Normal", "Pre-Seizure", "Seizure"]

class ModelRunner:
    """
    Thread-safe in-process inference engine for tusz_3class_eeg_model.h5.
    Expects input tensor of shape (batch, 1280, 22) or (1280, 22) with dtype float32.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.model: Optional[keras.Model] = None
        self._lock = threading.Lock()
        self._is_loaded: bool = False
        self._is_warm: bool = False
        self._input_shape = (None, 1280, 22)
        self._output_shape = (None, 3)
        self._classes: List[str] = CLASS_NAMES

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    def load(self, model_path: Union[str, Path]) -> None:
        """Load the pre-trained Keras model with PyTorch CPU backend."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {path}")

        logger.info(f"Loading Keras 3 model from {path} with backend='torch'...")
        with self._lock:
            try:
                self.model = keras.models.load_model(str(path), compile=False)
                self.model_path = path
                self._is_loaded = True
                self._input_shape = getattr(self.model, "input_shape", (None, 1280, 22))
                self._output_shape = getattr(self.model, "output_shape", (None, 3))
                logger.info(f"Model loaded successfully. Input: {self._input_shape}, Output: {self._output_shape}")
            except Exception as e:
                self._is_loaded = False
                logger.error(f"Failed to load model from {path}: {e}")
                raise RuntimeError(f"Failed to load model: {e}") from e

    def warmup(self) -> None:
        """Run a single zero-tensor inference pass to warm up PyTorch CPU kernels."""
        if not self._is_loaded or self.model is None:
            raise RuntimeError("Cannot warm up model: Model is not loaded.")
        
        logger.info("Executing model warmup inference pass...")
        dummy_input = np.zeros((1, 1280, 22), dtype=np.float32)
        with self._lock:
            with torch.no_grad():
                tensor_in = torch.from_numpy(dummy_input)
                _ = self.model(tensor_in)
        self._is_warm = True
        logger.info("Model warmup completed.")

    def validate_input(self, data: np.ndarray) -> np.ndarray:
        """
        Validate input array dimensions, dtype, and check for NaNs/Infs.
        Returns 3D array of shape (batch, 1280, 22).
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
        else:
            data = data.astype(np.float32, copy=False)

        if np.isnan(data).any():
            raise ValueError("Input EEG data contains NaN values.")
        if np.isinf(data).any():
            raise ValueError("Input EEG data contains Inf values.")

        if data.ndim == 2:
            if data.shape != (1280, 22):
                raise ValueError(f"Expected 2D input shape (1280, 22), but got {data.shape}")
            data = np.expand_dims(data, axis=0)
        elif data.ndim == 3:
            if data.shape[0] == 0:
                raise ValueError("Input batch contains 0 samples")
            if data.shape[1] != 1280 or data.shape[2] != 22:
                raise ValueError(f"Expected 3D input shape (batch, 1280, 22), but got {data.shape}")
        else:
            raise ValueError(f"Expected 2D or 3D array, but got array with {data.ndim} dimensions.")

        return data

    def predict(self, eeg_data: Union[np.ndarray, List], normalize: bool = True) -> Dict[str, Any]:
        """
        Execute model inference on an EEG window.
        
        Parameters:
            eeg_data: (1280, 22) or (batch, 1280, 22) array in microvolts.
            normalize: If True, applies per-channel Z-score normalization along time axis.
            
        Returns:
            Dict containing:
                "prediction": "Normal" | "Pre-Seizure" | "Seizure"
                "probabilities": {
                    "normal": float,
                    "pre_seizure": float,
                    "seizure": float
                }
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError("ModelRunner error: Model is not loaded.")

        arr = self.validate_input(eeg_data)

        if normalize:
            # Per-channel Z-score normalization across the 1280 time samples
            mean = np.mean(arr, axis=1, keepdims=True)
            std = np.std(arr, axis=1, keepdims=True) + 1e-6
            arr = (arr - mean) / std

        with self._lock:
            with torch.no_grad():
                tensor_in = torch.from_numpy(arr)
                out_tensor = self.model(tensor_in)
                probs = out_tensor.cpu().numpy()

        # Extract predictions for the first window
        p = probs[0]
        pred_idx = int(np.argmax(p))
        prediction_label = self._classes[pred_idx]

        return {
            "prediction": prediction_label,
            "probabilities": {
                "normal": float(p[0]),
                "pre_seizure": float(p[1]),
                "seizure": float(p[2]),
            }
        }

    def info(self) -> Dict[str, Any]:
        """Return model metadata and runtime state."""
        return {
            "model_name": "TUSZ_EEG_DEEP_CNN",
            "version": "1.0.0",
            "framework": "Keras 3 (PyTorch CPU backend)",
            "loaded": self._is_loaded,
            "warm": self._is_warm,
            "input_shape": list(self._input_shape),
            "output_shape": list(self._output_shape),
            "classes": self._classes,
            "sample_rate_hz": 256,
            "window_samples": 1280,
            "window_sec": 5.0,
            "channels": 22,
        }
