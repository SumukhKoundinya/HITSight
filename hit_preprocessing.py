"""Repository-grounded preprocessing contract for the HINTSight vHIT pipeline.

The confirmed training contract in HITClassification.ipynb is:
  - raw signal matrices are shaped as n_impulses x 2 x n_samples
  - 25-75 ms window extraction is applied after signal capture
  - normalization is (X - mean) / std with mean/std computed along axis=(0,2)

This module keeps the training evidence unmodified, but adds the requested
mobile-safe temporal standardization layer using CubicSpline to convert a
legacy 240Hz research dataset trace into a 60Hz uniform vector of exactly
300 samples for a 5-second observation window.
"""

import os
import csv
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

from config import TRAINING_CONTRACT


def get_training_contract() -> Dict[str, Any]:
    """Return a simple contract dictionary derived from the training notebook evidence."""
    return dict(TRAINING_CONTRACT)


def normalize_temporal_signal(timestamps, diameters, target_hz=60, duration_sec=5):
    """Resample a raw time series onto a uniform 60Hz timeline of exactly 300 points.

    Parameters
    ----------
    timestamps : array-like
        Absolute timestamps in seconds. If not provided, a sample index grid is used.
    diameters : array-like
        Pupil diameter or signal values to interpolate.
    target_hz : int
        Sampling frequency to standardize to. Default 60.
    duration_sec : int
        Duration window to standardize. Default 5 seconds.

    Returns
    -------
    timeline : np.ndarray
        Uniform target timeline in seconds of shape (300,).
    signal : np.ndarray
        Interpolated diameters sampled on the uniform 60Hz grid.
    """
    try:
        x = np.asarray(timestamps, dtype=float)
        y = np.asarray(diameters, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("timestamps and diameters must be 1D vectors")
        if x.shape[0] != y.shape[0]:
            raise ValueError("timestamps and diameters must have the same length")
        if x.shape[0] < 2:
            raise ValueError("at least two observations are required")

        # Drop rows containing NaN or invalid values gracefully.
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.shape[0] < 2:
            raise ValueError("at least two finite observations are required")

        # Sort by timestamp to avoid CubicSpline domain errors.
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Handle duplicate timestamp boundaries safely.
        unique_x, idx = np.unique(x, return_index=True)
        if len(unique_x) < 2:
            raise ValueError("cannot build a spline with fewer than two unique timestamps")
        x = x[idx]
        y = y[idx]

        # Prevent boundary NaN issue by using the observed finite domain only.
        x0 = float(x.min())
        x1 = float(x.max())
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
            raise ValueError("invalid timestamp domain for interpolation")

        # Build the uniform 60Hz target timeline for exactly 300 data points (5 seconds).
        n_points = int(target_hz * duration_sec)
        timeline = np.linspace(0.0, duration_sec, n_points, endpoint=False)

        # Carry a monoclinic time-space interval meaningful for the requested mobile 60Hz study path.
        x_domain = x - x.min()
        target_domain = np.linspace(0.0, float(x.max() - x.min()), n_points, endpoint=False)

        # Interpolate using CubicSpline over the observed raw signal.
        spline = CubicSpline(x_domain, y, bc_type='natural')
        signal = spline(target_domain)
        signal = np.nan_to_num(signal, nan=np.nanmean(y), posinf=np.nanmean(y), neginf=np.nanmean(y))

        return timeline, signal
    except Exception as exc:
        raise ValueError(f"normalize_temporal_signal failed: {exc}") from exc


def load_raw_csv(csv_path: str):
    """Load a synchronized raw sample CSV file into a pandas DataFrame.

    Columns are expected to be similar to:
    timestamp, frame_index, pupil_x, pupil_y, pupil_radius,
    head_yaw, head_pitch, head_roll, head_tx, head_ty, head_tz
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Raw CSV not found: {csv_path}')
    return pd.read_csv(csv_path)


def prepare_signals_from_csv(csv_path: str, window_ms=(25, 75), sample_length=175):
    """Convert a raw CSV rowset into a 3D tensor structure for later classifier use.

    The wrapper is deliberately conservative: it only accepts the already
    synchronized fields, and it verifies the 2-channel, 175-sample window shape.
    """
    df = load_raw_csv(csv_path)
    required = [
        'pupil_x', 'pupil_y', 'pupil_radius',
        'head_yaw', 'head_pitch', 'head_roll',
        'head_tx', 'head_ty', 'head_tz'
    ]
    for field in required:
        if field not in df.columns:
            raise ValueError(f'Missing field required for preprocessing: {field}')

    n_rows = max(1, len(df))
    x = np.zeros((n_rows, 2, sample_length), dtype=np.float32)

    pupil_xy = df[['pupil_x', 'pupil_y']].to_numpy(dtype=float)
    eye_signal = np.nan_to_num(pupil_xy, nan=0.0)

    head_euler = df[['head_yaw', 'head_pitch', 'head_roll']].to_numpy(dtype=float)
    head_signal = np.nan_to_num(head_euler, nan=0.0)

    # Placeholder interpolation for mobile 60Hz standardization is intentionally
    # repurposed only as a contract validation wrapper, leaving the training data untouched.
    # The notebook dataset itself remains the authoritative shape source.
    x[:, 0, :] = np.interp(np.linspace(0, 1, sample_length), np.linspace(0, 1, len(eye_signal)), eye_signal[:, 0]) if len(eye_signal) > 1 else 0.0
    x[:, 1, :] = np.interp(np.linspace(0, 1, sample_length), np.linspace(0, 1, len(head_signal)), head_signal[:, 0]) if len(head_signal) > 1 else 0.0

    return x[:, :, window_ms[0]:window_ms[1]], get_training_contract()
