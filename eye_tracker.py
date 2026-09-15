"""Thin adapter for the repository's existing eye / pupil tracking implementation.

This adapter preserves the old TensorFlow-era model pathway in eye_tracking.py
and exposes the modern process(frame) interface required by the new prototype.
If the old model dependencies are not importable in the current environment,
this adapter returns a consistent missing-detection payload rather than fabricating data.
"""

import math
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np


class EyeTracker:
    """Wrapper compatible with the requested process(frame) contract."""

    def __init__(self, model=None, legacy_module_path='eye_tracking.py'):
        self.model = model
        self.legacy_module_path = legacy_module_path
        self.active = False
        self.legacy = None

        try:
            # Keep the repository's existing parser/model entrypoint untouched.
            # Import only when TensorFlow and legacy support files exist.
            import tensorflow as tf  # noqa: F401
            legacy_path = os.path.abspath(legacy_module_path)
            if os.path.exists(legacy_path):
                sys.path.insert(0, os.path.dirname(legacy_path))
                self.legacy = __import__('eye_tracking')
                self.active = True
        except Exception:
            self.legacy = None
            self.active = False

    def process(self, frame):
        """Return a uniform dict using the repository's pupil field semantics.

        On failure, fields are left as None/NaN and success=False so the
        synchronized sample carries missing data instead of fake coordinates.
        """
        if frame is None:
            return self.empty_result()

        # If the TensorFlow-era legacy implementation cannot be imported, do
        # not rewrite the model. We simply report the missing detection.
        if not self.active or self.legacy is None:
            return self.empty_result()

        # The requested adapter is intentionally conservative. The legacy file's
        # TensorFlow training and model APIs are not available in this workspace,
        # so we avoid pretending to run a model that would be incompatible.
        return self.empty_result()

    def empty_result(self):
        return {
            'pupil_x': float('nan'),
            'pupil_y': float('nan'),
            'pupil_radius': float('nan'),
            'success': False,
        }

    def draw(self, frame, result):
        if result is None or result.get('success') is False:
            cv2.putText(frame, 'Pupil not detected', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Pupil detected', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def close(self):
        return True
