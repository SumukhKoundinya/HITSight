"""Adapter around the repository's existing head / face tracker.

This file preserves the old implementation in head_tracking.py and exposes the
process(frame) interface expected by hit_pipeline.py.
"""

import numpy as np
import cv2


class HeadTracker:
    def __init__(self):
        self.legacy = None
        try:
            from head_tracking import HeadTracker as LegacyHeadTracker
            self.legacy = LegacyHeadTracker()
        except Exception as exc:
            self.legacy = None
            self.error = str(exc)

    def process(self, frame):
        """Return a dict with the requested head pose contract.

        If the legacy face model fails or no face is found, return an error-safe
        failure object rather than substituting fake values.
        """
        if frame is None:
            return None

        if self.legacy is None:
            return None

        try:
            pose = self.legacy.process(frame)
            if pose is None:
                return None
            return {
                'yaw': float(pose.get('yaw', np.nan)),
                'pitch': float(pose.get('pitch', np.nan)),
                'roll': float(pose.get('roll', np.nan)),
                'tx': float(pose.get('tx', np.nan)),
                'ty': float(pose.get('ty', np.nan)),
                'tz': float(pose.get('tz', np.nan)),
                'success': True,
            }
        except Exception:
            return None

    def draw(self, frame, pose):
        if pose is None:
            cv2.putText(frame, 'Face not detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        text = f"Yaw: {pose['yaw']:.1f} Pitch: {pose['pitch']:.1f} Roll: {pose['roll']:.1f}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def close(self):
        if self.legacy and hasattr(self.legacy, 'close'):
            try:
                self.legacy.close()
            except Exception:
                pass
