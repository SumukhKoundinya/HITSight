"""
HINTSight vHIT preprocessing pipeline
======================================
Turns a ~10 second phone/webcam recording of a head-impulse test into the
2-channel (head angular velocity, eye angular velocity) signal the pretrained
TCN1D CNN expects, using MediaPipe's FaceLandmarker (iris + head pose) for
pupil tracking and head tracking.

Pipeline stages:
  1. Capture frames (webcam or video file) and run FaceLandmarker per frame.
  2. Derive head yaw (deg) from the facial transformation matrix and eye
     horizontal gaze angle (deg) from the iris position within the eye
     aperture, for whichever side ("left"/"right") is being tested.
  3. Differentiate both angle traces into angular velocity (deg/s) and
     smooth with a Savitzky-Golay filter to suppress landmark jitter.
  4. Detect the head-impulse (the brief, fast head rotation) and extract a
     window around it, resampled onto the same 1 ms grid / 175-sample length
     used when the CNN was trained, then apply the identical 25-75 ms crop.
  5. Normalize with the CNN's stored mean/std and run inference, returning
     both the class probabilities and the fusion-ready embedding.

Caveats: gaze-angle estimation from 2D iris landmarks is a geometric
approximation (not a scleral search coil / true eye tracker), and webcam frame
rates (~30-60 fps) are far below a clinical vHIT system's native sampling
rate, so the resampled signal is a best-effort approximation suitable for a
screening prototype, not a diagnostic-grade measurement.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from hitsight_cnn import load_cnn

# ============================================================
# MEDIAPIPE FACE LANDMARKER MODEL ASSET
# ============================================================

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# MediaPipe FaceMesh topology (with iris refinement) landmark indices
LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER = 468, 473
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374

SAMPLE_PERIOD_MS = 1.0  # matches the 1 ms/sample grid baked into cnn4_tcn_best.pt
RAW_LENGTH = 175
WINDOW_START, WINDOW_END = 25, 75  # matches HITClassification.ipynb 25-75 ms crop


def ensure_model_asset() -> str:
    if not os.path.exists(_MODEL_PATH):
        print("Downloading MediaPipe FaceLandmarker model asset (one-time)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _rotation_matrix_to_euler(R):
    """3x3 rotation matrix -> (yaw, pitch, roll) in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def _gaze_horizontal_angle(landmarks, iris_idx, inner_idx, outer_idx, fov_deg=45.0):
    """Approximate horizontal eye-in-head gaze angle (deg) from iris position
    projected onto the inner-outer eye-corner axis, normalized to [-1, 1]."""
    iris = landmarks[iris_idx]
    inner = landmarks[inner_idx]
    outer = landmarks[outer_idx]

    axis_x, axis_y = outer.x - inner.x, outer.y - inner.y
    axis_len_sq = axis_x ** 2 + axis_y ** 2 + 1e-8

    ratio = ((iris.x - inner.x) * axis_x + (iris.y - inner.y) * axis_y) / axis_len_sq
    return (ratio - 0.5) * fov_deg


class FaceTracker:
    """Thin wrapper around MediaPipe FaceLandmarker (VIDEO running mode)."""

    def __init__(self):
        base_options = mp.tasks.BaseOptions(model_asset_path=ensure_model_asset())
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process_frame(self, frame_bgr, timestamp_ms: int):
        # MediaPipe requires strictly increasing timestamps per landmarker instance
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        yaw = pitch = roll = None
        if result.facial_transformation_matrixes:
            R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
            yaw, pitch, roll = _rotation_matrix_to_euler(R)

        eye_h_left = _gaze_horizontal_angle(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER)
        eye_h_right = _gaze_horizontal_angle(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)

        return {
            "timestamp_ms": timestamp_ms,
            "head_yaw": yaw,
            "eye_h_left": eye_h_left,
            "eye_h_right": eye_h_right,
        }

    def close(self):
        self.landmarker.close()


# ============================================================
# CAPTURE
# ============================================================


def capture_time_series(source=0, duration_sec: float = 10.0) -> list[dict]:
    """Reads frames from a webcam (source=int index) or a video file
    (source=path) and returns a list of per-frame tracking records."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    tracker = FaceTracker()
    records = []
    is_live = isinstance(source, int)
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if is_live:
                elapsed_ms = (time.time() - start_time) * 1000.0
                if elapsed_ms / 1000.0 >= duration_sec:
                    break
            else:
                elapsed_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if duration_sec is not None and elapsed_ms / 1000.0 >= duration_sec:
                    break

            rec = tracker.process_frame(frame, int(elapsed_ms))
            if rec is not None:
                records.append(rec)
    finally:
        cap.release()
        tracker.close()

    if len(records) < 5:
        raise RuntimeError("Face/iris tracking failed on too few frames; ensure good lighting and a visible face.")

    return records


# ============================================================
# SIGNAL PROCESSING
# ============================================================


def _to_velocity(timestamps_s: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Angle (deg) time series -> smoothed angular velocity (deg/s)."""
    window = min(9, len(angle_deg) - (1 - len(angle_deg) % 2))
    window = max(window, 5) if len(angle_deg) >= 5 else len(angle_deg)
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        angle_smoothed = savgol_filter(angle_deg, window_length=window, polyorder=2)
    else:
        angle_smoothed = angle_deg
    return np.gradient(angle_smoothed, timestamps_s)


def _detect_impulse_center(head_velocity: np.ndarray, timestamps_ms: np.ndarray) -> float:
    """Finds the timestamp (ms) of peak head-rotation speed = the impulse."""
    peak_idx = int(np.argmax(np.abs(head_velocity)))
    return timestamps_ms[peak_idx]


def _resample_window(timestamps_ms, values, center_ms, pre_ms=25, total_len=RAW_LENGTH):
    """Interpolates `values` onto a uniform 1 ms grid spanning
    [center_ms - pre_ms, center_ms - pre_ms + total_len) samples."""
    grid = center_ms - pre_ms + np.arange(total_len) * SAMPLE_PERIOD_MS
    interp = interp1d(timestamps_ms, values, kind="linear", bounds_error=False,
                       fill_value=(values[0], values[-1]))
    return interp(grid)


def build_signal_from_records(records: list[dict], side: str = "left"):
    """Converts raw per-frame tracking records into the (2, 175) raw signal
    (head velocity, eye velocity) expected before the CNN's 25-75 ms crop."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    timestamps_ms = np.array([r["timestamp_ms"] for r in records], dtype=np.float64)
    head_yaw = np.array([r["head_yaw"] for r in records], dtype=np.float64)
    eye_key = "eye_h_left" if side == "left" else "eye_h_right"
    eye_angle = np.array([r[eye_key] for r in records], dtype=np.float64)

    valid = ~np.isnan(head_yaw) & ~np.isnan(eye_angle)
    timestamps_ms, head_yaw, eye_angle = timestamps_ms[valid], head_yaw[valid], eye_angle[valid]
    if len(timestamps_ms) < 5:
        raise RuntimeError("Not enough valid tracked frames (head pose or iris missing) to build a signal.")

    timestamps_s = timestamps_ms / 1000.0
    head_velocity = _to_velocity(timestamps_s, head_yaw)
    eye_velocity = _to_velocity(timestamps_s, eye_angle)

    impulse_center_ms = _detect_impulse_center(head_velocity, timestamps_ms)

    head_window = _resample_window(timestamps_ms, head_velocity, impulse_center_ms)
    eye_window = _resample_window(timestamps_ms, eye_velocity, impulse_center_ms)

    return np.stack([head_window, eye_window], axis=0).astype(np.float32)  # (2, 175)


# ============================================================
# END-TO-END: VIDEO -> CNN EMBEDDING / PREDICTION
# ============================================================


class VHITClassifier:
    def __init__(self, checkpoint_path="cnn4_tcn_best.pt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.mean, self.std, self.classes = load_cnn(checkpoint_path, device=self.device)

    def signal_to_tensor(self, raw_signal_2x175: np.ndarray) -> torch.Tensor:
        windowed = raw_signal_2x175[:, WINDOW_START:WINDOW_END]  # (2, 50)
        normalized = (windowed - self.mean[0]) / self.std[0]
        return torch.tensor(normalized[np.newaxis, ...], dtype=torch.float32, device=self.device)

    def predict(self, raw_signal_2x175: np.ndarray) -> dict:
        x = self.signal_to_tensor(raw_signal_2x175)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            embedding = self.model.extract_features(x).cpu().numpy()[0]  # 260-d, for fusion model

        pred_class = self.classes[int(np.argmax(probs))]
        return {
            "predicted_class": pred_class,
            "class_probabilities": dict(zip(self.classes, probs.tolist())),
            "embedding": embedding,
        }

    def classify_source(self, source, duration_sec: float = 10.0, side: str = "left") -> dict:
        records = capture_time_series(source, duration_sec=duration_sec)
        raw_signal = build_signal_from_records(records, side=side)
        result = self.predict(raw_signal)
        result["raw_signal"] = raw_signal
        result["side"] = side
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the HINTSight vHIT preprocessing + CNN pipeline.")
    parser.add_argument("--video", type=str, default=None, help="Path to a recorded video file. Omit to use the webcam.")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (webcam mode).")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    args = parser.parse_args()

    source = args.video if args.video else 0
    classifier = VHITClassifier()
    output = classifier.classify_source(source, duration_sec=args.duration, side=args.side)

    print("\nPredicted class:", output["predicted_class"])
    print("Class probabilities:")
    for cls, p in output["class_probabilities"].items():
        print(f"  {cls}: {p:.4f}")
