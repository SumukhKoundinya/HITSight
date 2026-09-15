"""
Phone video -> segmented eyes + head/eye motion -> HIT CNN -> stroke fusion.

The legacy segmentEyeRegion.py uses two fixed eye crops in a 320-pixel-wide
region: left=[2:120, 160:320] and right=[2:120, 0:160]. This file keeps that
crop convention, but exposes a callable Keras segmentation adapter so a phone
video can be processed frame by frame.

The Keras segmentation model must return either:
  * a mask shaped (H, W), (H, W, 1), or (1, H, W, 1), or
  * pupil coordinates shaped (2,) or (1, 2), interpreted as x, y.

The late-fusion checkpoint must be produced by the ten-real-feature version of
train_fusion_model.py. Metadata order is:
gender, age, hypertension, heart_disease, ever_married, work_type,
Residence_type, avg_glucose_level, bmi, smoking_status.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter

from hitsight_cnn import load_cnn
from vhit_pipeline import FaceTracker


LEGACY_LEFT_CROP = (2, 120, 160, 320)
LEGACY_RIGHT_CROP = (2, 120, 0, 160)
CNN_RAW_LENGTH = 175
CNN_CROP_START, CNN_CROP_END = 25, 75


class LateFusionMLP(nn.Module):
    def __init__(self, meta_dim, cnn_dim, hidden, dropout):
        super().__init__()
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.cnn_branch = nn.Sequential(
            nn.Linear(cnn_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, meta_x, cnn_x):
        meta_branch = self.meta_branch(meta_x)
        cnn_branch = self.cnn_branch(cnn_x)
        return self.fusion_head(torch.cat([meta_branch, cnn_branch], dim=1)).squeeze(1)


class KerasEyeSegmenter:
    """Loads a Keras pupil segmenter and converts its output to a centroid."""

    def __init__(self, model_path: str, threshold: float = 0.5, input_size: Optional[tuple[int, int]] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Segmentation model not found: {model_path}")
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.threshold = threshold
        shape = self.model.input_shape
        if input_size is None:
            if len(shape) < 3 or shape[1] is None or shape[2] is None:
                raise ValueError("Use --seg-width and --seg-height for a dynamic Keras input shape")
            input_size = (int(shape[1]), int(shape[2]))
        self.input_size = input_size

    def _prepare(self, eye_crop: np.ndarray) -> np.ndarray:
        width, height = self.input_size[1], self.input_size[0]
        image = cv2.resize(eye_crop, (width, height), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32) / 255.0
        return image[None, ...]

    def predict_centroid(self, eye_crop: np.ndarray) -> Optional[tuple[float, float]]:
        prediction = np.asarray(self.model.predict(self._prepare(eye_crop), verbose=0))
        prediction = np.squeeze(prediction)

        if prediction.ndim == 1 and prediction.size == 2:
            x, y = prediction
            if max(abs(x), abs(y)) <= 1.5:
                return float(x * eye_crop.shape[1]), float(y * eye_crop.shape[0])
            return float(x), float(y)

        if prediction.ndim == 2:
            mask = prediction
        elif prediction.ndim == 3 and prediction.shape[-1] == 1:
            mask = prediction[..., 0]
        else:
            raise ValueError(f"Unsupported segmentation output shape: {prediction.shape}")

        mask = cv2.resize(mask.astype(np.float32), (eye_crop.shape[1], eye_crop.shape[0]))
        binary = mask >= self.threshold
        moments = cv2.moments(binary.astype(np.uint8))
        if moments["m00"] == 0:
            return None
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


@dataclass
class FrameMeasurement:
    timestamp_s: float
    head_yaw: float
    eye_x: float
    eye_y: float


class PhoneHITFusionPipeline:
    def __init__(self, segmentation_model: str, fusion_checkpoint: str = "AI_Training/hintsight_fusion_model.pt", side: str = "left"):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.eye_segmenter = KerasEyeSegmenter(segmentation_model)
        self.head_tracker = FaceTracker()
        self.cnn, self.cnn_mean, self.cnn_std, self.cnn_classes = load_cnn("AI_Training/cnn4_tcn_best.pt")

        checkpoint = torch.load(fusion_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["meta_dim"] != 10:
            raise ValueError(
            f"Expected a ten-feature fusion checkpoint, got meta_dim={checkpoint['meta_dim']}. "
                "Retrain train_fusion_model.py without synthetic variables first."
            )
        self.meta_scaler = checkpoint["meta_scaler"]
        self.cnn_scaler = checkpoint["cnn_scaler"]
        self.fusion_model = LateFusionMLP(
            checkpoint["meta_dim"], checkpoint["cnn_dim"], checkpoint["hidden"], checkpoint["dropout"]
        )
        self.fusion_model.load_state_dict(checkpoint["model_state_dict"])
        self.fusion_model.eval()

    @staticmethod
    def _crop(frame: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
        y1, y2, x1, x2 = crop
        height, width = frame.shape[:2]
        if width < x2 or height < y2:
            scale = max(x2 / width, y2 / height)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        return frame[y1:y2, x1:x2]

    def _measure_frame(self, frame: np.ndarray, timestamp_s: float) -> Optional[FrameMeasurement]:
        tracking = self.head_tracker.process_frame(frame, int(timestamp_s * 1000))
        if tracking is None or tracking["head_yaw"] is None:
            return None
        left_crop = self._crop(frame, LEGACY_LEFT_CROP)
        right_crop = self._crop(frame, LEGACY_RIGHT_CROP)
        left_centroid = self.eye_segmenter.predict_centroid(cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY))
        right_centroid = self.eye_segmenter.predict_centroid(cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY))
        centroid = left_centroid if self.side == "left" else right_centroid
        if centroid is None or left_centroid is None or right_centroid is None:
            return None
        return FrameMeasurement(timestamp_s, tracking["head_yaw"], centroid[0], centroid[1])

    @staticmethod
    def _velocity(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        if len(values) < 5:
            raise RuntimeError("At least five valid tracked frames are required")
        window = min(9, len(values) if len(values) % 2 else len(values) - 1)
        smooth = savgol_filter(values, window_length=max(5, window), polyorder=2) if window >= 5 else values
        return np.gradient(smooth, timestamps)

    def _cnn_signal(self, measurements: list[FrameMeasurement]) -> np.ndarray:
        timestamps = np.array([item.timestamp_s for item in measurements], dtype=np.float64)
        head = self._velocity(np.array([item.head_yaw for item in measurements]), timestamps)
        eye = self._velocity(np.array([item.eye_x for item in measurements]), timestamps)
        peak = int(np.argmax(np.abs(head)))
        center_s = timestamps[peak]
        grid = center_s - 0.025 + np.arange(CNN_RAW_LENGTH) / 1000.0
        head_window = np.interp(grid, timestamps, head)
        eye_window = np.interp(grid, timestamps, eye)
        return np.stack([head_window, eye_window]).astype(np.float32)

    def run(self, video_path: str, metadata: dict, duration_sec: float = 10.0) -> dict:
        required = {
            "gender", "age", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status",
        }
        missing = required - metadata.keys()
        if missing:
            raise ValueError(f"Missing metadata fields: {sorted(missing)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        measurements = []
        try:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_s = frame_index / fps
                if timestamp_s >= duration_sec:
                    break
                measurement = self._measure_frame(frame, timestamp_s)
                if measurement is not None:
                    measurements.append(measurement)
                frame_index += 1
        finally:
            cap.release()
            self.head_tracker.close()

        raw_signal = self._cnn_signal(measurements)
        signal = raw_signal[:, CNN_CROP_START:CNN_CROP_END]
        signal = (signal - self.cnn_mean[0]) / self.cnn_std[0]
        signal_tensor = torch.tensor(signal[None, ...], dtype=torch.float32)
        with torch.no_grad():
            logits = self.cnn(signal_tensor)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
            embedding = self.cnn.extract_features(signal_tensor).numpy()

        meta_row = np.array([[
            1 if str(metadata["gender"]).lower().startswith("m") else 0,
            float(metadata["age"]), int(bool(metadata["hypertension"])),
            int(bool(metadata["heart_disease"])), int(bool(metadata["ever_married"])),
            {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}[metadata["work_type"]],
            1 if metadata["Residence_type"] == "Urban" else 0,
            float(metadata["avg_glucose_level"]), float(metadata["bmi"]),
            {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}[metadata["smoking_status"]],
        ]], dtype=np.float32)
        meta_scaled = self.meta_scaler.transform(meta_row)
        cnn_scaled = self.cnn_scaler.transform(embedding)
        with torch.no_grad():
            stroke_probability = torch.sigmoid(self.fusion_model(
                torch.tensor(meta_scaled, dtype=torch.float32),
                torch.tensor(cnn_scaled, dtype=torch.float32),
            )).item()

        return {
            "frames_used": len(measurements),
            "hit_predicted_class": self.cnn_classes[int(np.argmax(probabilities))],
            "hit_class_probabilities": dict(zip(self.cnn_classes, probabilities.tolist())),
            "stroke_risk_probability": stroke_probability,
            "stroke_risk_flag": stroke_probability >= 0.5,
        }


def main():
    parser = argparse.ArgumentParser(description="Run phone HIT video through segmentation, tracking, HIT CNN, and fusion")
    parser.add_argument("video")
    parser.add_argument("--segmentation-model", required=True)
    parser.add_argument("--fusion-checkpoint", default="AI_Training/hintsight_fusion_model.pt")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--gender", choices=["male", "female"], required=True)
    parser.add_argument("--hypertension", action="store_true")
    parser.add_argument("--heart-disease", action="store_true")
    parser.add_argument("--ever-married", action="store_true")
    parser.add_argument("--work-type", choices=["Private", "Self-employed", "Govt_job", "children", "Never_worked"], required=True)
    parser.add_argument("--residence-type", choices=["Urban", "Rural"], required=True)
    parser.add_argument("--avg-glucose-level", type=float, required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--smoking-status", choices=["never smoked", "formerly smoked", "smokes", "Unknown"], required=True)
    args = parser.parse_args()
    metadata = vars(args)
    result = PhoneHITFusionPipeline(args.segmentation_model, args.fusion_checkpoint, args.side).run(
        args.video, metadata, args.duration
    )
    print(result)


if __name__ == "__main__":
    main()
