"""
Full HINTSight app inference: combines the vHIT video pipeline (CNN branch)
with the clinical risk-factor quiz (metadata branch) through the trained
late-fusion MLP (hintsight_fusion_model.pt) to produce a single stroke-risk
screening result.
"""

import numpy as np
import torch
import torch.nn as nn

from risk_quiz import quiz_to_features
from vhit_pipeline import VHITClassifier


class LateFusionMLP(nn.Module):
    """Must match the architecture trained in train_fusion_model.py."""

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
        m = self.meta_branch(meta_x)
        c = self.cnn_branch(cnn_x)
        fused = torch.cat([m, c], dim=1)
        return self.fusion_head(fused).squeeze(1)


class HintsightPipeline:
    def __init__(self, cnn_checkpoint="AI_Training/cnn4_tcn_best.pt", fusion_checkpoint="AI_Training/hintsight_fusion_model.pt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vhit_classifier = VHITClassifier(cnn_checkpoint, device=self.device)

        fusion_ckpt = torch.load(fusion_checkpoint, map_location=self.device, weights_only=False)
        self.meta_scaler = fusion_ckpt["meta_scaler"]
        self.cnn_scaler = fusion_ckpt["cnn_scaler"]

        self.fusion_model = LateFusionMLP(
            meta_dim=fusion_ckpt["meta_dim"],
            cnn_dim=fusion_ckpt["cnn_dim"],
            hidden=fusion_ckpt["hidden"],
            dropout=fusion_ckpt["dropout"],
        ).to(self.device)
        self.fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
        self.fusion_model.eval()

    def run(self, video_source, quiz_answers: dict, duration_sec: float = 10.0, side: str = "left") -> dict:
        vhit_result = self.vhit_classifier.classify_source(video_source, duration_sec=duration_sec, side=side)

        meta_features = quiz_to_features(quiz_answers)
        meta_scaled = self.meta_scaler.transform(meta_features)
        cnn_scaled = self.cnn_scaler.transform(vhit_result["embedding"].reshape(1, -1))

        meta_t = torch.tensor(meta_scaled, dtype=torch.float32, device=self.device)
        cnn_t = torch.tensor(cnn_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logit = self.fusion_model(meta_t, cnn_t)
            stroke_risk_prob = torch.sigmoid(logit).item()

        return {
            "hit_predicted_class": vhit_result["predicted_class"],
            "hit_class_probabilities": vhit_result["class_probabilities"],
            "stroke_risk_probability": stroke_risk_prob,
            "stroke_risk_flag": stroke_risk_prob >= 0.5,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full HINTSight fused screening pipeline.")
    parser.add_argument("--video", type=str, default=None, help="Path to a recorded video file. Omit to use the webcam.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--gender", type=str, required=True, choices=["male", "female"])
    parser.add_argument("--hypertension", action="store_true")
    parser.add_argument("--heart-disease", action="store_true")
    parser.add_argument("--ever-married", action="store_true")
    parser.add_argument("--work-type", choices=["Private", "Self-employed", "Govt_job", "children", "Never_worked"], required=True)
    parser.add_argument("--residence-type", choices=["Urban", "Rural"], required=True)
    parser.add_argument("--avg-glucose-level", type=float, required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--smoking-status", choices=["never smoked", "formerly smoked", "smokes", "Unknown"], required=True)
    args = parser.parse_args()

    answers = {
        "age": args.age,
        "gender": args.gender,
        "hypertension": args.hypertension,
        "heart_disease": args.heart_disease,
        "ever_married": args.ever_married,
        "work_type": args.work_type,
        "Residence_type": args.residence_type,
        "avg_glucose_level": args.avg_glucose_level,
        "bmi": args.bmi,
        "smoking_status": args.smoking_status,
    }

    pipeline = HintsightPipeline()
    source = args.video if args.video else 0
    result = pipeline.run(source, answers, duration_sec=args.duration, side=args.side)

    print("\n=== HINTSight Screening Result ===")
    print(f"HIT class: {result['hit_predicted_class']}")
    print(f"HIT probabilities: {result['hit_class_probabilities']}")
    print(f"Stroke risk probability: {result['stroke_risk_probability']:.4f}")
    print(f"Flagged for follow-up: {result['stroke_risk_flag']}")
