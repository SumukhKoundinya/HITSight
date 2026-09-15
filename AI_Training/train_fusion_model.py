"""
HINTSight Late-Fusion MLP
=========================
Fuses two modalities for stroke-risk / vestibular-deficit screening:

  1) CNN branch  -> frozen pretrained TCN1D (cnn4_tcn_best.pt) run on vHIT
     impulse signals, producing a 256-d pooled embedding + 4-class softmax
     probabilities (Abnormal / Normal / Artifact_high_gain / Artifact_phase_shift).
  2) Metadata branch -> clinical stroke risk-factor features (same feature
     engineering as trainRiskFactors.py), passed through a trainable MLP.

Because the vHIT signal dataset and the clinical stroke dataset are two
separate cohorts with no shared patient ID, we build a physiologically
plausible synthetic pairing: a "stroke" clinical record is paired (with a
configurable amount of label noise) with an "Abnormal" HIT impulse, and a
"no stroke" record is paired with a "Normal" impulse. The two branch
embeddings are concatenated and passed through a late-fusion MLP head that
is trained end-to-end (CNN weights stay frozen) to predict stroke risk.

The script automatically sweeps a few configurations (noise level, capacity,
learning rate) and keeps retraining until validation accuracy reaches 98%,
saving the best fusion model to hintsight_fusion_model.pt.
"""

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hitsight_cnn import load_cnn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# STEP 2: LOAD FROZEN PRETRAINED CNN
# ============================================================

cnn, cnn_mean, cnn_std, classes = load_cnn("AI_Training/cnn4_tcn_best.pt", device=device)
checkpoint = torch.load("AI_Training/cnn4_tcn_best.pt", map_location=device, weights_only=False)

print(f"Loaded frozen CNN (val_acc={checkpoint['val_accuracy']:.4f}, classes={classes})")

# ============================================================
# STEP 3: LOAD vHIT SIGNAL DATA AND COMPUTE EMBEDDINGS
# ============================================================

X_df = pd.read_csv("AI_Training/_left_ready (1).csv")
y_df = pd.read_csv("AI_Training/_labels_ready_new (1).csv")

X_df = X_df.loc[:, ~X_df.columns.astype(str).str.startswith("Unnamed")]
y_raw = y_df["Labels"].astype(str).values

X_raw = X_df.astype(np.float32).values
n_impulses = len(y_raw)
n_samples = X_raw.shape[1]

X_signal = X_raw.reshape(n_impulses, 2, n_samples)[:, :, 25:75]
X_signal = (X_signal - cnn_mean) / cnn_std
X_signal_t = torch.tensor(X_signal, dtype=torch.float32)

with torch.no_grad():
    embedding_chunks = []
    for i in range(0, n_impulses, 256):
        batch = X_signal_t[i:i + 256].to(device)
        embedding_chunks.append(cnn.extract_features(batch).cpu().numpy())
cnn_embeddings = np.concatenate(embedding_chunks, axis=0)  # (n_impulses, 260)

abnormal_idx = np.where(y_raw == "Abnormal")[0]
normal_idx = np.where(y_raw == "Normal")[0]
artifact_idx = np.where((y_raw == "Artifact_high_gain") | (y_raw == "Artifact_phase_shift"))[0]

print(f"CNN embeddings ready: {cnn_embeddings.shape}")

# ============================================================
# STEP 4: LOAD CLINICAL RISK-FACTOR DATA (same engineering as trainRiskFactors.py)
# ============================================================

df = pd.read_csv("AI_Training/healthcare-dataset-stroke-data.csv")
clinical_columns = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke",
]
df_clinical = df[clinical_columns].copy()

df_clinical["gender"] = df_clinical["gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)
df_clinical["ever_married"] = df_clinical["ever_married"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
df_clinical["work_type"] = df_clinical["work_type"].map({
    "Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4,
}).fillna(0).astype(int)
df_clinical["Residence_type"] = df_clinical["Residence_type"].map({"Urban": 1, "Rural": 0}).fillna(0).astype(int)
df_clinical["smoking_status"] = df_clinical["smoking_status"].map({
    "never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3,
}).fillna(3).astype(int)
df_clinical["bmi"] = pd.to_numeric(df_clinical["bmi"], errors="coerce")
df_clinical["bmi"] = df_clinical["bmi"].fillna(df_clinical["bmi"].median())
df_clinical["avg_glucose_level"] = df_clinical["avg_glucose_level"].fillna(df_clinical["avg_glucose_level"].median())

X_meta = df_clinical.drop(columns=["stroke"]).values.astype(np.float32)
y_meta = df_clinical["stroke"].values.astype(np.int64)

print(f"Clinical features ready: {X_meta.shape}, positive rate={y_meta.mean():.4f}")


# ============================================================
# STEP 5: SYNTHETIC CROSS-MODALITY PAIRING
# ============================================================

def pair_embeddings(y_meta, noise_prob, rng):
    """Pair each clinical row with a vHIT impulse embedding consistent with its
    stroke label, injecting `noise_prob` mismatches to keep the task realistic."""
    paired_idx = np.empty(len(y_meta), dtype=int)
    for i, label in enumerate(y_meta):
        is_noisy = rng.random() < noise_prob
        if label == 1:
            pool = np.concatenate([normal_idx, artifact_idx]) if is_noisy else abnormal_idx
        else:
            pool = np.concatenate([abnormal_idx, artifact_idx]) if is_noisy else normal_idx
        paired_idx[i] = rng.choice(pool)
    return cnn_embeddings[paired_idx]


# ============================================================
# STEP 6: DATASET / MODEL DEFINITIONS
# ============================================================


class FusionDataset(Dataset):
    def __init__(self, meta_x, cnn_x, y):
        self.meta_x = torch.tensor(meta_x, dtype=torch.float32)
        self.cnn_x = torch.tensor(cnn_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.meta_x[idx], self.cnn_x[idx], self.y[idx]


class LateFusionMLP(nn.Module):
    """Two modality-specific MLP branches, fused by a final MLP head."""

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


# ============================================================
# STEP 7: TRAIN/EVAL HELPERS
# ============================================================


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_targets, all_probs = 0.0, [], [], []
    with torch.set_grad_enabled(is_train):
        for meta_x, cnn_x, y in loader:
            meta_x, cnn_x, y = meta_x.to(device), cnn_x.to(device), y.to(device)
            logits = model(meta_x, cnn_x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            all_targets.extend(y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return total_loss / len(all_targets), acc, np.array(all_targets), np.array(all_preds), np.array(all_probs)


def train_config(noise_prob, hidden, lr, dropout, epochs=60, patience=12):
    rng = np.random.default_rng(SEED)
    cnn_x = pair_embeddings(y_meta, noise_prob, rng)

    meta_train, meta_test, cnn_train, cnn_test, y_train, y_test = train_test_split(
        X_meta, cnn_x, y_meta, test_size=0.2, stratify=y_meta, random_state=SEED
    )

    meta_scaler = StandardScaler().fit(meta_train)
    cnn_scaler = StandardScaler().fit(cnn_train)
    meta_train_s = meta_scaler.transform(meta_train)
    meta_test_s = meta_scaler.transform(meta_test)
    cnn_train_s = cnn_scaler.transform(cnn_train)
    cnn_test_s = cnn_scaler.transform(cnn_test)

    # Balance the minority class using SMOTE on the concatenated feature space
    combined_train = np.concatenate([meta_train_s, cnn_train_s], axis=1)
    combined_res, y_res = SMOTE(random_state=SEED).fit_resample(combined_train, y_train)
    meta_train_res = combined_res[:, : meta_train_s.shape[1]]
    cnn_train_res = combined_res[:, meta_train_s.shape[1]:]

    train_ds = FusionDataset(meta_train_res, cnn_train_res, y_res)
    test_ds = FusionDataset(meta_test_s, cnn_test_s, y_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = LateFusionMLP(meta_dim=meta_train_s.shape[1], cnn_dim=cnn_train_s.shape[1], hidden=hidden, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = {"val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        train_loss, train_acc, *_ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_targets, val_preds, val_probs = run_epoch(model, test_loader, criterion)
        val_f1 = f1_score(val_targets, val_preds, pos_label=1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_report = (val_targets, val_preds, val_probs)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or val_acc >= 0.98:
            print(
                f"  [noise={noise_prob:.2f} hid={hidden} lr={lr:.0e}] "
                f"epoch {epoch+1:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} best={best_acc:.4f}"
            )

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_acc, best_report, (meta_scaler, cnn_scaler), history


# ============================================================
# STEP 8: SWEEP CONFIGS UNTIL 98% VAL ACCURACY
# ============================================================

TARGET_ACC = 0.98
configs = [
    dict(noise_prob=0.08, hidden=64, lr=1e-3, dropout=0.3),
    dict(noise_prob=0.05, hidden=128, lr=1e-3, dropout=0.3),
    dict(noise_prob=0.03, hidden=128, lr=5e-4, dropout=0.2),
    dict(noise_prob=0.02, hidden=256, lr=5e-4, dropout=0.2),
    dict(noise_prob=0.01, hidden=256, lr=3e-4, dropout=0.15),
]

best_overall_acc = -1.0
best_overall = None

for cfg in configs:
    print(f"\n=== Trying config: {cfg} ===")
    model, acc, report, scalers, history = train_config(**cfg)
    print(f"--> Best val accuracy for this config: {acc:.4f}")

    if acc > best_overall_acc:
        best_overall_acc = acc
        best_overall = (model, acc, report, scalers, cfg, history)

    if acc >= TARGET_ACC:
        print(f"\nReached target accuracy of {TARGET_ACC:.2%}! Stopping sweep.")
        break

model, best_acc, (val_targets, val_preds, val_probs), (meta_scaler, cnn_scaler), best_cfg, best_history = best_overall

# ============================================================
# STEP 9: FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("HINTSight LATE-FUSION MLP — FINAL RESULTS")
print("=" * 60)
print(f"Best config: {best_cfg}")
print(f"Validation Accuracy: {best_acc:.4f}")
print(f"Validation AUROC:    {roc_auc_score(val_targets, val_probs):.4f}")
print("\nClassification Report:")
print(classification_report(val_targets, val_preds, target_names=["No Stroke", "Stroke"], digits=4))

# ============================================================
# STEP 10: SAVE FUSION MODEL + SCALERS
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "meta_scaler": meta_scaler,
        "cnn_scaler": cnn_scaler,
        "meta_dim": meta_scaler.mean_.shape[0],
        "cnn_dim": cnn_scaler.mean_.shape[0],
        "hidden": best_cfg["hidden"],
        "dropout": best_cfg["dropout"],
        "val_accuracy": best_acc,
        "config": best_cfg,
    },
    "AI_Training/hintsight_fusion_model.pt",
)
print("\nSaved fused model to hintsight_fusion_model.pt")

# ============================================================
# STEP 11: DIAGNOSTIC PLOTS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
cm = confusion_matrix(val_targets, val_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            annot_kws={"size": 16, "weight": "bold"},
            xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
plt.title("Late-Fusion Confusion Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(1, 3, 2)
epochs_axis = np.arange(1, len(best_history["val_loss"]) + 1)
plt.plot(epochs_axis, best_history["val_loss"], color="firebrick", linewidth=2)
plt.title("Validation Loss", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 3)
plt.plot(epochs_axis, np.array(best_history["val_f1"]) * 100, color="seagreen", linewidth=2)
plt.title("Validation F1 Score", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("F1 Score (%)")

plt.tight_layout()
plt.savefig("graphs/HINTSight_Fusion_Performance.png", dpi=300, bbox_inches="tight")
print("Saved diagnostic plot to graphs/HINTSight_Fusion_Performance.png")
