import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==========================================
# STEP 1: LOAD AND CLEAN DATA
# ==========================================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

clinical_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
df_clinical = df[clinical_columns].copy()

# Fix NaNs and encode categories
df_clinical['gender'] = df_clinical['gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)
df_clinical['bmi'] = pd.to_numeric(df_clinical['bmi'], errors='coerce')
bmi_median = df_clinical['bmi'].median()
df_clinical['bmi'] = df_clinical['bmi'].fillna(bmi_median)

glucose_median = df_clinical['avg_glucose_level'].median()
df_clinical['avg_glucose_level'] = df_clinical['avg_glucose_level'].fillna(glucose_median)

# ==========================================
# STEP 2: ENCODE DATASET PREDICTORS
# ==========================================
predictor_columns = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status',
]
df_predictors = df[predictor_columns].copy()
df_predictors['gender'] = df_predictors['gender'].map({'Male': 1, 'Female': 0}).fillna(0)
df_predictors['bmi'] = pd.to_numeric(df_predictors['bmi'], errors='coerce').fillna(bmi_median)
df_predictors['avg_glucose_level'] = df_predictors['avg_glucose_level'].fillna(glucose_median)
X = pd.get_dummies(df_predictors, dtype=float)
y = df['stroke']

# ==========================================
# STEP 3: SPLIT AND RESAMPLE (SMOTE)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(
    f"Training classes after balancing: "
    f"no stroke={(y_train_balanced == 0).sum()}, "
    f"stroke={(y_train_balanced == 1).sum()}"
)

# ==========================================
# STEP 4: SCALE AND TRAIN
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Gradient-boosted trees trained on the balanced training split.
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train_balanced)

# ==========================================
# STEP 5: EVALUATE AND OVERWRITE SAVED FILES
# ==========================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- HINTSight Module 5: UPGRADED Evaluation Report ---")
print(classification_report(y_test, y_pred))
print(f" New Area Under the ROC Curve (AUROC): {roc_auc_score(y_test, y_prob):.4f}\n")

# Overwrite with your high-performing model artifacts
with open("hintsight_stroke_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("hintsight_scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print(" Success! High-performance model and scaler updated.")


# ==========================================
# STEP 6: PLOT PERFORMANCE METRICS & CHARTS (SPREAD OUT)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score

print("\n Generating diagnostic performance graphs...")

# Compute metrics dynamically for the curves
trees_range = np.arange(10, 151, 10)
train_losses = []
test_accuracies = []

for num_trees in trees_range:
    sub_model = XGBClassifier(
        n_estimators=num_trees,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    sub_model.fit(X_train_scaled, y_train_balanced)
    
    sub_train_probs = sub_model.predict_proba(X_train_scaled)
    sub_test_preds = sub_model.predict(X_test_scaled)
    
    train_losses.append(log_loss(y_train_balanced, sub_train_probs))
    test_accuracies.append(accuracy_score(y_test, sub_test_preds) * 100.0) 

final_acc = accuracy_score(y_test, y_pred) * 100.0

# Set global visual style for professional charts
sns.set_theme(style="whitegrid", palette="muted")

# Setup a wider canvas with extra height to guarantee zero overlap
plt.figure(figsize=(20, 6))

# Graph 1: Training Loss Curve (Corrected to Trees)
plt.subplot(1, 3, 1)
plt.plot(trees_range, train_losses, color='red', linewidth=2)
plt.title("Training Loss over Number of Trees", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Number of Estimators (Trees)", fontsize=12, labelpad=10)
plt.ylabel("Log Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Graph 2: Testing Accuracy Curve (Corrected to Trees)
plt.subplot(1, 3, 2)
plt.plot(trees_range, test_accuracies, color='blue', linewidth=2)
plt.title("Test Accuracy over Number of Trees", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Number of Estimators (Trees)", fontsize=12, labelpad=10)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Graph 3: Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={'size': 16, 'weight': 'bold'},
            xticklabels=['Perfect VOR (0)', 'Deficit (1)'], 
            yticklabels=['Perfect VOR (0)', 'Deficit (1)'])
plt.title(f"Confusion Matrix (Acc: {final_acc:.2f}%)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
plt.ylabel("True Label", fontsize=12, labelpad=10)

# --- THE ABSOLUTE FIX FOR THE SQUISHING ---
# By calling subplots_adjust with explicitly massive horizontal spacing (wspace),
# we hard-force the charts to move away from each other on the canvas.
plt.subplots_adjust(wspace=0.4, top=0.85, bottom=0.15, left=0.08, right=0.95)

# Save with tight bounding parameters so no labels are chopped off by the margins
plt.savefig('ISEF_Performance_Graphs.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n The 3-panel presentation graph has been saved as 'ISEF_Performance_Graphs.png'!")

# ==========================================
# STEP 7: FEATURE INFLUENCE GRAPH
# ==========================================
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
factor_names = {
    'gender': 'Gender',
    'age': 'Age',
    'hypertension': 'Hypertension',
    'heart_disease': 'Heart Disease',
    'ever_married': 'Ever Married',
    'work_type': 'Work Type',
    'Residence_type': 'Residence Type',
    'avg_glucose_level': 'Average Glucose',
    'bmi': 'BMI',
    'smoking_status': 'Smoking Status',
}

factor_importance = {}
for encoded_name, importance in feature_importance.items():
    factor_name = next(
        (name for name in factor_names if encoded_name == name or encoded_name.startswith(f'{name}_')),
        encoded_name,
    )
    factor_importance[factor_name] = factor_importance.get(factor_name, 0) + importance
feature_importance = pd.Series(factor_importance).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(
    [factor_names.get(name, name) for name in feature_importance.index],
    feature_importance.values * 100,
    color=sns.color_palette('muted', len(feature_importance)),
)
plt.title('Factors Influencing Stroke Prediction', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('XGBoost Feature Importance (%)', fontsize=12)
plt.ylabel('Clinical Risk Factor', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.35)
plt.xlim(0, max(feature_importance.values * 100) * 1.18)

for bar, importance in zip(bars, feature_importance.values * 100):
    plt.text(
        bar.get_width() + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f'{importance:.1f}',
        va='center',
        fontsize=10,
    )

plt.tight_layout()
plt.savefig('HINTSight_Metadata_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved feature influence graph to 'HINTSight_Metadata_Feature_Importance.png'!")

# ==========================================
# STEP 8: LEAVE-ONE-FACTOR-OUT ABLATION
# ==========================================
def train_and_score(features):
    X_train_ablation, X_test_ablation, y_train_ablation, y_test_ablation = train_test_split(
        features, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_ablation, y_train_ablation = SMOTE(
        sampling_strategy=1.0, random_state=42
    ).fit_resample(X_train_ablation, y_train_ablation)
    ablation_scaler = StandardScaler().fit(X_train_ablation)
    ablation_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    ablation_model.fit(
        ablation_scaler.transform(X_train_ablation), y_train_ablation
    )
    predictions = ablation_model.predict(ablation_scaler.transform(X_test_ablation))
    return f1_score(y_test_ablation, predictions, pos_label=1)


baseline_stroke_f1 = f1_score(y_test, y_pred, pos_label=1)
ablation_f1_scores = {'None': baseline_stroke_f1}
for factor in predictor_columns:
    encoded_columns = [
        column for column in X.columns
        if column == factor or column.startswith(f'{factor}_')
    ]
    ablation_f1_scores[factor_names.get(factor, factor)] = train_and_score(
        X.drop(columns=encoded_columns)
    )

ablation_f1_scores = pd.Series(ablation_f1_scores) * 100
most_influential_factor = ablation_f1_scores.drop('None').idxmin()
print(
    f"\nMost influential factor by lowest leave-one-out Stroke F1: "
    f"{most_influential_factor} ({ablation_f1_scores[most_influential_factor]:.1f}%)"
)

plt.figure(figsize=(10, 6))
bars = plt.bar(
    ablation_f1_scores.index,
    ablation_f1_scores.values,
    color=sns.color_palette('muted', len(ablation_f1_scores)),
)
plt.title('Different Factors Performance', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Excluded Dataset Factor', fontsize=12)
plt.ylabel('Stroke F1 Score (%)', fontsize=12)
plt.setp(plt.gca().get_xticklabels(), rotation=35, ha='right')
plt.ylim(0, max(ablation_f1_scores.values) * 1.18)
plt.grid(axis='y', linestyle='--', alpha=0.35)

for bar, score in zip(bars, ablation_f1_scores.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{score:.0f}',
        va='bottom',
        ha='center',
        fontsize=10,
    )

plt.tight_layout()
plt.savefig('HINTSight_Metadata_Factor_Ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved F1 leave-one-factor-out graph to 'HINTSight_Metadata_Factor_Ablation.png'!")