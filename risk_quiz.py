"""
Converts clinical risk-factor quiz answers into the feature vector expected by
the metadata branch of the HINTSight late-fusion model (same column order
used in trainRiskFactors.py / train_fusion_model.py).
"""

import numpy as np

# Order must match the encoded metadata columns in train_fusion_model.py.
FEATURE_ORDER = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]


def quiz_to_features(answers: dict) -> np.ndarray:
    """
    answers keys:
      gender: "male" | "female"
      age: float (years)
      hypertension: bool
      heart_disease: bool
    ever_married: bool
    work_type: "Private" | "Self-employed" | "Govt_job" | "children" | "Never_worked"
    Residence_type: "Urban" | "Rural"
    avg_glucose_level: float (mg/dL)
      bmi: float
    smoking_status: "never smoked" | "formerly smoked" | "smokes" | "Unknown"
    """
    required = set(FEATURE_ORDER)
    missing = required - answers.keys()
    if missing:
        raise ValueError(f"Missing quiz answers: {sorted(missing)}")

    gender = 1 if str(answers["gender"]).lower().startswith("m") else 0
    ever_married = 1 if bool(answers["ever_married"]) else 0
    work_type_codes = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
    residence_type = 1 if str(answers["Residence_type"]).lower() == "urban" else 0
    smoking_codes = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}

    row = [
        gender,
        float(answers["age"]),
        int(bool(answers["hypertension"])),
        int(bool(answers["heart_disease"])),
        ever_married,
        work_type_codes[str(answers["work_type"])],
        residence_type,
        float(answers["avg_glucose_level"]),
        float(answers["bmi"]),
        smoking_codes[str(answers["smoking_status"])],
    ]
    return np.array(row, dtype=np.float32).reshape(1, -1)
