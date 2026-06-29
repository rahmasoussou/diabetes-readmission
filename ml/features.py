"""
Feature Engineering — ML Pipeline (v3)
========================================
v3 : utilise les médicaments (21 colonnes) + diagnostics ICD-9
     encodés en groupes cliniques → AUC cible 0.68+
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ─── Médicaments ──────────────────────────────────────────────────
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide_metformin", "glipizide_metformin",
    "glimepiride_pioglitazone", "metformin_rosiglitazone",
    "metformin_pioglitazone",
]

# ─── Features numériques brutes ───────────────────────────────────
NUMERIC_FEATURES = [
    "age_num", "time_in_hospital", "num_medications",
    "num_lab_procedures", "num_procedures", "number_diagnoses",
    "num_emergency", "num_inpatient", "num_outpatient",
    "meds_per_day", "total_visits",
]

# ─── Features catégorielles ───────────────────────────────────────
CATEGORICAL_FEATURES = [
    "race", "gender", "a1c_result", "glucose_serum",
    "change_in_meds", "diabetes_meds",
] + MED_COLS


def _diag_group(code: str) -> int:
    """
    Convertit un code ICD-9 en groupe clinique (0-9).
    Basé sur la littérature sur la réhospitalisation diabétique.
    """
    if not code or code in ("Unknown", "nan", ""):
        return 0
    code = str(code).strip()
    try:
        if code.startswith("E") or code.startswith("V"):
            return 9
        n = float(code)
        if 390 <= n <= 459 or n == 785:   return 1  # Circulatoire
        if 460 <= n <= 519 or n == 786:   return 2  # Respiratoire
        if 520 <= n <= 579 or n == 787:   return 3  # Digestif
        if 800 <= n <= 999:               return 4  # Traumatisme
        if 710 <= n <= 739:               return 5  # Musculo-squelettique
        if 580 <= n <= 629 or n == 788:   return 6  # Génito-urinaire
        if 140 <= n <= 239:               return 7  # Néoplasme
        if 250 <= n <= 250.99:            return 8  # Diabète
        return 0
    except (ValueError, TypeError):
        return 0


def fit_encoders(df: pd.DataFrame) -> dict:
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].fillna("No").astype(str))
            encoders[col] = le
    return encoders


def build_features(df: pd.DataFrame, encoders: dict = None) -> pd.DataFrame:
    features = pd.DataFrame()

    # ── 1. Numériques brutes ──────────────────────────────────────
    for col in NUMERIC_FEATURES:
        features[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # ── 2. Catégorielles encodées ─────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and encoders and col in encoders:
            le   = encoders[col]
            vals = df[col].fillna("No").astype(str)
            vals = vals.apply(lambda x: x if x in le.classes_ else "No")
            features[col + "_enc"] = le.transform(vals)
        else:
            features[col + "_enc"] = 0

    # ── 3. Features médicaments agrégées ─────────────────────────
    # Compter les médicaments actifs (Up/Down/Steady ≠ No)
    med_enc_cols = [c + "_enc" for c in MED_COLS if c + "_enc" in features.columns]
    features["n_active_meds"] = sum(
        (features[c] > 0).astype(int) for c in med_enc_cols
    )

    # Médicaments augmentés (Up)
    for col in MED_COLS:
        enc_col = col + "_enc"
        if enc_col in features.columns and encoders and col in encoders:
            le = encoders[col]
            up_idx = list(le.classes_).index("Up") if "Up" in le.classes_ else -1
            features[col + "_up"] = (features[enc_col] == up_idx).astype(int)

    # Insulin spécifiquement (très prédictif)
    if "insulin_enc" in features.columns and encoders and "insulin" in encoders:
        le = encoders["insulin"]
        features["insulin_active"] = (features["insulin_enc"] > 0).astype(int)
        up_idx = list(le.classes_).index("Up") if "Up" in le.classes_ else -1
        features["insulin_up"] = (features["insulin_enc"] == up_idx).astype(int)

    # ── 4. Diagnostics ICD-9 ─────────────────────────────────────
    for d in ["diag_1", "diag_2", "diag_3"]:
        col = d + "_group"
        if d in df.columns:
            features[col] = df[d].apply(_diag_group)
        else:
            features[col] = 0

    # Diabète comme diagnostic principal
    features["diag1_diabetes"] = (features["diag_1_group"] == 8).astype(int)
    # Maladie cardiovasculaire (fort prédicteur de réhospitalisation)
    features["has_circulatory"] = (
        (features["diag_1_group"] == 1) |
        (features["diag_2_group"] == 1) |
        (features["diag_3_group"] == 1)
    ).astype(int)

    # ── 5. Features dérivées cliniques ───────────────────────────
    features["high_history"] = (
        (features["num_inpatient"] >= 2) | (features["num_emergency"] >= 1)
    ).astype(int)

    if "a1c_result_enc" in features.columns:
        features["a1c_elevated"] = (features["a1c_result_enc"] >= 2).astype(int)

    features["complexity_score"] = (
        features["number_diagnoses"] * features["num_medications"] / 100.0
    )
    features["procedures_per_day"] = (
        (features["num_procedures"] + features["num_lab_procedures"])
        / (features["time_in_hospital"] + 1)
    )
    features["age_x_inpatient"] = features["age_num"] * features["num_inpatient"] / 100.0
    features["frequent_flyer"]  = (
        (features["num_inpatient"] >= 3) | (features["num_emergency"] >= 2)
    ).astype(int)
    features["long_stay"] = (features["time_in_hospital"] > 7).astype(int)

    if "change_in_meds_enc" in features.columns and "diabetes_meds_enc" in features.columns:
        features["med_instability"] = (
            (features["change_in_meds_enc"] >= 1) &
            (features["diabetes_meds_enc"] >= 1)
        ).astype(int)

    features["meds_per_diagnosis"] = (
        features["num_medications"] / (features["number_diagnoses"] + 1)
    )

    # Score composite enrichi
    a1c_e = features.get("a1c_elevated", 0)
    med_i = features.get("med_instability", 0)
    features["risk_composite"] = (
        features["high_history"] * 3 +
        a1c_e * 2 +
        features["frequent_flyer"] * 2 +
        features["long_stay"] +
        med_i +
        features.get("insulin_active", 0) +
        features["has_circulatory"]
    )

    return features


def save_artifacts(encoders: dict, feature_names: list, path: str = "/app/models") -> None:
    os.makedirs(path, exist_ok=True)
    joblib.dump(encoders,      os.path.join(path, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(path, "feature_names.pkl"))


def load_artifacts(path: str = "/app/models") -> tuple:
    encoders      = joblib.load(os.path.join(path, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(path, "feature_names.pkl"))
    return encoders, feature_names