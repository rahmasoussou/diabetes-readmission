"""
Tests unitaires — features.py (feature engineering)
======================================================
Lancer avec : pytest tests/test_features.py -v
"""

import sys
import pandas as pd
import pytest

sys.path.insert(0, "/app")

from features import (
    _diag_group, fit_encoders, build_features,
    MED_COLS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)


# ─── _diag_group ────────────────────────────────────────────────
@pytest.mark.parametrize("code,expected", [
    ("410", 1),      # circulatoire
    ("785", 1),      # circulatoire (code symptôme)
    ("486", 2),      # respiratoire
    ("530", 3),      # digestif
    ("820", 4),      # traumatisme
    ("715", 5),      # musculo-squelettique
    ("590", 6),      # génito-urinaire
    ("150", 7),      # néoplasme
    ("250.02", 8),   # diabète
    ("E950", 9),     # code E
    ("V10", 9),       # code V
    ("Unknown", 0),
    ("", 0),
    (None, 0),
    ("abc", 0),       # non convertible -> 0, ne doit pas lever d'exception
])
def test_diag_group(code, expected):
    assert _diag_group(code) == expected


# ─── fit_encoders / build_features ─────────────────────────────
@pytest.fixture
def sample_df():
    n = 20
    data = {
        "age_num": [65] * n,
        "time_in_hospital": [5] * n,
        "num_medications": [10] * n,
        "num_lab_procedures": [30] * n,
        "num_procedures": [1] * n,
        "number_diagnoses": [5] * n,
        "num_emergency": [0] * n,
        "num_inpatient": [1] * n,
        "num_outpatient": [0] * n,
        "meds_per_day": [2.0] * n,
        "total_visits": [1] * n,
        "race": ["Caucasian"] * n,
        "gender": ["Female"] * n,
        "a1c_result": ["Norm"] * n,
        "glucose_serum": ["None"] * n,
        "change_in_meds": ["No"] * n,
        "diabetes_meds": ["Yes"] * n,
        "diag_1": ["410"] * n,
        "diag_2": ["Unknown"] * n,
        "diag_3": ["Unknown"] * n,
    }
    for m in MED_COLS:
        data[m] = ["No"] * n
    return pd.DataFrame(data)


def test_fit_encoders_covers_categorical_columns(sample_df):
    encoders = fit_encoders(sample_df)
    for col in CATEGORICAL_FEATURES:
        if col in sample_df.columns:
            assert col in encoders


def test_build_features_shape_and_no_nans(sample_df):
    encoders = fit_encoders(sample_df)
    features = build_features(sample_df, encoders)
    assert len(features) == len(sample_df)
    assert not features.isnull().values.any()


def test_build_features_numeric_columns_present(sample_df):
    encoders = fit_encoders(sample_df)
    features = build_features(sample_df, encoders)
    for col in NUMERIC_FEATURES:
        assert col in features.columns


def test_build_features_unseen_category_falls_back_to_no(sample_df):
    """Une catégorie jamais vue à l'entraînement doit être gérée sans lever d'exception."""
    encoders = fit_encoders(sample_df)
    df_new = sample_df.copy()
    df_new.loc[0, "race"] = "CategorieJamaisVue"
    features = build_features(df_new, encoders)
    assert len(features) == len(df_new)


def test_build_features_without_encoders_returns_zeros(sample_df):
    """Sans encoders (ex: premier appel), les colonnes catégorielles encodées valent 0."""
    features = build_features(sample_df, encoders=None)
    for col in CATEGORICAL_FEATURES:
        enc_col = col + "_enc"
        if enc_col in features.columns:
            assert (features[enc_col] == 0).all()


def test_risk_composite_is_non_negative(sample_df):
    encoders = fit_encoders(sample_df)
    features = build_features(sample_df, encoders)
    assert (features["risk_composite"] >= 0).all()


def test_diag1_diabetes_flag(sample_df):
    df = sample_df.copy()
    df["diag_1"] = "250.01"  # diabète
    encoders = fit_encoders(df)
    features = build_features(df, encoders)
    assert (features["diag1_diabetes"] == 1).all()