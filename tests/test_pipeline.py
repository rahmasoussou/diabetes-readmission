"""
Tests unitaires — pipeline.py (ETL)
======================================
Lancer avec : pytest tests/test_pipeline.py -v
"""

import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, "/app/etl")

from pipeline import clean_data, engineer_features, MED_RENAME


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "encounter_id": [1, 2, 3, 4],
        "patient_nbr": [100, 100, 200, 300],  # 100 est dupliqué -> déduplication attendue
        "age": ["[60-70)", "[60-70)", "[70-80)", "[0-10)"],
        "time_in_hospital": [5, 5, 3, 1],
        "num_medications": [10, 10, 5, 2],
        "readmitted": ["<30", "<30", "NO", ">30"],
        "discharge_disposition_id": [1, 1, 11, 1],  # 11 = décès -> exclu
        "weight": ["?", "?", "?", "?"],
        "payer_code": ["?", "?", "?", "?"],
        "medical_specialty": ["?", "?", "?", "?"],
        "glyburide-metformin": ["No", "No", "Steady", np.nan],
        "insulin": ["Up", "Up", "No", np.nan],
        "diag_1": ["410", np.nan, "?", "250.01"],
        "number_outpatient": [0, 0, 1, 2],
        "number_emergency": [0, 0, 0, 1],
        "number_inpatient": [1, 1, 0, 0],
    })


def test_clean_data_removes_placeholder_columns(raw_df):
    cleaned = clean_data(raw_df)
    assert "weight" not in cleaned.columns
    assert "payer_code" not in cleaned.columns


def test_clean_data_excludes_deaths(raw_df):
    cleaned = clean_data(raw_df)
    # encounter_id=3 a discharge_disposition_id=11 (décès) -> doit être exclu
    assert 3 not in cleaned["encounter_id"].values


def test_clean_data_keeps_all_encounters(raw_df):
    """v3 : plus de déduplication — toutes les rencontres d'un même patient
    sont conservées. La fuite est empêchée côté ml/train.py (split groupé
    par patient_nbr), pas côté ETL."""
    cleaned = clean_data(raw_df)
    # patient_nbr=100 apparaît deux fois dans raw_df -> les deux doivent être conservées
    assert (cleaned["patient_nbr"] == 100).sum() == 2


def test_clean_data_creates_binary_target(raw_df):
    cleaned = clean_data(raw_df)
    assert "readmitted_30" in cleaned.columns
    assert set(cleaned["readmitted_30"].unique()).issubset({0, 1})


def test_engineer_features_age_mapping(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    assert "age_num" in engineered.columns
    # [60-70) doit être mappé à 65
    row = engineered[engineered["age"] == "[60-70)"]
    if len(row) > 0:
        assert row["age_num"].iloc[0] == 65


def test_engineer_features_meds_per_day_no_division_by_zero(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    assert engineered["meds_per_day"].notnull().all()
    assert np.isfinite(engineered["meds_per_day"]).all()


def test_engineer_features_renames_med_columns(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    for old, new in MED_RENAME.items():
        if old in raw_df.columns:
            assert new in engineered.columns
            assert old not in engineered.columns


def test_engineer_features_fills_medication_nan_with_no(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    assert engineered["insulin"].isnull().sum() == 0


def test_engineer_features_total_visits(raw_df):
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    assert "total_visits" in engineered.columns
    assert (engineered["total_visits"] >= 0).all()
