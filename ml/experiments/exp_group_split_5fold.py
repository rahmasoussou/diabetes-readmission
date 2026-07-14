"""
Expérience #2 bis — GroupKFold à 5 plis (stabilité du gain)

Suite demandée après exp_group_split.py : le gain observé sur UN SEUL
split groupé (+0.034 AUC-ROC) pourrait être un effet de tirage favorable.
Ce script répète la comparaison sur 5 plis (GroupKFold, toujours groupé
par patient_nbr — un même patient n'est jamais à cheval entre deux plis)
et rapporte la moyenne ± écart-type pour vérifier que le gain est stable.

Réutilise exactement le même feature engineering, les mêmes hyperparamètres
et la même source de données que exp_group_split.py, pour une comparaison
juste. Pas de calibration ni de seuil ici (hors périmètre du test, comme
dans exp_group_split.py) : on ne touche pas à train.py, features.py, ni
etl/pipeline.py de production.

Lancer avec :
  docker-compose exec ml-service python experiments/exp_group_split_5fold.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from dotenv import load_dotenv

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/etl")
from features import MED_COLS, NUMERIC_FEATURES, _diag_group
from pipeline import engineer_features, MED_RENAME  # réutilise le vrai code ETL

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NOMINAL_COLS = ["race", "gender", "a1c_result", "glucose_serum",
                "change_in_meds", "diabetes_meds"]

XGB_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.03,
    subsample=0.75, colsample_bytree=0.75, colsample_bylevel=0.75,
    min_child_weight=10, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="aucpr", early_stopping_rounds=50,
    random_state=42, verbosity=0,
)


def clean_data_no_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Identique à exp_group_split.py — pas de déduplication, on garde
    toutes les rencontres."""
    df = df.replace("?", np.nan)
    cols_to_drop = ["weight", "payer_code", "medical_specialty", "examide", "citoglipton"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    if "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    return df


def rename_for_features(df: pd.DataFrame) -> pd.DataFrame:
    col_rename = {
        "A1Cresult": "a1c_result", "max_glu_serum": "glucose_serum",
        "change": "change_in_meds", "diabetesMed": "diabetes_meds",
        "number_outpatient": "num_outpatient",
        "number_emergency": "num_emergency",
        "number_inpatient": "num_inpatient",
    }
    return df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})


def build_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Identique à exp_group_split.py."""
    f = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        f[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    med_active_count = pd.Series(0, index=df.index)
    for col in MED_COLS:
        if col in df.columns:
            vals = df[col].fillna("No").astype(str)
            med_active_count += (~vals.isin(["No"])).astype(int)
    f["n_active_meds"] = med_active_count

    if "insulin" in df.columns:
        ins = df["insulin"].fillna("No").astype(str)
        f["insulin_active"] = (ins != "No").astype(int)
        f["insulin_up"]     = (ins == "Up").astype(int)

    for d in ["diag_1", "diag_2", "diag_3"]:
        col = d + "_group"
        f[col] = df[d].apply(_diag_group) if d in df.columns else 0
    f["diag1_diabetes"]  = (f["diag_1_group"] == 8).astype(int)
    f["has_circulatory"] = ((f["diag_1_group"] == 1) | (f["diag_2_group"] == 1) | (f["diag_3_group"] == 1)).astype(int)

    f["a1c_elevated"] = df.get("a1c_result", "None").astype(str).isin([">7", ">8"]).astype(int)
    f["high_history"] = ((f["num_inpatient"] >= 2) | (f["num_emergency"] >= 1)).astype(int)
    f["complexity_score"]   = f["number_diagnoses"] * f["num_medications"] / 100.0
    f["procedures_per_day"] = (f["num_procedures"] + f["num_lab_procedures"]) / (f["time_in_hospital"] + 1)
    f["age_x_inpatient"]    = f["age_num"] * f["num_inpatient"] / 100.0
    f["frequent_flyer"]     = ((f["num_inpatient"] >= 3) | (f["num_emergency"] >= 2)).astype(int)
    f["long_stay"]          = (f["time_in_hospital"] > 7).astype(int)
    med_instab = (df.get("change_in_meds", "No").astype(str) == "Ch") & \
                 (df.get("diabetes_meds", "No").astype(str) == "Yes")
    f["med_instability"]   = med_instab.astype(int)
    f["meds_per_diagnosis"] = f["num_medications"] / (f["number_diagnoses"] + 1)
    f["risk_composite"] = (
        f["high_history"] * 3 + f["a1c_elevated"] * 2 + f["frequent_flyer"] * 2 +
        f["long_stay"] + f["med_instability"] + f.get("insulin_active", 0) + f["has_circulatory"]
    )
    return f


def build_features_label(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    f = build_common_features(df)
    for col in NOMINAL_COLS:
        le = encoders[col]
        vals = df[col].fillna("No").astype(str)
        fallback = "No" if "No" in le.classes_ else le.classes_[0]
        vals = vals.apply(lambda x: x if x in le.classes_ else fallback)
        f[col + "_enc"] = le.transform(vals)
    return f


def train_and_eval(df_train, df_test, y_train, y_test) -> dict:
    """Identique à exp_group_split.py : LabelEncoder ajusté sur le train
    du pli uniquement (pas de fuite d'info entre plis)."""
    encoders = {}
    for col in NOMINAL_COLS:
        le = LabelEncoder()
        le.fit(df_train[col].fillna("No").astype(str))
        encoders[col] = le
    X_train = build_features_label(df_train, encoders)
    X_test  = build_features_label(df_test, encoders)

    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
        "auc_pr":  round(float(average_precision_score(y_test, proba)), 4),
        "n_train": len(X_train), "n_test": len(X_test),
    }


if __name__ == "__main__":
    RAW_PATH = "/app/data/raw/diabetic_data.csv"
    if not os.path.exists(RAW_PATH):
        logger.error(f"Fichier introuvable : {RAW_PATH}")
        sys.exit(1)

    logger.info(f"Chargement des données brutes : {RAW_PATH}")
    df_raw = pd.read_csv(RAW_PATH, low_memory=False)

    df_clean = clean_data_no_dedup(df_raw)
    df_clean = engineer_features(df_clean)
    df_clean = rename_for_features(df_clean)
    for d in ["diag_1", "diag_2", "diag_3"]:
        if d in df_clean.columns:
            df_clean[d] = df_clean[d].fillna("Unknown").astype(str).str.strip()

    n_patients_uniques = df_clean["patient_nbr"].nunique()
    logger.info(f"  → {len(df_clean)} rencontres après nettoyage | {n_patients_uniques} patients uniques")

    y_all = df_clean["readmitted_30"].values
    groups = df_clean["patient_nbr"].values

    logger.info("── GroupKFold à 5 plis — toutes rencontres, split groupé par patient ──")
    gkf = GroupKFold(n_splits=5)
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df_clean, y_all, groups)):
        # Contrôle anti-fuite : aucun patient à cheval entre train et test du pli
        patients_train = set(groups[train_idx])
        patients_test  = set(groups[test_idx])
        overlap = patients_train & patients_test
        assert len(overlap) == 0, f"fuite de patients détectée au pli {fold_i} : {len(overlap)} patients en commun"

        r = train_and_eval(
            df_clean.iloc[train_idx], df_clean.iloc[test_idx],
            y_all[train_idx], y_all[test_idx],
        )
        r["fold"] = fold_i
        r["overlap_check"] = len(overlap)
        fold_results.append(r)
        logger.info(f"  Pli {fold_i}: AUC-ROC={r['auc_roc']} | AUC-PR={r['auc_pr']} | "
                     f"train={r['n_train']} | test={r['n_test']} | chevauchement={len(overlap)}")

    aucs = np.array([r["auc_roc"] for r in fold_results])
    aps  = np.array([r["auc_pr"] for r in fold_results])

    summary = {
        "folds": fold_results,
        "auc_roc_mean": round(float(aucs.mean()), 4),
        "auc_roc_std":  round(float(aucs.std()), 4),
        "auc_pr_mean":  round(float(aps.mean()), 4),
        "auc_pr_std":   round(float(aps.std()), 4),
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"AUC-ROC moyen sur 5 plis : {summary['auc_roc_mean']} ± {summary['auc_roc_std']}")
    logger.info(f"AUC-PR moyen sur 5 plis  : {summary['auc_pr_mean']} ± {summary['auc_pr_std']}")
    logger.info("Compare cette moyenne au résultat de exp_group_split.py (split unique) "
                 "pour juger si le gain est stable ou tenait au tirage.")
    logger.info(f"{'='*55}")

    os.makedirs("/app/experiments", exist_ok=True)
    with open("/app/experiments/exp_group_split_5fold_results.json", "w") as fjson:
        json.dump(summary, fjson, indent=2, ensure_ascii=False)
    logger.info("✓ Résultats sauvegardés dans experiments/exp_group_split_5fold_results.json")
