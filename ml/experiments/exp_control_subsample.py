"""
Expérience #2 ter — Test de contrôle : volume vs signal patients récurrents
 le gain observé sur
"toutes les rencontres + split groupé" (~0.645 AUC-ROC en moyenne sur 5
plis) pourrait venir simplement du fait qu'on a PLUS de données (99 343
lignes contre 69 990 dans la version dédupliquée actuelle), et pas d'un
vrai signal apporté par les rencontres répétées d'un même patient.

Ce script réentraîne le modèle sur un SOUS-ÉCHANTILLON du grand jeu
(toutes rencontres), ramené à ~56 000 lignes — la même taille que le jeu
de production actuel — avec split toujours groupé par patient_nbr.

Interprétation :
  - AUC reste proche de ~0.64        → le gain vient du signal des
                                        patients récurrents (le modèle
                                        apprend de leur historique, pas
                                        juste "plus de lignes")
  - AUC retombe vers ~0.615 (baseline) → le gain venait surtout du volume

Répété sur plusieurs graines aléatoires pour ne pas conclure sur un seul
tirage de sous-échantillon.


Lancer avec :
  docker-compose exec ml-service python experiments/exp_control_subsample.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
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

TARGET_N = 56000       # taille du jeu de production actuel (donnée par Madame)
N_REPEATS = 3           # plusieurs tirages pour ne pas conclure sur un seul


def clean_data_no_dedup(df: pd.DataFrame) -> pd.DataFrame:
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

    logger.info(f"  → {len(df_clean)} rencontres disponibles (toutes) | "
                 f"sous-échantillonnage à {TARGET_N} lignes, {N_REPEATS} répétitions")

    repeats = []
    for i, seed in enumerate([42, 7, 123][:N_REPEATS]):
        df_sub = df_clean.sample(n=TARGET_N, random_state=seed).reset_index(drop=True)
        prevalence = df_sub["readmitted_30"].mean()
        logger.info(f"── Répétition {i} (seed={seed}) : {len(df_sub)} lignes, "
                     f"{df_sub['patient_nbr'].nunique()} patients uniques, "
                     f"prévalence={prevalence:.2%} ──")

        y_sub = df_sub["readmitted_30"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(df_sub, y_sub, groups=df_sub["patient_nbr"]))

        patients_train = set(df_sub.iloc[train_idx]["patient_nbr"])
        patients_test  = set(df_sub.iloc[test_idx]["patient_nbr"])
        overlap = patients_train & patients_test

        r = train_and_eval(
            df_sub.iloc[train_idx], df_sub.iloc[test_idx],
            y_sub[train_idx], y_sub[test_idx],
        )
        r["seed"] = seed
        r["overlap_check"] = len(overlap)
        r["prevalence"] = round(float(prevalence), 4)
        repeats.append(r)
        logger.info(f"    AUC-ROC={r['auc_roc']} | AUC-PR={r['auc_pr']} | "
                     f"chevauchement={len(overlap)} | train={r['n_train']} | test={r['n_test']}")

    aucs = np.array([r["auc_roc"] for r in repeats])
    summary = {
        "target_n": TARGET_N,
        "repeats": repeats,
        "auc_roc_mean": round(float(aucs.mean()), 4),
        "auc_roc_std":  round(float(aucs.std()), 4),
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"AUC-ROC moyen sur {N_REPEATS} répétitions (n={TARGET_N}) : "
                 f"{summary['auc_roc_mean']} ± {summary['auc_roc_std']}")
    logger.info("Rappel — baseline actuelle (dédup, split aléatoire) : AUC-ROC ≈ 0.60-0.61")
    logger.info("Rappel — toutes rencontres, GroupKFold 5 plis (99 343 lignes) : AUC-ROC = 0.6451 ± 0.0102")
    if summary["auc_roc_mean"] >= 0.63:
        logger.info("→ L'AUC reste élevée malgré la réduction de taille : "
                     "le gain vient bien du signal des patients récurrents.")
    elif summary["auc_roc_mean"] <= 0.62:
        logger.info("→ L'AUC retombe proche de la baseline : "
                     "le gain observé venait surtout du volume de données.")
    else:
        logger.info("→ Résultat intermédiaire, à interpréter avec prudence.")
    logger.info(f"{'='*55}")

    os.makedirs("/app/experiments", exist_ok=True)
    with open("/app/experiments/exp_control_subsample_results.json", "w") as fjson:
        json.dump(summary, fjson, indent=2, ensure_ascii=False)
    logger.info("✓ Résultats sauvegardés dans experiments/exp_control_subsample_results.json")