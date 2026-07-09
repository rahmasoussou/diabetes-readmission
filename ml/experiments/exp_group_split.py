"""
Expérience #2 — Toutes les rencontres + split groupé par patient
====================================================================
 au lieu de ne garder qu'une rencontre par
patient (la déduplication actuelle dans etl/pipeline.py, qui jette
beaucoup de données), on peut garder TOUTES les rencontres — à condition
de s'assurer qu'un même patient n'est jamais à la fois dans le train et
dans le test (sinon le modèle "triche" en ayant déjà vu ce patient).

Ce script compare, à hyperparamètres strictement identiques :
  (A) baseline actuelle  : 1 rencontre/patient (déduplication ETL),
                           split aléatoire classique (comme train.py)
  (B) alternative testée : TOUTES les rencontres, split par GROUPE de
                           patient_nbr (aucun patient à cheval entre
                           train et test)

Simplification assumée et documentée : on fait un seul split groupé
(GroupShuffleSplit 80/20), pas un GroupKFold à 5 plis complet, pour
rester cohérent avec le protocole de l'expérience #1 (même logique de
comparaison A/B) et limiter le temps de calcul. Une vraie validation
croisée groupée serait l'étape suivante si ce premier test est concluant.

Ceci est une EXPÉRIENCE, pas un correctif : ne touche pas à train.py, à
features.py, ni à etl/pipeline.py de production.

Lancer avec :
  docker-compose exec ml-service python experiments/exp_group_split.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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


def get_engine():
    u = os.environ["POSTGRES_USER"]; p = os.environ["POSTGRES_PASSWORD"]
    h = os.environ["POSTGRES_HOST"]; d = os.environ["POSTGRES_DB"]
    return create_engine(f"postgresql://{u}:{p}@{h}/{d}")


def clean_data_no_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Identique à etl.pipeline.clean_data(), MOINS la ligne de
    déduplication (df.drop_duplicates(subset='patient_nbr', keep='first')).
    On garde ainsi toutes les rencontres de chaque patient."""
    df = df.replace("?", np.nan)
    cols_to_drop = ["weight", "payer_code", "medical_specialty", "examide", "citoglipton"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    if "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
    # PAS de drop_duplicates ici — c'est tout l'objet du test
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    return df


def rename_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes brutes du CSV vers les noms attendus par
    features.py (mêmes noms que ceux utilisés en base par l'ETL)."""
    col_rename = {
        "A1Cresult": "a1c_result", "max_glu_serum": "glucose_serum",
        "change": "change_in_meds", "diabetesMed": "diabetes_meds",
        "number_outpatient": "num_outpatient",
        "number_emergency": "num_emergency",
        "number_inpatient": "num_inpatient",
    }
    return df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})


def build_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Identique à exp_encoding.py — features numériques + dérivées,
    indépendantes du schéma d'encodage ou du mode de split."""
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
    logger.info(f"  → {len(df_raw)} rencontres brutes (toutes, avant dédup)")

    df_clean = clean_data_no_dedup(df_raw)
    df_clean = engineer_features(df_clean)   # vraie fonction ETL, réutilisée telle quelle
    df_clean = rename_for_features(df_clean)
    for d in ["diag_1", "diag_2", "diag_3"]:
        if d in df_clean.columns:
            df_clean[d] = df_clean[d].fillna("Unknown").astype(str).str.strip()

    n_patients_uniques = df_clean["patient_nbr"].nunique()
    logger.info(f"  → {len(df_clean)} rencontres après nettoyage | {n_patients_uniques} patients uniques")

    results = {}

    # ── (A) Baseline actuelle : 1 rencontre/patient, split aléatoire ──
    logger.info("── Version A : dédup actuelle (1 rencontre/patient) + split aléatoire ──")
    df_dedup = df_clean.drop_duplicates(subset="patient_nbr", keep="first")
    y_dedup = df_dedup["readmitted_30"].values
    train_idx, test_idx = train_test_split(
        np.arange(len(df_dedup)), test_size=0.2, random_state=42, stratify=y_dedup
    )
    res_a = train_and_eval(
        df_dedup.iloc[train_idx], df_dedup.iloc[test_idx],
        y_dedup[train_idx], y_dedup[test_idx],
    )
    results["dedup_random_split"] = res_a
    logger.info(f"  AUC-ROC={res_a['auc_roc']} | AUC-PR={res_a['auc_pr']} | train={res_a['n_train']} | test={res_a['n_test']}")

    # ── (B) Testée : toutes rencontres, split GROUPÉ par patient ─────
    logger.info("── Version B : toutes les rencontres + split groupé par patient_nbr ──")
    y_all = df_clean["readmitted_30"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx2, test_idx2 = next(gss.split(df_clean, y_all, groups=df_clean["patient_nbr"]))

    # Vérification de non-fuite : aucun patient à cheval entre train et test
    patients_train = set(df_clean.iloc[train_idx2]["patient_nbr"])
    patients_test  = set(df_clean.iloc[test_idx2]["patient_nbr"])
    overlap = patients_train & patients_test
    logger.info(f"  → Chevauchement de patients train/test : {len(overlap)} (doit être 0)")

    res_b = train_and_eval(
        df_clean.iloc[train_idx2], df_clean.iloc[test_idx2],
        y_all[train_idx2], y_all[test_idx2],
    )
    results["all_encounters_group_split"] = res_b
    results["all_encounters_group_split"]["n_patients_overlap_check"] = len(overlap)
    logger.info(f"  AUC-ROC={res_b['auc_roc']} | AUC-PR={res_b['auc_pr']} | train={res_b['n_train']} | test={res_b['n_test']}")

    # ── Conclusion ────────────────────────────────────────────────────
    delta_auc = res_b["auc_roc"] - res_a["auc_roc"]
    delta_ap  = res_b["auc_pr"]  - res_a["auc_pr"]
    results["conclusion"] = {
        "delta_auc_roc": round(delta_auc, 4),
        "delta_auc_pr":  round(delta_ap, 4),
        "verdict": (
            "Toutes les rencontres + split groupé améliore" if delta_auc > 0.005 else
            "Dédup actuelle reste préférable" if delta_auc < -0.005 else
            "Différence négligeable — pas d'impact significatif"
        ),
    }
    logger.info(f"\n{'='*55}")
    logger.info(f"Δ AUC-ROC (toutes rencontres − dédup) = {delta_auc:+.4f}")
    logger.info(f"Δ AUC-PR  (toutes rencontres − dédup) = {delta_ap:+.4f}")
    logger.info(f"Verdict : {results['conclusion']['verdict']}")
    logger.info(f"{'='*55}")

    os.makedirs("/app/experiments", exist_ok=True)
    with open("/app/experiments/exp_group_split_results.json", "w") as fjson:
        json.dump(results, fjson, indent=2, ensure_ascii=False)
    logger.info("✓ Résultats sauvegardés dans experiments/exp_group_split_results.json")
