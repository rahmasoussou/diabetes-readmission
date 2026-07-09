"""
Expérience #1 — LabelEncoder vs One-Hot pour les variables nominales
=======================================================================
 le LabelEncoder actuel impose un ordre
artificiel sur des catégories qui n'en ont pas (ex. race : Caucasian=0,
AfricanAmerican=1, Hispanic=2... le modèle peut "croire" que 2 > 1 > 0
a un sens, alors que ce sont juste des étiquettes).

Ce script compare, à split et hyperparamètres strictement identiques :
  (A) baseline actuelle  : LabelEncoder (comme dans ml/features.py / train.py)
  (B) alternative testée : One-Hot Encoding pour les variables nominales

Variables concernées par le test : race, gender, a1c_result, glucose_serum,
change_in_meds, diabetes_meds (les colonnes médicaments restent identiques
dans les deux versions — un one-hot sur 21 médicaments x ~4 niveaux
ajouterait ~80 colonnes creuses, hors du périmètre de ce test ciblé).

Ceci est une EXPÉRIENCE, pas un correctif : ne touche pas à train.py ni à
features.py de production. Résultat documenté dans experiments/README.md
qu'on que la conclusion soit positive ou négative.

Lancer avec :
  docker-compose exec ml-service python experiments/exp_encoding.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from dotenv import load_dotenv

sys.path.insert(0, "/app")
from features import MED_COLS, NUMERIC_FEATURES, _diag_group

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(INFO)s] %(message)s"
                     .replace("%(INFO)s", "INFO"))
logger = logging.getLogger(__name__)

# Variables nominales concernées par le test (hors colonnes médicaments,
# gérées séparément comme en production)
NOMINAL_COLS = ["race", "gender", "a1c_result", "glucose_serum",
                "change_in_meds", "diabetes_meds"]

# Hyperparamètres IDENTIQUES à ceux de train.py, pour une comparaison juste
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


def build_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features numériques + dérivées CLINIQUES, identiques dans les deux
    versions de l'expérience. Calculées directement sur les colonnes brutes
    (pas sur des colonnes encodées) pour ne dépendre d'aucun choix d'encodage,
    et garantir une comparaison qui isole vraiment l'effet de l'encodage."""
    f = pd.DataFrame()
    for col in NUMERIC_FEATURES:
        f[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # Médicaments (identique à la prod : LabelEncoder, hors périmètre du test)
    med_active_count = pd.Series(0, index=df.index)
    for col in MED_COLS:
        if col in df.columns:
            vals = df[col].fillna("No").astype(str)
            active = ~vals.isin(["No"])
            med_active_count += active.astype(int)
    f["n_active_meds"] = med_active_count

    if "insulin" in df.columns:
        ins = df["insulin"].fillna("No").astype(str)
        f["insulin_active"] = (ins != "No").astype(int)
        f["insulin_up"]     = (ins == "Up").astype(int)

    # Diagnostics ICD-9 (identique à la prod)
    for d in ["diag_1", "diag_2", "diag_3"]:
        col = d + "_group"
        f[col] = df[d].apply(_diag_group) if d in df.columns else 0
    f["diag1_diabetes"]  = (f["diag_1_group"] == 8).astype(int)
    f["has_circulatory"] = ((f["diag_1_group"] == 1) | (f["diag_2_group"] == 1) | (f["diag_3_group"] == 1)).astype(int)

    # Flags cliniques dérivés directement des colonnes BRUTES (pas *_enc) :
    # a1c/glucose ont un vrai ordre clinique (None < Norm < >7/>200 < >8/>300)
    # qu'un LabelEncoder alphabétique ne respecte PAS, donc on le fixe ici
    # explicitement pour que les deux versions de l'expérience partent du
    # même calcul clinique correct.
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
    """Version (A) — baseline actuelle : LabelEncoder sur les nominales."""
    f = build_common_features(df)
    for col in NOMINAL_COLS:
        le = encoders[col]
        vals = df[col].fillna("No").astype(str)
        fallback = "No" if "No" in le.classes_ else le.classes_[0]
        vals = vals.apply(lambda x: x if x in le.classes_ else fallback)
        f[col + "_enc"] = le.transform(vals)
    return f


def build_features_onehot(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """Version (B) — testée : One-Hot sur les nominales.
    handle_unknown='ignore' gère nativement les catégories jamais vues
    (toutes mises à 0), donc pas besoin de fallback manuel ici."""
    f = build_common_features(df)
    raw = df[NOMINAL_COLS].fillna("No").astype(str)
    onehot_arr = ohe.transform(raw)
    onehot_df = pd.DataFrame(
        onehot_arr,
        columns=ohe.get_feature_names_out(NOMINAL_COLS),
        index=df.index,
    )
    return pd.concat([f, onehot_df], axis=1)


def run_experiment(df: pd.DataFrame) -> dict:
    y = df["readmitted_30"].values

    # Split identique dans les 2 versions (même random_state que train.py)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]
    y_train, y_test   = y[df.index.get_indexer(train_idx)], y[df.index.get_indexer(test_idx)]

    results = {}

    # ── (A) Baseline : LabelEncoder ────────────────────────────────
    logger.info("── Version A : LabelEncoder (baseline actuelle) ──")
    encoders = {}
    for col in NOMINAL_COLS:
        le = LabelEncoder()
        le.fit(df_train[col].fillna("No").astype(str))
        encoders[col] = le

    X_train_a = build_features_label(df_train, encoders)
    X_test_a  = build_features_label(df_test, encoders)

    model_a = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    model_a.fit(X_train_a, y_train, eval_set=[(X_test_a, y_test)], verbose=False)
    proba_a = model_a.predict_proba(X_test_a)[:, 1]
    results["label_encoder"] = {
        "auc_roc": round(float(roc_auc_score(y_test, proba_a)), 4),
        "auc_pr":  round(float(average_precision_score(y_test, proba_a)), 4),
        "n_features": X_train_a.shape[1],
    }
    logger.info(f"  AUC-ROC={results['label_encoder']['auc_roc']} | AUC-PR={results['label_encoder']['auc_pr']} | {X_train_a.shape[1]} features")

    # ── (B) Testée : One-Hot ───────────────────────────────────────
    logger.info("── Version B : One-Hot Encoding (testée) ──")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df_train[NOMINAL_COLS].fillna("No").astype(str))

    X_train_b = build_features_onehot(df_train, ohe)
    X_test_b  = build_features_onehot(df_test, ohe)

    model_b = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    model_b.fit(X_train_b, y_train, eval_set=[(X_test_b, y_test)], verbose=False)
    proba_b = model_b.predict_proba(X_test_b)[:, 1]
    results["one_hot"] = {
        "auc_roc": round(float(roc_auc_score(y_test, proba_b)), 4),
        "auc_pr":  round(float(average_precision_score(y_test, proba_b)), 4),
        "n_features": X_train_b.shape[1],
    }
    logger.info(f"  AUC-ROC={results['one_hot']['auc_roc']} | AUC-PR={results['one_hot']['auc_pr']} | {X_train_b.shape[1]} features")

    # ── Conclusion ──────────────────────────────────────────────────
    delta_auc = results["one_hot"]["auc_roc"] - results["label_encoder"]["auc_roc"]
    delta_ap  = results["one_hot"]["auc_pr"]  - results["label_encoder"]["auc_pr"]
    results["conclusion"] = {
        "delta_auc_roc": round(delta_auc, 4),
        "delta_auc_pr":  round(delta_ap, 4),
        "verdict": (
            "One-Hot améliore légèrement" if delta_auc > 0.003 else
            "LabelEncoder reste préférable" if delta_auc < -0.003 else
            "Différence négligeable — pas d'impact significatif"
        ),
    }
    logger.info(f"\n{'='*55}")
    logger.info(f"Δ AUC-ROC (one-hot − label) = {delta_auc:+.4f}")
    logger.info(f"Δ AUC-PR  (one-hot − label) = {delta_ap:+.4f}")
    logger.info(f"Verdict : {results['conclusion']['verdict']}")
    logger.info(f"{'='*55}")

    return results


if __name__ == "__main__":
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM patients WHERE readmitted_30 IS NOT NULL", engine)
    logger.info(f"{len(df)} patients chargés pour l'expérience")

    results = run_experiment(df)

    os.makedirs("/app/experiments", exist_ok=True)
    with open("/app/experiments/exp_encoding_results.json", "w") as fjson:
        json.dump(results, fjson, indent=2, ensure_ascii=False)
    logger.info("✓ Résultats sauvegardés dans experiments/exp_encoding_results.json")
