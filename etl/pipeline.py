"""
Pipeline ETL — Prédiction Réhospitalisation Diabétiques (v2)
=============================================================
v2 : charge les médicaments (21 colonnes) et les diagnostics ICD-9
     → permet au modèle ML d'exploiter ces features très prédictives
"""

import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Colonnes médicaments présentes dans le CSV ───────────────────
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Renommage pour compatibilité SQL (pas de tirets)
MED_RENAME = {
    "glyburide-metformin":        "glyburide_metformin",
    "glipizide-metformin":        "glipizide_metformin",
    "glimepiride-pioglitazone":   "glimepiride_pioglitazone",
    "metformin-rosiglitazone":    "metformin_rosiglitazone",
    "metformin-pioglitazone":     "metformin_pioglitazone",
}


def get_db_engine():
    user     = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host     = os.environ["POSTGRES_HOST"]
    port     = os.environ.get("POSTGRES_PORT", "5432")
    db       = os.environ["POSTGRES_DB"]
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")


def load_raw_data(path: str) -> pd.DataFrame:
    logger.info(f"Chargement des données : {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  → {df.shape[0]} lignes, {df.shape[1]} colonnes chargées")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Nettoyage des données...")
    df = df.replace("?", np.nan)

    # Supprimer colonnes inutiles (mais garder médicaments et diagnostics)
    cols_to_drop = ["weight", "payer_code", "medical_specialty", "examide", "citoglipton"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Exclure décès et transferts
    if "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]

    # Dédupliquer par patient (garder premier séjour)
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    logger.info(f"  → {df.shape[0]} lignes après déduplication")

    # Cible binaire
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    logger.info(f"  → Taux réhospitalisation <30j : {df['readmitted_30'].mean():.2%}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Feature engineering...")

    # Âge numérique
    age_map = {
        "[0-10)": 5,   "[10-20)": 15, "[20-30)": 25,
        "[30-40)": 35, "[40-50)": 45, "[50-60)": 55,
        "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95
    }
    df["age_num"] = df["age"].map(age_map).fillna(55)

    # Features dérivées
    df["meds_per_day"] = df["num_medications"] / (df["time_in_hospital"] + 1)
    df["total_visits"] = (
        df.get("number_outpatient", pd.Series(0, index=df.index)).fillna(0) +
        df.get("number_emergency",  pd.Series(0, index=df.index)).fillna(0) +
        df.get("number_inpatient",  pd.Series(0, index=df.index)).fillna(0)
    )

    # Renommage colonnes visites
    rename_map = {
        "number_outpatient": "num_outpatient",
        "number_emergency":  "num_emergency",
        "number_inpatient":  "num_inpatient",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Renommage médicaments (tirets → underscores)
    df = df.rename(columns={k: v for k, v in MED_RENAME.items() if k in df.columns})

    # Remplir NaN médicaments par "No"
    med_cols_renamed = [MED_RENAME.get(c, c) for c in MED_COLS]
    for col in med_cols_renamed:
        if col in df.columns:
            df[col] = df[col].fillna("No")

    # Diagnostics : garder comme string, nettoyer
    for d in ["diag_1", "diag_2", "diag_3"]:
        if d in df.columns:
            df[d] = df[d].fillna("Unknown").astype(str).str.strip()

    logger.info(f"  → Features construites ({df.shape[1]} colonnes)")
    return df


def load_to_db(df: pd.DataFrame, engine) -> None:
    logger.info("Chargement en base PostgreSQL...")

    # Colonnes à charger (dans l'ordre de la table)
    med_cols_db = [MED_RENAME.get(c, c) for c in MED_COLS]

    cols_db = [
        "encounter_id", "patient_nbr", "race", "gender", "age", "age_num",
        "time_in_hospital", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "num_lab_procedures", "num_procedures",
        "num_medications", "number_diagnoses",
        "num_outpatient", "num_emergency", "num_inpatient",
        "diag_1", "diag_2", "diag_3",
        "A1Cresult", "max_glu_serum", "change", "diabetesMed",
    ] + [MED_RENAME.get(c, c) for c in MED_COLS] + [
        "meds_per_day", "total_visits", "readmitted", "readmitted_30"
    ]

    col_rename = {
        "A1Cresult":                "a1c_result",
        "max_glu_serum":            "glucose_serum",
        "change":                   "change_in_meds",
        "diabetesMed":              "diabetes_meds",
        "admission_type_id":        "admission_type",
        "discharge_disposition_id": "discharge_type",
        "admission_source_id":      "admission_source",
    }

    cols_available = [c for c in cols_db if c in df.columns]
    missing = [c for c in cols_db if c not in df.columns]
    if missing:
        logger.warning(f"  → Colonnes absentes (ignorées) : {missing}")

    df_out = df[cols_available].rename(columns=col_rename)

    # Vider les tables
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE predictions RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE patients RESTART IDENTITY CASCADE"))
        conn.commit()
    logger.info("  → Tables vidées")

    df_out.to_sql("patients", engine, if_exists="append", index=False, chunksize=500)
    logger.info(f"  → {len(df_out)} lignes chargées en base ✓")
    logger.info(f"  → Colonnes chargées : {list(df_out.columns)}")


if __name__ == "__main__":
    RAW_PATH = "/app/data/raw/diabetic_data.csv"
    if not os.path.exists(RAW_PATH):
        logger.error(f"Fichier introuvable : {RAW_PATH}")
        exit(1)
    engine = get_db_engine()
    df = load_raw_data(RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    load_to_db(df, engine)
    logger.info("Pipeline ETL v2 terminé avec succès ✓")