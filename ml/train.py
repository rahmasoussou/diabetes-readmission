"""
Entraînement XGBoost + SHAP (v2)
==================================
Améliorations v2 :
  - Hyperparamètres optimisés pour données médicales déséquilibrées
  - Optimisation du seuil de décision (maximise F1 sur classe positive)
  - Rapport complet : AUC-ROC, AUC-PR, recall, F1
  - SHAP global summary sauvegardé

Lancer avec :
  docker-compose exec ml-service python train.py
"""

import os
import logging
import joblib
import json
import hashlib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
    f1_score, precision_recall_curve,
)
from dotenv import load_dotenv
from features import fit_encoders, build_features, save_artifacts

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH     = "/app/models/xgboost_v1.pkl"
EXPLAINER_PATH = "/app/models/shap_explainer.pkl"


def get_engine():
    u = os.environ["POSTGRES_USER"]
    p = os.environ["POSTGRES_PASSWORD"]
    h = os.environ["POSTGRES_HOST"]
    d = os.environ["POSTGRES_DB"]
    return create_engine(f"postgresql://{u}:{p}@{h}/{d}")


def load_from_db(engine) -> pd.DataFrame:
    logger.info("Chargement des données depuis PostgreSQL...")
    df = pd.read_sql("SELECT * FROM patients WHERE readmitted_30 IS NOT NULL", engine)
    logger.info(f"  → {len(df)} patients chargés")
    return df


def compute_data_hash(df: pd.DataFrame) -> str:
    """Hash de traçabilité du DataFrame d'entraînement — 10 premiers caractères du SHA256.
    Permet de savoir, a posteriori, si un modèle a été entraîné sur exactement
    le même jeu de données (traçabilité / reproductibilité)."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:10]


def find_best_threshold(y_true, y_proba) -> float:
    """Trouve le seuil qui maximise le F1-score sur la classe positive."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    logger.info(f"  → Seuil optimal : {best_threshold:.3f} (F1={f1_scores[best_idx]:.4f})")
    return best_threshold


def train(df: pd.DataFrame) -> None:

    # 0. Traçabilité : hash du jeu de données utilisé pour cet entraînement
    data_hash = compute_data_hash(df)
    logger.info(f"  → Hash de traçabilité du dataset : {data_hash}")

    # 1. Features
    encoders = fit_encoders(df)
    X = build_features(df, encoders)
    y = df["readmitted_30"].values

    feature_names = list(X.columns)
    logger.info(f"  → {len(feature_names)} features construites (v2)")

    # 2. Split stratifié : train / test, puis un sous-split calibration dans train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Sous-split dédié à la calibration + early stopping, JAMAIS vu par le test final
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    logger.info(f"  → Fit: {len(X_fit)} | Calib: {len(X_calib)} | Test: {len(X_test)}")
    logger.info(f"  → Positifs fit: {y_fit.mean():.2%} | calib: {y_calib.mean():.2%} | test: {y_test.mean():.2%}")

    # 3. Déséquilibre de classes
    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
    logger.info(f"  → scale_pos_weight = {scale_pos_weight:.2f}")

    # 4. XGBoost — hyperparamètres optimisés pour données médicales
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=5,                # moins profond = moins d'overfitting
        learning_rate=0.03,         # plus lent = meilleure généralisation
        subsample=0.75,
        colsample_bytree=0.75,
        colsample_bylevel=0.75,     # nouveau : régularisation supplémentaire
        min_child_weight=10,        # plus élevé = moins sensible aux outliers
        gamma=0.1,                  # nouveau : élagage des feuilles inutiles
        reg_alpha=0.1,              # nouveau : régularisation L1
        reg_lambda=1.0,             # régularisation L2
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",        # optimise AUC-PR (mieux pour classes déséquilibrées)
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_fit, y_fit,
        eval_set=[(X_calib, y_calib)],
        verbose=100,
    )
    logger.info(f"  → Meilleure itération : {model.best_iteration}")

    # 4bis. Calibration des probabilités (isotonic, sur le split calibration —
    # jamais utilisé pour l'entraînement des arbres). Sans ça, les scores XGBoost
    # bruts sont bien classés (bon AUC) mais mal calibrés en valeur absolue
    # (ex. un score de 0.8 ne correspond pas vraiment à 80% de risque réel).
    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated_model.fit(X_calib, y_calib)

    # 5. Évaluation — sur le test final, jamais touché ni par le fit ni par la calibration
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

    # Seuil optimisé — recalculé sur les probabilités CALIBRÉES (le seuil change
    # après calibration, on ne peut pas réutiliser un seuil trouvé avant)
    best_threshold = find_best_threshold(y_calib, calibrated_model.predict_proba(X_calib)[:, 1])
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    auc  = roc_auc_score(y_test, y_pred_proba)
    ap   = average_precision_score(y_test, y_pred_proba)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"\n{'='*55}")
    logger.info(f"AUC-ROC  : {auc:.4f}  (objectif > 0.68)")
    logger.info(f"AUC-PR   : {ap:.4f}   (métrique principale — classes déséquilibrées)")
    logger.info(f"F1-Score : {f1:.4f}  (seuil={best_threshold:.3f})")
    logger.info(f"Matrice de confusion :\n{cm}")
    logger.info(f"Rapport :\n{report}")
    logger.info(f"{'='*55}")

    # 6. Validation croisée
    cv_model = xgb.XGBClassifier(
        n_estimators=model.best_iteration + 1,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        colsample_bylevel=0.75,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc")
    cv_ap  = cross_val_score(cv_model, X, y, cv=cv, scoring="average_precision")
    logger.info(f"Cross-val AUC-ROC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    logger.info(f"Cross-val AUC-PR  : {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")

    # 7. SHAP — basé sur le modèle XGBoost brut (TreeExplainer ne supporte pas le
    # wrapper de calibration). La calibration ne fait que recalibrer l'échelle des
    # probabilités, elle ne change pas l'ordre d'importance des features.
    logger.info("Construction de l'explainer SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:200])
    logger.info(f"  → SHAP values shape : {shap_values.shape}")

    # Top features SHAP globales
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)
    logger.info(f"  → Top 5 features SHAP :\n{mean_shap.head()}")

    # 8. Sauvegardes — on sauvegarde le modèle CALIBRÉ : c'est lui que l'API
    # charge et utilise pour predict_proba (interface identique à XGBClassifier).
    os.makedirs("/app/models", exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    joblib.dump(explainer,        EXPLAINER_PATH)
    save_artifacts(encoders, feature_names)

    meta = {
        "version":            "v2",
        "data_hash":          data_hash,
        "auc_roc":            round(auc, 4),
        "auc_pr":             round(ap, 4),
        "f1_score":           round(f1, 4),
        "best_threshold":     round(best_threshold, 4),
        "calibration_method": "isotonic",
        "cv_auc_mean":        round(float(cv_auc.mean()), 4),
        "cv_auc_std":         round(float(cv_auc.std()), 4),
        "cv_ap_mean":         round(float(cv_ap.mean()), 4),
        "n_fit":              len(X_fit),
        "n_calib":            len(X_calib),
        "n_test":             len(X_test),
        "feature_count":      len(feature_names),
        "best_iteration":     model.best_iteration,
        "n_features_v2":      len(feature_names),
    }
    with open("/app/models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("✓ Modèle v2, explainer et artefacts sauvegardés")
    logger.info(f"Résumé : AUC-ROC={auc:.4f} | AUC-PR={ap:.4f} | F1={f1:.4f}")


if __name__ == "__main__":
    engine = get_engine()
    df     = load_from_db(engine)
    train(df)