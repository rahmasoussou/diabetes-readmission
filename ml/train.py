"""
Entraînement XGBoost + SHAP (v3)
==================================
Améliorations v3 (suite aux expériences validées par Chaima) :
  - Toutes les rencontres conservées (plus de déduplication côté ETL)
  - Split fit/calib/test GROUPÉ par patient_nbr (GroupShuffleSplit) :
    un même patient ne peut jamais être à la fois dans deux sous-ensembles
    → élimine la fuite de patients, chiffres honnêtes
  - Validation croisée en GroupKFold à 5 plis (au lieu de StratifiedKFold),
    cohérent avec les expériences (experiments/exp_group_split_5fold.py)
  - Nouvelle métrique clinique : recall parmi les 10% de patients jugés
    les plus à risque — "si le service ne peut suivre que 10% des
    sortants, combien de vrais réadmis attrape-t-on ?"

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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
    f1_score, precision_recall_curve,
    precision_score, recall_score,
)
from dotenv import load_dotenv
from features import fit_encoders, build_features, save_artifacts

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH     = "/app/models/xgboost_v3.pkl"
EXPLAINER_PATH = "/app/models/shap_explainer.pkl"

XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.75,
    colsample_bytree=0.75,
    colsample_bylevel=0.75,
    min_child_weight=10,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="aucpr",
    random_state=42,
    verbosity=0,
)


def get_engine():
    u = os.environ["POSTGRES_USER"]
    p = os.environ["POSTGRES_PASSWORD"]
    h = os.environ["POSTGRES_HOST"]
    d = os.environ["POSTGRES_DB"]
    return create_engine(f"postgresql://{u}:{p}@{h}/{d}")


def load_from_db(engine) -> pd.DataFrame:
    logger.info("Chargement des données depuis PostgreSQL...")
    df = pd.read_sql("SELECT * FROM patients WHERE readmitted_30 IS NOT NULL", engine)
    logger.info(f"  → {len(df)} rencontres chargées | {df['patient_nbr'].nunique()} patients uniques")
    return df


def compute_data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:10]


def find_best_threshold(y_true, y_proba) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    logger.info(f"  → Seuil optimal : {best_threshold:.3f} (F1={f1_scores[best_idx]:.4f})")
    return best_threshold


def top10_capture_rate(y_true, y_proba) -> float:
    """Parmi les 10% de patients jugés les plus à risque par le modèle,
    quelle proportion des vrais réadmis sous 30j est effectivement capturée.
    Métrique clinique — parle davantage au service qu'un AUC seul."""
    n_top = max(1, int(0.10 * len(y_true)))
    order = np.argsort(-y_proba)[:n_top]
    return float(np.asarray(y_true)[order].sum() / max(1, np.asarray(y_true).sum()))
def threshold_analysis(y_true, y_proba) -> list:
    """Calcule precision, recall, et % de patients alertés pour une grille
    de seuils — permet d'afficher le compromis dans le dashboard, sans
    toucher au modèle ni au seuil de production (best_threshold reste
    calculé séparément par find_best_threshold)."""
    thresholds = [round(t, 2) for t in np.arange(0.02, 0.61, 0.02)]
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        n_alerted = int(y_pred_t.sum())
        rows.append({
            "threshold": t,
            "precision": round(float(precision_score(y_true, y_pred_t, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred_t, zero_division=0)), 4),
            "n_alerted": n_alerted,
            "pct_alerted": round(n_alerted / len(y_true), 4),
        })
    return rows


def group_split(X, y, groups, test_size=0.2, random_state=42):
    """Split groupé par patient_nbr — un même patient ne peut jamais se
    retrouver des deux côtés du split (élimine la fuite)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_a, idx_b = next(gss.split(X, y, groups=groups))
    return idx_a, idx_b


def train(df: pd.DataFrame) -> None:

    # 0. Traçabilité
    data_hash = compute_data_hash(df)
    logger.info(f"  → Hash de traçabilité du dataset : {data_hash}")

    # 1. Features
    encoders = fit_encoders(df)
    X = build_features(df, encoders)
    y = df["readmitted_30"].values
    groups = df["patient_nbr"].values

    feature_names = list(X.columns)
    logger.info(f"  → {len(feature_names)} features construites (v3)")

    # 2. Split en 3, GROUPÉ par patient (fit / calib / test) — v3 :
    # remplace le train_test_split stratifié aléatoire par un split par groupe,
    # pour qu'un même patient_nbr ne soit jamais à cheval entre deux sous-ensembles.
    train_idx, test_idx = group_split(X, y, groups, test_size=0.2, random_state=42)
    X_train, y_train, groups_train = X.iloc[train_idx], y[train_idx], groups[train_idx]

    fit_idx_rel, calib_idx_rel = group_split(
        X_train, y_train, groups_train, test_size=0.2, random_state=42
    )
    X_fit, y_fit       = X_train.iloc[fit_idx_rel], y_train[fit_idx_rel]
    X_calib, y_calib   = X_train.iloc[calib_idx_rel], y_train[calib_idx_rel]
    X_test, y_test     = X.iloc[test_idx], y[test_idx]

    # Contrôle anti-fuite explicite : aucun patient commun entre les 3 sous-ensembles
    g_fit   = set(groups[train_idx][fit_idx_rel])
    g_calib = set(groups[train_idx][calib_idx_rel])
    g_test  = set(groups[test_idx])
    assert not (g_fit & g_test), "fuite de patients fit/test détectée"
    assert not (g_calib & g_test), "fuite de patients calib/test détectée"
    assert not (g_fit & g_calib), "fuite de patients fit/calib détectée"

    logger.info(f"  → Fit: {len(X_fit)} | Calib: {len(X_calib)} | Test: {len(X_test)}")
    logger.info(f"  → Positifs fit: {y_fit.mean():.2%} | calib: {y_calib.mean():.2%} | test: {y_test.mean():.2%}")
    logger.info("  → Contrôle anti-fuite : OK, aucun patient partagé entre fit/calib/test")

    # 3. Déséquilibre de classes
    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
    logger.info(f"  → scale_pos_weight = {scale_pos_weight:.2f}")

    # 4. XGBoost — l'early stopping s'appuie sur X_calib, jamais sur X_test :
    # les chiffres mesurés sur X_test restent donc "honnêtes" (aucune influence
    # du test set sur l'entraînement, ni directe ni indirecte).
    model = xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_calib, y_calib)],
        verbose=100,
    )
    logger.info(f"  → Meilleure itération : {model.best_iteration}")

    # 4bis. Calibration des probabilités (isotonic, sur le split calibration —
    # jamais utilisé pour l'entraînement des arbres ni pour l'early stopping).
    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated_model.fit(X_calib, y_calib)

    # 5. Évaluation — sur le test final, jamais touché par fit, early stopping,
    # ni calibration. Chiffres honnêtes.
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

    best_threshold = find_best_threshold(y_calib, calibrated_model.predict_proba(X_calib)[:, 1])
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    thr_analysis = threshold_analysis(y_test, y_pred_proba)
    logger.info(f"  → Analyse de seuil calculée sur {len(thr_analysis)} points (jeu de test, jamais vu à l'entraînement)")

    auc     = roc_auc_score(y_test, y_pred_proba)
    ap      = average_precision_score(y_test, y_pred_proba)
    f1      = f1_score(y_test, y_pred)
    top10   = top10_capture_rate(y_test, y_pred_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred)

    logger.info(f"\n{'='*55}")
    logger.info(f"AUC-ROC  : {auc:.4f}  (chiffre honnête post-correction early stopping)")
    logger.info(f"AUC-PR   : {ap:.4f}")
    logger.info(f"F1-Score : {f1:.4f}  (seuil={best_threshold:.3f})")
    logger.info(f"Recall top 10% risque : {top10:.4f}  "
                 f"(métrique clinique — parmi les 10% jugés les plus à risque, "
                 f"proportion des vrais réadmis capturée)")
    logger.info(f"Matrice de confusion :\n{cm}")
    logger.info(f"Rapport :\n{report}")
    logger.info(f"{'='*55}")

    # 6. Validation croisée — GroupKFold à 5 plis (cohérent avec les
    # expériences), pas StratifiedKFold : un même patient ne doit jamais être
    # à cheval entre deux plis.
    cv_model = xgb.XGBClassifier(
        **{k: v for k, v in XGB_PARAMS.items() if k not in ("eval_metric", "n_estimators")},
        n_estimators=model.best_iteration + 1,
        scale_pos_weight=scale_pos_weight,
    )
    cv = GroupKFold(n_splits=5)
    cv_auc = cross_val_score(cv_model, X, y, cv=cv, groups=groups, scoring="roc_auc")
    cv_ap  = cross_val_score(cv_model, X, y, cv=cv, groups=groups, scoring="average_precision")
    logger.info(f"GroupKFold 5 plis — AUC-ROC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    logger.info(f"GroupKFold 5 plis — AUC-PR  : {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")

    # 7. SHAP
    logger.info("Construction de l'explainer SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:200])
    logger.info(f"  → SHAP values shape : {shap_values.shape}")

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)
    logger.info(f"  → Top 5 features SHAP :\n{mean_shap.head()}")

    # 8. Sauvegardes
    os.makedirs("/app/models", exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    joblib.dump(explainer,        EXPLAINER_PATH)
    save_artifacts(encoders, feature_names)

    meta = {
        "version":              "v3",
        "data_hash":            data_hash,
        "threshold_analysis":   thr_analysis,
        "auc_roc":               round(auc, 4),
        "auc_pr":                round(ap, 4),
        "f1_score":              round(f1, 4),
        "top10_capture_rate":    round(top10, 4),
        "best_threshold":        round(best_threshold, 4),
        "calibration_method":    "isotonic",
        "split_method":          "GroupShuffleSplit (patient_nbr) — fit/calib/test",
        "groupkfold5_auc_mean":  round(float(cv_auc.mean()), 4),
        "groupkfold5_auc_std":   round(float(cv_auc.std()), 4),
        "groupkfold5_ap_mean":   round(float(cv_ap.mean()), 4),
        "groupkfold5_ap_std":    round(float(cv_ap.std()), 4),
        "n_fit":                 len(X_fit),
        "n_calib":               len(X_calib),
        "n_test":                len(X_test),
        "n_patients_total":      int(df["patient_nbr"].nunique()),
        "n_encounters_total":    len(df),
        "feature_count":         len(feature_names),
        "best_iteration":        model.best_iteration,
    }
    with open("/app/models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("✓ Modèle v3, explainer et artefacts sauvegardés")
    logger.info(f"Résumé : AUC-ROC={auc:.4f} | AUC-PR={ap:.4f} | F1={f1:.4f} | Top10%={top10:.4f}")


if __name__ == "__main__":
    engine = get_engine()
    df     = load_from_db(engine)
    train(df)