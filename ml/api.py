"""
API FastAPI — Service de prédiction
=====================================
Endpoints :
  GET  /health         — vérification santé (pas d'auth)
  POST /token          — obtenir un JWT
  POST /predict        — score de risque + SHAP (JWT requis)
  GET  /model/info     — méta-données du modèle (JWT requis)
  GET  /predictions    — historique paginé des prédictions (JWT requis)
  GET  /stats          — statistiques agrégées du dashboard (JWT requis)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import create_engine, text
from features import build_features, load_artifacts

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────
JWT_SECRET    = os.environ["JWT_SECRET_KEY"]
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_H  = int(os.environ.get("JWT_EXPIRE_HOURS", 1))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

DASHBOARD_USER = os.environ.get("DASHBOARD_USERNAME", "medecin")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASSWORD")

# ─── App ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Diabetes Readmission API",
    description="Prédiction de réhospitalisation des patients diabétiques",
    version="1.1.0",
    docs_url="/docs",
    redoc_url=None,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()

# ─── Chargement du modèle au démarrage ────────────────────────────
model         = None
explainer     = None
encoders      = None
feature_names = None
model_meta    = {}

@app.on_event("startup")
def load_model():
    global model, explainer, encoders, feature_names, model_meta
    try:
        model     = joblib.load("/app/models/xgboost_v1.pkl")
        explainer = joblib.load("/app/models/shap_explainer.pkl")
        encoders, feature_names = load_artifacts("/app/models")
        with open("/app/models/model_meta.json") as f:
            model_meta = json.load(f)
        logger.info(f"Modèle chargé — AUC-ROC : {model_meta.get('auc_roc', 'N/A')} ✓")
    except FileNotFoundError:
        logger.warning("Modèle non trouvé. Lance d'abord : python train.py")


# ─── JWT ──────────────────────────────────────────────────────────
def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_H),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── DB ───────────────────────────────────────────────────────────
def get_engine():
    u = os.environ["POSTGRES_USER"]
    p = os.environ["POSTGRES_PASSWORD"]
    h = os.environ["POSTGRES_HOST"]
    d = os.environ["POSTGRES_DB"]
    return create_engine(f"postgresql://{u}:{p}@{h}/{d}")


# ─── Schémas Pydantic ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class PatientInput(BaseModel):
    age_num:            float = Field(..., ge=0, le=120)
    gender:             str   = Field("Unknown")
    race:               str   = Field("Unknown")
    time_in_hospital:   float = Field(..., ge=1, le=30)
    num_medications:    float = Field(..., ge=0, le=100)
    num_lab_procedures: float = Field(..., ge=0, le=130)
    num_procedures:     float = Field(..., ge=0, le=10)
    number_diagnoses:   float = Field(..., ge=1, le=16)
    num_outpatient:     float = Field(0, ge=0)
    num_emergency:      float = Field(0, ge=0)
    num_inpatient:      float = Field(0, ge=0)
    a1c_result:         str   = Field("None")
    glucose_serum:      str   = Field("None")
    change_in_meds:     str   = Field("No")
    diabetes_meds:      str   = Field("Yes")
    meds_per_day:       Optional[float] = None
    total_visits:       Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "age_num": 65, "gender": "Female", "race": "Caucasian",
                "time_in_hospital": 5, "num_medications": 15,
                "num_lab_procedures": 40, "num_procedures": 1,
                "number_diagnoses": 7, "num_outpatient": 0,
                "num_emergency": 1, "num_inpatient": 2,
                "a1c_result": ">7", "glucose_serum": "None",
                "change_in_meds": "Ch", "diabetes_meds": "Yes"
            }
        }


class PredictionResponse(BaseModel):
    risk_score:    float
    risk_level:    str
    top_factors:   dict
    model_version: str


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }


@app.post("/token")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    if body.username != DASHBOARD_USER or body.password != DASHBOARD_PASS:
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
    token = create_token(body.username)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
def predict(request: Request, patient: PatientInput, user: dict = Depends(verify_token)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible. Lance train.py d'abord.")

    data_dict = patient.dict()

    if data_dict["meds_per_day"] is None:
        data_dict["meds_per_day"] = data_dict["num_medications"] / (data_dict["time_in_hospital"] + 1)
    if data_dict["total_visits"] is None:
        data_dict["total_visits"] = (
            data_dict["num_outpatient"] +
            data_dict["num_emergency"] +
            data_dict["num_inpatient"]
        )

    df_input = pd.DataFrame([data_dict])
    X = build_features(df_input, encoders)[feature_names]

    risk_score = float(model.predict_proba(X)[0][1])

    if risk_score >= 0.5:
        risk_level = "ÉLEVÉ"
    elif risk_score >= 0.3:
        risk_level = "MODÉRÉ"
    else:
        risk_level = "FAIBLE"

    shap_vals = explainer.shap_values(X)
    explanations = {
        feat: round(float(val), 4)
        for feat, val in zip(feature_names, shap_vals[0])
    }
    top_factors = dict(
        sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO predictions (risk_score, risk_level, model_version, top_factors, requested_by)
                    VALUES (:score, :level, :version, CAST(:factors AS jsonb), :user)
                """),
                {
                    "score":   risk_score,
                    "level":   risk_level,
                    "version": MODEL_VERSION,
                    "factors": json.dumps(top_factors),
                    "user":    user.get("sub", "unknown"),
                }
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Impossible d'enregistrer la prédiction : {e}")

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        top_factors=top_factors,
        model_version=MODEL_VERSION,
    )


@app.get("/model/info")
def model_info(user: dict = Depends(verify_token)):
    return model_meta


# ─── NOUVEAU : Historique des prédictions ─────────────────────────
@app.get("/predictions")
def get_predictions(
    user:      dict = Depends(verify_token),
    limit:     int  = Query(50,  ge=1, le=200),
    offset:    int  = Query(0,   ge=0),
    risk_level: str = Query(None, description="Filtrer par niveau : ÉLEVÉ, MODÉRÉ, FAIBLE"),
):
    """
    Retourne l'historique paginé des prédictions.
    Filtrable par niveau de risque.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Filtre optionnel par niveau
            where = "WHERE risk_level = :level" if risk_level else ""
            params = {"limit": limit, "offset": offset}
            if risk_level:
                params["level"] = risk_level

            rows = conn.execute(
                text(f"""
                    SELECT
                        id,
                        risk_score,
                        risk_level,
                        model_version,
                        top_factors,
                        requested_by,
                        predicted_at
                    FROM predictions
                    {where}
                    ORDER BY predicted_at DESC
                    LIMIT :limit OFFSET :offset
                """),
                params,
            ).fetchall()

            total = conn.execute(
                text(f"SELECT COUNT(*) FROM predictions {where}"),
                {"level": risk_level} if risk_level else {},
            ).scalar()

        return {
            "total":   total,
            "limit":   limit,
            "offset":  offset,
            "results": [
                {
                    "id":            r[0],
                    "risk_score":    round(r[1], 4),
                    "risk_level":    r[2],
                    "model_version": r[3],
                    "top_factors":   r[4],
                    "requested_by":  r[5],
                    "predicted_at":  r[6].isoformat() if r[6] else None,
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.error(f"Erreur /predictions : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── NOUVEAU : Statistiques agrégées ──────────────────────────────
@app.get("/stats")
def get_stats(user: dict = Depends(verify_token)):
    """
    Statistiques globales pour le dashboard :
    - Répartition des niveaux de risque
    - Score moyen / min / max
    - Évolution des prédictions sur les 30 derniers jours
    - Top facteurs globaux (agrégation SHAP)
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:

            # 1. Totaux et répartition par niveau
            dist = conn.execute(text("""
                SELECT
                    COUNT(*)                                        AS total,
                    COUNT(*) FILTER (WHERE risk_level = 'ÉLEVÉ')   AS eleve,
                    COUNT(*) FILTER (WHERE risk_level = 'MODÉRÉ')  AS modere,
                    COUNT(*) FILTER (WHERE risk_level = 'FAIBLE')  AS faible,
                    ROUND(AVG(risk_score)::numeric, 4)              AS score_moyen,
                    ROUND(MIN(risk_score)::numeric, 4)              AS score_min,
                    ROUND(MAX(risk_score)::numeric, 4)              AS score_max
                FROM predictions
            """)).fetchone()

            # 2. Évolution quotidienne (30 derniers jours)
            trend = conn.execute(text("""
                SELECT
                    DATE(predicted_at)                              AS jour,
                    COUNT(*)                                        AS nb,
                    ROUND(AVG(risk_score)::numeric, 4)              AS score_moyen,
                    COUNT(*) FILTER (WHERE risk_level = 'ÉLEVÉ')   AS nb_eleve
                FROM predictions
                WHERE predicted_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(predicted_at)
                ORDER BY jour
            """)).fetchall()

            # 3. Agrégation des top_factors sur toutes les prédictions
            factors_raw = conn.execute(text("""
                SELECT top_factors FROM predictions
                WHERE top_factors IS NOT NULL
                LIMIT 500
            """)).fetchall()

        # Agrégation Python des facteurs SHAP
        factor_totals: dict = {}
        factor_counts: dict = {}
        for row in factors_raw:
            factors = row[0] if isinstance(row[0], dict) else {}
            for k, v in factors.items():
                factor_totals[k] = factor_totals.get(k, 0.0) + abs(float(v))
                factor_counts[k] = factor_counts.get(k, 0) + 1

        top_global = sorted(
            [
                {"feature": k, "mean_abs_shap": round(factor_totals[k] / factor_counts[k], 4)}
                for k in factor_totals
            ],
            key=lambda x: x["mean_abs_shap"],
            reverse=True,
        )[:8]

        return {
            "total":       int(dist[0]) if dist[0] else 0,
            "eleve":       int(dist[1]) if dist[1] else 0,
            "modere":      int(dist[2]) if dist[2] else 0,
            "faible":      int(dist[3]) if dist[3] else 0,
            "score_moyen": float(dist[4]) if dist[4] else 0.0,
            "score_min":   float(dist[5]) if dist[5] else 0.0,
            "score_max":   float(dist[6]) if dist[6] else 0.0,
            "trend": [
                {
                    "jour":        str(r[0]),
                    "nb":          int(r[1]),
                    "score_moyen": float(r[2]) if r[2] else 0.0,
                    "nb_eleve":    int(r[3]),
                }
                for r in trend
            ],
            "top_factors_global": top_global,
        }

    except Exception as e:
        logger.error(f"Erreur /stats : {e}")
        raise HTTPException(status_code=500, detail=str(e))