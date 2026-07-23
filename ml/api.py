"""
API FastAPI — Service de prédiction (v2)
==========================================
Endpoints :
  GET  /health              — vérification santé (pas d'auth)
  POST /token                — obtenir un JWT (authentifie contre la table `users`)
  POST /predict               — score de risque + SHAP (JWT requis)
  POST /predict/batch          — score de risque pour plusieurs patients (JWT requis)
  POST /predict/pdf             — rapport PDF (JWT requis)
  GET  /model/info               — méta-données du modèle (JWT requis)
  GET  /predictions               — historique paginé des prédictions (JWT requis)
  GET  /stats                      — statistiques agrégées du dashboard (JWT requis)

Changements v2 :
  - Authentification multi-utilisateurs via la table `users` (bcrypt), fini le
    compte unique DASHBOARD_USERNAME/DASHBOARD_PASSWORD partagé par tout le monde.
    Les comptes se créent avec scripts/create_user.py.
  - Journal d'audit (`audit_log`) pour toute action sensible.
  - Endpoint /predict/batch pour scorer plusieurs patients en un seul appel.
  - AUC injectée dynamiquement dans le rapport PDF (plus de valeur codée en dur).
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, status, Query
from fastapi.responses import StreamingResponse
import io
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from passlib.context import CryptContext
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
MODEL_VERSION_FALLBACK = os.environ.get("MODEL_VERSION", "unknown")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MAX_BATCH_SIZE = 50

# ─── App ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Diabetes Readmission API",
    description="Prédiction de réhospitalisation des patients diabétiques",
    version="2.0.0",
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

MODEL_CANDIDATES = ["/app/models/xgboost_v3.pkl", "/app/models/xgboost_model.pkl", "/app/models/xgboost_v1.pkl"]


@app.on_event("startup")
def load_model():
    global model, explainer, encoders, feature_names, model_meta
    try:
        model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError("Aucun modèle trouvé")
        model     = joblib.load(model_path)
        explainer = joblib.load("/app/models/shap_explainer.pkl")
        encoders, feature_names = load_artifacts("/app/models")
        with open("/app/models/model_meta.json") as f:
            model_meta = json.load(f)
        logger.info(f"Modèle chargé ({model_path}) — AUC-ROC : {model_meta.get('auc_roc', 'N/A')} ✓")
    except FileNotFoundError:
        logger.warning("Modèle non trouvé. Lance d'abord : python train.py")


def model_version() -> str:
    return model_meta.get("version", MODEL_VERSION_FALLBACK)


# ─── DB ───────────────────────────────────────────────────────────
def get_engine():
    u = os.environ["POSTGRES_USER"]
    p = os.environ["POSTGRES_PASSWORD"]
    h = os.environ["POSTGRES_HOST"]
    d = os.environ["POSTGRES_DB"]
    return create_engine(f"postgresql://{u}:{p}@{h}/{d}")


def log_audit(username: str, action: str, detail: dict = None, ip: str = None) -> None:
    """Best-effort : ne doit jamais faire échouer la requête appelante."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO audit_log (username, action, detail, ip_address)
                    VALUES (:username, :action, CAST(:detail AS jsonb), :ip)
                """),
                {
                    "username": username,
                    "action": action,
                    "detail": json.dumps(detail or {}),
                    "ip": ip,
                },
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Impossible d'écrire l'audit log : {e}")


# ─── JWT ──────────────────────────────────────────────────────────
def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
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


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Vérifie les identifiants contre la table `users` (hash bcrypt)."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT username, password_hash, role, is_active
                    FROM users WHERE username = :username
                """),
                {"username": username},
            ).fetchone()
    except Exception as e:
        logger.error(f"Erreur DB lors de l'authentification : {e}")
        return None

    if row is None or not row[3]:  # inexistant ou désactivé
        return None
    if not pwd_context.verify(password, row[1]):
        return None

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE users SET last_login = NOW() WHERE username = :username"),
                {"username": username},
            )
            conn.commit()
    except Exception:
        pass

    return {"username": row[0], "role": row[2]}


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
    patient_label:       Optional[str] = None  # libre, utile pour /predict/batch

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


class BatchPredictionResponse(BaseModel):
    patient_label: Optional[str] = None
    risk_score:    float
    risk_level:    str
    top_factors:   dict


class BatchRequest(BaseModel):
    patients: List[PatientInput]


# ─── Fonctions internes partagées ─────────────────────────────────
def _prepare_row(patient: PatientInput) -> dict:
    data_dict = patient.dict()
    if data_dict["meds_per_day"] is None:
        data_dict["meds_per_day"] = data_dict["num_medications"] / (data_dict["time_in_hospital"] + 1)
    if data_dict["total_visits"] is None:
        data_dict["total_visits"] = (
            data_dict["num_outpatient"] + data_dict["num_emergency"] + data_dict["num_inpatient"]
        )
    return data_dict


def _risk_level(score: float) -> str:
    """Niveaux de risque basés sur le seuil optimisé (F1) calculé par train.py
    et sauvegardé dans model_meta.json, plutôt que sur un seuil arbitraire codé en dur."""
    best_threshold = model_meta.get("best_threshold", 0.5)
    moderate_threshold = best_threshold * 0.6  # zone tampon avant le seuil optimal
    if score >= best_threshold:
        return "ÉLEVÉ"
    elif score >= moderate_threshold:
        return "MODÉRÉ"
    return "FAIBLE"


def _score_dataframe(df_input: pd.DataFrame):
    X = build_features(df_input, encoders)[feature_names]
    scores = model.predict_proba(X)[:, 1]
    shap_vals = explainer.shap_values(X)
    return scores, shap_vals


def _record_prediction(risk_score: float, risk_level: str, top_factors: dict, username: str) -> None:
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
                    "version": model_version(),
                    "factors": json.dumps(top_factors),
                    "user":    username,
                }
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Impossible d'enregistrer la prédiction : {e}")


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": model_version(),
    }


@app.post("/token")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    user = authenticate_user(body.username, body.password)
    if user is None:
        log_audit(body.username, "LOGIN_FAILED", ip=get_remote_address(request))
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
    token = create_token(user["username"], user["role"])
    log_audit(user["username"], "LOGIN", ip=get_remote_address(request))
    return {"access_token": token, "token_type": "bearer"}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
def predict(request: Request, patient: PatientInput, user: dict = Depends(verify_token)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible. Lance train.py d'abord.")

    data_dict = _prepare_row(patient)
    df_input = pd.DataFrame([data_dict])
    scores, shap_vals = _score_dataframe(df_input)

    risk_score = float(scores[0])
    risk_level = _risk_level(risk_score)

    explanations = {
        feat: round(float(val), 4)
        for feat, val in zip(feature_names, shap_vals[0])
    }
    top_factors = dict(
        sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    )

    username = user.get("sub", "unknown")
    _record_prediction(risk_score, risk_level, top_factors, username)
    log_audit(username, "PREDICT", detail={"risk_level": risk_level}, ip=get_remote_address(request))

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        top_factors=top_factors,
        model_version=model_version(),
    )


@app.post("/predict/batch", response_model=List[BatchPredictionResponse])
@limiter.limit("10/minute")
def predict_batch(request: Request, body: BatchRequest, user: dict = Depends(verify_token)):
    """
    Score plusieurs patients en un seul appel (jusqu'à MAX_BATCH_SIZE).
    Utile pour la comparaison de patients ou l'analyse de cohortes,
    évite les appels séquentiels multiples depuis le dashboard.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible. Lance train.py d'abord.")
    if len(body.patients) == 0:
        raise HTTPException(status_code=422, detail="Liste de patients vide")
    if len(body.patients) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail=f"Maximum {MAX_BATCH_SIZE} patients par appel")

    rows = [_prepare_row(p) for p in body.patients]
    df_input = pd.DataFrame(rows)
    scores, shap_vals = _score_dataframe(df_input)

    username = user.get("sub", "unknown")
    results = []
    for i, patient in enumerate(body.patients):
        risk_score = float(scores[i])
        risk_level = _risk_level(risk_score)
        explanations = {
            feat: round(float(val), 4)
            for feat, val in zip(feature_names, shap_vals[i])
        }
        top_factors = dict(
            sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )
        _record_prediction(risk_score, risk_level, top_factors, username)
        results.append(BatchPredictionResponse(
            patient_label=patient.patient_label,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            top_factors=top_factors,
        ))

    log_audit(username, "PREDICT_BATCH", detail={"n": len(results)}, ip=get_remote_address(request))
    return results


@app.get("/model/info")
def model_info(user: dict = Depends(verify_token)):
    return model_meta


# ─── Historique des prédictions ───────────────────────────────────
@app.get("/predictions")
def get_predictions(
    request:   Request,
    user:      dict = Depends(verify_token),
    limit:     int  = Query(50,  ge=1, le=200),
    offset:    int  = Query(0,   ge=0),
    risk_level: str = Query(None, description="Filtrer par niveau : ÉLEVÉ, MODÉRÉ, FAIBLE"),
):
    """Retourne l'historique paginé des prédictions. Filtrable par niveau de risque."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            where = "WHERE risk_level = :level" if risk_level else ""
            params = {"limit": limit, "offset": offset}
            if risk_level:
                params["level"] = risk_level

            rows = conn.execute(
                text(f"""
                    SELECT id, risk_score, risk_level, model_version, top_factors, requested_by, predicted_at
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

        log_audit(user.get("sub", "unknown"), "VIEW_HISTORY", ip=get_remote_address(request))

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


# ─── Statistiques agrégées ─────────────────────────────────────────
@app.get("/stats")
def get_stats(user: dict = Depends(verify_token)):
    try:
        engine = get_engine()
        with engine.connect() as conn:
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

            factors_raw = conn.execute(text("""
                SELECT top_factors FROM predictions
                WHERE top_factors IS NOT NULL
                LIMIT 500
            """)).fetchall()

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


# ─── Rapport PDF ────────────────────────────────────────────────
@app.post("/predict/pdf")
@limiter.limit("10/minute")
def predict_pdf(request: Request, patient: PatientInput, user: dict = Depends(verify_token)):
    """Génère un rapport PDF complet pour une prédiction patient."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    data_dict = _prepare_row(patient)
    df_input = pd.DataFrame([data_dict])
    scores, shap_vals = _score_dataframe(df_input)
    risk_score = float(scores[0])
    risk_level = _risk_level(risk_score)

    top_factors = dict(
        sorted(
            {f: float(v) for f, v in zip(feature_names, shap_vals[0])}.items(),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
    )

    try:
        from pdf_report import generate_pdf
        pdf_bytes = generate_pdf(
            patient_data  = data_dict,
            risk_score    = risk_score,
            risk_level    = risk_level,
            top_factors   = top_factors,
            model_version = model_version(),
            requested_by  = user.get("sub", "medecin"),
            model_auc     = model_meta.get("auc_roc"),
        )
        filename = f"rapport_ClinAI_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        log_audit(user.get("sub", "unknown"), "DOWNLOAD_PDF", ip=get_remote_address(request))
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Erreur génération PDF : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur PDF : {str(e)}")