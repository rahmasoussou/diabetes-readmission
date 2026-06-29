"""
Tests unitaires — API FastAPI
================================
Lancer avec : pytest tests/ -v
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, "/app/ml")

os.environ.setdefault("JWT_SECRET_KEY",       "test_secret_key_pour_les_tests")
os.environ.setdefault("JWT_ALGORITHM",        "HS256")
os.environ.setdefault("JWT_EXPIRE_HOURS",     "1")
os.environ.setdefault("DASHBOARD_USERNAME",   "medecin")
os.environ.setdefault("DASHBOARD_PASSWORD",   "test_password")
os.environ.setdefault("POSTGRES_USER",        "test")
os.environ.setdefault("POSTGRES_PASSWORD",    "test")
os.environ.setdefault("POSTGRES_HOST",        "localhost")
os.environ.setdefault("POSTGRES_DB",          "test")
os.environ.setdefault("MODEL_VERSION",        "v1")

from fastapi.testclient import TestClient
from jose import jwt


# ─── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def client():
    with patch("api.load_model"):   # ne pas charger le vrai modèle
        from api import app
        return TestClient(app)


@pytest.fixture
def valid_token():
    payload = {
        "sub": "medecin",
        "exp": datetime.utcnow() + timedelta(hours=1),
    }
    return jwt.encode(payload, "test_secret_key_pour_les_tests", algorithm="HS256")


@pytest.fixture
def patient_payload():
    return {
        "age_num": 65, "gender": "Female", "race": "Caucasian",
        "time_in_hospital": 5, "num_medications": 15,
        "num_lab_procedures": 40, "num_procedures": 1,
        "number_diagnoses": 7, "num_outpatient": 0,
        "num_emergency": 1, "num_inpatient": 2,
        "a1c_result": ">7", "glucose_serum": "None",
        "change_in_meds": "Ch", "diabetes_meds": "Yes",
    }


# ─── Tests santé ──────────────────────────────────────────────────
def test_health_endpoint(client):
    """L'endpoint /health répond sans auth."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"


# ─── Tests authentification ───────────────────────────────────────
def test_login_success(client):
    """Login avec bons identifiants → JWT retourné."""
    resp = client.post("/token", json={
        "username": "medecin",
        "password": "test_password",
    })
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password(client):
    """Login avec mauvais mot de passe → 401."""
    resp = client.post("/token", json={
        "username": "medecin",
        "password": "mauvais_mot_de_passe",
    })
    assert resp.status_code == 401


def test_predict_no_token(client, patient_payload):
    """Prédiction sans token → rejetée."""
    resp = client.post("/predict", json=patient_payload)
    assert resp.status_code in [401, 403]


def test_predict_invalid_token(client, patient_payload):
    """Prédiction avec token invalide → rejetée."""
    resp = client.post(
        "/predict",
        json=patient_payload,
        headers={"Authorization": "Bearer token_faux"},
    )
    assert resp.status_code == 401


def test_predict_expired_token(client, patient_payload):
    """Prédiction avec token expiré → rejetée."""
    expired = jwt.encode(
        {"sub": "medecin", "exp": datetime.utcnow() - timedelta(hours=1)},
        "test_secret_key_pour_les_tests",
        algorithm="HS256",
    )
    resp = client.post(
        "/predict",
        json=patient_payload,
        headers={"Authorization": f"Bearer {expired}"},
    )
    assert resp.status_code == 401


# ─── Tests validation des données ─────────────────────────────────
def test_predict_age_out_of_range(client, patient_payload, valid_token):
    """Âge hors limite → erreur de validation."""
    bad_payload = {**patient_payload, "age_num": 200}
    resp = client.post(
        "/predict",
        json=bad_payload,
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert resp.status_code == 422  # Unprocessable Entity (Pydantic)


def test_predict_negative_medications(client, patient_payload, valid_token):
    """Médicaments négatifs → erreur de validation."""
    bad_payload = {**patient_payload, "num_medications": -5}
    resp = client.post(
        "/predict",
        json=bad_payload,
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert resp.status_code == 422


# ─── Tests de la prédiction (avec mock modèle) ────────────────────
def test_predict_with_mock_model(client, patient_payload, valid_token):
    """Prédiction avec modèle mocké → structure de réponse correcte."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    mock_model.predict.return_value = np.array([0])

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, 0.05, -0.1]])

    mock_encoders = {}
    mock_features = ["age_num", "time_in_hospital", "num_medications",
                     "num_lab_procedures", "number_diagnoses"]

    with patch("api.model",        mock_model), \
         patch("api.explainer",    mock_explainer), \
         patch("api.encoders",     mock_encoders), \
         patch("api.feature_names", mock_features), \
         patch("api.get_engine"):

        resp = client.post(
            "/predict",
            json=patient_payload,
            headers={"Authorization": f"Bearer {valid_token}"},
        )

    assert resp.status_code == 200
    data = resp.json()

    assert "risk_score"    in data
    assert "risk_level"    in data
    assert "top_factors"   in data
    assert "model_version" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_level"] in ["FAIBLE", "MODÉRÉ", "ÉLEVÉ"]
