"""
Tests unitaires — API FastAPI (v2)
====================================
Lancer avec : pytest tests/ -v
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, "/app")

# Ces 3 variables sont forcées (pas de setdefault) car le conteneur charge déjà
# le vrai JWT_SECRET_KEY via env_file: .env — sans ça, le token signé par la
# fixture valid_token() (avec la clé de test ci-dessous) ne correspondrait pas
# à la clé utilisée par l'API pour le vérifier, et tout renverrait 401.
os.environ["JWT_SECRET_KEY"]    = "test_secret_key_pour_les_tests"
os.environ["JWT_ALGORITHM"]     = "HS256"
os.environ["JWT_EXPIRE_HOURS"]  = "1"
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
        "role": "medecin",
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
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"


# ─── Tests authentification (via table users mockée) ──────────────
def test_login_success(client):
    with patch("api.authenticate_user", return_value={"username": "medecin", "role": "medecin"}):
        resp = client.post("/token", json={"username": "medecin", "password": "bon_mot_de_passe"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password(client):
    with patch("api.authenticate_user", return_value=None):
        resp = client.post("/token", json={"username": "medecin", "password": "mauvais"})
    assert resp.status_code == 401


def test_login_inactive_or_unknown_user(client):
    with patch("api.authenticate_user", return_value=None):
        resp = client.post("/token", json={"username": "inconnu", "password": "x"})
    assert resp.status_code == 401


def test_predict_no_token(client, patient_payload):
    resp = client.post("/predict", json=patient_payload)
    assert resp.status_code in [401, 403]


def test_predict_invalid_token(client, patient_payload):
    resp = client.post(
        "/predict", json=patient_payload,
        headers={"Authorization": "Bearer token_faux"},
    )
    assert resp.status_code == 401


def test_predict_expired_token(client, patient_payload):
    expired = jwt.encode(
        {"sub": "medecin", "exp": datetime.utcnow() - timedelta(hours=1)},
        "test_secret_key_pour_les_tests", algorithm="HS256",
    )
    resp = client.post(
        "/predict", json=patient_payload,
        headers={"Authorization": f"Bearer {expired}"},
    )
    assert resp.status_code == 401


# ─── Tests validation des données ─────────────────────────────────
def test_predict_age_out_of_range(client, patient_payload, valid_token):
    bad_payload = {**patient_payload, "age_num": 200}
    resp = client.post(
        "/predict", json=bad_payload,
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert resp.status_code == 422


def test_predict_negative_medications(client, patient_payload, valid_token):
    bad_payload = {**patient_payload, "num_medications": -5}
    resp = client.post(
        "/predict", json=bad_payload,
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert resp.status_code == 422


# ─── Tests de la prédiction (avec mock modèle) ────────────────────
def _mocked_model_context():
    import numpy as np
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([
        [0.1, -0.2, 0.3, 0.05, -0.1],
        [0.2, -0.1, 0.1, 0.02, -0.05],
    ])
    mock_encoders = {}
    mock_features = ["age_num", "time_in_hospital", "num_medications",
                      "num_lab_procedures", "number_diagnoses"]
    return mock_model, mock_explainer, mock_encoders, mock_features


def test_predict_with_mock_model(client, patient_payload, valid_token):
    mock_model, mock_explainer, mock_encoders, mock_features = _mocked_model_context()

    with patch("api.model",         mock_model), \
         patch("api.explainer",     mock_explainer), \
         patch("api.encoders",      mock_encoders), \
         patch("api.feature_names", mock_features), \
         patch("api._record_prediction"), \
         patch("api.log_audit"):

        resp = client.post(
            "/predict", json=patient_payload,
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


def test_predict_batch(client, patient_payload, valid_token):
    """/predict/batch retourne un résultat par patient, dans le même ordre."""
    mock_model, mock_explainer, mock_encoders, mock_features = _mocked_model_context()

    with patch("api.model",         mock_model), \
         patch("api.explainer",     mock_explainer), \
         patch("api.encoders",      mock_encoders), \
         patch("api.feature_names", mock_features), \
         patch("api._record_prediction"), \
         patch("api.log_audit"):

        resp = client.post(
            "/predict/batch",
            json={"patients": [
                {**patient_payload, "patient_label": "A"},
                {**patient_payload, "patient_label": "B"},
            ]},
            headers={"Authorization": f"Bearer {valid_token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["patient_label"] == "A"
    assert data[1]["patient_label"] == "B"
    for item in data:
        assert 0 <= item["risk_score"] <= 1


def test_predict_batch_too_large(client, patient_payload, valid_token):
    mock_model, mock_explainer, mock_encoders, mock_features = _mocked_model_context()
    with patch("api.model", mock_model):
        resp = client.post(
            "/predict/batch",
            json={"patients": [patient_payload] * 51},
            headers={"Authorization": f"Bearer {valid_token}"},
        )
    assert resp.status_code == 422


def test_predict_batch_empty(client, valid_token):
    mock_model = MagicMock()
    with patch("api.model", mock_model):
        resp = client.post(
            "/predict/batch",
            json={"patients": []},
            headers={"Authorization": f"Bearer {valid_token}"},
        )
    assert resp.status_code == 422