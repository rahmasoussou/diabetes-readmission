"""
Tests unitaires — train.py (fonctions utilitaires uniquement)
================================================================
On ne lance pas un entraînement XGBoost complet dans les tests (trop lourd/lent),
on teste uniquement les fonctions pures : seuil optimal, hash de traçabilité.
Lancer avec : docker-compose exec ml-service pytest tests/test_train.py -v
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "test")

sys.path.insert(0, "/app")

from train import find_best_threshold, compute_data_hash


def test_find_best_threshold_returns_value_in_range():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.2, size=500), 0, 1)
    threshold = find_best_threshold(y_true, y_proba)
    assert 0.0 <= threshold <= 1.0


def test_find_best_threshold_perfect_separation():
    y_true  = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.95])
    threshold = find_best_threshold(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    assert (y_pred == y_true).all()


def test_compute_data_hash_is_deterministic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    h1 = compute_data_hash(df)
    h2 = compute_data_hash(df)
    assert h1 == h2
    assert len(h1) == 10


def test_compute_data_hash_changes_with_data():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3, 4]})
    assert compute_data_hash(df1) != compute_data_hash(df2)