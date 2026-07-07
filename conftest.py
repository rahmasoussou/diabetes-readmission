import sys
import os

# Ajoute /app au path pour que les imports fonctionnent dans les containers
sys.path.insert(0, "/app")

# Variables d'environnement de test
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_testing_only")
os.environ.setdefault("DASHBOARD_PASSWORD", "test")
os.environ.setdefault("DASHBOARD_USERNAME", "test")
