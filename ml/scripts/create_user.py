"""
Créer / mettre à jour un compte praticien dans la table `users`.
================================================================
Usage :
  docker-compose exec ml-service python /app/ml/scripts/create_user.py \
      --username medecin --role medecin

  (le mot de passe est demandé de façon masquée, jamais en argument CLI)
"""
import argparse
import getpass
import os
import sys

from passlib.context import CryptContext
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_engine():
    u = os.environ["POSTGRES_USER"]
    p = os.environ["POSTGRES_PASSWORD"]
    h = os.environ["POSTGRES_HOST"]
    d = os.environ["POSTGRES_DB"]
    return create_engine(f"postgresql://{u}:{p}@{h}/{d}")


def main():
    parser = argparse.ArgumentParser(description="Créer/mettre à jour un compte praticien")
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", default="medecin", choices=["medecin", "admin"])
    args = parser.parse_args()

    password = getpass.getpass("Mot de passe : ")
    confirm = getpass.getpass("Confirmer le mot de passe : ")
    if password != confirm:
        print("❌ Les mots de passe ne correspondent pas.")
        sys.exit(1)
    if len(password) < 10:
        print("❌ Le mot de passe doit faire au moins 10 caractères.")
        sys.exit(1)

    password_hash = pwd_context.hash(password)
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO users (username, password_hash, role, is_active)
                VALUES (:username, :hash, :role, TRUE)
                ON CONFLICT (username)
                DO UPDATE SET password_hash = :hash, role = :role, is_active = TRUE
            """),
            {"username": args.username, "hash": password_hash, "role": args.role},
        )
        conn.commit()

    print(f"✓ Compte '{args.username}' ({args.role}) créé/mis à jour avec succès.")


if __name__ == "__main__":
    main()
