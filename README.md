# 🏥 Prédiction de Réhospitalisation des Patients Diabétiques

Projet de stage — Pipeline complet ML avec Docker, sécurité JWT et dashboard Streamlit.

---

## Architecture

```
postgres  ──→  etl  ──→  ml-service (FastAPI)  ──→  streamlit
   :5432           (pipeline)      :8000                :8501
```

Tous les services tournent sur le réseau Docker isolé `diabetes-net`.

---


## Méthodologie (v3) — toutes les rencontres + split par patient

Depuis la v3, le pipeline a évolué sur deux points, validés par expérimentation
(voir `experiments/`) :

- **ETL (`etl/pipeline.py`)** : toutes les rencontres sont conservées (plus de
  déduplication par patient). Un même patient peut avoir plusieurs lignes en
  base (une par hospitalisation).
- **Entraînement (`ml/train.py`)** : pour éviter toute fuite de données, le
  split train/validation/test (fit/calib/test) est **groupé par
  `patient_nbr`** (`GroupShuffleSplit`) — un même patient ne se retrouve
  jamais dans deux sous-ensembles différents. L'early stopping s'appuie
  uniquement sur le jeu de calibration, jamais sur le test.

Résultats (validation croisée `GroupKFold`, 5 plis) :

| Métrique | Valeur |
|---|---|
| AUC-ROC | 0,6402 ± 0,0085 |
| AUC-PR | 0,2034 ± 0,0068 |
| Recall parmi les 10 % de patients les plus à risque | 0,2121 |

Ce gain a été vérifié comme provenant du signal apporté par les patients
récurrents (leur historique d'hospitalisations), et non simplement du volume
de données supplémentaire — voir le test de contrôle dans `experiments/`.
---

## Démarrage rapide

### 1. Prérequis
- Docker Desktop installé et lancé
- Python 3.11+ (pour les tests locaux)
- Dataset Kaggle : https://www.kaggle.com/datasets/brandao/diabetes

### 2. Configuration
```bash
# Copier le fichier de configuration
cp .env.example .env

# Générer une clé JWT sécurisée
python -c "import secrets; print(secrets.token_hex(32))"
# → Coller la valeur dans .env à JWT_SECRET_KEY

# Modifier les mots de passe dans .env
```

### 3. Placer les données
```
data/raw/diabetic_data.csv   ← fichier téléchargé depuis Kaggle
```

### 4. Lancer tout le projet
```bash
# Construire et démarrer tous les services
docker-compose up --build -d

# Vérifier que tout tourne
docker-compose ps

# Étape 1 : Charger les données (ETL)
docker-compose exec etl python pipeline.py

# Étape 2 : Entraîner le modèle
docker-compose exec ml-service python train.py

# Étape 3 : Créer le premier compte praticien (table `users` vide sur une install neuve)
docker-compose exec ml-service python /app/scripts/create_user.py --username medecin --role medecin
# → mot de passe demandé de façon masquée (jamais en argument CLI)

# Dashboard disponible sur :
# → http://localhost:8501
# → Utilisateur : medecin  |  Mot de passe : (celui saisi à l'étape 3)
```

> ⚠️ Sur une installation neuve, la table `users` est vide : tant que l'étape 3
> n'a pas été exécutée, toute tentative de connexion à l'API ou au dashboard
> échouera (401), même avec des identifiants qui semblent corrects.

---

## Structure du projet

```
diabetes-readmission/
├── docker-compose.yml        ← orchestration des services
├── .env                      ← secrets (JAMAIS committé)
├── .env.example              ← modèle sans secrets (committé)
├── .gitignore
│
├── db/
│   └── init.sql              ← schéma PostgreSQL
│
├── etl/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pipeline.py           ← chargement + nettoyage + DB
│
├── ml/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── features.py           ← feature engineering (partagé)
│   ├── train.py              ← entraînement XGBoost + SHAP
│   └── api.py                ← FastAPI + JWT + rate limiting
│
├── dashboard/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                ← Streamlit
│
├── experiments/               ← scripts d'expérimentation (encodage, split
│                                  par patient, GroupKFold, test de contrôle)
│
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_pipeline.py
│   └── test_train.py
│
├── data/raw/                 ← données brutes (non committé)
└── models/                   ← modèles entraînés (non committé)
```

---

## Sécurité

| Couche | Mesure |
|--------|--------|
| Secrets | `.env` jamais committé, variables Docker |
| API | JWT HS256, expiration 1h |
| Rate limiting | 30 req/min sur `/predict`, 10/min sur `/token` |
| Base de données | Rôle `readonly`, extension `pgcrypto` |
| Réseau | Docker network isolé `diabetes-net` |
| CORS | Uniquement `localhost:8501` autorisé |
| Traçabilité | Toutes les prédictions enregistrées en DB |

---

## Commandes utiles

```bash
# Logs d'un service
docker-compose logs -f ml-service

# Accéder à la base de données
docker-compose exec postgres psql -U diabetes_user -d diabetes_db

# Consulter les prédictions
# (dans psql) SELECT * FROM predictions ORDER BY predicted_at DESC LIMIT 10;

# Lancer les tests (les tests importent train.py/api.py via /app,
# ils doivent donc tourner DANS le conteneur ml-service, pas en local)
docker-compose exec ml-service pytest -v

# Arrêter tous les services
docker-compose down

# Arrêter et supprimer les volumes (reset complet)
docker-compose down -v
```

---

## Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**
- Source : https://www.kaggle.com/datasets/brandao/diabetes
- 101 766 séjours hospitaliers bruts, 99 343 conservés après nettoyage
  (toutes les rencontres, sans déduplication par patient depuis la v3)
- 130 hôpitaux américains
- 50+ variables cliniques par patient

---

## Stack technique

- **Python** 3.11 — Pandas, scikit-learn, XGBoost, SHAP
- **PostgreSQL** 15 — stockage structuré
- **FastAPI** — API REST sécurisée
- **Streamlit** — interface praticien
- **Docker** — conteneurisation complète
- **JWT** — authentification sans état