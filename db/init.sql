CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'diabetes_readonly') THEN
    CREATE ROLE diabetes_readonly;
  END IF;
END $$;

GRANT CONNECT ON DATABASE diabetes_db TO diabetes_readonly;

CREATE TABLE IF NOT EXISTS patients (
    id                      SERIAL PRIMARY KEY,
    encounter_id            INTEGER UNIQUE NOT NULL,
    patient_nbr             INTEGER NOT NULL,
    race                    VARCHAR(50),
    gender                  VARCHAR(20),
    age                     VARCHAR(20),
    age_num                 INTEGER,
    time_in_hospital        INTEGER,
    admission_type          INTEGER,
    discharge_type          INTEGER,
    admission_source        INTEGER,
    num_lab_procedures      INTEGER,
    num_procedures          INTEGER,
    num_medications         INTEGER,
    number_diagnoses        INTEGER,
    num_outpatient          INTEGER DEFAULT 0,
    num_emergency           INTEGER DEFAULT 0,
    num_inpatient           INTEGER DEFAULT 0,

    -- Diagnostics ICD-9
    diag_1                  VARCHAR(20),
    diag_2                  VARCHAR(20),
    diag_3                  VARCHAR(20),

    -- Résultats cliniques
    a1c_result              VARCHAR(20),
    glucose_serum           VARCHAR(20),
    change_in_meds          VARCHAR(10),
    diabetes_meds           VARCHAR(10),

    -- Médicaments (21 colonnes)
    metformin               VARCHAR(10),
    repaglinide             VARCHAR(10),
    nateglinide             VARCHAR(10),
    chlorpropamide          VARCHAR(10),
    glimepiride             VARCHAR(10),
    acetohexamide           VARCHAR(10),
    glipizide               VARCHAR(10),
    glyburide                VARCHAR(10),
    tolbutamide              VARCHAR(10),
    pioglitazone             VARCHAR(10),
    rosiglitazone            VARCHAR(10),
    acarbose                 VARCHAR(10),
    miglitol                 VARCHAR(10),
    troglitazone             VARCHAR(10),
    tolazamide                VARCHAR(10),
    insulin                   VARCHAR(10),
    glyburide_metformin       VARCHAR(10),
    glipizide_metformin       VARCHAR(10),
    glimepiride_pioglitazone  VARCHAR(10),
    metformin_rosiglitazone   VARCHAR(10),
    metformin_pioglitazone    VARCHAR(10),

    -- Features dérivées ETL
    meds_per_day             FLOAT,
    total_visits              INTEGER,

    -- Cible
    readmitted                VARCHAR(10),
    readmitted_30              INTEGER,
    created_at                 TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    encounter_id    INTEGER REFERENCES patients(encounter_id),
    risk_score      FLOAT NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_level      VARCHAR(10),
    model_version   VARCHAR(20) NOT NULL,
    top_factors     JSONB,
    requested_by    VARCHAR(50),
    predicted_at    TIMESTAMP DEFAULT NOW()
);

-- Table users : désormais réellement utilisée par l'API (voir ml/api.py)
-- password_hash est un hash bcrypt (jamais de mot de passe en clair)
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    role            VARCHAR(20) DEFAULT 'medecin',
    is_active       BOOLEAN DEFAULT TRUE,
    last_login      TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Journal d'audit : qui a consulté/quoi, au-delà des seules prédictions
CREATE TABLE IF NOT EXISTS audit_log (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(50) NOT NULL,
    action          VARCHAR(50) NOT NULL,   -- ex: LOGIN, PREDICT, PREDICT_BATCH, VIEW_HISTORY, DOWNLOAD_PDF
    detail          JSONB,
    ip_address      VARCHAR(64),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patients_readmitted   ON patients(readmitted_30);
CREATE INDEX IF NOT EXISTS idx_patients_age          ON patients(age_num);
CREATE INDEX IF NOT EXISTS idx_patients_insulin      ON patients(insulin);
CREATE INDEX IF NOT EXISTS idx_predictions_encounter ON predictions(encounter_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date      ON predictions(predicted_at);
CREATE INDEX IF NOT EXISTS idx_users_username        ON users(username);
CREATE INDEX IF NOT EXISTS idx_audit_username_date   ON audit_log(username, created_at);

GRANT SELECT ON ALL TABLES IN SCHEMA public TO diabetes_readonly;

CREATE OR REPLACE VIEW v_patient_risk AS
SELECT
    p.encounter_id, p.age, p.gender, p.time_in_hospital,
    p.num_medications, p.a1c_result, p.insulin, p.readmitted,
    pr.risk_score, pr.risk_level, pr.predicted_at
FROM patients p
LEFT JOIN predictions pr ON p.encounter_id = pr.encounter_id
ORDER BY pr.risk_score DESC NULLS LAST;
