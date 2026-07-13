"""
Configuration centrale du projet AI-BI : Détection d'anémie par transfer learning.

Toutes les constantes (chemins, schéma de features, seuils cliniques OMS,
hypothèses économiques pour les métriques métier) sont centralisées ici afin
d'être facilement auditables et modifiables par le jury.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Charge les variables d'environnement depuis .env (clés IA + Postgres), si dispo.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"          # CSV consommés par Power BI
REPORTS_DIR = BASE_DIR / "reports"          # figures (PNG)

for _d in (PROCESSED_DIR, MODELS_DIR, EXPORTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Fichier clinique (cible de fine-tuning) : 1004 patients réels, Chattogram.
CLINICAL_XLSX = DATA_DIR / "anemia_raw.xlsx"
# Dossier du dataset de pré-entraînement (Pediatric Anemia, Mendeley y7v7ff3wpj).
PRETRAIN_DIR = DATA_DIR / "pretrain_pediatric_x"

# --------------------------------------------------------------------------- #
# Base PostgreSQL (opérationnelle + KPI métier consommés par Power BI)
# --------------------------------------------------------------------------- #
DB = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5432"),
    "dbname": os.getenv("PGDATABASE", "anemia"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}


def db_dsn() -> str:
    """Chaîne de connexion libpq (psycopg / langgraph)."""
    return (f"host={DB['host']} port={DB['port']} dbname={DB['dbname']} "
            f"user={DB['user']} password={DB['password']}")


def db_uri() -> str:
    """URI PostgreSQL (checkpointer langgraph PostgresSaver)."""
    return (f"postgresql://{DB['user']}:{DB['password']}@{DB['host']}:{DB['port']}/"
            f"{DB['dbname']}")


# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Schéma commun (aligné entre les deux datasets)
# --------------------------------------------------------------------------- #
# Features brutes partagées par les deux jeux de données.
BASE_FEATURES = ["sex", "age", "hgb", "rbc", "hct", "mcv", "mch", "mchc"]
# Feature dérivée cliniquement motivée : indice de Mentzer = MCV / RBC
# (utilisé pour distinguer carence en fer vs thalassémie).
DERIVED_FEATURES = ["mentzer_index"]
MODEL_FEATURES = BASE_FEATURES + DERIVED_FEATURES
# Features continues à standardiser (sex reste binaire 0/1).
CONTINUOUS_FEATURES = ["age", "hgb", "rbc",
                       "hct", "mcv", "mch", "mchc", "mentzer_index"]

TARGET = "target"           # 0 = non anémique, 1 = anémique

# Renommage des colonnes brutes -> schéma commun.
CLINICAL_RENAME = {
    "Gender": "sex",
    "Age": "age",
    "HGB(Hemoglobin)": "hgb",
    "RBC": "rbc",
    "PCV/HCT": "hct",
    "MCV": "mcv",
    "MCH": "mch",
    "MCHC": "mchc",
    "Decision_Class": "target",
}
PRETRAIN_RENAME = {
    "Gender": "sex",
    "Age": "age",
    "Hb": "hgb",
    "RBC": "rbc",
    "PCV": "hct",
    "MCV": "mcv",
    "MCH": "mch",
    "MCHC": "mchc",
    "Decision_Class": "target",
}

# --------------------------------------------------------------------------- #
# Seuils cliniques OMS (g/dL) — hémoglobine
# --------------------------------------------------------------------------- #
# Seuil d'anémie par sexe (adulte).
HB_ANEMIA_THRESHOLD = {"M": 13.0, "F": 12.0}
# Bandes de sévérité OMS (simplifiées, adulte) utilisées pour l'entonnoir métier.
#   Sévère   : Hb < 8.0
#   Modérée  : 8.0 <= Hb < 10.0
#   Légère   : 10.0 <= Hb < seuil d'anémie (sexe)
#   Aucune   : Hb >= seuil d'anémie
HB_SEVERE = 8.0
HB_MODERATE = 10.0

# Bornes physiologiques plausibles (garde-fou qualité des données).
PLAUSIBLE_RANGES = {
    "age": (0, 120),
    "hgb": (2.0, 25.0),
    "rbc": (1.0, 8.0),
    "hct": (10.0, 65.0),
    "mcv": (50.0, 130.0),
    "mch": (10.0, 45.0),
    "mchc": (25.0, 40.0),
}

# --------------------------------------------------------------------------- #
# Hypothèses économiques (illustratives, éditables) — métriques métier
# --------------------------------------------------------------------------- #
# Ces valeurs sont des HYPOTHÈSES explicites pour chiffrer l'impact métier.
BUSINESS = {
    # temps clinicien / interprétation CBC manuelle
    "minutes_revue_manuelle": 8.0,
    "minutes_triage_auto": 0.5,          # temps de revue d'un cas priorisé par l'IA
    "cout_clinicien_par_minute": 0.60,   # € / minute (coût chargé)
    "cout_inference_ia_par_cas": 0.002,  # € / prédiction (compute)
    # € / bilan martial de confirmation (faux positifs)
    "cout_test_confirmation": 15.0,
    # € / cas manqué (faux négatif : complications)
    "cout_cas_manque": 120.0,
}

# Split
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # part du train restant utilisée en validation
