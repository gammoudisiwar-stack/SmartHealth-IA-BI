"""
Couche de données PostgreSQL — base opérationnelle de l'agent + KPI métier.

C'est le SYSTÈME D'ENREGISTREMENT : chaque décision de l'agent est journalisée
dans `ledger` via une PROCÉDURE STOCKÉE (sp_log_decision). Les KPI métier sont
exposés comme des VUES SQL (vw_*) que Power BI interroge nativement (connecteur
PostgreSQL) — plus aucun fichier CSV intermédiaire.

Objets créés par init_schema() :
  Tables      : sessions, ledger
  Procédures  : sp_create_session(text,text,text), sp_log_decision(jsonb)
  Vues (KPI)  : vw_patient_ledger, vw_ops_kpis, vw_referral_funnel,
                vw_action_distribution, vw_case_mix, vw_urgency_mix,
                vw_sla_by_urgency, vw_equity_gap, vw_agent_daily

Connexion : variables PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD (.env).
"""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# Coefficient d'heures cliniciennes économisées / cas (depuis les hypothèses métier).
_HRS_COEF = (C.BUSINESS["minutes_revue_manuelle"] -
             C.BUSINESS["minutes_triage_auto"]) / 60.0

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    started_at  TIMESTAMP NOT NULL DEFAULT now(),
    channel     TEXT NOT NULL,
    operator    TEXT
);

CREATE TABLE IF NOT EXISTS ledger (
    event_id            BIGSERIAL PRIMARY KEY,
    session_id          TEXT REFERENCES sessions(session_id),
    ts                  TIMESTAMP NOT NULL DEFAULT now(),
    patient_ref         TEXT,
    sex                 TEXT,
    age                 DOUBLE PRECISION,
    age_band            TEXT,
    hgb                 DOUBLE PRECISION,
    rbc                 DOUBLE PRECISION,
    hct                 DOUBLE PRECISION,
    mcv                 DOUBLE PRECISION,
    mch                 DOUBLE PRECISION,
    mchc                DOUBLE PRECISION,
    mentzer_index       DOUBLE PRECISION,
    proba_anemie        DOUBLE PRECISION,
    prediction          INTEGER,
    risk_band           TEXT,
    severity            TEXT,
    anemia_type         TEXT,
    recommended_action  TEXT,
    urgency             TEXT,
    urgency_sla_h       INTEGER,
    handled_within_h    DOUBLE PRECISION,
    sla_met             INTEGER,
    triage_minutes      DOUBLE PRECISION,
    cost_avoided        DOUBLE PRECISION,
    oms_rule_flag       INTEGER,
    concordance         INTEGER,
    clinician_decision  TEXT,
    override_flag       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger (ts);
CREATE INDEX IF NOT EXISTS idx_ledger_session ON ledger (session_id);

-- Procédure : ouvrir / mettre à jour une session -------------------------- --
CREATE OR REPLACE PROCEDURE sp_create_session(p_id TEXT, p_channel TEXT, p_operator TEXT)
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO sessions (session_id, channel, operator)
    VALUES (p_id, p_channel, p_operator)
    ON CONFLICT (session_id) DO UPDATE
        SET channel = EXCLUDED.channel, operator = EXCLUDED.operator;
END; $$;

-- Procédure : journaliser une décision (charge utile JSONB) --------------- --
CREATE OR REPLACE PROCEDURE sp_log_decision(p JSONB)
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO ledger (
        session_id, ts, patient_ref, sex, age, age_band, hgb, rbc, hct, mcv, mch, mchc,
        mentzer_index, proba_anemie, prediction, risk_band, severity, anemia_type,
        recommended_action, urgency, urgency_sla_h, handled_within_h, sla_met,
        triage_minutes, cost_avoided, oms_rule_flag, concordance, clinician_decision,
        override_flag
    ) VALUES (
        p->>'session_id',
        COALESCE((p->>'ts')::timestamp, now()),
        p->>'patient_ref', p->>'sex', (p->>'age')::float8, p->>'age_band',
        (p->>'hgb')::float8, (p->>'rbc')::float8, (p->>'hct')::float8, (p->>'mcv')::float8,
        (p->>'mch')::float8, (p->>'mchc')::float8, (p->>'mentzer_index')::float8,
        (p->>'proba_anemie')::float8, (p->>'prediction')::int, p->>'risk_band',
        p->>'severity', p->>'anemia_type', p->>'recommended_action', p->>'urgency',
        (p->>'urgency_sla_h')::int, (p->>'handled_within_h')::float8, (p->>'sla_met')::int,
        (p->>'triage_minutes')::float8, (p->>'cost_avoided')::float8,
        (p->>'oms_rule_flag')::int, (p->>'concordance')::int, p->>'clinician_decision',
        (p->>'override')::int
    );
END; $$;

-- Vues KPI (consommées par Power BI) -------------------------------------- --
CREATE OR REPLACE VIEW vw_patient_ledger AS
    SELECT event_id, session_id, ts, patient_ref, sex, age, age_band, hgb, mentzer_index,
           proba_anemie, prediction, risk_band, severity, anemia_type, recommended_action,
           urgency, urgency_sla_h, handled_within_h, sla_met, cost_avoided,
           clinician_decision, override_flag
    FROM ledger;

CREATE OR REPLACE VIEW vw_ops_kpis AS
WITH agg AS (
    SELECT COUNT(*) AS n,
           COALESCE(SUM(prediction), 0) AS detected,
           AVG(prediction) AS taux,
           SUM(cost_avoided) AS cout,
           AVG(cost_avoided) AS cout_par_cas,
           AVG(sla_met) AS sla,
           AVG(handled_within_h) AS delai,
           AVG(override_flag) AS ovr,
           AVG(concordance) AS conc,
           COUNT(*) FILTER (WHERE urgency = 'Urgent') AS urgents
    FROM ledger
)
SELECT metrique, valeur, unite, note FROM (
    VALUES
      ('consultations_traitees',        (SELECT n::numeric FROM agg),                    'cas',    'activité de l''agent'),
      ('cas_anemie_detectes',           (SELECT detected::numeric FROM agg),             'cas',    'priorisés pour revue'),
      ('taux_detection',                (SELECT ROUND(taux::numeric, 3) FROM agg),       'ratio',  'part de cas positifs'),
      ('cout_evite_total',              (SELECT ROUND(cout::numeric, 0) FROM agg),       '€',      'sur la période'),
      ('cout_evite_par_cas',            (SELECT ROUND(cout_par_cas::numeric, 2) FROM agg), '€',    'moyenne / cas'),
      ('heures_clinicien_economisees',  (SELECT ROUND(n * {_HRS_COEF}, 1) FROM agg),     'heures', 'vs revue manuelle NFS'),
      ('taux_conformite_sla',           (SELECT ROUND(sla::numeric, 3) FROM agg),        'ratio',  'pris en charge dans les délais'),
      ('delai_prise_en_charge_h',       (SELECT ROUND(delai::numeric, 1) FROM agg),      'heures', 'moyenne réelle'),
      ('taux_override_clinicien',       (SELECT ROUND(ovr::numeric, 3) FROM agg),        'ratio',  'corrections de l''IA'),
      ('taux_concordance_oms',          (SELECT ROUND(conc::numeric, 3) FROM agg),       'ratio',  'IA vs règle OMS'),
      ('cas_urgents',                   (SELECT urgents::numeric FROM agg),              'cas',    'sévères à référer vite')
) AS t(metrique, valeur, unite, note);

CREATE OR REPLACE VIEW vw_referral_funnel AS
    SELECT 'Patients dépistés' AS etape, COUNT(*) AS n_patients, 1 AS ordre FROM ledger
    UNION ALL SELECT 'Anémie détectée', COUNT(*) FILTER (WHERE prediction = 1), 2 FROM ledger
    UNION ALL SELECT 'Orientés (bilan/référé)', COUNT(*) FILTER (WHERE prediction = 1), 3 FROM ledger
    UNION ALL SELECT 'Référés hématologie', COUNT(*) FILTER (WHERE recommended_action ILIKE '%hématologie%'), 4 FROM ledger
    UNION ALL SELECT 'Cas urgents', COUNT(*) FILTER (WHERE urgency = 'Urgent'), 5 FROM ledger;

CREATE OR REPLACE VIEW vw_action_distribution AS
    SELECT recommended_action, COUNT(*) AS n_patients,
           ROUND((COUNT(*)::numeric / NULLIF(SUM(COUNT(*)) OVER (), 0)), 3) AS part
    FROM ledger GROUP BY recommended_action ORDER BY n_patients DESC;

CREATE OR REPLACE VIEW vw_case_mix AS
    SELECT anemia_type, COUNT(*) AS n_patients,
           ROUND((COUNT(*)::numeric / NULLIF(SUM(COUNT(*)) OVER (), 0)), 3) AS part
    FROM ledger GROUP BY anemia_type ORDER BY n_patients DESC;

CREATE OR REPLACE VIEW vw_urgency_mix AS
    SELECT urgency, COUNT(*) AS n_patients,
           ROUND((COUNT(*)::numeric / NULLIF(SUM(COUNT(*)) OVER (), 0)), 3) AS part
    FROM ledger GROUP BY urgency;

CREATE OR REPLACE VIEW vw_sla_by_urgency AS
    SELECT urgency, COUNT(*) AS n_patients,
           ROUND(AVG(sla_met)::numeric, 3) AS taux_sla,
           ROUND(AVG(handled_within_h)::numeric, 1) AS delai_moyen_h
    FROM ledger GROUP BY urgency;

CREATE OR REPLACE VIEW vw_equity_gap AS
    SELECT age_band AS bande_age, sex AS sexe, COUNT(*) AS n_patients,
           ROUND(AVG(prediction)::numeric, 3) AS taux_detection_ia,
           ROUND(AVG(oms_rule_flag)::numeric, 3) AS taux_regle_oms,
           ROUND((AVG(prediction) - AVG(oms_rule_flag))::numeric, 3) AS ecart_ia_oms
    FROM ledger GROUP BY age_band, sex;

CREATE OR REPLACE VIEW vw_agent_daily AS
    SELECT ts::date AS jour, COUNT(*) AS consultations,
           COALESCE(SUM(prediction), 0) AS cas_detectes,
           ROUND(SUM(cost_avoided)::numeric, 0) AS cout_evite,
           ROUND(AVG(handled_within_h)::numeric, 1) AS delai_moyen_h,
           ROUND(AVG(sla_met)::numeric, 3) AS taux_sla
    FROM ledger GROUP BY ts::date ORDER BY jour;
"""


def connect() -> psycopg.Connection:
    """Ouvre une connexion PostgreSQL (autocommit)."""
    return psycopg.connect(C.db_dsn(), autocommit=True)


def init_schema(reset: bool = False) -> None:
    """Crée tables, procédures et vues. Si reset=True, vide les données."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        if reset:
            cur.execute(
                "TRUNCATE ledger RESTART IDENTITY CASCADE; DELETE FROM sessions;")


def _jsonable(d: dict) -> dict:
    """Convertit les scalaires NumPy en types Python natifs (sérialisables JSON)."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def create_session(session_id: str, channel: str, operator: str | None = None) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CALL sp_create_session(%s, %s, %s)",
                    (session_id, channel, operator))


def log_event(decision: dict) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CALL sp_log_decision(%s)", (Jsonb(_jsonable(decision)),))


def insert_events(events: list[dict]) -> int:
    if not events:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany("CALL sp_log_decision(%s)", [
                        (Jsonb(_jsonable(e)),) for e in events])
    return len(events)


def read_ledger():
    """Retourne le grand livre complet en DataFrame."""
    import pandas as pd
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM ledger ORDER BY ts", conn)


if __name__ == "__main__":
    init_schema()
    with connect() as conn, conn.cursor() as cur:
        n = cur.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
    print(
        f"Schéma PostgreSQL prêt (base '{C.DB['dbname']}') — {n} événements.")
