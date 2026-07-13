"""
Couche « procédures » — point d'accès unique aux données de l'agent et aux KPI.

S'appuie sur les PROCÉDURES STOCKÉES et les VUES PostgreSQL définies dans db.py.
L'agent, l'interface et Power BI passent par ici : les KPI métier sont donc
réellement ENREGISTRÉS (procédures) puis RELUS (vues) depuis la base.

  sp_create_session(session_id, channel, operator)   -> CALL sp_create_session
  sp_list_sessions()                                 -> sessions + nb d'événements
  sp_session_ledger(session_id)                      -> décisions d'une session
  sp_log_decision(decision)                          -> CALL sp_log_decision
  sp_get_kpis(session_id=None)                        -> KPI métier (vue vw_ops_kpis)
"""
from __future__ import annotations

import sys
from pathlib import Path

from psycopg.rows import dict_row

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import db  # noqa: E402


def sp_create_session(session_id: str, channel: str = "chat",
                      operator: str | None = None) -> str:
    db.init_schema()
    db.create_session(session_id, channel, operator)
    return session_id


def sp_list_sessions() -> list[dict]:
    db.init_schema()
    with db.connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT s.session_id, "
            "       to_char(s.started_at, 'YYYY-MM-DD HH24:MI') AS started_at, "
            "       s.channel, s.operator, COUNT(l.event_id) AS n_events "
            "FROM sessions s LEFT JOIN ledger l ON l.session_id = s.session_id "
            "GROUP BY s.session_id, s.started_at, s.channel, s.operator "
            "ORDER BY s.started_at DESC"
        ).fetchall()


def sp_session_ledger(session_id: str):
    import pandas as pd
    with db.connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM ledger WHERE session_id = %(sid)s ORDER BY ts",
            conn, params={"sid": session_id})


def sp_log_decision(decision: dict) -> None:
    db.log_event(decision)


def sp_get_kpis(session_id: str | None = None) -> dict:
    """KPI métier. Global -> vue vw_ops_kpis ; par session -> agrégat filtré."""
    with db.connect() as conn, conn.cursor() as cur:
        if session_id is None:
            rows = cur.execute(
                "SELECT metrique, valeur FROM vw_ops_kpis").fetchall()
            kpis = {m: float(v) if v is not None else None for m, v in rows}
            if not kpis or kpis.get("consultations_traitees", 0) == 0:
                return {"consultations_traitees": 0, "message": "Aucune activité enregistrée."}
            return kpis

        coef = (C.BUSINESS["minutes_revue_manuelle"] -
                C.BUSINESS["minutes_triage_auto"]) / 60.0
        row = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(prediction), 0), AVG(prediction), "
            "       SUM(cost_avoided), AVG(sla_met), AVG(override_flag), AVG(concordance), "
            "       COUNT(*) FILTER (WHERE urgency = 'Urgent') "
            "FROM ledger WHERE session_id = %s", (session_id,)).fetchone()
    n = row[0] or 0
    if n == 0:
        return {"consultations_traitees": 0, "message": "Aucune activité pour cette session."}
    return {
        "consultations_traitees": n,
        "cas_anemie_detectes": int(row[1]),
        "taux_detection": round(float(row[2]), 3),
        "cout_evite_total": round(float(row[3] or 0), 0),
        "heures_clinicien_economisees": round(n * coef, 1),
        "taux_conformite_sla": round(float(row[4]), 3) if row[4] is not None else None,
        "taux_override_clinicien": round(float(row[5]), 3) if row[5] is not None else None,
        "taux_concordance_oms": round(float(row[6]), 3) if row[6] is not None else None,
        "cas_urgents": int(row[7]),
    }
