"""
Agent ReAct (LangChain) de triage de l'anémie — AnemiaCareAgent.

Architecture (calquée sur l'exemple exam_chat_agent) :
  - Modèle de raisonnement : LLM (Azure OpenAI / OpenAI) via llm.get_chat_model().
  - Agent ReAct : langchain.agents.create_agent — le LLM RAISONNE et choisit
    quels OUTILS appeler (rien n'est codé en dur, aucun if/else de workflow).
  - Mémoire de threads : checkpointer SQLite (langgraph) -> chaque session
    (thread_id) est PERSISTÉE et peut être reprise / listée dans l'interface.
  - Outils (@tool) :
        classify_anemia(...)    -> pipeline DÉTERMINISTE (modèle finetuned.keras
                                   + indice de Mentzer + parcours de soins),
                                   journalise la décision dans la base.
        get_operational_kpis()  -> KPI métier RELUS depuis la base (procedures).
  Le LLM n'invente jamais de diagnostic : il délègue aux outils, source de vérité.

CLI :
  python src/care_agent.py --bootstrap          # remplit la base (hors-ligne)
  python src/care_agent.py --message "Femme 62 ans, Hb 8.1, RBC 3.4, HCT 27, MCV 79, MCH 24, MCHC 31"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import db  # noqa: E402
import llm  # noqa: E402
import procedures as proc  # noqa: E402
from predict import predict_frame  # noqa: E402

# Indice de Mentzer : seuil classique de distinction du différentiel.
MENTZER_THRESHOLD = 13.0
# Cibles de délai (heures) de prise en charge par niveau d'urgence.
SLA_HOURS = {"Urgent": 4, "Prioritaire": 24, "Standard": 72, "Aucun": 168}
_REQUIRED = ["sex", "age", "hgb", "rbc", "hct", "mcv", "mch", "mchc"]


def severity_band(hgb: float, sex: str) -> str:
    """Bande de sévérité OMS (simplifiée, adulte) à partir de l'hémoglobine."""
    thr = C.HB_ANEMIA_THRESHOLD.get(sex, 12.0)
    if hgb < C.HB_SEVERE:
        return "Sévère"
    if hgb < C.HB_MODERATE:
        return "Modérée"
    if hgb < thr:
        return "Légère"
    return "Aucune"


def age_band(age: float) -> str:
    """Tranche d'âge (cohortes du cockpit)."""
    for lo, hi, label in [(0, 12, "0-12"), (13, 18, "13-18"), (19, 35, "19-35"),
                          (36, 50, "36-50"), (51, 65, "51-65"), (66, 200, "66+")]:
        if lo <= age <= hi:
            return label
    return "NA"


# --------------------------------------------------------------------------- #
# Outils cliniques déterministes (source de vérité, auditables)
# --------------------------------------------------------------------------- #
def differential(prediction: int, mentzer: float) -> str:
    """Type d'anémie probable à partir de l'indice de Mentzer."""
    if prediction == 0:
        return "Aucune"
    if mentzer is None or np.isnan(mentzer):
        return "Indéterminé"
    return "Thalassémie suspectée" if mentzer < MENTZER_THRESHOLD else "Carence en fer probable"


def recommend_pathway(prediction: int, severity: str, risk: str,
                      anemia_type: str) -> tuple[str, str]:
    """Retourne (action recommandée, niveau d'urgence)."""
    if severity == "Sévère":
        return "Référer en urgence (transfusion à évaluer)", "Urgent"
    if prediction == 1:
        if anemia_type == "Thalassémie suspectée":
            return "Référer en hématologie (électrophorèse de l'Hb)", "Prioritaire"
        if severity == "Modérée":
            return "Bilan martial + prescription fer, recontrôle 4 sem.", "Prioritaire"
        return "Bilan martial (ferritine, CRP), conseil diététique", "Standard"
    if risk == "Modéré":
        return "Surveillance : recontrôle NFS à 3 mois", "Standard"
    return "Aucune action — résultat rassurant", "Aucun"


def economics(prediction: int) -> float:
    """€ économisés sur ce cas (temps clinicien évité + complications évitées)."""
    b = C.BUSINESS
    minutes_saved = b["minutes_revue_manuelle"] - b["minutes_triage_auto"]
    saved = minutes_saved * \
        b["cout_clinicien_par_minute"] - b["cout_inference_ia_par_cas"]
    if prediction == 1:  # cas détecté et orienté tôt -> fraction de complication évitée
        saved += 0.15 * b["cout_cas_manque"]
    return round(saved, 2)


def score_patient(patient: dict, session_id: str, ts: datetime | None = None,
                  patient_ref: str | None = None, persist: bool = True) -> dict:
    """Pipeline déterministe complet sur un patient. Journalise si persist=True."""
    scored = predict_frame(pd.DataFrame([patient])).iloc[0]
    sex = str(patient.get("sex", "F")).strip().upper()[:1]
    age = float(patient.get("age", np.nan))
    mentzer = float(patient["mcv"]) / float(patient["rbc"]
                                            ) if patient.get("rbc") else np.nan
    prediction = int(scored["prediction"])
    risk = str(scored["bande_risque"])
    severity = severity_band(float(patient["hgb"]), sex)
    anemia_type = differential(prediction, mentzer)
    action, urgency = recommend_pathway(
        prediction, severity, risk, anemia_type)
    oms_flag = int(float(patient["hgb"]) <
                   C.HB_ANEMIA_THRESHOLD.get(sex, 12.0))

    decision = {
        "session_id": session_id,
        "ts": (ts or datetime.now()).isoformat(timespec="seconds"),
        "patient_ref": patient_ref or f"P-{uuid.uuid4().hex[:6]}",
        "sex": sex, "age": age,
        "age_band": age_band(age) if not np.isnan(age) else "NA",
        "hgb": float(patient["hgb"]), "rbc": float(patient["rbc"]),
        "hct": float(patient["hct"]), "mcv": float(patient["mcv"]),
        "mch": float(patient["mch"]), "mchc": float(patient["mchc"]),
        "mentzer_index": round(float(mentzer), 2),
        "proba_anemie": round(float(scored["proba_anemie"]), 4),
        "prediction": prediction, "risk_band": risk, "severity": severity,
        "anemia_type": anemia_type, "recommended_action": action,
        "urgency": urgency, "urgency_sla_h": SLA_HOURS[urgency],
        "triage_minutes": C.BUSINESS["minutes_triage_auto"],
        "cost_avoided": economics(prediction),
        "oms_rule_flag": oms_flag, "concordance": int(prediction == oms_flag),
    }
    if persist:
        proc.sp_log_decision(decision)
    return decision


def narrate(d: dict) -> str:
    """Résumé lisible d'une décision (repli hors-ligne / CLI)."""
    verdict = "ANÉMIE probable" if d["prediction"] == 1 else "pas d'anémie"
    lines = [
        f"🩺 Patient {d['sex']}, {int(d['age'])} ans — Hb {d['hgb']} g/dL",
        f"   • Probabilité d'anémie : {d['proba_anemie']*100:.0f} %  → {verdict} (risque {d['risk_band']})",
        f"   • Sévérité OMS : {d['severity']}",
    ]
    if d["prediction"] == 1:
        lines.append(
            f"   • Différentiel (Mentzer {d['mentzer_index']}) : {d['anemia_type']}")
    lines += [
        f"   • Action recommandée : {d['recommended_action']}",
        f"   • Urgence : {d['urgency']} (à voir sous {d['urgency_sla_h']} h)",
    ]
    if not d["concordance"]:
        lines.append(
            "   ⚠️ Divergence IA / règle OMS — à revoir par un clinicien.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Extraction hors-ligne (repli si aucun LLM configuré)
# --------------------------------------------------------------------------- #
_FIELD_PATTERNS = {
    "hgb": r"(?:hgb|h[ée]moglobine|hb)\D{0,6}(\d{1,2}(?:[.,]\d)?)",
    "rbc": r"(?:rbc|gr|h[ée]maties)\D{0,6}(\d(?:[.,]\d{1,2})?)",
    "hct": r"(?:hct|h[ée]matocrite|pcv)\D{0,6}(\d{2}(?:[.,]\d)?)",
    "mcv": r"(?:mcv|vgm)\D{0,6}(\d{2,3}(?:[.,]\d)?)",
    "mch": r"(?:mch|tcmh)\D{0,6}(\d{2}(?:[.,]\d)?)",
    "mchc": r"(?:mchc|ccmh)\D{0,6}(\d{2}(?:[.,]\d)?)",
    "age": r"(?:âge|age|ans)\D{0,4}(\d{1,3})",
}


def extract_regex(text: str) -> dict:
    """Repli hors-ligne : extrait {sex, age, hgb, ...} par expressions régulières."""
    t = text.lower()
    out: dict = {}
    m_sex = re.search(
        r"\b(femme|f[ée]minin|female|\bf\b|homme|masculin|male|\bm\b)", t)
    if m_sex:
        out["sex"] = "M" if m_sex.group(1)[0] in ("h", "m") else "F"
    for field, pat in _FIELD_PATTERNS.items():
        m = re.search(pat, t)
        if m:
            out[field] = float(m.group(1).replace(",", "."))
    return out


# --------------------------------------------------------------------------- #
# Invite système de l'agent
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "Tu es un assistant clinique de triage de l'anémie, destiné à des soignants. "
    "Ton rôle : dialoguer en français, comprendre les valeurs d'hémogramme (NFS) "
    "fournies en texte libre ou issues d'un rapport de laboratoire océrisé, et "
    "demander poliment celles qui manquent parmi : sexe (M/F), âge, hémoglobine (hgb), "
    "globules rouges (rbc), hématocrite (hct), VGM (mcv), TCMH (mch), CCMH (mchc).\n\n"
    "Règles STRICTES :\n"
    "1. Tu ne poses JAMAIS de diagnostic toi-même. Dès que les 8 valeurs sont connues, "
    "tu APPELLES l'outil `classify_anemia` (modèle IA + règles cliniques = source de vérité).\n"
    "2. Tu n'inventes AUCUNE valeur biologique ; si une valeur manque, tu la demandes.\n"
    "3. Pour toute question sur l'activité, la performance ou les coûts (« combien de "
    "patients aujourd'hui ? », « taux de détection ? »), tu appelles `get_operational_kpis`.\n"
    "4. Après un outil, explique le résultat en langage clair et prudent, et rappelle "
    "qu'un clinicien valide la décision finale."
)


# --------------------------------------------------------------------------- #
# Agent ReAct
# --------------------------------------------------------------------------- #
class AnemiaCareAgent:
    """Agent ReAct LangChain avec mémoire de threads persistée (SQLite)."""

    def __init__(self, temperature: float = 0.1) -> None:
        self._temperature = temperature
        self._agent = None
        self._checkpointer = None
        self._active_session: str | None = None
        db.init_schema()

    # -- Outils (fabriques fermées sur self, à la manière de l'exemple) ----- #
    def _make_classify_tool(self):
        from langchain_core.tools import tool

        @tool
        def classify_anemia(
            sex: Annotated[str, "Sexe : 'M' ou 'F'"],
            age: Annotated[float, "Âge en années"],
            hgb: Annotated[float, "Hémoglobine (g/dL)"],
            rbc: Annotated[float, "Globules rouges (10^6/µL)"],
            hct: Annotated[float, "Hématocrite (%)"],
            mcv: Annotated[float, "VGM / MCV (fL)"],
            mch: Annotated[float, "TCMH / MCH (pg)"],
            mchc: Annotated[float, "CCMH / MCHC (g/dL)"],
        ) -> str:
            """Exécute le pipeline déterministe de triage de l'anémie (modèle IA
            finetuned.keras + indice de Mentzer + parcours de soins) et JOURNALISE
            la décision dans la base. À appeler uniquement quand les 8 valeurs sont
            connues. Retourne la décision structurée (JSON)."""
            patient = {"sex": sex, "age": age, "hgb": hgb, "rbc": rbc,
                       "hct": hct, "mcv": mcv, "mch": mch, "mchc": mchc}
            d = score_patient(
                patient, session_id=self._active_session or "S-ADHOC")
            keep = ("patient_ref", "proba_anemie", "prediction", "risk_band",
                    "severity", "anemia_type", "recommended_action", "urgency",
                    "urgency_sla_h", "mentzer_index", "cost_avoided", "concordance")
            return json.dumps({k: d[k] for k in keep}, ensure_ascii=False)

        return classify_anemia

    def _make_kpis_tool(self):
        from langchain_core.tools import tool

        @tool
        def get_operational_kpis() -> str:
            """Retourne les KPI métier OPÉRATIONNELS courants, relus depuis la base :
            nombre de consultations, cas détectés, taux de détection, coût évité,
            heures cliniciennes économisées, conformité SLA, taux d'override. JSON."""
            return json.dumps(proc.sp_get_kpis(), ensure_ascii=False)

        return get_operational_kpis

    # -- Construction paresseuse de l'agent -------------------------------- #
    @property
    def agent(self):
        if self._agent is not None:
            return self._agent
        from langchain.agents import create_agent

        try:  # threads persistés dans PostgreSQL (mémoire de conversation)
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
            from psycopg.rows import dict_row
            conn = Connection.connect(
                C.db_uri(), autocommit=True, prepare_threshold=0, row_factory=dict_row)
            self._checkpointer = PostgresSaver(conn)
            self._checkpointer.setup()
        except Exception:
            from langgraph.checkpoint.memory import InMemorySaver
            self._checkpointer = InMemorySaver()

        middleware = []
        try:  # résumé automatique des longues conversations (comme l'exemple)
            from langchain.agents.middleware import SummarizationMiddleware
            middleware = [SummarizationMiddleware(
                model=llm.get_chat_model(temperature=0.0),
                trigger=[("tokens", 80000), ("messages", 50)],
                keep=("messages", 10),
            )]
        except Exception:
            middleware = []

        self._agent = create_agent(
            model=llm.get_chat_model(self._temperature),
            tools=[self._make_classify_tool(), self._make_kpis_tool()],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
            middleware=middleware,
        )
        return self._agent

    # -- Sessions ----------------------------------------------------------- #
    def new_session(self, channel: str = "chat", operator: str | None = None) -> str:
        """Crée une nouvelle session (thread) et retourne son identifiant."""
        sid = f"S-{uuid.uuid4().hex[:8]}"
        proc.sp_create_session(sid, channel=channel, operator=operator)
        return sid

    def invoke(self, message: str, thread_id: str) -> str:
        """Traite un message dans un thread donné. Persiste via le checkpointer."""
        self._active_session = thread_id
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke({"messages": [{"role": "user", "content": message}]},
                                   config)
        messages = result.get("messages", [])
        return getattr(messages[-1], "content", "") if messages else ""

    def ingest_document(self, file_bytes: bytes, thread_id: str) -> str:
        """Océrise un rapport NFS puis laisse l'agent raisonner sur le texte."""
        import doc_intelligence as di
        if not di.available():
            raise RuntimeError("Azure Document Intelligence non configuré "
                               "(AZURE_DOC_INTEL_ENDPOINT / _KEY).")
        text = di.ocr_text(file_bytes)
        prompt = ("Voici le texte d'un rapport de laboratoire océrisé. Extrais les "
                  "valeurs NFS et évalue le patient.\n\n---\n" + text)
        return self.invoke(prompt, thread_id)


# --------------------------------------------------------------------------- #
# Bootstrap hors-ligne : rejouer la cohorte comme un flux opérationnel
# --------------------------------------------------------------------------- #
def bootstrap(days: int = 30, seed: int = C.RANDOM_SEED) -> int:
    """Génère un grand livre réaliste (sans LLM) en scorant tous les patients."""
    rng = np.random.default_rng(seed)
    clin = (pd.read_csv(C.PROCESSED_DIR / "clinical_clean.csv")
            .drop(columns=["sex"]).rename(columns={"sex_label": "sex"}))

    db.init_schema(reset=True)
    proc.sp_create_session(
        "S-BOOTSTRAP", channel="batch", operator="simulation")

    now = datetime.now()
    events: list[dict] = []
    now = datetime.now()
    events: list[dict] = []
    for i, row in clin.iterrows():
        patient = {k: row[k] for k in _REQUIRED}
        ts = now - timedelta(days=float(rng.uniform(0, days)),
                             hours=float(rng.uniform(0, 24)))
        d = score_patient(patient, session_id="S-BOOTSTRAP", ts=ts,
                          patient_ref=f"P-{i+1:05d}", persist=False)
        sla = d["urgency_sla_h"]
        handled = float(rng.gamma(shape=2.0, scale=sla / 3.0))
        d["handled_within_h"] = round(handled, 1)
        d["sla_met"] = int(handled <= sla)
        overrides = int(rng.random() < 0.08)
        d["override"] = overrides
        d["clinician_decision"] = "Rejeté" if overrides else (
            "Accepté" if rng.random() < 0.97 else "En attente")
        events.append(d)

    n = db.insert_events(events)
    pos = sum(e["prediction"] for e in events)
    print(f"[Bootstrap] {n} événements journalisés sur {days} j "
          f"| {pos} cas d'anémie détectés | base PostgreSQL '{C.DB['dbname']}'")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Agent ReAct de triage de l'anémie.")
    ap.add_argument("--bootstrap", action="store_true",
                    help="Rejoue toute la cohorte pour remplir la base (hors-ligne).")
    ap.add_argument("--days", type=int, default=30,
                    help="Fenêtre du bootstrap (jours).")
    ap.add_argument(
        "--message", help="Message pour l'agent (nécessite un LLM configuré).")
    args = ap.parse_args()

    if args.bootstrap:
        bootstrap(days=args.days)
        return

    if not llm.available():
        # Démo hors-ligne : extraction regex + narration déterministe.
        msg = args.message or "Femme 62 ans, Hb 8.1, RBC 3.4, HCT 27, MCV 79, MCH 24, MCHC 31"
        fields = extract_regex(msg)
        missing = [f for f in _REQUIRED if f not in fields]
        if missing:
            print("LLM non configuré et valeurs manquantes :", ", ".join(missing))
            return
        proc.sp_create_session("S-CLI", channel="chat", operator="cli")
        print(narrate(score_patient(fields, session_id="S-CLI")))
        return

    agent = AnemiaCareAgent()
    sid = agent.new_session(channel="chat", operator="cli")
    msg = args.message or "Femme 62 ans, Hb 8.1, RBC 3.4, HCT 27, MCV 79, MCH 24, MCHC 31"
    print(agent.invoke(msg, thread_id=sid))


if __name__ == "__main__":
    main()
