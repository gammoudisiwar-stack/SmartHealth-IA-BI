"""
Interface Streamlit — deux modes :

  1. 💬 Agent de triage (ReAct + threads) : conversation, upload de rapport NFS
     (océrisation Document Intelligence), KPI métier en direct depuis PostgreSQL.
  2. 🔬 Inférence modèle : expose DIRECTEMENT le réseau entraîné
     (models/finetuned.keras, AUC test 0.995) sur un patient saisi à la main.

Lancement :  streamlit run chat_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
import care_agent as ca  # noqa: E402
import llm  # noqa: E402
import procedures as proc  # noqa: E402
from predict import predict_frame  # noqa: E402

st.set_page_config(page_title="Anémie — IA & Triage",
                   page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Initialisation de l'agent…")
def get_agent():
    return ca.AnemiaCareAgent()


def _offline_reply(message: str, thread_id: str) -> str:
    fields = ca.extract_regex(message)
    missing = [f for f in ca._REQUIRED if f not in fields]
    if missing:
        return ("(mode hors-ligne) Il me manque : " + ", ".join(missing)
                + ".\nExemple : « Femme 62 ans, Hb 8.1, RBC 3.4, HCT 27, MCV 79, MCH 24, MCHC 31 »")
    return ca.narrate(ca.score_patient(fields, session_id=thread_id))


# --------------------------------------------------------------------------- #
# État de session
# --------------------------------------------------------------------------- #
st.session_state.setdefault("threads", {})
st.session_state.setdefault("active", None)
llm_ok = llm.available()


def _new_session() -> str:
    if llm_ok:
        sid = get_agent().new_session(channel="chat", operator="streamlit")
    else:
        sid = f"S-{os.urandom(4).hex()}"
        proc.sp_create_session(sid, channel="chat",
                               operator="streamlit-offline")
    st.session_state.threads[sid] = []
    st.session_state.active = sid
    return sid


# --------------------------------------------------------------------------- #
# Page 1 — Agent conversationnel
# --------------------------------------------------------------------------- #
def page_agent() -> None:
    if not llm_ok:
        st.sidebar.warning(
            "LLM non configuré → mode hors-ligne (extraction simple).")
    if st.sidebar.button("➕ Nouvelle session", use_container_width=True):
        _new_session()

    st.sidebar.markdown("### Sessions")
    for s in proc.sp_list_sessions():
        sid = s["session_id"]
        if st.sidebar.button(f"{sid} · {s['n_events']} cas", key=f"sel_{sid}",
                             use_container_width=True):
            st.session_state.active = sid
            st.session_state.threads.setdefault(sid, [])

    st.sidebar.markdown("### 📄 Rapport de laboratoire")
    upload = st.sidebar.file_uploader("Téléverser (PDF / image)",
                                      type=["pdf", "png", "jpg", "jpeg"])

    if st.session_state.active is None:
        _new_session()
    active = st.session_state.active
    st.header(f"💬 Conversation — session `{active}`")

    kpis = proc.sp_get_kpis()
    cols = st.columns(4)
    cols[0].metric("Consultations", int(
        kpis.get("consultations_traitees", 0) or 0))
    cols[1].metric("Cas détectés", int(
        kpis.get("cas_anemie_detectes", 0) or 0))
    cols[2].metric("Coût évité (€)",
                   f"{kpis.get('cout_evite_total', 0) or 0:,.0f}")
    cols[3].metric("Heures économisées", kpis.get(
        "heures_clinicien_economisees", 0) or 0)

    if upload is not None and st.button("Analyser le rapport téléversé"):
        st.session_state.threads.setdefault(active, [])
        with st.spinner("Océrisation et analyse…"):
            try:
                reply = get_agent().ingest_document(upload.getvalue(), thread_id=active)
            except Exception as exc:
                reply = f"Impossible d'analyser le document : {exc}"
        st.session_state.threads[active].append(("user", f"📄 {upload.name}"))
        st.session_state.threads[active].append(("assistant", reply))

    for role, content in st.session_state.threads.get(active, []):
        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Décrivez le patient ou posez une question…"):
        st.session_state.threads.setdefault(
            active, []).append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("L'agent réfléchit…"):
                try:
                    reply = get_agent().invoke(prompt, thread_id=active) if llm_ok \
                        else _offline_reply(prompt, active)
                except Exception as exc:
                    reply = f"Erreur : {exc}"
            st.markdown(reply)
        st.session_state.threads[active].append(("assistant", reply))


# --------------------------------------------------------------------------- #
# Page 2 — Inférence directe du modèle entraîné
# --------------------------------------------------------------------------- #
def page_inference() -> None:
    st.header("🔬 Inférence directe — modèle entraîné")
    st.caption("Réseau profond `finetuned.keras` (transfer learning). "
               "Performance test : AUC 0.995 · sensibilité 0.977 · F1 0.903.")

    with st.form("inf"):
        c = st.columns(4)
        sex = c[0].selectbox("Sexe", ["F", "M"])
        age = c[1].number_input("Âge", 0, 120, 62)
        hgb = c[2].number_input("Hémoglobine (g/dL)", 2.0, 25.0, 8.1, 0.1)
        rbc = c[3].number_input("RBC (10⁶/µL)", 1.0, 8.0, 3.4, 0.01)
        c2 = st.columns(4)
        hct = c2[0].number_input("Hématocrite (%)", 10.0, 65.0, 27.0, 0.1)
        mcv = c2[1].number_input("MCV (fL)", 50.0, 130.0, 79.0, 0.1)
        mch = c2[2].number_input("MCH (pg)", 10.0, 45.0, 24.0, 0.1)
        mchc = c2[3].number_input("MCHC (g/dL)", 25.0, 40.0, 31.0, 0.1)
        submitted = st.form_submit_button("Prédire", type="primary")

    if not submitted:
        st.info(
            "Renseignez l'hémogramme puis cliquez **Prédire** pour interroger le modèle.")
        return

    patient = {"sex": sex, "age": age, "hgb": hgb, "rbc": rbc,
               "hct": hct, "mcv": mcv, "mch": mch, "mchc": mchc}
    res = predict_frame(pd.DataFrame([patient])).iloc[0]
    proba = float(res["proba_anemie"])
    pred = int(res["prediction"])
    mentzer = mcv / rbc if rbc else float("nan")
    anemia_type = ca.differential(pred, mentzer)
    oms = int(res["regle_oms_anemique"])

    m = st.columns(3)
    m[0].metric("Probabilité d'anémie", f"{proba*100:.1f} %")
    m[1].metric("Prédiction du modèle", "ANÉMIQUE" if pred else "Non anémique")
    m[2].metric("Sévérité OMS (Hb)", str(res["severite_oms"]))
    st.progress(min(max(proba, 0.0), 1.0))

    d = st.columns(3)
    d[0].metric("Bande de risque", str(res["bande_risque"]))
    d[1].metric("Indice de Mentzer", f"{mentzer:.1f}")
    d[2].metric("Différentiel", anemia_type if pred else "—")

    if pred == oms:
        st.success("✅ Prédiction du modèle concordante avec la règle OMS (Hb).")
    else:
        st.warning(
            "⚠️ Divergence modèle / règle OMS — à confirmer par un clinicien.")
    st.caption("Aide à la décision — la validation finale revient au clinicien.")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 Anémie — IA")
mode = st.sidebar.radio("Mode", ["💬 Agent de triage", "🔬 Inférence modèle"])
st.sidebar.markdown("---")

if mode.startswith("💬"):
    page_agent()
else:
    page_inference()
