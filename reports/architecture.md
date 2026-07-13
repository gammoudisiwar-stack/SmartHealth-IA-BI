# Architecture — SmartHealth IA & BI (diagrammes)

> Diagrammes Mermaid. Pour les exporter en image (PNG/SVG) pour le rapport :
>
> - **VS Code** : ouvrir l'aperçu Markdown (Ctrl+Shift+V) → clic droit sur le diagramme → _Copy Image_, ou
> - **mermaid.live** : coller le bloc `mermaid` → _Actions → PNG/SVG_.

## 1. Architecture globale & stack technique

```mermaid
flowchart LR
  subgraph Data["Donnees (Mendeley)"]
    D1["Cohorte pediatrique<br/>~1000 patients"]
    D2["Cohorte clinique<br/>1004 patients"]
  end
  subgraph Train["Entrainement — Notebooks Jupyter"]
    N1["1_data_preparation.ipynb<br/>nettoyage + features + splits"]
    N2["2_training_evaluation.ipynb<br/>transfer learning + eval + SHAP"]
  end
  M["finetuned.keras<br/>+ scaler.joblib"]
  subgraph App["Application — Python"]
    UI["Streamlit — chat_app.py"]
    AG["Agent de triage (ReAct)"]
    PR["predict.py — inference"]
  end
  DB[("PostgreSQL<br/>ledger + vues KPI")]
  BI["Power BI<br/>AnemieCockpit.pbip"]

  D1 --> N1
  D2 --> N1
  N1 --> N2 --> M
  M --> PR --> AG
  UI --> AG
  AG -->|sp_log_decision| DB
  DB -->|vues vw_*| BI
```

**Stack :** Python 3.13 · TensorFlow/Keras (MLP transfer learning) · scikit-learn ·
LangChain + LangGraph (agent ReAct + memoire) · Azure OpenAI (gpt-5.4) ·
Azure AI Document Intelligence (OCR) · PostgreSQL 18 (psycopg) · Streamlit (UI) ·
Power BI (PBIR / connecteur Npgsql).

## 2. Architecture de l'agent (ReAct)

```mermaid
flowchart TB
  U["Utilisateur / soignant"] -->|"message ou rapport NFS"| UI["Streamlit — chat_app.py"]
  UI -->|"fichier PDF/image"| DOC["Document Intelligence<br/>OCR prebuilt-layout"]
  DOC --> AG
  UI --> AG["AnemiaCareAgent<br/>LangChain create_agent"]
  AG <-->|"raisonnement"| LLM["Azure OpenAI — gpt-5.4"]
  AG -->|"thread persiste"| CP[("PostgresSaver<br/>checkpointer")]

  subgraph Tools["Outils deterministes (source de verite)"]
    T1["classify_anemia<br/>modele finetuned.keras"]
    T2["get_operational_kpis"]
  end
  AG --> T1
  AG --> T2
  T1 --> SP["score_patient<br/>+ indice de Mentzer + parcours de soins"]
  SP -->|"CALL sp_log_decision"| DB[("PostgreSQL — ledger")]
  T2 -->|"SELECT vw_ops_kpis"| DB
```

**Principe :** le LLM RAISONNE et choisit les outils ; il n'invente aucun diagnostic.
Les outils cliniques sont deterministes (modele + regles) et journalisent chaque
decision dans PostgreSQL via une procedure stockee.

## 3. Architecture BI (Power BI ↔ PostgreSQL)

```mermaid
flowchart LR
  subgraph PG["PostgreSQL — base anemia"]
    L[("table ledger<br/>1 ligne / decision agent")]
    subgraph Views["Vues SQL — KPI metier"]
      V1["vw_ops_kpis"]
      V2["vw_referral_funnel"]
      V3["vw_case_mix"]
      V4["vw_agent_daily"]
      V5["vw_equity_gap"]
      V6["vw_urgency_mix / vw_sla_by_urgency"]
    end
  end
  L --> V1
  L --> V2
  L --> V3
  L --> V4
  L --> V5
  L --> V6

  subgraph PBIP["AnemieCockpit.pbip (format PBIR)"]
    SM["Modele semantique TMDL<br/>+ 7 mesures DAX"]
    RP["Page Cockpit<br/>9 visuels"]
  end
  Views -->|"connecteur Npgsql (import)"| SM --> RP
  RP --> DASH["Cockpit operationnel<br/>cartes | funnel | donut | courbe | table equite"]
```

**Flux :** l'agent ecrit dans `ledger` → les vues `vw_*` recalculent les KPI en
direct → Power BI les importe via le connecteur PostgreSQL natif (aucun CSV
intermediaire) → rafraichir dans Power BI met a jour le tableau de bord.
