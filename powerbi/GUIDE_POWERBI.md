# Cockpit Power BI — branché sur PostgreSQL

Le tableau de bord lit **directement les vues SQL** de la base `anemia` (aucun CSV).
Les KPI métier proviennent de l'activité de l'**agent de triage** (table `ledger`),
exposée via des **vues** `vw_*`.

---

## 1. Pré-requis

- **PostgreSQL** en service (`postgresql-18`) avec la base `anemia` peuplée :
  ```powershell
  .\.venv\Scripts\python.exe src\care_agent.py --bootstrap   # peuple ledger + vues
  .\.venv\Scripts\python.exe src\build_pbip.py               # (re)génère le PBIP
  ```
- **Power BI Desktop** avec le connecteur **PostgreSQL (Npgsql)**.
  Si absent : Power BI propose de l'installer au premier import PostgreSQL.

## 2. Ouvrir le projet

Power BI Desktop → **Fichier → Ouvrir** → `powerbi/AnemieCockpit.pbip`.

Le modèle est déjà câblé : serveur `localhost:5432`, base `anemia`, schéma `public`.
À la première ouverture, saisir les identifiants PostgreSQL (`postgres` / `postgres`).

## 3. Vues disponibles (déjà importées)

| Vue                      | Rôle                                                      | Visuel suggéré     |
| ------------------------ | --------------------------------------------------------- | ------------------ |
| `vw_ops_kpis`            | Cartes métier (consultations, coût évité, SLA, override…) | Cartes             |
| `vw_referral_funnel`     | Entonnoir dépistage → référés → urgents                   | Entonnoir / barres |
| `vw_case_mix`            | Différentiel (carence fer vs thalassémie)                 | Anneau             |
| `vw_action_distribution` | Répartition des actions recommandées                      | Barres             |
| `vw_urgency_mix`         | Répartition par urgence                                   | Anneau             |
| `vw_sla_by_urgency`      | Respect du SLA par urgence                                | Barres + ligne     |
| `vw_equity_gap`          | Écart de détection IA vs règle OMS par cohorte            | Matrice            |
| `vw_agent_daily`         | Série temporelle quotidienne                              | Courbe             |
| `vw_patient_ledger`      | Détail patient + **mesures DAX**                          | Table              |

## 4. Mesures DAX (sur `vw_patient_ledger`)

Déjà définies : `Consultations`, `CasDetectes`, `TauxDetection`, `CoutEvite`,
`TauxConformiteSLA`, `TauxOverride`, `DelaiPriseEnChargeH`.
Les déposer dans des **cartes** pour un cockpit dynamique et filtrable.

## 5. Actualiser

L'agent journalise en continu dans PostgreSQL. Dans Power BI : **Accueil → Actualiser**
pour recharger les vues (aucune régénération de fichier nécessaire).
