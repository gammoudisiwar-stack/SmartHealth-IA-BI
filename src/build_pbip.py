"""
Génère le projet Power BI **PBIP** branché NATIVEMENT sur PostgreSQL.

Plus aucun CSV intermédiaire : chaque table du modèle sémantique pointe vers une
VUE SQL (vw_*) de la base `anemia` via le connecteur PostgreSQL (Power Query
`PostgreSQL.Database`). Les KPI métier sont donc lus en direct depuis la base,
alimentée par l'agent de triage.

Pré-requis Power BI Desktop : le fournisseur **Npgsql** (connecteur PostgreSQL).
Ouverture :  Power BI Desktop -> Fichier -> Ouvrir -> powerbi/AnemieCockpit.pbip

Exécution :  python src/build_pbip.py
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

PBIP_DIR = C.BASE_DIR / "powerbi"
MODEL_DIR = PBIP_DIR / "AnemieCockpit.SemanticModel"
REPORT_DIR = PBIP_DIR / "AnemieCockpit.Report"

PG_SERVER = f"{C.DB['host']}:{C.DB['port']}"
PG_DATABASE = C.DB["dbname"]

# Schéma (colonne -> type) de chaque VUE. Types : s=texte, i=entier, d=décimal, t=date/heure.
TABLES: dict[str, list[tuple[str, str]]] = {
    "vw_ops_kpis": [("metrique", "s"), ("valeur", "d"), ("unite", "s"), ("note", "s")],
    "vw_referral_funnel": [("etape", "s"), ("n_patients", "i"), ("ordre", "i")],
    "vw_action_distribution": [("recommended_action", "s"), ("n_patients", "i"), ("part", "d")],
    "vw_case_mix": [("anemia_type", "s"), ("n_patients", "i"), ("part", "d")],
    "vw_urgency_mix": [("urgency", "s"), ("n_patients", "i"), ("part", "d")],
    "vw_sla_by_urgency": [("urgency", "s"), ("n_patients", "i"), ("taux_sla", "d"),
                          ("delai_moyen_h", "d")],
    "vw_equity_gap": [("bande_age", "s"), ("sexe", "s"), ("n_patients", "i"),
                      ("taux_detection_ia", "d"), ("taux_regle_oms", "d"), ("ecart_ia_oms", "d")],
    "vw_agent_daily": [("jour", "t"), ("consultations", "i"), ("cas_detectes", "i"),
                       ("cout_evite", "d"), ("delai_moyen_h", "d"), ("taux_sla", "d")],
    "vw_patient_ledger": [("event_id", "i"), ("session_id", "s"), ("ts", "t"),
                          ("patient_ref", "s"), ("sex",
                                                 "s"), ("age", "d"), ("age_band", "s"),
                          ("hgb", "d"), ("mentzer_index",
                                         "d"), ("proba_anemie", "d"),
                          ("prediction", "i"), ("risk_band", "s"), ("severity", "s"),
                          ("anemia_type", "s"), ("recommended_action",
                                                 "s"), ("urgency", "s"),
                          ("urgency_sla_h", "i"), ("handled_within_h",
                                                   "d"), ("sla_met", "i"),
                          ("cost_avoided", "d"), ("clinician_decision", "s"),
                          ("override_flag", "i")],
}

# Mesures DAX (rattachées à vw_patient_ledger) — KPI métier depuis l'activité agent.
MEASURES = [
    ("Consultations", "COUNTROWS(vw_patient_ledger)", "0"),
    ("CasDetectes", "SUM(vw_patient_ledger[prediction])", "0"),
    ("TauxDetection", "DIVIDE([CasDetectes], [Consultations])", "0.0%"),
    ("CoutEvite", "SUM(vw_patient_ledger[cost_avoided])", "0"),
    ("TauxConformiteSLA", "AVERAGE(vw_patient_ledger[sla_met])", "0.0%"),
    ("TauxOverride", "AVERAGE(vw_patient_ledger[override_flag])", "0.0%"),
    ("DelaiPriseEnChargeH",
     "AVERAGE(vw_patient_ledger[handled_within_h])", "0.0"),
]

_TMDL_TYPE = {"s": "string", "i": "int64", "d": "double", "t": "dateTime"}


def _guid() -> str:
    return str(uuid.uuid4())


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _table_tmdl(name: str, cols: list[tuple[str, str]]) -> str:
    lines = [f"table {name}", f"\tlineageTag: {_guid()}", ""]
    for col, t in cols:
        lines += [
            f"\tcolumn {col}",
            f"\t\tdataType: {_TMDL_TYPE[t]}",
            f"\t\tlineageTag: {_guid()}",
            "\t\tsummarizeBy: none",
            f"\t\tsourceColumn: {col}",
            "",
        ]
    if name == "vw_patient_ledger":
        for mname, expr, fmt in MEASURES:
            lines += [f"\tmeasure {mname} = {expr}", f"\t\tformatString: {fmt}",
                      f"\t\tlineageTag: {_guid()}", ""]
    # Partition Power Query (M) — source PostgreSQL native (vue).
    m = (
        "\t\t\tlet\n"
        f'\t\t\t    Source = PostgreSQL.Database("{PG_SERVER}", "{PG_DATABASE}"),\n'
        f'\t\t\t    data = Source{{[Schema="public", Item="{name}"]}}[Data]\n'
        "\t\t\tin\n"
        "\t\t\t    data"
    )
    lines += [f"\tpartition {name} = m",
              "\t\tmode: import", "\t\tsource =", m, ""]
    return "\n".join(lines) + "\n"


def build_model() -> None:
    _write(MODEL_DIR / ".platform", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "SemanticModel", "displayName": "AnemieCockpit"},
        "config": {"version": "2.0", "logicalId": _guid()},
    }, indent=2))
    _write(MODEL_DIR / "definition.pbism", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json",
        "version": "4.2", "settings": {}}, indent=2))
    _write(MODEL_DIR / "definition" / "database.tmdl",
           "database\n\tcompatibilityLevel: 1601\n")
    _write(MODEL_DIR / "definition" / "model.tmdl",
           "model Model\n\tculture: fr-FR\n\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
           "\tsourceQueryCulture: fr-FR\n")
    # Supprime les anciennes tables (modèle CSV) avant de régénérer les vues.
    tables_dir = MODEL_DIR / "definition" / "tables"
    if tables_dir.exists():
        for old in tables_dir.glob("*.tmdl"):
            old.unlink()
    # Supprime toute relation auto-détectée par Power BI (aucune n'est requise ici,
    # et une relation résiduelle qui pointe vers une colonne absente casse le modèle).
    rel = MODEL_DIR / "definition" / "relationships.tmdl"
    if rel.exists():
        rel.unlink()
    for name, cols in TABLES.items():
        _write(tables_dir / f"{name}.tmdl", _table_tmdl(name, cols))


def build_report() -> None:
    # Repart d'un dossier de pages propre (supprime d'éventuelles pages résiduelles).
    pages_dir = REPORT_DIR / "definition" / "pages"
    if pages_dir.exists():
        shutil.rmtree(pages_dir)
    _write(REPORT_DIR / ".platform", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "Report", "displayName": "AnemieCockpit"},
        "config": {"version": "2.0", "logicalId": _guid()},
    }, indent=2))
    _write(REPORT_DIR / "definition.pbir", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0",
        "datasetReference": {"byPath": {"path": "../AnemieCockpit.SemanticModel"}},
    }, indent=2))
    _write(REPORT_DIR / "definition" / "version.json", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json",
        "version": "2.0.0"}, indent=2))
    _write(REPORT_DIR / "definition" / "report.json", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/3.3.0/schema.json",
        "themeCollection": {}}, indent=2))
    _write(REPORT_DIR / "definition" / "pages" / "pages.json", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.1.0/schema.json",
        "pageOrder": ["cockpit"], "activePageName": "cockpit"}, indent=2))
    _write(REPORT_DIR / "definition" / "pages" / "cockpit" / "page.json", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.1.0/schema.json",
        "name": "cockpit", "displayName": "Cockpit opérationnel", "displayOption": "FitToPage",
        "height": 720, "width": 1280}, indent=2))


# --------------------------------------------------------------------------- #
# Visuels du rapport (dashboard prêt à l'emploi)
# --------------------------------------------------------------------------- #
def _measure(entity: str, prop: str) -> dict:
    return {"field": {"Measure": {"Expression": {"SourceRef": {"Entity": entity}},
                                  "Property": prop}},
            "queryRef": f"{entity}.{prop}", "nativeQueryRef": prop}


def _column(entity: str, prop: str) -> dict:
    return {"field": {"Column": {"Expression": {"SourceRef": {"Entity": entity}},
                                 "Property": prop}},
            "queryRef": f"{entity}.{prop}", "nativeQueryRef": prop}


def _sum(entity: str, prop: str) -> dict:
    return {"field": {"Aggregation": {"Expression": {"Column": {
        "Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}, "Function": 0}},
        "queryRef": f"Sum({entity}.{prop})", "nativeQueryRef": prop}


def _visual(vtype: str, x: int, y: int, w: int, h: int, tab: int, roles: dict) -> dict:
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.10.0/schema.json",
        "name": uuid.uuid4().hex[:20],
        "position": {"x": x, "y": y, "z": tab, "width": w, "height": h, "tabOrder": tab},
        "visual": {
            "visualType": vtype,
            "query": {"queryState": {r: {"projections": p} for r, p in roles.items()}},
        },
    }


def build_visuals() -> int:
    """Génère un cockpit prêt à l'emploi : cartes, graphiques et table."""
    vdir = REPORT_DIR / "definition" / "pages" / "cockpit" / "visuals"
    if vdir.exists():
        shutil.rmtree(vdir)
    L = "vw_patient_ledger"
    visuals = [
        # Bandeau de cartes (mesures DAX).
        _visual("card", 16, 24, 300, 110, 0, {
                "Values": [_measure(L, "Consultations")]}),
        _visual("card", 328, 24, 300, 110, 1, {
                "Values": [_measure(L, "CasDetectes")]}),
        _visual("card", 640, 24, 300, 110, 2, {
                "Values": [_measure(L, "CoutEvite")]}),
        _visual("card", 952, 24, 312, 110, 3, {
                "Values": [_measure(L, "TauxConformiteSLA")]}),
        # Ligne du milieu : entonnoir, différentiel, urgence.
        _visual("clusteredColumnChart", 16, 150, 400, 270, 4,
                {"Category": [_column("vw_referral_funnel", "etape")],
                 "Y": [_sum("vw_referral_funnel", "n_patients")]}),
        _visual("donutChart", 432, 150, 380, 270, 5,
                {"Category": [_column("vw_case_mix", "anemia_type")],
                 "Y": [_sum("vw_case_mix", "n_patients")]}),
        _visual("clusteredColumnChart", 828, 150, 436, 270, 6,
                {"Category": [_column("vw_urgency_mix", "urgency")],
                 "Y": [_sum("vw_urgency_mix", "n_patients")]}),
        # Ligne du bas : activité quotidienne + équité.
        _visual("lineChart", 16, 436, 620, 268, 7,
                {"Category": [_column("vw_agent_daily", "jour")],
                 "Y": [_sum("vw_agent_daily", "consultations")]}),
        _visual("tableEx", 652, 436, 612, 268, 8,
                {"Values": [_column("vw_equity_gap", "bande_age"),
                            _column("vw_equity_gap", "sexe"),
                            _sum("vw_equity_gap", "taux_detection_ia"),
                            _sum("vw_equity_gap", "taux_regle_oms"),
                            _sum("vw_equity_gap", "ecart_ia_oms")]}),
    ]
    for v in visuals:
        _write(vdir / v["name"] / "visual.json", json.dumps(v, indent=2))
    return len(visuals)


def build_pbip() -> None:
    _write(PBIP_DIR / "AnemieCockpit.pbip", json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
        "version": "1.0",
        "artifacts": [{"report": {"path": "AnemieCockpit.Report"}}],
        "settings": {"enableAutoRecovery": True}}, indent=2))


def main() -> None:
    build_model()
    build_report()
    n_vis = build_visuals()
    build_pbip()
    print(
        f"Projet PBIP (PostgreSQL) généré -> {PBIP_DIR / 'AnemieCockpit.pbip'}")
    print(f"  {len(TABLES)} vues + {len(MEASURES)} mesures DAX + {n_vis} visuels "
          f"branchés sur PostgreSQL {PG_SERVER}/{PG_DATABASE}")
    print("  Pré-requis : connecteur Npgsql installé dans Power BI Desktop.")


if __name__ == "__main__":
    main()
