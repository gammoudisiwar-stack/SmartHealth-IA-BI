"""
Diagrammes d'architecture rendus en IMAGES (matplotlib, sans dépendance externe).

Produit dans reports/images/ :
  - archi_global.png     stack technique de bout en bout
  - archi_agent.png      agent ReAct (LLM + outils déterministes + base)
  - archi_powerbi.png    PostgreSQL (vues) -> Power BI (PBIR)

Copie aussi toutes les figures d'évaluation (reports/*.png) dans reports/images/
pour regrouper toutes les images du rapport au même endroit.

Exécution :  python src/make_architecture_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")
sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

IMG_DIR = C.REPORTS_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

TEAL, DARK, SAND, GREY = "#12A19A", "#213A44", "#E8A33D", "#7C8B92"
plt.rcParams.update({"savefig.dpi": 170, "font.size": 10})


def _box(ax, cx, cy, w, h, text, fc=TEAL, tc="white", fs=9.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=DARK, lw=1.3, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", color=tc, fontsize=fs,
            zorder=3, fontweight="bold")


def _arrow(ax, p1, p2, label=None, color=GREY):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.5, zorder=1,
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, label, fontsize=7.5,
                color=DARK, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1), zorder=4)


def _canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", color=DARK)
    return fig, ax


def archi_global() -> None:
    fig, ax = _canvas(15, 5, "Architecture globale & stack technique")
    _box(ax, 1.3, 3.5, 2.1, 0.9, "Cohorte\npédiatrique", SAND, DARK)
    _box(ax, 1.3, 1.6, 2.1, 0.9, "Cohorte\nclinique (1004)", SAND, DARK)
    _box(ax, 4.0, 3.5, 2.2, 0.9, "1_data_\npreparation.ipynb")
    _box(ax, 4.0, 1.6, 2.2, 0.9, "2_training_\nevaluation.ipynb")
    _box(ax, 6.7, 2.55, 2.0, 0.9, "finetuned\n.keras", DARK)
    _box(ax, 9.2, 4.0, 2.1, 0.85, "Streamlit\nchat_app.py")
    _box(ax, 9.2, 2.55, 2.1, 0.85, "Agent ReAct")
    _box(ax, 9.2, 1.1, 2.1, 0.85, "predict.py")
    _box(ax, 11.9, 2.55, 2.0, 0.9, "PostgreSQL\nledger + vues", DARK)
    _box(ax, 14.0, 2.55, 1.7, 0.9, "Power BI")
    _arrow(ax, (2.35, 3.5), (2.9, 3.5))
    _arrow(ax, (2.35, 1.6), (2.9, 3.2))
    _arrow(ax, (4.0, 3.05), (4.0, 2.05))
    _arrow(ax, (5.1, 1.9), (5.7, 2.4))
    _arrow(ax, (7.7, 2.55), (8.15, 2.55))
    _arrow(ax, (9.2, 3.57), (9.2, 2.98))
    _arrow(ax, (9.2, 1.53), (9.2, 2.12))
    _arrow(ax, (10.25, 2.55), (10.9, 2.55), "log")
    _arrow(ax, (12.9, 2.55), (13.15, 2.55), "vw_*")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "archi_global.png")
    plt.close(fig)


def archi_agent() -> None:
    fig, ax = _canvas(11, 8, "Architecture de l'agent (ReAct)")
    _box(ax, 5.5, 7.3, 3.0, 0.8, "Utilisateur / soignant", DARK)
    _box(ax, 5.5, 6.1, 3.0, 0.8, "Streamlit — chat_app.py")
    _box(ax, 2.0, 4.9, 2.6, 0.9, "Document\nIntelligence (OCR)", SAND, DARK)
    _box(ax, 5.5, 4.9, 2.6, 0.9, "AnemiaCareAgent\n(create_agent)")
    _box(ax, 9.0, 4.9, 2.4, 0.9, "Azure OpenAI\ngpt-5.4", DARK)
    _box(ax, 9.0, 3.4, 2.4, 0.8, "PostgresSaver\n(threads)", GREY, "white", 8.5)
    _box(ax, 3.0, 3.2, 2.4, 0.85, "classify_anemia\n(finetuned.keras)")
    _box(ax, 6.0, 3.2, 2.2, 0.85, "get_operational\n_kpis")
    _box(ax, 3.0, 1.6, 2.6, 0.85, "score_patient\n+ Mentzer + parcours")
    _box(ax, 6.0, 0.7, 2.4, 0.85, "PostgreSQL — ledger", DARK)
    _arrow(ax, (5.5, 6.9), (5.5, 6.5))
    _arrow(ax, (4.3, 5.9), (2.6, 5.35), "PDF/img")
    _arrow(ax, (2.4, 4.45), (4.5, 4.95))
    _arrow(ax, (5.5, 5.7), (5.5, 5.35))
    _arrow(ax, (6.8, 4.9), (7.8, 4.9), "raisonne")
    _arrow(ax, (7.8, 4.6), (6.8, 4.75))
    _arrow(ax, (6.8, 4.5), (8.4, 3.8), "thread")
    _arrow(ax, (4.6, 4.45), (3.4, 3.65))
    _arrow(ax, (5.7, 4.45), (6.1, 3.65))
    _arrow(ax, (3.0, 2.75), (3.0, 2.05))
    _arrow(ax, (3.6, 1.35), (5.2, 0.95), "sp_log_decision")
    _arrow(ax, (6.0, 2.75), (6.0, 1.15), "vw_ops_kpis")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "archi_agent.png")
    plt.close(fig)


def archi_powerbi() -> None:
    fig, ax = _canvas(14, 6.4, "Architecture BI — PostgreSQL vers Power BI")
    _box(ax, 1.9, 3.2, 2.3, 1.0, "table ledger\n1 ligne / décision", DARK)
    views = [("vw_ops_kpis", 5.4), ("vw_referral_funnel", 4.45),
             ("vw_case_mix", 3.5), ("vw_agent_daily", 2.55),
             ("vw_equity_gap", 1.6), ("vw_urgency / sla", 0.65)]
    for name, y in views:
        _box(ax, 5.6, y, 2.6, 0.7, name, TEAL, "white", 8.5)
        _arrow(ax, (3.05, 3.2), (4.3, y))
    _box(ax, 9.6, 3.1, 2.6, 1.0, "Modèle sémantique\nTMDL + 7 mesures DAX", DARK)
    _box(ax, 12.6, 3.1, 2.3, 1.0, "Page Cockpit\n9 visuels")
    for _, y in views:
        _arrow(ax, (6.9, y), (8.3, 3.1))
    _arrow(ax, (10.9, 3.1), (11.45, 3.1), "Npgsql")
    ax.text(5.6, 6.0, "Vues SQL — KPI métier", ha="center", color=DARK,
            fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "archi_powerbi.png")
    plt.close(fig)


def main() -> None:
    archi_global()
    archi_agent()
    archi_powerbi()
    # Jeu d'images FINAL du rapport (diagrammes + évaluation validée).
    keep = {"archi_global.png", "archi_agent.png", "archi_powerbi.png"}
    final_eval = ["cv_scores", "roc_pr", "confusion_matrix", "calibration",
                  "shap_importance"]
    for name in final_eval:
        src = C.REPORTS_DIR / f"{name}.png"
        if src.exists():
            shutil.copy2(src, IMG_DIR / f"{name}.png")
            keep.add(f"{name}.png")
    # Ne conserve QUE les images finales du rapport.
    for p in IMG_DIR.glob("*.png"):
        if p.name not in keep:
            p.unlink()
    print(f"Images finales du rapport dans {IMG_DIR} :")
    for p in sorted(IMG_DIR.glob("*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
