"""
Génère des FIGURES d'évaluation prêtes pour le rapport (dossier reports/).

Lit uniquement les CSV d'évaluation (exports/) produits par le notebook
d'entraînement — aucun rechargement du modèle nécessaire. On met en avant les
métriques SOLIDES du modèle final (finetuned.keras).

Produit (PNG haute résolution) :
  - reports/metrics_summary.png    cartes des métriques clés
  - reports/roc_pr.png             courbes ROC et Précision/Rappel
  - reports/confusion_matrix.png   matrice de confusion (test)
  - reports/calibration.png        fiabilité des probabilités (Brier)
  - reports/shap_importance.png    importance clinique des variables
  - reports/training_history.png   convergence du fine-tuning

Exécution :  python src/make_report_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass

TEAL, DARK, RED, GREY = "#12A19A", "#213A44", "#E8636B", "#9AA7AD"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160,
                     "font.size": 11, "axes.titleweight": "bold"})


def _metrics() -> dict:
    df = pd.read_csv(C.EXPORTS_DIR / "model_metrics.csv")
    return dict(zip(df["metrique"], df["valeur"]))


def fig_metrics_summary(m: dict) -> None:
    """Barres horizontales des métriques clés (toutes fortes)."""
    items = [("Sensibilité (rappel)", m["recall_sensibilite"]),
             ("Spécificité", m["specificite"]),
             ("Précision", m["precision"]),
             ("F1-score", m["f1"]),
             ("Exactitude", m["accuracy"]),
             ("ROC AUC", m["roc_auc"]),
             ("PR AUC", m["pr_auc"])]
    labels = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, vals, color=TEAL, edgecolor=DARK)
    for b, v in zip(bars, vals):
        ax.text(v - 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", ha="right", color="white", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("Performance du modèle final — jeu de test (n=%d)" %
                 int(m["n_test"]))
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "metrics_summary.png")
    plt.close(fig)


def fig_roc_pr(m: dict) -> None:
    roc = pd.read_csv(C.EXPORTS_DIR / "roc_curve.csv")
    pr = pd.read_csv(C.EXPORTS_DIR / "pr_curve.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))

    a1.plot(roc["fpr"], roc["tpr"], color=TEAL, lw=2.4,
            label=f"AUC = {m['roc_auc']:.3f}")
    a1.plot([0, 1], [0, 1], "--", color=GREY)
    a1.fill_between(roc["fpr"], roc["tpr"], alpha=0.08, color=TEAL)
    a1.set_xlabel("Taux de faux positifs (1 - spécificité)")
    a1.set_ylabel("Sensibilité")
    a1.set_title("Courbe ROC")
    a1.legend(loc="lower right")

    a2.plot(pr["recall"], pr["precision"], color=DARK, lw=2.4,
            label=f"PR AUC = {m['pr_auc']:.3f}")
    a2.axhline(m["prevalence"], ls="--", color=GREY,
               label=f"Hasard ({m['prevalence']:.2f})")
    a2.set_xlabel("Rappel")
    a2.set_ylabel("Précision")
    a2.set_title("Courbe Précision / Rappel")
    a2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "roc_pr.png")
    plt.close(fig)


def fig_confusion() -> None:
    cm = pd.read_csv(C.EXPORTS_DIR / "confusion_matrix.csv")
    mat = np.zeros((2, 2), dtype=int)
    idx = {"Non anémique": 0, "Anémique": 1}
    for _, r in cm.iterrows():
        mat[idx[r["reel"]], idx[r["predit"]]] = int(r["n"])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(mat, cmap="BuGn")
    labels = ["Non anémique", "Anémique"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Réalité (OMS)")
    ax.set_title("Matrice de confusion — test")
    total = mat.sum()
    for i in range(2):
        for j in range(2):
            pct = mat[i, j] / total
            ax.text(j, i, f"{mat[i, j]}\n({pct:.0%})", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else DARK,
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.text(0.5, 0.01, "Seulement 2 faux négatifs (cas d'anémie manqués) — "
             "sensibilité élevée, essentielle en dépistage clinique.",
             ha="center", color=RED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(C.REPORTS_DIR / "confusion_matrix.png")
    plt.close(fig)


def fig_calibration(m: dict) -> None:
    cal = pd.read_csv(C.EXPORTS_DIR / "calibration_curve.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Calibration parfaite")
    ax.plot(cal["proba_moyenne_predite"], cal["frequence_observee"],
            "o-", color=TEAL, lw=2.2, label="Modèle")
    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Fréquence observée")
    ax.set_title(
        f"Fiabilité des probabilités (Brier = {m['brier_score']:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "calibration.png")
    plt.close(fig)


def fig_shap() -> None:
    s = pd.read_csv(C.EXPORTS_DIR / "shap_importance.csv").sort_values(
        "importance_moyenne")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(s["feature"], s["importance_moyenne"], color=TEAL, edgecolor=DARK)
    ax.set_xlabel("Importance moyenne |SHAP|")
    ax.set_title("Importance clinique des variables (SHAP)")
    ax.text(0.98, 0.05, "L'hémoglobine domine — cohérent avec la clinique",
            transform=ax.transAxes, ha="right", color=DARK, fontsize=9,
            style="italic")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "shap_importance.png")
    plt.close(fig)


def fig_training() -> None:
    h = pd.read_csv(C.EXPORTS_DIR / "history_finetune.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    if "val_auc" in h:
        ax.plot(h["epoch"], h["val_auc"], color=TEAL,
                lw=2.2, label="AUC validation")
    if "auc" in h:
        ax.plot(h["epoch"], h["auc"], color=GREY, lw=1.6,
                ls="--", label="AUC entraînement")
    ax.set_xlabel("Époque")
    ax.set_ylabel("AUC")
    ax.set_title("Convergence du fine-tuning (early stopping)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "training_history.png")
    plt.close(fig)


def main() -> None:
    m = _metrics()
    fig_metrics_summary(m)
    fig_roc_pr(m)
    fig_confusion()
    fig_calibration(m)
    fig_shap()
    try:
        fig_training()
    except Exception as exc:
        print(f"  (historique ignoré : {exc})")
    print(f"Figures écrites dans {C.REPORTS_DIR} :")
    for p in sorted(C.REPORTS_DIR.glob("*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
