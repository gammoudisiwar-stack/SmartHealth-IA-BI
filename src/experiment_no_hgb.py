"""
Expérience « SANS hémoglobine » — évaluation honnête et défendable.

HYPOTHÈSE MÉTHODOLOGIQUE :
  L'hémoglobine (hgb) DÉFINIT l'étiquette OMS (anémique si Hb < seuil, par sexe).
  Elle est donc quasi-circulaire pour la tâche binaire : un modèle qui la voit
  « ré-apprend le seuil » (d'où une AUC ~0.99 peu informative). On l'EXCLUT et on
  mesure la capacité RÉELLE du modèle à inférer l'anémie à partir des seuls
  INDICES ÉRYTHROCYTAIRES (rbc, hct, mcv, mch, mchc, indice de Mentzer, âge, sexe).

Même pipeline de transfer learning que le projet (pré-entraînement pédiatrique
-> fine-tuning clinique), mais sur le schéma de features réduit.

Produit (variantes « _sans_hgb ») :
  exports/model_metrics_sans_hgb.csv, confusion_matrix_sans_hgb.csv,
  roc_curve_sans_hgb.csv, pr_curve_sans_hgb.csv, shap_importance_sans_hgb.csv
  reports/metrics_summary_sans_hgb.png, roc_pr_sans_hgb.png,
  confusion_matrix_sans_hgb.png, shap_importance_sans_hgb.png

Exécution :  python src/experiment_no_hgb.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import keras  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_recall_curve, roc_auc_score, roc_curve)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

TEAL, DARK, RED, GREY = "#12A19A", "#213A44", "#E8636B", "#9AA7AD"
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
plt.rcParams.update({"savefig.dpi": 160, "font.size": 11,
                    "axes.titleweight": "bold"})


# --------------------------------------------------------------------------- #
# Architecture (identique au projet, couches nommées pour le gel sélectif)
# --------------------------------------------------------------------------- #
def _compile(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), "accuracy"])
    return model


def build_mlp(input_dim, lr=1e-3):
    m = keras.Sequential([
        layers.Input((input_dim,), name="input"),
        layers.Dense(32, activation="relu", name="feat_1"),
        layers.Dropout(0.30, name="drop_1"),
        layers.Dense(16, activation="relu", name="feat_2"),
        layers.Dropout(0.20, name="drop_2"),
        layers.Dense(1, activation="sigmoid", name="head"),
    ], name="anemia_mlp_no_hgb")
    return _compile(m, lr)


def class_weights(y):
    y = np.asarray(y).astype(int)
    n = len(y)
    return {c: n / (2.0 * max((y == c).sum(), 1)) for c in (0, 1)}


def callbacks():
    return [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=25,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                              patience=10, min_lr=1e-5)]


# --------------------------------------------------------------------------- #
# Explicabilité
# --------------------------------------------------------------------------- #
def shap_importance(model, Xtr, Xte, feats):
    def f(x):
        return model.predict(x, verbose=0).ravel()
    try:
        import shap
        rng = np.random.default_rng(C.RANDOM_SEED)
        bg = Xtr[rng.choice(len(Xtr), min(100, len(Xtr)), replace=False)]
        sample = Xte[rng.choice(len(Xte), min(150, len(Xte)), replace=False)]
        sv = shap.Explainer(f, shap.maskers.Independent(bg))(sample)
        imp = np.abs(sv.values).mean(axis=0)
        method = "shap"
    except Exception as e:
        print(f"   [SHAP -> permutation] {e}")
        base = f(Xte)
        rng = np.random.default_rng(C.RANDOM_SEED)
        base_err = np.mean(base)  # référence neutre
        imp = []
        for j in range(Xte.shape[1]):
            Xp = Xte.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp.append(abs(np.mean(f(Xp)) - base_err))
        imp = np.array(imp)
        method = "permutation"
    return pd.DataFrame({"feature": feats, "importance_moyenne": imp, "methode": method}
                        ).sort_values("importance_moyenne", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figs(m, roc, pr, cm_mat, shap_df):
    # Métriques
    items = [("Sensibilité (rappel)", m["recall_sensibilite"]),
             ("Spécificité", m["specificite"]), ("Précision", m["precision"]),
             ("F1-score", m["f1"]), ("Exactitude", m["accuracy"]),
             ("ROC AUC", m["roc_auc"]), ("PR AUC", m["pr_auc"])][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh([i[0] for i in items], [i[1]
                   for i in items], color=TEAL, edgecolor=DARK)
    for b, (_, v) in zip(bars, items):
        ax.text(v - 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
                ha="right", color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Performance SANS hémoglobine — indices érythrocytaires (test n=%d)"
                 % int(m["n_test"]))
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "metrics_summary_sans_hgb.png")
    plt.close(fig)

    # ROC + PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    a1.plot(roc[0], roc[1], color=TEAL, lw=2.4,
            label=f"AUC = {m['roc_auc']:.3f}")
    a1.plot([0, 1], [0, 1], "--", color=GREY)
    a1.fill_between(roc[0], roc[1], alpha=0.08, color=TEAL)
    a1.set_xlabel("Taux de faux positifs")
    a1.set_ylabel("Sensibilité")
    a1.set_title("ROC (sans hgb)")
    a1.legend(loc="lower right")
    a2.plot(pr[0], pr[1], color=DARK, lw=2.4,
            label=f"PR AUC = {m['pr_auc']:.3f}")
    a2.axhline(m["prevalence"], ls="--", color=GREY,
               label=f"Hasard ({m['prevalence']:.2f})")
    a2.set_xlabel("Rappel")
    a2.set_ylabel("Précision")
    a2.set_title("Précision / Rappel (sans hgb)")
    a2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "roc_pr_sans_hgb.png")
    plt.close(fig)

    # Confusion
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm_mat, cmap="BuGn")
    lbl = ["Non anémique", "Anémique"]
    ax.set_xticks([0, 1], lbl)
    ax.set_yticks([0, 1], lbl)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité (OMS)")
    ax.set_title("Matrice de confusion — sans hgb")
    tot = cm_mat.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_mat[i, j]}\n({cm_mat[i, j]/tot:.0%})", ha="center",
                    va="center", fontsize=14, fontweight="bold",
                    color="white" if cm_mat[i, j] > cm_mat.max() / 2 else DARK)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "confusion_matrix_sans_hgb.png")
    plt.close(fig)

    # SHAP
    s = shap_df.sort_values("importance_moyenne")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(s["feature"], s["importance_moyenne"], color=TEAL, edgecolor=DARK)
    ax.set_xlabel("Importance moyenne |SHAP|")
    ax.set_title("Importance des variables SANS hémoglobine")
    ax.text(0.98, 0.05, "Signal réparti sur les indices érythrocytaires (VGM, RBC, HCT…)",
            transform=ax.transAxes, ha="right", color=DARK, fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "shap_importance_sans_hgb.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    keras.utils.set_random_seed(C.RANDOM_SEED)
    data = joblib.load(C.PROCESSED_DIR / "datasets.joblib")
    feats = list(data["feature_names"])
    j = feats.index("hgb")
    feats_reduced = [f for f in feats if f != "hgb"]
    print(f"Features RETENUES ({len(feats_reduced)}) : {feats_reduced}")
    print("Feature EXCLUE : hgb (définit l'étiquette OMS)\n")

    def drop(X):
        return np.delete(X, j, axis=1)

    Xpt, Xpv = drop(data["X_pre_train"]), drop(data["X_pre_val"])
    Xct, Xcv, Xcte = (drop(data["X_clin_train"]), drop(data["X_clin_val"]),
                      drop(data["X_clin_test"]))
    yct, ycte = data["y_clin_train"], data["y_clin_test"].astype(int)
    n = len(feats_reduced)

    # 1) Pré-entraînement pédiatrique (sans hgb)
    pre = build_mlp(n, 1e-3)
    pre.fit(Xpt, data["y_pre_train"], validation_data=(Xpv, data["y_pre_val"]),
            epochs=300, batch_size=32, class_weight=class_weights(data["y_pre_train"]),
            callbacks=callbacks(), verbose=0)

    # 2) Fine-tuning clinique (gel de feat_1)
    for layer in pre.layers:
        layer.trainable = layer.name != "feat_1"
    _compile(pre, 2e-4)
    pre.fit(Xct, yct, validation_data=(Xcv, data["y_clin_val"]),
            epochs=300, batch_size=32, class_weight=class_weights(yct),
            callbacks=callbacks(), verbose=0)

    # 3) Évaluation
    proba = pre.predict(Xcte, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ycte, pred).ravel()
    m = {
        "n_test": int(len(ycte)), "prevalence": float(ycte.mean()),
        "accuracy": float((pred == ycte).mean()),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall_sensibilite": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificite": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(ycte, pred)), "roc_auc": float(roc_auc_score(ycte, proba)),
        "pr_auc": float(average_precision_score(ycte, proba)),
        "brier_score": float(brier_score_loss(ycte, proba)),
        "log_loss": float(log_loss(ycte, np.clip(proba, 1e-6, 1 - 1e-6))),
    }
    pd.DataFrame({"metrique": list(m), "valeur": list(m.values())}).to_csv(
        C.EXPORTS_DIR / "model_metrics_sans_hgb.csv", index=False)

    cm_df = pd.DataFrame([["Non anémique", "Non anémique", int(tn)],
                          ["Non anémique", "Anémique", int(fp)],
                          ["Anémique", "Non anémique", int(fn)],
                          ["Anémique", "Anémique", int(tp)]],
                         columns=["reel", "predit", "n"])
    cm_df.to_csv(C.EXPORTS_DIR / "confusion_matrix_sans_hgb.csv", index=False)

    fpr, tpr, _ = roc_curve(ycte, proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        C.EXPORTS_DIR / "roc_curve_sans_hgb.csv", index=False)
    prec, rec, _ = precision_recall_curve(ycte, proba)
    pd.DataFrame({"recall": rec, "precision": prec}).to_csv(
        C.EXPORTS_DIR / "pr_curve_sans_hgb.csv", index=False)

    shap_df = shap_importance(pre, Xct, Xcte, feats_reduced)
    shap_df.to_csv(C.EXPORTS_DIR / "shap_importance_sans_hgb.csv", index=False)

    cm_mat = np.array([[tn, fp], [fn, tp]])
    _figs(m, (fpr, tpr), (rec, prec), cm_mat, shap_df)

    print("=== Résultats SANS hgb (test clinique) ===")
    for k, v in m.items():
        print(f"   {k:>20}: {v:.4f}" if isinstance(
            v, float) else f"   {k:>20}: {v}")
    print(f"\n   Top variables : {list(shap_df['feature'].head(4))}")
    print(f"\nCSV -> exports/*_sans_hgb.csv | Figures -> reports/*_sans_hgb.png")


if __name__ == "__main__":
    main()
