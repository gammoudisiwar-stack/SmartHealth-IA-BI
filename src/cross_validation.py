"""
Validation croisée (K-fold) — métrique HONNÊTE et défendable pour le rapport.

Plutôt qu'un unique split test (potentiellement « chanceux »), on évalue le
modèle par transfer learning en validation croisée stratifiée 5 plis sur toute
la cohorte clinique. On rapporte la MOYENNE ± ÉCART-TYPE — une lecture rigoureuse
qui montre la stabilité (et non un score isolé).

Produit :
  exports/cv_metrics.csv      (par pli + moyenne/écart-type)
  reports/cv_scores.png       (AUC & F1 par pli + moyenne±écart-type)

Exécution :  python src/cross_validation.py
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
from sklearn.metrics import (accuracy_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

TEAL, DARK, SAND, GREY = "#12A19A", "#213A44", "#E8A33D", "#7C8B92"
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
plt.rcParams.update({"savefig.dpi": 165, "font.size": 11,
                    "axes.titleweight": "bold"})

N_SPLITS = 5
# Variables EXCLUES : mesures directes de la masse érythrocytaire qui définissent
# (ou approchent très fortement) l'étiquette OMS d'anémie (Hb < seuil). Les retirer
# force le modèle à prédire l'anémie à partir de la seule MORPHOLOGIE des globules
# rouges (VGM, TCMH, CCMH, Mentzer) : tâche honnête et plus difficile.
# Mettre [] pour l'évaluation transfer learning complète.
DROP_FEATURES = ["hgb", "hct", "rbc"]


def _class_weights(y):
    y = np.asarray(y).astype(int)
    n = len(y)
    return {c: n / (2.0 * max((y == c).sum(), 1)) for c in (0, 1)}


def _build_scratch(input_dim):
    """MLP identique, entraîné de zéro (jeu de features réduit)."""
    from keras import layers
    net = keras.Sequential([
        layers.Input((input_dim,), name="input"),
        layers.Dense(32, activation="relu", name="feat_1"),
        layers.Dropout(0.30), layers.Dense(
            16, activation="relu", name="feat_2"),
        layers.Dropout(0.20), layers.Dense(
            1, activation="sigmoid", name="head"),
    ])
    return net


def _fit(Xtr, ytr, Xval, yval, use_transfer):
    """Transfer (features complètes) ou from-scratch (jeu réduit)."""
    if use_transfer:
        net = keras.models.load_model(C.MODELS_DIR / "pretrained.keras")
        for layer in net.layers:
            layer.trainable = layer.name != "feat_1"
        lr = 2e-4
    else:
        net = _build_scratch(Xtr.shape[1])
        lr = 1e-3
    net.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(name="auc")])
    net.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=300, batch_size=32,
            class_weight=_class_weights(ytr),
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                       patience=20, restore_best_weights=True)], verbose=0)
    return net


def main() -> None:
    keras.utils.set_random_seed(C.RANDOM_SEED)
    data = joblib.load(C.PROCESSED_DIR / "datasets.joblib")
    # Reconstitue toute la cohorte clinique (déjà standardisée).
    X = np.vstack(
        [data["X_clin_train"], data["X_clin_val"], data["X_clin_test"]])
    y = np.concatenate([data["y_clin_train"], data["y_clin_val"],
                        data["y_clin_test"]]).astype(int)

    # Sélection des variables (exclusion optionnelle des mesures de masse).
    feats = list(data["feature_names"])
    keep_idx = [i for i, f in enumerate(feats) if f not in DROP_FEATURES]
    kept = [feats[i] for i in keep_idx]
    use_transfer = len(kept) == len(feats)
    X = X[:, keep_idx]
    if DROP_FEATURES:
        print(f"Variables EXCLUES : {DROP_FEATURES}")
    print(f"Variables RETENUES ({len(kept)}) : {kept}")
    print(f"Validation croisée {N_SPLITS} plis — cohorte clinique n={len(y)} "
          f"(prévalence {y.mean():.1%})\n")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=C.RANDOM_SEED)
    rows = []
    for k, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        Xtr_all, ytr_all = X[tr_idx], y[tr_idx]
        Xtr, Xval, ytr, yval = train_test_split(
            Xtr_all, ytr_all, test_size=0.15, stratify=ytr_all,
            random_state=C.RANDOM_SEED)
        net = _fit(Xtr, ytr, Xval, yval, use_transfer)
        proba = net.predict(X[te_idx], verbose=0).ravel()
        pred = (proba >= 0.5).astype(int)
        yte = y[te_idx]
        rows.append({
            "pli": k,
            "auc": roc_auc_score(yte, proba),
            "f1": f1_score(yte, pred),
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
        })
        print(f"  Pli {k} : AUC={rows[-1]['auc']:.3f}  F1={rows[-1]['f1']:.3f}  "
              f"Acc={rows[-1]['accuracy']:.3f}")

    df = pd.DataFrame(rows)
    mean, std = df.mean(numeric_only=True), df.std(numeric_only=True)
    summary = pd.concat([df,
                         pd.DataFrame(
                             [{"pli": "moyenne", **mean[df.columns[1:]].to_dict()}]),
                         pd.DataFrame([{"pli": "ecart_type", **std[df.columns[1:]].to_dict()}])],
                        ignore_index=True)
    summary.to_csv(C.EXPORTS_DIR / "cv_metrics.csv", index=False)

    print(
        f"\n=== Résultat validé (moyenne ± écart-type sur {N_SPLITS} plis) ===")
    for c in ["auc", "f1", "accuracy", "precision", "recall"]:
        print(f"   {c:>10}: {mean[c]:.3f} ± {std[c]:.3f}")

    # --- Figure ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(1, N_SPLITS + 1)
    ax.bar(x - 0.19, df["auc"], width=0.38,
           color=TEAL, edgecolor=DARK, label="AUC")
    ax.bar(x + 0.19, df["f1"], width=0.38,
           color=SAND, edgecolor=DARK, label="F1")
    ax.axhline(mean["auc"], color=TEAL, ls="--", lw=1.4)
    ax.axhline(mean["f1"], color=SAND, ls="--", lw=1.4)
    ax.set_xticks(x, [f"Pli {i}" for i in x])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    task = ("cohorte clinique" if use_transfer
            else "anémie par morphologie (sans hgb/hct/rbc)")
    ax.set_title(f"Validation croisée {N_SPLITS} plis — {task}")
    ax.legend(loc="lower left", ncol=2)
    ax.text(0.98, 0.05,
            f"AUC {mean['auc']:.3f} ± {std['auc']:.3f}   |   "
            f"F1 {mean['f1']:.3f} ± {std['f1']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", color=DARK,
            fontsize=10, bbox=dict(fc="white", ec=GREY, boxstyle="round,pad=0.4"))
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "cv_scores.png")
    plt.close(fig)
    print(f"\nFigure -> {C.REPORTS_DIR / 'cv_scores.png'}")


if __name__ == "__main__":
    main()
