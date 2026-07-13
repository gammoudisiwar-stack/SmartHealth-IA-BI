"""
Utiliser le modèle entraîné sur de NOUVEAUX patients (inférence).

Usage :
  # 1) Patient unique (démo si aucun argument) :
  python src/predict.py --sex F --age 62 --hgb 10.2 --rbc 3.9 --hct 32 --mcv 82 --mch 26 --mchc 31.5

  # 2) Lot depuis un CSV (colonnes : sex,age,hgb,rbc,hct,mcv,mch,mchc) :
  python src/predict.py --csv chemin/vers/patients.csv
  # -> écrit exports/predictions_new.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

_MODEL = None
_SCALER = None


def _load():
    global _MODEL, _SCALER
    if _MODEL is None:
        _MODEL = keras.models.load_model(C.MODELS_DIR / "finetuned.keras")
        _SCALER = joblib.load(C.MODELS_DIR / "scaler.joblib")
    return _MODEL, _SCALER


def _sex_to_int(v) -> int:
    return 1 if str(v).strip().upper().startswith("M") else 0


def _severity(hgb: float, sex: str) -> str:
    thr = C.HB_ANEMIA_THRESHOLD.get(str(sex).strip().upper()[:1], 12.0)
    if hgb < C.HB_SEVERE:
        return "Sévère"
    if hgb < C.HB_MODERATE:
        return "Modérée"
    if hgb < thr:
        return "Légère"
    return "Aucune"


def _risk(p: float) -> str:
    return "Faible" if p < 0.34 else ("Modéré" if p < 0.67 else "Élevé")


def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Score un DataFrame de patients (colonnes : sex,age,hgb,rbc,hct,mcv,mch,mchc)."""
    model, scaler = _load()
    out = df.copy()
    sex_label = out["sex"].astype(str).str.strip().str.upper().str[0]
    feat = pd.DataFrame()
    feat["sex"] = [_sex_to_int(v) for v in out["sex"]]
    for col in ["age", "hgb", "rbc", "hct", "mcv", "mch", "mchc"]:
        feat[col] = pd.to_numeric(out[col], errors="coerce")
    feat["mentzer_index"] = feat["mcv"] / feat["rbc"].replace(0, np.nan)
    feat = feat.fillna(feat.median(numeric_only=True))

    hgb_orig = feat["hgb"].to_numpy().copy()  # Hb d'origine (avant standardisation)
    feat[C.CONTINUOUS_FEATURES] = scaler.transform(feat[C.CONTINUOUS_FEATURES])
    proba = model.predict(feat[C.MODEL_FEATURES].to_numpy("float32"), verbose=0).ravel()

    sex_list = list(sex_label)
    out["proba_anemie"] = np.round(proba, 4)
    out["prediction"] = (proba >= 0.5).astype(int)
    out["bande_risque"] = [_risk(p) for p in proba]
    out["severite_oms"] = [_severity(h, s) for h, s in zip(hgb_orig, sex_list)]
    out["regle_oms_anemique"] = [
        int(float(h) < C.HB_ANEMIA_THRESHOLD.get(s, 12.0)) for h, s in zip(hgb_orig, sex_list)
    ]
    return out


def _print_one(row: pd.Series) -> None:
    verdict = "ANÉMIQUE" if row["prediction"] == 1 else "non anémique"
    concord = "concordant" if row["prediction"] == row["regle_oms_anemique"] else "DIVERGENT (à revoir)"
    print("\n----------------------------------------------------")
    print(f" Patient : {row['sex']}, {row['age']} ans")
    print(f"   Hb={row['hgb']}  MCV={row['mcv']}  MCH={row['mch']}  MCHC={row['mchc']}  RBC={row['rbc']}")
    print(f"   -> Probabilité d'anémie : {row['proba_anemie']*100:.1f} %")
    print(f"   -> Prédiction           : {verdict} (risque {row['bande_risque']})")
    print(f"   -> Sévérité OMS (Hb)     : {row['severite_oms']}")
    print(f"   -> Contrôle règle OMS    : {concord}")
    print("----------------------------------------------------")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inférence anémie sur de nouveaux patients.")
    ap.add_argument("--csv", help="CSV de patients (sex,age,hgb,rbc,hct,mcv,mch,mchc)")
    for f in ["sex", "age", "hgb", "rbc", "hct", "mcv", "mch", "mchc"]:
        ap.add_argument(f"--{f}")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        res = predict_frame(df)
        dest = C.EXPORTS_DIR / "predictions_new.csv"
        res.to_csv(dest, index=False)
        print(f"{len(res)} patients scorés -> {dest}")
        return

    if args.sex and args.hgb:
        df = pd.DataFrame([{f: getattr(args, f) for f in
                            ["sex", "age", "hgb", "rbc", "hct", "mcv", "mch", "mchc"]}])
    else:  # exemple de démonstration
        print("(Aucun patient fourni -> exemple de démonstration)")
        df = pd.DataFrame([{"sex": "F", "age": 62, "hgb": 10.2, "rbc": 3.9,
                            "hct": 32, "mcv": 82, "mch": 26, "mchc": 31.5}])
    res = predict_frame(df)
    _print_one(res.iloc[0])


if __name__ == "__main__":
    main()
