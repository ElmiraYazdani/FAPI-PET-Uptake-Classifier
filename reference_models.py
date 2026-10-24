"""
Conventional quantitative reference models evaluated in the same framework (as in the previous analysis).

Same 1,020 foci and the same patient-grouped outer folds (StratifiedGroupKFold, 4 folds, seed 100) as the radiomics
experiments. Within each outer-training partition: median imputation (only if a value is missing) -> ComBat
(batch = center, no covariates) -> z-scoring -> logistic regression (L2, C = 1, balanced class weights). The four
variables are log-transformed before modelling because of their strong right skew. No feature selection or tuning.
PET intensities are SUV and do not depend on the grey-level discretization, so the PET-64 file is used.

Usage:
    python reference_models.py --data ../01_data --out ..
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from fapi_pipeline.harmonization import ComBat

COLS = {
    "SUVmax": "PET_IB_MaximumIntensity(IBSI:84IY)",
    "SUVmean": "PET_IB_MeanIntensity(IBSI:Q4LE)",
    "Volume": "PET_Mo_Volume(IBSI:RNU0)[cm3]",
    "TLF": "PET_IB_TotalLesionGlycolysis(IBSI:No)",
}
MODELS = {
    "Ref: SUVmax": ["SUVmax"],
    "Ref: SUVmean": ["SUVmean"],
    "Ref: Volume": ["Volume"],
    "Ref: TLF": ["TLF"],
    "Ref: SUVmax+SUVmean+Volume+TLF": ["SUVmax", "SUVmean", "Volume", "TLF"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../01_data")
    ap.add_argument("--out", default="..")
    ap.add_argument("--file", default="exp1_pet64.csv")
    ap.add_argument("--tag", default="Reference_conventional")
    a = ap.parse_args()

    df = pd.read_csv(Path(a.data) / a.file)
    y = df.label.to_numpy()
    g = df.patient_id.to_numpy()
    b = df.center.to_numpy()
    feats = pd.DataFrame({k: np.log(df[v].clip(lower=1e-6)) for k, v in COLS.items()})
    print("missing values in reference variables:", feats.isna().sum().to_dict())

    rows = []
    outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100)
    for k, (tr, te) in enumerate(outer.split(feats, y, g), start=1):
        for name, cols in MODELS.items():
            imp = SimpleImputer(strategy="median").fit(feats[cols].to_numpy()[tr])
            Xtr, Xte = imp.transform(feats[cols].to_numpy()[tr]), imp.transform(feats[cols].to_numpy()[te])
            cb = ComBat().fit(Xtr, b[tr])
            Xtr, Xte = cb.transform(Xtr, b[tr]), cb.transform(Xte, b[te])
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), y[tr])
            p = m.predict_proba(sc.transform(Xte))[:, 1]
            for i, idx in enumerate(te):
                rows.append({"setting": "reference", "model": name, "outer_fold": k, "patient_id": int(g[idx]),
                             "region_id": df.region_id.iat[idx], "center": int(b[idx]), "label": int(y[idx]),
                             "prob": float(p[i]), "pred": int(p[i] >= 0.5)})
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "oof_predictions.csv", index=False)
    print(f"saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
