"""
Post hoc analysis by type of physiological focus (no re-training).

Re-evaluates the out-of-fold predictions of the highest-ROC-AUC classifier of each experiment and of the conventional
reference models separately for (a) the ten organ regions per patient (foci 1-10) versus all pathological foci and
(b) the physiological focal-uptake foci (non-organ Class 0 foci) versus all pathological foci. 95% CIs from the
patient-cluster bootstrap (2,000 resamples of the patients present in the subset; seed 100).

Usage:
    python negative_subtype.py --out .. --data ../01_data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize import B, SEED, fast_auc


def boot_ci(g, rng):
    pts = np.unique(g.patient_id)
    by = {p: np.where(g.patient_id.to_numpy() == p)[0] for p in pts}
    y, p = g.label.to_numpy(), g.prob.to_numpy()
    vals = []
    for _ in range(B):
        idx = np.concatenate([by[q] for q in rng.choice(pts, len(pts), replace=True)])
        if y[idx].min() != y[idx].max():
            vals.append(fast_auc(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    ap.add_argument("--data", default="../01_data")
    a = ap.parse_args()
    oof = pd.read_csv(Path(a.out) / "summary" / "all_oof_predictions.csv")
    perf = pd.read_csv(Path(a.out) / "summary" / "performance_summary.csv")
    d = pd.read_csv(Path(a.data) / "exp1_pet64.csv")[["region_id", "focus", "label"]]
    kind = dict(zip(d.region_id, np.where(d.label == 1, "pathological",
                                          np.where(d.focus <= 10, "organ", "focal physiological"))))
    models = [(s, perf[perf.setting == s].sort_values("AUC", ascending=False).model.iloc[0])
              for s in ["exp1_pet64", "exp2_pet128", "exp3_pet64_ct", "exp4_pet128_ct", "exp5_ct"]]
    models += [("reference", m) for m in ["Ref: SUVmax+SUVmean+Volume+TLF", "Ref: Volume", "Ref: SUVmean",
                                          "Ref: TLF", "Ref: SUVmax"]]
    rows = []
    for s, m in models:
        g = oof[(oof.setting == s) & (oof.model == m)].copy()
        g["kind"] = g.region_id.map(kind)
        for neg in ["organ", "focal physiological"]:
            h = g[g.kind.isin([neg, "pathological"])]
            lo, hi = boot_ci(h, np.random.default_rng(SEED))
            rows.append({"setting": s, "model": m, "negatives": neg, "n_neg": int((h.label == 0).sum()),
                         "n_pos": int(h.label.sum()), "n_patients": h.patient_id.nunique(),
                         "AUC": fast_auc(h.label.to_numpy(), h.prob.to_numpy()), "AUC_CI_low": lo, "AUC_CI_high": hi,
                         "specificity_at_0.5": float((h[h.label == 0].pred == 0).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(Path(a.out) / "summary" / "negative_subtype_auc.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
