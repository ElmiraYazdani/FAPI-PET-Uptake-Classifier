"""
Effect of patient clustering on uncertainty (Editor comment 5).

Compares the patient-cluster bootstrap 95% CIs (from summarize.py) with
region-level DeLong intervals that ignore clustering, and computes
patient-weighted ROC-AUCs (each region weighted by 1 / number of regions of its
patient, so that every patient contributes equally).


Usage:
    python clustering_effect.py --out ..
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MAIN = ["exp1_pet64", "exp2_pet128", "exp3_pet64_ct", "exp4_pet128_ct"]


def delong_var(y, p):
    pos, neg = p[y == 1], p[y == 0]
    v10 = np.array([(x > neg).mean() + 0.5 * (x == neg).mean() for x in pos])
    v01 = np.array([(pos > x).mean() + 0.5 * (pos == x).mean() for x in neg])
    return v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)


def row(g, r):
    y, p = g.label.to_numpy(), g.prob.to_numpy()
    auc, se = roc_auc_score(y, p), np.sqrt(delong_var(y, p))
    w = 1 / g.groupby("patient_id").patient_id.transform("size").to_numpy()
    return {"auc": auc, "naive_lo": auc - 1.96 * se, "naive_hi": min(1.0, auc + 1.96 * se),
            "clus_lo": r.AUC_CI_low, "clus_hi": r.AUC_CI_high,
            "width_ratio": (r.AUC_CI_high - r.AUC_CI_low) / (2 * 1.96 * se),
            "patient_weighted_auc": roc_auc_score(y, p, sample_weight=w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    root = Path(ap.parse_args().out) / "summary"
    perf = pd.read_csv(root / "performance_summary.csv")
    oof = pd.read_csv(root / "all_oof_predictions.csv")
    N = oof[oof.setting == MAIN[0]].region_id.nunique()
    rows = []
    for s in MAIN + ["exp5_ct"]:
        r = perf[perf.setting == s].sort_values("AUC", ascending=False).iloc[0]
        rows.append({"setting": s, "model": r.model, **row(oof[(oof.setting == s) & (oof.model == r.model)], r)})
    for m in ["Ref: SUVmax+SUVmean+Volume+TLF", "Ref: Volume"]:
        r = perf[(perf.setting == "reference") & (perf.model == m)].iloc[0]
        rows.append({"setting": "reference", "model": m, **row(oof[(oof.setting == "reference") & (oof.model == m)], r)})
    pd.DataFrame(rows).to_csv(root / "clustering_effect.csv", index=False)
    ratios = [row(g, perf[(perf.setting == s) & (perf.model == m)].iloc[0])["width_ratio"]
              for (s, m), g in oof[oof.setting.isin(MAIN)].groupby(["setting", "model"])]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"CI width ratio across {len(ratios)} combinations: median {np.median(ratios):.2f}, "
          f"range {min(ratios):.2f}-{max(ratios):.2f}")
    de = np.square(ratios)  # approximate design effect (variance ratio)
    print(f"Design effect: median {np.median(de):.2f}, IQR {np.percentile(de, 25):.2f}-{np.percentile(de, 75):.2f}; "
          f"effective n of {N} foci: {N / np.median(de):.0f} "
          f"(IQR {N / np.percentile(de, 75):.0f}-{N / np.percentile(de, 25):.0f})")


if __name__ == "__main__":
    main()
