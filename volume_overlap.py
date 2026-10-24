"""
Volume-overlap sensitivity analysis (Editor comment 8).

Re-evaluates the existing out-of-fold predictions (no re-training) in the subset of
regions whose volume lies within the range in which normal-organ and tumor-lesion
regions overlap (smallest normal-organ volume to largest tumor-lesion volume). If a
model only re-encoded region size, its discrimination should collapse in this subset.
Patient-cluster bootstrap (2,000 resamples) gives 95% CIs and paired differences
versus the volume-only reference model; Holm adjustment across the comparisons.


Usage:
    python volume_overlap.py --out .. --data ../01_data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize import B, SEED, fast_auc, holm

VOL = "PET_Mo_Volume(IBSI:RNU0)[cm3]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    ap.add_argument("--data", default="../01_data")
    a = ap.parse_args()
    oof = pd.read_csv(Path(a.out) / "summary" / "all_oof_predictions.csv")
    perf = pd.read_csv(Path(a.out) / "summary" / "performance_summary.csv")
    d = pd.read_csv(Path(a.data) / "exp1_pet64.csv")[["region_id", "patient_id", "label", VOL]]
    lo, hi = d[d.label == 0][VOL].min(), d[d.label == 1][VOL].max()
    sub_ids = set(d[(d[VOL] >= lo) & (d[VOL] <= hi)].region_id)
    sub = d[d.region_id.isin(sub_ids)]

    models = [(s, perf[perf.setting == s].sort_values("AUC", ascending=False).model.iloc[0])
              for s in ["exp1_pet64", "exp2_pet128", "exp3_pet64_ct", "exp4_pet128_ct", "exp5_ct"]]
    models += [("reference", "Ref: Volume"), ("reference", "Ref: SUVmax+SUVmean+Volume+TLF"),
               ("reference", "Ref: TLF"), ("reference", "Ref: SUVmean"), ("reference", "Ref: SUVmax")]

    preds = {}
    for s, m in models:
        g = oof[(oof.setting == s) & (oof.model == m) & oof.region_id.isin(sub_ids)].sort_values("region_id")
        preds[(s, m)] = g
    ref = preds[("reference", "Ref: Volume")]
    pts = np.unique(ref.patient_id)
    by_pat = {p: np.where(ref.patient_id.to_numpy() == p)[0] for p in pts}
    rng = np.random.default_rng(SEED)
    idx_list = [np.concatenate([by_pat[p] for p in rng.choice(pts, len(pts), replace=True)]) for _ in range(B)]

    def boot(g):
        y, p = g.label.to_numpy(), g.prob.to_numpy()
        return np.array([fast_auc(y[i], p[i]) if y[i].min() != y[i].max() else np.nan for i in idx_list])

    conv = preds[("reference", "Ref: SUVmax+SUVmean+Volume+TLF")]
    rows, bvol, bconv = [], boot(ref), boot(conv)
    auc_conv = fast_auc(conv.label.to_numpy(), conv.prob.to_numpy())
    for (s, m), g in preds.items():
        assert (g.region_id.to_numpy() == ref.region_id.to_numpy()).all()
        b = boot(g)
        dlt = b - bvol
        dlt = dlt[~np.isnan(dlt)]
        p = max(min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean())), 1 / len(dlt))
        auc = fast_auc(g.label.to_numpy(), g.prob.to_numpy())
        dc = (b - bconv)[~np.isnan(b - bconv)]
        pc = max(min(1.0, 2 * min((dc <= 0).mean(), (dc >= 0).mean())), 1 / len(dc))
        rows.append({"Delta_vs_conventional": auc - auc_conv, "DeltaC_CI_low": np.percentile(dc, 2.5),
                     "DeltaC_CI_high": np.percentile(dc, 97.5), "P_boot_conv": pc,
                     "setting": s, "model": m, "n_regions": len(g), "n_neg": int((1 - g.label).sum()),
                     "n_pos": int(g.label.sum()), "AUC": auc, "AUC_CI_low": np.nanpercentile(b, 2.5),
                     "AUC_CI_high": np.nanpercentile(b, 97.5),
                     "Delta_vs_volume": auc - fast_auc(ref.label.to_numpy(), ref.prob.to_numpy()),
                     "Delta_CI_low": np.percentile(dlt, 2.5), "Delta_CI_high": np.percentile(dlt, 97.5), "P_boot": p})
    res = pd.DataFrame(rows)
    mask = ~((res.setting == "reference") & (res.model == "Ref: Volume"))
    res.loc[mask, "P_Holm"] = holm(res.loc[mask, "P_boot"].to_numpy())
    radi = res.setting != "reference"
    res.loc[radi, "P_Holm_conv"] = holm(res.loc[radi, "P_boot_conv"].to_numpy())
    res.to_csv(Path(a.out) / "summary" / "volume_overlap_subset.csv", index=False)
    res["overlap_low_cm3"], res["overlap_high_cm3"], res["n_patients"] = lo, hi, sub.patient_id.nunique()
    res.to_csv(Path(a.out) / "summary" / "volume_overlap_subset.csv", index=False)
    print(f"Overlap range {lo:.1f}-{hi:.1f} cm3: {len(sub)} foci ({int((1 - sub.label).sum())} class 0, "
          f"{int(sub.label.sum())} class 1) from {sub.patient_id.nunique()} patients")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
