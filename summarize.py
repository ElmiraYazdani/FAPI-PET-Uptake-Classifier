"""
Performance summary, patient-cluster bootstrap inference and calibration.

All results are computed from the out-of-fold prediction files written by
run_experiment.py and reference_models.py (one row per focus x model).
Adapted from the previous analysis (Revision_Analysis/code/summarize.py): experiment names, the four
configuration contrasts and the label-permutation leakage check. Comparison families as in the manuscript:
within-experiment (incl. CT-only), configuration contrasts (6 per classifier, incl. vs CT-only), ComBat vs no ComBat,
radiomics (top per experiment, incl. CT-only) vs conventional.

* Point estimates: pooled out-of-fold metrics (all regions, one prediction each)
  and the mean +/- SD of the four outer-fold ROC-AUCs.
* Uncertainty: patient-cluster bootstrap (2000 resamples of the 24 patients
  with replacement; all regions of a sampled patient are kept together),
  percentile 95% CIs. The intervals are conditional on the fixed out-of-fold
  predictions and do not capture the variability of model re-training.
* Paired comparisons: AUC differences computed on the same bootstrap patient
  samples for both models; two-sided bootstrap P values; Holm adjustment within
  each comparison family.
* Calibration: Brier score, calibration intercept (calibration-in-the-large,
  logistic model with logit(p) as offset) and calibration slope.

Usage:
    python summarize.py --out ..
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, cohen_kappa_score, confusion_matrix

B = 2000
SEED = 100
MAIN = ["exp1_pet64", "exp2_pet128", "exp3_pet64_ct", "exp4_pet128_ct"]
LABELS = {"exp1_pet64": "Exp 1: PET-64", "exp2_pet128": "Exp 2: PET-128", "exp3_pet64_ct": "Exp 3: PET-64 + CT",
          "exp4_pet128_ct": "Exp 4: PET-128 + CT", "exp5_ct": "CT-only", "reference": "Conventional",
          "exp1_pet64_nocombat": "Exp 1: PET-64, no ComBat", "exp2_pet128_nocombat": "Exp 2: PET-128, no ComBat",
          "exp3_pet64_ct_nocombat": "Exp 3: PET-64 + CT, no ComBat",
          "exp4_pet128_ct_nocombat": "Exp 4: PET-128 + CT, no ComBat"}


def fast_auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:
        slope = np.nan
    try:
        intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        intercept = np.nan
    return intercept, slope


def point_metrics(y, p, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    ci, cs = calib(y, p)
    return {"AUC": fast_auc(y, p), "PR_AUC": average_precision_score(y, p),
            "Accuracy": (tp + tn) / len(y), "Balanced_accuracy": (sens + spec) / 2,
            "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv,
            "F1": 2 * tp / (2 * tp + fp + fn), "Kappa": cohen_kappa_score(y, pred),
            "Brier": brier_score_loss(y, p), "Cal_intercept": ci, "Cal_slope": cs,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def boot_indices(patients, region_patient, rng):
    by_pat = {pt: np.where(region_patient == pt)[0] for pt in patients}
    for _ in range(B):
        draw = rng.choice(patients, size=len(patients), replace=True)
        yield np.concatenate([by_pat[pt] for pt in draw])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    a = ap.parse_args()
    root = Path(a.out)
    summ = root / "summary"
    summ.mkdir(exist_ok=True)

    frames = [pd.read_csv(f) for f in sorted(root.glob("*/oof_predictions.csv"))]
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(summ / "all_oof_predictions.csv", index=False)

    # Wide probability matrices per region set for paired bootstrap
    rows, boot_store = [], {}
    for (setting, model), g in oof.groupby(["setting", "model"], sort=False):
        g = g.sort_values("region_id")
        y, p, pred = g.label.to_numpy(), g.prob.to_numpy(), g.pred.to_numpy()
        m = point_metrics(y, p, pred)
        fold_auc = g.groupby("outer_fold").apply(lambda f: fast_auc(f.label.to_numpy(), f.prob.to_numpy()),
                                                 include_groups=False)
        m.update({"setting": setting, "model": model, "n_regions": len(g), "n_pos": int(y.sum()),
                  "n_neg": int((1 - y).sum()), "Fold_AUC_mean": fold_auc.mean(), "Fold_AUC_SD": fold_auc.std(ddof=0)})
        rows.append(m)
        boot_store[(setting, model)] = g

    res = pd.DataFrame(rows)

    # Patient-cluster bootstrap for each model, using shared resamples per region set
    rng = np.random.default_rng(SEED)
    region_sets = {}
    for key, g in boot_store.items():
        region_sets.setdefault(tuple(g.region_id), []).append(key)
    boot_auc = {}
    ci_rows = []
    for regions, keys in region_sets.items():
        g0 = boot_store[keys[0]]
        pts = np.unique(g0.patient_id)
        idx_list = list(boot_indices(pts, g0.patient_id.to_numpy(), np.random.default_rng(SEED)))
        for key in keys:
            g = boot_store[key]
            y, p, pred = g.label.to_numpy(), g.prob.to_numpy(), g.pred.to_numpy()
            aucs, prs, bals, kaps, briers = [], [], [], [], []
            for idx in idx_list:
                yy, pp, dd = y[idx], p[idx], pred[idx]
                if yy.min() == yy.max():
                    continue
                aucs.append(fast_auc(yy, pp))
                prs.append(average_precision_score(yy, pp))
                se = (dd[yy == 1] == 1).mean()
                sp = (dd[yy == 0] == 0).mean()
                bals.append((se + sp) / 2)
                kaps.append(cohen_kappa_score(yy, dd))
                briers.append(np.mean((pp - yy) ** 2))
            boot_auc[key] = np.array([fast_auc(y[idx], p[idx]) if y[idx].min() != y[idx].max() else np.nan
                                      for idx in idx_list])
            q = lambda v: np.nanpercentile(v, [2.5, 97.5])
            ci_rows.append({"setting": key[0], "model": key[1],
                            "AUC_CI_low": q(aucs)[0], "AUC_CI_high": q(aucs)[1],
                            "PR_AUC_CI_low": q(prs)[0], "PR_AUC_CI_high": q(prs)[1],
                            "BalAcc_CI_low": q(bals)[0], "BalAcc_CI_high": q(bals)[1],
                            "Kappa_CI_low": q(kaps)[0], "Kappa_CI_high": q(kaps)[1],
                            "Brier_CI_low": q(briers)[0], "Brier_CI_high": q(briers)[1],
                            "n_boot_valid": len(aucs)})
    res = res.merge(pd.DataFrame(ci_rows), on=["setting", "model"])
    res["setting_label"] = res.setting.map(LABELS)
    res.sort_values(["setting", "AUC"], ascending=[True, False]).to_csv(summ / "performance_summary.csv", index=False)

    def paired(k1, k2):
        d = boot_auc[k1] - boot_auc[k2]
        d = d[~np.isnan(d)]
        a1 = res.set_index(["setting", "model"]).loc[k1, "AUC"]
        a2 = res.set_index(["setting", "model"]).loc[k2, "AUC"]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        p = max(p, 1 / len(d))
        return {"A": f"{LABELS.get(k1[0], k1[0])} | {k1[1]}", "B": f"{LABELS.get(k2[0], k2[0])} | {k2[1]}",
                "AUC_A": a1, "AUC_B": a2, "Delta_AUC": a1 - a2,
                "CI_low": np.percentile(d, 2.5), "CI_high": np.percentile(d, 97.5), "P_boot": p}

    comps = []
    # Family 1: all 45 pairwise classifier comparisons within each main setting
    for s in MAIN + ["exp5_ct"]:
        models = sorted(oof[oof.setting == s].model.unique())
        fam = [paired((s, m1), (s, m2)) for m1, m2 in itertools.combinations(models, 2)]
        for r_ in fam:
            r_["family"] = f"within {LABELS[s]}"
        adj = holm([r_["P_boot"] for r_ in fam])
        for r_, pa in zip(fam, adj):
            r_["P_Holm"] = pa
        comps += fam
    # Family 2: modality and discretization contrasts for each classifier (same 1,020 foci)
    fam = []
    for m in sorted(oof[oof.setting == "exp3_pet64_ct"].model.unique()):
        for s1, s2 in [("exp3_pet64_ct", "exp1_pet64"), ("exp4_pet128_ct", "exp2_pet128"),
                       ("exp1_pet64", "exp2_pet128"), ("exp3_pet64_ct", "exp4_pet128_ct"),
                       ("exp3_pet64_ct", "exp5_ct"), ("exp4_pet128_ct", "exp5_ct")]:
            fam.append(paired((s1, m), (s2, m)))
    for r_ in fam:
        r_["family"] = "configuration contrasts (same classifier)"
    for r_, pa in zip(fam, holm([r_["P_boot"] for r_ in fam])):
        r_["P_Holm"] = pa
    comps += fam
    # Family 2b: ComBat vs no ComBat for each classifier and main setting
    fam = []
    for s_ in MAIN:
        if (s_ + "_nocombat") not in set(oof.setting):
            continue
        for m in sorted(oof[oof.setting == s_].model.unique()):
            fam.append(paired((s_, m), (s_ + "_nocombat", m)))
    for r_ in fam:
        r_["family"] = "ComBat vs no ComBat (same classifier)"
    for r_, pa in zip(fam, holm([r_["P_boot"] for r_ in fam])):
        r_["P_Holm"] = pa
    comps += fam
    # Family 3: each setting's highest-AUC radiomics model vs conventional reference models
    fam = []
    refs = sorted(oof[oof.setting == "reference"].model.unique())
    for s in MAIN + ["exp5_ct"]:
        top = res[res.setting == s].sort_values("AUC", ascending=False).model.iloc[0]
        for rm in refs:
            fam.append(paired((s, top), ("reference", rm)))
    for r_ in fam:
        r_["family"] = "radiomics (top per setting) vs conventional"
    for r_, pa in zip(fam, holm([r_["P_boot"] for r_ in fam])):
        r_["P_Holm"] = pa
    comps += fam
    pd.DataFrame(comps).to_csv(summ / "paired_comparisons.csv", index=False)

    # Label-permutation leakage check: with shuffled labels, out-of-fold AUCs should be close to 0.5
    perm = [pd.read_csv(f).assign(run=f.parent.name) for f in sorted(root.glob("permutation_tests/*/oof_predictions.csv"))]
    if perm:
        prm = pd.concat(perm)
        prm.groupby(["run", "setting", "model"]).apply(lambda g: fast_auc(g.label.to_numpy(), g.prob.to_numpy()),
                                                include_groups=False).rename("AUC_permuted_labels")             .reset_index().to_csv(summ / "permutation_check.csv", index=False)

    meta = {"bootstrap_resamples": B, "seed": SEED, "resampling_unit": "patient",
            "p_value": "two-sided percentile bootstrap", "multiplicity": "Holm within family",
            "n_comparisons_per_family": pd.DataFrame(comps).groupby("family").size().to_dict()}
    with open(summ / "summary_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(res[["setting", "model", "AUC", "AUC_CI_low", "AUC_CI_high", "Fold_AUC_mean", "Fold_AUC_SD",
               "Balanced_accuracy", "Kappa", "Brier", "Cal_intercept", "Cal_slope"]]
          .sort_values(["setting", "AUC"], ascending=[True, False]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
