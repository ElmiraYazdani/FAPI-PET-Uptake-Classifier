"""
Manuscript-ready tables for the four experiments (CSV; an .xlsx workbook is also written if openpyxl is installed).

Inputs: summary/performance_summary.csv, summary/paired_comparisons.csv, summary/permutation_check.csv,
        <Exp folder>/{fold_metrics,selected_features,best_params}.csv, 01_data/data_check_report.json
Output: tables/

Usage:
    python make_tables.py --out ..
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from run_experiment import EXPERIMENTS
from summarize import LABELS, MAIN

METRICS = ["AUC", "AUC_CI_low", "AUC_CI_high", "Fold_AUC_mean", "Fold_AUC_SD", "PR_AUC", "Accuracy",
           "Balanced_accuracy", "BalAcc_CI_low", "BalAcc_CI_high", "Sensitivity", "Specificity", "PPV", "NPV", "F1",
           "Kappa", "Kappa_CI_low", "Kappa_CI_high", "Brier", "Cal_intercept", "Cal_slope", "TN", "FP", "FN", "TP"]


def ci(r, m, lo, hi):
    return f"{r[m]:.3f} ({r[lo]:.3f}-{r[hi]:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    a = ap.parse_args()
    root = Path(a.out)
    tab = root / "tables"
    tab.mkdir(exist_ok=True)
    perf = pd.read_csv(root / "summary" / "performance_summary.csv")
    comp = pd.read_csv(root / "summary" / "paired_comparisons.csv")
    rep = json.load(open(root / "01_data" / "data_check_report.json"))
    T = {}

    # 1. datasets
    rows = []
    for s in MAIN:
        e = rep["experiments"][s]
        sel = pd.read_csv(root / EXPERIMENTS[s][1] / "selected_features.csv")
        o = sel[sel.inner_fold == 0]
        rows.append({"Experiment": LABELS[s], "Foci": e["observations"], "Patients": e["patients"],
                     "Class 0 (physiological)": e["class_counts"]["0"], "Class 1 (pathological)": e["class_counts"]["1"],
                     "Candidate features": e["candidate_features"], "PET features": e["PET_features"],
                     "CT features": e["CT_features"], "Missing cells (imputed in training folds)": e["missing_cells"],
                     "After volume filter (outer folds)": "/".join(map(str, o.n_after_volume)),
                     "After redundancy filter": "/".join(map(str, o.n_after_redundancy)),
                     "After mRMR": "/".join(map(str, o.n_after_mrmr)),
                     "Final selected (outer folds 1-4)": "/".join(map(str, o.n_features))})
    T["T1_datasets_and_feature_counts"] = pd.DataFrame(rows)

    # 2. all models, all metrics
    allm = perf[perf.setting.isin(MAIN + ["reference"])].copy()
    allm["Experiment"] = allm.setting.map(LABELS)
    T["T2_performance_all_models"] = allm[["Experiment", "model", "n_regions", "n_neg", "n_pos"] + METRICS] \
        .sort_values(["Experiment", "AUC"], ascending=[True, False])

    # 3. consolidated: same classifier across the four experiments (pooled out-of-fold AUC with 95% CI)
    piv = {}
    for s in MAIN:
        for _, r in perf[perf.setting == s].iterrows():
            piv.setdefault(r.model, {})[LABELS[s]] = ci(r, "AUC", "AUC_CI_low", "AUC_CI_high")
    T["T3_consolidated_AUC_by_classifier"] = pd.DataFrame(piv).T.reset_index().rename(columns={"index": "Classifier"})

    # 4. consolidated: best classifier per experiment (exploratory, highest pooled AUC) with full metrics
    rows = []
    for s in MAIN:
        sub = perf[perf.setting == s].sort_values("AUC", ascending=False)
        r = sub.iloc[0]
        rows.append({"Experiment": LABELS[s], "Classifier (highest pooled AUC)": r.model,
                     "AUC (95% CI)": ci(r, "AUC", "AUC_CI_low", "AUC_CI_high"),
                     "Fold AUC mean +/- SD": f"{r.Fold_AUC_mean:.3f} +/- {r.Fold_AUC_SD:.3f}",
                     "Median AUC of 10 classifiers": round(sub.AUC.median(), 3),
                     "Balanced accuracy (95% CI)": ci(r, "Balanced_accuracy", "BalAcc_CI_low", "BalAcc_CI_high"),
                     "Accuracy": round(r.Accuracy, 3), "Sensitivity": round(r.Sensitivity, 3),
                     "Specificity": round(r.Specificity, 3), "Precision (PPV)": round(r.PPV, 3),
                     "NPV": round(r.NPV, 3), "F1": round(r.F1, 3),
                     "Kappa (95% CI)": ci(r, "Kappa", "Kappa_CI_low", "Kappa_CI_high"),
                     "Brier": round(r.Brier, 3), "Calibration intercept": round(r.Cal_intercept, 3),
                     "Calibration slope": round(r.Cal_slope, 3), "TN/FP/FN/TP": f"{r.TN}/{r.FP}/{r.FN}/{r.TP}"})
    for _, r in perf[perf.setting == "reference"].sort_values("AUC", ascending=False).iterrows():
        rows.append({"Experiment": "Conventional reference", "Classifier (highest pooled AUC)": r.model,
                     "AUC (95% CI)": ci(r, "AUC", "AUC_CI_low", "AUC_CI_high"),
                     "Fold AUC mean +/- SD": f"{r.Fold_AUC_mean:.3f} +/- {r.Fold_AUC_SD:.3f}",
                     "Balanced accuracy (95% CI)": ci(r, "Balanced_accuracy", "BalAcc_CI_low", "BalAcc_CI_high"),
                     "Accuracy": round(r.Accuracy, 3), "Sensitivity": round(r.Sensitivity, 3),
                     "Specificity": round(r.Specificity, 3), "Precision (PPV)": round(r.PPV, 3),
                     "NPV": round(r.NPV, 3), "F1": round(r.F1, 3),
                     "Kappa (95% CI)": ci(r, "Kappa", "Kappa_CI_low", "Kappa_CI_high"),
                     "Brier": round(r.Brier, 3), "Calibration intercept": round(r.Cal_intercept, 3),
                     "Calibration slope": round(r.Cal_slope, 3), "TN/FP/FN/TP": f"{r.TN}/{r.FP}/{r.FN}/{r.TP}"})
    T["T4_consolidated_best_per_experiment"] = pd.DataFrame(rows)

    # 5. statistical comparisons
    T["T5_configuration_contrasts"] = comp[comp.family == "configuration contrasts (same classifier)"]
    T["T6_radiomics_vs_conventional"] = comp[comp.family == "radiomics (top per setting) vs conventional"]
    T["T7_pairwise_within_experiment"] = comp[comp.family.str.startswith("within")]

    # 6. fold composition, features, hyperparameters
    fm, fs, bp = [], [], []
    for s in MAIN:
        d = root / EXPERIMENTS[s][1]
        f = pd.read_csv(d / "fold_metrics.csv")
        fm.append(f.drop_duplicates("outer_fold")[["setting", "outer_fold", "n_train", "n_test", "n_test_neg",
                                                   "n_test_pos", "test_patients", "n_features"]])
        sel = pd.read_csv(d / "selected_features.csv")
        o = sel[sel.inner_fold == 0]
        cnt = pd.Series([x for v in o.features for x in v.split("; ")]).value_counts()
        fs.append(pd.DataFrame({"setting": s, "feature": cnt.index, "outer_folds_selected (of 4)": cnt.values}))
        bp.append(pd.read_csv(d / "best_params.csv"))
    fm = pd.concat(fm)
    fm["Experiment"] = fm.setting.map(LABELS)
    T["T8_outer_fold_composition"] = fm
    T["T9_selected_features_outer_folds"] = pd.concat(fs)
    T["T10_best_hyperparameters"] = pd.concat(bp)
    if (root / "summary" / "permutation_check.csv").exists():
        T["T11_permutation_leakage_check"] = pd.read_csv(root / "summary" / "permutation_check.csv")

    for k, v in T.items():
        v.to_csv(tab / f"{k}.csv", index=False)
    try:
        with pd.ExcelWriter(tab / "Results_tables.xlsx") as xw:
            for k, v in T.items():
                v.to_excel(xw, sheet_name=k[:31], index=False)
    except ImportError:
        print("openpyxl not installed: CSV tables only")
    print(T["T4_consolidated_best_per_experiment"].iloc[:, :6].to_string(index=False))
    print(T["T3_consolidated_AUC_by_classifier"].to_string(index=False))


if __name__ == "__main__":
    main()
