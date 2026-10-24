"""
Run the leakage-free nested, patient-grouped cross-validation for one experiment (all foci of uptake).

Experiments (datasets written by build_datasets.py):
    exp1_pet64      PET-64 only          145 candidate features
    exp2_pet128     PET-128 only         145 candidate features
    exp3_pet64_ct   PET-64 + CT          290 candidate features
    exp4_pet128_ct  PET-128 + CT         290 candidate features

Configuration identical to the previous analysis (Revision_Analysis/code/run_setting.py) except
USE_YEOJOHNSON = True, fold-wise median imputation (see fapi_pipeline/nested_cv.py) and the SVM solver
iteration cap (fapi_pipeline/models.py).

Usage:
    python run_experiment.py --exp exp1_pet64 --data ../01_data --out ..
"""

import argparse
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fapi_pipeline.models import SVM_MAX_ITER
from fapi_pipeline.nested_cv import run_nested_cv

CONFIG = {
    "RANDOM_STATE": 100,
    "N_OUTER_FOLDS": 4,
    "N_INNER_FOLDS": 3,
    "N_OPTUNA_TRIALS": 30,
    "VOLUME_CORR_THRESHOLD": 0.50,
    "REDUNDANCY_CORR_THRESHOLD": 0.90,
    "MRMR_PER_MODALITY": 5,
    "STABILITY_N_ITERATIONS": 2000,
    "STABILITY_SUBSAMPLE_RATIO": 0.60,
    "STABILITY_SELECTION_THRESHOLD": 0.75,
    "STABILITY_FALLBACK_STEP": 0.05,
    "STABILITY_FALLBACK_MIN_THRESHOLD": 0.55,
    "STABILITY_MIN_FEATURES": 2,
    "ELASTICNET_L1_RATIOS": [0.60, 0.70, 0.80, 0.90, 0.95],
    "ELASTICNET_CS": [0.001, 0.01, 0.05, 0.10, 0.50],
    "ELASTICNET_COEF_THRESHOLD": 1e-3,
    "CLASS_THRESHOLD": 0.5,
    "USE_COMBAT": True,
    "USE_YEOJOHNSON": True,
}

EXPERIMENTS = {
    "exp1_pet64": ("exp1_pet64.csv", "Exp1_PET64", "PET-64"),
    "exp2_pet128": ("exp2_pet128.csv", "Exp2_PET128", "PET-128"),
    "exp3_pet64_ct": ("exp3_pet64_ct.csv", "Exp3_PET64_CT", "PET-64 + CT"),
    "exp4_pet128_ct": ("exp4_pet128_ct.csv", "Exp4_PET128_CT", "PET-128 + CT"),
    "exp5_ct": ("exp5_ct.csv", "Sens_CT_only", "CT only (sensitivity analysis)"),
}
VOLUME_COL = "PET_Mo_Volume(IBSI:RNU0)[cm3]"
CT_VOLUME_COL = "CT_Mo_Volume(IBSI:RNU0)[cm3]"   # volume filter of the CT-only analysis (as previously)
META = ["patient_id", "region_id", "focus", "center", "label"]


def main():
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=EXPERIMENTS)
    ap.add_argument("--data", default="../01_data")
    ap.add_argument("--out", default="..")
    ap.add_argument("--trials", type=int, default=None, help="override Optuna trials (smoke tests only)")
    ap.add_argument("--stab_iter", type=int, default=None, help="override stability iterations (smoke tests only)")
    ap.add_argument("--permute", type=int, default=None, help="leakage check: shuffle labels with this seed")
    ap.add_argument("--tag", default=None, help="override output folder name")
    ap.add_argument("--no_combat", action="store_true", help="sensitivity analysis without ComBat")
    a = ap.parse_args()

    cfg = dict(CONFIG)
    if a.trials:
        cfg["N_OPTUNA_TRIALS"] = a.trials
    if a.stab_iter:
        cfg["STABILITY_N_ITERATIONS"] = a.stab_iter
    if a.no_combat:
        cfg["USE_COMBAT"] = False

    fname, folder, label = EXPERIMENTS[a.exp]
    df = pd.read_csv(Path(a.data) / fname)
    feats = [c for c in df.columns if c.startswith(("PET_", "CT_"))]
    assert not set(feats) & set(META) and "label" not in feats and len(feats) + len(META) == df.shape[1]
    vol_col = CT_VOLUME_COL if a.exp == "exp5_ct" else VOLUME_COL
    assert df[vol_col].notna().all()
    if a.permute is not None:
        df["label"] = np.random.default_rng(a.permute).permutation(df["label"].to_numpy())
    tag = a.tag or (folder + ("_noComBat" if a.no_combat else "") + (f"_perm{a.permute}" if a.permute is not None else ""))
    setting = a.exp + ("_nocombat" if a.no_combat else "")
    out = Path(a.out) / ("permutation_tests" if a.permute is not None else "") / tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run_log.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"{label} ({a.exp}): {len(df)} foci, {df.patient_id.nunique()} patients, "
        f"labels {df.label.value_counts().sort_index().to_dict()}, {len(feats)} candidate features, "
        f"{int(df[feats].isna().sum().sum())} missing cells")
    t0 = time.time()
    oof, folds, feat, params, shap_store, fitted = run_nested_cv(df, feats, vol_col, cfg, setting, log)

    oof.to_csv(out / "oof_predictions.csv", index=False)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    feat.to_csv(out / "selected_features.csv", index=False)
    params.to_csv(out / "best_params.csv", index=False)
    with open(out / "shap_values.pkl", "wb") as fh:
        pickle.dump(shap_store, fh)
    if a.permute is None:
        with open(out / "outer_fold_models.pkl", "wb") as fh:
            pickle.dump(fitted, fh)

    import sklearn, xgboost, lightgbm, optuna, imblearn, shap
    meta = {"experiment": a.exp, "label": label, "output_folder": tag, "data_file": str(Path(a.data) / fname),
            "volume_col": vol_col, "n_foci": len(df), "n_patients": int(df.patient_id.nunique()),
            "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
            "n_candidate_features": len(feats), "permuted_labels_seed": a.permute, "config": cfg,
            "svm_max_iter": SVM_MAX_ITER,
            "runtime_min": round((time.time() - t0) / 60, 1),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                         "lightgbm": lightgbm.__version__, "optuna": optuna.__version__,
                         "imbalanced-learn": imblearn.__version__, "shap": shap.__version__}}
    with open(out / "run_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"Finished in {meta['runtime_min']} min")


if __name__ == "__main__":
    main()
