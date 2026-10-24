"""
Model interpretation (Figure 9): SHAP for the final model of each setting.

Performance estimates come exclusively from nested cross-validation. For
interpretation, the complete pipeline is applied once to the full cohort of an
experiment, exactly as it is applied to an outer-training set:
constant-feature removal -> median imputation -> Yeo-Johnson -> ComBat (batch = center) -> four-stage feature
selection -> z-scoring -> SMOTE -> Optuna tuning by patient-grouped 3-fold CV
(30 TPE trials) -> fit. The classifier is the tree ensemble with the highest
pooled out-of-fold AUC in that experiment. SHAP values (TreeExplainer) are
computed for the 1,020 original foci (synthetic SMOTE samples excluded) and
describe this fitted model; they are not performance estimates. The fitted final pipelines are saved.

Usage:
    python final_model_shap.py --data ../01_data --out ..
"""

import argparse
import sys
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from fapi_pipeline.models import build_model
from fapi_pipeline.nested_cv import preprocess
from run_experiment import CONFIG, EXPERIMENTS, VOLUME_COL

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SETTINGS = {k: v[0] for k, v in EXPERIMENTS.items()}
TREES = ["Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM"]
GRID = {("PET Only", "64"): "exp1_pet64", ("PET Only", "128"): "exp2_pet128",
        ("PET/CT", "64"): "exp3_pet64_ct", ("PET/CT", "128"): "exp4_pet128_ct"}
HEADER = {"fontsize": 20, "fontweight": "bold", "family": "Times New Roman"}


def short(f):
    s = f.split("(IBSI")[0]
    for a, b in [("Mo_", "Morph_")]:
        s = s.replace(a, b)
    mod = "CT" if s.startswith("CT_") else "PET"
    s = s.replace("PET_", "", 1).replace("CT_", "", 1) + f"_{mod}"
    if len(s) > 24:  # wrap long names at '-' or '_' boundaries
        import re
        tokens = re.findall(r"[^-_]+[-_]?", s)
        lines, line = [], ""
        for t in tokens:
            if len(line) + len(t) > 24 and line:
                lines.append(line)
                line = ""
            line += t
        lines.append(line)
        s = "\n".join(lines)
    return s


def fit_final(df, model_name, cfg):
    rs = cfg["RANDOM_STATE"]
    feats = [c for c in df.columns if c.startswith(("PET_", "CT_"))]
    X = df[feats].to_numpy(float)
    y = df.label.to_numpy(int)
    g = df.patient_id.to_numpy()
    b = df.center.to_numpy()
    vol = df[VOLUME_COL].to_numpy(float)

    cache = []
    for tr, va in StratifiedGroupKFold(3, shuffle=True, random_state=rs).split(X, y, g):
        Xb, yb, _, Xv, _, _, _ = preprocess(X[tr], y[tr], b[tr], vol[tr], X[va], b[va], feats, cfg, rs)
        cache.append((Xb, yb, Xv, y[va]))

    def objective(trial):
        proto = build_model(trial, model_name, rs)
        return float(np.mean([roc_auc_score(yv, clone(proto).fit(Xb, yb).predict_proba(Xv)[:, 1])
                              for Xb, yb, Xv, yv in cache]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rs))
    study.optimize(objective, n_trials=cfg["N_OPTUNA_TRIALS"])

    # Full-cohort preprocessing; the 'evaluation' partition is the cohort itself so that
    # SHAP is computed on the original (non-synthetic) regions in the fitted feature space.
    Xb, yb, X_all_s, _, names, log, fprep = preprocess(X, y, b, vol, X, b, feats, cfg, rs)
    model = build_model(optuna.trial.FixedTrial(study.best_params), model_name, rs).fit(Xb, yb)
    sv = shap.TreeExplainer(model).shap_values(X_all_s)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    # Colour by harmonized feature values on their original scale (z-scores are monotone in them).
    return {"model": model_name, "features": names, "shap": sv, "X": X_all_s,
            "best_params": study.best_params, "inner_auc": study.best_value, "fs_log": log,
            "preprocessor": fprep, "fitted_model": model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../01_data")
    ap.add_argument("--out", default="..")
    a = ap.parse_args()
    root = Path(a.out)
    perf = pd.read_csv(root / "summary" / "performance_summary.csv")
    sel = pd.concat([pd.read_csv(root / EXPERIMENTS[s][1] / "selected_features.csv") for s in SETTINGS])
    outer = sel[sel.inner_fold == 0]

    results, rows = {}, []
    for s, fname in SETTINGS.items():
        sub = perf[(perf.setting == s) & perf.model.isin(TREES)].sort_values("AUC", ascending=False)
        model_name, kappa, auc = sub.model.iloc[0], sub.Kappa.iloc[0], sub.AUC.iloc[0]
        res = fit_final(pd.read_csv(Path(a.data) / fname), model_name, CONFIG)
        res.update({"kappa_cv": kappa, "auc_cv": auc})
        results[s] = res
        fold_sets = [set(x.split("; ")) for x in outer[outer.setting == s].features]
        for k, f in enumerate(res["features"]):
            rows.append({"setting": s, "model": model_name, "feature": f, "feature_short": short(f),
                         "mean_abs_shap": float(np.abs(res["shap"][:, k]).mean()),
                         "outer_folds_selected": sum(f in fs for fs in fold_sets)})
        print(s, model_name, res["features"])

    pd.DataFrame(rows).sort_values(["setting", "mean_abs_shap"], ascending=[True, False]) \
        .to_csv(root / "summary" / "final_model_shap_importance.csv", index=False)
    with open(root / "summary" / "final_model_shap.pkl", "wb") as fh:
        pickle.dump(results, fh)

    plot_figure9(results, root)


def plot_figure9(results, root):
    """Figure 9: each panel rendered as its own figure (official SHAP beeswarm), then composed in the 2 x 2 template."""
    import io
    from PIL import Image
    plt.style.use("seaborn-v0_8-white")
    panels = {}
    for s, res in results.items():
        plt.rcParams.update({"font.size": 14})
        fig = plt.figure(figsize=(6.6, 4.6))
        ax = fig.add_subplot(111)
        exp = shap.Explanation(values=res["shap"], data=res["X"], feature_names=[short(f) for f in res["features"]])
        shap.plots.beeswarm(exp, max_display=10, ax=ax, show=False, plot_size=None, s=16)
        scale = "probability" if res["model"] in ("Random Forest", "Extra Trees") else "log-odds"
        ax.set_xlabel(f"SHAP value ({scale})", fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=13.5); ax.tick_params(axis="x", labelsize=13)
        ax.set_title(f"{res['model']} (final model)\nnested-CV AUC = {res['auc_cv']:.3f}; κ = {res['kappa_cv']:.3f}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, dpi=600, bbox_inches="tight"); plt.close(fig)
        panels[s] = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    fig = plt.figure(figsize=(14, 10.4))
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.06)
    for (r, c), s in GRID.items():
        i, j = ["PET Only", "PET/CT"].index(r), ["64", "128"].index(c)
        ax = fig.add_subplot(gs[i, j]); ax.imshow(panels[s]); ax.axis("off")
        if i == 0:
            fig.text((ax.get_position().x0 + ax.get_position().x1) / 2, 0.97, f"Gray Level ({c})",
                     ha="center", va="center", **HEADER)
        if j == 0:
            b = ax.get_position()
            fig.text(0.02, (b.y0 + b.y1) / 2, r, rotation=90, ha="center", va="center", **HEADER)
    (root / "figures_manuscript").mkdir(exist_ok=True)
    out = root / "figures_manuscript" / "Figure9_SHAP.png"   # manuscript Figure 9 (same template as before)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        _root = Path("..")
        plot_figure9(pickle.load(open(_root / "summary" / "final_model_shap.pkl", "rb")), _root)
    else:
        main()
