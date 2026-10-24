"""
Manuscript figures regenerated from the verified out-of-fold predictions, using
the layout and colour template of the originally submitted figures:
2 x 2 panels (rows: PET Only, PET/CT; columns: Gray Level 64, 128), all ten
classifiers in the matplotlib tab10 colours (fixed colour per classifier),
the highest-pooled-AUC classifier of each panel drawn with a thicker line.

    Figure 2  ROC curves
    Figure 3  Decision curves
    Figure 4  Pairwise comparison heatmaps (patient-cluster bootstrap, Holm)
    Figure 5  Confusion matrices (highest-AUC classifier per setting)
    Figure 6  Calibration curves (quantile binning) with Brier scores
    Figure 7  t-SNE coloured by patient
    Figure 8  t-SNE coloured by class
    Figure 9  SHAP summary (held-out outer-test regions)
    Fig. S    the same ROC layout for the no-ComBat and sensitivity analyses

Copied from the previous analysis (Revision_Analysis/code); only the experiment names, data files, class names
and the t-SNE missing-value handling (median imputation; the updated data contain a few missing cells) were changed.
Constants and helpers are imported by figures_print.py, which produces the manuscript figures.

Usage:
    python figures_original_style.py --out .. --data ../01_data
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-whitegrid")

MODEL_ORDER = ["Logistic Regression", "RBF-SVM", "Linear SVM", "Random Forest", "Gradient Boosting",
               "Extra Trees", "KNN", "Gaussian Naive Bayes", "XGBoost", "LightGBM"]
COLORS = dict(zip(MODEL_ORDER, plt.get_cmap("tab10").colors))
GRID = {("PET Only", "64"): "exp1_pet64", ("PET Only", "128"): "exp2_pet128",
        ("PET/CT", "64"): "exp3_pet64_ct", ("PET/CT", "128"): "exp4_pet128_ct"}
ROWS, COLS = ["PET Only", "PET/CT"], ["64", "128"]
DATA_FILES = {"exp1_pet64": "exp1_pet64.csv", "exp2_pet128": "exp2_pet128.csv",
              "exp3_pet64_ct": "exp3_pet64_ct.csv", "exp4_pet128_ct": "exp4_pet128_ct.csv"}
CLASS_NAMES = {0: "Class 0 (physiological uptake)", 1: "Class 1 (pathological uptake)"}
REF_MODEL = "Ref: SUVmax+SUVmean+Volume+TLF"
SHAP_CMAP = LinearSegmentedColormap.from_list("shap", ["#008bfb", "#ff0052"])
HEADER = {"fontsize": 26, "fontweight": "bold", "family": "Times New Roman"}
TITLE = {"fontsize": 13, "fontweight": "bold", "pad": 10}
LABEL = {"fontsize": 11, "fontweight": "bold", "labelpad": 8}


def frame(figsize=(18, 16), grid=GRID, rows=ROWS, cols=COLS):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=figsize)
    axes = np.atleast_2d(axes)
    fig._hdr = (rows, cols)
    return fig, axes


def finalize(fig, axes, path):
    """Lays out panels inside a margin and writes row/column headers into that margin."""
    rows, cols = fig._hdr
    fig.tight_layout(rect=(0.06, 0, 1, 0.94))
    for j, c in enumerate(cols):
        x = np.mean([axes[0, j].get_position().x0, axes[0, j].get_position().x1])
        fig.text(x, 0.975, f"Gray Level ({c})" if c in ("64", "128") else c, ha="center", va="center", **HEADER)
    for i, r in enumerate(rows):
        b = axes[i, 0].get_position()
        fig.text(0.02, (b.y0 + b.y1) / 2, r, rotation=90, ha="center", va="center", **HEADER)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def best_model(perf, s):
    return perf[perf.setting == s].sort_values("AUC", ascending=False).model.iloc[0]


def fig_roc(oof, perf, path, grid=GRID, rows=ROWS, cols=COLS):
    fig, axes = frame(grid=grid, rows=rows, cols=cols)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax, s = axes[i, j], grid[(r, c)]
            best = best_model(perf, s)
            for m in MODEL_ORDER:
                g = oof[(oof.setting == s) & (oof.model == m)]
                f, t, _ = roc_curve(g.label, g.prob)
                auc = perf[(perf.setting == s) & (perf.model == m)].AUC.iloc[0]
                ax.plot(f, t, color=COLORS[m], lw=3.5 if m == best else 2.0,
                        label=f"{m}{' (highest AUC)' if m == best else ''} (AUC = {auc:.3f})", zorder=3 if m == best else 2)
            ref = oof[(oof.setting == "reference") & (oof.model == REF_MODEL)]
            if len(ref) and (ref.region_id.nunique() == oof[oof.setting == s].region_id.nunique()):
                f, t, _ = roc_curve(ref.label, ref.prob)
                auc = perf[(perf.setting == "reference") & (perf.model == REF_MODEL)].AUC.iloc[0]
                ax.plot(f, t, color="black", ls="-.", lw=2.0, label=f"Conventional model (AUC = {auc:.3f})", zorder=4)
            ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.5, label="Chance (AUC = 0.500)")
            ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
            ax.set_xlabel("False Positive Rate (1 - Specificity)", **LABEL)
            ax.set_ylabel("True Positive Rate (Sensitivity)", **LABEL)
            ax.set_title("ROC Curves from Pooled Outer-Fold Predictions", **TITLE)
            ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", edgecolor="gainsboro")
            ax.grid(True, linestyle=":", alpha=0.6)
    finalize(fig, axes, path)


def net_benefit(y, p, thr):
    n = len(y)
    return np.array([(np.sum((p >= t) & (y == 1)) - np.sum((p >= t) & (y == 0)) * t / (1 - t)) / n for t in thr])


def fig_dca(oof, perf, path):
    thr = np.linspace(0.01, 0.80, 80)
    fig, axes = frame()
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            ax, s = axes[i, j], GRID[(r, c)]
            best = best_model(perf, s)
            for m in MODEL_ORDER:
                g = oof[(oof.setting == s) & (oof.model == m)]
                ax.plot(thr, net_benefit(g.label.to_numpy(), g.prob.to_numpy(), thr), color=COLORS[m],
                        lw=3.5 if m == best else 1.5, alpha=1.0 if m == best else 0.75,
                        label=f"{m} (highest AUC)" if m == best else m, zorder=3 if m == best else 2)
            y = oof[(oof.setting == s) & (oof.model == best)].label.to_numpy()
            prev = y.mean()
            ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), ls="--", color="black", lw=2, label="Treat All")
            ax.plot(thr, np.zeros_like(thr), color="gray", lw=2, label="Treat None")
            ref = oof[(oof.setting == "reference") & (oof.model == REF_MODEL)]
            ax.plot(thr, net_benefit(ref.label.to_numpy(), ref.prob.to_numpy(), thr), color="black", ls="-.", lw=2,
                    label="Conventional model")
            ax.set_xlim(0, 0.8); ax.set_ylim(-0.1, 0.75)
            ax.set_xlabel("Threshold Probability", **LABEL); ax.set_ylabel("Net Benefit", **LABEL)
            ax.set_title("Decision Curve Analysis", **TITLE)
            ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white", edgecolor="gainsboro")
            ax.grid(True, linestyle=":", alpha=0.5)
    finalize(fig, axes, path)


def fig_calibration(oof, perf, path, bins=10):
    fig, axes = frame()
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            ax, s = axes[i, j], GRID[(r, c)]
            best = best_model(perf, s)
            ax.plot([0, 1], [0, 1], ls="--", color="black", lw=2, label="Perfect calibration")
            for m in MODEL_ORDER:
                g = oof[(oof.setting == s) & (oof.model == m)]
                q = pd.qcut(g.prob.rank(method="first"), bins, labels=False)
                cal = g.groupby(q).agg(p=("prob", "mean"), o=("label", "mean"))
                row = perf[(perf.setting == s) & (perf.model == m)].iloc[0]
                isb = m == best
                ax.plot(cal.p, cal.o, color=COLORS[m], lw=3.5 if isb else 1.5, alpha=1 if isb else 0.65,
                        marker="s" if isb else "o", ms=8 if isb else 4,
                        label=f"{m} ({'highest AUC, ' if isb else ''}Brier = {row.Brier:.4f})", zorder=3 if isb else 2)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean Predicted Probability", **LABEL)
            ax.set_ylabel("Observed Event Frequency (Actual)", **LABEL)
            ax.set_title("Calibration Curves (Quantile Binning)", **TITLE)
            ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="gainsboro")
            ax.grid(True, linestyle=":", alpha=0.5)
    finalize(fig, axes, path)


def fig_confusion(perf, path):
    fig, axes = frame(figsize=(14, 13))
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            ax, s = axes[i, j], GRID[(r, c)]
            row = perf[perf.setting == s].sort_values("AUC", ascending=False).iloc[0]
            cm = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
            ax.grid(False)
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
            for (a, b), v in np.ndenumerate(cm):
                ax.text(b, a, str(v), ha="center", va="center", fontsize=16,
                        color="white" if v > cm.max() * 0.5 else "black")
            ax.set_xticks([0, 1], ["Pred_0", "Pred_1"], fontsize=12)
            ax.set_yticks([0, 1], ["Actual_0", "Actual_1"], fontsize=12)
            ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
            ax.set_xlabel("Prediction", fontsize=13, labelpad=8); ax.set_ylabel("Actual", fontsize=13)
            mod = "PET" if r == "PET Only" else "PET/CT"
            ax.set_title(f"Confusion Matrix, {mod} (Bin{c})\nHighest-AUC classifier: {row.model}", fontsize=14, pad=34)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finalize(fig, axes, path)


def fig_pairwise(comp, perf, s, label, path):
    fam = comp[comp.family == f"within {label}"]
    order = MODEL_ORDER
    n = len(order)
    P = np.full((n, n), np.nan)
    D = np.full((n, n), np.nan)
    for _, r in fam.iterrows():
        a, b = r.A.split(" | ")[1], r.B.split(" | ")[1]
        i, j = order.index(a), order.index(b)
        P[i, j] = P[j, i] = r.P_Holm
        D[i, j], D[j, i] = r.Delta_AUC, -r.Delta_AUC
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 9.5))
    for ax in (a1, a2):
        ax.grid(False)
    L = -np.log10(np.clip(P, 1e-12, 1))
    im1 = a1.imshow(L, cmap="YlOrRd", vmin=0, vmax=max(2.0, np.nanmax(L)))
    lim = np.nanmax(np.abs(D))
    im2 = a2.imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a1.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white" if L[i, j] > 1.2 else "black")
            a2.text(j, i, f"{D[i, j]:+.3f}" + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white" if abs(D[i, j]) > 0.6 * lim else "black")
    for ax in (a1, a2):
        ax.set_xticks(range(n), order, rotation=40, ha="right", fontsize=11)
        ax.set_yticks(range(n), order, fontsize=11)
    a1.set_title(f"{label}: Holm-adjusted P (patient-cluster bootstrap)\n$-\\log_{{10}}(P)$, higher = more significant",
                 fontsize=13, fontweight="bold")
    a2.set_title("ΔAUC (row − column)\nRed = row better, Blue = column better; * Holm P < 0.05",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04).set_label("$-\\log_{10}(P_{Holm})$", fontsize=11)
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04).set_label("ΔAUC", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def fig_tsne(data_dir, path_patient, path_class):
    emb = {}
    for s, fname in DATA_FILES.items():
        d = pd.read_csv(data_dir / fname)
        feats = [c for c in d.columns if c.startswith(("PET_", "CT_"))]
        X = d[feats].to_numpy(float)
        X = X[:, X.std(0) > 1e-12]
        Z = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(StandardScaler().fit_transform(X))
        emb[s] = (Z, d.patient_id.to_numpy(), d.label.to_numpy())
    tab20 = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors[:4]
    for path, mode in [(path_patient, "patient"), (path_class, "class")]:
        fig, axes = frame()
        for i, r in enumerate(ROWS):
            for j, c in enumerate(COLS):
                ax, s = axes[i, j], GRID[(r, c)]
                Z, pid, lab = emb[s]
                if mode == "patient":
                    for k, p in enumerate(sorted(np.unique(pid))):
                        m = pid == p
                        ax.scatter(Z[m, 0], Z[m, 1], s=28, color=tab20[k], edgecolor="k", linewidth=0.3, alpha=0.8,
                                   label=f"P{p}")
                    ax.legend(title="Patient ID", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1,
                              markerscale=1.2, frameon=True)
                    ax.set_title("t-SNE Visualization: Region Clustering by Patient", fontsize=13)
                else:
                    for k, (cl, name, col) in enumerate([(0, "Class 0 (normal-organ region)", "#FF7043"),
                                                         (1, "Class 1 (tumor-lesion region)", "#00ACC1")]):
                        m = lab == cl
                        ax.scatter(Z[m, 0], Z[m, 1], s=28, color=col, alpha=0.7, label=name)
                    ax.legend(loc="best", fontsize=9, frameon=True)
                    ax.set_title("t-SNE Visualization: Separation by Class", fontsize=13)
                ax.set_xlabel("t-SNE Dimension 1", fontsize=11); ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        finalize(fig, axes, path)


def short(f):
    s = f.split("(IBSI")[0]
    for a, b in [("INTENSITY-BASED_", "IB_"), ("INTENSITY-HISTOGRAM_", "IH_"), ("MORPHOLOGICAL_", "Morph_")]:
        s = s.replace(a, b)
    mod = "CT" if s.startswith("CT_") else "PET"
    return s.replace("PET_", "").replace("CT_", "") + f"_{mod}"


def fig_shap(root, perf, path, top_k=6):
    trees = ["Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM"]
    fig, axes = frame(figsize=(20, 16))
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            ax, s = axes[i, j], GRID[(r, c)]
            sub = perf[(perf.setting == s) & perf.model.isin(trees)].sort_values("AUC", ascending=False)
            model = sub.model.iloc[0]
            kappa = sub.Kappa.iloc[0]
            store = [x for x in pickle.load(open(root / s / "shap_values.pkl", "rb")) if x["model"] == model]
            vals = {}
            for rec in store:
                for k, f in enumerate(rec["features"]):
                    vals.setdefault(f, {"shap": [], "x": [], "folds": 0})
                    vals[f]["shap"].append(rec["shap"][:, k])
                    vals[f]["x"].append(rec["X"][:, k])
                    vals[f]["folds"] += 1
            imp = sorted(vals, key=lambda f: -np.mean(np.abs(np.concatenate(vals[f]["shap"]))))[:top_k]
            rng = np.random.default_rng(0)
            for y, f in enumerate(reversed(imp)):
                sv = np.concatenate(vals[f]["shap"])
                xv = np.concatenate(vals[f]["x"])
                lo, hi = np.percentile(xv, [5, 95])
                col = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
                ax.scatter(sv, y + rng.normal(0, 0.08, len(sv)), c=col, cmap=SHAP_CMAP, s=10, alpha=0.8, linewidths=0)
            ax.axvline(0, color="#999999", lw=1)
            ax.set_yticks(range(len(imp)), [f"{short(f)} ({vals[f]['folds']}/4)" for f in reversed(imp)], fontsize=10)
            ax.grid(axis="x", linestyle=":", alpha=0.4); ax.grid(axis="y", visible=False)
            scale = "probability" if model in ("Random Forest", "Extra Trees") else "log-odds"
            ax.set_xlabel(f"SHAP Value (Impact on Model Output, {scale})", fontsize=10, fontweight="bold")
            ax.set_title(f"SHAP Summary Plot (held-out outer-test regions)\nModel: {model} (highest pooled AUC) | κ = {kappa:.3f}",
                         fontsize=11.5, fontweight="bold")
            sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP)
            cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
            cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=10)
    finalize(fig, axes, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../outputs")
    ap.add_argument("--data", default="../data_rebuilt")
    a = ap.parse_args()
    root = Path(a.out)
    fd = root / "figures_manuscript"
    fd.mkdir(exist_ok=True)
    oof = pd.read_csv(root / "summary" / "all_oof_predictions.csv")
    perf = pd.read_csv(root / "summary" / "performance_summary.csv")
    comp = pd.read_csv(root / "summary" / "paired_comparisons.csv")

    fig_roc(oof, perf, fd / "Figure2_ROC.png")
    fig_dca(oof, perf, fd / "Figure3_DCA.png")
    labels = {"pet64": "PET-only bin64", "pet128": "PET-only bin128", "petct64": "PET/CT bin64", "petct128": "PET/CT bin128"}
    for s, lab in labels.items():
        fig_pairwise(comp, perf, s, lab, fd / f"Figure4_pairwise_{s}.png")
    fig_confusion(perf, fd / "Figure5_confusion.png")
    fig_calibration(oof, perf, fd / "Figure6_calibration.png")
    fig_tsne(Path(a.data), fd / "Figure7_tSNE_patient.png", fd / "Figure8_tSNE_class.png")
    # Figure 9 (SHAP) is produced by final_model_shap.py
    nc = {("PET Only", "64"): "pet64_nocombat", ("PET Only", "128"): "pet128_nocombat",
          ("PET/CT", "64"): "petct64_nocombat", ("PET/CT", "128"): "petct128_nocombat"}
    fig_roc(oof, perf, fd / "FigureS_ROC_noComBat.png", grid=nc)
    sens = {("Sensitivity", "CT-only (727)"): "ct", ("Sensitivity", "PET-only bin64 (734)"): "pet64_all",
            ("Sensitivity", "PET-only bin128 (734)"): "pet128_all"}
    fig_roc(oof, perf, fd / "FigureS_ROC_sensitivity.png", grid=sens, rows=["Sensitivity"],
            cols=["CT-only (727)", "PET-only bin64 (734)", "PET-only bin128 (734)"])
    print("written to", fd)


if __name__ == "__main__":
    main()
