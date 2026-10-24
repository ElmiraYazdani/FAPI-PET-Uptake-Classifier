"""
Publication figures regenerated from the verified out-of-fold prediction files.

Each multi-panel figure highlights, per setting, (1) the classifier with the
highest pooled ROC-AUC (exploratory selection), (2) the combined conventional
reference model (SUVmax + SUVmean + volume + TLF) and (3) the volume-only
reference model; the remaining nine classifiers are drawn as thin grey lines.

Usage:
    python figures.py --out ..

Adapted from the previous analysis (Revision_Analysis/code/figures.py): experiment names and folders; figures are
saved as PNG (600 dpi) and PDF; the CT-only / 734-region sensitivity ROC figure is not produced.
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
from sklearn.metrics import roc_curve

from run_experiment import EXPERIMENTS
from summarize import LABELS, MAIN

FOLDER = {k: v[1] for k, v in EXPERIMENTS.items()}

C_TOP, C_REF, C_VOL = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED = "#b5b4af", "#0b0b0b", "#52514e"
REF_ALL, REF_VOL = "Ref: SUVmax+SUVmean+Volume+TLF", "Ref: Volume"
PANEL = ["A", "B", "C", "D", "E", "F"]

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def _grid(ax):
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def _series(oof, perf, s):
    top = perf[perf.setting == s].sort_values("AUC", ascending=False).iloc[0]
    others = [m for m in perf[perf.setting == s].model if m != top.model]
    return top, others


def roc_panel(oof, perf, settings, path):
    fig, axes = plt.subplots(1, len(settings), figsize=(4.0 * len(settings), 4.3), sharey=True)
    ref = oof[oof.setting == "reference"]
    for i, (ax, s) in enumerate(zip(np.atleast_1d(axes), settings)):
        top, others = _series(oof, perf, s)
        for m in others:
            g = oof[(oof.setting == s) & (oof.model == m)]
            f, t, _ = roc_curve(g.label, g.prob)
            ax.plot(f, t, color=GREY, lw=0.8, zorder=1)
        for g, col, lab in [(ref[ref.model == REF_VOL], C_VOL, "Volume only"),
                            (ref[ref.model == REF_ALL], C_REF, "Conventional combined"),
                            (oof[(oof.setting == s) & (oof.model == top.model)], C_TOP, top.model)]:
            f, t, _ = roc_curve(g.label, g.prob)
            auc = perf[(perf.setting == g.setting.iat[0]) & (perf.model == g.model.iat[0])].iloc[0]
            ax.plot(f, t, color=col, lw=2, zorder=3,
                    label=f"{lab}: {auc.AUC:.3f} ({auc.AUC_CI_low:.3f}-{auc.AUC_CI_high:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
        ax.plot([], [], color=GREY, lw=0.8, label="Other classifiers")
        _grid(ax)
        ax.set_title(f"{PANEL[i]}  {LABELS[s]}", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("1 - specificity")
        if i == 0:
            ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", fontsize=7.5, frameon=False, title="Pooled AUC (95% patient-cluster CI)",
                  title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def calibration_panel(oof, perf, settings, path, bins=10):
    fig, axes = plt.subplots(1, len(settings), figsize=(4.0 * len(settings), 4.3), sharey=True)
    ref = oof[oof.setting == "reference"]
    for i, (ax, s) in enumerate(zip(np.atleast_1d(axes), settings)):
        top, _ = _series(oof, perf, s)
        for g, col, lab in [(ref[ref.model == REF_ALL], C_REF, "Conventional combined"),
                            (oof[(oof.setting == s) & (oof.model == top.model)], C_TOP, top.model)]:
            q = pd.qcut(g.prob, bins, duplicates="drop")
            cal = g.groupby(q, observed=True).agg(p=("prob", "mean"), o=("label", "mean"))
            r = perf[(perf.setting == g.setting.iat[0]) & (perf.model == g.model.iat[0])].iloc[0]
            ax.plot(cal.p, cal.o, color=col, lw=2, marker="o", ms=5, mec="white", mew=1,
                    label=f"{lab}: Brier {r.Brier:.3f}, intercept {r.Cal_intercept:.2f}, slope {r.Cal_slope:.2f}")
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8, label="Ideal")
        _grid(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{PANEL[i]}  {LABELS[s]}", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Mean predicted probability (decile)")
        if i == 0:
            ax.set_ylabel("Observed proportion of Class 1")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def net_benefit(y, p, thr):
    n = len(y)
    out = []
    for t in thr:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def dca_panel(oof, perf, settings, path):
    thr = np.linspace(0.05, 0.95, 91)
    fig, axes = plt.subplots(1, len(settings), figsize=(4.0 * len(settings), 4.3), sharey=True)
    ref = oof[oof.setting == "reference"]
    for i, (ax, s) in enumerate(zip(np.atleast_1d(axes), settings)):
        top, _ = _series(oof, perf, s)
        g0 = oof[(oof.setting == s) & (oof.model == top.model)]
        prev = g0.label.mean()
        ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), color=MUTED, lw=1, ls="--", label="Classify all as Class 1")
        ax.axhline(0, color=INK, lw=0.8, label="Classify none")
        for g, col, lab in [(ref[ref.model == REF_VOL], C_VOL, "Volume only"),
                            (ref[ref.model == REF_ALL], C_REF, "Conventional combined"),
                            (g0, C_TOP, top.model)]:
            ax.plot(thr, net_benefit(g.label.to_numpy(), g.prob.to_numpy(), thr), color=col, lw=2, label=lab)
        _grid(ax)
        ax.set_ylim(-0.1, max(0.7, prev + 0.05))
        ax.set_title(f"{PANEL[i]}  {LABELS[s]}", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Threshold probability")
        if i == 0:
            ax.set_ylabel("Net benefit")
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def confusion_panel(perf, settings, path):
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#2a78d6"])
    fig, axes = plt.subplots(1, len(settings), figsize=(3.3 * len(settings), 3.4))
    for i, (ax, s) in enumerate(zip(np.atleast_1d(axes), settings)):
        r = perf[perf.setting == s].sort_values("AUC", ascending=False).iloc[0]
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], dtype=int)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{v}", ha="center", va="center", fontsize=12,
                    color="white" if v > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"]); ax.set_yticks([0, 1], ["Class 0", "Class 1"])
        ax.spines[:].set_visible(False)
        ax.set_title(f"{PANEL[i]}  {LABELS[s]}\n{r.model} (n0={r.n_neg}, n1={r.n_pos})", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def pairwise_heatmap(comp, perf, s, path):
    fam = comp[comp.family == f"within {LABELS[s]}"].copy()
    order = perf[perf.setting == s].sort_values("AUC", ascending=False).model.tolist()
    n = len(order)
    mat = np.full((n, n), np.nan)
    sig = np.zeros((n, n), dtype=bool)
    for _, r in fam.iterrows():
        a, b = r.A.split(" | ")[1], r.B.split(" | ")[1]
        i, j = order.index(a), order.index(b)
        mat[i, j], mat[j, i] = r.Delta_AUC, -r.Delta_AUC
        sig[i, j] = sig[j, i] = r.P_Holm < 0.05
    lim = np.nanmax(np.abs(mat))
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4f3ef", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{mat[i, j]:+.2f}" + ("*" if sig[i, j] else ""), ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(n), order, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n), order, fontsize=10)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("AUC(row) - AUC(column)", fontsize=10)
    ax.set_title(f"{LABELS[s]}: paired AUC differences\n* Holm-adjusted patient-cluster bootstrap P < 0.05 (45 comparisons)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def shap_panel(root, perf, settings, path, top_k=8):
    fig, axes = plt.subplots(len(settings), 1, figsize=(8.0, 2.6 * len(settings)))
    tables = []
    for i, (ax, s) in enumerate(zip(np.atleast_1d(axes), settings)):
        tree = perf[(perf.setting == s) & perf.model.isin(["Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM"])]
        model = tree.sort_values("AUC", ascending=False).model.iloc[0]
        store = pickle.load(open(root / FOLDER[s] / "shap_values.pkl", "rb"))
        rows = []
        for rec in [r for r in store if r["model"] == model]:
            for j, f in enumerate(rec["features"]):
                rows.append({"feature": f, "fold": rec["outer_fold"], "mean_abs_shap": np.abs(rec["shap"][:, j]).mean()})
        t = pd.DataFrame(rows).groupby("feature").agg(mean_abs_shap=("mean_abs_shap", "mean"), n_folds=("fold", "nunique"))
        t = t.sort_values("mean_abs_shap", ascending=False)
        tables.append(t.assign(setting=s, model=model).reset_index())
        t = t.head(top_k)[::-1]
        ax.barh(range(len(t)), t.mean_abs_shap, color=C_TOP, height=0.6)
        ax.set_yticks(range(len(t)), [f"{_short(f)} ({n}/4)" for f, n in zip(t.index, t.n_folds)], fontsize=8.5)
        _grid(ax)
        ax.set_title(f"{PANEL[i]}  {LABELS[s]} - {model}", loc="left", fontsize=10.5, color=INK)
        scale = "probability scale" if model in ("Random Forest", "Extra Trees") else "log-odds scale"
        ax.set_xlabel(f"Mean |SHAP value| on held-out test regions ({scale})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)
    return pd.concat(tables)


def _short(f):
    s = f.split("(IBSI")[0]
    for a, b in [("IB_", "IB "), ("IH_", "IH "), ("Mo_", "Shape "), ("_", " ")]:
        s = s.replace(a, b)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    a = ap.parse_args()
    root = Path(a.out)
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    oof = pd.read_csv(root / "summary" / "all_oof_predictions.csv")
    perf = pd.read_csv(root / "summary" / "performance_summary.csv")
    comp = pd.read_csv(root / "summary" / "paired_comparisons.csv")
    main4 = MAIN
    roc_panel(oof, perf, main4, fig_dir / "Fig_ROC_four_settings.png")
    calibration_panel(oof, perf, main4, fig_dir / "Fig_calibration_four_settings.png")
    dca_panel(oof, perf, main4, fig_dir / "Fig_DCA_four_settings.png")
    confusion_panel(perf, main4, fig_dir / "Fig_confusion_matrices.png")
    for s in main4:
        pairwise_heatmap(comp, perf, s, fig_dir / f"Fig_pairwise_{s}.png")
    shap_tab = shap_panel(root, perf, main4, fig_dir / "Fig_SHAP_four_settings.png")
    shap_tab.to_csv(root / "summary" / "shap_feature_importance.csv", index=False)
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
