"""
Print-size versions of the manuscript figures (Editor comment 13: readability).

Same template as figures_original_style.py (row headers PET Only / PET/CT, column
headers Gray Level 64 / 128, tab10 colour per classifier, thicker line for the
highest-AUC classifier), but sized for a 170-mm journal column: canvases of about
12 inches, fonts of 12-15 pt (>= 6.5 pt after reduction), one shared legend per
figure instead of twelve per-panel legends (AUC values are given in Table 2), and a
single-panel pairwise heatmap.

Usage:
    python figures_print.py --out .. --data ../01_data
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from figures_original_style import (CLASS_NAMES, COLORS, DATA_FILES, GRID, MODEL_ORDER, REF_MODEL, ROWS, COLS,
                                    best_model, net_benefit)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12.5, "axes.labelweight": "bold"})
HEADER = {"fontsize": 20, "fontweight": "bold", "family": "Times New Roman"}
W, H = 12.0, 11.0


def frame(grid=GRID, rows=ROWS, cols=COLS, size=(W, H)):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=size)
    axes = np.atleast_2d(axes)
    fig._hdr = (rows, cols)
    return fig, axes


def finalize(fig, axes, path, legend_handles=None, ncol=4, bottom=0.13):
    rows, cols = fig._hdr
    fig.tight_layout(rect=(0.06, bottom if legend_handles else 0, 1, 0.95))
    for j, c in enumerate(cols):
        b = axes[0, j].get_position()
        fig.text((b.x0 + b.x1) / 2, 0.978, f"Gray Level ({c})" if c in ("64", "128") else c,
                 ha="center", va="center", **HEADER)
    for i, r in enumerate(rows):
        b = axes[i, 0].get_position()
        fig.text(0.018, (b.y0 + b.y1) / 2, r, rotation=90, ha="center", va="center", **HEADER)
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.53, 0.0), columnspacing=1.2, handlelength=2.2)
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def model_handles(extra):
    hs = [Line2D([0], [0], color=COLORS[m], lw=2.5, label=m) for m in MODEL_ORDER]
    return hs + extra


def fig_roc(oof, perf, path, grid=GRID, rows=ROWS, cols=COLS, size=(W, H), ncol=4, bottom=0.13):
    fig, axes = frame(grid, rows, cols, size)
    ref = oof[(oof.setting == "reference") & (oof.model == REF_MODEL)]
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax, s = axes[i, j], grid[(r, c)]
            best = best_model(perf, s)
            for m in MODEL_ORDER:
                g = oof[(oof.setting == s) & (oof.model == m)]
                f, t, _ = roc_curve(g.label, g.prob)
                ax.plot(f, t, color=COLORS[m], lw=4.0 if m == best else 1.8, zorder=3 if m == best else 2)
            if ref.region_id.nunique() == oof[oof.setting == s].region_id.nunique():
                f, t, _ = roc_curve(ref.label, ref.prob)
                ax.plot(f, t, color="black", ls="-.", lw=2.2, zorder=4)
            ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.4)
            ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
            ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
            ax.text(0.97, 0.05, f"Thick line: {best}", transform=ax.transAxes, ha="right", fontsize=12,
                    bbox=dict(facecolor="white", edgecolor="gainsboro", boxstyle="round,pad=0.3"))
            ax.grid(True, linestyle=":", alpha=0.6)
    extra = [Line2D([0], [0], color="black", ls="-.", lw=2.2, label="Conventional model"),
             Line2D([0], [0], color="gray", ls="--", lw=1.4, label="Chance")]
    finalize(fig, axes, path, model_handles(extra), ncol=ncol, bottom=bottom)


def fig_dca(oof, perf, path):
    thr = np.linspace(0.01, 0.80, 80)
    fig, axes = frame()
    ref = oof[(oof.setting == "reference") & (oof.model == REF_MODEL)]
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            ax, s = axes[i, j], GRID[(r, c)]
            best = best_model(perf, s)
            for m in MODEL_ORDER:
                g = oof[(oof.setting == s) & (oof.model == m)]
                ax.plot(thr, net_benefit(g.label.to_numpy(), g.prob.to_numpy(), thr), color=COLORS[m],
                        lw=4.0 if m == best else 1.5, alpha=1 if m == best else 0.8, zorder=3 if m == best else 2)
            prev = oof[(oof.setting == s) & (oof.model == best)].label.mean()
            ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), ls="--", color="black", lw=2)
            ax.plot(thr, np.zeros_like(thr), color="gray", lw=2)
            ax.plot(thr, net_benefit(ref.label.to_numpy(), ref.prob.to_numpy(), thr), color="black", ls="-.", lw=2.2)
            ax.set_xlim(0, 0.8); ax.set_ylim(-0.1, 0.75)
            ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
            ax.text(0.03, 0.05, f"Thick line: {best}", transform=ax.transAxes, ha="left", fontsize=12,
                    bbox=dict(facecolor="white", edgecolor="gainsboro", boxstyle="round,pad=0.3"))
            ax.grid(True, linestyle=":", alpha=0.5)
    extra = [Line2D([0], [0], color="black", ls="-.", lw=2.2, label="Conventional model"),
             Line2D([0], [0], color="black", ls="--", lw=2, label="Treat all"),
             Line2D([0], [0], color="gray", lw=2, label="Treat none")]
    finalize(fig, axes, path, model_handles(extra))


def fig_calibration(oof, perf, path, bins=10):
    fig, axes = frame()
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            ax, s = axes[i, j], GRID[(r, c)]
            best = best_model(perf, s)
            ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.8)
            for m in MODEL_ORDER:
                g = oof[(oof.setting == s) & (oof.model == m)]
                q = pd.qcut(g.prob.rank(method="first"), bins, labels=False)
                cal = g.groupby(q).agg(p=("prob", "mean"), o=("label", "mean"))
                isb = m == best
                ax.plot(cal.p, cal.o, color=COLORS[m], lw=4.0 if isb else 1.4, alpha=1 if isb else 0.7,
                        marker="s" if isb else "o", ms=7 if isb else 3.5, zorder=3 if isb else 2)
            row = perf[(perf.setting == s) & (perf.model == best)].iloc[0]
            ax.text(0.97, 0.05, f"Thick line: {best}\nBrier {row.Brier:.3f}; intercept {row.Cal_intercept:.2f}; "
                                f"slope {row.Cal_slope:.2f}", transform=ax.transAxes, ha="right", fontsize=11.5,
                    bbox=dict(facecolor="white", edgecolor="gainsboro", boxstyle="round,pad=0.3"))
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed proportion of Class 1")
            ax.grid(True, linestyle=":", alpha=0.5)
    extra = [Line2D([0], [0], color="black", ls="--", lw=1.8, label="Perfect calibration")]
    finalize(fig, axes, path, model_handles(extra))


def fig_confusion(perf, path):
    fig, axes = frame(size=(10.5, 10))
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            ax, s = axes[i, j], GRID[(r, c)]
            row = perf[perf.setting == s].sort_values("AUC", ascending=False).iloc[0]
            cm = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
            ax.grid(False)
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
            for (a_, b_), v in np.ndenumerate(cm):
                ax.text(b_, a_, str(v), ha="center", va="center", fontsize=20,
                        color="white" if v > cm.max() * 0.5 else "black")
            ax.set_xticks([0, 1], ["Pred_0", "Pred_1"], fontsize=13)
            ax.set_yticks([0, 1], ["Actual_0", "Actual_1"], fontsize=13)
            ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
            ax.set_xlabel("Prediction", fontsize=14, labelpad=6); ax.set_ylabel("Actual", fontsize=14)
            ax.set_title(f"Highest-AUC classifier: {row.model}", fontsize=14, pad=30)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finalize(fig, axes, path)


def fig_pairwise(comp, perf, s, label, path):
    fam = comp[comp.family == f"within {label}"]
    order = MODEL_ORDER
    n = len(order)
    D = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    for _, r in fam.iterrows():
        a_, b_ = r.A.split(" | ")[1], r.B.split(" | ")[1]
        i, j = order.index(a_), order.index(b_)
        D[i, j], D[j, i] = r.Delta_AUC, -r.Delta_AUC
        P[i, j] = P[j, i] = r.P_Holm
    fig, ax = plt.subplots(figsize=(10, 8.6))
    ax.grid(False)
    lim = np.nanmax(np.abs(D))
    im = ax.imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{D[i, j]:+.2f}" + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white" if abs(D[i, j]) > 0.6 * lim else "black")
    labels = [m.replace("Gaussian Naive Bayes", "Gaussian NB").replace("Logistic Regression", "Logistic Reg.")
              for m in order]
    ax.set_xticks(range(n), labels, rotation=40, ha="right", fontsize=13)
    ax.set_yticks(range(n), labels, fontsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ΔAUC (row − column)", fontsize=13)
    ax.set_title(f"{label}: * Holm-adjusted P < 0.05 (paired patient-cluster bootstrap)", fontsize=13.5,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def fig_tsne(data_dir, path_patient, path_class):
    emb = {}
    for s, fname in DATA_FILES.items():
        d = pd.read_csv(data_dir / fname)
        X = d[[c for c in d.columns if c.startswith(("PET_", "CT_"))]].to_numpy(float)
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = X[:, X.std(0) > 1e-12]
        emb[s] = (TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
                  .fit_transform(StandardScaler().fit_transform(X)), d.patient_id.to_numpy(), d.label.to_numpy())
    tab = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors[:4]
    for path, mode in [(path_patient, "patient"), (path_class, "class")]:
        fig, axes = frame()
        for i, r in enumerate(ROWS):
            for j, c in enumerate(COLS):
                ax, s = axes[i, j], GRID[(r, c)]
                Z, pid, lab = emb[s]
                if mode == "patient":
                    for k, p in enumerate(sorted(np.unique(pid))):
                        m = pid == p
                        ax.scatter(Z[m, 0], Z[m, 1], s=22, color=tab[k], edgecolor="k", linewidth=0.25, alpha=0.85)
                else:
                    for cl, col in [(0, "#FF7043"), (1, "#00ACC1")]:
                        m = lab == cl
                        ax.scatter(Z[m, 0], Z[m, 1], s=22, color=col, alpha=0.75)
                ax.set_xlabel("t-SNE dimension 1"); ax.set_ylabel("t-SNE dimension 2")
        if mode == "patient":
            finalize(fig, axes, path)
        else:
            hs = [Line2D([0], [0], marker="o", ls="", color="#FF7043", ms=9, label=CLASS_NAMES[0]),
                  Line2D([0], [0], marker="o", ls="", color="#00ACC1", ms=9, label=CLASS_NAMES[1])]
            finalize(fig, axes, path, hs, ncol=2, bottom=0.06)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    ap.add_argument("--data", default="../01_data")
    a = ap.parse_args()
    root = Path(a.out)
    fd = root / "figures_manuscript"
    fd.mkdir(exist_ok=True)
    oof = pd.read_csv(root / "summary" / "all_oof_predictions.csv")
    perf = pd.read_csv(root / "summary" / "performance_summary.csv")
    comp = pd.read_csv(root / "summary" / "paired_comparisons.csv")
    fig_roc(oof, perf, fd / "Figure2_ROC.png")
    fig_dca(oof, perf, fd / "Figure3_DCA.png")
    from summarize import LABELS
    labels = {s: LABELS[s] for s in GRID.values()}
    for s, lab in labels.items():
        fig_pairwise(comp, perf, s, lab, fd / f"Figure4_pairwise_{s}.png")
    fig_confusion(perf, fd / "Figure5_confusion.png")
    fig_calibration(oof, perf, fd / "Figure6_calibration.png")
    fig_tsne(Path(a.data), fd / "Figure7_tSNE_patient.png", fd / "Figure8_tSNE_class.png")
    nc = {k: v + "_nocombat" for k, v in GRID.items()}
    fig_roc(oof, perf, fd / "FigureS_ROC_noComBat.png", grid=nc)
    sens = {("Sensitivity", "CT-only (1,020 foci)"): "exp5_ct"}
    fig_roc(oof, perf, fd / "FigureS_ROC_sensitivity.png", grid=sens, rows=["Sensitivity"],
            cols=["CT-only (1,020 foci)"], size=(8.5, 10.5), ncol=2, bottom=0.2)
    print("written to", fd)


if __name__ == "__main__":
    main()
