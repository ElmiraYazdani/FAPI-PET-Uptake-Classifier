"""
Manuscript-format tables from the new results (same content and layout as the previous manuscript tables).

    Table 2   pooled out-of-fold ROC-AUC (95% CI) of the ten classifiers in the four experiments (bold: highest)
    Table 3   highest-ROC-AUC classifier of each experiment and of CT-only, and the conventional reference models
    Tables S1-S13 and Figures S1-S5 of Additional file 1 (as built previously by manuscript_tools/build_supplement.py,
    in the order of the revised manuscript; Table S13 is new)
    Additional file 2 (out-of-fold predictions) and Additional file 3 (paired comparisons)

Adapted from Revision_Analysis/manuscript_tools/ed03_results.py (Tables 2-3) and build_supplement.py (S1-S12).
Every number and every count in the captions is read from the results; nothing is typed by hand. The documents are
written to <out>/tables_manuscript only - no manuscript or submission file is modified. Class names follow the
updated target (0 = physiological uptake, 1 = pathological uptake).

Usage:
    python tables_manuscript.py --out ..
"""
import argparse
import ast
from pathlib import Path

import docx
import pandas as pd
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Cm, Pt
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

TURQ, RED = WD_COLOR_INDEX.TURQUOISE, WD_COLOR_INDEX.RED
MAIN = ["exp1_pet64", "exp2_pet128", "exp3_pet64_ct", "exp4_pet128_ct"]
NOCOMBAT = [s + "_nocombat" for s in MAIN]
FOLDER = {"exp1_pet64": "Exp1_PET64", "exp2_pet128": "Exp2_PET128", "exp3_pet64_ct": "Exp3_PET64_CT",
          "exp4_pet128_ct": "Exp4_PET128_CT", "exp5_ct": "Sens_CT_only"}
LABEL = {"exp1_pet64": "PET-only bin64", "exp2_pet128": "PET-only bin128", "exp3_pet64_ct": "PET/CT bin64",
         "exp4_pet128_ct": "PET/CT bin128", "exp5_ct": "CT-only",
         "exp1_pet64_nocombat": "PET-only bin64, no ComBat", "exp2_pet128_nocombat": "PET-only bin128, no ComBat",
         "exp3_pet64_ct_nocombat": "PET/CT bin64, no ComBat", "exp4_pet128_ct_nocombat": "PET/CT bin128, no ComBat"}
SUMLAB = {"Exp 1: PET-64": "PET-only bin64", "Exp 2: PET-128": "PET-only bin128", "Exp 3: PET-64 + CT": "PET/CT bin64",
          "Exp 4: PET-128 + CT": "PET/CT bin128", "CT-only": "CT-only", "Conventional": "Conventional"}
MODEL_ORDER = ["Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM", "KNN",
               "RBF-SVM", "Gaussian Naive Bayes", "Linear SVM", "Logistic Regression"]
REF_ALL, REF_VOL = "Ref: SUVmax+SUVmean+Volume+TLF", "Ref: Volume"
REF_NAMES = {REF_ALL: "Conventional: SUVmax + SUVmean + volume + TLF", REF_VOL: "Conventional: volume",
             "Ref: TLF": "Conventional: TLF", "Ref: SUVmean": "Conventional: SUVmean",
             "Ref: SUVmax": "Conventional: SUVmax"}
TEMPLATE = Path(r"C:\Users\Elmira\Documents\0Manuscripts-Research under review\Moradi-FAPI\Submission\BMC MI"
                r"\Supplementary File.docx")


def f3(x):
    return f"{x:.3f}"


def ci(r, k="AUC"):
    return f"{r[k]:.3f} ({r[k + '_CI_low']:.3f}–{r[k + '_CI_high']:.3f})"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def lab(x):
    """Label used in the comparison files ('Exp 1: PET-64 | XGBoost') -> manuscript label."""
    return SUMLAB.get(x, x)


def new_doc(landscape=True):
    doc = docx.Document(TEMPLATE) if TEMPLATE.exists() else docx.Document()  # template is only read
    body = doc.element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"):
            body.remove(el)
    sec = doc.sections[0]
    if landscape and sec.page_width < sec.page_height:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(1.8))
    return doc


def add_par(doc, text, bold=False, size=None, space_after=6, highlight=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold or None
    if size:
        r.font.size = Pt(size)
    if highlight is not None:
        r.font.highlight_color = highlight
    p.paragraph_format.space_after = Pt(space_after)
    return p


def add_table(doc, header, rows, widths, size=8, bold_cells=()):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        borders.append(e)
    t._tbl.tblPr.append(borders)
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths[j])
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            r.bold = True if (i == 0 or (i, j) in bold_cells) else None
    doc.add_paragraph()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="..")
    a = ap.parse_args()
    root = Path(a.out)
    summ = root / "summary"
    dst = root / "tables_manuscript"
    dst.mkdir(exist_ok=True)
    perf = pd.read_csv(summ / "performance_summary.csv")
    comp = pd.read_csv(summ / "paired_comparisons.csv")
    shap_final = pd.read_csv(summ / "final_model_shap_importance.csv")
    oof = pd.read_csv(summ / "all_oof_predictions.csv")
    vo = pd.read_csv(summ / "volume_overlap_subset.csv")
    g0 = perf[perf.setting == MAIN[0]].iloc[0]
    N, N0, N1 = int(g0.n_regions), int(g0.n_neg), int(g0.n_pos)
    NP = oof[oof.setting == MAIN[0]].patient_id.nunique()
    TOP = {s: perf[perf.setting == s].sort_values("AUC", ascending=False).model.iloc[0] for s in MAIN + ["exp5_ct"]}
    A = lambda s, m: perf[(perf.setting == s) & (perf.model == m)].iloc[0]
    cohort = (f"{N:,} foci from {NP} patients; {N0} physiological-uptake (Class 0) and {N1} pathological-uptake "
              f"(Class 1) foci in every setting")
    xls = {}

    # ------------------------------------------------------------ Tables 2 and 3
    doc = new_doc()
    hdr = ["Classifier"] + [LABEL[s] for s in MAIN]
    rows, bold = [], []
    for i, m in enumerate(MODEL_ORDER, start=1):
        rows.append([m] + [ci(A(s, m)) for s in MAIN])
        bold += [(i, j) for j, s in enumerate(MAIN, start=1) if TOP[s] == m]
    add_par(doc, f"Table 2. Pooled out-of-fold ROC-AUC (95% patient-cluster bootstrap CI) of the ten classifiers in "
                 f"the four experimental settings ({cohort}). The highest value in each setting is shown in bold. "
                 f"bin64/bin128, PET discretization with 64/128 gray levels.", bold=True, size=10)
    add_table(doc, hdr, rows, [3.4, 3.2, 3.2, 3.2, 3.2], size=9, bold_cells=bold)
    xls["Table2"] = pd.DataFrame(rows, columns=hdr)

    hdr3 = ["Configuration – model", "ROC-AUC (95% CI)", "PR-AUC", "Balanced accuracy", "Sensitivity", "Specificity",
            "F1-score", "κ", "Brier score"]
    spec = [(s, TOP[s], f"{LABEL[s]} – {TOP[s]}") for s in MAIN + ["exp5_ct"]] + \
           [("reference", m, REF_NAMES[m]) for m in [REF_ALL, REF_VOL, "Ref: TLF", "Ref: SUVmean", "Ref: SUVmax"]]
    rows = []
    for s, m, name in spec:
        r = A(s, m)
        rows.append([name, ci(r), f3(r.PR_AUC), f3(r.Balanced_accuracy), f3(r.Sensitivity), f3(r.Specificity),
                     f3(r.F1), f3(r.Kappa), f3(r.Brier)])
    add_par(doc, f"Table 3. Classification performance of the highest-ROC-AUC classifier of each setting (exploratory "
                 f"selection) and of the conventional reference models (logistic regression), computed from pooled "
                 f"out-of-fold predictions of the same {N:,} foci. Threshold-dependent metrics use a fixed threshold "
                 f"of 0.5. κ, Cohen's kappa between predicted and reference classes; TLF, total lesion FAPI.",
            bold=True, size=10)
    add_table(doc, hdr3, rows, [5.2, 3.2, 1.6, 2.2, 1.9, 1.9, 1.6, 1.4, 1.6], size=9)
    xls["Table3"] = pd.DataFrame(rows, columns=hdr3)
    doc.save(dst / "Tables_2_3.docx")

    # ------------------------------------------------------------ Additional file 1
    doc = new_doc()
    add_par(doc, "Additional file 1", bold=True, size=14)
    add_par(doc, "Radiomics-Based Machine Learning for Classification of Physiological and Pathological Uptake on "
                 "[68Ga]Ga-FAPI-46 PET/CT: supplementary tables and figures", bold=True, size=11)
    add_par(doc, f"All values were computed from the out-of-fold prediction file (Additional file 2; {cohort}). "
                 "ROC-AUC, PR-AUC, balanced accuracy, κ and Brier score are pooled out-of-fold estimates; 95% "
                 "confidence intervals (CIs) are percentile intervals from 2,000 patient-cluster bootstrap resamples. "
                 "Threshold-dependent metrics use a fixed threshold of 0.5. bin64/bin128, PET discretization with "
                 "64/128 gray levels; PPV, positive predictive value; TLF, total lesion FAPI.", size=9, highlight=TURQ)
    hdr = ["Classifier", "ROC-AUC (95% CI)", "Fold ROC-AUC, mean ± SD", "PR-AUC", "Accuracy", "Balanced accuracy",
           "Sensitivity", "Specificity", "PPV", "F1-score", "κ (95% CI)", "Brier score"]
    w = [3.4, 3.1, 2.6, 1.4, 1.5, 1.7, 1.6, 1.6, 1.3, 1.4, 3.0, 1.5]
    for k, s in enumerate(MAIN, start=1):
        g = perf[perf.setting == s].set_index("model")
        rows = [[m, ci(g.loc[m]), f"{g.loc[m].Fold_AUC_mean:.3f} ± {g.loc[m].Fold_AUC_SD:.3f}", f3(g.loc[m].PR_AUC),
                 f3(g.loc[m].Accuracy), f3(g.loc[m].Balanced_accuracy), f3(g.loc[m].Sensitivity),
                 f3(g.loc[m].Specificity), f3(g.loc[m].PPV), f3(g.loc[m].F1), ci(g.loc[m], "Kappa"), f3(g.loc[m].Brier)]
                for m in MODEL_ORDER]
        add_par(doc, f"Table S{k}. Performance of the ten classifiers in the {LABEL[s]} setting ({N:,} foci).",
                bold=True, size=10)
        add_table(doc, hdr, rows, w)
        xls[f"TableS{k}"] = pd.DataFrame(rows, columns=hdr)

    sens = ["exp5_ct"] + NOCOMBAT
    rows = [[m] + [ci(A(s, m)) for s in sens] for m in MODEL_ORDER]
    add_par(doc, f"Table S5. Sensitivity analyses: pooled out-of-fold ROC-AUC (95% CI) for CT-only radiomics ({N:,} "
                 "foci) and the four main settings repeated without ComBat harmonization. (The previous PET-only "
                 "analysis of regions without CT features does not apply: all foci have PET and CT features.)",
            bold=True, size=10)
    add_table(doc, ["Classifier"] + [LABEL[s] for s in sens], rows, [3.4] + [4.4] * 5)
    xls["TableS5"] = pd.DataFrame(rows, columns=["Classifier"] + [LABEL[s] for s in sens])

    rows = []
    for m in MODEL_ORDER:
        row = [m]
        for s in MAIN:
            r = A(s, m)
            icpt = f"{r.Cal_intercept:.2f}" if abs(r.Cal_intercept) < 100 else "n.e."
            row.append(f"{r.Brier:.3f} / {icpt} / {r.Cal_slope:.2f}")
        rows.append(row)
    add_par(doc, "Table S6. Calibration of the ten classifiers: Brier score / calibration intercept "
                 "(calibration-in-the-large) / calibration slope, from pooled out-of-fold predictions. Ideal values: "
                 "intercept 0 and slope 1. n.e., not estimable (degenerate probabilities).", bold=True, size=10)
    add_table(doc, ["Classifier"] + [LABEL[s] for s in MAIN], rows, [3.6, 5.2, 5.2, 5.2, 5.2])
    xls["TableS6"] = pd.DataFrame(rows, columns=["Classifier"] + [LABEL[s] for s in MAIN])

    fam = comp[comp.family == "radiomics (top per setting) vs conventional"]
    rows = []
    for _, r in fam.iterrows():
        a_, b_ = r.A.split(" | "), r.B.split(" | ")[1]
        rows.append([f"{lab(a_[0])}: {a_[1]}", REF_NAMES[b_], f3(r.AUC_A), f3(r.AUC_B),
                     f"{r.Delta_AUC:+.3f} ({r.CI_low:+.3f} to {r.CI_high:+.3f})", pfmt(r.P_boot), pfmt(r.P_Holm)])
    h = ["Radiomics model", "Reference model", "ROC-AUC radiomics", "ROC-AUC reference", "ΔAUC (95% CI)", "P",
         "Holm-adjusted P"]
    add_par(doc, f"Table S7. Highest-ROC-AUC radiomics classifier of each setting versus conventional quantitative "
                 f"reference models (logistic regression; same {N:,} foci and outer folds). ΔAUC and 95% CI from "
                 f"paired patient-cluster bootstrap; Holm adjustment across the {len(fam)} comparisons.",
            bold=True, size=10)
    add_table(doc, h, rows, [5.2, 5.6, 2.4, 2.4, 4.4, 1.8, 2.4])
    xls["TableS7"] = pd.DataFrame(rows, columns=h)

    fam = comp[comp.family == "configuration contrasts (same classifier)"]
    rows = []
    for _, r in fam.iterrows():
        rows.append([r.A.split(" | ")[1], f"{lab(r.A.split(' | ')[0])} vs {lab(r.B.split(' | ')[0])}", f3(r.AUC_A),
                     f3(r.AUC_B), f"{r.Delta_AUC:+.3f} ({r.CI_low:+.3f} to {r.CI_high:+.3f})", pfmt(r.P_Holm)])
    h = ["Classifier", "Contrast", "ROC-AUC first", "ROC-AUC second", "ΔAUC (95% CI)", "Holm-adjusted P"]
    add_par(doc, f"Table S8. Effect of imaging modality and gray-level discretization for each classifier (same "
                 f"{N:,} foci). ΔAUC = first minus second configuration; paired patient-cluster bootstrap; Holm "
                 f"adjustment across the {len(fam)} comparisons.", bold=True, size=10)
    add_table(doc, h, rows, [4.0, 7.0, 2.4, 2.4, 4.6, 2.6], size=7.5)
    xls["TableS8"] = pd.DataFrame(rows, columns=h)

    rows = []
    for s in MAIN + ["exp5_ct"]:
        f = pd.read_csv(root / FOLDER[s] / "selected_features.csv")
        fm = pd.read_csv(root / FOLDER[s] / "fold_metrics.csv").drop_duplicates("outer_fold").set_index("outer_fold")
        for _, r in f[f.inner_fold == 0].iterrows():
            before, after = ast.literal_eval(r.smote_before), ast.literal_eval(r.smote_after)
            feats = "; ".join(x.split("(IBSI")[0] for x in r.features.split("; "))
            rows.append([LABEL[s], int(r.outer_fold), fm.loc[r.outer_fold].test_patients.replace(",", ", "),
                         f"{fm.loc[r.outer_fold].n_test_neg}/{fm.loc[r.outer_fold].n_test_pos}",
                         f"{r.n_after_volume} → {r.n_after_redundancy} → {r.n_after_mrmr} → {r.n_features}",
                         feats, f"{before[0]}/{before[1]} → {after[0]}/{after[1]}"])
    h = ["Setting", "Outer fold", "Test patients", "Test foci (0/1)", "Features per stage", "Selected features",
         "Training foci (0/1) before → after SMOTE"]
    add_par(doc, "Table S9. Outer-fold composition, feature-selection stages and class counts before and after "
                 "SMOTE (applied to the outer-training partition only). Feature counts: after volume-dependence "
                 "filtering → redundancy reduction → mRMR → stability selection (final). Class counts are "
                 "Class 0/Class 1. Test partitions were never oversampled.", bold=True, size=10)
    add_table(doc, h, rows, [2.6, 1.2, 3.0, 1.8, 3.0, 9.0, 3.6], size=7)
    xls["TableS9"] = pd.DataFrame(rows, columns=h)

    rows = [[LABEL[r.setting], r.model, r.feature.split("(IBSI")[0], f3(r.mean_abs_shap),
             f"{int(r.outer_folds_selected)}/4"] for _, r in shap_final.iterrows()]
    h = ["Setting", "Model", "Feature", "Mean |SHAP|", "Selected in outer folds"]
    add_par(doc, f"Table S10. Features of the final models used for interpretation (Figure 9): mean absolute SHAP "
                 f"value over the {N:,} foci and number of outer folds in which the same feature was selected during "
                 f"nested cross-validation. SHAP values are on the log-odds scale for XGBoost, LightGBM and Gradient "
                 f"Boosting and on the probability scale for Random Forest and Extra Trees.", bold=True, size=10)
    add_table(doc, h, rows, [4.0, 3.2, 11.0, 2.6, 3.4])
    xls["TableS10"] = pd.DataFrame(rows, columns=h)

    rows = []
    for _, r in vo.iterrows():
        name = REF_NAMES[r.model] if r.setting == "reference" else f"{LABEL[r.setting]}: {r.model}"
        is_vol = r.model == REF_VOL
        rows.append([name, f"{r.AUC:.3f} ({r.AUC_CI_low:.3f}–{r.AUC_CI_high:.3f})",
                     "—" if is_vol else f"{r.Delta_vs_volume:+.3f} ({r.Delta_CI_low:+.3f} to {r.Delta_CI_high:+.3f})",
                     "—" if is_vol else pfmt(r.P_Holm),
                     f"{r.Delta_vs_conventional:+.3f} ({r.DeltaC_CI_low:+.3f} to {r.DeltaC_CI_high:+.3f})"
                     if r.setting != "reference" else "—",
                     pfmt(r.P_Holm_conv) if r.setting != "reference" else "—"])
    h = ["Model", "ROC-AUC (95% CI)", "ΔAUC vs volume (95% CI)", "Holm-adjusted P", "ΔAUC vs conventional (95% CI)",
         "Holm-adjusted P"]
    r0 = vo.iloc[0]
    add_par(doc, f"Table S11. Post hoc volume-overlap sensitivity analysis: the out-of-fold predictions were "
                 f"re-evaluated (without re-training) in the {int(r0.n_regions)} foci whose volume lay within the "
                 f"range shared by both classes ({r0.overlap_low_cm3:.1f}–{r0.overlap_high_cm3:.1f} cm³; "
                 f"{int(r0.n_neg)} physiological-uptake and {int(r0.n_pos)} pathological-uptake foci from "
                 f"{int(r0.n_patients)} patients). ΔAUC versus the volume-only and the combined conventional models "
                 f"from paired patient-cluster bootstrap with Holm adjustment.", bold=True, size=10)
    add_table(doc, h, rows, [6.4, 3.6, 4.4, 2.4, 4.4, 2.4])
    xls["TableS11"] = pd.DataFrame(rows, columns=h)

    rows = []
    for s in MAIN:
        for m in MODEL_ORDER:
            g = oof[(oof.setting == s) & (oof.model == m)]
            cells = []
            for k in sorted(g.outer_fold.unique()):
                hh = g[g.outer_fold == k]
                cells.append(f"{roc_auc_score(hh.label, hh.prob):.3f} / {accuracy_score(hh.label, hh.pred):.3f} / "
                             f"{cohen_kappa_score(hh.label, hh.pred):.3f}")
            rows.append([LABEL[s], m] + cells)
    fm = pd.read_csv(root / FOLDER[MAIN[0]] / "fold_metrics.csv").drop_duplicates("outer_fold").set_index("outer_fold")
    fold_hdr = [f"Fold {k} ({fm.loc[k].n_test_neg}/{fm.loc[k].n_test_pos})" for k in sorted(fm.index)]
    add_par(doc, "Table S12. Fold-level results: ROC-AUC / accuracy / Cohen's κ of each classifier in each outer test "
                 "fold, computed from the same out-of-fold predictions as all other results (Additional file 2). The "
                 "outer folds and their test foci (Class 0/Class 1, in parentheses) are identical in all settings; "
                 "threshold 0.5.", bold=True, size=10)
    add_table(doc, ["Setting", "Classifier"] + fold_hdr, rows, [3.0, 3.6, 4.6, 4.6, 4.6, 4.6], size=7)
    xls["TableS12"] = pd.DataFrame(rows, columns=["Setting", "Classifier"] + fold_hdr)

    fd = root / "figures_manuscript"
    figs = [
        ("Figure4_pairwise_exp1_pet64.png", "Figure S1. Pairwise comparison of the ten classifiers in the PET-only "
         "bin64 setting: difference in pooled out-of-fold ROC-AUC (row minus column); * Holm-adjusted P < 0.05 "
         "(paired patient-cluster bootstrap, 45 comparisons). All P values are listed in Additional file 3."),
        ("Figure4_pairwise_exp2_pet128.png", "Figure S2. Pairwise comparison of the ten classifiers in the PET-only "
         "bin128 setting."),
        ("Figure4_pairwise_exp4_pet128_ct.png", "Figure S3. Pairwise comparison of the ten classifiers in the PET/CT "
         "bin128 setting."),
        ("FigureS_ROC_noComBat.png", "Figure S4. ROC curves of the four main settings repeated without ComBat "
         "harmonization."),
        ("FigureS_ROC_sensitivity.png", f"Figure S5. ROC curves of the CT-only sensitivity analysis ({N:,} foci)."),
    ]
    # Table S13 (new): discrimination by type of physiological focus (post hoc; negative_subtype.py)
    ns = pd.read_csv(summ / "negative_subtype_auc.csv")
    rows = []
    for _, r in ns.iterrows():
        name = REF_NAMES[r.model] if r.setting == "reference" else f"{LABEL[r.setting]}: {r.model}"
        rows.append([name, "Organ regions" if r.negatives == "organ" else "Physiological focal-uptake foci",
                     f"{int(r.n_neg)}/{int(r.n_pos)}", f"{r.AUC:.3f} ({r.AUC_CI_low:.3f}–{r.AUC_CI_high:.3f})",
                     f3(r["specificity_at_0.5"])])
    h = ["Model", "Class 0 foci compared with all Class 1 foci", "Foci (0/1)", "ROC-AUC (95% CI)",
         "Specificity at 0.5"]
    add_par(doc, "Table S13. Post hoc analysis by type of physiological focus: the out-of-fold predictions were "
                 "re-evaluated (without re-training) separately for the organ regions and for the physiological "
                 "focal-uptake foci, each compared with all pathological foci. 95% CIs from 2,000 patient-cluster "
                 "bootstrap resamples.", bold=True, size=10)
    add_table(doc, h, rows, [6.4, 5.4, 2.2, 3.6, 2.6])
    xls["TableS13"] = pd.DataFrame(rows, columns=h)

    # Reorder the table blocks to the numbering used in the revised manuscript and renumber the captions
    order = {9: 1, 1: 2, 2: 3, 3: 4, 4: 5, 12: 6, 8: 7, 7: 8, 11: 9, 6: 10, 10: 11, 5: 12, 13: 13}
    body = doc.element.body
    caps = [pp for pp in doc.paragraphs if pp.text.startswith("Table S")]
    blocks = {}
    for cp in caps:
        k = int(cp.text.split(".")[0][7:])
        tbl = cp._p.getnext()
        spacer = tbl.getnext()
        blocks[k] = (cp, [cp._p, tbl, spacer])
    anchor = caps[0]._p.getprevious()
    for old_k, new_k in sorted(order.items(), key=lambda x: x[1]):
        cp, els = blocks[old_k]
        for r in cp.runs:
            r.text = r.text.replace(f"Table S{old_k}.", f"Table S{new_k}.", 1)
            r.font.highlight_color = RED if new_k == 6 else TURQ
        for el in els:
            anchor.addnext(el)
            anchor = el
    xls = {("TableS" + str(order[int(k[6:])]) if k.startswith("TableS") else k): v for k, v in xls.items()}
    for f, cap in figs:
        doc.add_paragraph().add_run().add_picture(str(fd / f), width=Cm(24 if "sensitivity" not in f else 14))
        add_par(doc, cap, size=10, highlight=TURQ)
    doc.save(dst / "Additional_file_1_Supplementary_Tables_Figures.docx")

    comp.rename(columns={"A": "model_A", "B": "model_B", "AUC_A": "roc_auc_A", "AUC_B": "roc_auc_B",
                         "Delta_AUC": "delta_auc_A_minus_B", "CI_low": "delta_ci_low", "CI_high": "delta_ci_high",
                         "P_boot": "p_bootstrap", "P_Holm": "p_holm", "family": "comparison_family"}) \
        .to_csv(dst / "Additional_file_3_paired_comparisons.csv", index=False)
    o = oof.copy()
    o["configuration"] = o.setting.map(lambda s: LABEL.get(s, "Conventional reference models"))
    o["model"] = o.model.map(lambda m: REF_NAMES.get(m, m))
    cols = ["configuration", "model", "patient_id", "region_id", "center", "label", "outer_fold", "prob", "pred"]
    o[cols].rename(columns={"region_id": "focus_id", "label": "reference_label", "prob": "oof_probability",
                            "pred": "predicted_class"}).to_csv(dst / "Additional_file_2_out_of_fold_predictions.csv",
                                                               index=False)
    try:
        with pd.ExcelWriter(dst / "Manuscript_tables.xlsx") as xw:
            for k, v in xls.items():
                v.to_excel(xw, sheet_name=k, index=False)
    except ImportError:
        pass
    print("written to", dst, "|", ", ".join(xls))


if __name__ == "__main__":
    main()
