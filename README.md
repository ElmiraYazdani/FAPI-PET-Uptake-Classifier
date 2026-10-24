# 🧬 FAPI-PET-Uptake-Classifier

<p align="center">
  <b>Radiomics-Based Machine Learning for Classification of Physiological and Pathological Uptake on [<sup>68</sup>Ga]Ga-FAPI-46 PET/CT: A Multicenter Proof-of-Concept Study</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/Imaging-PET%2FCT-green.svg" alt="PET/CT" />
  <img src="https://img.shields.io/badge/Method-Radiomics-orange.svg" alt="Radiomics" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-purple.svg" alt="SHAP" />
  <img src="https://img.shields.io/badge/Validation-Nested%20CV-lightgrey.svg" alt="Nested cross-validation" />
</p>

---

## 📌 Overview

This repository contains the analysis pipeline for a multicenter proof-of-concept study investigating radiomics-based machine learning (ML) to distinguish physiological from pathological uptake on [⁶⁸Ga]Ga-FAPI-46 PET/CT in patients with cancer.

The framework combines PET and CT radiomic feature extraction, hierarchical feature selection, machine learning classification, multicenter harmonization, and SHapley Additive exPlanations (SHAP). Model development and evaluation use **leakage-controlled nested, patient-grouped cross-validation**.

Four principal feature configurations are evaluated: PET with 64 gray levels, PET with 128 gray levels, PET-64 combined with CT, and PET-128 combined with CT. The radiomics models are also compared with conventional quantitative PET reference models.

The project is intended for research and methodological development. The results are exploratory and do not establish clinical diagnostic utility.

## 🔬 Study Dataset

The study dataset comprised **1,020 segmented foci from 24 patients** examined at two centers.

| Class | Number of foci | Description |
|---|---:|---|
| Physiological uptake (Target = 0) | 487 | 240 regions from ten normal organs and 247 physiological focal-uptake foci |
| Physician-designated tumor lesions (Target = 1) | 533 | Lesions identified and delineated by nuclear medicine physicians |
| **Total** | **1,020** | Retrospective multicenter cohort |

Physiological focal-uptake foci were included because non-pathological uptake can resemble pathological lesions and represents a more challenging classification task than distinguishing normal-organ regions from lesions alone.

**Reference-standard limitation:** The pathological class consists of physician-designated tumor lesions. Lesion-level confirmation by histopathology, longitudinal imaging follow-up, or other imaging modalities was not systematically documented. Consequently, model performance reflects agreement with physician-assigned labels rather than independently validated diagnostic accuracy for malignancy.

Patient-level imaging data are not included in this repository.

## 🧪 Experimental Configurations

The four principal experiments differ in imaging modality and PET gray-level discretization.

| Experiment | Features | Script key |
|---|---|---|
| Exp 1 | PET, 64 gray levels | `exp1_pet64` |
| Exp 2 | PET, 128 gray levels | `exp2_pet128` |
| Exp 3 | PET-64 + CT | `exp3_pet64_ct` |
| Exp 4 | PET-128 + CT | `exp4_pet128_ct` |

PET features use absolute intensity discretization over a fixed SUVbw range of 0–40. CT features use a fixed intensity range of −1000 to 3000 HU and a bin width of 10 HU. Images are resampled to an isotropic voxel size of 1 × 1 × 1 mm³ before feature extraction.

Radiomic features are extracted using **LIFEx v25.06.1** and include shape descriptors, first-order intensity statistics, and higher-order texture features.

## ✨ Key Features

- **Ten ML classifiers:** Logistic regression, linear SVM, RBF-kernel SVM, random forest, extra trees, gradient boosting, XGBoost, LightGBM, k-nearest neighbors, and Gaussian naive Bayes.
- **Four-stage hierarchical feature selection:** Volume-dependence filtering, redundancy reduction, modality-balanced minimum redundancy maximum relevance (mRMR), and Elastic Net-based stability selection.
- **Leakage-controlled preprocessing:** Training-partition-specific imputation, Yeo–Johnson transformation, ComBat harmonization, and feature selection.
- **Nested patient-grouped cross-validation:** Four outer folds and three inner folds, keeping all foci from a patient within the same fold.
- **Hyperparameter optimization:** Optuna with the Tree-structured Parzen Estimator (TPE) sampler and inner-fold ROC-AUC as the optimization objective.
- **Model explainability:** SHAP analysis of feature contributions to fitted model predictions.
- **Conventional reference models:** Logistic regression based on SUVmax, SUVmean, focus volume, and total lesion FAPI (TLF), individually and in combination.
- **Statistical assessment:** Patient-cluster bootstrap confidence intervals, paired model comparisons, and Holm adjustment within comparison families.
- **Sensitivity analyses:** CT-only radiomics, analyses without ComBat, volume-overlap analyses, and label-permutation checks.
- **Reproducible outputs:** Out-of-fold predictions, fold-level metrics, selected features, optimized hyperparameters, SHAP values, figures, and manuscript tables.

## ⚙️ Methodology

### Preprocessing and model development

The following pipeline is applied within the appropriate training partitions of the nested cross-validation procedure:

1. **Constant-feature removal** to exclude features without variation.
2. **Median imputation** of missing feature values.
3. **Yeo–Johnson transformation** to reduce feature skewness.
4. **ComBat harmonization** using imaging center as the batch variable, without class-label covariates.
5. **Hierarchical feature selection:**
   - Volume-dependence filtering: remove features with absolute Spearman correlation \(|\rho| > 0.5\) with region volume.
   - Redundancy reduction: remove redundant features using the specified correlation threshold \(|r| > 0.9\).
   - Modality-balanced mRMR ranking with \(k=5\).
   - Elastic Net-based stability selection.
6. **Z-score standardization** of selected features.
7. **SMOTE** applied to training data to address class imbalance.
8. **Classifier training and hyperparameter optimization** using the inner cross-validation folds.

All data-dependent transformations, harmonization parameters, feature-selection decisions, scaling parameters, and model-fitting steps are learned from the relevant training partitions. Held-out evaluation foci are not used to fit these steps.

### Validation and statistical analysis

- **Outer cross-validation:** StratifiedGroupKFold, four folds, with patients kept together.
- **Inner cross-validation:** Three folds for hyperparameter optimization.
- **Optimization:** Optuna TPE, 30 trials per experiment.
- **Random seed:** 100 for the principal analysis.
- **Classification threshold:** 0.5 for threshold-dependent classification metrics.
- **Uncertainty estimation:** Patient-cluster bootstrap with 2,000 resamples.
- **Multiple comparisons:** Holm adjustment within predefined comparison families.
- **Leakage check:** The full pipeline is rerun with permuted labels.

The fitted pipelines are saved for the outer folds to support inspection and reproducibility.


## 📥 Input Data

The dataset-building script expects three LIFEx-exported CSV files:

```text
CT.csv
PET_64.csv
PET_128.csv
```

Each export should contain the following identifiers and label:

| Column | Description |
|---|---|
| `INFO_PatientName` | Patient identifier used to group foci |
| `INFO_NameOfRoi` | ROI name, including focus-identification information |
| `Target` | Class label: 0 = physiological, 1 = pathological |

The exports are expected to contain the radiomic feature columns required by the pipeline. The supplied workflow assumes 145 radiomic features per export; verify the columns against the exported files before running the analysis.

**Data matching:** PET and CT records are matched by patient and focus number, not by row position. This avoids incorrect modality pairing when rows are ordered differently across exports.

**Center assignment:** Patients P1–P12 belong to Center 1, and patients P13–P24 belong to Center 2.

The dataset-building step validates and combines the exports. Patient-level imaging data are not distributed with the repository.

## 🚀 Installation and Execution

Run the following commands from the repository directory. The commands assume that the scripts and dependency file described below are present.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the analysis datasets

```bash
python build_datasets.py --src <folder_with_CT_and_PET_CSVs> --out ../01_data
```

Replace `<folder_with_CT_and_PET_CSVs>` with the directory containing `CT.csv`, `PET_64.csv`, and `PET_128.csv`.

### 3. Run the four principal experiments and permutation checks

```bash
bash run_all.sh
```

### 4. Evaluate conventional reference models

```bash
python reference_models.py
```

This evaluates conventional models based on SUVmax, SUVmean, volume, and TLF, individually and in combination.

### 5. Run sensitivity analyses

```bash
bash run_manuscript_extras.sh
```

This script runs the CT-only analysis, the four experiments without ComBat, and additional permutation checks using seeds 2 and 3.

### 6. Generate summaries, tables, and figures

```bash
python summarize.py
python volume_overlap.py
python clustering_effect.py
python final_model_shap.py
python figures_print.py
python tables_manuscript.py
python make_tables.py
```

The scripts generate pooled performance summaries, volume-overlap and clustering analyses, SHAP outputs, manuscript figures, manuscript tables, and supplementary tables.

**Figure scripts:** `figures_print.py` produces the manuscript-style figures. `figures.py` generates an alternative default pipeline style that was not used in the manuscript.

### Notebook alternative

`FAPI_classification_all_foci.ipynb` provides a notebook implementation of the same analysis workflow for local execution or Kaggle. Set `DATA_DIR` to the appropriate location of the input data before running it.

## 🗂️ Repository and Output Layout

The following describes the expected organization of the code and generated outputs. Output directories are relative to the parent directory (`..`) as specified by the pipeline.

```text
FAPI-PET-Uptake-Classifier/
├── build_datasets.py
├── run_all.sh
├── run_manuscript_extras.sh
├── reference_models.py
├── summarize.py
├── volume_overlap.py
├── clustering_effect.py
├── final_model_shap.py
├── figures_print.py
├── figures.py
├── tables_manuscript.py
├── make_tables.py
├── FAPI_classification_all_foci.ipynb
├── requirements.txt
└── README.md

../01_data/                    # Derived datasets and data checks
../Exp*/                       # Per-experiment predictions, metrics,
                               # selected features, hyperparameters,
                               # SHAP values, fitted outer-fold pipelines
../Reference_conventional/     # Conventional PET reference models
../permutation_tests/          # Label-permutation checks
../Exp*_noComBat/              # Experiments without ComBat
../Sens_CT_only/               # CT-only sensitivity analysis
../summary/                    # Pooled metrics, confidence intervals,
                               # paired model comparisons
../figures_manuscript/         # Manuscript figures (600 dpi PNG and PDF)
../tables_manuscript/          # Manuscript tables and Additional files 1–3
../tables/                     # Additional CSV/XLSX tables
../figures_pipeline_default_style/ # Alternative figure style
```

The actual repository may contain additional files or a different directory layout. Consult the committed scripts and configuration files if an output path differs.

## 🔍 Explainability and Calibration

SHAP is used to describe how selected radiomic features contribute to model predictions. These explanations characterize the behavior of the fitted models and should not be interpreted as evidence of underlying biological mechanisms.

Calibration is assessed using the Brier score, calibration intercept, calibration slope, and decile-based calibration curves. In the revised study, predicted probabilities were not well calibrated; therefore, they should not be interpreted as absolute probabilities of pathology.

Decision Curve Analysis is included as a descriptive analysis and does not, by itself, demonstrate clinical benefit.

## 🧾 Provenance and Implementation Notes

This analysis was adapted from the previous analysis in `Revision_Analysis/code`.

- `fapi_pipeline/feature_selection.py` and `harmonization.py` were retained unchanged.
- `nested_cv.py` adds preprocessing steps 2–3 (median imputation and Yeo–Johnson transformation) and saves the fitted pipelines.
- `models.py` adds an SVM solver iteration cap (`max_iter = 1_000_000`) because one linear-SVM fit did not finish within two hours during the first run.
- The hyperparameter search spaces were unchanged.

These notes document the stated changes relative to the previous analysis and help trace the implementation used for the revised manuscript.

## 📚 Citation

If you use this repository in your research, please cite the associated publication.

**Manuscript title:** *Radiomics-Based Machine Learning for Classification of Physiological and Pathological Uptake on [⁶⁸Ga]Ga-FAPI-46 PET/CT: A Multicenter Proof-of-Concept Study.*

Publication details and DOI will be added when available.

Provisional BibTeX entry:

```bibtex
@article{yazdani2026fapi_pet_radiomics,
  title   = {Radiomics-Based Machine Learning for Classification of Physiological and Pathological Uptake on [68Ga]Ga-FAPI-46 PET/CT: A Multicenter Proof-of-Concept Study},
  author  = {Yazdani, Elmira and others},
  journal = {To be updated},
  year    = {2026}
}
```


---

<p align="center">
  <b>Radiomics • Machine Learning • Explainable AI • FAPI PET/CT</b>
</p>
