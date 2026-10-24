"""
Leakage-free nested, patient-grouped cross-validation.

For every (inner or outer) training/evaluation split, the following operations
are estimated on the training partition only and then applied, unchanged, to
the evaluation partition:

    1. removal of features that are constant in the training partition
    2. median imputation of missing values               fit on train -> apply   (new: updated data contain NaN)
    3. Yeo-Johnson transformation per feature           fit on train -> apply   (new: cfg['USE_YEOJOHNSON'])
    4. ComBat (batch = center, no covariates)          fit on train -> apply
       (can be switched off with cfg['USE_COMBAT'] = False for sensitivity analyses)
    5. four-stage feature selection (uses train labels)  fit on train
    6. z-score standardization                         fit on train -> apply
    7. SMOTE                                            train only
    8. classifier fitting                               train only

Inner loop (3-fold StratifiedGroupKFold within the outer-training patients):
Optuna/TPE chooses hyperparameters by mean inner-validation ROC-AUC. Steps 1-7
are recomputed for each inner split (they do not depend on hyperparameters, so
they are cached across trials). Outer loop (4-fold StratifiedGroupKFold):
steps 1-8 are refitted on all outer-training patients with the selected
hyperparameters, and probabilities are predicted for the untouched outer-test
patients. No probability recalibration is performed; class predictions use a
fixed 0.5 threshold.
"""

import time
import warnings

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler

from .feature_selection import select_features
from .harmonization import ComBat
from .models import MODEL_NAMES, TREE_MODELS, build_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def smote(X, y, rs):
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        return X, y
    return SMOTE(random_state=rs, k_neighbors=min(5, counts.min() - 1)).fit_resample(X, y)


class FoldPreprocessor:
    """Steps 1-6 fitted on a training partition; transform() applies them unchanged to new foci."""

    def fit(self, Xtr, ytr, btr, vtr, names, cfg, rs, use_combat=True):
        self.nz_ = np.nanstd(Xtr, axis=0) > 1e-12
        nm = [n for n, k in zip(names, self.nz_) if k]
        Z = Xtr[:, self.nz_]
        self.imputer_ = SimpleImputer(strategy="median").fit(Z)
        Z = self.imputer_.transform(Z)
        self.yj_ = None
        if cfg.get("USE_YEOJOHNSON", False):
            self.yj_ = PowerTransformer(method="yeo-johnson", standardize=False).fit(Z)
            Z = self._yj(Z)
        self.combat_ = ComBat().fit(Z, btr) if use_combat else None
        if self.combat_ is not None:
            Z = self.combat_.transform(Z, btr)
        self.sel_, self.log_ = select_features(Z, ytr, nm, vtr, cfg, rs)
        self.scaler_ = StandardScaler().fit(Z[:, self.sel_])
        self.names_ = [nm[i] for i in self.sel_]
        self.Ztr_ = self.scaler_.transform(Z[:, self.sel_])
        return self

    def _yj(self, Z):
        out = self.yj_.transform(Z)
        bad = ~np.isfinite(out).all(axis=0)  # numerical safeguard: keep the untransformed feature
        out[:, bad] = Z[:, bad]
        return out

    def transform(self, X, b):
        Z = self.imputer_.transform(X[:, self.nz_])
        if self.yj_ is not None:
            Z = self._yj(Z)
        if self.combat_ is not None:
            Z = self.combat_.transform(Z, b)
        return self.scaler_.transform(Z[:, self.sel_])


def preprocess(Xtr, ytr, btr, vtr, Xev, bev, names, cfg, rs, use_combat=True):
    """Fits steps 1-7 on the training partition and applies steps 1-6 to the evaluation partition."""
    fp = FoldPreprocessor().fit(Xtr, ytr, btr, vtr, names, cfg, rs, use_combat)
    Xtr_s, Xev_s, log = fp.Ztr_, fp.transform(Xev, bev), dict(fp.log_)
    n_before = {int(k): int(v) for k, v in zip(*np.unique(ytr, return_counts=True))}
    Xtr_b, ytr_b = smote(Xtr_s, ytr, rs)
    n_after = {int(k): int(v) for k, v in zip(*np.unique(ytr_b, return_counts=True))}
    log.update({"smote_before": n_before, "smote_after": n_after})
    return Xtr_b, ytr_b, Xtr_s, Xev_s, fp.names_, log, fp


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def _shap_values(model, name, X):
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def run_nested_cv(df: pd.DataFrame, feature_cols, volume_col, cfg, setting, log=print):
    rs = cfg["RANDOM_STATE"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    g = df["patient_id"].to_numpy()
    b = df["center"].to_numpy()
    vol = df[volume_col].to_numpy(dtype=float)

    outer = StratifiedGroupKFold(n_splits=cfg["N_OUTER_FOLDS"], shuffle=True, random_state=rs)
    oof, fold_rows, feat_rows, param_rows, shap_store, fitted = [], [], [], [], [], []

    for k, (tr, te) in enumerate(outer.split(X, y, g), start=1):
        t0 = time.time()
        # ---- inner splits: preprocessing cached once per split ----
        inner = StratifiedGroupKFold(n_splits=cfg["N_INNER_FOLDS"], shuffle=True, random_state=rs)
        cache = []
        for j, (itr, iva) in enumerate(inner.split(X[tr], y[tr], g[tr]), start=1):
            a, c = tr[itr], tr[iva]
            Xb, yb, _, Xv, sel_names, ilog, _ = preprocess(X[a], y[a], b[a], vol[a], X[c], b[c],
                                                        feature_cols, cfg, rs, cfg.get("USE_COMBAT", True))
            cache.append((Xb, yb, Xv, y[c]))
            feat_rows.append({"setting": setting, "outer_fold": k, "inner_fold": j,
                              "n_features": len(sel_names), "features": "; ".join(sel_names),
                              "n_after_volume": ilog["n_after_volume"],
                              "n_after_redundancy": ilog["n_after_redundancy"],
                              "n_after_mrmr": ilog["n_after_mrmr"],
                              "stability_threshold": ilog["stability_threshold_used"],
                              "smote_before": ilog["smote_before"], "smote_after": ilog["smote_after"]})

        # ---- outer preprocessing ----
        Xb, yb, Xtr_s, Xte, sel_names, olog, fprep = preprocess(X[tr], y[tr], b[tr], vol[tr], X[te], b[te],
                                                          feature_cols, cfg, rs, cfg.get("USE_COMBAT", True))
        feat_rows.append({"setting": setting, "outer_fold": k, "inner_fold": 0,
                          "n_features": len(sel_names), "features": "; ".join(sel_names),
                          "n_after_volume": olog["n_after_volume"],
                          "n_after_redundancy": olog["n_after_redundancy"],
                          "n_after_mrmr": olog["n_after_mrmr"],
                          "stability_threshold": olog["stability_threshold_used"],
                          "smote_before": olog["smote_before"], "smote_after": olog["smote_after"]})

        for name in MODEL_NAMES:
            def objective(trial):
                proto = build_model(trial, name, rs)
                aucs = []
                for Xb_i, yb_i, Xv_i, yv_i in cache:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            m = clone(proto).fit(Xb_i, yb_i)
                        aucs.append(roc_auc_score(yv_i, _proba(m, Xv_i)))
                    except Exception:
                        aucs.append(0.5)
                return float(np.mean(aucs))

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rs))
            study.optimize(objective, n_trials=cfg["N_OPTUNA_TRIALS"], show_progress_bar=False)
            model = build_model(optuna.trial.FixedTrial(study.best_params), name, rs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(Xb, yb)
            fitted.append({"outer_fold": k, "model_name": name, "preprocessor": fprep, "model": model,
                           "best_params": study.best_params, "selected_features": sel_names})
            p_te = _proba(model, Xte)
            p_tr = _proba(model, Xtr_s)
            param_rows.append({"setting": setting, "model": name, "outer_fold": k,
                               "best_inner_auc": study.best_value, "best_params": study.best_params})
            fold_rows.append({"setting": setting, "model": name, "outer_fold": k,
                              "n_train": len(tr), "n_test": len(te),
                              "n_test_pos": int(y[te].sum()), "n_test_neg": int((1 - y[te]).sum()),
                              "test_patients": ",".join(map(str, sorted(np.unique(g[te])))),
                              "train_auc": roc_auc_score(y[tr], p_tr),
                              "test_auc": roc_auc_score(y[te], p_te) if len(np.unique(y[te])) > 1 else np.nan,
                              "best_inner_auc": study.best_value, "n_features": len(sel_names)})
            for i, idx in enumerate(te):
                oof.append({"setting": setting, "model": name, "outer_fold": k,
                            "patient_id": int(g[idx]), "region_id": df["region_id"].iat[idx],
                            "center": int(b[idx]), "label": int(y[idx]),
                            "prob": float(p_te[i]), "pred": int(p_te[i] >= 0.5)})
            if name in TREE_MODELS:
                try:
                    sv = _shap_values(model, name, Xte)
                    shap_store.append({"setting": setting, "model": name, "outer_fold": k,
                                       "features": sel_names, "shap": sv, "X": Xte,
                                       "region_id": df["region_id"].iloc[te].tolist()})
                except Exception as e:
                    log(f"SHAP failed for {name} fold {k}: {e}")
        log(f"[{setting}] outer fold {k} done in {time.time() - t0:.0f}s; {len(sel_names)} features: {sel_names}")

    return (pd.DataFrame(oof), pd.DataFrame(fold_rows), pd.DataFrame(feat_rows),
            pd.DataFrame(param_rows), shap_store, fitted)
