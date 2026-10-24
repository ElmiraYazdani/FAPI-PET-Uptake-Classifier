"""
Four-stage feature selection, fitted on a training partition only.

Stage 1  Volume-dependence filter: features with |Spearman rho| > threshold
         against the region volume (PET volume; CT volume for CT-only) are
         removed. No hypothesis test or FDR correction is involved.
Stage 2  Redundancy reduction: features ranked by univariable training-set
         ROC-AUC (max(AUC, 1-AUC)); greedily keep the best-ranked feature and
         drop features with |Pearson r| > threshold to it.
Stage 3  Modality-balanced mRMR: mutual-information relevance minus mean
         |Pearson r| redundancy, greedy forward selection, up to k per modality
         (modality identified by the 'PET_'/'CT_' prefix).
Stage 4  Elastic-Net stability selection on stratified subsamples, with
         threshold relaxation and a minimum number of retained features.
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler


def stage1_volume(X, names, volume, threshold):
    rv = rankdata(volume)
    rx = np.apply_along_axis(rankdata, 0, X)
    rx_c = rx - rx.mean(0)
    rv_c = rv - rv.mean()
    denom = np.sqrt((rx_c ** 2).sum(0) * (rv_c ** 2).sum())
    rho = np.where(denom > 0, (rx_c * rv_c[:, None]).sum(0) / np.where(denom > 0, denom, 1), 0.0)
    keep = np.where(np.abs(rho) <= threshold)[0]
    return keep, dict(zip(names, np.abs(rho).round(4).tolist()))


def _uni_auc(x, y):
    r = rankdata(x)
    n1 = y.sum()
    n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(auc, 1 - auc)


def stage2_redundancy(X, y, threshold):
    p = X.shape[1]
    aucs = np.array([_uni_auc(X[:, j], y) for j in range(p)])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
    remaining = set(range(p))
    selected = []
    for j in np.argsort(-aucs):
        if j not in remaining:
            continue
        selected.append(j)
        remaining.discard(j)
        remaining -= {o for o in remaining if abs(corr[j, o]) > threshold}
    return np.array(sorted(selected)), aucs


def stage3_mrmr(X, y, names, k_per_modality, random_state):
    keep = []
    for prefix in ("CT_", "PET_"):
        idx = [i for i, n in enumerate(names) if n.startswith(prefix)]
        if not idx:
            continue
        Xs = X[:, idx]
        rel = mutual_info_classif(Xs, y, random_state=random_state, n_neighbors=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.nan_to_num(np.corrcoef(Xs, rowvar=False)))
        corr = np.atleast_2d(corr)
        chosen, rest = [], list(range(len(idx)))
        for _ in range(min(k_per_modality, len(idx))):
            scores = [rel[c] - (corr[c, chosen].mean() if chosen else 0.0) for c in rest]
            best = rest[int(np.argmax(scores))]
            chosen.append(best)
            rest.remove(best)
        keep += [idx[c] for c in chosen]
    return np.array(sorted(keep))


def stage4_stability(X, y, cfg, random_state):
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(random_state)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    counts = np.zeros(X.shape[1])
    ok = 0
    for it in range(cfg["STABILITY_N_ITERATIONS"]):
        s0 = rng.choice(idx0, max(int(len(idx0) * cfg["STABILITY_SUBSAMPLE_RATIO"]), 2), replace=False)
        s1 = rng.choice(idx1, max(int(len(idx1) * cfg["STABILITY_SUBSAMPLE_RATIO"]), 2), replace=False)
        sub = np.concatenate([s0, s1])
        l1 = rng.choice(cfg["ELASTICNET_L1_RATIOS"])
        C = rng.choice(cfg["ELASTICNET_CS"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1, max_iter=2000, tol=1e-4,
                                       random_state=random_state + it).fit(Xs[sub], y[sub])
            counts += (np.abs(m.coef_[0]) > cfg["ELASTICNET_COEF_THRESHOLD"])
            ok += 1
        except Exception:
            pass
    freq = counts / max(ok, 1)
    thr = cfg["STABILITY_SELECTION_THRESHOLD"]
    keep = np.where(freq >= thr)[0]
    while len(keep) < cfg["STABILITY_MIN_FEATURES"] and thr - cfg["STABILITY_FALLBACK_STEP"] >= cfg["STABILITY_FALLBACK_MIN_THRESHOLD"] - 1e-9:
        thr = round(thr - cfg["STABILITY_FALLBACK_STEP"], 4)
        keep = np.where(freq >= thr)[0]
    if len(keep) < cfg["STABILITY_MIN_FEATURES"]:
        keep = np.sort(np.argsort(-freq)[:cfg["STABILITY_MIN_FEATURES"]])
        thr = float(freq[keep].min())
    return keep, freq, thr


def select_features(X: np.ndarray, y: np.ndarray, names: List[str], volume: np.ndarray,
                    cfg: Dict, random_state: int) -> Tuple[np.ndarray, Dict]:
    """Returns column indices (into X) of the selected features and a stage log."""
    log = {"n_input": X.shape[1]}
    cur = np.arange(X.shape[1])

    k1, _ = stage1_volume(X, names, volume, cfg["VOLUME_CORR_THRESHOLD"])
    cur = cur[k1]
    log["n_after_volume"] = len(cur)

    k2, _ = stage2_redundancy(X[:, cur], y, cfg["REDUNDANCY_CORR_THRESHOLD"])
    cur = cur[k2]
    log["n_after_redundancy"] = len(cur)

    k3 = stage3_mrmr(X[:, cur], y, [names[i] for i in cur], cfg["MRMR_PER_MODALITY"], random_state)
    cur = cur[k3]
    log["n_after_mrmr"] = len(cur)

    k4, freq, thr = stage4_stability(X[:, cur], y, cfg, random_state)
    log["stability_freq"] = dict(zip([names[i] for i in cur], np.round(freq, 4).tolist()))
    log["stability_threshold_used"] = thr
    cur = cur[k4]
    log["n_final"] = len(cur)
    log["selected"] = [names[i] for i in cur]
    return cur, log
