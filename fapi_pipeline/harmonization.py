"""
ComBat harmonization with an explicit fit/transform interface.

Parametric empirical-Bayes ComBat (Johnson et al., Biostatistics 2007) with the
imaging center as the batch variable and no biological covariates. Because no
outcome covariate is used, held-out patients are harmonized with parameters
estimated from training patients only and without their labels.
"""

import numpy as np


class ComBat:
    def __init__(self, max_iter: int = 1000, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _aprior(d):
        m, s2 = d.mean(), d.var(ddof=1)
        return (2 * s2 + m ** 2) / s2

    @staticmethod
    def _bprior(d):
        m, s2 = d.mean(), d.var(ddof=1)
        return (m * s2 + m ** 3) / s2

    def fit(self, X: np.ndarray, batch: np.ndarray):
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch)
        self.batches_ = np.unique(batch)
        n, p = X.shape

        batch_means = np.vstack([X[batch == b].mean(axis=0) for b in self.batches_])
        weights = np.array([(batch == b).sum() / n for b in self.batches_])
        self.grand_mean_ = weights @ batch_means
        resid = X - batch_means[np.searchsorted(self.batches_, batch)]
        self.var_pooled_ = (resid ** 2).mean(axis=0)
        # Features without pooled variance are passed through unchanged
        self.active_ = self.var_pooled_ > 1e-12
        vp = np.where(self.active_, self.var_pooled_, 1.0)

        s = (X - self.grand_mean_) / np.sqrt(vp)
        gamma_hat = np.vstack([s[batch == b].mean(axis=0) for b in self.batches_])
        delta_hat = np.vstack([np.maximum(s[batch == b].var(axis=0, ddof=1), 1e-8) for b in self.batches_])

        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.ones_like(delta_hat)
        for i, b in enumerate(self.batches_):
            sb = s[batch == b]
            nb = sb.shape[0]
            # Empirical-Bayes priors are pooled across features; with fewer than
            # three features they are not estimable, so the batch location/scale
            # estimates are used directly (non-EB ComBat).
            if p < 3 or delta_hat[i].var(ddof=1) <= 0:
                gamma_star[i], delta_star[i] = gamma_hat[i], delta_hat[i]
                continue
            g_bar = gamma_hat[i].mean()
            t2 = gamma_hat[i].var(ddof=1)
            a_pr = self._aprior(delta_hat[i])
            b_pr = self._bprior(delta_hat[i])
            g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
            for _ in range(self.max_iter):
                g_new = (nb * t2 * gamma_hat[i] + d_old * g_bar) / (nb * t2 + d_old)
                ssq = ((sb - g_new) ** 2).sum(axis=0)
                d_new = (b_pr + 0.5 * ssq) / (nb / 2.0 + a_pr - 1.0)
                change = max(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12)),
                             np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-12)))
                g_old, d_old = g_new, d_new
                if change < self.tol:
                    break
            gamma_star[i], delta_star[i] = g_old, d_old

        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        return self

    def transform(self, X: np.ndarray, batch: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch)
        unknown = np.setdiff1d(np.unique(batch), self.batches_)
        if unknown.size:
            raise ValueError(f"Batches {unknown} were not present during fitting")
        vp = np.where(self.active_, self.var_pooled_, 1.0)
        idx = np.searchsorted(self.batches_, batch)
        s = (X - self.grand_mean_) / np.sqrt(vp)
        adj = (s - self.gamma_star_[idx]) / np.sqrt(self.delta_star_[idx])
        out = adj * np.sqrt(vp) + self.grand_mean_
        out[:, ~self.active_] = X[:, ~self.active_]
        return out

    def fit_transform(self, X, batch):
        return self.fit(X, batch).transform(X, batch)
