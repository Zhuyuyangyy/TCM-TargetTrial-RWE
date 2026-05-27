"""Causal inference estimation engines: IPW, AIPW, and doubly-robust estimation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


@dataclass
class CausalEstimate:
    estimate: float
    ci_lower: float
    ci_upper: float
    se: float
    method: str
    n_obs: int
    n_effective: float
    weights_summary: Optional[dict] = None


class CAE:
    """Conditional Average Effect -- standard regression-based estimator."""
    def __init__(self, model=None):
        self.model = model or LogisticRegression(max_iter=1000)

    def estimate(self, df, treatment_col, outcome_col, covariate_cols) -> CausalEstimate:
        from sklearn.linear_model import LinearRegression
        X = df[covariate_cols + [treatment_col]].values
        y = df[outcome_col].values
        ols = LinearRegression().fit(X, y)
        treat_idx = len(covariate_cols)
        treat_coef = ols.coef_[treat_idx]
        residuals = y - ols.predict(X)
        se = np.sqrt(np.mean(residuals**2) / len(y))
        return CausalEstimate(
            estimate=treat_coef, ci_lower=treat_coef - 1.96*se,
            ci_upper=treat_coef + 1.96*se, se=se,
            method="CAE_regression", n_obs=len(df), n_effective=len(df),
        )


class IPW:
    """Inverse Probability of Treatment Weighting estimator."""
    def __init__(self, ps_model=None, trim_percentile=99.0, stabilize=True):
        self.ps_model = ps_model or LogisticRegression(max_iter=1000)
        self.trim_percentile = trim_percentile
        self.stabilize = stabilize

    def fit_propensity_scores(self, df, treatment_col, covariate_cols):
        X = df[covariate_cols].values
        t = df[treatment_col].values
        self.ps_model.fit(X, t)
        ps = self.ps_model.predict_proba(X)[:, 1]
        return np.clip(ps, 0.01, 0.99)

    def compute_weights(self, df, treatment_col, ps):
        t = df[treatment_col].values.astype(float)
        if self.stabilize:
            p_treat = t.mean()
            weights = t * p_treat / ps + (1 - t) * (1 - p_treat) / (1 - ps)
        else:
            weights = t / ps + (1 - t) / (1 - ps)
        cap = np.percentile(weights, self.trim_percentile)
        return np.minimum(weights, cap)

    def estimate(self, df, treatment_col, outcome_col, covariate_cols) -> CausalEstimate:
        ps = self.fit_propensity_scores(df, treatment_col, covariate_cols)
        weights = self.compute_weights(df, treatment_col, ps)
        t = df[treatment_col].values.astype(float)
        y = df[outcome_col].values
        w1, w0 = weights * t, weights * (1 - t)
        mu1 = np.sum(w1 * y) / np.sum(w1)
        mu0 = np.sum(w0 * y) / np.sum(w0)
        ate = mu1 - mu0
        # Bootstrap SE
        rng = np.random.RandomState(42)
        boot = np.empty(500)
        for b in range(500):
            idx = rng.choice(len(df), len(df), replace=True)
            df_b = df.iloc[idx].reset_index(drop=True)
            ps_b = self.fit_propensity_scores(df_b, treatment_col, covariate_cols)
            w_b = self.compute_weights(df_b, treatment_col, ps_b)
            t_b = df_b[treatment_col].values.astype(float)
            y_b = df_b[outcome_col].values
            w1b, w0b = w_b * t_b, w_b * (1 - t_b)
            boot[b] = (np.sum(w1b*y_b)/np.sum(w1b)) - (np.sum(w0b*y_b)/np.sum(w0b))
        se = np.std(boot)
        return CausalEstimate(
            estimate=ate, ci_lower=ate - 1.96*se, ci_upper=ate + 1.96*se,
            se=se, method="IPW", n_obs=len(df),
            n_effective=np.sum(weights)**2 / np.sum(weights**2),
            weights_summary={"mean": float(np.mean(weights)), "sd": float(np.std(weights)),
                             "min": float(np.min(weights)), "max": float(np.max(weights))},
        )


class AIPW:
    """Augmented Inverse Probability Weighting (doubly robust) estimator."""
    def __init__(self, ps_model=None, outcome_model=None, trim_percentile=99.0):
        self.ps_model = ps_model or LogisticRegression(max_iter=1000)
        self.outcome_model = outcome_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42)
        self.trim_percentile = trim_percentile

    def estimate(self, df, treatment_col, outcome_col, covariate_cols) -> CausalEstimate:
        X = df[covariate_cols].values
        t = df[treatment_col].values.astype(float)
        y = df[outcome_col].values
        self.ps_model.fit(X, t)
        ps = np.clip(self.ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
        from sklearn.base import clone
        m1, m0 = clone(self.outcome_model), clone(self.outcome_model)
        mask1, mask0 = t == 1, t == 0
        m1.fit(X[mask1], y[mask1])
        m0.fit(X[mask0], y[mask0])
        mu1 = m1.predict(X)
        mu0 = m0.predict(X)
        aipw_t = t * (y - mu1) / ps + mu1
        aipw_c = (1-t) * (y - mu0) / (1-ps) + mu0
        ate = np.mean(aipw_t - aipw_c)
        if_scores = aipw_t - aipw_c - ate
        se = np.std(if_scores) / np.sqrt(len(df))
        return CausalEstimate(
            estimate=ate, ci_lower=ate - 1.96*se, ci_upper=ate + 1.96*se,
            se=se, method="AIPW", n_obs=len(df), n_effective=float(len(df)),
        )
