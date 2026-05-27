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
    """Conditional Average Effect -- OLS regression-based estimator."""
    def __init__(self):
        pass

    def estimate(self, df, treatment_col, outcome_col, covariate_cols) -> CausalEstimate:
        from sklearn.linear_model import LinearRegression
        X_raw = df[covariate_cols + [treatment_col]].values
        y = df[outcome_col].values
        ols = LinearRegression(fit_intercept=True).fit(X_raw, y)
        treat_idx = len(covariate_cols)
        treat_coef = ols.coef_[treat_idx]
        residuals = y - ols.predict(X_raw)
        # Add constant column for correct (X'X)^{-1} SE calculation
        X = np.column_stack([np.ones(len(y)), X_raw])
        n, p = X.shape
        mse = np.sum(residuals**2) / (n - p) if n > p else np.sum(residuals**2) / n
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            # treat_idx+1 because column 0 is the intercept
            se = np.sqrt(mse * XtX_inv[treat_idx + 1, treat_idx + 1])
        except np.linalg.LinAlgError:
            se = np.sqrt(mse / n)
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


class TMLE:
    """Targeted Maximum Likelihood Estimation (TMLE) for continuous outcomes.

    Two-stage doubly-robust, locally efficient estimator:
      Stage 1: Initial outcome model Q(A,X) via cross-fitting with ML.
      Stage 2: Targeting step solves the efficient influence curve (EIC)
               estimating equation via a parametric fluctuation.

    For continuous Y: linear fluctuation  Q* = Q + eps * H(A,X)
      where H(A,X) = A/g(X) - (1-A)/(1-g(X)) is the clever covariate.

    Provides ATE estimates with inference based on the EIC.
    Reference: van der Laan & Rose (2011), Targeted Learning.
    """

    def __init__(self, outcome_model=None, ps_model=None, trim_bounds=(0.01, 0.99),
                 n_splits=5, seed=42):
        self.outcome_model = outcome_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        )
        self.ps_model = ps_model or LogisticRegression(max_iter=1000)
        self.trim_bounds = trim_bounds
        self.n_splits = n_splits
        self.seed = seed

    # ------------------------------------------------------------------
    def estimate(self, df, treatment_col, outcome_col, covariate_cols) -> CausalEstimate:
        X = df[covariate_cols].values.astype(float)
        t = df[treatment_col].values.astype(float)
        y = df[outcome_col].values.astype(float)
        n = len(y)

        # ---- Stage 1: Cross-fitted initial predictions ----
        from sklearn.model_selection import KFold
        from sklearn.base import clone as _clone

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        Q1_hat = np.zeros(n)
        Q0_hat = np.zeros(n)
        g_hat  = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            t_tr = t[train_idx]

            # Propensity model P(A=1|X)
            ps = _clone(self.ps_model)
            ps.fit(X_tr, t_tr)
            g_hat[val_idx] = np.clip(
                ps.predict_proba(X_val)[:, 1],
                self.trim_bounds[0], self.trim_bounds[1],
            )

            # Outcome models Q1(X)=E[Y|A=1,X], Q0(X)=E[Y|A=0,X]
            m1 = _clone(self.outcome_model)
            m0 = _clone(self.outcome_model)
            idx1 = t_tr == 1
            idx0 = t_tr == 0
            if idx1.sum() > 5:
                m1.fit(X_tr[idx1], y_tr[idx1])
                Q1_hat[val_idx] = m1.predict(X_val)
            if idx0.sum() > 5:
                m0.fit(X_tr[idx0], y_tr[idx0])
                Q0_hat[val_idx] = m0.predict(X_val)

        # ---- Stage 2: Targeting (fluctuation) step ----
        # Clever covariate: H(A,X) = A/g(X) - (1-A)/(1-g(X))
        H = t / g_hat - (1.0 - t) / (1.0 - g_hat)

        # For continuous Y, use linear fluctuation:
        #   Q_eps(A,X) = Q(A,X) + eps * H(A,X)
        # Solve: 0 = sum_i H_i * (Y_i - Q_hat(A_i,X_i) - eps * H_i)
        # => eps = sum_i H_i * (Y_i - Q_hat_i) / sum_i H_i^2
        Q_obs = t * Q1_hat + (1.0 - t) * Q0_hat  # Q under observed treatment
        residuals = y - Q_obs
        denom = np.mean(H ** 2)
        epsilon = np.mean(H * residuals) / denom if denom > 1e-12 else 0.0

        # Fluctuated predictions for each treatment arm
        # For A=1: H = 1/g(X);   For A=0: H = -1/(1-g(X))
        H1 = 1.0 / g_hat
        H0 = -1.0 / (1.0 - g_hat)
        Q1_star = Q1_hat + epsilon * H1
        Q0_star = Q0_hat + epsilon * H0

        ate = float(np.mean(Q1_star - Q0_star))

        # ---- Inference via efficient influence curve ----
        # EIC = A/g*(Y - Q1*) - (1-A)/(1-g*)*(Y - Q0*) + (Q1* - Q0*) - psi
        eic = (t / g_hat * (y - Q1_star)
               - (1.0 - t) / (1.0 - g_hat) * (y - Q0_star)
               + (Q1_star - Q0_star)
               - ate)
        se = float(np.std(eic) / np.sqrt(n))

        return CausalEstimate(
            estimate=ate,
            ci_lower=ate - 1.96 * se,
            ci_upper=ate + 1.96 * se,
            se=se,
            method="TMLE",
            n_obs=n,
            n_effective=float(n),
            weights_summary={
                "epsilon": float(epsilon),
                "ps_mean": float(np.mean(g_hat)),
                "ps_min": float(np.min(g_hat)),
                "ps_max": float(np.max(g_hat)),
            },
        )
