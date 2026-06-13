"""Propensity score estimation, IPTW, matching, and stratification.

DISCLAIMER: All results in this module are computed on synthetic or semi-realistic
data for method validation only. They do not represent real clinical findings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class PSResult:
    propensity_scores: np.ndarray
    weights: Optional[np.ndarray] = None
    matched_indices: Optional[list] = None
    standardized_mean_diffs: dict = field(default_factory=dict)
    method: str = "logistic"


@dataclass
class BalanceResult:
    standardized_mean_differences: dict
    variance_ratios: dict
    overall_balance: str


class PropensityScoreAnalyzer:
    MODELS = {
        "logistic": lambda: LogisticRegression(max_iter=1000),
        "gbm": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.model = self.MODELS.get(model_type, self.MODELS["logistic"])()
        self._scaler = StandardScaler()

    def estimate_propensity_scores(self, df, treatment_col, covariate_cols):
        X = self._scaler.fit_transform(df[covariate_cols].values)
        t = df[treatment_col].values
        self.model.fit(X, t)
        return np.clip(self.model.predict_proba(X)[:, 1], 0.01, 0.99)

    def compute_iptw(self, ps, treatment, stabilize=True):
        t = treatment.astype(float)
        if stabilize:
            p_t = t.mean()
            return t * p_t / ps + (1 - t) * (1 - p_t) / (1 - ps)
        return t / ps + (1 - t) / (1 - ps)

    def match(self, df, treatment_col, ps, caliper=0.05, ratio=1):
        treated_idx = np.where(df[treatment_col].values == 1)[0]
        control_idx = np.where(df[treatment_col].values == 0)[0]
        ps_t = ps[treated_idx]
        ps_c = ps[control_idx]
        matched, used = [], set()
        for i, ti in enumerate(treated_idx):
            dists = np.abs(ps_c - ps_t[i])
            order = control_idx[np.argsort(dists)]
            cnt = 0
            for ci in order:
                if ci in used: continue
                if dists[np.where(control_idx == ci)[0][0]] > caliper: break
                matched.append((int(ti), int(ci)))
                used.add(ci)
                cnt += 1
                if cnt >= ratio: break
        return matched

    def assess_balance(self, df, treatment_col, covariate_cols, weights=None):
        t = df[treatment_col].values.astype(bool)
        smds, vr = {}, {}
        for col in covariate_cols:
            x = df[col].values.astype(float)
            w = weights if weights is not None else np.ones(len(x))
            w1, w0 = w[t], w[~t]
            x1, x0 = x[t], x[~t]
            m1, m0 = np.average(x1, weights=w1), np.average(x0, weights=w0)
            s = np.sqrt((np.var(x1) + np.var(x0)) / 2)
            smds[col] = abs(m1 - m0) / s if s > 0 else 0
            v1 = np.average((x1 - m1)**2, weights=w1) if len(x1) > 1 else 0
            v0 = np.average((x0 - m0)**2, weights=w0) if len(x0) > 1 else 0
            vr[col] = v1 / v0 if v0 > 0 else float("inf")
        mx = max(smds.values()) if smds else 0
        return BalanceResult(smds, vr, "good" if mx < 0.1 else ("adequate" if mx < 0.25 else "poor"))

    def full_analysis(self, df, treatment_col, covariate_cols, method="iptw", caliper=0.05):
        ps = self.estimate_propensity_scores(df, treatment_col, covariate_cols)
        t = df[treatment_col].values
        weights = matched = None
        if method == "iptw":
            weights = self.compute_iptw(ps, t)
        elif method == "matching":
            matched = self.match(df, treatment_col, ps, caliper=caliper)
        balance = self.assess_balance(df, treatment_col, covariate_cols, weights)
        return PSResult(ps, weights, matched, balance.standardized_mean_differences,
                        f"PS_{method}_{self.model_type}")


class OverlapWeightEstimator:
    """Overlap (entropy) weighting for causal inference.

    Overlap weights: w(A,X) = A*(1-e(X)) + (1-A)*e(X), where e(X) is the
    propensity score.  These weights naturally down-weight subjects near the
    treatment decision boundary (extreme PS values), eliminating the need
    for arbitrary trimming.  The resulting estimator targets the ATO
    (average treatment effect in the overlap population).

    Reference: Li, Morgan & Zaslavsky (2018), JASA.
    """

    def __init__(self, ps_model=None):
        self.ps_model = ps_model or LogisticRegression(max_iter=1000)
        self._scaler = StandardScaler()

    def fit_propensity(self, df, treatment_col, covariate_cols):
        X = self._scaler.fit_transform(df[covariate_cols].values)
        t = df[treatment_col].values
        self.ps_model.fit(X, t)
        ps = np.clip(self.ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
        return ps

    def overlap_weights(self, ps, treatment):
        """Compute overlap weights: A*(1-ps) + (1-A)*ps."""
        t = treatment.astype(float)
        return t * (1 - ps) + (1 - t) * ps

    def entropy_weights(self, ps, treatment):
        """Entropy weights: minimize KL divergence (Li & Li, 2019)."""
        t = treatment.astype(float)
        w1 = -ps * np.log(ps)              # treated contribution
        w0 = -(1 - ps) * np.log(1 - ps)    # control contribution
        return t * w1 + (1 - t) * w0

    def estimate_ato(self, df, treatment_col, outcome_col, covariate_cols):
        """Estimate the ATE in the overlap population using overlap weights.

        Returns (ate, se, ci_lower, ci_upper, weights, ps).
        """
        ps = self.fit_propensity(df, treatment_col, covariate_cols)
        t = df[treatment_col].values.astype(float)
        y = df[outcome_col].values.astype(float)
        w = self.overlap_weights(ps, t)

        w1 = w * t
        w0 = w * (1 - t)
        mu1 = np.sum(w1 * y) / np.sum(w1)
        mu0 = np.sum(w0 * y) / np.sum(w0)
        ate = mu1 - mu0

        # Sandwich / robust SE via influence function
        n = len(y)
        D1 = w1 * (y - mu1) / np.mean(w1)
        D0 = w0 * (y - mu0) / np.mean(w0)
        D = D1 - D0 - ate
        se = np.std(D) / np.sqrt(n)

        return {
            "estimate": float(ate),
            "se": float(se),
            "ci_lower": float(ate - 1.96 * se),
            "ci_upper": float(ate + 1.96 * se),
            "weights": w,
            "ps": ps,
            "n_effective": float(np.sum(w) ** 2 / np.sum(w ** 2)),
            "method": "overlap_weighting",
        }

    def full_analysis(self, df, treatment_col, outcome_col, covariate_cols):
        """Run overlap weighting and return PSResult-compatible object."""
        result = self.estimate_ato(df, treatment_col, outcome_col, covariate_cols)
        psa = PropensityScoreAnalyzer()
        balance = psa.assess_balance(df, treatment_col, covariate_cols, result["weights"])
        return {
            **result,
            "balance": balance.overall_balance,
            "smds": balance.standardized_mean_differences,
            "variance_ratios": balance.variance_ratios,
        }
