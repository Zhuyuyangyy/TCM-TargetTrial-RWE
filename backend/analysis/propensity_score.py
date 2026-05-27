"""Propensity score estimation, IPTW, matching, and stratification."""
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
