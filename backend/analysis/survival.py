"""Survival analysis: Kaplan-Meier, Cox PH, and RMST.

DISCLAIMER: This module currently operates on synthetic or semi-realistic data
for method validation only. Results should not be interpreted as real clinical
evidence. Integration with real TCM hospital EHR data is required for
publication-grade research.

PLANNED: Cloning + censoring methods for immortal time bias correction.
Immortal time bias arises when the treatment assignment window includes a period
during which the outcome cannot occur (e.g., patients must survive long enough
to receive TCM adjunctive therapy). A future release will implement:
  - Clone-censor-weight (CCW) approach (Hernan, Robins 2024)
  - Time-zero alignment with grace period handling
  - Landmark analysis as a complementary sensitivity check
See: Hernan MA, Robins JM. Causal Inference: What If. Chapter 17.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class KMResult:
    times: np.ndarray
    survival_function: np.ndarray
    confidence_interval_lower: np.ndarray
    confidence_interval_upper: np.ndarray
    median_survival: Optional[float] = None


@dataclass
class CoxResult:
    hazard_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    coefficients: dict
    concordance: float
    log_likelihood: float


@dataclass
class RMSTResult:
    rmst_treatment: float
    rmst_control: float
    rmst_difference: float
    ci_lower: float
    ci_upper: float
    tau: float


class SurvivalAnalyzer:
    def kaplan_meier(self, df, time_col, event_col, group_col=None, alpha=0.05, weights=None):
        """Kaplan-Meier survival estimator.

        Parameters
        ----------
        weights : array-like, optional
            Per-observation weights (e.g., IPW weights). If provided, a
            weighted KM estimator is used via the KaplanMeierFitter's
            weights parameter.

        NOTE: Results are based on synthetic/semi-realistic data for validation.
        """
        from lifelines import KaplanMeierFitter
        groups = df[group_col].unique() if group_col else ["all"]
        results = {}
        for g in groups:
            if group_col:
                mask = df[group_col] == g
                subset = df[mask]
                g_weights = weights[mask.values] if weights is not None else None
            else:
                subset = df
                g_weights = weights
            kmf = KaplanMeierFitter()
            fit_kwargs = dict(durations=subset[time_col], event_observed=subset[event_col], alpha=alpha)
            if g_weights is not None:
                fit_kwargs["weights"] = g_weights
            kmf.fit(**fit_kwargs)
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_survival_function_
            med = kmf.median_survival_time_
            results[str(g)] = KMResult(
                times=sf.index.values, survival_function=sf.values.flatten(),
                confidence_interval_lower=ci.iloc[:, 0].values,
                confidence_interval_upper=ci.iloc[:, 1].values,
                median_survival=float(med) if not np.isnan(med) else None,
            )
        return results

    def cox_ph(self, df, time_col, event_col, treatment_col, covariate_cols=None,
               alpha=0.05, weights=None):
        """Cox Proportional Hazards model.

        Parameters
        ----------
        weights : array-like, optional
            Per-observation weights (e.g., IPW/stabilized weights). When
            provided, a weighted Cox model is fitted, adjusting for
            confounding via inverse probability weighting.

        NOTE: Results are based on synthetic/semi-realistic data for validation.
        """
        from lifelines import CoxPHFitter
        cols = [treatment_col] + (covariate_cols or [])
        fit_df = df[[time_col, event_col] + cols].copy()
        fit_kwargs = dict(duration_col=time_col, event_col=event_col)
        cph = CoxPHFitter(alpha=alpha)
        if weights is not None:
            fit_df["_ipw_weights"] = np.asarray(weights)
            cph.fit(fit_df, weights_col="_ipw_weights", **fit_kwargs)
        else:
            cph.fit(fit_df, **fit_kwargs)
        s = cph.summary
        r = s.loc[treatment_col]
        return CoxResult(
            hazard_ratio=float(np.exp(r["coef"])),
            ci_lower=float(np.exp(r["coef lower 95%"])),
            ci_upper=float(np.exp(r["coef upper 95%"])),
            p_value=float(r["p"]),
            coefficients={c: float(s.loc[c, "coef"]) for c in cols if c in s.index},
            concordance=float(cph.concordance_index_),
            log_likelihood=float(cph.log_likelihood_),
        )

    def rmst(self, df, time_col, event_col, treatment_col, tau=None, alpha=0.05,
             weights=None):
        """Restricted Mean Survival Time (RMST) difference.

        Parameters
        ----------
        weights : array-like, optional
            Per-observation weights (e.g., IPW weights). When provided,
            the weighted Kaplan-Meier estimator is used to compute RMST
            for each treatment arm.

        NOTE: Results are based on synthetic/semi-realistic data for validation.
        """
        from lifelines import KaplanMeierFitter
        treat = df[treatment_col].values.astype(bool)
        times = df[time_col].values
        events = df[event_col].values
        if tau is None:
            tau = np.percentile(times, 90)

        def _rmst(t_arr, e_arr, t_max, w_arr=None):
            kmf = KaplanMeierFitter()
            fit_kw = dict(durations=t_arr, event_observed=e_arr)
            if w_arr is not None:
                fit_kw["weights"] = w_arr
            kmf.fit(**fit_kw)
            sf = kmf.survival_function_
            t_vals = sf.index.values
            s_vals = sf.values.flatten()
            mask = t_vals <= t_max
            t_r, s_r = t_vals[mask], s_vals[mask]
            if len(t_r) == 0: return 0.0
            dt = np.diff(np.concatenate([[0], t_r]))
            return float(np.sum(s_r * dt))

        w1 = weights[treat] if weights is not None else None
        w0 = weights[~treat] if weights is not None else None
        rmst1 = _rmst(times[treat], events[treat], tau, w1)
        rmst0 = _rmst(times[~treat], events[~treat], tau, w0)
        diff = rmst1 - rmst0
        rng = np.random.RandomState(42)
        boot = []
        for _ in range(500):
            idx = rng.choice(len(df), len(df), replace=True)
            b = df.iloc[idx]
            bt = b[treatment_col].values.astype(bool)
            bw = weights[idx] if weights is not None else None
            bw1 = bw[bt] if bw is not None else None
            bw0 = bw[~bt] if bw is not None else None
            boot.append(_rmst(b[time_col].values[bt], b[event_col].values[bt], tau, bw1)
                        - _rmst(b[time_col].values[~bt], b[event_col].values[~bt], tau, bw0))
        se = np.std(boot)
        return RMSTResult(rmst_treatment=rmst1, rmst_control=rmst0, rmst_difference=diff,
                          ci_lower=diff - 1.96*se, ci_upper=diff + 1.96*se, tau=tau)
