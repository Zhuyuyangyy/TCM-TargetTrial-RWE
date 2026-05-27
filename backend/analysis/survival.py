"""Survival analysis: Kaplan-Meier, Cox PH, and RMST."""
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
    def kaplan_meier(self, df, time_col, event_col, group_col=None, alpha=0.05):
        from lifelines import KaplanMeierFitter
        groups = df[group_col].unique() if group_col else ["all"]
        results = {}
        for g in groups:
            subset = df[df[group_col] == g] if group_col else df
            kmf = KaplanMeierFitter()
            kmf.fit(durations=subset[time_col], event_observed=subset[event_col], alpha=alpha)
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

    def cox_ph(self, df, time_col, event_col, treatment_col, covariate_cols=None, alpha=0.05):
        from lifelines import CoxPHFitter
        cols = [treatment_col] + (covariate_cols or [])
        cph = CoxPHFitter(alpha=alpha)
        cph.fit(df[[time_col, event_col] + cols], duration_col=time_col, event_col=event_col)
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

    def rmst(self, df, time_col, event_col, treatment_col, tau=None, alpha=0.05):
        from lifelines import KaplanMeierFitter
        treat = df[treatment_col].values.astype(bool)
        times = df[time_col].values
        events = df[event_col].values
        if tau is None:
            tau = np.percentile(times, 90)

        def _rmst(t_arr, e_arr, t_max):
            kmf = KaplanMeierFitter()
            kmf.fit(t_arr, e_arr)
            sf = kmf.survival_function_
            t_vals = sf.index.values
            s_vals = sf.values.flatten()
            mask = t_vals <= t_max
            t_r, s_r = t_vals[mask], s_vals[mask]
            if len(t_r) == 0: return 0.0
            dt = np.diff(np.concatenate([[0], t_r]))
            return float(np.sum(s_r * dt))

        rmst1 = _rmst(times[treat], events[treat], tau)
        rmst0 = _rmst(times[~treat], events[~treat], tau)
        diff = rmst1 - rmst0
        rng = np.random.RandomState(42)
        boot = []
        for _ in range(500):
            idx = rng.choice(len(df), len(df), replace=True)
            b = df.iloc[idx]
            bt = b[treatment_col].values.astype(bool)
            boot.append(_rmst(b[time_col].values[bt], b[event_col].values[bt], tau)
                        - _rmst(b[time_col].values[~bt], b[event_col].values[~bt], tau))
        se = np.std(boot)
        return RMSTResult(rmst_treatment=rmst1, rmst_control=rmst0, rmst_difference=diff,
                          ci_lower=diff - 1.96*se, ci_upper=diff + 1.96*se, tau=tau)
