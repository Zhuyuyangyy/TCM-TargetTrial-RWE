"""Target Trial Emulator -- orchestrates the full target trial emulation pipeline.

DISCLAIMER: All results produced by this emulator are computed on synthetic or
semi-realistic data for method validation purposes only. They do not represent
real clinical findings and should not be used for clinical decision-making.

PLANNED: Cloning + censoring for immortal time bias correction.
See backend/analysis/survival.py for details.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from backend.models.trial_protocol import TargetTrialProtocol
from backend.models.causal_engine import IPW, AIPW, CausalEstimate
from backend.analysis.propensity_score import PropensityScoreAnalyzer, PSResult
from backend.analysis.survival import SurvivalAnalyzer, CoxResult, RMSTResult


@dataclass
class TrialEmulationResult:
    protocol_id: str
    n_eligible: int
    n_treated: int
    n_control: int
    causal_estimate: Optional[CausalEstimate] = None
    ps_result: Optional[PSResult] = None
    cox_result: Optional[CoxResult] = None
    rmst_result: Optional[RMSTResult] = None
    covariate_balance: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)


class TargetTrialEmulator:
    def __init__(self, protocol: TargetTrialProtocol):
        self.protocol = protocol
        self.ps_analyzer = PropensityScoreAnalyzer()
        self.survival_analyzer = SurvivalAnalyzer()

    def apply_eligibility(self, df):
        elig = self.protocol.eligibility
        mask = pd.Series(True, index=df.index)
        if "age" in df.columns:
            mask &= df["age"] >= elig.min_age
            mask &= df["age"] <= elig.max_age
        for d in elig.excluded_diagnoses:
            if d in df.columns: mask &= df[d] != 1
        for d in elig.required_diagnoses:
            if d in df.columns: mask &= df[d] == 1
        for var, val in elig.min_lab_values.items():
            if var in df.columns: mask &= df[var] >= val
        for var, val in elig.max_lab_values.items():
            if var in df.columns: mask &= df[var] <= val
        return df[mask].reset_index(drop=True)

    def emulate(self, df, treatment_col, outcome_col, covariate_cols,
                time_col=None, causal_method="aipw"):
        warns = []
        eligible = self.apply_eligibility(df)
        if len(eligible) < 50:
            warns.append(f"Small sample after eligibility: n={len(eligible)}")
        n_treated = int(eligible[treatment_col].sum())
        n_control = len(eligible) - n_treated
        ps_result = self.ps_analyzer.full_analysis(eligible, treatment_col, covariate_cols, method="iptw")
        engine = AIPW() if causal_method == "aipw" else IPW()
        causal_est = engine.estimate(eligible, treatment_col, outcome_col, covariate_cols)
        cox_result = rmst_result = None
        # Pass IPW weights to survival models for confounding adjustment
        ipw_weights = ps_result.weights
        if time_col and time_col in eligible.columns:
            try:
                cox_result = self.survival_analyzer.cox_ph(
                    eligible, time_col, outcome_col, treatment_col, covariate_cols,
                    weights=ipw_weights,
                )
            except Exception as e:
                warns.append(f"Cox PH failed: {e}")
            try:
                rmst_result = self.survival_analyzer.rmst(
                    eligible, time_col, outcome_col, treatment_col,
                    weights=ipw_weights,
                )
            except Exception as e:
                warns.append(f"RMST failed: {e}")
        balance = self.ps_analyzer.assess_balance(eligible, treatment_col, covariate_cols, ps_result.weights)
        return TrialEmulationResult(
            protocol_id=self.protocol.trial_id, n_eligible=len(eligible),
            n_treated=n_treated, n_control=n_control,
            causal_estimate=causal_est, ps_result=ps_result,
            cox_result=cox_result, rmst_result=rmst_result,
            covariate_balance=balance.standardized_mean_differences, warnings=warns,
        )
