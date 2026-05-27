#!/usr/bin/env python3
"""Run the full TCM-TargetTrial-RWE pipeline on semi-realistic NSCLC data.

Pipeline:
  1. Generate / load semi-realistic EHR data (3000 NSCLC patients)
  2. Handle missing data (median imputation for this demo)
  3. Scale covariates for stable PS estimation
  4. Propensity score estimation (logistic)
  5. Covariate balance diagnostics (SMD before/after, Love plot data)
  6. IPW, AIPW, and TMLE causal estimation
  7. Survival analysis (Cox PH, RMST)
  8. Cost-effectiveness (ICER, NMB)
  9. Sensitivity analysis (E-value, tipping point)
  10. Print comprehensive results table
"""
from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# ======================================================================
# Utility
# ======================================================================
def hr_line(width: int = 70, char: str = "=") -> str:
    return char * width


def section(title: str) -> None:
    print(f"\n{hr_line()}")
    print(f"  {title}")
    print(hr_line())


def fmt(val, decimals: int = 4, width: int = 12) -> str:
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, float):
        return f"{val:.{decimals}f}".rjust(width)
    return str(val).rjust(width)


# ======================================================================
# 1. Generate semi-realistic data
# ======================================================================
def generate_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Import and run the semi-realistic EHR generator."""
    from scripts.generate_semireal_ehr import _generate_semireal_ehr
    return _generate_semireal_ehr(n=n, seed=seed)


# ======================================================================
# 2. Preprocess: impute missing, scale covariates
# ======================================================================
def preprocess(df: pd.DataFrame, covariate_cols: list[str]) -> pd.DataFrame:
    """Simple imputation: median for numeric columns. Returns copy."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def scale_covariates(df: pd.DataFrame, covariate_cols: list[str]) -> pd.DataFrame:
    """Standardize covariates (zero mean, unit variance) for stable PS estimation."""
    from sklearn.preprocessing import StandardScaler
    df = df.copy()
    scaler = StandardScaler()
    df[covariate_cols] = scaler.fit_transform(df[covariate_cols])
    return df


# ======================================================================
# Main pipeline
# ======================================================================
def run_full_analysis(n: int = 3000, seed: int = 42) -> dict:
    """Execute the full analysis pipeline and return all results."""
    results = {}
    t0 = time.time()

    # ---- Generate data ----
    section("1. GENERATING SEMI-REALISTIC NSCLC DATA")
    df_raw = generate_data(n=n, seed=seed)
    print(f"  Raw data shape: {df_raw.shape}")
    print(f"  Missing cells  : {df_raw.isna().sum().sum()}")
    print(f"  Treatment group: {df_raw['treatment'].sum()} treated, "
          f"{(1-df_raw['treatment']).sum():.0f} control")
    print(f"  TCM adjunctive : {df_raw['tcm_therapy'].sum()} ({df_raw['tcm_therapy'].mean()*100:.1f}%)")

    # ---- Define analysis variables ----
    treatment_col = "treatment"
    outcome_col = "death"          # binary event indicator for survival
    time_col = "survival_months"
    cost_col = "total_cost_k_cny"
    effect_col = "qol_score"

    covariate_cols = [
        "age", "sex", "bmi", "stage", "egfr_mutation", "pd_l1_tps",
        "tumor_size_cm", "ecog_score", "charlson_index",
        "albumin", "creatinine", "hemoglobin", "nlr", "platelet",
        "tcm_therapy", "chemo_cycles",
    ]

    # ---- Preprocess ----
    section("2. PREPROCESSING")
    df = preprocess(df_raw, covariate_cols)
    print(f"  Imputed NAs   : {df_raw.isna().sum().sum()} cells -> {df.isna().sum().sum()} remaining")
    print(f"  Treatment     : {treatment_col}")
    print(f"  Outcome       : {outcome_col} (survival event)")
    print(f"  Time          : {time_col}")
    print(f"  Covariates    : {len(covariate_cols)} variables")

    # Scale for stable logistic regression
    df_scaled = scale_covariates(df, covariate_cols)
    print(f"  Covariates standardized for PS estimation")

    # ==============================================================
    # 3. Propensity Score Estimation
    # ==============================================================
    section("3. PROPENSITY SCORE ESTIMATION")
    from backend.analysis.propensity_score import PropensityScoreAnalyzer

    psa = PropensityScoreAnalyzer(model_type="logistic")
    ps = psa.estimate_propensity_scores(df_scaled, treatment_col, covariate_cols)
    print(f"  PS range : [{ps.min():.4f}, {ps.max():.4f}]")
    print(f"  PS mean  : {ps.mean():.4f}")
    print(f"  PS sd    : {ps.std():.4f}")

    weights = psa.compute_iptw(ps, df[treatment_col].values.astype(float), stabilize=True)
    print(f"  IPTW range: [{weights.min():.4f}, {weights.max():.4f}]")
    results["ps_mean"] = ps.mean()

    # ==============================================================
    # 4. Covariate Balance Diagnostics
    # ==============================================================
    section("4. COVARIATE BALANCE DIAGNOSTICS")
    from backend.analysis.diagnostics import (
        CovariateBalanceDiag, LovePlotData, PropensityDistribution, diagnose,
    )

    # Run diagnostics on ORIGINAL scale (for interpretable SMDs)
    diag = diagnose(df, treatment_col, covariate_cols, weights, ps)
    print(diag.summary())

    # Overlap weights
    from backend.analysis.propensity_score import OverlapWeightEstimator
    ols = OverlapWeightEstimator()
    ps_ols = ols.fit_propensity(df_scaled, treatment_col, covariate_cols)
    ow = ols.overlap_weights(ps_ols, df[treatment_col].values.astype(float))
    bal_ow = CovariateBalanceDiag.compute(df, treatment_col, covariate_cols, weights=ow)
    print(f"\n  Overlap-weighted max |SMD|: {bal_ow.max_smd_after:.4f}")

    results["balance"] = diag.balance_after.overall
    results["max_smd_before"] = diag.balance_before.max_smd_before
    results["max_smd_after"] = diag.balance_after.max_smd_after

    # ==============================================================
    # 5. Causal Estimation (IPW, AIPW, TMLE)
    # ==============================================================
    section("5. CAUSAL ESTIMATION (binary outcome: death)")
    from backend.models.causal_engine import IPW, AIPW, TMLE

    estimators = {}

    print("  Running IPW (bootstrap=50)...")
    ipw_est = IPW(n_bootstrap=50, trim_percentile=99)
    est_ipw = ipw_est.estimate(df_scaled, treatment_col, outcome_col, covariate_cols)
    estimators["IPW"] = est_ipw

    print("  Running AIPW...")
    aipw_est = AIPW()
    est_aipw = aipw_est.estimate(df_scaled, treatment_col, outcome_col, covariate_cols)
    estimators["AIPW"] = est_aipw

    print("  Running TMLE (5-fold cross-fitting)...")
    tmle_est = TMLE(n_splits=5)
    est_tmle = tmle_est.estimate(df_scaled, treatment_col, outcome_col, covariate_cols)
    estimators["TMLE"] = est_tmle

    # Overlap weighting
    print("  Running Overlap Weighting...")
    ow_result = ols.estimate_ato(df_scaled, treatment_col, outcome_col, covariate_cols)
    estimators["OverlapW"] = type("E", (), {
        "estimate": ow_result["estimate"],
        "ci_lower": ow_result["ci_lower"],
        "ci_upper": ow_result["ci_upper"],
        "se": ow_result["se"],
        "method": "OverlapWeighting",
        "n_effective": ow_result["n_effective"],
    })()

    print(f"\n  {'Method':<20s} {'Estimate':>10s} {'SE':>10s} {'95% CI':>25s} {'n_eff':>10s}")
    print(f"  {'-'*75}")
    for name, est in estimators.items():
        ci = f"[{est.ci_lower:.4f}, {est.ci_upper:.4f}]"
        neff = f"{est.n_effective:.0f}" if hasattr(est, "n_effective") else "N/A"
        print(f"  {name:<20s} {est.estimate:>10.4f} {est.se:>10.4f} {ci:>25s} {neff:>10s}")

    results["ipw_est"] = est_ipw.estimate
    results["aipw_est"] = est_aipw.estimate
    results["tmle_est"] = est_tmle.estimate

    # ==============================================================
    # 6. Survival Analysis
    # ==============================================================
    section("6. SURVIVAL ANALYSIS")
    from backend.analysis.survival import SurvivalAnalyzer

    sa = SurvivalAnalyzer()

    # Cox PH (use original scale for interpretable coefficients)
    print("\n  --- Cox Proportional Hazards ---")
    try:
        cox = sa.cox_ph(df, time_col, outcome_col, treatment_col, covariate_cols)
        print(f"  Hazard Ratio     : {cox.hazard_ratio:.4f}")
        print(f"  95% CI           : [{cox.ci_lower:.4f}, {cox.ci_upper:.4f}]")
        print(f"  p-value          : {cox.p_value:.4e}")
        print(f"  Concordance index: {cox.concordance:.4f}")
        results["cox_hr"] = cox.hazard_ratio
    except Exception as e:
        print(f"  Cox PH failed: {e}")
        results["cox_hr"] = None

    # Kaplan-Meier + RMST
    print("\n  --- Restricted Mean Survival Time (RMST) ---")
    try:
        rmst = sa.rmst(df, time_col, outcome_col, treatment_col)
        print(f"  RMST (treated)   : {rmst.rmst_treatment:.2f} months")
        print(f"  RMST (control)   : {rmst.rmst_control:.2f} months")
        print(f"  RMST difference  : {rmst.rmst_difference:.2f} months")
        print(f"  95% CI           : [{rmst.ci_lower:.2f}, {rmst.ci_upper:.2f}]")
        print(f"  tau              : {rmst.tau:.2f} months")
        results["rmst_diff"] = rmst.rmst_difference
    except Exception as e:
        print(f"  RMST failed: {e}")
        results["rmst_diff"] = None

    # KM median survival
    print("\n  --- Kaplan-Meier Median Survival ---")
    try:
        km = sa.kaplan_meier(df, time_col, outcome_col, group_col=treatment_col)
        for grp, res in km.items():
            med = f"{res.median_survival:.2f}" if res.median_survival else "NR"
            print(f"  Group {grp}: median = {med} months")
    except Exception as e:
        print(f"  KM failed: {e}")

    # ==============================================================
    # 7. Cost-Effectiveness
    # ==============================================================
    section("7. COST-EFFECTIVENESS ANALYSIS")
    from backend.models.cost_effectiveness import CostEffectivenessAnalyzer

    cea = CostEffectivenessAnalyzer(wtp=50000)
    try:
        ce = cea.analyze(df, treatment_col, cost_col, effect_col, n_bootstrap=500)
        print(f"  WTP threshold    : {ce.wtp:,.0f} (k CNY/QALY proxy)")
        print(f"  Delta cost       : {ce.delta_cost:,.1f} k CNY")
        print(f"  Delta effect     : {ce.delta_effect:,.2f} (QoL)")
        print(f"  ICER             : {ce.icer:,.1f} k CNY per QoL point")
        print(f"  NMB              : {ce.nmb:,.1f} k CNY")
        print(f"  P(cost-effective): {ce.prob_cost_effective:.3f}")
        if ce.bootstrap_ci:
            print(f"  NMB 95% CI       : [{ce.bootstrap_ci[0]:,.1f}, {ce.bootstrap_ci[1]:,.1f}]")
        results["icer"] = ce.icer
        results["nmb"] = ce.nmb
        results["prob_ce"] = ce.prob_cost_effective
    except Exception as e:
        print(f"  Cost-effectiveness failed: {e}")
        results["icer"] = None

    # ==============================================================
    # 8. Sensitivity Analysis
    # ==============================================================
    section("8. SENSITIVITY ANALYSIS")
    from backend.analysis.sensitivity import SensitivityAnalyzer

    senz = SensitivityAnalyzer()

    # E-value for the HR
    print("\n  --- E-value (unmeasured confounding) ---")
    if results.get("cox_hr") is not None:
        hr = results["cox_hr"]
        # E-value is for risk ratios; convert HR if needed
        rr = hr if hr >= 1 else 1 / hr
        ev = senz.e_value(rr)
        print(f"  {ev.interpretation}")
        results["e_value"] = ev.e_value_point
    else:
        print("  (Skipped - no HR available)")

    # Tipping point analysis
    print("\n  --- Tipping Point Analysis ---")
    tip = senz.tipping_point(
        observed_estimate=-np.log(results.get("cox_hr", 0.70)),
        se=0.05,
        prevalence_exposure=0.3,
    )
    print(f"  {tip.interpretation}")
    results["tipping_gamma"] = tip.tipping_gamma

    # ==============================================================
    # 9. Summary Table
    # ==============================================================
    section("COMPREHENSIVE RESULTS SUMMARY")
    elapsed = time.time() - t0
    print(f"""
  Dataset
  -------
  N patients              : {n:,}
  Treatment (new drug)    : {df['treatment'].sum():,} ({df['treatment'].mean()*100:.1f}%)
  TCM adjunctive          : {df['tcm_therapy'].sum():,} ({df['tcm_therapy'].mean()*100:.1f}%)
  Events (death)          : {df['death'].sum():,} ({df['death'].mean()*100:.1f}%)
  True HR (data-generating): 0.700

  Propensity Score & Balance
  --------------------------
  Max |SMD| before IPTW   : {results.get('max_smd_before', 0):.4f}
  Max |SMD| after IPTW    : {results.get('max_smd_after', 0):.4f}
  Overall balance         : {results.get('balance', 'N/A').upper()}

  Causal Estimates (ATE on death risk)
  -------------------------------------
  IPW                     : {fmt(results.get('ipw_est'), 4)}
  AIPW                    : {fmt(results.get('aipw_est'), 4)}
  TMLE                    : {fmt(results.get('tmle_est'), 4)}

  Survival Analysis
  -----------------
  Cox HR (treatment)      : {fmt(results.get('cox_hr'), 4)}
  RMST difference (months): {fmt(results.get('rmst_diff'), 2)}

  Cost-Effectiveness
  ------------------
  ICER (k CNY / QoL pt)  : {fmt(results.get('icer'), 1)}
  NMB (k CNY)             : {fmt(results.get('nmb'), 1)}
  P(cost-effective)       : {fmt(results.get('prob_ce'), 3)}

  Sensitivity
  -----------
  E-value                 : {fmt(results.get('e_value'), 3)}
  Tipping-point gamma     : {fmt(results.get('tipping_gamma'), 2)}

  Wall-clock time         : {elapsed:.1f}s
""")

    print(hr_line())
    print("  Pipeline complete.")
    print(hr_line())

    return results


# ======================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full TCM-TargetTrial-RWE analysis")
    parser.add_argument("-n", type=int, default=3000, help="Number of patients")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_full_analysis(n=args.n, seed=args.seed)
