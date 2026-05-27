#!/usr/bin/env python3
"""Generate semi-realistic EHR data for NSCLC patients (n=3000).

The data-generating process mimics real-world patterns seen in Chinese hospital
EHRs for non-small cell lung cancer (NSCLC):

  * Age ~ N(62, 10), clipped [40, 80]
  * Stage distribution: I=15%, II=25%, III=35%, IV=25%
  * Treatment (new drug) assignment confounded by age, stage, and performance
    status via a logistic model
  * TCM adjunctive therapy given to ~40% of patients, confounded by stage
  * Survival time from Weibull distribution with true HR=0.70 for treatment
  * Costs follow a log-normal distribution correlated with stage and treatment
  * Lab values have 10% MAR missingness (depend on observed covariates)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _generate_semireal_ehr(n: int = 3000, seed: int = 42,
                           output: str = "data/semireal_ehr.csv") -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1. Demographics
    # ------------------------------------------------------------------
    age = rng.normal(62, 10, n).clip(40, 80).astype(int)
    sex = rng.binomial(1, 0.58, n)  # 58% male
    smoking = rng.choice(["never", "former", "current"], n, p=[0.15, 0.50, 0.35])
    bmi = rng.normal(23.5, 3.5, n).clip(15, 40).round(1)

    # ------------------------------------------------------------------
    # 2. Tumour characteristics
    # ------------------------------------------------------------------
    stage = rng.choice([1, 2, 3, 4], n, p=[0.15, 0.25, 0.35, 0.25])
    histology = rng.choice(
        ["adenocarcinoma", "squamous", "large_cell", "other"],
        n, p=[0.55, 0.30, 0.10, 0.05],
    )
    egfr_mutation = rng.binomial(1, 0.30, n)
    alk_rearrange = rng.binomial(1, 0.07, n)
    pd_l1_tps = rng.exponential(25, n).clip(0, 100).round(1)
    tumor_size_cm = (1.5 + 0.8 * stage + rng.normal(0, 0.8, n)).clip(0.5, 10).round(1)

    # ------------------------------------------------------------------
    # 3. Performance status & comorbidities
    # ------------------------------------------------------------------
    # ECOG 0-1 favourable, 2-3 unfavourable  (correlated with age & stage)
    ecog_base = 0.4 * ((age - 50) / 15) + 0.5 * (stage - 1)
    ecog = (ecog_base + rng.normal(0, 0.6, n)).clip(0, 3).astype(int)
    ecog = np.minimum(ecog, 3)

    charlson_index = (1 + 0.5 * ((age - 50) / 10) + 0.3 * stage
                      + rng.poisson(1, n)).clip(0, 12).astype(int)

    # ------------------------------------------------------------------
    # 4. Lab values (before introducing missingness)
    # ------------------------------------------------------------------
    albumin = (42 - 1.0 * stage + rng.normal(0, 4, n)).clip(20, 55).round(1)
    creatinine = (70 + 0.5 * age + rng.normal(0, 20, n)).clip(30, 250).round(1)
    hemoglobin = (135 - 3.0 * stage + rng.normal(0, 12, n)).clip(70, 180).round(1)
    wbc = (6.0 + 0.5 * stage + rng.normal(0, 2.0, n)).clip(1.5, 20).round(1)
    lymphocyte = (1.8 - 0.1 * stage + rng.normal(0, 0.5, n)).clip(0.3, 5.0).round(2)
    neutrophil = (5.0 + 0.6 * stage + rng.normal(0, 1.5, n)).clip(1.0, 15.0).round(1)
    nlr = (neutrophil / lymphocyte).round(2)
    platelet = (230 + 10 * stage + rng.normal(0, 50, n)).clip(50, 600).astype(int)
    alt = (25 + rng.exponential(10, n)).clip(5, 200).round(1)
    ast = (28 + rng.exponential(12, n)).clip(5, 250).round(1)

    # ------------------------------------------------------------------
    # 5. Treatment assignment (confounded by age, stage, ecog)
    # ------------------------------------------------------------------
    # Logistic model: younger, lower-stage, better ECOG -> more likely to
    # receive the new drug
    logit_treat = (-1.2
                   - 0.03 * (age - 60)
                   - 0.5 * (stage - 2)
                   - 0.4 * ecog
                   + 0.2 * (albumin > 38).astype(float)
                   - 0.15 * (nlr > 3).astype(float)
                   + 0.1 * (sex == 0).astype(float))
    prob_treat = 1 / (1 + np.exp(-logit_treat))
    treatment = rng.binomial(1, prob_treat)

    # ------------------------------------------------------------------
    # 6. TCM adjunctive therapy (~40%, confounded by stage)
    # ------------------------------------------------------------------
    logit_tcm = (-0.3
                 + 0.3 * (stage >= 3).astype(float)
                 + 0.15 * (age > 55).astype(float)
                 - 0.1 * ecog)
    prob_tcm = 1 / (1 + np.exp(-logit_tcm))
    tcm_therapy = rng.binomial(1, prob_tcm)
    tcm_formula = np.where(tcm_therapy == 1,
                           rng.choice(["Fuzheng_Guben", "Yiqi_Yangyin", "Qingre_Jiedu"],
                                      n, p=[0.45, 0.30, 0.25]),
                           "none")

    # ------------------------------------------------------------------
    # 7. Treatment details
    # ------------------------------------------------------------------
    chemo_regimen = rng.choice(
        ["cisplatin_pemetrexed", "carboplatin_paclitaxel", "cisplatin_gemcitabine",
         "docetaxel_monotherapy"],
        n, p=[0.35, 0.30, 0.20, 0.15],
    )
    chemo_cycles = rng.poisson(4, n).clip(1, 8).astype(int)
    chemo_completed = (chemo_cycles >= 4).astype(int)
    immunotherapy = rng.binomial(1, 0.25, n)
    target_therapy = rng.binomial(1, 0.20, n) * egfr_mutation  # only if EGFR+

    # ------------------------------------------------------------------
    # 8. Survival (Weibull, true HR=0.70 for treatment)
    # ------------------------------------------------------------------
    shape = 1.5  # Weibull shape parameter
    # Linear predictor on the log-hazard scale
    lp = (np.log(0.01)
          + np.log(0.70) * treatment
          + np.log(0.80) * tcm_therapy     # small TCM benefit
          + 0.020 * (age - 60)
          + 0.30 * (stage == 2).astype(float)
          + 0.55 * (stage == 3).astype(float)
          + 0.80 * (stage == 4).astype(float)
          + 0.15 * (smoking == "current").astype(float)
          + 0.10 * (ecog >= 2).astype(float)
          - 0.015 * (albumin - 40)
          + 0.04 * np.log(np.clip(nlr, 0.5, None)))
    scale = np.exp(-lp / shape)  # Weibull scale
    surv_time = rng.weibull(shape, n) * scale

    # Administrative censoring at 60 months + random censoring
    admin_censor = 60.0
    random_censor = rng.exponential(80, n)
    censor_time = np.minimum(admin_censor, random_censor)
    observed_time = np.minimum(surv_time, censor_time).clip(0.1)
    death = (surv_time <= censor_time).astype(int)

    # Disease-free survival (shorter)
    dfs_lp = lp + 0.3  # worse
    dfs_scale = np.exp(-dfs_lp / shape)
    dfs_time = rng.weibull(shape, n) * dfs_scale
    dfs_observed = np.minimum(dfs_time, observed_time)
    recurrence = (dfs_time <= observed_time).astype(int)

    # ------------------------------------------------------------------
    # 9. Quality of life (EQ-5D-like score, 0-100)
    # ------------------------------------------------------------------
    qol = (70 - 5 * stage + 6 * treatment + 3 * tcm_therapy
           + rng.normal(0, 12, n)).clip(0, 100).round(1)

    # ------------------------------------------------------------------
    # 10. Costs (log-normal, correlated with stage & treatment)
    # ------------------------------------------------------------------
    log_cost = (9.5 + 0.25 * stage + 0.30 * treatment + 0.10 * tcm_therapy
                + 0.05 * chemo_cycles + rng.normal(0, 0.4, n))
    total_cost_k_cny = np.exp(log_cost).round(1)

    # Separate drug and non-drug costs
    drug_cost_k = (total_cost_k_cny * rng.uniform(0.3, 0.6, n)).round(1)
    hospital_cost_k = (total_cost_k_cny - drug_cost_k).round(1)

    # ------------------------------------------------------------------
    # 11. Hospital & year
    # ------------------------------------------------------------------
    hospital = rng.choice(
        ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"],
        n, p=[0.30, 0.25, 0.25, 0.20],
    )
    year = rng.choice(range(2016, 2025), n)

    # ------------------------------------------------------------------
    # 12. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(n)],
        # Demographics
        "age": age,
        "sex": sex,
        "smoking_status": smoking,
        "bmi": bmi,
        # Tumour
        "stage": stage,
        "histology": histology,
        "egfr_mutation": egfr_mutation,
        "alk_rearrangement": alk_rearrange,
        "pd_l1_tps": pd_l1_tps,
        "tumor_size_cm": tumor_size_cm,
        # Performance & comorbidity
        "ecog_score": ecog,
        "charlson_index": charlson_index,
        # Labs
        "albumin": albumin,
        "creatinine": creatinine,
        "hemoglobin": hemoglobin,
        "wbc": wbc,
        "lymphocyte_count": lymphocyte,
        "neutrophil_count": neutrophil,
        "nlr": nlr,
        "platelet": platelet,
        "alt": alt,
        "ast": ast,
        # Treatment
        "treatment": treatment,
        "chemo_regimen": chemo_regimen,
        "chemo_cycles": chemo_cycles,
        "chemo_completed": chemo_completed,
        "immunotherapy": immunotherapy,
        "target_therapy": target_therapy,
        # TCM
        "tcm_therapy": tcm_therapy,
        "tcm_formula": tcm_formula,
        # Outcomes
        "survival_months": observed_time.round(2),
        "death": death,
        "dfs_months": dfs_observed.round(2),
        "recurrence": recurrence,
        "qol_score": qol,
        # Costs
        "total_cost_k_cny": total_cost_k_cny,
        "drug_cost_k_cny": drug_cost_k,
        "hospital_cost_k_cny": hospital_cost_k,
        # Metadata
        "hospital": hospital,
        "year_of_diagnosis": year,
    })

    # ------------------------------------------------------------------
    # 13. Inject MAR missingness (~10%) for lab values
    #     Missingness depends on observed covariates (MAR mechanism)
    # ------------------------------------------------------------------
    mar_base = rng.uniform(0, 1, n)
    for col in ["albumin", "creatinine", "hemoglobin", "wbc",
                "lymphocyte_count", "neutrophil_count", "nlr",
                "platelet", "alt", "ast"]:
        # MAR: higher missingness for older / higher-stage patients
        mar_prob = 0.08 + 0.003 * (age - 50) / 10 + 0.01 * (stage - 2)
        mar_prob = np.clip(mar_prob, 0.02, 0.20)
        miss_mask = rng.uniform(0, 1, n) < mar_prob
        df.loc[miss_mask, col] = np.nan

    # Also ~5% missing for pd_l1_tps (often not ordered)
    df.loc[rng.random(n) < 0.08, "pd_l1_tps"] = np.nan

    # ------------------------------------------------------------------
    # 14. Save
    # ------------------------------------------------------------------
    out = Path(__file__).resolve().parent.parent / output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Summary
    print(f"Semi-realistic NSCLC EHR data generated: {out}")
    print(f"  N patients       : {n}")
    print(f"  Treatment (new)  : {treatment.sum()} ({treatment.mean()*100:.1f}%)")
    print(f"  TCM adjunctive   : {tcm_therapy.sum()} ({tcm_therapy.mean()*100:.1f}%)")
    print(f"  Deaths observed  : {death.sum()} ({death.mean()*100:.1f}%)")
    print(f"  Mean surv months : {observed_time.mean():.1f}")
    print(f"  True HR (treat)  : 0.70")
    print(f"  Stage dist       : I={np.mean(stage==1)*100:.0f}%, II={np.mean(stage==2)*100:.0f}%, "
          f"III={np.mean(stage==3)*100:.0f}%, IV={np.mean(stage==4)*100:.0f}%")
    print(f"  Missing labs     : ~{df[['albumin','hemoglobin','nlr']].isna().mean().mean()*100:.0f}%")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate semi-realistic NSCLC EHR data")
    parser.add_argument("-n", type=int, default=3000, help="Number of patients")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-o", "--output", default="data/semireal_ehr.csv", help="Output CSV path")
    args = parser.parse_args()
    _generate_semireal_ehr(args.n, args.seed, args.output)
