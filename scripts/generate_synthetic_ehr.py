#!/usr/bin/env python3
"""Generate synthetic EHR data for TCM-TargetTrial-RWE testing.
Creates 2000 patients with realistic NSCLC treatment patterns."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_synthetic_ehr(n_patients=2000, seed=42, output="data/synthetic_ehr.csv"):
    rng = np.random.RandomState(seed)
    print(f"Generating synthetic EHR data for {n_patients} patients...")
    age = rng.normal(62, 10, n_patients).clip(30, 85).astype(int)
    sex = rng.binomial(1, 0.6, n_patients)
    smoking = rng.choice(["never", "former", "current"], n_patients, p=[0.15, 0.55, 0.30])
    bmi = rng.normal(24, 4, n_patients).clip(15, 45)
    histology = rng.choice(["adenocarcinoma", "squamous", "large_cell", "other"],
                           n_patients, p=[0.55, 0.30, 0.10, 0.05])
    stage = rng.choice([2, 3], n_patients, p=[0.55, 0.45])
    egfr = rng.binomial(1, 0.3, n_patients)
    pd_l1 = rng.exponential(25, n_patients).clip(0, 100)
    ki67 = rng.beta(2, 5, n_patients) * 100
    albumin = rng.normal(38, 5, n_patients).clip(20, 55)
    creat = rng.normal(80, 25, n_patients).clip(30, 200)
    lymph = rng.normal(1.8, 0.6, n_patients).clip(0.3, 5.0)
    neutro = rng.normal(5.0, 2.0, n_patients).clip(1.0, 15.0)
    nlr = neutro / lymph
    ef = rng.normal(62, 8, n_patients).clip(35, 80)
    hb = rng.normal(130, 15, n_patients).clip(80, 180)

    # Confounded treatment assignment
    p_logit = (-1.5 + 0.8*(stage==2).astype(float) - 0.03*(age-60)
               + 0.3*(albumin>35).astype(float) - 0.2*(nlr<3).astype(float)
               + 0.15*(sex==0).astype(float))
    prop = 1 / (1 + np.exp(-p_logit))
    tcm = rng.binomial(1, prop)

    surgery = rng.choice(["lobectomy", "pneumonectomy", "segmentectomy"],
                         n_patients, p=[0.70, 0.15, 0.15])
    chemo = rng.choice(["cisplatin_pemetrexed", "carboplatin_paclitaxel", "cisplatin_gemcitabine"],
                       n_patients, p=[0.45, 0.35, 0.20])
    cycles = rng.poisson(4, n_patients).clip(1, 6)
    completed = (cycles >= 4).astype(int)
    hospital = rng.choice(["Hospital_A", "Hospital_B", "Hospital_C"], n_patients, p=[0.4, 0.35, 0.25])
    year = rng.choice(range(2015, 2025), n_patients)

    # Outcomes: true HR ~ 0.70
    base_h = 0.015
    lp = (np.log(base_h) + np.log(0.70)*tcm + 0.02*(age-60)
          + 0.3*(stage==3).astype(float) + 0.15*(smoking=="current").astype(float).astype(float)
          - 0.01*(albumin-38) + 0.05*np.log(np.clip(nlr, 0.5, None)))
    surv_time = rng.exponential(np.exp(-lp))
    obs_time = np.minimum(surv_time, 60)
    death = (surv_time <= 60).astype(int)

    dfs_h = base_h * 1.5
    dfs_t = rng.exponential(1/(dfs_h*np.exp(np.log(0.70)*tcm + 0.3*(stage==3).astype(float))))
    dfs_months = np.minimum(dfs_t, obs_time)
    recurrence = (dfs_t <= obs_time).astype(int)

    qol = (rng.normal(65, 15, n_patients) + 8*tcm + rng.normal(0, 5, n_patients)).clip(0, 100)
    cd48 = (rng.normal(1.5, 0.5, n_patients) + 0.3*tcm + rng.normal(0, 0.2, n_patients)).clip(0.2, 5.0)
    cost = (rng.normal(80, 20, n_patients) + 15*tcm + 5*cycles + rng.normal(0, 10, n_patients)).clip(10, 500)

    df = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(n_patients)],
        "age": age, "sex": sex, "smoking_status": smoking, "bmi": bmi.round(1),
        "histology": histology, "stage": stage, "egfr_mutation": egfr,
        "pd_l1_tps": pd_l1.round(1), "ki67": ki67.round(1),
        "albumin": albumin.round(1), "creatinine": creat.round(1),
        "lymphocyte_count": lymph.round(2), "neutrophil_count": neutro.round(2),
        "nlr": nlr.round(2), "ejection_fraction": ef.round(1), "hemoglobin": hb.round(1),
        "hospital": hospital, "year_of_diagnosis": year,
        "surgery_type": surgery, "chemo_regimen": chemo, "chemo_cycles": cycles,
        "chemo_completed": completed, "tcm_treatment": tcm,
        "tcm_formula": np.where(tcm==1, "Fuzheng_Guben", "none"),
        "survival_months": obs_time.round(2), "death": death,
        "dfs_months": dfs_months.round(2), "recurrence": recurrence,
        "qol_score": qol.round(1), "cd4_cd8_ratio": cd48.round(2),
        "total_cost_k_cny": cost.round(1),
    })

    # Inject missing
    for col, pct in {"bmi":0.05, "pd_l1_tps":0.12, "ki67":0.15, "albumin":0.03,
                     "nlr":0.04, "ejection_fraction":0.08, "cd4_cd8_ratio":0.10, "qol_score":0.07}.items():
        df.loc[rng.random(n_patients) < pct, col] = np.nan

    out = Path(__file__).resolve().parent.parent / output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(f"  Patients: {n_patients}, TCM: {tcm.sum()} ({tcm.mean()*100:.1f}%)")
    print(f"  Events: {death.sum()}, True HR: 0.70")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=2000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="data/synthetic_ehr.csv")
    args = parser.parse_args()
    generate_synthetic_ehr(args.n, args.seed, args.output)
