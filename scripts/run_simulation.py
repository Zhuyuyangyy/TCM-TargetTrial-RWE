#!/usr/bin/env python3
"""Run a complete target trial emulation simulation using synthetic EHR data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.trial_protocol import TargetTrialProtocol
from backend.analysis.target_trial import TargetTrialEmulator
from backend.analysis.survival import SurvivalAnalyzer
from backend.analysis.sensitivity import SensitivityAnalyzer
from backend.models.cost_effectiveness import CostEffectivenessAnalyzer
from backend.data.missing_handler import MICEHandler


def main():
    sep = "=" * 70
    print(sep)
    print("TCM-TargetTrial-RWE: Full Simulation Pipeline")
    print(sep)

    # 1. Load protocol
    print("")
    print("[1/7] Loading trial protocol...")
    proto_path = Path(__file__).resolve().parent.parent / "data" / "trial_protocols.yaml"
    protocol = TargetTrialProtocol.from_yaml(str(proto_path))
    print(f"  Protocol: {protocol.title}")
    print(f"  ID: {protocol.trial_id}")

    # 2. Load data
    print("")
    print("[2/7] Loading EHR data...")
    data_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_ehr.csv"
    if not data_path.exists():
        print("  Generating synthetic data...")
        from scripts.generate_synthetic_ehr import generate_synthetic_ehr
        generate_synthetic_ehr()
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} patients")

    # 3. Handle missing
    print("")
    print("[3/7] Handling missing data (MICE)...")
    mice = MICEHandler()
    diag = mice.diagnose(df)
    print(f"  Missing: {diag['total_missing']} ({diag['pct_missing']:.1f}%)")
    num_cols = ["bmi", "pd_l1_tps", "ki67", "albumin", "nlr", "ejection_fraction",
                "cd4_cd8_ratio", "qol_score"]
    df_imp = mice.impute_single(df, num_cols)
    print(f"  Remaining NaN: {df_imp[num_cols].isnull().sum().sum()}")

    # 4. Target trial emulation
    print("")
    print("[4/7] Target Trial Emulation...")
    covs = ["age", "sex", "stage", "egfr_mutation", "pd_l1_tps",
            "albumin", "nlr", "ejection_fraction"]
    df_imp["smoking_current"] = (df_imp["smoking_status"] == "current").astype(int)

    emulator = TargetTrialEmulator(protocol)
    result = emulator.emulate(df_imp, "tcm_treatment", "death", covs, "survival_months")
    print(f"  Eligible: {result.n_eligible} (TCM: {result.n_treated}, Ctrl: {result.n_control})")
    print(f"  ATE: {result.causal_estimate.estimate:.4f}")
    ci_lo = result.causal_estimate.ci_lower
    ci_hi = result.causal_estimate.ci_upper
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    if result.cox_result:
        hr = result.cox_result.hazard_ratio
        hr_lo = result.cox_result.ci_lower
        hr_hi = result.cox_result.ci_upper
        print(f"  Cox HR: {hr:.3f} ({hr_lo:.3f}-{hr_hi:.3f})")
    if result.covariate_balance:
        print(f"  Max SMD: {max(result.covariate_balance.values()):.4f}")

    # 5. Survival analysis
    print("")
    print("[5/7] Survival analysis...")
    sa = SurvivalAnalyzer()
    km = sa.kaplan_meier(df_imp, "survival_months", "death", "tcm_treatment")
    for g, r in km.items():
        label = "TCM" if g == "1" else "Control"
        print(f"  {label}: median OS = {r.median_survival:.1f} months")
    rmst = sa.rmst(df_imp, "survival_months", "death", "tcm_treatment")
    print(f"  RMST diff: {rmst.rmst_difference:.2f} months "
          f"(95% CI: {rmst.ci_lower:.2f}-{rmst.ci_upper:.2f})")

    # 6. Sensitivity
    print("")
    print("[6/7] Sensitivity analysis...")
    sens = SensitivityAnalyzer()
    if result.cox_result:
        ev = sens.e_value(result.cox_result.hazard_ratio,
                          result.cox_result.ci_lower, result.cox_result.ci_upper)
        print(f"  E-value: {ev.e_value_point:.3f}")
    tp = sens.tipping_point(result.causal_estimate.estimate, result.causal_estimate.se)
    print(f"  Tipping point gamma: {tp.tipping_gamma:.3f}")

    # 7. Cost-effectiveness
    print("")
    print("[7/7] Cost-effectiveness...")
    cea = CostEffectivenessAnalyzer(wtp=50000)
    df_ce = df_imp.copy()
    df_ce["cost_cny"] = df_ce["total_cost_k_cny"] * 1000
    ce = cea.analyze(df_ce, "tcm_treatment", "cost_cny", "survival_months")
    print(f"  ICER: {ce.icer:.0f} CNY/month")
    print(f"  NMB: {ce.nmb:.0f} CNY")
    print(f"  P(CE): {ce.prob_cost_effective:.3f}")

    print("")
    print(sep)
    print("Simulation complete!")
    print(sep)


if __name__ == "__main__":
    main()
