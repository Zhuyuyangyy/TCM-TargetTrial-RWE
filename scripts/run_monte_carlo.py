#!/usr/bin/env python3
"""Monte Carlo simulation for TTE estimator evaluation.

Generates synthetic EHR data with known ground truth (true HR),
runs IPW/AIPW/TMLE estimators across multiple seeds,
and computes bias, coverage, MSE, and statistical power.

Usage:
    python scripts/run_monte_carlo.py
    python scripts/run_monte_carlo.py --n-sims 500 --n-patients 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.causal_engine import IPW, AIPW, TMLE, CAE


def generate_one_dataset(
    n_patients: int,
    true_hr: float,
    seed: int,
    confounding_strength: float = 1.0,
) -> pd.DataFrame:
    """Generate one synthetic EHR dataset with known causal effect.

    Args:
        n_patients: Sample size.
        true_hr: True hazard ratio (treatment effect).
        seed: Random seed.
        confounding_strength: How strongly confounders affect treatment assignment.
    """
    rng = np.random.RandomState(seed)

    age = rng.normal(62, 10, n_patients).clip(30, 85)
    sex = rng.binomial(1, 0.6, n_patients)
    stage = rng.choice([2, 3], n_patients, p=[0.55, 0.45])
    albumin = rng.normal(38, 5, n_patients).clip(20, 55)
    nlr = rng.lognormal(1.0, 0.5, n_patients).clip(0.5, 20)
    pd_l1 = rng.exponential(25, n_patients).clip(0, 100)

    # Confounded treatment assignment
    logit = (
        -1.5
        + confounding_strength * 0.8 * (stage == 2).astype(float)
        - confounding_strength * 0.03 * (age - 60)
        + confounding_strength * 0.3 * (albumin > 35).astype(float)
        - confounding_strength * 0.2 * (nlr < 3).astype(float)
    )
    prop = 1 / (1 + np.exp(-logit))
    treatment = rng.binomial(1, prop)

    # Outcome with known treatment effect
    log_hr = np.log(true_hr)
    lp = (
        log_hr * treatment
        + 0.02 * (age - 60)
        + 0.3 * (stage == 3).astype(float)
        + 0.15 * (sex == 1).astype(float)
        - 0.01 * (albumin - 38)
        + 0.05 * np.log(np.clip(nlr, 0.5, None))
    )
    outcome = rng.normal(0, 1, n_patients) + lp  # Continuous outcome

    return pd.DataFrame({
        "treatment": treatment,
        "outcome": outcome,
        "age": age,
        "sex": sex,
        "stage": stage,
        "albumin": albumin,
        "nlr": nlr,
        "pd_l1": pd_l1,
        "propensity_true": prop,
    })


def run_one_simulation(
    df: pd.DataFrame,
    true_effect: float,
    estimators: Dict[str, object],
) -> Dict[str, Dict]:
    """Run all estimators on one dataset."""
    covariates = ["age", "sex", "stage", "albumin", "nlr", "pd_l1"]
    results = {}

    for name, estimator in estimators.items():
        try:
            est = estimator.estimate(df, "treatment", "outcome", covariates)
            bias = est.estimate - true_effect
            covered = est.ci_lower <= true_effect <= est.ci_upper
            results[name] = {
                "estimate": float(est.estimate),
                "bias": float(bias),
                "se": float(est.se),
                "ci_lower": float(est.ci_lower),
                "ci_upper": float(est.ci_upper),
                "covered": bool(covered),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def compute_metrics(all_results: List[Dict[str, Dict]], true_effect: float) -> Dict:
    """Compute simulation metrics across all runs."""
    metrics = {}
    estimator_names = set()
    for r in all_results:
        estimator_names.update(r.keys())

    for name in sorted(estimator_names):
        estimates = []
        biases = []
        covered = []
        ses = []

        for r in all_results:
            if name in r and "error" not in r[name]:
                estimates.append(r[name]["estimate"])
                biases.append(r[name]["bias"])
                covered.append(r[name]["covered"])
                ses.append(r[name]["se"])

        if not estimates:
            continue

        estimates = np.array(estimates)
        biases = np.array(biases)

        metrics[name] = {
            "n_valid": len(estimates),
            "mean_estimate": float(np.mean(estimates)),
            "empirical_bias": float(np.mean(biases)),
            "relative_bias_pct": float(100 * np.mean(biases) / abs(true_effect)) if true_effect != 0 else 0,
            "empirical_se": float(np.std(estimates)),
            "mean_model_se": float(np.mean(ses)),
            "mse": float(np.mean(biases ** 2)),
            "rmse": float(np.sqrt(np.mean(biases ** 2))),
            "coverage_95": float(np.mean(covered)),
            "power": float(np.mean(np.array(estimates) / np.array(ses) > 1.96)),
        }

    return metrics


def generate_latex_table(metrics: Dict, true_effect: float) -> str:
    """Generate a LaTeX-ready results table."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Monte Carlo Simulation Results (true effect = %.3f)}" % true_effect)
    lines.append(r"\label{tab:simulation}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Estimator & Bias & Rel.\ Bias (\%) & RMSE & SE & Coverage & Power \\")
    lines.append(r"\midrule")

    for name, m in sorted(metrics.items()):
        lines.append(
            f"{name.upper()} & {m['empirical_bias']:.4f} & {m['relative_bias_pct']:.1f} "
            f"& {m['rmse']:.4f} & {m['mean_model_se']:.4f} "
            f"& {m['coverage_95']:.3f} & {m['power']:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_text_report(metrics: Dict, true_effect: float, n_sims: int, n_patients: int) -> str:
    """Generate a text report."""
    lines = [
        "=" * 70,
        "MONTE CARLO SIMULATION RESULTS",
        "=" * 70,
        f"True effect (ATE): {true_effect}",
        f"Sample size: {n_patients}",
        f"Number of simulations: {n_sims}",
        "",
        f"{'Estimator':10s} {'Bias':>8s} {'RelBias%':>9s} {'RMSE':>8s} {'SE':>8s} {'Coverage':>9s} {'Power':>7s}",
        "-" * 70,
    ]

    for name, m in sorted(metrics.items()):
        lines.append(
            f"{name.upper():10s} {m['empirical_bias']:8.4f} {m['relative_bias_pct']:8.1f}% "
            f"{m['rmse']:8.4f} {m['mean_model_se']:8.4f} {m['coverage_95']:8.3f} {m['power']:7.3f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Bias: difference between estimated and true effect")
    lines.append("- Coverage: proportion of 95% CIs containing the true effect (target: 0.95)")
    lines.append("- Power: proportion of simulations rejecting H0 (detecting effect)")
    lines.append("")

    # Best estimator
    best = min(metrics.items(), key=lambda x: x[1]["rmse"])
    lines.append(f"Best estimator by RMSE: {best[0].upper()} (RMSE={best[1]['rmse']:.4f})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo TTE estimator evaluation")
    parser.add_argument("--n-sims", type=int, default=200, help="Number of simulations")
    parser.add_argument("--n-patients", type=int, default=2000, help="Sample size per sim")
    parser.add_argument("--true-hr", type=float, default=0.70, help="True hazard ratio")
    parser.add_argument("--confounding", type=float, default=1.0, help="Confounding strength")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_effect = np.log(args.true_hr)  # Log-scale ATE

    estimators = {
        "cae": CAE(),
        "ipw": IPW(n_bootstrap=200),
        "aipw": AIPW(),
        "tmle": TMLE(),
    }

    print(f"Running {args.n_sims} simulations (n={args.n_patients}, true_HR={args.true_hr})...")

    all_results = []
    for i in range(args.n_sims):
        if (i + 1) % 50 == 0:
            print(f"  Simulation {i+1}/{args.n_sims}")

        df = generate_one_dataset(args.n_patients, args.true_hr, seed=i, confounding_strength=args.confounding)
        results = run_one_simulation(df, true_effect, estimators)
        all_results.append(results)

    metrics = compute_metrics(all_results, true_effect)

    # Generate reports
    text_report = generate_text_report(metrics, true_effect, args.n_sims, args.n_patients)
    print(text_report)

    latex_table = generate_latex_table(metrics, true_effect)

    # Save
    with open(output_dir / "monte_carlo_report.txt", "w") as f:
        f.write(text_report)
    with open(output_dir / "monte_carlo_table.tex", "w") as f:
        f.write(latex_table)
    with open(output_dir / "monte_carlo_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
