"""Sensitivity analysis for unmeasured confounding: E-value and tipping point."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EValueResult:
    point_estimate: float
    e_value_point: float
    e_value_ci: Optional[float]
    interpretation: str


@dataclass
class TippingPointResult:
    tipping_gamma: float
    tipping_rr: float
    interpretation: str
    gamma_grid: np.ndarray
    adjusted_estimates: np.ndarray


class SensitivityAnalyzer:
    def e_value(self, estimate, ci_lower=None, ci_upper=None, rare_outcome=True):
        if estimate == 0:
            return EValueResult(0, float("inf"), None,
                                "E-value is undefined for a null point estimate (RR=0).")
        rr = abs(estimate) if abs(estimate) >= 1 else 1 / abs(estimate)
        ev = rr + np.sqrt(rr * (rr - 1))
        ev_ci = None
        if ci_lower is not None and abs(ci_lower) > 1e-12:
            rr_ci = abs(ci_lower) if abs(ci_lower) >= 1 else 1 / abs(ci_lower)
            ev_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
        parts = [
            f"E-value for point estimate: {ev:.3f}.",
            f"An unmeasured confounder associated with both treatment and outcome",
            f"by a risk ratio of at least {ev:.3f} each could explain away the observed association.",
        ]
        if ev_ci:
            parts.append(f"E-value for CI bound: {ev_ci:.3f}.")
        return EValueResult(estimate, ev, ev_ci, " ".join(parts))

    def tipping_point(self, observed_estimate, se, prevalence_exposure=0.3,
                      gamma_range=(1.0, 5.0), n_points=100,
                      confounder_prev_range=(0.1, 0.5)):
        gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)
        adjusted = np.empty(n_points)
        p_c = confounder_prev_range[1]
        for i, g in enumerate(gammas):
            bias = p_c * (g - 1) + 1
            adjusted[i] = observed_estimate - np.log(bias)
        tipping_gamma = gammas[-1]
        for i in range(len(adjusted) - 1):
            if adjusted[i] >= 0 and adjusted[i + 1] < 0:
                tipping_gamma = gammas[i]
                break
        interp = (f"The treatment effect would be nullified if an unmeasured confounder "
                  f"increased the odds of treatment and outcome by a factor of {tipping_gamma:.2f}. "
                  f"If no confounder of this strength is plausible, the result is robust.")
        return TippingPointResult(float(tipping_gamma), float(tipping_gamma), interp, gammas, adjusted)
