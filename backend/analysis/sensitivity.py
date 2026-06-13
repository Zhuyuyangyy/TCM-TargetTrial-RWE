"""Sensitivity analysis for unmeasured confounding: E-value and tipping point.

DISCLAIMER: All results in this module are computed on synthetic or semi-realistic
data generated for method validation purposes only. They do not represent real
clinical findings and should not be used for clinical decision-making.
"""
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


# Valid estimate types for E-value computation.
# The E-value formula (VanderWeele & Ding, 2017) applies to risk ratios (RR >= 1).
# Other effect measures must be converted to RR before computing E-values.
_VALID_EVALUE_TYPES = {"risk_ratio", "odds_ratio", "hazard_ratio"}


class SensitivityAnalyzer:
    def e_value(self, estimate, ci_lower=None, ci_upper=None, rare_outcome=True,
                estimate_type="risk_ratio"):
        """Compute the E-value for a given estimate.

        Parameters
        ----------
        estimate : float
            The point estimate. Must be a positive number.
        ci_lower : float, optional
            Lower bound of the confidence interval.
        ci_upper : float, optional
            Upper bound of the confidence interval.
        rare_outcome : bool
            If True, treats the estimate as approximate RR (rare outcome assumption).
        estimate_type : str
            Type of estimate. Must be one of: 'risk_ratio', 'odds_ratio', 'hazard_ratio'.
            This is used for validation and interpretation. The E-value formula
            (VanderWeele & Ding, 2017) is defined for risk ratios. If an odds_ratio
            or hazard_ratio is provided, a warning is included in the interpretation
            noting that conversion assumptions apply.

        Returns
        -------
        EValueResult

        Raises
        ------
        ValueError
            If estimate_type is not recognized, or estimate is not positive.
        """
        # --- Input validation ---
        if estimate_type not in _VALID_EVALUE_TYPES:
            raise ValueError(
                f"estimate_type must be one of {_VALID_EVALUE_TYPES}, got '{estimate_type}'. "
                f"The E-value formula (VanderWeele & Ding, 2017) applies to risk ratios. "
                f"For mean differences or other effect measures, convert to a risk ratio first."
            )

        if not isinstance(estimate, (int, float)):
            raise TypeError(f"estimate must be numeric, got {type(estimate).__name__}")

        if np.isnan(estimate) or np.isinf(estimate):
            raise ValueError(f"estimate must be a finite number, got {estimate}")

        if estimate <= 0:
            raise ValueError(
                f"E-value requires a positive estimate (RR-scale), got {estimate}. "
                f"The E-value is defined for risk ratios > 0."
            )

        if ci_lower is not None and (np.isnan(ci_lower) or np.isinf(ci_lower)):
            raise ValueError(f"ci_lower must be a finite number, got {ci_lower}")
        if ci_upper is not None and (np.isnan(ci_upper) or np.isinf(ci_upper)):
            raise ValueError(f"ci_upper must be a finite number, got {ci_upper}")

        # --- E-value computation ---
        rr = abs(estimate) if abs(estimate) >= 1 else 1 / abs(estimate)
        ev = rr + np.sqrt(rr * (rr - 1))
        ev_ci = None
        if ci_lower is not None and abs(ci_lower) > 1e-12:
            rr_ci = abs(ci_lower) if abs(ci_lower) >= 1 else 1 / abs(ci_lower)
            ev_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))

        # --- Build interpretation ---
        conversion_note = ""
        if estimate_type in ("odds_ratio", "hazard_ratio"):
            conversion_note = (
                f" Note: the input was provided as a {estimate_type.replace('_', ' ')}; "
                f"the E-value formula assumes a risk ratio. For non-rare outcomes, "
                f"the OR-to-RR or HR-to-RR conversion may introduce bias."
            )

        parts = [
            f"E-value for point estimate: {ev:.3f}.",
            f"An unmeasured confounder associated with both treatment and outcome",
            f"by a risk ratio of at least {ev:.3f} each could explain away the observed association.",
        ]
        if ev_ci:
            parts.append(f"E-value for CI bound: {ev_ci:.3f}.")
        if conversion_note:
            parts.append(conversion_note)
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
