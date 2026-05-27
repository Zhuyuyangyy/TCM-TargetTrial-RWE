"""Diagnostic tools for target trial emulation quality assessment.

Provides:
  - CovariateBalanceDiag: compute SMD before and after weighting
  - LovePlotData: generate data structure for Love plot visualization
  - PropensityDistribution: compare PS distributions between treatment arms
  - diagnose(): convenience function returning a full diagnostic report
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BalanceRow:
    covariate: str
    smd_before: float
    smd_after: float
    variance_ratio_before: float
    variance_ratio_after: float


@dataclass
class CovariateBalanceDiag:
    """Compute standardised mean differences (SMD) before and after weighting.

    SMD = |mean(X_treated) - mean(X_control)| / sqrt((var_t + var_c)/2)

    A well-balanced covariate has |SMD| < 0.1 after weighting.
    """
    covariates: list[str]
    balance_rows: list[BalanceRow] = field(default_factory=list)
    max_smd_before: float = 0.0
    max_smd_after: float = 0.0
    overall: str = ""

    @classmethod
    def compute(cls, df: pd.DataFrame, treatment_col: str,
                covariate_cols: list[str],
                weights: Optional[np.ndarray] = None) -> "CovariateBalanceDiag":
        t = df[treatment_col].values.astype(bool)
        w = weights if weights is not None else np.ones(len(df))
        rows: list[BalanceRow] = []
        for col in covariate_cols:
            x = df[col].values.astype(float)
            mask = ~np.isnan(x)
            if mask.sum() < 10:
                continue
            x_m, t_m, w_m = x[mask], t[mask], w[mask]
            # Before weighting (all weights = 1)
            m1_b = np.mean(x_m[t_m])
            m0_b = np.mean(x_m[~t_m])
            s_b = np.sqrt((np.var(x_m[t_m]) + np.var(x_m[~t_m])) / 2)
            smd_b = abs(m1_b - m0_b) / s_b if s_b > 1e-12 else 0.0
            vr_b = np.var(x_m[t_m]) / np.var(x_m[~t_m]) if np.var(x_m[~t_m]) > 1e-12 else 1.0
            # After weighting
            w1, w0 = w_m[t_m], w_m[~t_m]
            m1_a = np.average(x_m[t_m], weights=w1)
            m0_a = np.average(x_m[~t_m], weights=w0)
            v1_a = np.average((x_m[t_m] - m1_a) ** 2, weights=w1)
            v0_a = np.average((x_m[~t_m] - m0_a) ** 2, weights=w0)
            s_a = np.sqrt((v1_a + v0_a) / 2)
            smd_a = abs(m1_a - m0_a) / s_a if s_a > 1e-12 else 0.0
            vr_a = v1_a / v0_a if v0_a > 1e-12 else 1.0
            rows.append(BalanceRow(col, smd_b, smd_a, vr_b, vr_a))
        max_b = max((r.smd_before for r in rows), default=0)
        max_a = max((r.smd_after for r in rows), default=0)
        overall = "good" if max_a < 0.1 else ("adequate" if max_a < 0.25 else "poor")
        return cls(
            covariates=covariate_cols,
            balance_rows=rows,
            max_smd_before=max_b,
            max_smd_after=max_a,
            overall=overall,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"covariate": r.covariate,
             "smd_before": r.smd_before,
             "smd_after": r.smd_after,
             "vr_before": r.variance_ratio_before,
             "vr_after": r.variance_ratio_after,
             "balanced": r.smd_after < 0.1}
            for r in self.balance_rows
        ])

    def summary(self) -> str:
        df = self.to_dataframe()
        n_bal = df["balanced"].sum()
        n_total = len(df)
        lines = [
            "=" * 60,
            "COVARIATE BALANCE DIAGNOSTIC",
            "=" * 60,
            f"Overall balance : {self.overall.upper()}",
            f"Max |SMD| before: {self.max_smd_before:.4f}",
            f"Max |SMD| after : {self.max_smd_after:.4f}",
            f"Balanced (|SMD|<0.1): {n_bal}/{n_total} covariates",
            "-" * 60,
            f"{'Covariate':<25s} {'SMD before':>10s} {'SMD after':>10s} {'OK?':>5s}",
            "-" * 60,
        ]
        for _, row in df.iterrows():
            ok = "Yes" if row["balanced"] else "NO"
            lines.append(f"{row['covariate']:<25s} {row['smd_before']:>10.4f} "
                         f"{row['smd_after']:>10.4f} {ok:>5s}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LovePlotData:
    """Data for a Love plot (dot plot of |SMD| before/after weighting).

    The Love plot is the standard visual diagnostic for covariate balance.
    Points to the right of the dashed line (|SMD|=0.1) indicate imbalance.
    """
    covariates: list[str]
    smd_before: np.ndarray
    smd_after: np.ndarray
    threshold: float = 0.1

    @classmethod
    def from_balance(cls, balance: CovariateBalanceDiag,
                     threshold: float = 0.1) -> "LovePlotData":
        covs = [r.covariate for r in balance.balance_rows]
        sb = np.array([r.smd_before for r in balance.balance_rows])
        sa = np.array([r.smd_after for r in balance.balance_rows])
        return cls(covariates=covs, smd_before=sb, smd_after=sa, threshold=threshold)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "covariate": self.covariates,
            "smd_before": self.smd_before,
            "smd_after": self.smd_after,
        })

    def summary(self) -> str:
        lines = [
            "LOVE PLOT DATA",
            f"Threshold: |SMD| = {self.threshold}",
            f"{'Covariate':<25s} {'Before':>10s} {'After':>10s} {'Flag':>5s}",
            "-" * 55,
        ]
        for c, b, a in zip(self.covariates, self.smd_before, self.smd_after):
            flag = "***" if a > self.threshold else ""
            lines.append(f"{c:<25s} {b:>10.4f} {a:>10.4f} {flag:>5s}")
        return "\n".join(lines)


@dataclass
class PropensityDistribution:
    """Compare propensity score distributions between treatment arms.

    Checks overlap (region of common support) and extreme weights.
    """
    ps_treated: np.ndarray
    ps_control: np.ndarray
    overlap_min: float = 0.0
    overlap_max: float = 1.0
    pct_trimmed_treated: float = 0.0
    pct_trimmed_control: float = 0.0
    ks_statistic: float = 0.0

    @classmethod
    def compute(cls, ps: np.ndarray, treatment: np.ndarray,
                trim_bounds: tuple[float, float] = (0.01, 0.99)) -> "PropensityDistribution":
        ps_t = ps[treatment.astype(bool)]
        ps_c = ps[~treatment.astype(bool)]
        overlap_min = max(ps_t.min(), ps_c.min())
        overlap_max = min(ps_t.max(), ps_c.max())
        pct_trim_t = 100 * np.mean((ps_t < trim_bounds[0]) | (ps_t > trim_bounds[1]))
        pct_trim_c = 100 * np.mean((ps_c < trim_bounds[0]) | (ps_c > trim_bounds[1]))
        # Two-sample KS statistic
        all_ps = np.concatenate([ps_t, ps_c])
        all_labels = np.concatenate([np.ones(len(ps_t)), np.zeros(len(ps_c))])
        order = np.argsort(all_ps)
        sorted_ps = all_ps[order]
        sorted_labels = all_labels[order]
        ecdf_t = np.cumsum(sorted_labels == 1) / len(ps_t)
        ecdf_c = np.cumsum(sorted_labels == 0) / len(ps_c)
        ks = float(np.max(np.abs(ecdf_t - ecdf_c)))
        return cls(
            ps_treated=ps_t, ps_control=ps_c,
            overlap_min=float(overlap_min), overlap_max=float(overlap_max),
            pct_trimmed_treated=float(pct_trim_t),
            pct_trimmed_control=float(pct_trim_c),
            ks_statistic=ks,
        )

    def summary(self) -> str:
        lines = [
            "PROPENSITY SCORE DISTRIBUTION DIAGNOSTIC",
            "-" * 55,
            f"Treated PS  : mean={np.mean(self.ps_treated):.4f}, "
            f"sd={np.std(self.ps_treated):.4f}, "
            f"range=[{np.min(self.ps_treated):.3f}, {np.max(self.ps_treated):.3f}]",
            f"Control PS  : mean={np.mean(self.ps_control):.4f}, "
            f"sd={np.std(self.ps_control):.4f}, "
            f"range=[{np.min(self.ps_control):.3f}, {np.max(self.ps_control):.3f}]",
            f"Common support: [{self.overlap_min:.4f}, {self.overlap_max:.4f}]",
            f"KS statistic: {self.ks_statistic:.4f} "
            f"({'good overlap' if self.ks_statistic < 0.25 else 'POOR overlap - check positivity'})",
            f"Extreme PS trimmed: treated={self.pct_trimmed_treated:.1f}%, "
            f"control={self.pct_trimmed_control:.1f}%",
        ]
        return "\n".join(lines)


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report combining all checks."""
    balance_before: CovariateBalanceDiag
    balance_after: CovariateBalanceDiag
    love_plot: LovePlotData
    ps_dist: PropensityDistribution
    effective_sample_size: float = 0.0
    weight_stability: float = 0.0  # ratio of max to min weight

    def summary(self) -> str:
        sections = [
            self.balance_after.summary(),
            "",
            self.love_plot.summary(),
            "",
            self.ps_dist.summary(),
            "",
            "SUMMARY STATISTICS",
            "-" * 55,
            f"Effective sample size (n_eff): {self.effective_sample_size:.0f}",
            f"Weight stability (max/min)   : {self.weight_stability:.2f}",
            f"Balance status               : {self.balance_after.overall.upper()}",
        ]
        if self.balance_after.overall == "poor":
            sections.append("WARNING: Poor covariate balance after weighting.")
            sections.append("Consider: (1) different PS model, (2) trimming, "
                            "(3) overlap weights, (4) checking positivity violations.")
        if self.ks_statistic > 0.25:
            sections.append("WARNING: Large PS distribution difference (KS > 0.25).")
            sections.append("Treatment populations may not overlap well.")
        return "\n".join(sections)

    @property
    def ks_statistic(self) -> float:
        return self.ps_dist.ks_statistic


def diagnose(df: pd.DataFrame, treatment_col: str, covariate_cols: list[str],
             weights: np.ndarray, ps: np.ndarray) -> DiagnosticReport:
    """Run full diagnostic suite on a weighted analysis.

    Parameters
    ----------
    df : DataFrame with patient data
    treatment_col : binary treatment column
    covariate_cols : list of covariate column names
    weights : IPW / overlap weights
    ps : propensity scores

    Returns
    -------
    DiagnosticReport with balance, Love plot, PS distribution, and summary.
    """
    t = df[treatment_col].values.astype(float)

    # Balance before weighting
    bal_before = CovariateBalanceDiag.compute(df, treatment_col, covariate_cols,
                                               weights=None)
    # Balance after weighting
    bal_after = CovariateBalanceDiag.compute(df, treatment_col, covariate_cols,
                                              weights=weights)
    # Love plot data
    love = LovePlotData.from_balance(bal_after)

    # PS distribution
    ps_dist = PropensityDistribution.compute(ps, t)

    # Effective sample size
    w = weights
    n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))
    w_stab = float(np.max(w) / np.min(w)) if np.min(w) > 0 else float("inf")

    return DiagnosticReport(
        balance_before=bal_before,
        balance_after=bal_after,
        love_plot=love,
        ps_dist=ps_dist,
        effective_sample_size=n_eff,
        weight_stability=w_stab,
    )
