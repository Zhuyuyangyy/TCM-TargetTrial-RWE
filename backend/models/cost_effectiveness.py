"""Cost-effectiveness analysis: ICER and Net Monetary Benefit."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class CEResult:
    icer: Optional[float]
    nmb: float
    delta_cost: float
    delta_effect: float
    wtp: float
    method: str
    bootstrap_ci: Optional[tuple[float, float]] = None
    prob_cost_effective: Optional[float] = None


class CostEffectivenessAnalyzer:
    def __init__(self, wtp: float = 50000.0):
        self.wtp = wtp

    def compute_icer(self, cost_t, cost_c, effect_t, effect_c) -> float:
        dc, de = cost_t - cost_c, effect_t - effect_c
        if abs(de) < 1e-10:
            return float("inf") if dc > 0 else float("-inf") if dc < 0 else float("nan")
        return dc / de

    def compute_nmb(self, cost_t, cost_c, effect_t, effect_c) -> float:
        return self.wtp * (effect_t - effect_c) - (cost_t - cost_c)

    def analyze(self, df, treatment_col, cost_col, effect_col, n_bootstrap=1000, seed=42) -> CEResult:
        treat = df[treatment_col].values.astype(bool)
        costs, effects = df[cost_col].values, df[effect_col].values
        c1, c0 = costs[treat].mean(), costs[~treat].mean()
        e1, e0 = effects[treat].mean(), effects[~treat].mean()
        icer = self.compute_icer(c1, c0, e1, e0)
        nmb = self.compute_nmb(c1, c0, e1, e0)
        rng = np.random.RandomState(seed)
        boot_nmb, boot_ce = [], []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(df), len(df), replace=True)
            b = df.iloc[idx]
            bt = b[treatment_col].values.astype(bool)
            bc, be = b[cost_col].values, b[effect_col].values
            bn = self.compute_nmb(bc[bt].mean(), bc[~bt].mean(), be[bt].mean(), be[~bt].mean())
            boot_nmb.append(bn)
            boot_ce.append(bn > 0)
        return CEResult(
            icer=icer, nmb=nmb, delta_cost=c1-c0, delta_effect=e1-e0, wtp=self.wtp,
            method="ICER_NMB_bootstrap",
            bootstrap_ci=(np.percentile(boot_nmb, 2.5), np.percentile(boot_nmb, 97.5)),
            prob_cost_effective=float(np.mean(boot_ce)),
        )
