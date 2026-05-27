"""Tests for survival analysis."""
import numpy as np
import pandas as pd
import pytest
from backend.analysis.survival import SurvivalAnalyzer


def _make_surv(n=500, seed=42):
    rng = np.random.RandomState(seed)
    t = rng.binomial(1, 0.5, n)
    base = 0.02
    h = base * np.exp(-0.4 * t)
    time = rng.exponential(1 / h)
    event = rng.binomial(1, 0.8, n)
    return pd.DataFrame({"time": np.clip(time, 0.1, None), "event": event,
                          "treatment": t, "age": rng.normal(60, 10, n),
                          "stage": rng.choice([1, 2, 3], n)})


class TestKaplanMeier:
    def test_km_runs(self):
        df = _make_surv()
        sa = SurvivalAnalyzer()
        r = sa.kaplan_meier(df, "time", "event", "treatment")
        assert "0" in r and "1" in r
        for g, km in r.items():
            assert len(km.times) > 0
            assert all(np.diff(km.survival_function) <= 0.01)

    def test_km_overall(self):
        df = _make_surv()
        r = SurvivalAnalyzer().kaplan_meier(df, "time", "event")
        assert "all" in r
        assert r["all"].median_survival is not None


class TestCoxPH:
    def test_cox_runs(self):
        df = _make_surv(800)
        r = SurvivalAnalyzer().cox_ph(df, "time", "event", "treatment", ["age", "stage"])
        assert 0 < r.hazard_ratio < 2
        assert 0.5 <= r.concordance <= 1.0

    def test_hr_direction(self):
        df = _make_surv(2000, 42)
        r = SurvivalAnalyzer().cox_ph(df, "time", "event", "treatment")
        assert r.hazard_ratio < 1.0


class TestRMST:
    def test_rmst_runs(self):
        df = _make_surv(500)
        r = SurvivalAnalyzer().rmst(df, "time", "event", "treatment")
        assert r.rmst_treatment > 0 and r.rmst_control > 0 and r.tau > 0
