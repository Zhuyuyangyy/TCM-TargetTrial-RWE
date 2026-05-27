"""Tests for causal inference engines."""
import numpy as np
import pandas as pd
import pytest
from backend.models.causal_engine import IPW, AIPW, CAE, CausalEstimate


def _make_data(n=2000, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    ps = 1 / (1 + np.exp(-(0.5*X1 + 0.3*X2)))
    t = rng.binomial(1, ps)
    true_eff = 2.0
    y = 1.0 + true_eff*t + 0.8*X1 + 0.5*X2 + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": t, "outcome": y, "X1": X1, "X2": X2}), true_eff


class TestIPW:
    def test_recovers_effect(self):
        df, te = _make_data(3000, 42)
        r = IPW().estimate(df, "treatment", "outcome", ["X1", "X2"])
        assert abs(r.estimate - te) < 0.5
        assert r.method == "IPW"

    def test_ci_contains_truth(self):
        df, te = _make_data(3000, 123)
        r = IPW().estimate(df, "treatment", "outcome", ["X1", "X2"])
        assert r.ci_lower <= te <= r.ci_upper

    def test_weights_summary(self):
        df, _ = _make_data()
        r = IPW().estimate(df, "treatment", "outcome", ["X1", "X2"])
        assert r.weights_summary is not None
        assert r.weights_summary["mean"] > 0


class TestAIPW:
    def test_recovers_effect(self):
        df, te = _make_data(3000, 42)
        r = AIPW().estimate(df, "treatment", "outcome", ["X1", "X2"])
        assert abs(r.estimate - te) < 0.5
        assert r.method == "AIPW"

    def test_ci(self):
        df, te = _make_data(3000, 456)
        r = AIPW().estimate(df, "treatment", "outcome", ["X1", "X2"])
        assert r.ci_lower < r.estimate < r.ci_upper


class TestCAE:
    def test_runs(self):
        df, _ = _make_data()
        r = CAE().estimate(df, "treatment", "outcome", ["X1", "X2"])
        assert isinstance(r, CausalEstimate)
