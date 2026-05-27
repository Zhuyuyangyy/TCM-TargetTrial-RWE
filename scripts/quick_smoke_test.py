"""Quick smoke test: generate synthetic EHR data and run full causal pipeline.

Usage:  python scripts/quick_smoke_test.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0

def ok(name, msg=""):
    global PASS; PASS += 1
    print(f"  [PASS] {name}  {msg}")

def fail(name, err):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {name}  {err}")


# ── synthetic data ───────────────────────────────────────────────────────────
def generate(n=500, seed=42):
    rng = np.random.RandomState(seed)
    age      = rng.normal(60, 10, n)
    sex      = rng.binomial(1, 0.5, n)
    bmi      = rng.normal(25, 4, n)
    comorb   = rng.poisson(2, n).astype(float)
    tcm_hist = rng.binomial(1, 0.4, n).astype(float)

    logit_t = -1.0 + 0.02 * (age - 60) + 0.3 * tcm_hist + 0.15 * comorb
    ps_true = 1 / (1 + np.exp(-logit_t))
    treatment = rng.binomial(1, ps_true).astype(float)

    # True ATE ~ 2.5
    outcome = 30 + 2.5 * treatment + 0.1 * age - 0.5 * sex + 0.3 * bmi               + 0.8 * comorb + rng.normal(0, 5, n)

    return pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi, "comorbidity": comorb,
        "tcm_history": tcm_hist, "treatment": treatment, "outcome": outcome,
    })

COVARS = ["age", "sex", "bmi", "comorbidity", "tcm_history"]


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  TCM-TargetTrial-RWE  —  Smoke Test")
print("=" * 70)

df = generate()
print(f"\n  Synthetic data: {len(df)} rows, treatment rate={df['treatment'].mean():.2%}")
print(f"  Covariates: {COVARS}")
print()

# ── 1. IPW ───────────────────────────────────────────────────────────────────
print("  --- IPW Estimator ---")
try:
    from backend.models.causal_engine import IPW
    r = IPW(stabilize=True).estimate(df, "treatment", "outcome", COVARS)
    ok("IPW.estimate", f"ATE={r.estimate:.3f}  95%CI=({r.ci_lower:.3f}, {r.ci_upper:.3f})")
    assert abs(r.estimate - 2.5) < 2.0, f"ATE {r.estimate} too far from 2.5"
    ok("IPW plausibility", f"|ATE-2.5|={abs(r.estimate-2.5):.2f}")
except Exception as e:
    fail("IPW", e)

# ── 2. AIPW ──────────────────────────────────────────────────────────────────
print("  --- AIPW Estimator ---")
try:
    from backend.models.causal_engine import AIPW
    r = AIPW().estimate(df, "treatment", "outcome", COVARS)
    ok("AIPW.estimate", f"ATE={r.estimate:.3f}  95%CI=({r.ci_lower:.3f}, {r.ci_upper:.3f})")
    assert abs(r.estimate - 2.5) < 2.0, f"ATE {r.estimate} too far from 2.5"
    ok("AIPW plausibility", f"|ATE-2.5|={abs(r.estimate-2.5):.2f}")
except Exception as e:
    fail("AIPW", e)

# ── 3. TMLE ──────────────────────────────────────────────────────────────────
print("  --- TMLE Estimator ---")
try:
    from backend.models.causal_engine import TMLE
    r = TMLE().estimate(df, "treatment", "outcome", COVARS)
    ok("TMLE.estimate", f"ATE={r.estimate:.3f}  95%CI=({r.ci_lower:.3f}, {r.ci_upper:.3f})  SE={r.se:.3f}")
    assert r.method == "TMLE"
    ok("TMLE method label")
    assert abs(r.estimate - 2.5) < 2.0, f"ATE {r.estimate} too far from 2.5"
    ok("TMLE plausibility", f"|ATE-2.5|={abs(r.estimate-2.5):.2f}")
except Exception as e:
    fail("TMLE", e)

# ── 4. Propensity Score / IPTW ───────────────────────────────────────────────
print("  --- PropensityScoreAnalyzer ---")
try:
    from backend.analysis.propensity_score import PropensityScoreAnalyzer
    psa = PropensityScoreAnalyzer()
    result = psa.full_analysis(df, "treatment", COVARS, method="iptw")
    ok("PSA.full_analysis", f"method={result.method}")
    bal = psa.assess_balance(df, "treatment", COVARS, result.weights)
    ok("PSA.balance", f"overall={bal.overall_balance}")
    assert bal.overall_balance in ("good", "adequate"), f"balance={bal.overall_balance}"
    ok("Balance plausibility")
except Exception as e:
    fail("PSA", e)

# ── 5. Overlap Weights ───────────────────────────────────────────────────────
print("  --- OverlapWeightEstimator ---")
try:
    from backend.analysis.propensity_score import OverlapWeightEstimator
    ols = OverlapWeightEstimator()
    res = ols.full_analysis(df, "treatment", "outcome", COVARS)
    ok("Overlap.full_analysis", f"ATE={res['estimate']:.3f}  SE={res['se']:.3f}  balance={res['balance']}")
    assert abs(res["estimate"] - 2.5) < 2.0
    ok("Overlap plausibility", f"|ATE-2.5|={abs(res['estimate']-2.5):.2f}")
    assert "n_effective" in res
    ok("Overlap n_effective", f"n_eff={res['n_effective']:.1f}")
except Exception as e:
    fail("Overlap", e)

# ── 6. CAE ───────────────────────────────────────────────────────────────────
print("  --- CAE (Regression) Estimator ---")
try:
    from backend.models.causal_engine import CAE
    r = CAE().estimate(df, "treatment", "outcome", COVARS)
    ok("CAE.estimate", f"ATE={r.estimate:.3f}  95%CI=({r.ci_lower:.3f}, {r.ci_upper:.3f})")
except Exception as e:
    fail("CAE", e)

# ── 7. FastAPI app + balance endpoint ────────────────────────────────────────
print("  --- FastAPI App + Balance Endpoint ---")
try:
    from backend.main import app
    routes = [r.path for r in app.routes]
    ok("App import", f"{len(routes)} routes")
    has_balance = "/api/analysis/balance" in routes or any("balance" in r for r in routes)
    assert has_balance, f"/balance not in routes: {routes}"
    ok("Balance endpoint registered")
except Exception as e:
    fail("App import / route check", e)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL:
    print("  *** SOME TESTS FAILED ***")
    sys.exit(1)
else:
    print("  *** ALL TESTS PASSED ***")
    sys.exit(0)
