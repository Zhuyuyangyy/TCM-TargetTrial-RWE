"""Analysis API endpoints."""
import io
from typing import Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from backend.models.causal_engine import IPW, AIPW
from backend.analysis.propensity_score import PropensityScoreAnalyzer
from backend.analysis.survival import SurvivalAnalyzer
from backend.analysis.sensitivity import SensitivityAnalyzer
from backend.models.cost_effectiveness import CostEffectivenessAnalyzer

router = APIRouter()
_data_store: dict[str, pd.DataFrame] = {}


class PropensityRequest(BaseModel):
    treatment_col: str
    covariate_cols: list[str]
    model_type: str = "logistic"
    method: str = "iptw"


class CausalRequest(BaseModel):
    treatment_col: str
    outcome_col: str
    covariate_cols: list[str]
    method: str = "aipw"


class SurvivalRequest(BaseModel):
    time_col: str
    event_col: str
    treatment_col: str
    covariate_cols: list[str] = []
    analysis_type: str = "all"


class SensitivityRequest(BaseModel):
    estimate: float
    se: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    analysis_type: str = "both"


class CERequest(BaseModel):
    treatment_col: str
    cost_col: str
    effect_col: str
    wtp: float = 50000.0


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    key = file.filename or "data"
    _data_store[key] = df
    return {"filename": key, "rows": len(df), "columns": list(df.columns)}


@router.post("/propensity")
async def run_propensity(req: PropensityRequest, data_key: str = "data"):
    if data_key not in _data_store:
        raise HTTPException(400, "Upload data first")
    df = _data_store[data_key]
    psa = PropensityScoreAnalyzer(model_type=req.model_type)
    result = psa.full_analysis(df, req.treatment_col, req.covariate_cols, method=req.method)
    balance = psa.assess_balance(df, req.treatment_col, req.covariate_cols, result.weights)
    return {"method": result.method, "balance": balance.overall_balance,
            "smd": {k: round(v, 4) for k, v in balance.standardized_mean_differences.items()}}


@router.post("/causal")
async def run_causal(req: CausalRequest, data_key: str = "data"):
    if data_key not in _data_store:
        raise HTTPException(400, "Upload data first")
    df = _data_store[data_key]
    engine = AIPW() if req.method == "aipw" else IPW()
    r = engine.estimate(df, req.treatment_col, req.outcome_col, req.covariate_cols)
    return {"method": r.method, "estimate": round(r.estimate, 6),
            "ci_lower": round(r.ci_lower, 6), "ci_upper": round(r.ci_upper, 6),
            "se": round(r.se, 6), "n_obs": r.n_obs}


@router.post("/survival")
async def run_survival(req: SurvivalRequest, data_key: str = "data"):
    if data_key not in _data_store:
        raise HTTPException(400, "Upload data first")
    df = _data_store[data_key]
    sa = SurvivalAnalyzer()
    results = {}
    if req.analysis_type in ("km", "all"):
        km = sa.kaplan_meier(df, req.time_col, req.event_col, req.treatment_col)
        results["kaplan_meier"] = {g: {"median_survival": r.median_survival} for g, r in km.items()}
    if req.analysis_type in ("cox", "all"):
        c = sa.cox_ph(df, req.time_col, req.event_col, req.treatment_col, req.covariate_cols)
        results["cox_ph"] = {"hazard_ratio": round(c.hazard_ratio, 4), "ci_lower": round(c.ci_lower, 4),
                             "ci_upper": round(c.ci_upper, 4), "p_value": round(c.p_value, 6)}
    if req.analysis_type in ("rmst", "all"):
        r = sa.rmst(df, req.time_col, req.event_col, req.treatment_col)
        results["rmst"] = {"difference": round(r.rmst_difference, 2), "ci_lower": round(r.ci_lower, 2),
                           "ci_upper": round(r.ci_upper, 2), "tau": round(r.tau, 2)}
    return results


@router.post("/sensitivity")
async def run_sensitivity(req: SensitivityRequest):
    sa = SensitivityAnalyzer()
    results = {}
    if req.analysis_type in ("evalue", "both"):
        ev = sa.e_value(req.estimate, req.ci_lower, req.ci_upper)
        results["e_value"] = {"point_estimate": round(ev.e_value_point, 4),
                              "ci_bound": round(ev.e_value_ci, 4) if ev.e_value_ci else None,
                              "interpretation": ev.interpretation}
    if req.analysis_type in ("tipping", "both"):
        tp = sa.tipping_point(req.estimate, req.se)
        results["tipping_point"] = {"tipping_gamma": round(tp.tipping_gamma, 4),
                                    "interpretation": tp.interpretation}
    return results


@router.post("/cost_effectiveness")
async def run_ce(req: CERequest, data_key: str = "data"):
    if data_key not in _data_store:
        raise HTTPException(400, "Upload data first")
    cea = CostEffectivenessAnalyzer(wtp=req.wtp)
    r = cea.analyze(_data_store[data_key], req.treatment_col, req.cost_col, req.effect_col)
    return {"icer": round(r.icer, 2) if r.icer else None, "nmb": round(r.nmb, 2),
            "prob_cost_effective": round(r.prob_cost_effective, 4)}


class BalanceRequest(BaseModel):
    treatment_col: str
    covariate_cols: list[str]
    model_type: str = "logistic"
    weights_type: str = "iptw"  # "iptw", "overlap", "none"


@router.post("/balance")
async def run_balance(req: BalanceRequest, data_key: str = "data"):
    """SMD diagnostics endpoint.  Computes standardized mean differences
    (unweighted and weighted) plus variance ratios for covariate balance
    assessment.

    weights_type options:
      - "none":   unweighted (crude) balance
      - "iptw":   inverse probability of treatment weights
      - "overlap": overlap (entropy) weights
    """
    if data_key not in _data_store:
        raise HTTPException(400, "Upload data first")
    df = _data_store[data_key]

    psa = PropensityScoreAnalyzer(model_type=req.model_type)
    ps = psa.estimate_propensity_scores(df, req.treatment_col, req.covariate_cols)
    t = df[req.treatment_col].values

    # Unweighted balance
    bal_unw = psa.assess_balance(df, req.treatment_col, req.covariate_cols, weights=None)

    result = {
        "unweighted": {
            "smd": {k: round(v, 4) for k, v in bal_unw.standardized_mean_differences.items()},
            "variance_ratio": {k: round(v, 4) for k, v in bal_unw.variance_ratios.items()},
            "overall": bal_unw.overall_balance,
            "max_smd": round(max(bal_unw.standardized_mean_differences.values()), 4),
        },
    }

    if req.weights_type == "iptw":
        weights = psa.compute_iptw(ps, t, stabilize=True)
        bal_w = psa.assess_balance(df, req.treatment_col, req.covariate_cols, weights)
        result["weighted_iptw"] = {
            "smd": {k: round(v, 4) for k, v in bal_w.standardized_mean_differences.items()},
            "variance_ratio": {k: round(v, 4) for k, v in bal_w.variance_ratios.items()},
            "overall": bal_w.overall_balance,
            "max_smd": round(max(bal_w.standardized_mean_differences.values()), 4),
        }
    elif req.weights_type == "overlap":
        from backend.analysis.propensity_score import OverlapWeightEstimator
        ols = OverlapWeightEstimator()
        ow = ols.overlap_weights(ps, t.astype(float))
        bal_w = psa.assess_balance(df, req.treatment_col, req.covariate_cols, ow)
        result["weighted_overlap"] = {
            "smd": {k: round(v, 4) for k, v in bal_w.standardized_mean_differences.items()},
            "variance_ratio": {k: round(v, 4) for k, v in bal_w.variance_ratios.items()},
            "overall": bal_w.overall_balance,
            "max_smd": round(max(bal_w.standardized_mean_differences.values()), 4),
        }

    return result
