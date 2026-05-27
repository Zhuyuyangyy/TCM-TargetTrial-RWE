"""Protocol management API endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.models.trial_protocol import TargetTrialProtocol

router = APIRouter()
_protocols: dict[str, TargetTrialProtocol] = {}


class ProtocolCreateRequest(BaseModel):
    trial_id: str
    title: str
    description: str = ""
    eligibility: dict = {}
    treatment_strategies: list[dict] = []
    primary_outcome: Optional[dict] = None
    secondary_outcomes: list[dict] = []
    follow_up_weeks: int = 52
    adjustment_variables: list[str] = []
    effect_measure: str = "hazard_ratio"


class ProtocolResponse(BaseModel):
    trial_id: str
    title: str
    description: str
    n_strategies: int
    follow_up_weeks: int


@router.get("/", response_model=list[ProtocolResponse])
async def list_protocols():
    return [ProtocolResponse(trial_id=p.trial_id, title=p.title, description=p.description,
                             n_strategies=len(p.treatment_strategies), follow_up_weeks=p.follow_up_weeks)
            for p in _protocols.values()]


@router.post("/", response_model=ProtocolResponse)
async def create_protocol(req: ProtocolCreateRequest):
    if req.trial_id in _protocols:
        raise HTTPException(status_code=409, detail="Protocol already exists")
    p = TargetTrialProtocol(trial_id=req.trial_id, title=req.title, description=req.description,
                            follow_up_weeks=req.follow_up_weeks,
                            adjustment_variables=req.adjustment_variables,
                            effect_measure=req.effect_measure)
    _protocols[req.trial_id] = p
    return ProtocolResponse(trial_id=p.trial_id, title=p.title, description=p.description,
                            n_strategies=len(p.treatment_strategies), follow_up_weeks=p.follow_up_weeks)


@router.get("/{trial_id}")
async def get_protocol(trial_id: str):
    if trial_id not in _protocols:
        raise HTTPException(status_code=404, detail="Protocol not found")
    return _protocols[trial_id].to_dict()
