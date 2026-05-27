"""Target Trial Protocol dataclass."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import date
import yaml


@dataclass
class EligibilityCriteria:
    min_age: int = 18
    max_age: int = 100
    required_diagnoses: list[str] = field(default_factory=list)
    excluded_diagnoses: list[str] = field(default_factory=list)
    min_lab_values: dict[str, float] = field(default_factory=dict)
    max_lab_values: dict[str, float] = field(default_factory=dict)
    required_prior_treatments: list[str] = field(default_factory=list)
    excluded_prior_treatments: list[str] = field(default_factory=list)


@dataclass
class TreatmentStrategy:
    name: str
    description: str = ""
    tcm_formula: Optional[str] = None
    components: list[str] = field(default_factory=list)
    dosage: Optional[str] = None
    duration_weeks: Optional[int] = None
    comparator: bool = False


@dataclass
class OutcomeDefinition:
    name: str
    outcome_type: str
    variable: str
    time_variable: Optional[str] = None
    direction: str = "lower"
    minimal_clinically_important_difference: Optional[float] = None


@dataclass
class TargetTrialProtocol:
    trial_id: str
    title: str
    version: str = "1.0"
    created_date: str = field(default_factory=lambda: date.today().isoformat())
    description: str = ""
    eligibility: EligibilityCriteria = field(default_factory=EligibilityCriteria)
    treatment_strategies: list[TreatmentStrategy] = field(default_factory=list)
    primary_outcome: Optional[OutcomeDefinition] = None
    secondary_outcomes: list[OutcomeDefinition] = field(default_factory=list)
    follow_up_weeks: int = 52
    grace_period_weeks: int = 4
    adjustment_variables: list[str] = field(default_factory=list)
    effect_measure: str = "hazard_ratio"

    @classmethod
    def from_yaml(cls, path: str) -> "TargetTrialProtocol":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        eligibility = EligibilityCriteria(**data.get("eligibility", {}))
        strategies = [TreatmentStrategy(**s) for s in data.get("treatment_strategies", [])]
        primary = OutcomeDefinition(**data["primary_outcome"]) if "primary_outcome" in data else None
        secondary = [OutcomeDefinition(**s) for s in data.get("secondary_outcomes", [])]
        return cls(
            trial_id=data["trial_id"], title=data["title"],
            version=data.get("version", "1.0"), description=data.get("description", ""),
            eligibility=eligibility, treatment_strategies=strategies,
            primary_outcome=primary, secondary_outcomes=secondary,
            follow_up_weeks=data.get("follow_up_weeks", 52),
            grace_period_weeks=data.get("grace_period_weeks", 4),
            adjustment_variables=data.get("adjustment_variables", []),
            effect_measure=data.get("effect_measure", "hazard_ratio"),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
