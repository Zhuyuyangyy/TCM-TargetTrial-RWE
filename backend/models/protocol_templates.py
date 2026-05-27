"""Pre-configured protocol templates for TCM target trial emulation.

These templates provide ready-to-use protocols for common TCM research
questions. Each template can be customized and used directly with the
TargetTrialEmulator.
"""
from __future__ import annotations

from typing import Dict, List

from .trial_protocol import (
    EligibilityCriteria,
    OutcomeDefinition,
    TargetTrialProtocol,
    TreatmentStrategy,
)


# ---------------------------------------------------------------------------
# Template 1: Buzhong Yiqi Tang for fatigue in cancer patients
# ---------------------------------------------------------------------------
BUZHONG_YIQI_CANCER_FATIGUE = TargetTrialProtocol(
    trial_id="TTE-001",
    title="补中益气汤治疗癌因性疲乏的目标试验仿真",
    description=(
        "Emulates a target trial comparing Buzhong Yiqi Tang vs. "
        "standard supportive care for cancer-related fatigue in "
        "patients with solid tumors."
    ),
    eligibility=EligibilityCriteria(
        min_age=18,
        max_age=80,
        required_diagnoses=["solid_tumor", "cancer_related_fatigue"],
        excluded_diagnoses=["psychiatric_disorder", "severe_organ_failure"],
        min_lab_values={"hemoglobin": 80, "albumin": 25},
    ),
    treatment_strategies=[
        TreatmentStrategy(
            name="Buzhong Yiqi Tang",
            description="补中益气汤加减",
            tcm_formula="buzhong_yiqi_tang",
            components=["黄芪", "人参", "白术", "甘草", "当归", "陈皮", "升麻", "柴胡"],
            dosage="每日1剂，分2次服",
            duration_weeks=8,
        ),
        TreatmentStrategy(
            name="Standard Supportive Care",
            description="常规对症支持治疗",
            comparator=True,
            duration_weeks=8,
        ),
    ],
    primary_outcome=OutcomeDefinition(
        name="Fatigue severity change",
        outcome_type="continuous",
        variable="fatigue_score",
        time_variable="weeks",
        direction="lower",
        minimal_clinically_important_difference=3.0,
    ),
    secondary_outcomes=[
        OutcomeDefinition(
            name="Quality of life",
            outcome_type="continuous",
            variable="qol_score",
            direction="higher",
        ),
        OutcomeDefinition(
            name="Body weight change",
            outcome_type="continuous",
            variable="weight_kg",
            direction="higher",
        ),
    ],
    follow_up_weeks=12,
    grace_period_weeks=2,
    adjustment_variables=[
        "age", "sex", "cancer_stage", "chemotherapy_regimen",
        "baseline_fatigue_score", "hemoglobin", "albumin",
    ],
    effect_measure="mean_difference",
)


# ---------------------------------------------------------------------------
# Template 2: Liuwei Dihuang Wan for diabetic nephropathy
# ---------------------------------------------------------------------------
LIUWEI_DIHUANG_DN = TargetTrialProtocol(
    trial_id="TTE-002",
    title="六味地黄丸治疗糖尿病肾病的目标试验仿真",
    description=(
        "Emulates a target trial comparing Liuwei Dihuang Wan + standard care "
        "vs. standard care alone for diabetic nephropathy progression."
    ),
    eligibility=EligibilityCriteria(
        min_age=30,
        max_age=75,
        required_diagnoses=["type2_diabetes", "diabetic_nephropathy"],
        excluded_diagnoses=["type1_diabetes", "dialysis", "renal_transplant"],
        min_lab_values={"eGFR": 15, "hemoglobin": 90},
        max_lab_values={"UACR": 3000},
    ),
    treatment_strategies=[
        TreatmentStrategy(
            name="Liuwei Dihuang Wan + Standard Care",
            description="六味地黄丸 + 标准降糖降压治疗",
            tcm_formula="liuwei_dihuang_wan",
            components=["熟地黄", "山茱萸", "山药", "泽泻", "茯苓", "牡丹皮"],
            dosage="每次8丸，每日3次",
            duration_weeks=52,
        ),
        TreatmentStrategy(
            name="Standard Care Alone",
            description="标准降糖降压治疗（ACEI/ARB + 降糖药）",
            comparator=True,
            duration_weeks=52,
        ),
    ],
    primary_outcome=OutcomeDefinition(
        name="eGFR decline",
        outcome_type="continuous",
        variable="eGFR",
        time_variable="weeks",
        direction="higher",
        minimal_clinically_important_difference=5.0,
    ),
    secondary_outcomes=[
        OutcomeDefinition(
            name="UACR change",
            outcome_type="continuous",
            variable="UACR",
            direction="lower",
        ),
        OutcomeDefinition(
            name="HbA1c",
            outcome_type="continuous",
            variable="HbA1c",
            direction="lower",
        ),
        OutcomeDefinition(
            name="Composite renal endpoint",
            outcome_type="binary",
            variable="renal_composite",
            direction="lower",
        ),
    ],
    follow_up_weeks=52,
    grace_period_weeks=4,
    adjustment_variables=[
        "age", "sex", "BMI", "diabetes_duration", "baseline_eGFR",
        "baseline_UACR", "HbA1c", "systolic_BP", "ACEI_ARB_use",
        "insulin_use", "smoking_status",
    ],
    effect_measure="hazard_ratio",
)


# ---------------------------------------------------------------------------
# Template 3: Danshen + Aspirin for stable angina
# ---------------------------------------------------------------------------
DANSHEN_ASPIRIN_ANGINA = TargetTrialProtocol(
    trial_id="TTE-003",
    title="丹参饮联合阿司匹林治疗稳定型心绞痛的目标试验仿真",
    description=(
        "Emulates a target trial comparing Danshen-based formula + aspirin "
        "vs. aspirin alone for stable angina pectoris."
    ),
    eligibility=EligibilityCriteria(
        min_age=40,
        max_age=80,
        required_diagnoses=["stable_angina"],
        excluded_diagnoses=["acute_coronary_syndrome", "severe_heart_failure"],
        min_lab_values={"LVEF": 40},
        max_lab_values={"troponin_I": 0.04},
    ),
    treatment_strategies=[
        TreatmentStrategy(
            name="Danshen Yin + Aspirin",
            description="丹饮加味 + 阿司匹林 100mg/d",
            tcm_formula="danshen_yin",
            components=["丹参", "檀香", "砂仁"],
            dosage="每日1剂 + 阿司匹林100mg/d",
            duration_weeks=24,
        ),
        TreatmentStrategy(
            name="Aspirin Alone",
            description="阿司匹林 100mg/d",
            comparator=True,
            duration_weeks=24,
        ),
    ],
    primary_outcome=OutcomeDefinition(
        name="Angina frequency",
        outcome_type="continuous",
        variable="angina_episodes_per_week",
        time_variable="weeks",
        direction="lower",
    ),
    secondary_outcomes=[
        OutcomeDefinition(
            name="Exercise tolerance",
            outcome_type="continuous",
            variable="exercise_duration_sec",
            direction="higher",
        ),
        OutcomeDefinition(
            name="Nitroglycerin use",
            outcome_type="continuous",
            variable="ntg_doses_per_week",
            direction="lower",
        ),
    ],
    follow_up_weeks=24,
    grace_period_weeks=2,
    adjustment_variables=[
        "age", "sex", "BMI", "hypertension", "diabetes",
        "baseline_angina_frequency", "LVEF", "baseline_exercise_duration",
        "statin_use", "beta_blocker_use", "CCB_use",
    ],
    effect_measure="mean_difference",
)


# ---------------------------------------------------------------------------
# Template 4: Xiao Chaihu Tang for chronic hepatitis B
# ---------------------------------------------------------------------------
XIAO_CHAIHU_HBV = TargetTrialProtocol(
    trial_id="TTE-004",
    title="小柴胡汤联合核苷类似物治疗慢性乙型肝炎的目标试验仿真",
    description=(
        "Emulates a target trial comparing Xiao Chaihu Tang + NUC "
        "vs. NUC alone for chronic hepatitis B."
    ),
    eligibility=EligibilityCriteria(
        min_age=18,
        max_age=65,
        required_diagnoses=["chronic_hepatitis_b"],
        excluded_diagnoses=["hepatocellular_carcinoma", "decompensated_cirrhosis", "HIV_coinfection"],
        min_lab_values={"ALT": 40, "HBV_DNA": 2000},
    ),
    treatment_strategies=[
        TreatmentStrategy(
            name="Xiao Chaihu Tang + NUC",
            description="小柴胡汤 + 核苷类似物（恩替卡韦/替诺福韦）",
            tcm_formula="xiao_chaihu_tang",
            components=["柴胡", "黄芩", "人参", "半夏", "甘草", "生姜", "大枣"],
            dosage="每日1剂 + 恩替卡韦0.5mg/d",
            duration_weeks=48,
        ),
        TreatmentStrategy(
            name="NUC Alone",
            description="核苷类似物单药（恩替卡韦0.5mg/d）",
            comparator=True,
            duration_weeks=48,
        ),
    ],
    primary_outcome=OutcomeDefinition(
        name="HBV DNA suppression",
        outcome_type="binary",
        variable="hbv_dna_undetectable",
        direction="higher",
    ),
    secondary_outcomes=[
        OutcomeDefinition(
            name="HBeAg seroconversion",
            outcome_type="binary",
            variable="hbeag_seroconversion",
            direction="higher",
        ),
        OutcomeDefinition(
            name="ALT normalization",
            outcome_type="binary",
            variable="alt_normal",
            direction="higher",
        ),
        OutcomeDefinition(
            name="Liver fibrosis score",
            outcome_type="continuous",
            variable="fibroscan_kpa",
            direction="lower",
        ),
    ],
    follow_up_weeks=48,
    grace_period_weeks=4,
    adjustment_variables=[
        "age", "sex", "baseline_HBV_DNA", "baseline_ALT",
        "HBeAg_status", "liver_stiffness", "platelet_count",
    ],
    effect_measure="risk_ratio",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PROTOCOL_TEMPLATES: Dict[str, TargetTrialProtocol] = {
    "buzhong_yiqi_cancer_fatigue": BUZHONG_YIQI_CANCER_FATIGUE,
    "liuwei_dihuang_dn": LIUWEI_DIHUANG_DN,
    "danshen_aspirin_angina": DANSHEN_ASPIRIN_ANGINA,
    "xiao_chaihu_hbv": XIAO_CHAIHU_HBV,
}


def list_templates() -> List[Dict]:
    """List all available protocol templates."""
    return [
        {
            "id": tid,
            "trial_id": p.trial_id,
            "title": p.title,
            "description": p.description[:100] + "...",
            "n_strategies": len(p.treatment_strategies),
            "follow_up_weeks": p.follow_up_weeks,
            "effect_measure": p.effect_measure,
        }
        for tid, p in PROTOCOL_TEMPLATES.items()
    ]


def get_template(template_id: str) -> TargetTrialProtocol:
    """Get a protocol template by ID."""
    if template_id not in PROTOCOL_TEMPLATES:
        raise ValueError(f"Unknown template: {template_id}. Available: {list(PROTOCOL_TEMPLATES.keys())}")
    return PROTOCOL_TEMPLATES[template_id]
