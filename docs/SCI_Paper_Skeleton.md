# Target Trial Emulation Framework for Evaluating Adjunctive Traditional Chinese Medicine in Non-Small Cell Lung Cancer: A Real-World Evidence Study

---

## Structured Abstract

**Background:** Real-world evidence (RWE) on the effectiveness of adjunctive Traditional Chinese Medicine (TCM) in non-small cell lung cancer (NSCLC) is often limited by confounding, immortal time bias, and lack of causal interpretability. Target trial emulation (TTE) offers a methodological bridge between randomized trials and observational data.

**Objective:** To develop and validate a target trial emulation framework that estimates the causal effect of adjunctive TCM therapy on overall survival, progression-free survival, and cost-effectiveness in NSCLC patients using real-world clinical data.

**Methods:** We define a hypothetical target trial protocol specifying eligibility criteria, treatment strategies, assignment procedures, outcome measures, and analysis plans. Observational EHR and cancer registry data are mapped to this protocol. The causal inference engine employs inverse probability weighting (IPW), augmented IPW (AIPW), and g-computation under the potential outcomes framework. Survival outcomes are analyzed via weighted Kaplan-Meier and Cox proportional hazards models. Cost-effectiveness is evaluated through incremental cost-effectiveness ratios (ICER) with probabilistic sensitivity analysis.

**Results:** [To be completed upon study execution]

**Conclusions:** [To be completed upon study execution]

**Keywords:** Target trial emulation; Traditional Chinese Medicine; Non-small cell lung cancer; Real-world evidence; Causal inference; Inverse probability weighting; Cost-effectiveness analysis

---

## 1. Introduction

### 1.1 Clinical Context
- NSCLC accounts for approximately 85% of lung cancers globally; 5-year survival remains poor
- TCM (e.g., Fufang Kushen injection, Aidi injection, oral herbal formulas) is widely used as adjunctive therapy in China
- Evidence base: heterogeneous retrospective studies, small RCTs with questionable methodology (CONSORT-TCM gaps)
- Regulatory and clinical demand for rigorous RWE to support TCM integration decisions

### 1.2 Problem Statement
- Observational TCM studies suffer from:
  - Confounding by indication (healthier patients more likely to receive TCM)
  - Immortal time bias (delay between diagnosis and TCM initiation)
  - Lack of structured causal protocol
  - Poor reproducibility and transparency
- Existing systematic reviews report low certainty of evidence (GRADE low-moderate)

### 1.3 Target Trial Emulation as a Solution
- Hernan and Robins (2016): Emulating a target trial using observational data
- NEJM 2024 perspective: Growing regulatory acceptance of TTE-based RWE
- Advantages: explicit protocol, reduced bias, causal interpretability, reproducibility

### 1.4 Study Objectives
1. Define a target trial protocol for adjunctive TCM in NSCLC
2. Implement a causal inference engine (IPW, AIPW, g-computation)
3. Estimate causal effects on OS, PFS, and QoL
4. Conduct cost-effectiveness analysis alongside the emulated trial
5. Validate against known RCT benchmarks where available

---

## 2. Related Work

### 2.1 TCM Real-World Evidence in Oncology
- Li et al. (2023): Meta-analysis of TCM adjunctive therapy in NSCLC (32 RCTs, n=3200), HR=0.72 for OS, GRADE low-moderate
- Zhang et al. (2022): Registry-based study of Aidi injection in advanced NSCLC, propensity score matching (PSM), median OS gain 3.2 months
- Wang et al. (2024): Systematic review identifying methodological gaps in TCM oncology RWE
- National Cancer Center (China): Real-world data platform for TCM oncology outcomes

### 2.2 Target Trial Emulation Methodology
- Hernan MA, Robins JM (2016). Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. Am J Epidemiol, 183(8):758-764.
- NEJM 2024: Target Trial Emulation - A Framework for Causal Inference from Observational Data (Editorial/Perspective)
- Dickerman et al. (2019): TTE applied to statin use and cancer mortality
- Danaei et al. (2013): Comparative effectiveness using TTE framework
- FDA Real-World Evidence Framework (2018, updated 2023): regulatory guidance for TTE studies

### 2.3 Causal Inference in Observational Studies
- Robins (1986): G-estimation and structural nested models
- van der Laan and Rose (2011): Targeted Learning (TMLE)
- Bang and Robins (2005): Augmented inverse probability weighting (AIPW) - doubly robust
- Hernan and Robins (2020). Causal Inference: What If. Boca Raton: Chapman and Hall/CRC.
- Stuart (2010): Matching methods for causal inference - review
- Austin (2011): Propensity score matching in medical research - practical guide

---

## 3. Methods

### 3.1 Target Trial Protocol Specification

| Component | Specification |
|-----------|--------------|
| Eligibility | Age >= 18; histologically confirmed NSCLC (stage IIIB-IV); ECOG 0-2; receiving first-line platinum-based chemotherapy |
| Treatment strategies | (A) Platinum-based chemo + adjunctive TCM >= 2 cycles vs. (B) Platinum-based chemo alone |
| Assignment | Emulated via observed treatment assignment in EHR; ignorable given measured covariates |
| Outcome | Primary: OS; Secondary: PFS, ORR, QoL (EORTC QLQ-C30), adverse events |
| Follow-up | From treatment initiation (time zero) to death, loss to follow-up, or administrative censoring (36 months) |
| Causal contrast | Per-protocol effect (ATE) and intention-to-treat analog |
| Analysis plan | IPW, AIPW, g-computation; sensitivity analysis for unmeasured confounding |

#### 3.1.1 Time Zero Definition
- Index date: Date of first chemotherapy cycle
- TCM initiation: Must begin within 60 days of index date (grace period)
- Immortal time handled via cloning + censoring approach

#### 3.1.2 Eligibility Criteria Mapping
- ICD-10 codes: C34.x
- TNM staging from pathology reports
- TCM prescription codes from hospital pharmacy systems

### 3.2 Causal Inference Engine

#### 3.2.1 Directed Acyclic Graph (DAG)
- Exposure: TCM use (binary)
- Outcome: Survival time
- Confounders: Age, sex, stage, ECOG, histology, biomarkers, comorbidities
- Structure: Common causes of both TCM assignment and survival outcome

#### 3.2.2 Inverse Probability Weighting (IPW)
- Treatment model: Logistic regression / GBM for P(TCM | X)
- Stabilized weights: SW = P(TCM) / P(TCM | X)
- Truncation at 1st and 99th percentiles to reduce variance
- Balance diagnostics: standardized mean differences (SMD < 0.1)

#### 3.2.3 Augmented Inverse Probability Weighting (AIPW)
- Doubly robust: consistent if either treatment or outcome model correct
- AIPW estimator: mu_AIPW = (1/n) * sum[ I(A=a)/P_hat(A|X) * (Y - mu_hat(a,X)) + mu_hat(a,X) ]
- Outcome model: flexible ML (SuperLearner ensemble: LASSO, RF, XGBoost, GLM)
- Cross-fitting to avoid Donsker conditions (Chernozhukov et al., 2018)

#### 3.2.4 G-Computation
- Parametric g-formula for time-varying treatments
- Monte Carlo simulation under hypothetical intervention regimes
- Bootstrap confidence intervals (1000 iterations)

#### 3.2.5 Sensitivity Analysis for Unmeasured Confounding
- E-value calculation (VanderWeele and Ding, 2017)
- Probabilistic bias analysis (Lash et al., 2009)
- Tipping point analysis

### 3.3 Survival Analysis

#### 3.3.1 Primary Analysis
- Weighted Kaplan-Meier estimator for OS and PFS
- Weighted Cox proportional hazards model: HR with 95% CI
- Restricted mean survival time (RMST) difference at 12, 24, 36 months

#### 3.3.2 Secondary Analyses
- Competing risks (Fine-Gray) for cancer-specific death vs. other causes
- Landmark analysis at 3, 6, 12 months to address immortal time bias
- Subgroup analyses: by stage, histology (adenocarcinoma vs. squamous), TCM type, treatment duration

#### 3.3.3 Proportional Hazards Assessment
- Schoenfeld residuals test
- Time-varying coefficient models if PH violated

### 3.4 Cost-Effectiveness Analysis

#### 3.4.1 Perspective and Time Horizon
- Healthcare system perspective (Chinese SMI reimbursement)
- Lifetime horizon with 3% annual discounting
- Partitioned survival model: PFS -> Progression -> Death

#### 3.4.2 Costs
- Direct medical: chemotherapy drugs, TCM preparations, hospitalization, monitoring, adverse event management
- Unit costs from National Drug Reimbursement List and hospital charge data
- Indirect costs: productivity loss (human capital approach)

#### 3.4.3 Outcomes
- Quality-adjusted life years (QALYs): EQ-5D-5L utility values mapped from EORTC QLQ-C30
- Life years gained (LYG)

#### 3.4.4 Analysis
- Incremental cost-effectiveness ratio (ICER) = Delta Cost / Delta QALY
- Willingness-to-pay threshold: 3x GDP per capita (approximately 258,000 CNY/QALY)
- Probabilistic sensitivity analysis (PSA): 10,000 Monte Carlo iterations
- One-way deterministic sensitivity analysis on key parameters
- Value of information analysis (EVPPI)

---

## 4. Experimental Setup

### 4.1 Data Sources
- Multi-center hospital EHR data (3-5 tertiary hospitals, 2015-2024)
- Chinese Lung Cancer Registry (subset)
- Hospital pharmacy TCM dispensing records
- Mortality linkage via National Death Index / local vital statistics

### 4.2 Study Population
- Estimated N = 2,500-4,000 NSCLC patients
- TCM group: approximately 40% (based on prior institutional data)
- Standard care group: approximately 60%
- Inclusion: stage IIIB-IV, first-line treatment, >= 18 years
- Exclusion: prior TCM use for cancer, other active malignancies, incomplete staging

### 4.3 Covariates (Measured Confounders)
- Demographics: age, sex, BMI, smoking status, education
- Clinical: stage, histology, ECOG, gene mutation status (EGFR, ALK, ROS1, PD-L1), comorbidity index (Charlson)
- Treatment: chemotherapy regimen, cycles received, dose intensity, surgery/radiation history
- Laboratory: albumin, hemoglobin, LDH, tumor markers (CEA, CYFRA21-1, NSE)
- TCM specifics: formula type, duration, frequency, concurrent herbal components

### 4.4 Software and Reproducibility
- R (v4.3+): WeightIt, survival, survminer, geex, ltmle
- Python (v3.10+): EconML, causalml, lifelines, scikit-survival
- TMLE implementation: tmle R package or dowhy Python
- DAG visualization: dagitty R package
- Reporting: CONSORT-Target Trial checklist adaptation
- Version control: Git; containerized environment (Docker)

### 4.5 Sample Size and Power
- Based on expected HR = 0.75 (OS), alpha = 0.05, power = 0.80
- Required events: approximately 350 deaths (Schoenfeld formula)
- Effective sample size after weighting: evaluated via design effect
- Monte Carlo simulation for finite-sample performance of AIPW

### 4.6 Validation Strategy
- Internal: k-fold cross-validation (k=5) for treatment and outcome models
- External: temporal validation (pre-2020 training, 2020+ testing)
- Benchmark: compare emulated trial estimates against published RCT results where overlap exists

---

## 5. Ethical Considerations

- Institutional Review Board (IRB) approval at each participating center
- De-identified data; waiver of informed consent for retrospective analysis
- Compliance with China Personal Information Protection Law (PIPL)
- Study registration: ClinicalTrials.gov (pending) or Chinese Clinical Trial Registry

---

## 6. Expected Contributions

1. First rigorously designed target trial emulation for TCM in oncology
2. Open-source causal inference pipeline adaptable to other TCM indications
3. Causal effect estimates with explicit bias quantification
4. Cost-effectiveness evidence to inform China National Reimbursement Drug List (NRDL) decisions
5. Methodological template for TCM RWE aligned with international standards (NEJM, ICH E9(R1))

---

## 7. Limitations

- Residual confounding from unmeasured variables (e.g., patient preference, lifestyle factors)
- Generalizability limited to Chinese hospital settings
- TCM heterogeneity (multiple formulas, dosing variability)
- Data quality dependency on EHR completeness
- Potential informative censoring (non-random dropout)

---

## References

1. Hernan MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. Am J Epidemiol. 2016;183(8):758-764.
2. Hernan MA, Robins JM. Causal Inference: What If. Boca Raton: Chapman and Hall/CRC; 2020.
3. Robins JM. A new approach to causal inference in mortality studies with a sustained exposure period. Math Model. 1986;7(9-12):1393-1512.
4. Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models. Biometrics. 2005;61(4):962-973.
5. van der Laan MJ, Rose S. Targeted Learning: Causal Inference for Observational and Experimental Data. New York: Springer; 2011.
6. Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters. Econometrics J. 2018;21(1):C1-C68.
7. VanderWeele TJ, Ding P. Sensitivity Analysis in Observational Research: Introducing the E-Value. Ann Intern Med. 2017;167(4):268-274.
8. Dickerman BA, Garcia-Albeniz X, Logan RW, et al. Avoidable flaws in observational analyses: an application to statins and cancer. Nat Med. 2019;25(10):1601-1606.
9. Li M, Wang Y, Zhang Q, et al. Adjunctive Traditional Chinese Medicine therapy in advanced NSCLC: a systematic review and meta-analysis of randomized controlled trials. J Ethnopharmacol. 2023;301:115792.
10. Zhang L, Chen W, Liu X, et al. Real-world evidence of Aidi injection in NSCLC: a propensity score matched cohort study. Chin J Integr Med. 2022;28(5):387-394.
11. Stuart EA. Matching methods for causal inference: A review and a look forward. Stat Sci. 2010;25(1):1-21.
12. Austin PC. An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies. Multivariate Behav Res. 2011;46(3):399-424.
13. Lash TL, Fox MP, Fink AK. Applying Quantitative Bias Analysis to Epidemiologic Data. New York: Springer; 2009.
14. U.S. Food and Drug Administration. Framework for FDA Real-World Evidence Program. 2018. Updated 2023.
15. Wang S, Liu H, Zhao Y, et al. Methodological quality assessment of real-world evidence studies in Traditional Chinese Medicine oncology: a systematic review. BMC Complement Med Ther. 2024;24(1):89.

---

**Corresponding Author:** [Name, Affiliation, Email]
**Funding:** [Grant Number, Agency]
**Conflicts of Interest:** [Declared]

---
**Document Version:** v1.0
**Last Updated:** 2026-05-27
**Status:** Draft / Working Paper
