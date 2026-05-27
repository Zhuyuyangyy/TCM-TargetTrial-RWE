# Claim-Evidence Table: Target Trial Emulation for TCM in NSCLC

---

## Purpose

This table maps each key scientific claim in the TCM-TargetTrial-RWE project to supporting evidence, methodological justification, and potential challenges. It serves as a quality assurance document for peer review preparation and internal validation.

---

| ID | Claim | Category | Supporting Evidence | Methodological Basis | Status | Risk Level |
|----|-------|----------|-------------------|---------------------|--------|------------|
| C01 | Target trial emulation can validly estimate causal effects from observational data | Methodology | Hernan & Robins 2016 (Am J Epidemiol); NEJM 2024 TTE perspective; FDA RWE Framework 2018/2023 | Potential outcomes framework; consistency, positivity, exchangeability assumptions | Established | Low |
| C02 | IPW can balance measured confounders between TCM and non-TCM groups | Methodology | Robins 1986; Austin 2011; Stuart 2010 | Re-weighting to create pseudo-population where treatment assignment is independent of confounders | Established | Low |
| C03 | AIPW is doubly robust: consistent if either treatment or outcome model is correctly specified | Methodology | Bang & Robins 2005; Chernozhukov et al. 2018 | Semiparametric efficiency theory; influence function-based estimation | Established | Low |
| C04 | SuperLearner ensemble reduces model misspecification risk compared to single-model approaches | Methodology | van der Laan & Rose 2011; van der Laan et al. 2007 | Cross-validated risk minimization; oracle inequality guarantees | Established | Low |
| C05 | Cloning + censoring approach addresses immortal time bias in TCM initiation timing | Methodology | Hernan et al. 2016; Dickerman et al. 2019 | Structural nested models; g-estimation for time-varying treatments | Established | Medium |
| C06 | Adjunctive TCM improves overall survival in advanced NSCLC (expected HR approximately 0.72-0.80) | Clinical | Li et al. 2023 meta-analysis (HR=0.72, 32 RCTs); Zhang et al. 2022 RWE study (OS gain 3.2 months) | Pooled RCT evidence; PSM cohort studies | Moderate evidence | Medium |
| C07 | Real-world EHR data can be reliably mapped to target trial eligibility criteria | Data | ICD-10 coding standards; pathology report TNM staging; pharmacy dispensing records | Data quality audits; validation against chart review sample | To be validated | Medium |
| C08 | E-value analysis can meaningfully quantify vulnerability to unmeasured confounding | Sensitivity | VanderWeele & Ding 2017; Mathur et al. 2018 | Bounding bias from unmeasured confounders using minimum strength of association | Established | Low |
| C09 | Partitioned survival model (PFS to Progression to Death) is appropriate for NSCLC CEA | Economic | ISPOR guidelines; NICE TAs for NSCLC; published NSCLC CEA models | Partitioned survival analysis widely accepted in oncology HTA | Established | Low |
| C10 | QALYs can be derived from EORTC QLQ-C30 via EQ-5D mapping in Chinese NSCLC population | Economic | Yang et al. 2018 mapping algorithm; Luo et al. 2020 Chinese EQ-5D value set | Validated crosswalk algorithms; population-specific value sets | Moderate evidence | Medium |
| C11 | The WTP threshold of 3x GDP per capita (approximately 258,000 CNY/QALY) is appropriate for China CEA | Economic | WHO-CHOICE recommendation; Woods et al. 2016; Chinese HTA guidelines 2020 | Health economics convention; emerging Chinese-specific evidence | Accepted convention | Low-Medium |
| C12 | Cross-fitting in AIPW avoids Donsker conditions and improves finite-sample performance | Methodology | Chernozhukov et al. 2018; Zheng & van der Laan 2011 | Sample splitting for nuisance parameter estimation; debiased ML theory | Established | Low |
| C13 | Weighted Cox model with robust variance provides valid inference under IPW | Methodology | Cole & Hernan 2004; Xie & Liu 2005 | Sandwich variance estimator for weighted estimating equations | Established | Low |
| C14 | RMST provides a clinically interpretable alternative to HR when PH assumption is violated | Methodology | Royston & Parmar 2013; Uno et al. 2014 | Restricted mean time lost; area under survival curve | Established | Low |
| C15 | TCM heterogeneity can be meaningfully addressed through subgroup and type-specific analyses | Clinical | Wang et al. 2024 review; clinical expert consensus | Pre-specified subgroup definitions by TCM category (injection, oral, external) | To be validated | Medium-High |
| C16 | Multi-center EHR data from 3-5 hospitals provides sufficient sample size (N=2500-4000) | Data | Institutional NSCLC registry volumes; published Chinese NSCLC RWE studies | Power calculation based on Schoenfeld formula for 350 events | To be confirmed | Medium |
| C17 | CONSORT-Target Trial reporting checklist adaptation is feasible for this study | Reporting | CONSORT-TCM extension; STROBE-TTE; SPIRIT adaptations | Existing reporting frameworks for TTE and TCM can be integrated | To be implemented | Low |
| C18 | The framework is generalizable to other TCM indications beyond NSCLC | Generalizability | Modular architecture design; configuration-driven protocol specification | Software engineering: separation of framework logic from disease-specific parameters | To be validated | Medium |
| C19 | Mortality linkage via Chinese vital statistics provides complete survival ascertainment | Data | National Death Index protocols; local vital statistics linkage procedures | Deterministic and probabilistic record linkage algorithms | To be validated | Medium |
| C20 | Results will be robust to violations of the no unmeasured confounders assumption | Assumption | E-value analysis; probabilistic bias analysis; tipping point analysis | Multiple complementary sensitivity analysis approaches | To be demonstrated | High |

---

## Evidence Strength Rating Scale

| Rating | Definition |
|--------|-----------|
| Established | Well-validated in peer-reviewed literature with multiple independent confirmations |
| Moderate evidence | Supported by some studies but with limitations or inconsistency |
| To be validated | Novel application or assumption requiring empirical verification in this study |
| To be demonstrated | Core assumption that must be shown through sensitivity analyses |
| To be implemented | Planned methodological component not yet executed |

## Risk Level Definitions

| Level | Definition | Action Required |
|-------|-----------|----------------|
| Low | Well-established method or evidence; minimal threat to validity | Standard implementation |
| Medium | Some uncertainty; requires careful implementation and sensitivity analysis | Enhanced monitoring and validation |
| High | Core assumption with limited control; could undermine primary conclusions | Multiple sensitivity analyses; explicit discussion in limitations |

---

## Evidence-to-Claim Mapping Summary

- Total claims: 20
- Methodology claims: 10 (C01-C05, C08, C12-C14, C17)
- Clinical claims: 3 (C06, C15, C18)
- Data/Infrastructure claims: 4 (C07, C16, C19, C20)
- Economic claims: 3 (C09-C11)

- Established: 12 (60%)
- Moderate evidence: 2 (10%)
- To be validated: 4 (20%)
- To be demonstrated: 1 (5%)
- To be implemented: 1 (5%)

---

## Key Risk Areas

1. **C20 (Unmeasured confounding)** - Highest risk. Core causal assumption. Must be addressed through comprehensive sensitivity analyses (E-value, probabilistic bias, tipping point). Even with robust methods, this remains the primary threat to causal interpretation.

2. **C15 (TCM heterogeneity)** - High risk. Wide variety of TCM formulas, dosing, and treatment durations may not be adequately captured in pre-specified subgroups. Requires clinical expert input and potentially post-hoc exploratory analyses.

3. **C05 (Immortal time bias)** - Medium risk. The cloning + censoring approach is theoretically sound but complex to implement correctly. Requires careful data engineering and validation against simulated data with known truth.

4. **C07 (Data mapping)** - Medium risk. EHR coding quality varies across hospitals. Requires pilot data quality assessment and potential manual chart review for validation sample.

---

**Document Version:** v1.0
**Last Updated:** 2026-05-27
**Maintained By:** [Research Team]
**Review Cycle:** Updated with each major study milestone
