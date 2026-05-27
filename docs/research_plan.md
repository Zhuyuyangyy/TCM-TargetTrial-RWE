# Research Plan: Target Trial Emulation for TCM using Real-World Evidence

## Background

Traditional Chinese Medicine (TCM) has been widely used in China for adjuvant
cancer therapy, but high-quality evidence from randomized controlled trials (RCTs)
remains limited. The target trial emulation framework (Hernan & Robins, 2016)
provides a principled approach to estimating causal treatment effects from
observational data by explicitly specifying the protocol of the hypothetical RCT.

## Research Questions

### Primary Question
Does adjuvant TCM therapy (Fuzheng Guben decoction) improve overall survival
in patients with stage II-III NSCLC after surgical resection, compared to
standard care alone?

### Secondary Questions
1. Does TCM adjuvant therapy improve disease-free survival?
2. Does TCM adjuvant therapy improve quality of life (EORTC QLQ-C30)?
3. Does TCM adjuvant therapy improve immune function (CD4/CD8 ratio)?
4. Is TCM adjuvant therapy cost-effective?

## Methodology

### Phase 1: Protocol Specification (Months 1-2)
- Specify eligibility criteria based on clinical guidelines
- Define treatment strategies and comparators
- Identify primary and secondary outcomes
- Select confounders based on DAG (Directed Acyclic Graph)

### Phase 2: Data Preparation (Months 2-3)
- Extract and harmonize EHR data from 3 hospitals
- Apply variable mapping and standardization
- Handle missing data via MICE with chained equations
- Assess data quality and completeness

### Phase 3: Analysis (Months 3-5)
1. Propensity Score Estimation: Logistic regression, GBM, random forest
2. Covariate Balance: SMD assessment before/after weighting
3. Treatment Effect Estimation: IPW, AIPW (doubly robust), PS matching
4. Survival Analysis: Kaplan-Meier, Cox PH, RMST
5. Sensitivity Analysis: E-value, tipping point, negative controls

### Phase 4: Cost-Effectiveness (Months 5-6)
- Compute ICER with bootstrap uncertainty
- Cost-effectiveness acceptability curves
- Net Monetary Benefit at various WTP thresholds

### Phase 5: Reporting (Months 6-7)
- STROBE guidelines for observational studies
- CONSORT adaptation for target trial emulation

## Expected Outputs

1. Hazard ratio for overall survival (TCM vs. standard care)
2. RMST difference with confidence intervals
3. E-value assessing robustness to unmeasured confounding
4. ICER and cost-effectiveness analysis
5. Covariate balance tables (Love plots)

## Innovation Points

1. First systematic TTE for TCM interventions
2. TCM-specific confounders: syndrome differentiation, constitution types
3. Doubly robust estimation (AIPW)
4. Comprehensive sensitivity analysis calibrated for TCM research
5. First NMB analysis for TCM adjuvant therapy

## Target Journals

1. Annals of Internal Medicine (IF ~50)
2. BMJ (IF ~30)
3. JAMA Network Open (IF ~13)
4. Phytomedicine (IF ~7)
5. BMC Complementary Medicine and Therapies (IF ~3)

## Timeline

| Phase | Months | Deliverable |
|-------|--------|-------------|
| Protocol design | 1-2 | Protocol YAML, DAG diagram |
| Data preparation | 2-3 | Clean dataset, variable mapping |
| Main analysis | 3-5 | Effect estimates, tables |
| Sensitivity | 5-6 | E-values, tipping points |
| Cost-effectiveness | 6-7 | ICER, CEAC curves |
| Manuscript | 7-9 | First draft |
| Submission | 9-10 | Journal submission |
