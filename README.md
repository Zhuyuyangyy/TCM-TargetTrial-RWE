# TCM-TargetTrial-RWE

**Target Trial Emulation Framework for Traditional Chinese Medicine using Real-World Evidence**

## Overview

This framework implements the target trial emulation (TTE) paradigm for evaluating
Traditional Chinese Medicine (TCM) interventions using electronic health record (EHR)
data. It bridges the gap between randomized controlled trials and observational studies
by applying rigorous causal inference methods to real-world data.

## Key Features

- **Target Trial Protocol Design**: Formal specification of eligibility, treatment strategies, outcomes, and follow-up using structured YAML protocols
- **Causal Inference Engine**: Inverse Probability Weighting (IPW), Augmented IPW (AIPW), propensity score matching, and stratification
- **Survival Analysis**: Kaplan-Meier, Cox Proportional Hazards, Restricted Mean Survival Time (RMST)
- **Sensitivity Analysis**: E-value calculation, tipping point analysis for unmeasured confounding
- **Cost-Effectiveness**: Incremental Cost-Effectiveness Ratio (ICER) and Net Monetary Benefit (NMB) analysis
- **EHR Processing**: Automated data cleaning, variable mapping, and missing data handling via MICE

## Quick Start



## API Endpoints

| Method | Path                      | Description                      |
|--------|---------------------------|----------------------------------|
| GET    | /api/protocols            | List trial protocols             |
| POST   | /api/protocols            | Create a new protocol            |
| POST   | /api/analysis/propensity  | Run propensity score analysis    |
| POST   | /api/analysis/survival    | Run survival analysis            |
| POST   | /api/analysis/causal      | Run causal estimation (IPW/AIPW) |
| POST   | /api/analysis/sensitivity | Run sensitivity analysis         |
| GET    | /health                   | Health check                     |

## License

MIT
