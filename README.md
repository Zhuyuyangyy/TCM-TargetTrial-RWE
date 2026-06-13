<div align="center">

# TCM-TargetTrial-RWE

**Target Trial Emulation Framework for Traditional Chinese Medicine Using Real-World Evidence**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)

</div>

---

## Overview

TCM-TargetTrial-RWE is a computational framework that applies the target trial emulation paradigm to generate real-world evidence (RCT-quality causal inference) for Traditional Chinese Medicine interventions. The framework implements the Hernan and Robins target trial framework, adapted for the unique characteristics of TCM clinical practice including syndrome-based treatment selection, formula customization, and integration with Western medicine comparators.

The system provides a complete causal inference pipeline: protocol specification, propensity score estimation with multiple model backends, four causal estimation strategies (IPW, AIPW, TMLE, and Overlap Weighting), survival analysis (Kaplan-Meier, Cox PH, RMST), and comprehensive sensitivity analysis for unmeasured confounding (E-value and tipping point methods). Four pre-configured protocol templates for classic TCM formula-disease pairs enable rapid study design.

### Key Research Contributions

- **Target trial emulation for TCM**: First framework to formalize TCM observational studies as emulated target trials following the Hernan & Robins paradigm
- **Multi-method causal estimation**: IPW, AIPW (doubly robust with cross-fitting), TMLE (targeted maximum likelihood with cross-fitting), and Overlap Weighting with unified API
- **TCM-aware protocol design**: Structured representation of TCM formulas, syndrome patterns, and treatment strategies alongside standard clinical variables
- **Comprehensive sensitivity analysis**: E-value and tipping point methods for assessing robustness to unmeasured confounding

> **IMPORTANT**: This framework currently uses synthetic and semi-realistic data for method validation only. All results are for benchmarking causal inference pipelines and do not represent real clinical findings. See the [Important Note](#important-note) section.

---

## Key Features

### Target Trial Protocol Framework
- **Structured protocol specification**: Eligibility criteria, treatment strategies, primary/secondary outcomes, follow-up periods, and adjustment variables
- **TCM formula integration**: Native support for formula names, herb components, dosage, and treatment duration
- **Protocol templates**: 4 pre-configured templates for classic TCM research questions
- **YAML import/export**: Protocols can be defined in YAML and loaded programmatically

### Causal Estimation Engine

| Method | Type | Key Property |
|--------|------|-------------|
| IPW | Inverse Probability Weighting | Stabilized weights, bootstrap SE |
| AIPW | Augmented IPW | Doubly robust, **cross-fitted** (5-fold), IF-based inference |
| TMLE | Targeted MLE | Locally efficient, cross-fitted, EIC-based inference |
| Overlap Weighting | Entropy/Overlap | Natural trimming of extreme PS, targets ATO |

### Propensity Score Analysis
- **Multiple model backends**: Logistic regression, Gradient Boosting, Random Forest
- **IPTW computation**: Stabilized and unstabilized inverse probability weights
- **Propensity score matching**: Caliper-based matching with configurable ratio
- **Balance diagnostics**: Standardized mean differences (SMD) and variance ratios for covariate balance assessment

### Survival Analysis
- **Kaplan-Meier**: Group-stratified survival curves with confidence intervals and median survival
- **Cox Proportional Hazards**: Hazard ratios with 95% CI, concordance index, log-likelihood
- **Restricted Mean Survival Time (RMST)**: Bootstrap-based RMST difference with confidence intervals

### Sensitivity Analysis
- **E-value**: Minimum strength of unmeasured confounding needed to explain away the observed association
- **Tipping point analysis**: Gamma parameter at which the treatment effect becomes null
- **Visual interpretation**: Automated plain-language interpretation of robustness

### Pre-configured Protocol Templates

| ID | Formula | Research Question | Primary Outcome | Effect Measure |
|----|---------|-------------------|-----------------|----------------|
| `buzhong_yiqi_cancer_fatigue` | Buzhong Yiqi Tang | Cancer-related fatigue | Fatigue score change | Mean difference |
| `liuwei_dihuang_dn` | Liuwei Dihuang Wan | Diabetic nephropathy | eGFR decline | Hazard ratio |
| `danshen_aspirin_angina` | Danshen Yin + Aspirin | Stable angina pectoris | Angina frequency | Mean difference |
| `xiao_chaihu_hbv` | Xiao Chaihu Tang + NUC | Chronic hepatitis B | HBV DNA suppression | Risk ratio |

---

## Architecture

```
backend/
  main.py                          FastAPI application entry point
  config.py                        Application settings
  api/
    protocol.py                    Protocol management and template endpoints
    analysis.py                    Analysis endpoints (propensity, causal, survival, sensitivity)
  models/
    trial_protocol.py              Protocol data structures (eligibility, treatment, outcome)
    protocol_templates.py          4 pre-configured protocol templates
    causal_engine.py               IPW, AIPW, TMLE, and CAE estimators
    cost_effectiveness.py          Cost-effectiveness analysis module
  analysis/
    propensity_score.py            PS estimation, IPTW, matching, overlap weighting
    survival.py                    Kaplan-Meier, Cox PH, RMST
    sensitivity.py                 E-value and tipping point analysis
    diagnostics.py                 Balance and diagnostic utilities
    target_trial.py                Target trial emulation orchestrator
  data/
    ehr_processor.py               EHR data processing
    missing_handler.py             Missing data handling
    variable_mapping.py            Variable name mapping
  tests/
    test_api.py                    API endpoint tests
    test_causal.py                 Causal estimator tests
    test_survival.py               Survival analysis tests
data/
  trial_protocols.yaml             YAML protocol definitions
scripts/
  run_full_analysis.py             End-to-end analysis pipeline
  run_simulation.py                Monte Carlo simulation
  run_monte_carlo.py               Monte Carlo convergence analysis
  generate_synthetic_ehr.py        Synthetic EHR data generator
  generate_semireal_ehr.py         Semi-realistic EHR data generator
  quick_smoke_test.py              Quick validation
tests/
  test_smoke.py                    Integration smoke tests
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/protocols/` | GET/POST | Protocol management (list, create) |
| `/api/protocols/templates/list` | GET | List available protocol templates |
| `/api/protocols/templates/{id}` | GET | Get template details |
| `/api/protocols/templates/{id}/load` | POST | Load template as a new protocol |
| `/api/analysis/upload` | POST | Upload EHR data |
| `/api/analysis/propensity` | POST | Propensity score analysis |
| `/api/analysis/causal` | POST | Causal estimation (IPW/AIPW/TMLE) |
| `/api/analysis/survival` | POST | Survival analysis (KM/Cox/RMST) |
| `/api/analysis/sensitivity` | POST | Sensitivity analysis (E-value/tipping point) |
| `/api/analysis/balance` | POST | Covariate balance diagnostics (SMD) |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI + Uvicorn | REST API with async support |
| Causal Inference | scikit-learn | Propensity score models (LR, GBM, RF) |
| Survival Analysis | lifelines | Kaplan-Meier, Cox PH, RMST |
| Statistical Computing | statsmodels, SciPy | Statistical tests and distributions |
| Data Processing | NumPy, Pandas | Numerical computation and data manipulation |
| Configuration | PyYAML, Pydantic | Typed config and validation |
| Testing | pytest, pytest-asyncio | Unit and integration tests |
| CI/CD | GitHub Actions | Lint (ruff) + test pipeline |

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/TCM-TargetTrial-RWE.git
cd TCM-TargetTrial-RWE

# Install in development mode (includes test dependencies)
pip install -e ".[dev]"
```

### Run the Server

```bash
uvicorn backend.main:app --port 8011 --reload
```

The API documentation is available at `http://localhost:8011/docs` (Swagger UI).

### Run a Full Analysis

```bash
# Run the complete target trial emulation pipeline
python scripts/run_full_analysis.py

# Generate synthetic EHR data for testing
python scripts/generate_synthetic_ehr.py
```

### Run Tests

```bash
pytest backend/tests/ -v
```

---

## Benchmarks

### Causal Estimator Comparison

| Property | IPW | AIPW | TMLE | Overlap |
|----------|-----|------|------|---------|
| Doubly Robust | No | Yes | Yes | No |
| Locally Efficient | No | No | Yes | No |
| Cross-fitting | No | Yes (K-fold) | Yes (K-fold) | No |
| Inference Method | Bootstrap | IF-based | EIC-based | IF-based |
| Extreme PS Handling | Trimming | Trimming | Trimming | Natural weighting |
| Target Estimand | ATE | ATE | ATE | ATO |

### Propensity Score Model Options

| Model | Library | Use Case |
|-------|---------|----------|
| Logistic Regression | scikit-learn | Linear treatment assignment |
| Gradient Boosting | scikit-learn | Non-linear relationships |
| Random Forest | scikit-learn | High-dimensional covariates |

### Balance Diagnostics Thresholds

| SMD Range | Interpretation |
|-----------|---------------|
| < 0.1 | Good balance |
| 0.1 - 0.25 | Adequate balance |
| > 0.25 | Poor balance (reweight/rematch recommended) |

---

## Research

### Target Trial Emulation Methodology

The framework implements the complete target trial emulation workflow:

1. **Protocol Specification**: Define eligibility criteria, treatment strategies, outcomes, and follow-up
2. **Data Processing**: Clean EHR data, handle missing values, map variables
3. **Propensity Score Estimation**: Fit treatment assignment model with balance diagnostics
4. **Causal Estimation**: Compute treatment effects using IPW, AIPW, TMLE, or Overlap Weighting
5. **Survival Analysis**: Kaplan-Meier curves, Cox PH, RMST for time-to-event outcomes
6. **Sensitivity Analysis**: E-value and tipping point for unmeasured confounding assessment

### Key References

- Hernan MA, Robins JM. *Causal Inference: What If*. Chapman & Hall/CRC, 2020.
- Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score weighting. *JASA*, 2018.
- van der Laan MJ, Rose S. *Targeted Learning*. Springer, 2011.
- VanderWeele TJ, Ding P. Sensitivity analysis in observational research. *Ann Intern Med*, 2017.

### Citation

If you use this framework in your research, please cite:

```bibtex
@software{tcm_targettrial_rwe,
  title   = {TCM-TargetTrial-RWE: Target Trial Emulation for TCM Using Real-World Evidence},
  year    = {2026},
  url     = {https://github.com/your-org/TCM-TargetTrial-RWE}
}
```

---

## Roadmap

- [ ] Integration with real TCM hospital HIS/EHR systems (currently synthetic data only)
- [ ] **SuperLearner ensemble** for PS and outcome models (currently single learners: GBM, LR, RF)
- [ ] **Clone-censor-weight (CCW)** for immortal time bias correction
- [ ] Landmark analysis for complementary sensitivity to immortal time bias
- [ ] Instrumental variable estimation (2SLS, LATE)
- [ ] Marginal structural models (MSM) for time-varying treatments
- [ ] Bayesian causal inference with informative priors
- [ ] Automated covariate selection via LASSO/elastic net
- [ ] Interactive balance diagnostic dashboard
- [ ] Support for multi-site data with privacy-preserving federated estimation

---

## Project Structure

```
TCM-TargetTrial-RWE/
|-- backend/
|   |-- api/                      # FastAPI route handlers
|   |-- analysis/                 # Core analysis algorithms
|   |-- data/                     # EHR data processing
|   |-- models/                   # Causal engines and protocol models
|   |-- tests/                    # Backend unit tests
|   |-- config.py                 # Application configuration
|   |-- main.py                   # FastAPI app entry point
|-- data/
|   |-- trial_protocols.yaml      # YAML protocol definitions
|-- scripts/
|   |-- run_full_analysis.py      # End-to-end analysis pipeline
|   |-- run_simulation.py         # Monte Carlo simulation
|   |-- run_monte_carlo.py        # Convergence analysis
|   |-- generate_synthetic_ehr.py # Synthetic data generation
|   |-- quick_smoke_test.py       # Quick validation
|-- tests/
|   |-- test_smoke.py             # Integration tests
|-- .github/workflows/ci.yml      # CI pipeline
|-- pyproject.toml                # Project metadata and dependencies
|-- REPRODUCE.md                  # Reproduction instructions
```

---

## Important Note

> **This framework currently uses synthetic and semi-realistic data for method validation and benchmarking only.** All results, including causal estimates, survival analyses, and sensitivity analyses, are computed on simulated data and do not represent real clinical findings. They should not be used for clinical decision-making or cited as evidence of treatment effectiveness.
>
> For publication-grade research, integration with real TCM hospital HIS/EHR systems is required. The causal estimation methods are fully implemented and validated against known analytical properties using synthetic data with known data-generating processes.
>
> **Known limitations of the current release:**
> - Individual learners (GBM, LR, RF) are used for PS and outcome models; **SuperLearner ensemble is planned but not yet implemented**
> - **Immortal time bias** is not addressed; clone-censor-weight (CCW) and landmark methods are planned for a future release
> - E-value computation assumes risk ratio inputs; odds ratios and hazard ratios are accepted with a conversion warning

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, collaboration, or feedback, please open an issue on GitHub or contact the maintainers.

---

<div align="center">

**Generating rigorous real-world evidence for Traditional Chinese Medicine**

</div>
