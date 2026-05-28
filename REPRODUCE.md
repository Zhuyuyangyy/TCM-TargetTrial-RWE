# REPRODUCE.md - TCM-TargetTrial-RWE

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux / macOS / Windows
- **GPU**: Not required

## Install

```bash
cd TCM-TargetTrial-RWE
pip install -e ".[dev]"
```

Dependencies: fastapi, uvicorn, lifelines, scikit-learn, statsmodels, numpy, pandas, pyyaml, pydantic, scipy

## Smoke Test

```bash
pytest backend/tests/ -v
```

## Run Server

```bash
uvicorn backend.main:app --port 8011 --reload
```

## Expected Outputs

- Target Trial Emulation framework for TCM
- IPW / AIPW / TMLE causal estimators
- Overlap weighting (Li, Morgan & Zaslavsky 2018)
- E-value + tipping point sensitivity analysis
- 4 pre-configured protocol templates for classic TCM formulas

## Known Issues

- Uses synthetic clinical data (no real patient data)
- lifelines may have version compatibility issues
- No hardcoded paths detected
