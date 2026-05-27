"""Electronic Health Record data processor."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class EHRDataset:
    df: pd.DataFrame
    patient_id_col: str
    treatment_col: str
    outcome_col: str
    time_col: Optional[str] = None
    covariate_cols: list[str] = field(default_factory=list)
    n_patients: int = 0
    n_treated: int = 0
    n_control: int = 0


class EHRProcessor:
    def __init__(self, patient_id_col: str = "patient_id"):
        self.patient_id_col = patient_id_col

    def load(self, path):
        p = Path(path)
        if p.suffix == ".csv":
            return pd.read_csv(p)
        elif p.suffix in (".parquet", ".pq"):
            return pd.read_parquet(p)
        raise ValueError(f"Unsupported format: {p.suffix}")

    def basic_cleaning(self, df):
        df = df.drop_duplicates().dropna(subset=[self.patient_id_col])
        if df[self.patient_id_col].duplicated().any():
            df = df.sort_values(self.patient_id_col).drop_duplicates(
                subset=[self.patient_id_col], keep="first")
        return df.reset_index(drop=True)

    def prepare_for_analysis(self, df, treatment_col, outcome_col, covariate_cols, time_col=None):
        needed = [treatment_col, outcome_col] + covariate_cols
        if time_col: needed.append(time_col)
        subset = df[needed + [self.patient_id_col]].dropna().reset_index(drop=True)
        return EHRDataset(
            df=subset, patient_id_col=self.patient_id_col,
            treatment_col=treatment_col, outcome_col=outcome_col,
            time_col=time_col, covariate_cols=covariate_cols,
            n_patients=len(subset), n_treated=int(subset[treatment_col].sum()),
            n_control=int((1 - subset[treatment_col]).sum()),
        )

    def summarize(self, ds):
        df = ds.df
        summary = {"n_patients": ds.n_patients, "n_treated": ds.n_treated,
                    "n_control": ds.n_control, "covariates": {}}
        for col in ds.covariate_cols:
            if df[col].dtype in (float, int):
                summary["covariates"][col] = {
                    "mean": float(df[col].mean()), "sd": float(df[col].std()),
                    "min": float(df[col].min()), "max": float(df[col].max())}
            else:
                summary["covariates"][col] = df[col].value_counts().to_dict()
        return summary
