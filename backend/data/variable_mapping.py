"""Variable mapping between EHR fields and analysis variables."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import yaml


@dataclass
class VariableSpec:
    analysis_name: str
    ehr_names: list[str]
    var_type: str
    description: str = ""
    transform: Optional[str] = None
    reference_category: Optional[str] = None


class VariableMapper:
    def __init__(self, specs=None):
        self.specs = specs or []
        self._ehr_to_analysis = {}

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls([VariableSpec(**v) for v in data.get("variables", [])])

    def build_mapping(self, df):
        mapping = {}
        for spec in self.specs:
            for ehr_name in spec.ehr_names:
                if ehr_name in df.columns:
                    mapping[ehr_name] = spec.analysis_name
                    break
        self._ehr_to_analysis = mapping
        return mapping

    def apply(self, df):
        if not self._ehr_to_analysis:
            self.build_mapping(df)
        df = df.rename(columns=self._ehr_to_analysis)
        for spec in self.specs:
            if spec.analysis_name not in df.columns:
                continue
            if spec.transform == "log":
                import numpy as np
                df[spec.analysis_name] = pd.to_numeric(df[spec.analysis_name], errors="coerce")
                df[spec.analysis_name] = df[spec.analysis_name].apply(
                    lambda x: max(x, 0.001) if pd.notna(x) else x)
                df[spec.analysis_name] = np.log(df[spec.analysis_name])
            elif spec.transform == "standardize":
                col = df[spec.analysis_name]
                if col.dtype in (float, int):
                    df[spec.analysis_name] = (col - col.mean()) / col.std()
        return df

    def get_covariate_names(self):
        return [s.analysis_name for s in self.specs if s.var_type not in ("treatment", "outcome")]
