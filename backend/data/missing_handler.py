"""Missing data handling via Multiple Imputation by Chained Equations (MICE)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


@dataclass
class MICEConfig:
    max_iter: int = 20
    random_state: int = 42
    n_imputations: int = 5
    estimator: object = None
    sample_posterior: bool = True


class MICEHandler:
    def __init__(self, config=None):
        self.config = config or MICEConfig()
        self._imputers = []

    def _create_imputer(self):
        est = self.config.estimator or BayesianRidge()
        return IterativeImputer(estimator=est, max_iter=self.config.max_iter,
                                random_state=self.config.random_state,
                                sample_posterior=self.config.sample_posterior)

    def impute(self, df, numeric_cols=None):
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].values
        datasets = []
        self._imputers = []
        for i in range(self.config.n_imputations):
            imp = self._create_imputer()
            imp.set_params(random_state=self.config.random_state + i)
            X_imp = imp.fit_transform(X)
            self._imputers.append(imp)
            df_imp = df.copy()
            df_imp[numeric_cols] = X_imp
            datasets.append(df_imp)
        return datasets

    def impute_single(self, df, numeric_cols=None):
        return self.impute(df, numeric_cols)[0]

    def pool_estimates(self, estimates, variances):
        m = len(estimates)
        q_bar = np.mean(estimates)
        u_bar = np.mean(variances)
        b = np.var(estimates, ddof=1)
        total_var = u_bar + (1 + 1 / m) * b
        return q_bar, total_var

    def diagnose(self, df):
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        return {
            "total_rows": len(df), "total_missing": int(missing.sum()),
            "pct_missing": float(missing.sum() / df.size * 100),
            "columns": {col: {"n_missing": int(missing[col]), "pct": float(pct[col])}
                        for col in df.columns if missing[col] > 0},
        }
