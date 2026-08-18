"""Shared feature encoding.

Every model funnels its inputs through :func:`f1predict.data.contracts.
select_features` before anything else happens, so the leak guard is not
something a model can forget to call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.contracts import select_features

CATEGORICAL = ["circuit_id", "constructor_id", "driver_nationality"]


class FieldEncoder:
    """Select admissible features, one-hot the categoricals, optionally scale.

    Categories are learned on the training fold only. A constructor or circuit
    that appears for the first time in a test race becomes all-zeros rather than
    a new column, which keeps the matrix width stable across the walk-forward
    sweep.
    """

    def __init__(self, view: str = "post_quali", scale: bool = True):
        self.view = view
        self.scale = scale
        self.numeric_columns: list[str] = []
        self.categories: dict[str, list[str]] = {}
        self.scaler: StandardScaler | None = None
        self.medians: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "FieldEncoder":
        selected = select_features(train, self.view)

        categorical = [c for c in CATEGORICAL if c in selected.columns]
        self.categories = {
            c: sorted(selected[c].dropna().astype(str).unique().tolist())
            for c in categorical
        }
        self.numeric_columns = [
            c
            for c in selected.columns
            if c not in categorical
            and pd.api.types.is_numeric_dtype(selected[c])
        ]

        numeric = selected[self.numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        self.medians = numeric.median()

        matrix = self._assemble(selected)
        if self.scale:
            self.scaler = StandardScaler().fit(matrix)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        selected = select_features(df, self.view)
        matrix = self._assemble(selected)
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        return matrix

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    def _assemble(self, selected: pd.DataFrame) -> np.ndarray:
        numeric = pd.DataFrame(index=selected.index)
        for column in self.numeric_columns:
            values = (
                pd.to_numeric(selected[column], errors="coerce")
                if column in selected.columns
                else pd.Series(np.nan, index=selected.index)
            )
            fill = (
                self.medians[column]
                if self.medians is not None and column in self.medians
                else 0.0
            )
            numeric[column] = values.fillna(fill if pd.notna(fill) else 0.0)

        blocks = [numeric.to_numpy(dtype=float)]
        for column, levels in self.categories.items():
            present = (
                selected[column].astype(str)
                if column in selected.columns
                else pd.Series("", index=selected.index)
            )
            block = np.zeros((len(selected), len(levels)), dtype=float)
            index = {level: i for i, level in enumerate(levels)}
            for row, value in enumerate(present):
                position = index.get(value)
                if position is not None:
                    block[row, position] = 1.0
            blocks.append(block)

        if not blocks:
            return np.zeros((len(selected), 0))
        return np.hstack(blocks)

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_columns)
        for column, levels in self.categories.items():
            names.extend(f"{column}_{level}" for level in levels)
        return names
