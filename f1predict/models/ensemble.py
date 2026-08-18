"""Combining the individual approaches.

Two things are combined separately, because they are different quantities:

* **Ordering** -- component scores live on incomparable scales (a LambdaRank
  margin, an SVM's Platt probability, a Plackett-Luce log-strength), so they are
  standardised *within each race* before averaging. Averaging raw scores would
  simply hand the ranking to whichever model has the widest spread.
* **Probability** -- averaged as a mixture over the components' own
  distributions. A mixture is never worse calibrated than its worst member and
  is usually better than the average member, which matters because the site
  publishes these numbers.

Per the project's adoption rule this ships only if it beats the individual
models it is built from, on held-out data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import RaceModel


def _zscore_within_race(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros(len(scores))
    values = np.where(finite, scores, np.nan)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std <= 0:
        return np.zeros(len(scores))
    return np.nan_to_num((values - mean) / std, nan=0.0)


class EnsembleModel(RaceModel):
    """Weighted combination of other :class:`RaceModel` instances."""

    def __init__(
        self,
        components: list[RaceModel],
        weights: list[float] | None = None,
        name: str = "ensemble",
        view: str = "post_quali",
    ):
        if not components:
            raise ValueError("ensemble needs at least one component")
        self.components = components
        self.weights = np.asarray(
            weights if weights is not None else [1.0] * len(components), dtype=float
        )
        if len(self.weights) != len(components):
            raise ValueError("weights and components must be the same length")
        self.weights = self.weights / self.weights.sum()
        self.name = name
        self.view = view

    def fit(self, train: pd.DataFrame) -> "EnsembleModel":
        for component in self.components:
            component.fit(train)
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        stacked = np.vstack(
            [
                _zscore_within_race(
                    np.asarray(c.predict_scores(race), dtype=float)
                )
                for c in self.components
            ]
        )
        return self.weights @ stacked

    def predict_proba(self, race: pd.DataFrame) -> np.ndarray:
        stacked = np.vstack(
            [np.asarray(c.predict_proba(race), dtype=float) for c in self.components]
        )
        mixture = self.weights @ stacked
        total = mixture.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(len(race), 1.0 / max(len(race), 1))
        return mixture / total


def build_ensemble(view: str = "post_quali", name: str = "ensemble") -> EnsembleModel:
    from .choice import PlackettLuceModel
    from .original import OriginalMLP
    from .ranker import LambdaRankModel

    return EnsembleModel(
        components=[
            OriginalMLP(view=view),
            LambdaRankModel(view=view),
            PlackettLuceModel(view=view),
        ],
        name=name,
        view=view,
    )
