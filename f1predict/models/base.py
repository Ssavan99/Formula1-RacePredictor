"""Common interface for everything that predicts a race.

Baselines and learned models implement the same protocol so the backtester can
score them on identical folds. That symmetry is the point: "is this model better
than assuming the pole sitter wins?" is only answerable if both are run through
the same harness on the same races with the same metrics.

A model produces a *score per driver* within a race, higher meaning more likely
to win. Win probabilities are then a softmax over the field, which enforces the
constraint that exactly one of the drivers present will win. The original
approach -- independent per-driver binary classifiers, argmax over the field --
does not enforce that, and its "probabilities" do not sum to one across the
grid. For a site that publishes probabilities, that matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RaceModel(ABC):
    """Predicts a finishing order for one race at a time."""

    #: Which feature view this model needs. Set to "pre_quali" for models that
    #: must run before qualifying has happened.
    view: str = "post_quali"

    #: Human-readable name used in results tables and on the site.
    name: str = "unnamed"

    #: Set False for baselines that need no training.
    requires_fit: bool = True

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "RaceModel":
        """Fit on all races strictly before the race being predicted."""

    @abstractmethod
    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        """Score each row of a single race. Higher = more likely to win."""

    def predict_proba(self, race: pd.DataFrame) -> np.ndarray:
        """Win probabilities over the field, summing to 1."""
        return _softmax(np.asarray(self.predict_scores(race), dtype=float))

    def predict_order(self, race: pd.DataFrame) -> np.ndarray:
        """Positional indices of the field, best first."""
        return np.argsort(-np.asarray(self.predict_scores(race), dtype=float))


def _softmax(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    if not finite.any():
        return np.full(len(scores), 1.0 / max(len(scores), 1))
    # Missing scores should not win; park them below the field.
    filled = np.where(finite, scores, np.nanmin(scores[finite]) - 1e3)
    shifted = filled - filled.max()
    exp = np.exp(shifted)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(scores), 1.0 / len(scores))
    return exp / total


class FunctionBaseline(RaceModel):
    """A baseline defined by a column to rank on. No fitting required.

    Args:
        name: label for results tables.
        column: column to rank the field by.
        higher_is_better: whether a larger value means a better expected finish.
        view: feature view the column belongs to.
    """

    requires_fit = False

    def __init__(
        self,
        name: str,
        column: str,
        higher_is_better: bool = True,
        view: str = "post_quali",
    ):
        self.name = name
        self.column = column
        self.higher_is_better = higher_is_better
        self.view = view

    def fit(self, train: pd.DataFrame) -> "FunctionBaseline":
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        if self.column not in race.columns:
            return np.zeros(len(race))
        values = pd.to_numeric(race[self.column], errors="coerce").to_numpy(dtype=float)
        return values if self.higher_is_better else -values

    def predict_proba(self, race: pd.DataFrame) -> np.ndarray:
        """Rank-based probabilities, so units do not masquerade as confidence.

        Softmaxing the raw column would make the probability metrics a statement
        about that column's scale rather than about the rule's behaviour:
        ranking by championship points (0-400) and by points/100 give identical
        orderings but log-losses of 13.7 and 2.1. Converting to ranks first
        makes every baseline comparable, at the cost of these being ordinal
        confidences rather than calibrated estimates -- which is all a naive
        rule can honestly claim anyway.
        """
        scores = np.asarray(self.predict_scores(race), dtype=float)
        n = len(scores)
        if n == 0:
            return scores
        finite = np.isfinite(scores)
        if not finite.any():
            return np.full(n, 1.0 / n)
        ranks = pd.Series(np.where(finite, scores, -np.inf)).rank(
            method="average", ascending=False
        )
        # Zipf-like decay over positions: plausible shape, no scale dependence.
        weights = 1.0 / ranks.to_numpy(dtype=float)
        return weights / weights.sum()
