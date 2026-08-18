"""Retirement risk, modelled separately from pace.

A race result is two things happening at once: how fast a car is, and whether it
finishes. The pace models conflate them — a driver who is quick but fragile gets
one score, and the model has no way to express "fastest, but a one-in-eight
chance of not seeing the flag."

That matters here because it is not a rare event. Over the 2022-26 window the
pole sitter fails to finish in the top ten in **9.7%** of races, which is a large
share of the gap between the pole-sitter baseline (0.573) and the ~0.883 ceiling
implied by 88% of winners starting in the top three.

So: a separate classifier estimates P(retire) from pre-race features, and the
combined score becomes

    adjusted = pace_score * (1 - P(retire))

Whether that actually helps is an empirical question, and it is answered in the
backtest like everything else. It is entirely plausible that retirement is close
to irreducible noise at this feature resolution, in which case the wrapper adds
variance and gets rejected.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .base import RaceModel
from .encoding import FieldEncoder

#: A classified finish is "Finished" or "+N Lap(s)". Anything else is a retirement.
FINISHED_PATTERN = r"Finished|^\+"


def retirement_label(df: pd.DataFrame) -> np.ndarray:
    status = df.get("status")
    if status is None:
        return np.zeros(len(df))
    text = status.fillna("").astype(str)
    return (~text.str.contains(FINISHED_PATTERN, regex=True)).astype(float).to_numpy()


class RetirementModel:
    """P(this driver fails to finish), from pre-race features only."""

    def __init__(self, view: str = "post_quali"):
        self.view = view
        self.encoder: FieldEncoder | None = None
        self.model: GradientBoostingClassifier | None = None
        self.base_rate = 0.15

    def fit(self, train: pd.DataFrame) -> "RetirementModel":
        self.encoder = FieldEncoder(view=self.view, scale=False)
        X = self.encoder.fit_transform(train)
        y = retirement_label(train)
        self.base_rate = float(y.mean()) if len(y) else 0.15

        if len(np.unique(y)) < 2:
            self.model = None
            return self

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=3, random_state=1
            ).fit(X, y)
        return self

    def predict(self, race: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.encoder is None:
            return np.full(len(race), self.base_rate)
        proba = self.model.predict_proba(self.encoder.transform(race))
        if proba.shape[1] < 2:
            return np.full(len(race), self.base_rate)
        return proba[:, 1]


class ReliabilityAdjusted(RaceModel):
    """Wrap a pace model, discounting each driver by their retirement risk."""

    requires_fit = True

    def __init__(self, base: RaceModel, name: str | None = None):
        self.base = base
        self.view = base.view
        self.name = name or f"{base.name} + reliability"
        self.retirement = RetirementModel(view=base.view)

    def fit(self, train: pd.DataFrame) -> "ReliabilityAdjusted":
        self.base.fit(train)
        self.retirement.fit(train)
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        pace = np.asarray(self.base.predict_scores(race), dtype=float)
        risk = np.clip(np.asarray(self.retirement.predict(race), dtype=float), 0.0, 0.95)

        # Scores from different models live on different scales, and multiplying
        # a possibly-negative margin by a survival probability would flip signs.
        # Convert to a within-race probability first, then discount.
        survival = 1.0 - risk
        probability = self.base.predict_proba(race)
        adjusted = np.asarray(probability, dtype=float) * survival
        total = adjusted.sum()
        return adjusted / total if total > 0 else pace

    def predict_proba(self, race: pd.DataFrame) -> np.ndarray:
        scores = self.predict_scores(race)
        total = scores.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(len(race), 1.0 / max(len(race), 1))
        return scores / total
