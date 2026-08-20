"""Shrinkage toward the starting-grid prior.

## Why this exists

A diagnostic on the 103-race backtest window showed something specific about how
LambdaRank loses to "assume the pole sitter wins":

* It agrees with the pole sitter in 46% of races, and is **80.9%** correct there.
* It backs someone else in the other 54%, and is **32.1%** correct there — while
  simply taking the pole sitter would have been right **37.5%** of those races.

So the model is not weak at picking winners. It is *over-confident about
deviating*: every departure from pole costs about 5.4 points of accuracy on
average. It has `grid` as a feature and still under-weights it, because a tree
ensemble splits on whatever reduces training loss and grid is only decisive for
the top slot.

## What it does

Blend the model's within-race distribution with an empirical prior over starting
slots:

    p_final  ∝  (1 - w) * p_model  +  w * P(win | grid slot)

`P(win | grid)` is estimated from the training fold only — never the test race —
and smoothed, so a slot seen a handful of times cannot dominate. At `w = 0` this
is the base model; at `w = 1` it is a soft version of the pole-sitter rule. In
between, the model keeps its confident calls and stops making its marginal ones,
which is exactly the behaviour the diagnostic asks for.

`w` is chosen on a validation fold inside the training era, never on the
backtest window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import RaceModel

#: Laplace-style smoothing on the per-slot win rate: a slot with three
#: observations should not be trusted like one with eighty.
PRIOR_SMOOTHING = 8.0

#: Grid slots beyond this are pooled; the tail is sparse and nearly flat.
MAX_SLOT = 12


class GridAnchored(RaceModel):
    """Wrap a model, shrinking its predictions toward the grid-slot prior."""

    requires_fit = True

    def __init__(self, base: RaceModel, weight: float = 0.5, name: str | None = None):
        self.base = base
        self.weight = float(np.clip(weight, 0.0, 1.0))
        self.view = base.view
        self.name = name or f"{base.name} + grid anchor"
        self._slot_rate: dict[int, float] = {}
        self._base_rate = 0.05

    # -- the prior ----------------------------------------------------------

    def _fit_prior(self, train: pd.DataFrame) -> None:
        grid = pd.to_numeric(train.get("grid"), errors="coerce")
        won = pd.to_numeric(train.get("is_winner"), errors="coerce").fillna(0)
        usable = grid.notna() & (grid > 0)
        if not usable.any():
            return

        slots = np.minimum(grid[usable].to_numpy(int), MAX_SLOT)
        wins = won[usable].to_numpy(float)
        overall = float(wins.mean()) if len(wins) else 0.05
        self._base_rate = overall

        frame = pd.DataFrame({"slot": slots, "won": wins})
        grouped = frame.groupby("slot")["won"].agg(["sum", "count"])
        # Shrink each slot's rate toward the overall rate by its sample size.
        self._slot_rate = {
            int(slot): float(
                (row["sum"] + PRIOR_SMOOTHING * overall)
                / (row["count"] + PRIOR_SMOOTHING)
            )
            for slot, row in grouped.iterrows()
        }

    def _prior_for(self, race: pd.DataFrame) -> np.ndarray:
        grid = pd.to_numeric(race.get("grid"), errors="coerce").to_numpy(float)
        prior = np.array(
            [
                self._slot_rate.get(
                    int(min(slot, MAX_SLOT)) if np.isfinite(slot) and slot > 0 else -1,
                    self._base_rate,
                )
                for slot in grid
            ],
            dtype=float,
        )
        total = prior.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(len(race), 1.0 / max(len(race), 1))
        return prior / total

    # -- model interface ----------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "GridAnchored":
        self.base.fit(train)
        self._fit_prior(train)
        return self

    def predict_proba(self, race: pd.DataFrame) -> np.ndarray:
        model_p = np.asarray(self.base.predict_proba(race), dtype=float)
        model_p = np.where(np.isfinite(model_p), model_p, 0.0)
        if model_p.sum() <= 0:
            model_p = np.full(len(race), 1.0 / max(len(race), 1))
        else:
            model_p = model_p / model_p.sum()

        blended = (1.0 - self.weight) * model_p + self.weight * self._prior_for(race)
        total = blended.sum()
        return blended / total if total > 0 else model_p

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        # Scores and probabilities must agree, or the ranking metrics and the
        # calibration metrics would be describing two different models.
        return self.predict_proba(race)
