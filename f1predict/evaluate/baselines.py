"""Naive baselines.

These are the bar. A model that does not clear them is not doing anything a
one-line rule could not, and saying so plainly is more useful than a headline
accuracy with nothing to compare it against.

Measured on 2022-2026 (103 races), the strongest of these is "the pole sitter
wins" at roughly 0.59 top-1 -- and it swings between 0.46 and 0.73 across
individual seasons, which is why the backtest spans several seasons and reports
confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.base import FunctionBaseline, RaceModel


class PreviousWinnerBaseline(RaceModel):
    """Predict whoever won the most recent race in the training data."""

    name = "naive: previous winner"
    view = "pre_quali"
    requires_fit = True

    def __init__(self) -> None:
        self._last_winner: str | None = None

    def fit(self, train: pd.DataFrame) -> "PreviousWinnerBaseline":
        finished = train[train["finish_position"] == 1]
        if finished.empty:
            self._last_winner = None
            return self
        latest = finished.sort_values(["race_date", "round"]).iloc[-1]
        self._last_winner = str(latest["driver_id"])
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        if self._last_winner is None:
            return np.zeros(len(race))
        return (race["driver_id"].astype(str) == self._last_winner).to_numpy(float)


class RandomBaseline(RaceModel):
    """Uniformly random ordering. The true floor: ~1/20 per race."""

    name = "naive: random"
    view = "pre_quali"
    requires_fit = False

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def fit(self, train: pd.DataFrame) -> "RandomBaseline":
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        return self._rng.random(len(race))


def build_baselines(view: str = "post_quali") -> list[RaceModel]:
    """Baselines admissible for a given feature view.

    ``pre_quali`` drops the pole-sitter rule, because on the Tuesday before a
    race there is no grid yet -- which is exactly what makes pre-weekend
    prediction the harder problem.
    """
    models: list[RaceModel] = [
        FunctionBaseline(
            "naive: championship leader",
            "driver_points_before",
            higher_is_better=True,
            view="pre_quali",
        ),
        FunctionBaseline(
            "naive: most wins so far",
            "driver_wins_before",
            higher_is_better=True,
            view="pre_quali",
        ),
        FunctionBaseline(
            "naive: best recent form",
            "driver_form_5",
            higher_is_better=False,
            view="pre_quali",
        ),
        FunctionBaseline(
            "naive: strongest constructor",
            "constructor_form_5",
            higher_is_better=False,
            view="pre_quali",
        ),
        PreviousWinnerBaseline(),
        RandomBaseline(),
    ]
    if view == "post_quali":
        # The one to beat.
        models.insert(
            0,
            FunctionBaseline(
                "naive: pole sitter wins",
                "grid",
                higher_is_better=False,
                view="post_quali",
            ),
        )
    return models


def build_qualifying_baselines() -> list[RaceModel]:
    """Baselines for predicting qualifying itself."""
    return [
        FunctionBaseline(
            "naive: championship leader",
            "driver_points_before",
            higher_is_better=True,
            view="pre_quali",
        ),
        FunctionBaseline(
            "naive: strongest constructor",
            "constructor_form_5",
            higher_is_better=False,
            view="pre_quali",
        ),
        FunctionBaseline(
            "naive: best recent form",
            "driver_form_5",
            higher_is_better=False,
            view="pre_quali",
        ),
        RandomBaseline(),
    ]
