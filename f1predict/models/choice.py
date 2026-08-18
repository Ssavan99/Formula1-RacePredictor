"""Plackett-Luce / conditional logit over the field.

A discrete-choice model. Each driver gets a latent strength ``v_i = x_i . beta``
and the probability that driver *i* wins is ``exp(v_i)`` normalised over the
drivers actually on the grid. That normalisation is the point: win probabilities
sum to 1 across the field by construction, because exactly one driver wins.

The original approach cannot do this. Fitting an independent binary classifier
per driver and taking an argmax gives per-driver scores that do not form a
distribution over the field, so they cannot be published as win probabilities
without qualification. For a site whose headline output *is* a probability, that
is a substantive difference rather than a stylistic one.

Plackett-Luce extends the same idea to the whole finishing order: the winner is
chosen from the full field, the runner-up from those remaining, and so on. That
uses far more of each race than "who won" alone -- roughly ten ordered
comparisons per race instead of one -- which matters when the dataset is only a
few hundred races.

This is an old model, deliberately. At this sample size a well-specified linear
choice model is a real competitor to anything heavier, and it is interpretable:
the fitted coefficients are readable as driver and constructor strength.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .base import RaceModel
from .encoding import FieldEncoder

#: Positions beyond this are dominated by attrition order rather than pace, so
#: the likelihood only models the top of the order.
DEFAULT_DEPTH = 10


class PlackettLuceModel(RaceModel):
    name = "plackett-luce"

    def __init__(
        self,
        view: str = "post_quali",
        alpha: float = 1.0,
        depth: int = DEFAULT_DEPTH,
        max_iter: int = 300,
        name: str | None = None,
    ):
        """
        Args:
            alpha: L2 penalty on the coefficients. Features are standardised, so
                this is on a comparable scale across columns.
            depth: how many finishing positions to include in the likelihood.
        """
        self.view = view
        self.alpha = alpha
        self.depth = depth
        self.max_iter = max_iter
        if name:
            self.name = name
        self.encoder: FieldEncoder | None = None
        self.beta: np.ndarray | None = None

    # -- likelihood ---------------------------------------------------------

    @staticmethod
    def _negative_log_likelihood(
        beta: np.ndarray,
        races: list[tuple[np.ndarray, int]],
        alpha: float,
    ) -> tuple[float, np.ndarray]:
        """NLL and its gradient over all races.

        For each race, walk down the finishing order. At step *k* the chosen
        driver is the one who finished *k*-th, and the choice set is everyone
        who has not yet been placed -- the whole rest of the field, not just the
        part of it we bother to model. ``depth`` limits how many steps are
        modelled; it must not limit the choice set, or the denominator would
        pretend the back of the grid was not racing.

        Vectorised over each race with suffix accumulations, so cost is O(n*d)
        per race rather than O(depth*n*d).
        """
        total = 0.0
        gradient = np.zeros_like(beta)

        for X_ordered, depth in races:
            # X_ordered is already in finishing order.
            v = X_ordered @ beta
            shift = v.max()
            e = np.exp(v - shift)

            # suffix_e[k]  = sum_{j >= k} exp(v_j - shift)
            # suffix_ex[k] = sum_{j >= k} exp(v_j - shift) * x_j
            suffix_e = np.cumsum(e[::-1])[::-1]
            suffix_ex = np.cumsum((e[:, None] * X_ordered)[::-1], axis=0)[::-1]

            k = slice(0, depth)
            log_denominator = np.log(suffix_e[k])
            total -= float((v[k] - shift - log_denominator).sum())

            # E_p[x] at each step, minus the driver actually chosen there.
            expected = suffix_ex[k] / suffix_e[k][:, None]
            gradient += (expected - X_ordered[k]).sum(axis=0)

        # L2 penalty.
        total += alpha * 0.5 * float(beta @ beta)
        gradient += alpha * beta
        return total, gradient

    # -- fitting ------------------------------------------------------------

    def _races_from(self, df: pd.DataFrame, X: np.ndarray) -> list:
        races = []
        positions = pd.to_numeric(df["finish_position"], errors="coerce").to_numpy(float)
        offset = 0
        for _, group in df.groupby(["season", "round"], sort=False):
            size = len(group)
            block = X[offset : offset + size]
            block_positions = positions[offset : offset + size]
            offset += size

            finite = np.where(np.isfinite(block_positions))[0]
            if len(finite) < 2:
                continue
            # Full finishing order: the choice set at every step is drawn from
            # all of it. `depth` only caps how many steps are modelled.
            order = finite[np.argsort(block_positions[finite])]
            depth = min(self.depth, len(order) - 1)
            if depth < 1:
                continue
            races.append((block[order], depth))
        return races

    def fit(self, train: pd.DataFrame) -> "PlackettLuceModel":
        ordered = train.sort_values(["race_date", "round"]).reset_index(drop=True)
        self.encoder = FieldEncoder(view=self.view, scale=True)
        X = self.encoder.fit_transform(ordered)

        races = self._races_from(ordered, X)
        if not races:
            self.beta = np.zeros(X.shape[1])
            return self

        result = minimize(
            self._negative_log_likelihood,
            x0=np.zeros(X.shape[1]),
            args=(races, self.alpha),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.beta = result.x
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        if self.beta is None or self.encoder is None:
            return np.zeros(len(race))
        return self.encoder.transform(race) @ self.beta

    def coefficients(self) -> pd.Series:
        """Fitted strength coefficients, largest absolute effect first."""
        if self.beta is None or self.encoder is None:
            return pd.Series(dtype=float)
        series = pd.Series(self.beta, index=self.encoder.feature_names)
        return series.reindex(series.abs().sort_values(ascending=False).index)
