"""LightGBM LambdaRank.

A race is a ranking problem with a natural grouping: one race is one query, the
drivers are the candidates, and the finishing order is the relevance ordering.
LambdaRank optimises NDCG over that ordering directly, which is a closer match
to the actual task than the original's "classify each driver independently, then
take an argmax".

Gradient-boosted trees are chosen here on the merits rather than the fashion:
the dataset is a few thousand rows of heterogeneous tabular features with
non-linear interactions and missing values, which is the regime where GBDTs
still beat both linear models and neural networks. Learning-to-rank with
LambdaMART has been used for the same shape of problem in cycling and horse
racing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from .base import RaceModel
from .encoding import FieldEncoder

#: Relevance is highest for the winner and decays down the order. Capped below
#: at 0 so the tail of the field is not distinguished by noise.
MAX_RELEVANCE = 20


def relevance_from_position(positions: np.ndarray) -> np.ndarray:
    finite = np.isfinite(positions)
    relevance = np.zeros(len(positions), dtype=int)
    relevance[finite] = np.clip(
        MAX_RELEVANCE + 1 - positions[finite].astype(int), 0, MAX_RELEVANCE
    )
    return relevance


class LambdaRankModel(RaceModel):
    name = "lightgbm: lambdarank"

    def __init__(
        self,
        view: str = "post_quali",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 30,
        random_state: int = 1,
        name: str | None = None,
        target: str = "finish_position",
    ):
        if lgb is None:  # pragma: no cover
            raise ImportError("lightgbm is required for LambdaRankModel")
        self.view = view
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.target = target
        if name:
            self.name = name
        self.encoder: FieldEncoder | None = None
        self.model = None

    def fit(self, train: pd.DataFrame) -> "LambdaRankModel":
        # LightGBM's `group` argument assumes rows are contiguous per group and
        # in the same order as the sizes array. Sort here rather than trusting
        # the caller to have done it.
        ordered = train.sort_values(["race_date", "round"]).reset_index(drop=True)

        # Trees do not need scaling, and leaving values on their natural scale
        # keeps split thresholds interpretable.
        self.encoder = FieldEncoder(view=self.view, scale=False)
        X = self.encoder.fit_transform(ordered)

        group_sizes = (
            ordered.groupby(["season", "round"], sort=False).size().to_numpy()
        )
        positions = pd.to_numeric(
            ordered[self.target], errors="coerce"
        ).to_numpy(float)
        y = relevance_from_position(positions)

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
            # LightGBM's thread pool thrashes on a dataset this small: the same
            # fit takes 4.7s multi-threaded and 0.45s on one thread.
            n_jobs=1,
            force_col_wise=True,
            label_gain=list(range(MAX_RELEVANCE + 1)),
        )
        self.model.fit(X, y, group=group_sizes)
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.encoder is None:
            return np.zeros(len(race))
        return np.asarray(self.model.predict(self.encoder.transform(race)), dtype=float)

    def feature_importance(self) -> pd.Series:
        if self.model is None or self.encoder is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.feature_importances_, index=self.encoder.feature_names
        ).sort_values(ascending=False)
