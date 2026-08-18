"""The original 2023 approach, preserved.

Per-driver binary classification of "did this driver win", with the field ranked
by predicted win probability. Architectures and hyperparameters are exactly
those the original notebook settled on:

    MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu',
                  solver='adam', alpha=0.0001, learning_rate_init=0.001,
                  max_iter=5000, random_state=1)
    SVC(probability=True, gamma=0.1, C=10.0, kernel='sigmoid')

Exactly one thing changed: features now come through the leak guard, so the
four ``status_*`` columns encoding whether a driver finished *the race being
predicted* are no longer inputs. :class:`LeakyOriginalSVM` reproduces the old
configuration so the cost of that defect can be measured rather than asserted.

A structural note for the write-up: this approach treats each driver as an
independent binary problem and then takes an argmax, so its per-driver
"probabilities" do not sum to 1 across the grid. It can rank a field, but its
probabilities are not calibrated as win probabilities. That is a limitation of
the method, not of the fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .base import RaceModel
from .encoding import FieldEncoder

TARGET = "is_winner"


class _PerDriverClassifier(RaceModel):
    """Shared plumbing: fit a binary classifier, rank the field by P(win)."""

    requires_fit = True

    def __init__(self, view: str = "post_quali"):
        self.view = view
        self.encoder: FieldEncoder | None = None
        self.model = None
        self._fallback_score: float = 0.0

    def _make_estimator(self):
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> "_PerDriverClassifier":
        self.encoder = FieldEncoder(view=self.view, scale=True)
        X = self.encoder.fit_transform(train)
        y = train[TARGET].to_numpy(dtype=int)

        self.model = self._make_estimator()
        with warnings.catch_warnings():
            # These models are preserved as specified; convergence chatter over
            # 100+ refits would drown the backtest log.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X, y)
        self._fallback_score = float(y.mean())
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.encoder is None:
            return np.full(len(race), self._fallback_score)
        X = self.encoder.transform(race)
        proba = self.model.predict_proba(X)
        # Guard against a fold where only one class was present in training.
        if proba.shape[1] < 2:
            return np.full(len(race), self._fallback_score)
        return proba[:, 1]

    def predict_proba(self, race: pd.DataFrame) -> np.ndarray:
        """Normalise the per-driver win probabilities over the field.

        The base class softmaxes, which is right for models whose scores are
        log-scale. These scores are already probabilities in [0, 1], and
        softmaxing them would compress a genuine 0.30-vs-0.02 gap into near
        uniformity -- exp(0.30)/exp(0.02) is barely above 1.

        Normalising is the honest reading, but note what it papers over: these
        are independent per-driver estimates that happen to be rescaled to sum
        to 1, not a distribution the model ever reasoned about jointly. If they
        sum to far from 1 before rescaling, the model is poorly calibrated and
        the rescaling hides it.
        """
        scores = np.asarray(self.predict_scores(race), dtype=float)
        scores = np.where(np.isfinite(scores), np.clip(scores, 0.0, 1.0), 0.0)
        total = scores.sum()
        if total <= 0:
            return np.full(len(race), 1.0 / max(len(race), 1))
        return scores / total


class OriginalMLP(_PerDriverClassifier):
    name = "original: MLP"

    def _make_estimator(self):
        return MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=5000,
            random_state=1,
        )


class OriginalSVM(_PerDriverClassifier):
    name = "original: SVM"

    def _make_estimator(self):
        return SVC(probability=True, gamma=0.1, C=10.0, kernel="sigmoid", random_state=1)


class OriginalSVMTuned(_PerDriverClassifier):
    """The original SVM approach, retuned for the rebuilt feature space.

    The shipped `gamma=0.1, C=10.0, kernel='sigmoid'` were grid-searched against
    the original 88-column mostly-sparse matrix. In the rebuilt 25-column dense
    space they collapse: top-1 falls to 0.107 while rank correlation stays at
    0.54, which is the signature of a kernel mismatch rather than a method that
    does not work.

    Re-searching over 21 configurations on 2020-21 (seasons >= 2022 never
    loaded) selects `rbf, C=10.0, gamma='scale'`, which scores 0.538 against
    0.107 on the same inner validation. The original SVM is *kept unchanged* as
    `OriginalSVM`; this is the same approach improved in place, reported beside
    it rather than in place of it.
    """

    name = "original: SVM (retuned)"

    def _make_estimator(self):
        return SVC(probability=True, kernel="rbf", C=10.0, gamma="scale", random_state=1)


class LeakyOriginalSVM(RaceModel):
    """The original SVM *with the post-race leak reinstated*.

    Deliberately bypasses the leak guard to reproduce the 2023 configuration,
    in which ``status_Finished`` / ``status_Incident`` / ``status_Illness`` /
    ``status_Mechanical Issue`` were part of the feature matrix. Those encode
    whether the driver finished the race being predicted.

    This exists to measure the defect, not to compete. It is reported in the
    results table as an artifact and is never eligible for adoption or for
    making a prediction about a real upcoming race -- at prediction time the
    values it depends on do not exist.
    """

    name = "original: SVM (leaky, artifact)"
    view = "post_quali"
    requires_fit = True

    def __init__(self) -> None:
        self.encoder: FieldEncoder | None = None
        self.model: SVC | None = None
        self._fallback_score = 0.0

    @staticmethod
    def _status_frame(df: pd.DataFrame) -> pd.DataFrame:
        """One-hot the finishing status, as the original pipeline did."""
        status = df.get("status")
        if status is None:
            return pd.DataFrame(index=df.index)
        text = status.fillna("").astype(str)
        return pd.DataFrame(
            {
                "status_Finished": text.str.contains("Finished|^\\+", regex=True).astype(
                    float
                ),
                "status_Incident": text.str.contains(
                    "Accident|Collision|Spun|Damage", regex=True
                ).astype(float),
                "status_Illness": text.str.contains("Illness|Injury", regex=True).astype(
                    float
                ),
            },
            index=df.index,
        ).assign(
            **{
                "status_Mechanical Issue": lambda d: (
                    1.0 - d[["status_Finished", "status_Incident", "status_Illness"]].max(
                        axis=1
                    )
                )
            }
        )

    def _matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        clean = (
            self.encoder.fit_transform(df) if fit else self.encoder.transform(df)
        )
        return np.hstack([clean, self._status_frame(df).to_numpy(dtype=float)])

    def fit(self, train: pd.DataFrame) -> "LeakyOriginalSVM":
        self.encoder = FieldEncoder(view=self.view, scale=True)
        X = self._matrix(train, fit=True)
        y = train[TARGET].to_numpy(dtype=int)
        self.model = SVC(
            probability=True, gamma=0.1, C=10.0, kernel="sigmoid", random_state=1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X, y)
        self._fallback_score = float(y.mean())
        return self

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.encoder is None:
            return np.full(len(race), self._fallback_score)
        proba = self.model.predict_proba(self._matrix(race, fit=False))
        if proba.shape[1] < 2:
            return np.full(len(race), self._fallback_score)
        return proba[:, 1]


def build_original_models(view: str = "post_quali") -> list[RaceModel]:
    return [OriginalMLP(view=view), OriginalSVM(view=view)]
