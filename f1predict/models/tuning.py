"""Hyperparameter search, run strictly inside the training era.

The LightGBM settings shipped in the first version (300 trees, lr 0.05, 15
leaves) were guesses, and the original SVM's were grid-searched against a
feature matrix that no longer exists. Neither had ever been tuned against this
data.

**The search never touches the test window.** Seasons up to 2019 train, 2020-21
validate; 2022-26 — the backtest window — are not loaded at all. So the reported
backtest remains a clean estimate rather than a number the hyperparameters were
chosen to maximise.

    python -m f1predict.models.tuning

Writes `results/tuning.json`. Chosen values are then pasted into the model
defaults, with this module kept so the choice is reproducible and auditable
rather than folklore.
"""

from __future__ import annotations

import itertools
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATASET = Path("data/processed/races.parquet")
OUTPUT = Path("results/tuning.json")

#: Everything at or after this season is the held-out backtest window and is
#: never loaded here.
TEST_WINDOW_START = 2022

INNER_TRAIN_END = 2019
INNER_VALID = (2020, 2021)


def _score_on_races(model, valid: pd.DataFrame) -> dict:
    """Top-1 and podium hit rate over the inner validation races."""
    from ..evaluate.metrics import podium_hit_rate, top1_correct

    top1, podium = [], []
    for _, race in valid.groupby(["season", "round"], sort=False):
        race = race.sample(frac=1.0, random_state=0)  # never rank by row order
        actual = pd.to_numeric(race["finish_position"], errors="coerce").to_numpy(float)
        if not np.isfinite(actual).any():
            continue
        scores = np.asarray(model.predict_scores(race), dtype=float)
        top1.append(top1_correct(scores, actual))
        podium.append(podium_hit_rate(scores, actual))
    return {
        "top1": float(np.nanmean(top1)) if top1 else float("nan"),
        "podium": float(np.nanmean(podium)) if podium else float("nan"),
        "n_races": len(top1),
    }


def tune_lambdarank(train: pd.DataFrame, valid: pd.DataFrame, view: str) -> dict:
    from .ranker import LambdaRankModel

    grid = {
        "n_estimators": [150, 300, 600],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [10, 30, 60],
    }
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LambdaRankModel(view=view, **params).fit(train)
        score = _score_on_races(model, valid)
        results.append({**params, **score})
        log.debug("lambdarank %s -> top1 %.3f", params, score["top1"])

    results.sort(key=lambda r: (-(r["top1"] or 0), -(r["podium"] or 0)))
    return {"best": results[0], "n_configs": len(results), "top5": results[:5]}


def tune_svm(train: pd.DataFrame, valid: pd.DataFrame, view: str) -> dict:
    """Give the original SVM a fair retune in the current feature space.

    Its shipped values (gamma=0.1, C=10.0, sigmoid) were searched against the
    original 88-column sparse matrix; in the rebuilt 25-column dense space its
    top-1 collapses to ~0.11 while its rank correlation stays at 0.54. This
    tests whether the *method* is weak or only those settings are.
    """
    from sklearn.svm import SVC

    from .original import _PerDriverClassifier

    class _TunableSVM(_PerDriverClassifier):
        name = "svm-tuning"

        def __init__(self, view: str, **params):
            super().__init__(view=view)
            self.params = params

        def _make_estimator(self):
            return SVC(probability=True, random_state=1, **self.params)

    grid = {
        "kernel": ["rbf", "sigmoid", "linear"],
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
    }
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        if params["kernel"] == "linear" and params["gamma"] != "scale":
            continue  # gamma is ignored by a linear kernel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _TunableSVM(view=view, **params).fit(train)
        score = _score_on_races(model, valid)
        results.append({**params, **score})

    results.sort(key=lambda r: (-(r["top1"] or 0), -(r["podium"] or 0)))
    return {"best": results[0], "n_configs": len(results), "top5": results[:5]}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = pd.read_parquet(DATASET)

    # Hard guarantee: the backtest window is never seen by the search.
    df = df[df["season"] < TEST_WINDOW_START]
    train = df[df["season"] <= INNER_TRAIN_END]
    valid = df[df["season"].isin(INNER_VALID)]

    print(
        f"search space: train <= {INNER_TRAIN_END} ({len(train)} rows), "
        f"validate {INNER_VALID} ({valid.groupby(['season','round']).ngroups} races). "
        f"Seasons >= {TEST_WINDOW_START} not loaded."
    )

    out = {}
    for view in ("post_quali", "pre_quali"):
        print(f"\n--- {view} ---")
        lam = tune_lambdarank(train, valid, view)
        print(
            f"lambdarank best ({lam['n_configs']} configs): "
            f"top1={lam['best']['top1']:.3f} {({k: lam['best'][k] for k in ('n_estimators','learning_rate','num_leaves','min_child_samples')})}"
        )
        out[f"lambdarank_{view}"] = lam

        if view == "post_quali":
            svm = tune_svm(train, valid, view)
            print(
                f"svm best ({svm['n_configs']} configs): "
                f"top1={svm['best']['top1']:.3f} "
                f"{({k: svm['best'][k] for k in ('kernel','C','gamma')})}"
            )
            out["svm_post_quali"] = svm

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
