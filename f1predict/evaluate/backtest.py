"""Walk-forward backtesting.

For every race in the test window, each model is fitted on the races that came
strictly before it and then asked to predict that one race. No model ever sees a
row dated on or after the race it is predicting.

This replaces the original protocol -- a single train/test split on one season --
for two reasons:

* One season is ~22 races, so a single-season accuracy has a standard error near
  10 percentage points. Naive baselines on this data swing from 0.46 to 0.73
  across seasons. A single held-out season cannot separate a good model from a
  lucky one.
* The original's second split trained on rounds 1-11 of 2022 and then scored
  across all 22 rounds of 2022, so half the "test" races were in the training
  set. Making the harness own the split removes the chance of that recurring.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..models.base import RaceModel
from .metrics import PROBABILITY_METRICS, RACE_METRICS

log = logging.getLogger(__name__)

DEFAULT_TEST_SEASONS = (2022, 2023, 2024, 2025, 2026)

#: A model needs some history before its first prediction means anything.
MIN_TRAIN_RACES = 40


def race_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Unique races in chronological order."""
    return (
        df[["season", "round", "race_date", "race_name"]]
        .drop_duplicates(subset=["season", "round"])
        .sort_values(["race_date", "round"])
        .reset_index(drop=True)
    )


def walk_forward(
    df: pd.DataFrame,
    models: Sequence[RaceModel],
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
    refit_every: int = 1,
    min_train_races: int = MIN_TRAIN_RACES,
    target: str = "finish_position",
    progress: bool = True,
) -> pd.DataFrame:
    """Score each model on each race in ``test_seasons``.

    Args:
        refit_every: refit after this many races. 1 refits before every race,
            which is the honest default; raise it only if a model is too slow to
            refit 100+ times, and say so in the write-up if you do.
        target: column holding the true finishing order.

    Returns:
        One row per (model, race) with the per-race value of every metric.
    """
    test_seasons = set(test_seasons)
    df = df.sort_values(["race_date", "round"]).reset_index(drop=True)
    races = race_keys(df)
    test_races = races[races["season"].isin(test_seasons)]

    if test_races.empty:
        raise ValueError(f"no races found for seasons {sorted(test_seasons)}")

    log.info(
        "Backtesting %d models over %d races (%s)",
        len(models),
        len(test_races),
        ", ".join(str(s) for s in sorted(test_seasons)),
    )

    rows: list[dict] = []
    last_fit_at: dict[str, int] = {}
    started = time.monotonic()
    shuffler = np.random.default_rng(20260817)

    for counter, race in enumerate(test_races.itertuples(index=False), start=1):
        race_date = race.race_date
        train = df[df["race_date"] < race_date]
        current = df[(df["season"] == race.season) & (df["round"] == race.round)]

        if len(race_keys(train)) < min_train_races or current.empty:
            continue

        # The dataset is stored sorted by finishing position, so row 0 of every
        # race is its winner. Any metric that breaks a tie by index would then
        # read the answer off the row order rather than the prediction -- a
        # baseline that scores 1.0 for one driver and 0.0 for nineteen would
        # appear to rank the rest of the field perfectly. Shuffle each race
        # before scoring, with a fixed seed so runs stay reproducible.
        current = current.sample(frac=1.0, random_state=shuffler.integers(0, 2**31 - 1))

        actual = pd.to_numeric(current[target], errors="coerce").to_numpy(float)
        if not np.isfinite(actual).any():
            continue

        for model in models:
            try:
                # Refit when this model has not been fitted for `refit_every`
                # races. Tracking the race index rather than a counter that is
                # only incremented on the non-fitting branch: that pattern
                # refits every `refit_every + 1` races, so refit_every=1 would
                # silently score half the season with a one-race-stale model.
                if model.requires_fit:
                    previous = last_fit_at.get(model.name)
                    if previous is None or (counter - previous) >= refit_every:
                        model.fit(train)
                        last_fit_at[model.name] = counter

                scores = np.asarray(model.predict_scores(current), dtype=float)
                if len(scores) != len(current):
                    raise ValueError(
                        f"{model.name} returned {len(scores)} scores "
                        f"for {len(current)} drivers"
                    )
                probabilities = np.asarray(model.predict_proba(current), dtype=float)
            except Exception as exc:  # a broken model must not abort the sweep
                # Skipped races make a model's mean an average over only the
                # races it survived. `n_races` in the summary exposes that;
                # never compare two models without checking it.
                log.warning(
                    "%s failed on %s R%s: %s", model.name, race.season, race.round, exc
                )
                continue

            record = {
                "model": model.name,
                "view": model.view,
                "season": int(race.season),
                "round": int(race.round),
                "race_date": race_date,
                "race_name": race.race_name,
                "n_drivers": int(len(current)),
                "predicted_winner": str(
                    current.iloc[
                        int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
                    ]["driver_id"]
                ),
                "actual_winner": _actual_winner(current, actual),
            }
            for name, fn in RACE_METRICS.items():
                record[name] = fn(scores, actual)
            for name, fn in PROBABILITY_METRICS.items():
                record[name] = fn(probabilities, actual)
            rows.append(record)

        if progress and counter % 20 == 0:
            elapsed = time.monotonic() - started
            log.info(
                "  %d/%d races (%.0fs elapsed)", counter, len(test_races), elapsed
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(
            "backtest produced no rows; check min_train_races and the test window"
        )
    return result


def _actual_winner(current: pd.DataFrame, actual: np.ndarray) -> str:
    winners = np.where(actual == 1)[0]
    return str(current.iloc[int(winners[0])]["driver_id"]) if len(winners) else ""


def per_race_metric(per_race: pd.DataFrame, model: str, metric: str) -> np.ndarray:
    """Extract one model's per-race values, ordered by race, for paired tests."""
    subset = per_race[per_race["model"] == model].sort_values(["race_date", "round"])
    return subset[metric].to_numpy(dtype=float)


def align_for_comparison(
    per_race: pd.DataFrame, model_a: str, model_b: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Values for two models over exactly the races both of them scored."""
    a = per_race[per_race["model"] == model_a][
        ["season", "round", metric]
    ].rename(columns={metric: "a"})
    b = per_race[per_race["model"] == model_b][
        ["season", "round", metric]
    ].rename(columns={metric: "b"})
    merged = a.merge(b, on=["season", "round"], how="inner")
    return merged["a"].to_numpy(float), merged["b"].to_numpy(float)
