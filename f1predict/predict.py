"""Predict an upcoming race.

The awkward part of predicting a race that has not happened is that the whole
feature pipeline is built from *results*, and an upcoming race has none. Rather
than write a second, parallel feature builder for inference -- which is how
training and serving drift apart -- this appends placeholder rows for the
upcoming race to the historical table and runs the *same* functions over it.

That works precisely because every derived feature is shifted: the placeholder
rows contribute nothing to their own features, and they sit last
chronologically, so nothing earlier is disturbed. If that stops being true, the
leak guard and the shift tests fail rather than the predictions quietly rotting.

Entrants are taken from the most recent completed race, which is the best
available guess at the grid before a weekend starts.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data.features import (
    add_championship_state,
    add_form_features,
    add_targets,
    normalise_grid,
)
from .data.jolpica import JolpicaClient
from .data.weather import WeatherClient
from .models.base import RaceModel

log = logging.getLogger(__name__)

DATASET = Path("data/processed/races.parquet")
PREDICTIONS_DIR = Path("docs/data/predictions")


def entrants_for(history: pd.DataFrame) -> pd.DataFrame:
    """Driver/constructor pairings from the most recent completed race."""
    latest = history.sort_values(["race_date", "round"]).iloc[-1]
    field = history[
        (history["season"] == latest["season"]) & (history["round"] == latest["round"])
    ]
    return field[["driver_id", "constructor_id", "driver_nationality", "driver_dob"]].copy()


def build_upcoming_rows(
    history: pd.DataFrame,
    race: dict,
    weather: dict[str, float | None],
    grid: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Placeholder rows for an upcoming race, one per entrant."""
    field = entrants_for(history)
    circuit = race["Circuit"]

    rows = field.assign(
        season=int(race["season"]),
        round=int(race["round"]),
        race_date=race["date"],
        race_time_utc=race.get("time"),
        race_name=race["raceName"],
        circuit_id=circuit["circuitId"],
        circuit_lat=float(circuit["Location"]["lat"]),
        circuit_lon=float(circuit["Location"]["long"]),
        # Unknown until the race happens. Left null deliberately: these are
        # registered POST_RACE, so the guard rejects them as features anyway.
        finish_position=np.nan,
        position_text=None,
        points=0.0,
        status=None,
        laps=np.nan,
        quali_position=np.nan,
        quali_time_seconds=np.nan,
        quali_gap_to_pole=np.nan,
        is_pole=np.nan,
        sprint_points=0.0,
    )
    for column, value in weather.items():
        rows[column] = value

    # Grid is known only after qualifying. Before that it is null, and the
    # pre_quali view excludes it.
    rows["grid"] = (
        rows["driver_id"].map(grid).astype(float) if grid else np.nan
    )
    return rows


def prepare_features(history: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Run the training feature pipeline over history + the upcoming race."""
    combined = pd.concat([history, upcoming], ignore_index=True, sort=False)
    combined = combined.sort_values(["race_date", "round"]).reset_index(drop=True)

    combined = normalise_grid(combined)
    combined = add_championship_state(combined)
    combined = add_form_features(combined)
    combined = add_targets(combined)

    season, rnd = int(upcoming.iloc[0]["season"]), int(upcoming.iloc[0]["round"])
    return combined[
        (combined["season"] == season) & (combined["round"] == rnd)
    ].reset_index(drop=True)


def predict_race(
    models: Sequence[RaceModel],
    history: pd.DataFrame,
    race_features: pd.DataFrame,
) -> list[dict]:
    """Fit each model on all history and predict the upcoming race."""
    predictions = []
    for model in models:
        try:
            model.fit(history)
            scores = np.asarray(model.predict_scores(race_features), dtype=float)
            probabilities = np.asarray(model.predict_proba(race_features), dtype=float)
        except Exception as exc:
            log.error("model %s failed to predict: %s", model.name, exc)
            continue

        order = np.argsort(-scores)
        predictions.append(
            {
                "model": model.name,
                "view": model.view,
                "ranking": [
                    {
                        "position": rank,
                        "driver_id": str(race_features.iloc[int(i)]["driver_id"]),
                        "constructor_id": str(
                            race_features.iloc[int(i)]["constructor_id"]
                        ),
                        "win_probability": round(float(probabilities[int(i)]), 4),
                    }
                    for rank, i in enumerate(order, start=1)
                ],
            }
        )
    return predictions


def run(
    view: str = "pre_quali",
    models: Sequence[RaceModel] | None = None,
    client: JolpicaClient | None = None,
    weather_client: WeatherClient | None = None,
    today: date | None = None,
) -> dict | None:
    """Produce predictions for the next race, or ``None`` if there is none."""
    client = client or JolpicaClient()
    weather_client = weather_client or WeatherClient()
    today = today or datetime.now(timezone.utc).date()

    race = client.next_race(today=today)
    if race is None:
        log.warning("no upcoming race found")
        return None

    history = pd.read_parquet(DATASET)

    grid = None
    if view == "post_quali":
        grid = fetch_grid(client, race)
        if not grid:
            log.warning(
                "no qualifying results yet for %s R%s; cannot predict post_quali",
                race["season"],
                race["round"],
            )
            return None

    conditions = weather_client.race_weather(
        latitude=float(race["Circuit"]["Location"]["lat"]),
        longitude=float(race["Circuit"]["Location"]["long"]),
        race_date=date.fromisoformat(race["date"]),
        race_time_utc=race.get("time"),
    )

    upcoming = build_upcoming_rows(history, race, conditions, grid=grid)
    features = prepare_features(history, upcoming)

    if models is None:
        from .models.registry import build_production_models

        models = build_production_models(view=view)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "view": view,
        "race": {
            "season": int(race["season"]),
            "round": int(race["round"]),
            "name": race["raceName"],
            "date": race["date"],
            "circuit": race["Circuit"]["circuitId"],
        },
        "weather": conditions,
        "predictions": predict_race(models, history, features),
    }
    return payload


def fetch_grid(client: JolpicaClient, race: dict) -> dict[str, int]:
    """Starting grid from qualifying, empty if qualifying has not run."""
    try:
        payload = client.get(
            f"{race['season']}/{race['round']}/qualifying", season=race["season"]
        )
    except Exception as exc:
        log.warning("qualifying fetch failed: %s", exc)
        return {}
    races = payload["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    return {
        entry["Driver"]["driverId"]: int(entry["position"])
        for entry in races[0].get("QualifyingResults", [])
    }


def save(payload: dict, directory: Path = PREDICTIONS_DIR) -> Path:
    """Write predictions to a stable, race-scoped path.

    The race-scoped file is the permanent record -- committed before the race,
    so git history evidences that it was not written afterwards. A `latest_*`
    pointer is written alongside it purely so the site can fetch a fixed URL.
    """
    directory.mkdir(parents=True, exist_ok=True)
    race = payload["race"]
    path = directory / f"{race['season']}_{race['round']:02d}_{payload['view']}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    pointer = directory.parent / f"latest_{payload['view']}.json"
    pointer.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
