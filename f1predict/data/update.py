"""Incremental dataset refresh, cheap enough to run weekly in CI.

A cold backfill is ~176 Jolpica requests against a 200/hour limit. Doing that
every week would be wasteful and would sit one API hiccup away from failing, so
CI does not rebuild from scratch.

Instead: the processed table is committed to the repo, and this refetches only
the *current* season, splices those rows in, and recomputes derived features
over the whole history. Completed seasons never change, so nothing is lost, and
the weekly cost is around a dozen requests.

    python -m f1predict.data.update
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backfill import BUILD_INFO_PATH, DATASET_PATH
from .features import (
    add_championship_state,
    add_form_features,
    add_targets,
    attach_weather,
    flatten_qualifying,
    flatten_results,
    flatten_sprint,
    normalise_grid,
)
from .jolpica import JolpicaClient
from .weather import WeatherClient

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "season", "round", "race_date", "race_time_utc", "race_name",
    "circuit_id", "circuit_lat", "circuit_lon",
    "driver_id", "driver_nationality", "driver_dob", "constructor_id",
    "grid", "finish_position", "position_text", "points", "status", "laps",
    "quali_position", "quali_time_seconds", "quali_gap_to_pole", "is_pole",
    "sprint_points",
    "weather_temp_max", "weather_precipitation", "weather_windspeed_max",
    "weather_is_wet",
]


def fetch_season(client: JolpicaClient, season: int) -> pd.DataFrame:
    """Raw rows for one season."""
    df = flatten_results(client.results(season))
    if df.empty:
        return df

    qualifying = flatten_qualifying(client.qualifying(season))
    if not qualifying.empty:
        df = df.merge(qualifying, on=["season", "round", "driver_id"], how="left")

    if season >= 2021:
        sprint = flatten_sprint(client.sprint(season))
        if not sprint.empty:
            df = df.merge(sprint, on=["season", "round", "driver_id"], how="left")
    return df


def update(
    client: JolpicaClient | None = None,
    weather: WeatherClient | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    client = client or JolpicaClient()
    weather = weather or WeatherClient()
    season = season or datetime.now(timezone.utc).year

    if not DATASET_PATH.exists():
        raise SystemExit(
            f"{DATASET_PATH} not found. Run a full backfill first:\n"
            "  python -m f1predict.data.backfill"
        )

    existing = pd.read_parquet(DATASET_PATH)
    fresh = fetch_season(client, season)
    if fresh.empty:
        log.info("no results yet for %d; dataset unchanged", season)
        return existing

    # Keep only raw columns; derived ones are recomputed below so that a change
    # to feature logic propagates to history rather than only to new rows.
    keep = [c for c in RAW_COLUMNS if c in existing.columns]
    history = existing[existing["season"] != season][keep]

    fresh = attach_weather(fresh, weather)
    combined = pd.concat([history, fresh], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=["season", "round", "driver_id"], keep="last"
    )

    combined = normalise_grid(combined)
    combined = add_championship_state(combined)
    combined = add_form_features(combined)
    combined = add_targets(combined)
    combined = combined.sort_values(
        ["race_date", "round", "finish_position"]
    ).reset_index(drop=True)

    combined.to_parquet(DATASET_PATH, index=False)

    last = combined.sort_values("race_date").iloc[-1]
    info = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(combined)),
        "races": int(combined.groupby(["season", "round"]).ngroups),
        "last_race": {
            "season": int(last["season"]),
            "round": int(last["round"]),
            "name": str(last["race_name"]),
            "date": str(last["race_date"]),
        },
        "jolpica_requests": client.requests_made,
        "weather_requests": weather.requests_made,
    }
    BUILD_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info(
        "dataset now %d rows through %s (%d jolpica requests)",
        info["rows"],
        info["last_race"]["date"],
        info["jolpica_requests"],
    )
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = update(season=args.season)
    last = df.sort_values("race_date").iloc[-1]
    print(f"{len(df)} rows, latest: {last['race_name']} ({last['race_date']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
