"""Build the modelling table from Jolpica + Open-Meteo.

Every derived feature here answers the question "what was knowable *before* this
race?". Two mechanisms enforce that, and they catch different mistakes:

* :mod:`f1predict.data.contracts` catches *column-level* leaks -- a post-race
  field being handed to a model at all.
* The ``_shifted`` helpers below catch *row-level* leaks -- a rolling statistic
  that includes the very race it is used to predict. Every expanding/rolling
  aggregate in this module is shifted by one race within its group, without
  exception.

Championship standings are derived from results rather than fetched. The
standings endpoints would cost roughly one request per round per season, which
does not fit the 200/hour budget; cumulative points from race + sprint results
give the same information for a handful of requests. `validate_standings` checks
that derivation against the API's own end-of-season table.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd

from .jolpica import JolpicaClient
from .weather import WeatherClient

log = logging.getLogger(__name__)

#: Modelling window start. Earlier seasons are still fetched to warm up
#: career/form features, but are not emitted as training rows.
DEFAULT_START_SEASON = 2014

#: Seasons of lead-in fetched before ``start_season`` so that form and career
#: features are populated rather than null on the first modelled race.
WARMUP_SEASONS = 4


# ---------------------------------------------------------------------------
# Flattening the API payloads
# ---------------------------------------------------------------------------


def _parse_lap_time(value: str | None) -> float | None:
    """'1:23.456' -> 83.456 seconds. Returns None on blanks or junk."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except (ValueError, TypeError):
        return None


def flatten_results(races: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        for entry in race.get("Results", []):
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_date": race["date"],
                    "race_time_utc": race.get("time"),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "circuit_lat": float(race["Circuit"]["Location"]["lat"]),
                    "circuit_lon": float(race["Circuit"]["Location"]["long"]),
                    "driver_id": entry["Driver"]["driverId"],
                    "driver_nationality": entry["Driver"].get("nationality"),
                    "driver_dob": entry["Driver"].get("dateOfBirth"),
                    "constructor_id": entry["Constructor"]["constructorId"],
                    "grid": _to_int(entry.get("grid")),
                    "finish_position": _to_int(entry.get("position")),
                    "position_text": entry.get("positionText"),
                    "points": float(entry.get("points", 0) or 0),
                    "status": entry.get("status"),
                    "laps": _to_int(entry.get("laps")),
                }
            )
    return pd.DataFrame(rows)


def flatten_qualifying(races: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        for entry in race.get("QualifyingResults", []):
            # Best available session time: Q3 beats Q2 beats Q1.
            best = None
            for session in ("Q3", "Q2", "Q1"):
                best = _parse_lap_time(entry.get(session))
                if best is not None:
                    break
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": entry["Driver"]["driverId"],
                    "quali_position": _to_int(entry.get("position")),
                    "quali_time_seconds": best,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Gap to pole is only meaningful within a single session.
    pole = df.groupby(["season", "round"])["quali_time_seconds"].transform("min")
    df["quali_gap_to_pole"] = df["quali_time_seconds"] - pole
    df["is_pole"] = (df["quali_position"] == 1).astype(int)
    return df


def flatten_sprint(races: Iterable[dict]) -> pd.DataFrame:
    """Sprint points count toward the championship, so standings need them."""
    rows = []
    for race in races:
        for entry in race.get("SprintResults", []):
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": entry["Driver"]["driverId"],
                    "sprint_points": float(entry.get("points", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Derived pre-race features
# ---------------------------------------------------------------------------


def _shifted_expanding_mean(df: pd.DataFrame, group: list[str], column: str) -> pd.Series:
    """Expanding mean over strictly prior rows within ``group``."""
    return df.groupby(group, sort=False)[column].transform(
        lambda s: s.shift(1).expanding().mean()
    )


def _shifted_rolling_mean(
    df: pd.DataFrame, group: list[str], column: str, window: int
) -> pd.Series:
    """Rolling mean over the ``window`` rows strictly preceding each row."""
    return df.groupby(group, sort=False)[column].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def _shifted_cumsum(df: pd.DataFrame, group: list[str], column: str) -> pd.Series:
    return df.groupby(group, sort=False)[column].transform(
        lambda s: s.shift(1).cumsum()
    )


#: Columns produced by add_championship_state / add_form_features / add_targets.
#: Dropped before recomputation so the pipeline is idempotent -- update.py and
#: predict.py both re-run it over a table that already contains them, and a
#: merge onto an existing column silently produces `_x`/`_y` suffixes instead.
DERIVED_COLUMNS = [
    "total_race_points", "is_win",
    "driver_points_before", "driver_wins_before", "driver_standings_pos_before",
    "constructor_points_before", "constructor_wins_before",
    "constructor_standings_pos_before",
    "constructor_round_points", "constructor_round_wins",
    "did_not_finish", "driver_form_3", "driver_form_5", "driver_dnf_rate_5",
    "constructor_form_3", "constructor_form_5", "driver_career_starts",
    "driver_circuit_mean_finish", "constructor_circuit_mean_finish",
    "driver_age", "is_winner", "is_podium",
]


def _drop_derived(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DERIVED_COLUMNS if c in df.columns]
    return df.drop(columns=present) if present else df


def add_championship_state(df: pd.DataFrame) -> pd.DataFrame:
    """Points, wins and standings position *entering* each round.

    Derived from race + sprint points rather than the standings endpoints, which
    would blow the request budget. Shifted by one round within each season, so a
    driver's row never sees the points they are about to score.

    Idempotent: any previously computed derived columns are discarded first.
    """
    df = _drop_derived(df)
    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    df["total_race_points"] = df["points"].fillna(0) + df.get(
        "sprint_points", pd.Series(0, index=df.index)
    ).fillna(0)
    df["is_win"] = (df["finish_position"] == 1).astype(float)

    # Driver championship state entering the round.
    df["driver_points_before"] = _shifted_cumsum(
        df, ["season", "driver_id"], "total_race_points"
    ).fillna(0.0)
    df["driver_wins_before"] = _shifted_cumsum(
        df, ["season", "driver_id"], "is_win"
    ).fillna(0.0)

    # Constructor state: aggregate both cars, then broadcast back to drivers.
    per_round = (
        df.groupby(["season", "round", "constructor_id"], as_index=False)
        .agg(constructor_round_points=("total_race_points", "sum"),
             constructor_round_wins=("is_win", "sum"))
        .sort_values(["season", "round"])
    )
    per_round["constructor_points_before"] = _shifted_cumsum(
        per_round, ["season", "constructor_id"], "constructor_round_points"
    ).fillna(0.0)
    per_round["constructor_wins_before"] = _shifted_cumsum(
        per_round, ["season", "constructor_id"], "constructor_round_wins"
    ).fillna(0.0)

    df = df.merge(
        per_round[
            [
                "season",
                "round",
                "constructor_id",
                "constructor_points_before",
                "constructor_wins_before",
            ]
        ],
        on=["season", "round", "constructor_id"],
        how="left",
    )

    # Standings position = rank on points entering the round, wins as tiebreak.
    # F1's official tiebreak is a countback on best finishes; points-then-wins
    # agrees with it in all but a handful of cases. `validate_standings` reports
    # how often they disagree.
    df["driver_standings_pos_before"] = (
        df.groupby(["season", "round"], sort=False)
        .apply(
            lambda g: (
                g["driver_points_before"] * 1000 + g["driver_wins_before"]
            ).rank(ascending=False, method="min"),
            include_groups=False,
        )
        .reset_index(level=[0, 1], drop=True)
    )
    df["constructor_standings_pos_before"] = (
        df.groupby(["season", "round"], sort=False)
        .apply(
            lambda g: (
                g["constructor_points_before"] * 1000 + g["constructor_wins_before"]
            ).rank(ascending=False, method="dense"),
            include_groups=False,
        )
        .reset_index(level=[0, 1], drop=True)
    )
    return df


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling form, experience and circuit history -- all strictly backward."""
    df = df.sort_values(["race_date", "round", "driver_id"]).reset_index(drop=True)

    df["did_not_finish"] = (
        ~df["status"].fillna("").str.contains("Finished|\\+", regex=True)
    ).astype(float)

    df["driver_form_3"] = _shifted_rolling_mean(
        df, ["driver_id"], "finish_position", 3
    )
    df["driver_form_5"] = _shifted_rolling_mean(
        df, ["driver_id"], "finish_position", 5
    )
    df["driver_dnf_rate_5"] = _shifted_rolling_mean(
        df, ["driver_id"], "did_not_finish", 5
    )
    df["constructor_form_3"] = _shifted_rolling_mean(
        df, ["constructor_id"], "finish_position", 3
    )
    df["constructor_form_5"] = _shifted_rolling_mean(
        df, ["constructor_id"], "finish_position", 5
    )

    # Races started before this one, counted from the earliest fetched season
    # (start_season - WARMUP_SEASONS), NOT from a driver's true debut. For
    # drivers whose career predates the fetch window this understates
    # experience, and the value shifts if WARMUP_SEASONS changes. It is a
    # within-window experience proxy, not a career total.
    df["driver_career_starts"] = (
        df.groupby("driver_id", sort=False).cumcount().astype(float)
    )

    # Circuit affinity, prior visits only.
    df["driver_circuit_mean_finish"] = _shifted_expanding_mean(
        df, ["driver_id", "circuit_id"], "finish_position"
    )
    df["constructor_circuit_mean_finish"] = _shifted_expanding_mean(
        df, ["constructor_id", "circuit_id"], "finish_position"
    )

    # Age at race day.
    dob = pd.to_datetime(df["driver_dob"], errors="coerce")
    race_day = pd.to_datetime(df["race_date"], errors="coerce")
    df["driver_age"] = (race_day - dob).dt.days / 365.25

    return df


def normalise_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a pit-lane start into a back-of-grid position.

    Ergast (and Jolpica after it) encode a pit-lane start as ``grid = 0``. Left
    alone that sorts *ahead* of pole under any "lower grid is better" rule, so
    every consumer -- the pole-sitter baseline included -- would treat a pit-lane
    start as the best possible position. Map it to one past the back of that
    race's grid instead.
    """
    grid = pd.to_numeric(df["grid"], errors="coerce")
    back_of_grid = (
        grid.where(grid > 0).groupby([df["season"], df["round"]]).transform("max") + 1
    )
    df["grid"] = grid.where(grid > 0, back_of_grid)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["is_winner"] = (df["finish_position"] == 1).astype(int)
    df["is_podium"] = (df["finish_position"] <= 3).astype(int)
    return df


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


def attach_weather(df: pd.DataFrame, weather: WeatherClient) -> pd.DataFrame:
    """One request per race (not per driver), joined back onto every entry."""
    races = df[
        ["season", "round", "race_date", "race_time_utc", "circuit_lat", "circuit_lon"]
    ].drop_duplicates(subset=["season", "round"])

    records = []
    for row in races.itertuples(index=False):
        try:
            race_day = date.fromisoformat(str(row.race_date))
        except (ValueError, TypeError):
            continue
        summary = weather.race_weather(
            latitude=row.circuit_lat,
            longitude=row.circuit_lon,
            race_date=race_day,
            race_time_utc=row.race_time_utc,
        )
        records.append({"season": row.season, "round": row.round, **summary})

    if not records:
        return df
    return df.merge(pd.DataFrame(records), on=["season", "round"], how="left")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_dataset(
    client: JolpicaClient,
    weather: WeatherClient | None = None,
    start_season: int = DEFAULT_START_SEASON,
    end_season: int | None = None,
) -> pd.DataFrame:
    """Assemble the full modelling table.

    Seasons from ``start_season - WARMUP_SEASONS`` are fetched so that form and
    career features are warm, then trimmed to ``start_season`` on the way out.
    """
    end_season = end_season or datetime.utcnow().year
    fetch_from = start_season - WARMUP_SEASONS

    results, qualifying, sprints = [], [], []
    for season in range(fetch_from, end_season + 1):
        log.info("Fetching season %d", season)
        results.extend(client.results(season))
        qualifying.extend(client.qualifying(season))
        if season >= 2021:  # sprint format introduced in 2021
            sprints.extend(client.sprint(season))

    df = flatten_results(results)
    if df.empty:
        raise RuntimeError(f"no race results returned for {fetch_from}-{end_season}")

    quali_df = flatten_qualifying(qualifying)
    if not quali_df.empty:
        df = df.merge(quali_df, on=["season", "round", "driver_id"], how="left")

    sprint_df = flatten_sprint(sprints)
    if not sprint_df.empty:
        df = df.merge(sprint_df, on=["season", "round", "driver_id"], how="left")

    df = normalise_grid(df)
    df = add_championship_state(df)
    df = add_form_features(df)
    df = add_targets(df)

    # Trim before fetching weather: the warm-up seasons exist only to populate
    # form and career features, and their rows are discarded, so fetching
    # weather for them would be several dozen wasted requests.
    df = df[df["season"] >= start_season].reset_index(drop=True)

    if weather is not None:
        df = attach_weather(df, weather)

    df = attach_practice_features(df)

    return df.sort_values(["race_date", "round", "finish_position"]).reset_index(
        drop=True
    )


def attach_practice_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join cached FastF1 practice pace, if it has been backfilled.

    Optional by design: the practice table is produced by a separate, slow job,
    and the pipeline must still build a dataset without it. Seasons before 2018
    have no FastF1 coverage and simply carry nulls.
    """
    from .practice import FEATURE_CACHE, attach_practice

    try:
        practice = pd.read_parquet(FEATURE_CACHE)
    except (FileNotFoundError, OSError):
        log.info("no practice table at %s; skipping practice features", FEATURE_CACHE)
        practice = None
    return attach_practice(df, practice)


def validate_standings(df: pd.DataFrame, client: JolpicaClient, season: int) -> dict:
    """Compare derived end-of-season points against the API's own standings.

    Derivation is a deliberate trade against the request budget; this quantifies
    what it costs in accuracy rather than assuming it costs nothing.
    """
    payload = client.get(f"{season}/driverStandings", season=season)
    table = payload["MRData"]["StandingsTable"]["StandingsLists"]
    if not table:
        return {"season": season, "checked": 0}

    official = {
        s["Driver"]["driverId"]: float(s["points"]) for s in table[0]["DriverStandings"]
    }

    season_rows = df[df["season"] == season]
    if season_rows.empty:
        return {"season": season, "checked": 0}

    # Take each driver's *own last appearance*, not the season's final round.
    # Mid-season replacements and departures (de Vries 2023, Sargeant 2024,
    # Doohan 2025) have no row in the finale, and reading the finale alone
    # reports them as mismatches when the derivation is in fact correct.
    last = (
        season_rows.sort_values("round")
        .groupby("driver_id", as_index=False)
        .last()
        .set_index("driver_id")
    )
    derived = (last["driver_points_before"] + last["total_race_points"]).to_dict()

    mismatches = {
        driver: (round(derived.get(driver, float("nan")), 1), points)
        for driver, points in official.items()
        if not np.isclose(derived.get(driver, float("nan")), points, atol=0.51)
    }
    return {
        "season": season,
        "checked": len(official),
        "mismatches": len(mismatches),
        "detail": dict(list(mismatches.items())[:5]),
    }
