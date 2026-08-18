"""Feature availability contracts — the leak guard.

Every column in the modelling table is registered here with the point in time at
which its value becomes knowable. A feature matrix is then assembled *only* from
columns whose values exist before the moment we claim to be predicting from.

This module exists because the original pipeline shipped `status_Finished`,
`status_Incident`, `status_Mechanical Issue` and `status_Illness` into the model
inputs. Those encode whether the driver finished *the race being predicted*. On
the 2014-2022 table, 681 of 3707 rows have `status_Finished == 0`, and not one of
them is a race winner -- so the model was told which ~18% of the field to rule
out before it predicted anything.

Two design choices make that class of bug hard to repeat:

1. **Fail closed.** A column that is not registered is rejected, not admitted.
   Adding a feature therefore requires stating when it becomes known.
2. **Views are explicit.** `pre_quali` (predicting on Tuesday, no grid) and
   `post_quali` (predicting on Saturday, real grid) are separate contracts, so a
   grid-dependent feature cannot silently leak into a pre-weekend prediction.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence

import pandas as pd


class Availability(Enum):
    """When a column's value becomes knowable."""

    IDENTIFIER = "identifier"
    """Keys and metadata. Never a feature, but safe to carry alongside one."""

    PRE_QUALI = "pre_quali"
    """Known before the race weekend starts: standings entering the round,
    historical form, circuit, weather forecast, driver biography."""

    POST_QUALI = "post_quali"
    """Known only once qualifying has run: grid position, qualifying gaps."""

    POST_RACE = "post_race"
    """Only knowable after the race. Never admissible as a feature."""

    TARGET = "target"
    """What we are trying to predict."""


class LeakageError(AssertionError):
    """Raised when a feature matrix would contain non-predictive information."""


#: Views, in order of increasing information.
VIEWS: dict[str, tuple[Availability, ...]] = {
    "pre_quali": (Availability.PRE_QUALI,),
    "post_quali": (Availability.PRE_QUALI, Availability.POST_QUALI),
}


REGISTRY: dict[str, Availability] = {
    # --- identifiers -------------------------------------------------------
    "season": Availability.IDENTIFIER,
    "round": Availability.IDENTIFIER,
    "race_date": Availability.IDENTIFIER,
    "race_name": Availability.IDENTIFIER,
    "driver_id": Availability.IDENTIFIER,
    "driver_dob": Availability.IDENTIFIER,  # `driver_age` is the usable form
    "race_time_utc": Availability.IDENTIFIER,  # scheduling; used for weather window
    "circuit_lat": Availability.IDENTIFIER,  # join key for weather, not a feature
    "circuit_lon": Availability.IDENTIFIER,
    # Which team a driver is in, and which track they are at, are both known
    # well before the race and were one-hot features in the original model.
    # They are join keys as well as features; the registry classifies them by
    # when they become knowable, which is what the guard cares about.
    "constructor_id": Availability.PRE_QUALI,
    "circuit_id": Availability.PRE_QUALI,
    # --- targets -----------------------------------------------------------
    "finish_position": Availability.TARGET,
    "is_winner": Availability.TARGET,
    "is_podium": Availability.TARGET,
    "quali_position": Availability.TARGET,
    "is_pole": Availability.TARGET,
    # --- known before the weekend -----------------------------------------
    # Championship state *entering* this round. The original scraper already
    # shifted these correctly; we keep that and name them so it is unambiguous.
    "driver_points_before": Availability.PRE_QUALI,
    "driver_wins_before": Availability.PRE_QUALI,
    "driver_standings_pos_before": Availability.PRE_QUALI,
    "constructor_points_before": Availability.PRE_QUALI,
    "constructor_wins_before": Availability.PRE_QUALI,
    "constructor_standings_pos_before": Availability.PRE_QUALI,
    # Biography / experience
    "driver_age": Availability.PRE_QUALI,
    "driver_career_starts": Availability.PRE_QUALI,
    "driver_nationality": Availability.PRE_QUALI,
    # Rolling form over previous races (strictly prior rounds)
    "driver_form_3": Availability.PRE_QUALI,
    "driver_form_5": Availability.PRE_QUALI,
    "constructor_form_3": Availability.PRE_QUALI,
    "constructor_form_5": Availability.PRE_QUALI,
    "driver_dnf_rate_5": Availability.PRE_QUALI,
    # Circuit history for this driver / constructor, prior seasons only
    "driver_circuit_mean_finish": Availability.PRE_QUALI,
    "constructor_circuit_mean_finish": Availability.PRE_QUALI,
    # Weather. Forecast pre-race, reanalysis for historical rows. Both are
    # legitimately available before the race -- this is the one genuinely
    # predictive fact about the future that we can obtain in advance.
    "weather_temp_max": Availability.PRE_QUALI,
    "weather_precipitation": Availability.PRE_QUALI,
    "weather_windspeed_max": Availability.PRE_QUALI,
    "weather_is_wet": Availability.PRE_QUALI,
    # --- known only after qualifying --------------------------------------
    "grid": Availability.POST_QUALI,
    "quali_time_seconds": Availability.POST_QUALI,
    "quali_gap_to_pole": Availability.POST_QUALI,
    # --- BANNED: only knowable after the race ------------------------------
    # These are the columns the original model was reading. Kept in the
    # registry deliberately: naming them is what makes the guard catch them.
    "points": Availability.POST_RACE,
    "status": Availability.POST_RACE,
    "status_Finished": Availability.POST_RACE,
    "status_Illness": Availability.POST_RACE,
    "status_Incident": Availability.POST_RACE,
    "status_Mechanical Issue": Availability.POST_RACE,
    "laps": Availability.POST_RACE,
    "race_time_ms": Availability.POST_RACE,
    "fastest_lap_rank": Availability.POST_RACE,
    "podium": Availability.POST_RACE,  # original column name; ambiguous, banned
    "position_text": Availability.POST_RACE,
    # Intermediates used to *build* the shifted pre-race features. They describe
    # the race being predicted, so they are inputs to feature construction and
    # never features themselves. `driver_dnf_rate_5` is the admissible,
    # shifted form of `did_not_finish`; `driver_points_before` of these points.
    "did_not_finish": Availability.POST_RACE,
    "is_win": Availability.POST_RACE,
    "total_race_points": Availability.POST_RACE,
    # The sprint runs on the Saturday of the same weekend. Treated as post-race
    # rather than post-qualifying: it is part of the weekend's outcome, and the
    # gain from using it does not justify the risk of it standing in for form.
    "sprint_points": Availability.POST_RACE,
}


#: One-hot prefixes expand to many columns; resolve them by prefix.
PREFIX_REGISTRY: dict[str, Availability] = {
    "circuit_id_": Availability.PRE_QUALI,
    "constructor_": Availability.PRE_QUALI,
    "nationality_": Availability.PRE_QUALI,
    "status_": Availability.POST_RACE,
}


def availability_of(column: str) -> Availability | None:
    """Resolve a column to its availability, or ``None`` if unregistered.

    Exact matches win over prefix matches, so ``constructor_points_before`` is
    not captured by the ``constructor_`` one-hot prefix.
    """
    if column in REGISTRY:
        return REGISTRY[column]
    # Longest prefix first, so `status_` beats a hypothetical `stat_`.
    for prefix in sorted(PREFIX_REGISTRY, key=len, reverse=True):
        if column.startswith(prefix):
            return PREFIX_REGISTRY[prefix]
    return None


def classify(columns: Iterable[str]) -> dict[str, Availability | None]:
    return {c: availability_of(c) for c in columns}


def feature_columns(columns: Iterable[str], view: str) -> list[str]:
    """Return the subset of ``columns`` admissible as features for ``view``.

    This is a *selector*, so it is run against a full modelling table that
    legitimately contains targets, identifiers and post-race fields; those are
    excluded rather than treated as errors. The one thing it refuses is a column
    it cannot classify, because an unclassified column is exactly how the
    original ``status_*`` leak got in. Strict validation of an assembled matrix
    is :func:`assert_pre_race`.

    Raises:
        LeakageError: if any column is unregistered (fail closed).
        ValueError: if ``view`` is not a known view.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(VIEWS)}")

    allowed = set(VIEWS[view])
    columns = list(columns)
    resolved = classify(columns)

    unregistered = [c for c, a in resolved.items() if a is None]
    if unregistered:
        raise LeakageError(
            "Unregistered column(s) cannot be used as features: "
            f"{sorted(unregistered)}. Add them to f1predict.data.contracts."
            "REGISTRY with the point in time at which they become knowable. "
            "This guard fails closed on purpose."
        )

    return [c for c in columns if resolved[c] in allowed]


def assert_pre_race(df: pd.DataFrame, view: str = "post_quali") -> None:
    """Assert that every column of ``df`` is admissible as a feature for ``view``.

    Use on an already-assembled feature matrix, immediately before ``fit`` or
    ``predict``. ``feature_columns`` selects; this one refuses.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(VIEWS)}")

    allowed = set(VIEWS[view])
    resolved = classify(df.columns)

    problems: list[str] = []
    for column, availability in sorted(resolved.items()):
        if availability is None:
            problems.append(f"  {column!r}: unregistered")
        elif availability is Availability.POST_RACE:
            problems.append(f"  {column!r}: POST_RACE, only known after the race")
        elif availability is Availability.TARGET:
            problems.append(f"  {column!r}: TARGET, cannot be its own feature")
        elif availability is Availability.IDENTIFIER:
            problems.append(f"  {column!r}: IDENTIFIER, drop before fitting")
        elif availability not in allowed:
            problems.append(
                f"  {column!r}: {availability.value}, not available in view {view!r}"
            )

    if problems:
        raise LeakageError(
            f"Feature matrix is not admissible for view {view!r}:\n"
            + "\n".join(problems)
        )


def select_features(
    df: pd.DataFrame, view: str, extra_drop: Sequence[str] = ()
) -> pd.DataFrame:
    """Select a guaranteed-admissible feature matrix from ``df`` for ``view``.

    This is the only sanctioned way to build model inputs. It selects, then
    re-asserts, so a bug in selection still fails loudly rather than silently
    training on the future.
    """
    keep = [c for c in feature_columns(df.columns, view) if c not in set(extra_drop)]
    out = df[keep].copy()
    assert_pre_race(out, view)
    return out
