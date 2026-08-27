"""Practice-session pace from FastF1.

The models are weakest exactly where the value is: 88% of race winners start in
the top three, so the job is discriminating between front-runners, and
championship standings barely separate them. Free practice is the only public
source that says who is genuinely quick *this weekend* — a new floor, an upgrade
that worked, a car that suits the circuit.

Two signals, both expressed as **gaps to the session's best** rather than
absolute lap times, because a lap at Monza and a lap at Monaco are not
comparable:

* **Best-lap gap** — one-lap pace, a qualifying proxy.
* **Long-run gap** — median quick lap over stints of five or more laps,
  excluding in- and out-laps. This is the race-pace proxy and is the more
  informative of the two: single-lap pace is contaminated by fuel loads and
  engine modes in a way that sustained running is not.

Availability, which is what the contract cares about: practice runs Friday and
Saturday morning, so these are known by the time qualifying ends but **not** on
the Tuesday before a race. They are registered `POST_QUALI` and are therefore
absent from pre-weekend predictions by construction.

Coverage starts in 2018; earlier seasons return nothing and the features are
null. Sprint weekends have only one practice session, which is handled rather
than treated as an error.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/fastf1")
FEATURE_CACHE = Path("data/processed/practice.parquet")

#: FastF1 timing coverage is not reliable before this season.
FIRST_SEASON = 2018

SESSIONS = ("FP1", "FP2", "FP3")

#: A stint must be at least this long to count as a representative long run.
MIN_STINT_LAPS = 5

#: Discard laps slower than this multiple of the driver's best (traffic, cool-down).
QUICKLAP_THRESHOLD = 1.07


def _enable_cache() -> bool:
    try:
        import fastf1

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        logging.getLogger("fastf1").setLevel(logging.ERROR)
        return True
    except ImportError:
        log.warning("fastf1 not installed; practice features unavailable")
        return False


def load_session_laps(season: int, rnd: int) -> pd.DataFrame:
    """Every practice lap for one race weekend, across available sessions."""
    import fastf1

    frames = []
    for name in SESSIONS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                session = fastf1.get_session(season, rnd, name)
                session.load(telemetry=False, weather=False, messages=False)
            laps = session.laps
            if laps is None or len(laps) == 0:
                continue
            laps = laps.copy()
            laps["Session"] = name
            frames.append(laps)
        except Exception as exc:
            # Sprint weekends genuinely have fewer sessions; this is expected.
            log.debug("%s %s R%s unavailable: %s", season, name, rnd, exc)
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise_practice(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-driver pace summary for one weekend."""
    if laps.empty or "LapTime" not in laps.columns:
        return pd.DataFrame()

    laps = laps[laps["LapTime"].notna()].copy()
    if laps.empty:
        return pd.DataFrame()

    laps["lap_seconds"] = laps["LapTime"].dt.total_seconds()

    # Exclude in- and out-laps: they are pit-cycle artefacts, not pace.
    if "PitOutTime" in laps.columns and "PitInTime" in laps.columns:
        laps = laps[laps["PitOutTime"].isna() & laps["PitInTime"].isna()]
    if laps.empty:
        return pd.DataFrame()

    rows = []
    for driver, group in laps.groupby("Driver"):
        best = float(group["lap_seconds"].min())

        # Long runs: stints of at least MIN_STINT_LAPS, quick laps only.
        long_run = np.nan
        if "Stint" in group.columns:
            stint_sizes = group.groupby(["Session", "Stint"]).size()
            long_stints = stint_sizes[stint_sizes >= MIN_STINT_LAPS].index
            if len(long_stints) > 0:
                mask = group.set_index(["Session", "Stint"]).index.isin(long_stints)
                stint_laps = group[mask]
                quick = stint_laps[
                    stint_laps["lap_seconds"] <= best * QUICKLAP_THRESHOLD
                ]
                if len(quick) >= 3:
                    long_run = float(quick["lap_seconds"].median())

        rows.append(
            {
                "driver_code": str(driver),
                "practice_best_seconds": best,
                "practice_long_run_seconds": long_run,
                "practice_laps": int(len(group)),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    # Convert to gaps: absolute lap times are not comparable across circuits.
    summary["practice_best_gap"] = (
        summary["practice_best_seconds"] - summary["practice_best_seconds"].min()
    )
    if summary["practice_long_run_seconds"].notna().any():
        summary["practice_long_run_gap"] = (
            summary["practice_long_run_seconds"]
            - summary["practice_long_run_seconds"].min()
        )
    else:
        summary["practice_long_run_gap"] = np.nan

    return summary[
        ["driver_code", "practice_best_gap", "practice_long_run_gap", "practice_laps"]
    ]


def build_practice_table(
    seasons: list[int], calendar: pd.DataFrame, force: bool = False
) -> pd.DataFrame:
    """Practice features for every race in ``seasons``.

    ``calendar`` must carry season/round. Results are cached to parquet because a
    cold build is roughly six seconds per session.
    """
    if not _enable_cache():
        return pd.DataFrame()

    cached = pd.DataFrame()
    if FEATURE_CACHE.exists() and not force:
        cached = pd.read_parquet(FEATURE_CACHE)

    have = set()
    if not cached.empty:
        have = set(zip(cached["season"], cached["round"]))

    wanted = calendar[calendar["season"].isin(seasons)][
        ["season", "round"]
    ].drop_duplicates()

    frames = [cached] if not cached.empty else []
    for row in wanted.itertuples(index=False):
        season, rnd = int(row.season), int(row.round)
        if season < FIRST_SEASON or (season, rnd) in have:
            continue
        laps = load_session_laps(season, rnd)
        summary = summarise_practice(laps)
        if summary.empty:
            continue
        summary["season"] = season
        summary["round"] = rnd
        frames.append(summary)
        log.info("practice %s R%s: %d drivers", season, rnd, len(summary))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["season", "round", "driver_code"], keep="last"
    )
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURE_CACHE, index=False)
    return out


def attach_practice(df: pd.DataFrame, practice: pd.DataFrame) -> pd.DataFrame:
    """Join practice features onto the modelling table.

    FastF1 identifies drivers by three-letter code while Jolpica uses driverId,
    so the join goes through a code derived from the driver's surname. Drivers
    who appear only in a practice session (rookie run-outs) simply do not match
    any race row, which is the desired behaviour.
    """
    feature_columns = ["practice_best_gap", "practice_long_run_gap", "practice_laps"]

    # Idempotent: a caller may pass a frame that already carries these columns
    # from a previous attach (e.g. an incremental update re-deriving over the
    # whole history). Without dropping them first, `merge` suffixes both sides
    # as `_x`/`_y` instead of overwriting, which silently produces columns the
    # leak guard has never heard of.
    df = df.drop(columns=[c for c in feature_columns if c in df.columns])

    if practice is None or practice.empty:
        for column in feature_columns:
            df[column] = np.nan
        return df

    df["_code"] = _driver_code(df["driver_id"])

    merged = df.merge(
        practice.rename(columns={"driver_code": "_code"}),
        on=["season", "round", "_code"],
        how="left",
    )
    return merged.drop(columns=["_code"])


def _driver_code(driver_ids: pd.Series) -> pd.Series:
    """driverId -> FastF1's three-letter code.

    Jolpica ids are usually the surname (`norris`, `max_verstappen`), and
    FastF1's code is the first three letters of the surname upper-cased. The
    handful that disagree are listed explicitly.
    """
    overrides = {
        "max_verstappen": "VER",
        "kevin_magnussen": "MAG",
        "mick_schumacher": "MSC",
        "jolyon_palmer": "PAL",
        "carlos_sainz": "SAI",
        "nico_hulkenberg": "HUL",
        "sergio_perez": "PER",
        "kimi_antonelli": "ANT",
        "antonelli": "ANT",
        "de_vries": "DEV",
        "hulkenberg": "HUL",
        "zhou": "ZHO",
        "guanyu": "ZHO",
        "tsunoda": "TSU",
        "bearman": "BEA",
        "colapinto": "COL",
        "lawson": "LAW",
        "doohan": "DOO",
        "hadjar": "HAD",
        "bortoleto": "BOR",
    }

    def convert(value: str) -> str:
        key = str(value).lower()
        if key in overrides:
            return overrides[key]
        surname = key.split("_")[-1]
        return surname[:3].upper()

    return driver_ids.map(convert)
