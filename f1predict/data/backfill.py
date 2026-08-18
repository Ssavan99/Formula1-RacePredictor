"""One-shot local backfill of the modelling table.

Run once on a workstation, not in CI: a cold backfill is a few hundred Jolpica
requests against a 200/hour limit, and the client self-throttles rather than
failing. Subsequent runs are served from the on-disk cache and cost nothing, so
the scheduled job only ever fetches the races that have happened since.

    python -m f1predict.data.backfill --start-season 2014

Writes:
    data/processed/races.parquet   the modelling table
    data/processed/build_info.json provenance for what was built and when
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .features import DEFAULT_START_SEASON, build_dataset, validate_standings
from .jolpica import JolpicaClient
from .weather import WeatherClient

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")
DATASET_PATH = OUTPUT_DIR / "races.parquet"
BUILD_INFO_PATH = OUTPUT_DIR / "build_info.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="skip Open-Meteo (faster; weather features will be absent)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="fail rather than making network calls; use the cache only",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    client = JolpicaClient(offline=args.offline)
    weather = None if args.no_weather else WeatherClient(offline=args.offline)

    df = build_dataset(
        client,
        weather=weather,
        start_season=args.start_season,
        end_season=args.end_season,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)

    seasons = sorted(df["season"].unique().tolist())
    last_race = df.sort_values("race_date").iloc[-1]

    # Derivation of standings is a budget trade-off; measure what it costs.
    checks = [validate_standings(df, client, s) for s in seasons[:-1][-3:]]

    info = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "seasons": [int(s) for s in seasons],
        "races": int(df.groupby(["season", "round"]).ngroups),
        "last_race": {
            "season": int(last_race["season"]),
            "round": int(last_race["round"]),
            "name": str(last_race["race_name"]),
            "date": str(last_race["race_date"]),
        },
        "jolpica_requests": client.requests_made,
        "weather_requests": weather.requests_made if weather else 0,
        "standings_validation": checks,
    }
    BUILD_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Wrote {DATASET_PATH}")
    print(f"  rows    : {info['rows']}")
    print(f"  races   : {info['races']}")
    print(f"  seasons : {seasons[0]}-{seasons[-1]}")
    print(
        f"  last    : {info['last_race']['name']} "
        f"(R{info['last_race']['round']}, {info['last_race']['date']})"
    )
    print(
        f"  requests: {info['jolpica_requests']} jolpica, "
        f"{info['weather_requests']} open-meteo"
    )
    for check in checks:
        if check.get("checked"):
            print(
                f"  standings {check['season']}: "
                f"{check['mismatches']}/{check['checked']} mismatched"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
