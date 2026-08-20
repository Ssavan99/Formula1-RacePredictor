"""Command-line entry points used by the scheduled workflows.

    python -m f1predict.cli predict --view pre_quali --within-days 10
    python -m f1predict.cli predict --view post_quali --within-days 2
    python -m f1predict.cli settle
    python -m f1predict.cli next
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from .data.jolpica import JolpicaClient

log = logging.getLogger(__name__)


def cmd_predict(args: argparse.Namespace) -> int:
    from . import predict as predict_module

    client = JolpicaClient()
    today = datetime.now(timezone.utc).date()

    race = client.next_race(today=today)
    if race is None:
        print("No upcoming race on the calendar; nothing to do.")
        return 0

    race_date = datetime.fromisoformat(race["date"]).date()
    days_away = (race_date - today).days

    # The whole point of the fixed weekly schedule: most weeks there is nothing
    # to do, and that is a success, not a failure.
    if days_away > args.within_days and not args.force:
        print(
            f"Next race is {race['raceName']} on {race['date']} "
            f"({days_away} days away); outside the {args.within_days}-day "
            "window, so no prediction this run."
        )
        return 0

    payload = predict_module.run(
        view=args.view, client=client, today=today, with_llm=args.with_llm
    )
    if payload is None:
        # For post_quali this is the normal state before Saturday.
        print(f"Could not produce a {args.view} prediction yet.")
        return 0

    path = predict_module.save(payload)
    race_info = payload["race"]
    print(
        f"Wrote {path} for {race_info['name']} "
        f"(R{race_info['round']}, {race_info['date']})"
    )
    for prediction in payload["predictions"]:
        top = prediction["ranking"][0]
        print(
            f"  {prediction['model']:32s} -> {top['driver_id']:15s} "
            f"p={top['win_probability']:.3f}"
        )
    return 0


def cmd_settle(args: argparse.Namespace) -> int:
    from .settle import main as settle_main

    return settle_main()


def cmd_next(args: argparse.Namespace) -> int:
    client = JolpicaClient()
    today = datetime.now(timezone.utc).date()
    race = client.next_race(today=today)
    if race is None:
        print("No upcoming race found.")
        return 0
    race_date = datetime.fromisoformat(race["date"]).date()
    print(
        f"{race['raceName']} (round {race['round']}, {race['season']}) "
        f"on {race['date']} - {(race_date - today).days} days away"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1predict", description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    predict_parser = sub.add_parser("predict", help="predict the upcoming race")
    predict_parser.add_argument(
        "--view", default="pre_quali", choices=["pre_quali", "post_quali"]
    )
    predict_parser.add_argument(
        "--within-days",
        type=int,
        default=10,
        help="only predict if the next race is at most this many days away",
    )
    predict_parser.add_argument("--force", action="store_true")
    predict_parser.add_argument(
        "--with-llm",
        action="store_true",
        help=(
            "also run the LLM entrant. Needs GEMINI_API_KEY; skipped with a "
            "warning if absent, because a missing optional key must not stop "
            "the tabular models from publishing."
        ),
    )
    predict_parser.set_defaults(func=cmd_predict)

    settle_parser = sub.add_parser("settle", help="score predictions for finished races")
    settle_parser.set_defaults(func=cmd_settle)

    next_parser = sub.add_parser("next", help="show the next race on the calendar")
    next_parser.set_defaults(func=cmd_next)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
