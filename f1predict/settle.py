"""Score past predictions against what actually happened.

This is what makes the site's track record verifiable rather than decorative.
Predictions are committed before a race runs, so git history is the evidence
that they were not written afterwards; this module only ever reads that record
and appends outcomes to it.

Idempotent: a race already settled is skipped, so the weekly job can re-run
without double-counting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data.jolpica import JolpicaClient
from .predict import PREDICTIONS_DIR

log = logging.getLogger(__name__)

TRACK_RECORD = Path("docs/data/track_record.json")


def load_predictions(directory: Path = PREDICTIONS_DIR) -> list[dict]:
    if not directory.exists():
        return []
    payloads = []
    for path in sorted(directory.glob("*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("skipping unreadable prediction file %s", path)
    return payloads


def load_track_record(path: Path = TRACK_RECORD) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("track record unreadable; starting a new one")
    return {"settled": [], "updated_at": None}


def actual_result(client: JolpicaClient, season: int, rnd: int) -> dict[str, int]:
    """Finishing position by driver, empty if the race has not been run."""
    try:
        payload = client.get(f"{season}/{rnd}/results", season=season)
    except Exception as exc:
        log.warning("results fetch failed for %s R%s: %s", season, rnd, exc)
        return {}
    races = payload["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    out = {}
    for entry in races[0].get("Results", []):
        try:
            out[entry["Driver"]["driverId"]] = int(entry["position"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def score_prediction(prediction: dict, actual: dict[str, int]) -> dict:
    """Score one model's ranking for one race."""
    ranking = prediction["ranking"]
    if not ranking:
        return {}

    predicted_winner = ranking[0]["driver_id"]
    true_winner = next((d for d, p in actual.items() if p == 1), None)

    predicted_podium = {entry["driver_id"] for entry in ranking[:3]}
    true_podium = {d for d, p in actual.items() if p <= 3}

    probability_of_winner = next(
        (
            entry["win_probability"]
            for entry in ranking
            if entry["driver_id"] == true_winner
        ),
        None,
    )

    return {
        "model": prediction["model"],
        "view": prediction["view"],
        "predicted_winner": predicted_winner,
        "actual_winner": true_winner,
        "top1_correct": int(predicted_winner == true_winner) if true_winner else None,
        "podium_hits": len(predicted_podium & true_podium) if true_podium else None,
        "winner_probability": probability_of_winner,
        "winner_log_loss": (
            float(-np.log(max(probability_of_winner, 1e-15)))
            if probability_of_winner is not None
            else None
        ),
    }


def settle(client: JolpicaClient | None = None) -> dict:
    client = client or JolpicaClient()
    record = load_track_record()
    already = {
        (entry["season"], entry["round"], entry["view"]) for entry in record["settled"]
    }

    newly_settled = 0
    for payload in load_predictions():
        race = payload["race"]
        key = (race["season"], race["round"], payload["view"])
        if key in already:
            continue

        actual = actual_result(client, race["season"], race["round"])
        if not actual:
            continue  # race has not run yet

        for prediction in payload["predictions"]:
            scored = score_prediction(prediction, actual)
            if not scored:
                continue
            record["settled"].append(
                {
                    "season": race["season"],
                    "round": race["round"],
                    "race_name": race["name"],
                    "date": race["date"],
                    "view": payload["view"],
                    "predicted_at": payload.get("generated_at"),
                    **scored,
                }
            )
        newly_settled += 1

    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    record["summary"] = summarise_record(record["settled"])
    TRACK_RECORD.parent.mkdir(parents=True, exist_ok=True)
    TRACK_RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("settled %d newly completed race(s)", newly_settled)
    return record


def summarise_record(settled: list[dict]) -> list[dict]:
    """Per-model, per-view running totals for the site."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for entry in settled:
        buckets.setdefault((entry["model"], entry["view"]), []).append(entry)

    summary = []
    for (model, view), entries in sorted(buckets.items()):
        top1 = [e["top1_correct"] for e in entries if e["top1_correct"] is not None]
        podium = [e["podium_hits"] for e in entries if e["podium_hits"] is not None]
        losses = [
            e["winner_log_loss"] for e in entries if e["winner_log_loss"] is not None
        ]
        summary.append(
            {
                "model": model,
                "view": view,
                "races": len(entries),
                "top1_accuracy": round(float(np.mean(top1)), 4) if top1 else None,
                "podium_hits_per_race": (
                    round(float(np.mean(podium)), 3) if podium else None
                ),
                "winner_log_loss": round(float(np.mean(losses)), 3) if losses else None,
            }
        )
    return sorted(
        summary, key=lambda s: (s["top1_accuracy"] is None, -(s["top1_accuracy"] or 0))
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    record = settle()
    print(f"Track record: {len(record['settled'])} scored predictions")
    for row in record["summary"]:
        accuracy = (
            f"{row['top1_accuracy']:.3f}" if row["top1_accuracy"] is not None else "n/a"
        )
        print(
            f"  {row['model']:32s} {row['view']:11s} "
            f"{row['races']:3d} races  top-1 {accuracy}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
