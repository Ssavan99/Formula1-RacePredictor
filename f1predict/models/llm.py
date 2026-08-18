"""An LLM entrant, and an honest way to evaluate it.

## The problem this module works around

Every other model here is scored by replaying history: hide a past race's result,
predict it, check. An LLM cannot be scored that way, because it was trained on
text that describes those races. Ask it who won the 2024 Monaco Grand Prix and it
is not predicting, it is recalling. The score would look excellent and mean
nothing.

That is the same failure that motivated this project's leak guard -- outcome
information reaching the model -- just arriving through the weights instead of
through a column.

## The workaround

Only score it on races the model cannot have seen: those after its training
cutoff. Published cutoffs are vague and providers are inconsistent, so this
module **measures the cutoff empirically** rather than trusting a number in a
docs page. `probe_knowledge()` asks who won a series of races whose answers we
already have, walking forward in time. Where its accuracy collapses to chance is
where its knowledge ends, and races after that point are usable.

Two caveats that survive the workaround, and belong in any write-up:

* Some deployments have web search or grounding enabled, which reintroduces
  contamination regardless of cutoff. The probe partly detects this: a model that
  answers *recent* races correctly is either grounded or has a later cutoff than
  claimed. Either way it is not clean.
* The clean window is small. Roughly 30 races is not enough to separate this from
  the pole-sitter baseline. It is a preliminary signal, and it is labelled as one.

## Cost

At roughly two calls per race weekend, this is ~50 calls a year, against a free
tier of ~1,500 requests a *day*. It is free in practice, not merely in principle.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from .base import RaceModel

log = logging.getLogger(__name__)

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
DEFAULT_MODEL = "gemini-2.0-flash"
ENV_KEY = "GEMINI_API_KEY"

#: Races before this are assumed contaminated until the probe says otherwise.
#: Overwritten by `probe_knowledge` results, never guessed at in production.
ASSUMED_CUTOFF = date(2025, 1, 1)


class LLMUnavailable(RuntimeError):
    """No API key, or the endpoint could not be reached."""


def _api_key() -> str:
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise LLMUnavailable(
            f"{ENV_KEY} is not set. Create a free key at "
            "https://aistudio.google.com/apikey and export it, or add it as a "
            "repository secret for CI."
        )
    return key


def call_model(prompt: str, model: str = DEFAULT_MODEL, retries: int = 3) -> str:
    """One completion. Raises LLMUnavailable rather than returning junk."""
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            # Deterministic: this is a measurement, not a creative task.
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 400},
        }
    ).encode()

    url = API_URL.format(model=model) + f"?key={_api_key()}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise LLMUnavailable(f"model call failed after {retries} attempts: {last}")


# ---------------------------------------------------------------------------
# Measuring what it already knows
# ---------------------------------------------------------------------------


def probe_knowledge(
    df: pd.DataFrame, model: str = DEFAULT_MODEL, per_season: int = 3
) -> pd.DataFrame:
    """Ask who won races we know the answer to, walking forward through time.

    Recall stays high while the model has seen the results and collapses towards
    chance afterwards. That transition is the usable cutoff -- measured, not
    taken on trust.
    """
    winners = (
        df[df["finish_position"] == 1][["season", "round", "race_name", "driver_id"]]
        .sort_values(["season", "round"])
    )

    rows = []
    for season, group in winners.groupby("season"):
        # Spread the samples across the season rather than clustering early.
        picks = group.iloc[:: max(1, len(group) // per_season)][:per_season]
        for race in picks.itertuples(index=False):
            prompt = (
                f"Who won the {race.season} {race.race_name} in Formula 1?\n"
                "Reply with only the driver's surname, or exactly UNKNOWN if you "
                "do not know. Do not guess."
            )
            try:
                answer = call_model(prompt, model=model).strip()
            except LLMUnavailable as exc:
                log.warning("probe aborted: %s", exc)
                return pd.DataFrame(rows)

            surname = str(race.driver_id).split("_")[-1].lower()
            said_unknown = "unknown" in answer.lower()
            correct = (not said_unknown) and surname in answer.lower()
            rows.append(
                {
                    "season": int(race.season),
                    "round": int(race.round),
                    "race": race.race_name,
                    "actual": surname,
                    "answered": answer[:60],
                    "said_unknown": said_unknown,
                    "correct": bool(correct),
                }
            )
            time.sleep(1.0)  # stay far inside the free tier
    return pd.DataFrame(rows)


def infer_cutoff(probe: pd.DataFrame, threshold: float = 0.5) -> int | None:
    """Latest season whose winners the model still recalls better than chance.

    Anything at or before this is contaminated; races after it are candidates for
    an honest evaluation.
    """
    if probe.empty:
        return None
    by_season = probe.groupby("season")["correct"].mean()
    known = by_season[by_season >= threshold]
    return int(known.index.max()) if len(known) else None


# ---------------------------------------------------------------------------
# The entrant
# ---------------------------------------------------------------------------


class LLMPredictor(RaceModel):
    """Ranks the field by asking a language model, given pre-race context only.

    `requires_fit` is False: there is nothing to train. The prompt is built from
    exactly the same pre-race features the other models see, so the comparison is
    about reasoning rather than about who got more data.
    """

    name = "llm: gemini"
    view = "pre_quali"
    requires_fit = False

    def __init__(self, model: str = DEFAULT_MODEL, view: str = "pre_quali"):
        self.model = model
        self.view = view
        self._last_reason: str = ""

    def fit(self, train: pd.DataFrame) -> "LLMPredictor":
        return self

    def _prompt(self, race: pd.DataFrame) -> str:
        first = race.iloc[0]
        lines = []
        for row in race.itertuples(index=False):
            bits = [
                f"{row.driver_id} ({row.constructor_id})",
                f"championship points so far: {getattr(row, 'driver_points_before', 0):.0f}",
                f"wins so far: {getattr(row, 'driver_wins_before', 0):.0f}",
            ]
            form = getattr(row, "driver_form_5", None)
            if form is not None and np.isfinite(form):
                bits.append(f"avg finish last 5: {form:.1f}")
            grid = getattr(row, "grid", None)
            if self.view == "post_quali" and grid is not None and np.isfinite(grid):
                bits.append(f"starting grid: {int(grid)}")
            lines.append("- " + ", ".join(bits))

        weather = ""
        wet = getattr(first, "weather_is_wet", None)
        if wet is not None and np.isfinite(wet):
            weather = f"\nForecast: {'wet' if wet else 'dry'} race."

        return (
            f"Formula 1, {first.season} {first.race_name}, round {first['round']}.\n"
            f"{'Qualifying has run.' if self.view == 'post_quali' else 'Qualifying has NOT run yet.'}"
            f"{weather}\n\nEntries and their form entering this race:\n"
            + "\n".join(lines)
            + "\n\nRank the FIVE most likely race winners, most likely first. "
            "Use only the information above and general knowledge of the sport; "
            "do not use knowledge of this specific race's outcome.\n"
            'Reply as JSON only: {"ranking": ["driverid1", ..., "driverid5"], '
            '"reason": "one short sentence"}'
        )

    def predict_scores(self, race: pd.DataFrame) -> np.ndarray:
        drivers = [str(d) for d in race["driver_id"]]
        scores = np.zeros(len(race))
        try:
            answer = call_model(self._prompt(race), model=self.model)
        except LLMUnavailable as exc:
            log.warning("%s unavailable: %s", self.name, exc)
            return scores

        ranking, reason = _parse_ranking(answer)
        self._last_reason = reason
        index = {d.lower(): i for i, d in enumerate(drivers)}
        # Decaying scores so the ordering survives the softmax intact.
        for position, name in enumerate(ranking[:5]):
            key = str(name).strip().lower()
            slot = index.get(key)
            if slot is None:  # tolerate surname-only answers
                matches = [i for d, i in index.items() if key and key in d]
                slot = matches[0] if len(matches) == 1 else None
            if slot is not None:
                scores[slot] = 5.0 - position
        return scores

    @property
    def last_reason(self) -> str:
        return self._last_reason


def _parse_ranking(text: str) -> tuple[list[str], str]:
    """Pull the ranking out of a reply that may be wrapped in prose or fences."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            return (
                [str(x) for x in payload.get("ranking", [])],
                str(payload.get("reason", "")),
            )
        except json.JSONDecodeError:
            pass
    # Fall back to any quoted tokens, in order.
    return re.findall(r'"([a-z_]+)"', text.lower())[:5], ""
