"""Assemble everything the site reads.

The home tab shows a single prediction per target and one confidence number,
with no mention of how many models sit behind it. That is a presentation
decision, not a modelling one, so the choice has to be made honestly and
recorded rather than hand-waved.

**The house pick for each target is whichever model won the backtest on that
target's own metric** — top-1 accuracy for the race winner, podium hit rate for
the podium, top-1 for pole. Not an average of the models, and not a favourite.
An ensemble was built and measured as no better than its best component, so
blending would add a layer without adding accuracy.

The confidence shown is that model's backtested accuracy with its interval,
which is a claim about the *method* over 103 races, alongside the live
probability, which is a claim about this specific race. Both are needed: a model
that is right 51% of the time saying "62% Verstappen" means something different
from one right 80% of the time saying the same.

    python -m f1predict.publish
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

RESULTS = Path("results")
SITE_DATA = Path("docs/data")

#: Which backtest metric decides the house pick for each target.
TARGET_METRIC = {
    "winner": "top1_accuracy",
    "podium": "podium_hit_rate",
    "pole": "top1_accuracy",
}

#: Baselines are shown for honesty but are never the house pick — the point of
#: the site is what the models say, with the baseline visible beside it.
BASELINE_PREFIX = "naive"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("could not parse %s", path)
        return None


def choose_house_model(summary: list[dict], metric: str) -> dict | None:
    """Best non-baseline model on ``metric``, with its interval."""
    candidates = [
        row
        for row in summary
        if not row["model"].startswith(BASELINE_PREFIX)
        and "artifact" not in row["model"]
        and row.get(metric) is not None
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r[metric])
    return {
        "model": best["model"],
        "metric": metric,
        "value": best[metric],
        "lo": best.get(f"{metric}_lo"),
        "hi": best.get(f"{metric}_hi"),
        "n_races": best.get("n_races"),
    }


def best_baseline(summary: list[dict], metric: str) -> dict | None:
    candidates = [
        row
        for row in summary
        if row["model"].startswith(BASELINE_PREFIX) and row.get(metric) is not None
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r[metric])
    return {"model": best["model"], "metric": metric, "value": best[metric]}


def build() -> dict:
    post = _load("summary_post_quali_all.json")
    pre = _load("summary_pre_quali.json")
    leak = _load("original_leak.json")

    if post is None:
        raise SystemExit(
            "results/summary_post_quali_all.json not found. Run the backtest first."
        )

    post_summary = post["summary"]
    pre_summary = pre["summary"] if pre else []

    house = {
        "winner": choose_house_model(post_summary, TARGET_METRIC["winner"]),
        "podium": choose_house_model(post_summary, TARGET_METRIC["podium"]),
        "winner_pre_quali": choose_house_model(pre_summary, TARGET_METRIC["winner"]),
    }
    baselines = {
        "winner": best_baseline(post_summary, "top1_accuracy"),
        "podium": best_baseline(post_summary, "podium_hit_rate"),
    }

    payload = {
        "house": house,
        "baselines": baselines,
        "post_quali": post_summary,
        "pre_quali": pre_summary,
        "vs_reference_top1": post.get("vs_reference_top1", {}),
        "reference_baseline": post.get("reference_baseline"),
        "original_leak_analysis": leak,
        "n_races": post.get("n_races"),
        "test_seasons": post.get("test_seasons"),
    }

    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "backtest.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    payload = build()
    print("House picks (chosen by backtest, per target):")
    for target, choice in payload["house"].items():
        if not choice:
            print(f"  {target:18s} -- no model available")
            continue
        print(
            f"  {target:18s} {choice['model']:28s} "
            f"{choice['metric']}={choice['value']:.3f} "
            f"[{choice['lo']:.2f}, {choice['hi']:.2f}]"
        )
    print("\nBest baseline for comparison:")
    for target, base in payload["baselines"].items():
        if base:
            print(f"  {target:18s} {base['model']:28s} {base['value']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
