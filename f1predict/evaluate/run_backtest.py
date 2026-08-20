"""Run the walk-forward backtest and write the results table.

    python -m f1predict.evaluate.run_backtest --models baselines,original
    python -m f1predict.evaluate.run_backtest --models baselines,original,new

The first form establishes the honest number: what the original approach
achieves against naive baselines under a protocol with no leakage. The second
adds the candidate approaches, which are only adopted if they beat it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ..data.contracts import VIEWS
from ..models.base import RaceModel
from .backtest import DEFAULT_TEST_SEASONS, align_for_comparison, walk_forward
from .baselines import build_baselines
from .metrics import paired_bootstrap_delta, summarise

log = logging.getLogger(__name__)

DATASET = Path("data/processed/races.parquet")
RESULTS_DIR = Path("results")


def build_models(groups: set[str], view: str) -> list[RaceModel]:
    models: list[RaceModel] = []
    if "baselines" in groups:
        models.extend(build_baselines(view=view))
    if "original" in groups:
        from ..models.original import (
            LeakyOriginalSVM,
            OriginalMLP,
            OriginalSVM,
            OriginalSVMTuned,
        )

        # The original SVM is kept exactly as specified AND retuned alongside
        # it, so "the method was fine, the kernel was wrong" is a claim the
        # table supports rather than an assertion.
        models.extend(
            [OriginalMLP(view=view), OriginalSVM(view=view), OriginalSVMTuned(view=view)]
        )
        # The leak needs qualifying-era features to be comparable to the
        # original setup; it is an artifact either way, never a candidate.
        if view == "post_quali":
            models.append(LeakyOriginalSVM())
    if "new" in groups:
        from ..models.choice import PlackettLuceModel
        from ..models.ranker import LambdaRankModel

        models.extend([LambdaRankModel(view=view), PlackettLuceModel(view=view)])
    if "reliability" in groups:
        from ..models.ranker import LambdaRankModel
        from ..models.reliability import ReliabilityAdjusted

        models.append(
            ReliabilityAdjusted(
                LambdaRankModel(view=view), name="lambdarank + reliability"
            )
        )
    if "ensemble" in groups:
        from ..models.ensemble import build_ensemble

        models.append(build_ensemble(view=view))
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="baselines,original")
    parser.add_argument("--view", default="post_quali", choices=sorted(VIEWS))
    parser.add_argument("--refit-every", type=int, default=1)
    parser.add_argument(
        "--min-season", type=int, default=None,
        help="drop training rows before this season (practice pace starts 2018)")
    parser.add_argument("--tag", default=None, help="suffix for output filenames")
    parser.add_argument(
        "--drop-weather",
        action="store_true",
        help=(
            "exclude weather features. Historical weather comes from ERA5 "
            "reanalysis (what actually happened over the race window) while a "
            "live prediction only has a forecast, so backtest weather is "
            "optimistic. This flag measures how much that is worth."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if not DATASET.exists():
        raise SystemExit(
            f"{DATASET} not found. Run: python -m f1predict.data.backfill"
        )

    df = pd.read_parquet(DATASET)
    if args.drop_weather:
        df = df.drop(columns=[c for c in df.columns if c.startswith("weather_")])
    groups = {g.strip() for g in args.models.split(",") if g.strip()}
    models = build_models(groups, args.view)
    if not models:
        raise SystemExit(f"no models selected from {sorted(groups)}")

    per_race = walk_forward(
        df, models, test_seasons=DEFAULT_TEST_SEASONS,
        refit_every=args.refit_every, min_season=args.min_season,
    )
    table = summarise(per_race)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.view}_{'-'.join(sorted(groups))}"
    per_race.to_parquet(RESULTS_DIR / f"per_race_{tag}.parquet", index=False)
    table.to_csv(RESULTS_DIR / f"summary_{tag}.csv", index=False)

    # Compare everything against the strongest naive rule, paired by race.
    baseline_rows = table[table["model"].str.startswith("naive")]
    reference = (
        baseline_rows.iloc[0]["model"] if not baseline_rows.empty else table.iloc[0]["model"]
    )
    comparisons = {}
    for model in table["model"]:
        if model == reference:
            continue
        a, b = align_for_comparison(per_race, model, reference, "top1_accuracy")
        comparisons[model] = paired_bootstrap_delta(a, b)

    payload = {
        "view": args.view,
        "models": sorted(groups),
        "test_seasons": list(DEFAULT_TEST_SEASONS),
        "refit_every": args.refit_every,
        "n_races": int(per_race.groupby(["season", "round"]).ngroups),
        "reference_baseline": reference,
        "summary": table.to_dict(orient="records"),
        "vs_reference_top1": comparisons,
    }
    (RESULTS_DIR / f"summary_{tag}.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )

    _print_table(table, reference, comparisons, per_race)
    return 0


def _print_table(table, reference, comparisons, per_race) -> None:
    races = per_race.groupby(["season", "round"]).ngroups
    print()
    print(f"Walk-forward backtest over {races} races (2022-2026)")
    print(f"Reference baseline: {reference}")
    print()
    header = (
        f"{'model':38s} {'top-1':>16s} {'podium':>8s} {'rho':>7s} "
        f"{'logloss':>8s} {'races':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in table.itertuples(index=False):
        ci = f"{row.top1_accuracy:.3f} [{row.top1_accuracy_lo:.2f},{row.top1_accuracy_hi:.2f}]"
        print(
            f"{row.model:38s} {ci:>16s} {row.podium_hit_rate:8.3f} "
            f"{row.spearman_rho:7.3f} {row.winner_log_loss:8.3f} {row.n_races:6d}"
        )
    print()
    print(f"Paired difference in top-1 vs {reference}:")
    for model, delta in sorted(
        comparisons.items(), key=lambda kv: kv[1]["delta"], reverse=True
    ):
        verdict = "better" if delta["lower"] > 0 else (
            "worse" if delta["upper"] < 0 else "not distinguishable"
        )
        print(
            f"  {model:38s} {delta['delta']:+.3f} "
            f"[{delta['lower']:+.3f},{delta['upper']:+.3f}]  {verdict}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
