"""Score the LLM entrant on the window the probe proved it has not seen.

The probe (results/llm_probe.csv) found 100% recall of race winners through 2024
and 0% -- with an honest "UNKNOWN" every time -- from 2025 onward. So 2025-2026
is the only window where asking this model to predict is predicting rather than
remembering.

Every other model is re-scored on exactly those races so the comparison is
like-for-like. It is a small window and the write-up says so.

    python score_llm.py
"""

from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from f1predict.evaluate.backtest import walk_forward
from f1predict.evaluate.baselines import build_baselines
from f1predict.evaluate.metrics import summarise
from f1predict.models.llm import LLMPredictor
from f1predict.models.ranker import LambdaRankModel

CLEAN_SEASONS = (2025, 2026)

df = pd.read_parquet("data/processed/races.parquet")

# The LLM sees no grid, so compare it against pre-qualifying models and the
# baselines that are available at the same moment.
models = [
    LLMPredictor(view="pre_quali"),
    LambdaRankModel(view="pre_quali"),
    *build_baselines(view="pre_quali"),
]

per_race = walk_forward(
    df, models, test_seasons=CLEAN_SEASONS, refit_every=1, min_season=2018
)
table = summarise(per_race)

per_race.to_parquet("results/per_race_llm_window.parquet", index=False)
table.to_csv("results/summary_llm_window.csv", index=False)

races = per_race.groupby(["season", "round"]).ngroups
print(f"\nUncontaminated window only: {races} races, {CLEAN_SEASONS[0]}-{CLEAN_SEASONS[1]}")
print("(the LLM has never seen these results; the probe verified that)\n")
print(f"{'model':32s} {'top-1':>16s} {'podium':>8s} {'races':>6s}")
print("-" * 66)
for row in table.itertuples(index=False):
    ci = f"{row.top1_accuracy:.3f} [{row.top1_accuracy_lo:.2f},{row.top1_accuracy_hi:.2f}]"
    print(f"{row.model:32s} {ci:>16s} {row.podium_hit_rate:8.3f} {row.n_races:6d}")

print(
    f"\nAt n={races} the 95% interval on an accuracy is roughly +/-15 points, so "
    "this ranks the entrants only loosely. It is a preliminary signal and is "
    "labelled as one wherever it is shown."
)
