"""Measure where the language model's knowledge of F1 results actually ends.

An LLM cannot be scored by replaying history it was trained on: asked who won a
2024 race, it recalls rather than predicts. Published training cutoffs are vague
and providers are inconsistent, so this measures the boundary instead of
trusting one.

Method: ask who won a handful of races per season, walking forward through time,
with the model explicitly permitted to answer UNKNOWN. Recall stays high while
it has seen the results and falls towards chance afterwards. Races after that
transition are the ones it can be honestly scored on.

A model that answers *recent* races correctly is either grounded to live search
or has a later cutoff than advertised. Either way it is contaminated, and that
shows up here as recall that never drops.

    python probe_llm.py [--from-season 2021] [--per-season 4]
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from f1predict.models.llm import LLMUnavailable, infer_cutoff, probe_knowledge

parser = argparse.ArgumentParser()
parser.add_argument("--from-season", type=int, default=2021)
parser.add_argument("--per-season", type=int, default=4)
args = parser.parse_args()

df = pd.read_parquet("data/processed/races.parquet")
df = df[df["season"] >= args.from_season]

seasons = sorted(df["season"].unique())
print(
    f"probing {len(seasons)} seasons ({seasons[0]}-{seasons[-1]}), "
    f"{args.per_season} races each -- about "
    f"{len(seasons) * args.per_season} calls against a ~1500/day free tier\n"
)

try:
    probe = probe_knowledge(df, per_season=args.per_season)
except LLMUnavailable as exc:
    raise SystemExit(f"cannot reach the model: {exc}")

if probe.empty:
    raise SystemExit("probe returned nothing")

print(f"{'season':>7} {'recalled':>9} {'unknown':>8}  sample answer")
print("-" * 62)
for season, group in probe.groupby("season"):
    recalled = group["correct"].mean()
    unknown = group["said_unknown"].mean()
    example = group.iloc[0]
    mark = "CONTAMINATED" if recalled >= 0.5 else "usable"
    print(
        f"{season:>7} {recalled:>9.2f} {unknown:>8.2f}  "
        f"{example['actual']} -> {example['answered'][:26]!r}  {mark}"
    )

cutoff = infer_cutoff(probe)
print()
if cutoff is None:
    print("No season is recalled better than chance -- the whole window looks usable.")
else:
    clean = [s for s in probe["season"].unique() if s > cutoff]
    n_clean = int(df[df["season"].isin(clean)].groupby(["season", "round"]).ngroups)
    print(f"Knowledge appears to end after {cutoff}.")
    print(f"Seasons safe to score on: {clean or 'none'}  ({n_clean} races)")
    if n_clean < 40:
        print(
            f"\nCaveat that belongs in any write-up: {n_clean} races gives a 95% "
            "interval roughly +/-15 points on an accuracy, so this can only ever "
            "be a preliminary signal, not a comparison against the 0.573 baseline."
        )

probe.to_csv("results/llm_probe.csv", index=False)
print("\nwrote results/llm_probe.csv")
