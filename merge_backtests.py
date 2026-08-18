"""Merge per-race results from several backtest passes into one summary.

Splitting a sweep across passes is safe because walk_forward is deterministic:
the race order and the within-race shuffle both use fixed seeds, so every pass
scores identical folds. Paired comparisons across passes therefore remain valid.
"""
import glob, json, sys
import pandas as pd
from f1predict.evaluate.backtest import align_for_comparison
from f1predict.evaluate.metrics import paired_bootstrap_delta, summarise

tag = sys.argv[1] if len(sys.argv) > 1 else "post_quali_all"
parts = sorted(glob.glob("results/per_race_pass*.parquet"))
if not parts:
    raise SystemExit("no pass files found")

frames = [pd.read_parquet(p) for p in parts]

# Models whose code did not change in this round keep their existing results
# rather than being re-run: the MLP, the original SVM, the leaky artifact and
# the ensemble are byte-identical to the run that produced them, and that run
# already used the corrected, tie-invariant metrics.
CARRIED = ["original: MLP", "original: SVM", "original: SVM (leaky, artifact)", "ensemble"]
archive = "results/per_race_post_quali_all.parquet"
try:
    old = pd.read_parquet(archive)
    carried = old[old["model"].isin(CARRIED)]
    if not carried.empty:
        frames.append(carried)
        print(f"carried forward {carried['model'].nunique()} unchanged models from {archive}")
except FileNotFoundError:
    pass

per_race = pd.concat(frames, ignore_index=True)
per_race = per_race.drop_duplicates(subset=["model", "season", "round"], keep="last")
table = summarise(per_race)

per_race.to_parquet(f"results/per_race_{tag}.parquet", index=False)
table.to_csv(f"results/summary_{tag}.csv", index=False)

baselines = table[table["model"].str.startswith("naive")]
reference = baselines.iloc[0]["model"] if not baselines.empty else table.iloc[0]["model"]
comparisons = {}
for model in table["model"]:
    if model == reference:
        continue
    a, b = align_for_comparison(per_race, model, reference, "top1_accuracy")
    comparisons[model] = paired_bootstrap_delta(a, b)

json.dump({
    "view": "post_quali", "n_races": int(per_race.groupby(["season","round"]).ngroups),
    "test_seasons": [2022,2023,2024,2025,2026], "reference_baseline": reference,
    "summary": table.to_dict(orient="records"), "vs_reference_top1": comparisons,
}, open(f"results/summary_{tag}.json","w"), indent=2, default=float)

print(f"merged {len(parts)} passes -> {len(table)} models, "
      f"{per_race.groupby(['season','round']).ngroups} races\n")
print(f"{'model':36s} {'top-1':>16s} {'podium':>7s} {'rho':>7s} {'logloss':>8s}")
print("-"*80)
for r in table.itertuples(index=False):
    print(f"{r.model:36s} {r.top1_accuracy:.3f} [{r.top1_accuracy_lo:.2f},{r.top1_accuracy_hi:.2f}] "
          f"{r.podium_hit_rate:7.3f} {r.spearman_rho:7.3f} {r.winner_log_loss:8.3f}")
