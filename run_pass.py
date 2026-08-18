"""Run a named subset of models through the backtest and save one pass file.

Split into passes purely for wall-clock reasons; walk_forward is deterministic
(fixed race order, fixed within-race shuffle seed), so passes score identical
folds and merge_backtests.py can combine them without invalidating the paired
comparisons.

    python run_pass.py pass2 svm svmtuned leaky
"""
import sys, pandas as pd
from f1predict.evaluate.backtest import walk_forward
from f1predict.evaluate.metrics import summarise

args = sys.argv[1:]
min_season = None
if "--min-season" in args:
    i = args.index("--min-season")
    min_season = int(args[i + 1])
    del args[i:i + 2]
tag, names = args[0], args[1:]
df = pd.read_parquet("data/processed/races.parquet")

def build(name):
    from f1predict.models.original import (LeakyOriginalSVM, OriginalMLP,
                                           OriginalSVM, OriginalSVMTuned)
    from f1predict.models.ranker import LambdaRankModel
    from f1predict.models.reliability import ReliabilityAdjusted
    from f1predict.models.ensemble import build_ensemble
    return {
        "mlp": lambda: OriginalMLP(view="post_quali"),
        "svm": lambda: OriginalSVM(view="post_quali"),
        "svmtuned": lambda: OriginalSVMTuned(view="post_quali"),
        "leaky": lambda: LeakyOriginalSVM(),
        "reliability": lambda: ReliabilityAdjusted(
            LambdaRankModel(view="post_quali"), name="lambdarank + reliability"),
        "ensemble": lambda: build_ensemble(view="post_quali"),
    }[name]()

models = [build(n) for n in names]
per_race = walk_forward(df, models, refit_every=1, min_season=min_season)
per_race.to_parquet(f"results/per_race_{tag}.parquet", index=False)
print(summarise(per_race)[["model","top1_accuracy","podium_hit_rate","spearman_rho","winner_log_loss","n_races"]].to_string(index=False))
