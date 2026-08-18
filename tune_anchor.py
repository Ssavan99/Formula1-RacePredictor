"""Choose the grid-anchor weight on a validation fold inside the training era.

One smooth parameter, swept on races the backtest never sees. Reported alongside
the whole curve so the choice is auditable rather than a number that appeared.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from f1predict.evaluate.metrics import podium_hit_rate, top1_correct
from f1predict.models.anchored import GridAnchored
from f1predict.models.ranker import LambdaRankModel

df = pd.read_parquet("data/processed/races.parquet")
df = df[(df.season >= 2018) & (df.season < 2022)]          # never the test window
train = df[df.season <= 2020]
valid = df[df.season == 2021]

print(f"validation: {valid.groupby(['season','round']).ngroups} races (2021), "
      f"train {len(train)} rows (2018-2020)\n")
print(f"{'weight':>7} {'top-1':>7} {'podium':>7}")
print("-" * 24)

rows = []
for w in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]:
    model = GridAnchored(LambdaRankModel(view="post_quali"), weight=w).fit(train)
    t1, pod = [], []
    for _, race in valid.groupby(["season", "round"], sort=False):
        race = race.sample(frac=1.0, random_state=0)
        actual = pd.to_numeric(race.finish_position, errors="coerce").to_numpy(float)
        if not np.isfinite(actual).any():
            continue
        s = np.asarray(model.predict_scores(race), dtype=float)
        t1.append(top1_correct(s, actual)); pod.append(podium_hit_rate(s, actual))
    rows.append((w, float(np.nanmean(t1)), float(np.nanmean(pod))))
    print(f"{w:7.2f} {rows[-1][1]:7.3f} {rows[-1][2]:7.3f}")

best = max(rows, key=lambda r: (r[1], r[2]))
print(f"\nchosen on validation: weight={best[0]:.2f} (top-1 {best[1]:.3f})")
