"""Backfill FastF1 practice pace, in resumable chunks.

Cold-loading a session is slow (several seconds each, three sessions per race),
and long-running background jobs get suspended in this environment, so this runs
in bounded chunks: it skips races already cached and stops after --max-races.
Re-run until it reports nothing missing.

    python backfill_practice.py --max-races 40 --from-season 2021
"""
import argparse, logging, time, warnings
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("fastf1").setLevel(logging.ERROR)

from f1predict.data.practice import (FEATURE_CACHE, FIRST_SEASON, _enable_cache,
                                     load_session_laps, summarise_practice)

ap = argparse.ArgumentParser()
ap.add_argument("--max-races", type=int, default=40)
ap.add_argument("--from-season", type=int, default=FIRST_SEASON)
args = ap.parse_args()

if not _enable_cache():
    raise SystemExit("fastf1 not installed")

races = pd.read_parquet("data/processed/races.parquet")[["season", "round"]].drop_duplicates()
races = races[races["season"] >= args.from_season].sort_values(["season", "round"])

try:
    have_df = pd.read_parquet(FEATURE_CACHE)
    have = set(zip(have_df["season"], have_df["round"]))
    frames = [have_df]
except FileNotFoundError:
    have, frames = set(), []

todo = [(int(s), int(r)) for s, r in zip(races["season"], races["round"]) if (int(s), int(r)) not in have]
print(f"{len(have)} cached, {len(todo)} missing; doing up to {args.max_races} this run")

done, t0 = 0, time.time()
for season, rnd in todo[: args.max_races]:
    started = time.time()
    summary = summarise_practice(load_session_laps(season, rnd))
    if summary.empty:
        print(f"  {season} R{rnd}: no practice data")
        continue
    summary["season"], summary["round"] = season, rnd
    frames.append(summary)
    done += 1
    print(f"  {season} R{rnd}: {len(summary)} drivers ({time.time()-started:.1f}s)")

if frames:
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["season", "round", "driver_code"], keep="last")
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURE_CACHE, index=False)
    covered = out.groupby(["season", "round"]).ngroups
    rate = (time.time() - t0) / max(done, 1)
    print(f"\n{len(out)} rows, {covered} races covered, {len(todo)-done} still missing "
          f"({rate:.1f}s/race)")
