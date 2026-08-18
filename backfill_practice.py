"""One-shot practice backfill. ~6s per session, cached; run locally, never in CI."""
import logging, warnings, pandas as pd
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("fastf1").setLevel(logging.ERROR)
from f1predict.data.practice import build_practice_table, FIRST_SEASON

df = pd.read_parquet("data/processed/races.parquet")
cal = df[["season", "round"]].drop_duplicates()
seasons = sorted(s for s in cal.season.unique() if s >= FIRST_SEASON)
print(f"backfilling practice for seasons {seasons[0]}-{seasons[-1]}")
out = build_practice_table(seasons, cal)
print(f"practice rows: {len(out)}, races covered: {out.groupby(['season','round']).ngroups}")
