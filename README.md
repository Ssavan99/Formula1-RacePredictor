# Formula 1 Race Predictor

Three models predict every Formula 1 race. Their predictions are published
before the race runs, scored after it, and measured against baselines a
one-line rule could produce.

**[Live site →](https://ssavan99.github.io/Formula1-RacePredictor/)** — next
race, every model's prediction side by side, and a public track record.

---

## The headline result

**No model here beats "assume the pole sitter wins."**

Walk-forward over 103 races, 2022–2026. Top-1 winner accuracy, 95% bootstrap
intervals over races:

| Model | Top-1 winner | Podium | Spearman ρ | Log-loss |
|---|---|---|---|---|
| **naive: pole sitter wins** | **0.573** [0.48, 0.67] | **0.663** | 0.628 | 1.757 |
| lightgbm: lambdarank | 0.553 [0.46, 0.65] | 0.631 | **0.664** | 1.530 |
| original: MLP | 0.476 [0.38, 0.57] | 0.573 | 0.567 | 5.789 |
| plackett-luce | 0.456 [0.36, 0.55] | 0.628 | 0.652 | 1.702 |
| original: SVM (retuned) | 0.427 [0.33, 0.52] | 0.511 | 0.329 | 2.026 |
| original: SVM (as shipped in 2023) | 0.107 [0.05, 0.17] | 0.366 | 0.536 | 2.487 |
| naive: random | 0.039 [0.01, 0.08] | 0.172 | 0.009 | 3.017 |

At n=103 the intervals are wide enough that no model is statistically
separable from the pole-sitter rule in either direction. That is the honest
state of the problem, and it is stated first because it is the most useful
thing on this page.

LambdaRank closed most of the gap once its hyperparameters were actually tuned
(0.515 → 0.553; the shipped values had never been tested against this data), and
it now leads every model on ordering (ρ 0.664) and is far better calibrated than
the original, whose log-loss of 5.79 reflects emitting ≈0.99 for its pick and
≈0 for everyone else.

**The original SVM was never the problem — its kernel was.** `sigmoid` scores
0.107 here while keeping ρ = 0.536, the signature of a kernel mismatched to the
feature space rather than a method that does not work. Re-searching selects
`rbf`, which lifts it to **0.427**. The 2023 configuration is kept unchanged
alongside it, because the comparison is the point.

Full methodology, adoption decisions, and limitations: **[results/honest_baseline.md](results/honest_baseline.md)**.

## What qualifying is worth

Qualifying only exists from Saturday, so the project predicts twice and scores
both records separately.

| Model | Before qualifying | After qualifying |
|---|---|---|
| lightgbm: lambdarank | 0.456 | 0.515 |
| best baseline available | 0.470 (most wins so far) | 0.573 (pole sitter) |

Knowing the grid is worth about 6 points to the model. The single fact *who is
on pole* outperforms every model that does not have it.

## Sample output

```
$ python -m f1predict.cli predict --view pre_quali --within-days 10

Wrote docs/data/predictions/2026_12_pre_quali.json for Dutch Grand Prix (R12, 2026-08-23)
  original: MLP                    -> leclerc         p=0.525
  lightgbm: lambdarank             -> antonelli       p=0.367
  plackett-luce                    -> russell         p=0.149
```

Three models, three different picks — and probabilities that reflect how much
each is actually willing to commit.

## The three approaches

| Model | What it does | Why it is here |
|---|---|---|
| **original: MLP** | Per-driver binary classification of "did this driver win", field ranked by predicted probability | The project's original approach, preserved. Competitive on top-1; its probabilities are not trustworthy. |
| **lightgbm: lambdarank** | Learning-to-rank, one race = one query group, optimises NDCG over finishing order | A race *is* a ranking problem with natural groups. Best model overall. |
| **plackett-luce** | Discrete choice over the field; win probabilities sum to 1 by construction | The only model whose probabilities are a real distribution. Best of the three at picking the podium. |

**A weighted ensemble was built, backtested, and rejected** — it failed to beat
LambdaRank on any metric (top-1 −0.029 [−0.146, +0.087]). The code stays in
`f1predict/models/ensemble.py` so the negative result is reproducible.

**Deliberately not attempted:** transformers or deep tabular models. ~1,000
races × 20 drivers is far too small for them to beat gradient boosting, and
several published F1-ML results reporting near-perfect scores are leakage
artifacts of the same class this project found in its own history.

## The leak guard

The single most important piece of code here is
[`f1predict/data/contracts.py`](f1predict/data/contracts.py). Every column is
registered with the point in time at which its value becomes knowable, and a
feature matrix is assembled only from columns available before the moment being
predicted from. Unregistered columns are **rejected, not admitted** — adding a
feature requires stating when it becomes known.

It exists because the original pipeline fed `status_Finished`,
`status_Incident`, `status_Illness` and `status_Mechanical Issue` — the
finishing status of the race being predicted — straight into the model. Of 3707
rows, 681 have `status_Finished == 0`, and not one is a winner: the model was
told which ~18% of the field to rule out before predicting. Measured, that leak
was worth **+0.136 top-1**.

While building this project the guard caught nine further post-race columns
that the *new* feature pipeline had itself produced.

## Install

Python 3.10+.

```bash
git clone https://github.com/Ssavan99/Formula1-RacePredictor.git
cd Formula1-RacePredictor
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

On Windows use `.venv\Scripts\pip` instead.

## Usage

Build the dataset (once, ~15 minutes; self-throttles against a 200 req/hour
API limit and caches everything to disk):

```bash
python -m f1predict.data.backfill --start-season 2014 --verbose
```

Reproduce the headline number:

```bash
python -m f1predict.evaluate.run_backtest --models baselines,original,new --view post_quali
```

Predict the next race:

```bash
python -m f1predict.cli predict --view pre_quali --within-days 10
```

Other entry points:

```bash
python -m f1predict.cli next        # next race on the calendar
python -m f1predict.data.update     # incremental refresh, current season only
python -m f1predict.settle          # score past predictions
python -m pytest tests/ -q          # 98 tests
```

## How it stays current

Everything runs on free tiers: a public GitHub repo (unlimited Actions
minutes), GitHub Pages, the [Jolpica-F1](https://github.com/jolpica/jolpica-f1)
API, and [Open-Meteo](https://open-meteo.com/). No API keys, no accounts, no
paid services.

- **Tuesday 06:00 UTC** — refetch the calendar, refresh the dataset, settle any
  race that has now run, and if the next race is within 10 days, publish
  qualifying and pre-weekend race predictions.
- **Saturday 20:00 UTC** — if qualifying has run, publish a post-qualifying race
  prediction using the real grid.

The schedule is calendar-driven rather than triggered off the previous race: a
previous-race trigger has no season opener and breaks on calendar gaps and
back-to-back weekends. A standing weekly job that asks "is there a race soon?"
handles all three identically. The calendar is refetched every run, not once a
season — the 2026 schedule lists round 16 as "Bahrain Grand Prix in Malaysia",
a mid-season relocation an annual fetch would miss.

**Predictions are committed before the race takes place**, so git history is
the evidence the track record was not fitted after the fact.

## Data

| Source | Used for | Cost |
|---|---|---|
| [Jolpica-F1](https://github.com/jolpica/jolpica-f1) | Results, qualifying, sprints, calendar | Free, no key, 200 req/hour |
| [Open-Meteo](https://open-meteo.com/) | Race-window weather (archive + forecast) | Free, no key |

The original pipeline used the Ergast API, which was frozen after 2024 and shut
down in early 2025; Jolpica is its schema-compatible successor. Championship
standings are derived from race and sprint results rather than fetched, which
fits the request budget — validated at **0 mismatches** against the official
end-of-season tables for 2023, 2024 and 2025.

Weather is taken hourly and aggregated over the race window rather than the
whole day, because a grand prix is a two-hour event.

## Limitations

- **n = 103 races.** Most differences here are not statistically separable. The
  intervals are the result, not decoration.
- **Backtest weather is optimistic.** Historical rows use reanalysis — what
  actually happened during the race — while a live prediction has only a
  forecast. Figures involving weather are an upper bound on live performance.
- **The original SVM has not been retuned.** Its hyperparameters were searched
  against the original 88-column feature matrix; in the rebuilt 25-column space
  its ordering signal survives (ρ = 0.536) but its top pick collapses to 0.107.
  That is a fair report of those hyperparameters here, not a fair test of the
  method.
- **`driver_career_starts` counts from 2010**, not from a driver's true debut.
- **No tyre, pit-strategy, or practice-session data.** FastF1 exposes these
  free and is the obvious next source.

## Layout

```
f1predict/
  data/       Jolpica + Open-Meteo clients, feature builder, leak guard
  models/     original MLP/SVM, LambdaRank, Plackett-Luce, ensemble, registry
  evaluate/   walk-forward backtest, baselines, metrics, leak analysis
  predict.py  inference for an upcoming race
  settle.py   scoring past predictions into the track record
docs/         GitHub Pages site + published JSON
results/      backtest output and the written-up findings
tests/        98 tests, mostly about leakage
notebooks/    the original 2023 notebooks, unchanged
```

## Licence

MIT.
