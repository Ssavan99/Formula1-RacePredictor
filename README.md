# Pitwall — Formula 1 race predictions

Pole, winner and podium for the next Grand Prix. Every call is committed to git
**before** the race runs and scored against the result afterwards, so the track
record cannot be fitted after the fact.

**[Live site →](https://ssavan99.github.io/Formula1-RacePredictor/)**

---

## The headline number

Walk-forward over **103 races (2022–2026)**. Top-1 winner accuracy, 95% bootstrap
intervals resampling races:

| Model | Top-1 winner | Podium | Spearman ρ | Log-loss |
|---|---|---|---|---|
| **lambdarank + grid anchor** | **0.583** [0.49, 0.68] | 0.660 | 0.661 | **1.377** |
| **plackett-luce + grid anchor** | **0.583** [0.49, 0.68] | **0.673** | 0.651 | 1.399 |
| naive: pole sitter wins | 0.573 [0.48, 0.67] | 0.663 | 0.628 | 1.757 |
| lightgbm: lambdarank | 0.544 [0.45, 0.64] | 0.654 | 0.664 | 1.576 |
| plackett-luce | 0.495 [0.40, 0.59] | 0.628 | 0.646 | 1.651 |
| original: MLP (2023) | 0.476 [0.38, 0.57] | 0.544 | 0.577 | 5.023 |
| original: SVM (2023) | 0.087 [0.04, 0.15] | 0.372 | 0.494 | 2.503 |
| naive: random | 0.039 [0.01, 0.08] | 0.172 | 0.009 | 3.017 |

**Read that honestly.** The anchored models beat the pole-sitter rule by
**+0.0097 — one race in 103**, with an interval spanning zero. That is a tie, not
a victory, and saying otherwise would repeat the exact mistake this project was
built to expose. The real win is calibration: log-loss **1.377 vs 1.757**, an
interval nowhere near zero. These are the first models here to *match* a
one-line rule on accuracy while publishing probabilities you can actually trust.

Full methodology, every rejected experiment, and the limitations:
**[results/honest_baseline.md](results/honest_baseline.md)**

## Where the headroom is

Measured before any modelling:

| | |
|---|---|
| Winner started on pole | 57.3% |
| **Winner started top-3** | **88.3%** |
| Winner started outside top-5 | 6.8% |
| Pole sitter finished outside top-10 | 9.7% |

The task is not "find one driver in twenty" — it is **telling front-runners
apart**, which puts a realistic ceiling near 0.85 rather than 1.0.

That framing produced the one change that actually worked. Splitting the races by
whether the ranker agreed with the pole sitter:

| | Races | Correct |
|---|---|---|
| Agreed with pole | 47 | **0.809** |
| Backed someone else | 56 | **0.321** |
| …where pole would have given | 56 | 0.375 |

The model was never bad at picking winners — it **over-deviated**, losing ~5.4
points per departure. Shrinking it toward an empirical `P(win | grid slot)` is
what closed the gap. Making a model *less* willing to back its marginal opinions
beat every attempt to make it smarter.

## What the original project got wrong

The 2023 version reported **0.818 precision**. That number reproduces, and it is
not fraud — it is two separate defects:

- **A post-race feature.** `status_Finished` — whether the driver finished *the
  race being predicted* — was in the model's inputs. Of 3707 rows, 681 had it at
  zero and **not one was a winner**: the model was told which 18% of the grid to
  rule out in advance. Measured: worth **+0.136 top-1**.
- **An 11-race test set.** The 0.818 is reproduced exactly by scoring rounds
  12–22 only. At n=11 the 95% interval is **[0.48, 0.98]** — half the available
  range. Over the full season the same model gives 0.409, *below* the baseline.

`f1predict/data/contracts.py` makes the first structurally hard to repeat: every
column is registered with the moment its value becomes knowable, and
unregistered columns are **rejected, not admitted**. While rebuilding, that guard
caught nine further post-race columns the *new* pipeline had produced by accident.

## Models

| Model | Approach |
|---|---|
| **original: MLP / SVM** | The 2023 approach, preserved unchanged and still scored. Retained because it is part of the project's history. |
| **lightgbm: lambdarank** | Learning-to-rank — one race is a query, the drivers are candidates, finishing order is relevance. |
| **plackett-luce** | Discrete choice over the field; win probabilities sum to 1 by construction. |
| **+ grid anchor** | Both shrunk toward an empirical prior over starting slots. Weight chosen on 2021 alone, never the test window. |

**Built, measured, rejected** — code kept so the negative results stay
reproducible: a weighted ensemble (beat no component), a retirement-hazard model
(improved calibration, not accuracy), a practice-era re-tune (validation window
was 22 races — too thin to trust), and an LLM entrant (mid-table).

**Deliberately not attempted:** transformers or deep tabular models. ~1,000 races
× 20 drivers is far too small, and several published F1-ML papers reporting
near-perfect scores are leakage artifacts of the same class found here.

## The LLM, and why it cannot be backtested

An LLM trained on the internet has *read the results*. Asking it to predict the
2024 Monaco GP is asking it to recall — the same failure as the `status_*` leak,
arriving through the weights instead of a column.

So `scripts/probe_llm.py` **measures** where its knowledge ends rather than
trusting a published cutoff:

| Season | Winners recalled | Said UNKNOWN |
|---|---|---|
| 2020–2024 | **1.00** | 0.00 |
| **2025–2026** | **0.00** | **1.00** |

A clean step, and no sign of live-search grounding. Scored on those 35 genuinely
unseen races it lands **mid-table** — clear of random, level at ordering the
podium, below the tabular models at picking winners. At n=35 nothing is
separable, so that is evidence of absence of a large effect, not proof of a small
one.

## How it stays current

Everything runs on free tiers: public GitHub repo (unlimited Actions minutes),
GitHub Pages, [Jolpica-F1](https://github.com/jolpica/jolpica-f1),
[Open-Meteo](https://open-meteo.com/), [FastF1](https://docs.fastf1.dev/). No API
keys, no accounts, no paid services.

- **Tuesday 06:00 UTC** — refetch the calendar, refresh the dataset, settle any
  race that has run, and if the next is within 10 days publish qualifying and
  pre-weekend predictions.
- **Saturday 20:00 UTC** — if qualifying has run, publish a post-qualifying
  prediction using the real grid.

Calendar-driven on a fixed schedule rather than triggered off the previous race:
a previous-race trigger has no season opener and breaks on calendar gaps and
back-to-back weekends. The calendar is refetched every run, because it genuinely
changes mid-season — the 2026 schedule lists round 16 as "Bahrain Grand Prix in
Malaysia".

Models are **refit from scratch before every prediction** on all data through the
most recent race.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python -m f1predict.data.backfill --start-season 2014 --verbose   # once, ~15 min
python -m f1predict.evaluate.run_backtest --models baselines,original,new --view post_quali
python -m f1predict.cli predict --view pre_quali --within-days 10
python -m pytest tests/ -q                                        # 98 tests
```

Serve the site locally with `python serve_docs.py`.

## Layout

```
f1predict/     data layer + leak guard, models, evaluation, inference
docs/          the site (index.html), 3D hero, model, published JSON
  classic-2d.html   earlier build with a drivable 2D car, kept as a fallback
results/       backtest output and the written-up findings
scripts/       operational and research tooling (see scripts/README.md)
notebooks/     the original 2023 notebooks, unchanged
tests/         98 tests, mostly about leakage
```

## Limitations

- **n = 103.** Most differences here are not statistically separable. The
  intervals are the result, not decoration.
- **Backtest weather is optimistic.** Historical rows use ERA5 reanalysis — what
  actually happened during the race — while a live prediction only has a
  forecast. Figures involving weather are an upper bound on live performance.
- **Practice pace starts in 2018**, so training is restricted to 2018+.
- **`driver_career_starts` counts from 2010**, not a driver's true debut.
- **The live record is empty** until the first race after launch, and will need a
  full season before it means much.

## Credits

Data: [Jolpica-F1](https://github.com/jolpica/jolpica-f1) ·
[Open-Meteo](https://open-meteo.com/) · [FastF1](https://docs.fastf1.dev/)

3D model: *McLaren MP4/5* by
[dark_igorek](https://sketchfab.com/dark_igorek), CC Attribution, via Sketchfab.
Draco-compressed for the web (105.8 MB → 2.5 MB); see `docs/model/CREDITS.md`.

MIT licensed.
