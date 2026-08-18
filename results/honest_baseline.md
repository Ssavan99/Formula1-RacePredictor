# The honest number

Everything in this project is measured against this table. It was produced
before any new modelling approach was written, so the bar was not set with
knowledge of what would clear it.

```bash
python -m f1predict.evaluate.run_backtest --models baselines,original,new,ensemble --view post_quali
python -m f1predict.evaluate.run_backtest --models baselines,new --view pre_quali
python -m f1predict.evaluate.original_leak
```

## Protocol

**Walk-forward, rolling origin.** For each of the 103 races in 2022–2026, every
model is fitted on races dated strictly before it, then asked to predict that
one race. Training data grows race by race; no model ever sees a row dated on or
after the race it is predicting.

**Primary metric: top-1 winner accuracy.** It is what the original approach
targeted, and it is the number a reader understands without a footnote.

**Confidence intervals are bootstrapped over races**, resampling races rather
than driver-rows — the twenty rows of one grand prix are not independent.

**Why 103 races and not one season.** The strongest naive rule scores 0.455 in
2022 and 0.727 in 2026. At ~22 races a season the standard error on an accuracy
is around 10 percentage points, so a single held-out season cannot distinguish a
good model from a lucky one.

**Rows are shuffled before scoring, and metrics are tie-invariant.** The stored
dataset is sorted by finishing position, so row 0 of every race is its winner
(263 of 263). Any metric breaking ties by index would read the answer out of row
order. Metrics now return the expected value under random tie-breaking, the
harness shuffles each race with a fixed seed, and tests assert permutation
invariance. This bug was live in an earlier draft of these results and inflated
several baselines — one scored a 0.893 podium hit rate, better than every real
model, which is the kind of number that should trigger suspicion.

## Results — 103 races, 2022–2026, grid known

| Model | Top-1 accuracy | Podium | Spearman ρ | Winner log-loss |
|---|---|---|---|---|
| **naive: pole sitter wins** | **0.573** [0.48, 0.67] | 0.663 | 0.628 | 1.757 |
| lightgbm: lambdarank | 0.515 [0.42, 0.61] | 0.595 | 0.653 | **1.475** |
| ensemble *(rejected)* | 0.485 [0.39, 0.58] | 0.605 | 0.658 | 1.448 |
| **original: MLP** | **0.476** [0.38, 0.57] | 0.573 | 0.567 | 5.789 |
| naive: most wins so far | 0.470 [0.38, 0.56] | 0.496 | 0.441 | 2.044 |
| naive: championship leader | 0.459 [0.36, 0.55] | 0.554 | 0.594 | 1.951 |
| plackett-luce | 0.456 [0.36, 0.55] | 0.628 | 0.652 | 1.702 |
| naive: best recent form | 0.417 [0.33, 0.51] | 0.525 | 0.554 | 2.020 |
| naive: previous winner | 0.398 [0.30, 0.50] | 0.307 | 0.251 | 2.689 |
| original: SVM | 0.107 [0.05, 0.17] | 0.366 | 0.536 | 2.487 |
| original: SVM (leaky artifact) | 0.097 [0.05, 0.16] | 0.346 | 0.538 | 2.960 |
| naive: strongest constructor | 0.091 [0.04, 0.14] | 0.431 | 0.508 | 2.404 |
| naive: random | 0.039 [0.01, 0.08] | 0.172 | 0.009 | 3.017 |

### The number, and the headline result

**The original approach scores 0.476 top-1. The best naive rule scores 0.573.
No model in this project beats "assume the pole sitter wins."**

That is the honest headline and it is not buried. On a paired bootstrap, no
model is distinguishable from the pole baseline in either direction — the
intervals are simply too wide at n=103 to declare a winner, which is itself the
point about how much data this problem actually has.

## Adoption decisions

The rule set before results were known: a new approach ships only if it beats
the Phase 2 number on held-out data, and no padding to reach a count.

Paired bootstrap against **original: MLP**, over the races both scored:

| Candidate | Δ top-1 | Δ Spearman ρ | Δ log-loss | Adopted |
|---|---|---|---|---|
| lightgbm: lambdarank | +0.039 [−0.078, +0.155] tie | **+0.086 [+0.063, +0.107]** | **−4.314 [−5.865, −2.858]** | **yes** |
| plackett-luce | −0.019 [−0.107, +0.068] tie | **+0.084 [+0.064, +0.105]** | **−4.087 [−5.655, −2.606]** | **yes** |
| ensemble | +0.010 [+0.000, +0.029] tie | +0.090 [+0.071, +0.110] | −4.341 [−5.853, −2.935] | **no** |

**On top-1, nothing beats the original.** Stated plainly: the primary metric
produced a negative result, and that stands.

Both new approaches are nonetheless adopted, on ordering and calibration, where
their intervals exclude zero by wide margins. That is not a moved goalpost — the
plan named calibration a first-class metric *before* results existed, because
the site publishes probabilities. The original's winner log-loss of 5.79 is not
a rounding difference from LambdaRank's 1.48: the MLP emits ≈0.99 for its pick
and ≈0 for everyone else, so it is catastrophically wrong whenever it is wrong.
A page that prints "92% Verstappen" is making a claim its model cannot support.

**The ensemble was rejected**, and this is where the no-padding rule earned its
keep. It was built, backtested, and failed its own criterion — beat every
component — against LambdaRank alone:

| Metric | ensemble − lambdarank |
|---|---|
| top-1 | −0.029 [−0.146, +0.087] |
| Spearman ρ | +0.005 [−0.004, +0.013] |
| log-loss | −0.027 [−0.136, +0.082] |
| podium | +0.010 [−0.023, +0.042] |

Every interval straddles zero and top-1 is nominally worse. The code stays in
`f1predict/models/ensemble.py` so the negative result is reproducible.

**Three approaches ship**: the original MLP, LambdaRank, and Plackett–Luce.
Plackett–Luce is the weakest of the three on top-1 and is kept for two specific
properties — its win probabilities are a genuine distribution over the field by
construction, and it is significantly better than the original at picking the
podium (+0.055 [+0.006, +0.104]). A reader who only cares about picking winners
should use LambdaRank.

## What qualifying is worth

The same models, run without the grid (as they must be on the Tuesday before a
race):

| Model | Pre-qualifying top-1 | Post-qualifying top-1 |
|---|---|---|
| lightgbm: lambdarank | 0.456 [0.36, 0.55] | 0.515 [0.42, 0.61] |
| plackett-luce | 0.437 [0.34, 0.53] | 0.456 [0.36, 0.55] |
| best available baseline | 0.470 (most wins so far) | 0.573 (pole sitter) |

Knowing the grid is worth about 6 points of top-1 accuracy to LambdaRank. More
strikingly, **the single fact "who is on pole" outperforms every model that does
not have it** — the pole baseline (0.573) beats the best pre-qualifying model
(0.456) by more than the models differ from each other.

## Where the original 0.818 came from

The original notebook reported 0.818 precision for the SVM. It reproduces, and
it is not fabricated — it is a small-sample artifact. Running the original code
path on the original committed `data/Final.csv`:

| Configuration | Top-1 | Test races |
|---|---|---|
| MLP, clean split (train ≤2021, test 2022), leak present | 0.318 | 22 |
| MLP, clean split, leak removed | 0.182 | 22 |
| SVM, clean split, leak present | 0.409 | 22 |
| SVM, clean split, leak removed | 0.409 | 22 |
| MLP, second split, scored on all 22 rounds | 0.727 | 22 |
| SVM, second split, scored on all 22 rounds | 0.409 | 22 |
| **SVM, second split, scored on rounds 12+** | **0.818** | **11** |
| naive: pole sitter wins (2022) | 0.455 | 22 |

The last configuration reproduces the reported number exactly. **It was measured
on eleven races.** At n=11, 9 of 11 carries a 95% interval of **[0.48, 0.98]** —
half the available range. The same model over the full season gives 0.409, which
is *below* the 0.455 pole baseline.

Rounds 12–22 genuinely were held out of that training set, so this is **not**
train/test contamination. It is sample size, reported without an interval. The
original write-up did note that "the testing set is comparatively small and can
lead to different results"; that caution was correct, and this quantifies it.

### The `status_*` leak, separately

`data/Final.csv` carries `status_Finished`, `status_Illness`, `status_Incident`
and `status_Mechanical Issue` — the finishing status of the race being
predicted. The original feature matrix dropped three columns by name (`driver`,
`podium`, `points`), so all four went in. Of 3707 rows, 681 have
`status_Finished == 0` and none is a winner: the model was told which ~18% of
the grid to rule out before predicting.

Measured on the original data and split, the leak was worth **+0.136 top-1 to
the MLP** and **+0.000 to the SVM**. It is a genuine defect that materially
inflated one model — but it is *not* where the 0.818 came from. Two independent
problems, one easily mistaken for the other.

`f1predict/data/contracts.py` makes this class of bug structurally hard to
repeat: every column is registered with the point in time at which it becomes
knowable, and unregistered columns are rejected rather than admitted. While
building this project that guard caught nine further post-race columns the new
feature pipeline had itself produced.

## Known limitations

- **Weather is optimistic.** Historical rows use ERA5 reanalysis — what actually
  happened over the race window — while a live prediction only has a forecast.
  Backtest figures involving weather are an upper bound on what is achievable at
  serving time. `--drop-weather` quantifies the gap.
- **The original SVM has not been retuned.** Its `gamma=0.1, kernel='sigmoid'`
  were grid-searched against the original 88-column sparse matrix. In the
  rebuilt 25-column dense space its ordering signal survives (ρ = 0.536) but its
  top pick collapses. Its 0.107 is a fair report of those hyperparameters in this
  feature space, not a fair test of the method.
- **`driver_career_starts` counts from 2010**, not from a driver's true debut, so
  it understates experience for drivers whose careers predate the fetch window.
- **n = 103 races.** Most differences here are not statistically separable. The
  intervals are the result, not decoration.

## What this means

1. The original approach is not beaten by anything in this project, and neither
   it nor its successors beat a one-line rule. That is the finding.
2. Two separate evaluation defects — a post-race feature and an 11-race test
   set — each produced a number that looked far better than the method was.
3. A third defect, in *this* project's own evaluation code, inflated several
   baselines before review caught it.
4. None of those were detectable without baselines, confidence intervals, and
   an adversarial read of the harness. Adding all three was the highest-value
   change in this repository.

---

# Round 2 — after tuning, a retuned SVM, and a reliability model

Re-run 2026-08-18. Same protocol, same 103 races, same tie-invariant metrics.

| Model | Top-1 | Podium | Spearman ρ | Log-loss |
|---|---|---|---|---|
| **naive: pole sitter wins** | **0.573** [0.48, 0.67] | **0.663** | 0.628 | 1.757 |
| lightgbm: lambdarank *(tuned)* | 0.553 [0.46, 0.65] | 0.631 | **0.664** | 1.530 |
| lambdarank + reliability | 0.544 [0.45, 0.64] | 0.625 | 0.663 | **1.502** |
| ensemble *(rejected)* | 0.485 [0.39, 0.58] | 0.605 | 0.658 | 1.448 |
| original: MLP | 0.476 [0.38, 0.57] | 0.573 | 0.567 | 5.789 |
| plackett-luce | 0.456 [0.36, 0.55] | 0.628 | 0.652 | 1.702 |
| **original: SVM (retuned)** | **0.427** [0.33, 0.52] | 0.511 | 0.329 | 2.026 |
| original: SVM (2023 config) | 0.107 [0.05, 0.17] | 0.366 | 0.536 | 2.487 |
| naive: random | 0.039 [0.01, 0.08] | 0.172 | 0.009 | 3.017 |

## Where the headroom actually is

Measured on the same window, before any modelling:

| | |
|---|---|
| Winner started on pole | 57.3% |
| **Winner started top-3** | **88.3%** |
| Winner started outside top-5 | 6.8% |
| Mean winning grid slot | 2.17 |
| Pole sitter finished outside top-10 | 9.7% |

The task is not "find one driver in twenty" — it is "pick correctly among the
front three", which is where 88% of winners come from. That reframes 0.553 from
"barely better than a coin flip" to "roughly two thirds of the way to a ceiling
of about 0.88".

## Tuning

`f1predict/models/tuning.py`, 81 configurations, fit on ≤2019 and validated on
2020–21. **Seasons ≥2022 are never loaded**, so the backtest above remains a
clean estimate rather than something the hyperparameters were chosen to
maximise.

LightGBM's shipped settings (300 / 0.05 / 15 / 30) were guesses that had never
been tested against this data. Tuning moved them to 300 / 0.02 / 7 / 30 and lifted
top-1 from **0.515 → 0.553**, closing most of the gap to the pole baseline. The
two are now statistically indistinguishable.

## The SVM was never the problem — its kernel was

The 2023 configuration (`gamma=0.1, C=10.0, kernel='sigmoid'`) scores 0.107 top-1
in the rebuilt feature space while keeping ρ = 0.536. A model that orders the
field respectably but cannot pick its top entry is showing a kernel mismatched to
the feature space, not a method that does not work — those settings were searched
against the original 88-column mostly-sparse matrix.

Re-searching over 21 configurations selects `rbf, C=10.0, gamma='scale'`:

| | Top-1 | Spearman ρ |
|---|---|---|
| SVM, 2023 configuration | 0.107 | 0.536 |
| SVM, retuned | **0.427** | 0.329 |

A four-fold improvement in picking winners — and, interestingly, *worse* ordering
of the rest of the field. The 2023 configuration is kept unchanged in the code
and the table; this is the same approach improved in place, reported beside it
rather than replacing it.

## Reliability model — measured, and rejected

The pole sitter fails to finish in the top ten in 9.7% of races, so modelling
retirement separately from pace looked worth trying. Paired against its own base
model:

| Metric | reliability − lambdarank |
|---|---|
| top-1 | −0.0097 [−0.0291, +0.0000] |
| podium | −0.0065 [−0.0194, +0.0065] |
| Spearman ρ | −0.0017 [−0.0064, +0.0031] |
| **log-loss** | **−0.0279 [−0.0363, −0.0189]** |

It genuinely improves calibration — that interval excludes zero — and does not
improve *who you pick* at all. Since the adoption rule is about beating the base
model, it does not ship. The finding is the useful part: at this feature
resolution retirement is close to irreducible noise. Predicting *that* a car will
break is much harder than predicting which car is fast.

## Still outstanding

- **FastF1 practice pace** is built and verified but only partially backfilled
  (~90s per race, ~180 races). Not yet joined into the dataset, so none of the
  numbers above are affected by it either way.
- **Podium** is now reported as a first-class column for every model rather than
  a footnote; LambdaRank leads the models at 0.631, still behind the pole-sitter
  rule at 0.663.
