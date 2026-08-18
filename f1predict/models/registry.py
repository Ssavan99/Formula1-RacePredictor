"""The models that ship.

Three approaches run side by side. What each is here for, and what it is not:

* **original: MLP** -- the project's original approach, preserved. Competitive
  on top-1 and retained regardless; it is part of the project's history. Its
  probabilities are not trustworthy (see below).
* **lightgbm: lambdarank** -- the strongest model overall. Best top-1 of any
  model, decisively better ordering and calibration than the original.
* **plackett-luce** -- the only model whose win probabilities form a genuine
  distribution over the field by construction, and the most interpretable.

Two additions were built, measured and **rejected**, and their code is kept so
the negative results stay reproducible:

* **Ensemble** — did not beat LambdaRank on any metric (top-1 -0.029
  [-0.146, +0.087]). See `ensemble.py`.
* **Reliability-adjusted LambdaRank** — improves calibration for real (log-loss
  -0.0279 [-0.0363, -0.0189]) but does not improve *who* it picks (top-1 -0.0097
  [-0.0291, +0.0000]). See `reliability.py`. Retirement appears close to
  irreducible noise at this feature resolution.

`OriginalSVMTuned` is not in the shipping set either -- at 0.427 it sits below
the MLP and Plackett-Luce -- but it stays in the code and the results table
because it demonstrates that the original SVM's weakness was a kernel mismatch
(0.107 -> 0.427), not the method.

Honest framing, which the README leads with: **on top-1 winner accuracy no model
here beats "assume the pole sitter wins" (0.573), and none beats the original
with confidence.** The two additions are justified on ordering and calibration,
where they beat the original by margins whose intervals exclude zero. That
distinction matters for a site that publishes probabilities: the original's
winner log-loss is 5.79 against LambdaRank's 1.48, because it emits ~0.99 for
its pick and ~0 for everyone else and is therefore catastrophically wrong when
it is wrong.
"""

from __future__ import annotations

from .base import RaceModel

#: Models published on the site and scored in the live track record.
PRODUCTION_MODEL_NAMES = (
    "original: MLP",
    "lightgbm: lambdarank",
    "plackett-luce",
)


#: Shrinkage weight toward the grid prior, chosen on 2021 (inside the training
#: era, never the backtest window). See models/anchored.py for why.
ANCHOR_WEIGHT = 0.60


def build_production_models(view: str = "post_quali") -> list[RaceModel]:
    """The adopted race-winner models for a given feature view.

    The anchored variants lead: they match or edge the pole-sitter rule on top-1
    (0.583 vs 0.573, a nominal win well inside the noise) and beat it decisively
    on calibration (log-loss 1.377 vs 1.757, interval excluding zero). The
    unanchored LambdaRank and the original MLP are kept alongside so the effect
    of the anchor is visible rather than asserted.

    Anchoring only makes sense once a grid exists, so the pre-weekend view keeps
    the plain models.
    """
    from .anchored import GridAnchored
    from .choice import PlackettLuceModel
    from .original import OriginalMLP
    from .ranker import LambdaRankModel

    if view == "post_quali":
        return [
            GridAnchored(LambdaRankModel(view=view), weight=ANCHOR_WEIGHT,
                         name="lambdarank + grid anchor"),
            GridAnchored(PlackettLuceModel(view=view), weight=ANCHOR_WEIGHT,
                         name="plackett-luce + grid anchor"),
            LambdaRankModel(view=view),
            OriginalMLP(view=view),
        ]
    return [
        LambdaRankModel(view=view),
        PlackettLuceModel(view=view),
        OriginalMLP(view=view),
    ]


def build_qualifying_models(view: str = "pre_quali") -> list[RaceModel]:
    """Models predicting qualifying order.

    Qualifying is always predicted before the weekend, so the view is fixed to
    ``pre_quali``: there is no grid to condition on, because the grid is what
    is being predicted.

    The original approach is absent here by construction rather than by
    omission -- it classifies "did this driver win the race", which has no
    qualifying analogue without inventing a model the project never had.
    """
    from .choice import PlackettLuceModel
    from .ranker import LambdaRankModel

    return [
        LambdaRankModel(
            view="pre_quali",
            target="quali_position",
            name="lightgbm: lambdarank (qualifying)",
        ),
        PlackettLuceModel(
            view="pre_quali",
            target="quali_position",
            name="plackett-luce (qualifying)",
        ),
    ]
