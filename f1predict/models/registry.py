"""The models that ship.

Three approaches run side by side. What each is here for, and what it is not:

* **original: MLP** -- the project's original approach, preserved. Competitive
  on top-1 and retained regardless; it is part of the project's history. Its
  probabilities are not trustworthy (see below).
* **lightgbm: lambdarank** -- the strongest model overall. Best top-1 of any
  model, decisively better ordering and calibration than the original.
* **plackett-luce** -- the only model whose win probabilities form a genuine
  distribution over the field by construction, and the most interpretable.

A weighted ensemble of all three was built, backtested and **rejected**: it did
not beat LambdaRank on any metric (top-1 -0.029 [-0.146, +0.087], Spearman
+0.005 [-0.004, +0.013], log-loss -0.027 [-0.136, +0.082]). The code is kept in
`ensemble.py` so the negative result is reproducible rather than merely claimed.

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


def build_production_models(view: str = "post_quali") -> list[RaceModel]:
    """The adopted race-winner models for a given feature view."""
    from .choice import PlackettLuceModel
    from .original import OriginalMLP
    from .ranker import LambdaRankModel

    return [
        OriginalMLP(view=view),
        LambdaRankModel(view=view),
        PlackettLuceModel(view=view),
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
