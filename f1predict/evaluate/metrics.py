"""Scoring for race predictions.

The headline metric is top-1 winner accuracy, because it is what the original
project targeted and because it is the one number a reader intuitively
understands. It is also noisy: a season is about 22 races, so a single-season
accuracy carries a standard error near 10 percentage points. Two consequences
run through this module:

* every metric is reported with a bootstrap confidence interval over races, not
  as a bare point estimate;
* rank and calibration metrics sit alongside top-1, because a model can pick
  winners well while ordering the rest of the field badly, or be accurate while
  being wildly overconfident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

#: Clip probabilities before taking logs so a confident miss is finite.
EPS = 1e-15


def top1_correct(predicted_scores: np.ndarray, actual_positions: np.ndarray) -> float:
    """1.0 if the highest-scored driver actually won, else 0.0."""
    if len(predicted_scores) == 0:
        return np.nan
    picked = int(np.argmax(predicted_scores))
    return float(actual_positions[picked] == 1)


def podium_hit_rate(
    predicted_scores: np.ndarray, actual_positions: np.ndarray, k: int = 3
) -> float:
    """Fraction of the actual top-k that appear in the predicted top-k."""
    n = len(predicted_scores)
    if n == 0:
        return np.nan
    k = min(k, n)
    predicted_top = set(np.argsort(-predicted_scores)[:k])
    actual_top = {
        i for i in range(n) if np.isfinite(actual_positions[i]) and actual_positions[i] <= k
    }
    if not actual_top:
        return np.nan
    return len(predicted_top & actual_top) / len(actual_top)


def spearman_rho(predicted_scores: np.ndarray, actual_positions: np.ndarray) -> float:
    """Rank correlation between predicted order and actual finishing order."""
    mask = np.isfinite(actual_positions) & np.isfinite(predicted_scores)
    if mask.sum() < 3:
        return np.nan
    # Higher score should mean a lower (better) finishing position.
    rho, _ = spearmanr(-predicted_scores[mask], actual_positions[mask])
    return float(rho) if np.isfinite(rho) else np.nan


def ndcg_at_k(
    predicted_scores: np.ndarray, actual_positions: np.ndarray, k: int = 10
) -> float:
    """NDCG with relevance decreasing in true finishing position."""
    n = len(predicted_scores)
    if n == 0:
        return np.nan
    relevance = np.where(
        np.isfinite(actual_positions), np.maximum(0.0, 21.0 - actual_positions), 0.0
    )
    order = np.argsort(-predicted_scores)[:k]
    gains = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())

    ideal = np.sort(relevance)[::-1][:k]
    idiscounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float((ideal * idiscounts).sum())
    return dcg / idcg if idcg > 0 else np.nan


def winner_log_loss(probabilities: np.ndarray, actual_positions: np.ndarray) -> float:
    """Negative log probability assigned to the driver who actually won."""
    winners = np.where(actual_positions == 1)[0]
    if len(winners) == 0:
        return np.nan
    p = float(np.clip(probabilities[winners[0]], EPS, 1.0))
    return -np.log(p)


def winner_brier(probabilities: np.ndarray, actual_positions: np.ndarray) -> float:
    """Brier score over the field for the one-hot winner outcome."""
    if len(probabilities) == 0:
        return np.nan
    outcome = (actual_positions == 1).astype(float)
    return float(np.mean((probabilities - outcome) ** 2))


RACE_METRICS = {
    "top1_accuracy": top1_correct,
    "podium_hit_rate": podium_hit_rate,
    "spearman_rho": spearman_rho,
    "ndcg_at_10": ndcg_at_k,
}

PROBABILITY_METRICS = {
    "winner_log_loss": winner_log_loss,
    "winner_brier": winner_brier,
}


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI, resampling races.

    Races are the independent unit, so this resamples races rather than
    driver-rows -- the 20 rows of a single grand prix are anything but
    independent of one another.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lower = float(np.percentile(draws, 100 * alpha / 2))
    upper = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return float(values.mean()), lower, upper


def paired_bootstrap_delta(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap on ``a - b`` over the races both models scored.

    Pairing matters: two models evaluated on the same races share the easy ones
    and the chaotic ones, so an unpaired comparison overstates the uncertainty
    of the difference between them.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {"delta": np.nan, "lower": np.nan, "upper": np.nan, "p_better": np.nan}

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    draws = diff[idx].mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "lower": float(np.percentile(draws, 2.5)),
        "upper": float(np.percentile(draws, 97.5)),
        "p_better": float((draws > 0).mean()),
        "n_races": int(len(diff)),
    }


def summarise(per_race: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Aggregate per-race scores into a mean + CI table, one row per model."""
    rows = []
    for model, group in per_race.groupby("model", sort=False):
        row: dict[str, object] = {"model": model, "n_races": int(len(group))}
        for metric in list(RACE_METRICS) + list(PROBABILITY_METRICS):
            if metric not in group.columns:
                continue
            mean, lower, upper = bootstrap_ci(group[metric].to_numpy(), seed=seed)
            row[metric] = mean
            row[f"{metric}_lo"] = lower
            row[f"{metric}_hi"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values("top1_accuracy", ascending=False)
