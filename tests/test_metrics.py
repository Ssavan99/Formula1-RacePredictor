"""Tests for the scoring layer."""

import numpy as np
import pandas as pd
import pytest

from f1predict.evaluate.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    paired_bootstrap_delta,
    podium_hit_rate,
    spearman_rho,
    summarise,
    top1_correct,
    winner_brier,
    winner_log_loss,
)


class TestTop1:
    def test_correct_pick_scores_one(self):
        scores = np.array([0.9, 0.1, 0.2])
        positions = np.array([1.0, 2.0, 3.0])
        assert top1_correct(scores, positions) == 1.0

    def test_wrong_pick_scores_zero(self):
        scores = np.array([0.1, 0.9, 0.2])
        positions = np.array([1.0, 2.0, 3.0])
        assert top1_correct(scores, positions) == 0.0

    def test_empty_race_is_nan(self):
        assert np.isnan(top1_correct(np.array([]), np.array([])))


class TestPodium:
    def test_perfect_podium(self):
        scores = np.array([3.0, 2.0, 1.0, 0.0])
        positions = np.array([1.0, 2.0, 3.0, 4.0])
        assert podium_hit_rate(scores, positions) == 1.0

    def test_partial_podium(self):
        # Predicts drivers 0,1,2; actual podium is 0,1,3 -> 2 of 3.
        scores = np.array([3.0, 2.0, 1.0, 0.5])
        positions = np.array([1.0, 2.0, 4.0, 3.0])
        assert podium_hit_rate(scores, positions) == pytest.approx(2 / 3)

    def test_order_within_podium_does_not_matter(self):
        scores = np.array([1.0, 2.0, 3.0, 0.0])
        positions = np.array([1.0, 2.0, 3.0, 4.0])
        assert podium_hit_rate(scores, positions) == 1.0


class TestRankMetrics:
    def test_spearman_perfect_order(self):
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        positions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert spearman_rho(scores, positions) == pytest.approx(1.0)

    def test_spearman_reversed_order(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        positions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert spearman_rho(scores, positions) == pytest.approx(-1.0)

    def test_ndcg_perfect_is_one(self):
        scores = np.array([5.0, 4.0, 3.0])
        positions = np.array([1.0, 2.0, 3.0])
        assert ndcg_at_k(scores, positions) == pytest.approx(1.0)

    def test_ndcg_worse_ordering_scores_lower(self):
        positions = np.array([1.0, 2.0, 3.0, 4.0])
        good = ndcg_at_k(np.array([4.0, 3.0, 2.0, 1.0]), positions)
        bad = ndcg_at_k(np.array([1.0, 2.0, 3.0, 4.0]), positions)
        assert good > bad


class TestProbabilityMetrics:
    def test_log_loss_rewards_confidence_in_the_winner(self):
        positions = np.array([1.0, 2.0, 3.0])
        confident = winner_log_loss(np.array([0.9, 0.05, 0.05]), positions)
        unsure = winner_log_loss(np.array([0.34, 0.33, 0.33]), positions)
        assert confident < unsure

    def test_log_loss_punishes_confident_miss(self):
        positions = np.array([1.0, 2.0, 3.0])
        wrong = winner_log_loss(np.array([0.01, 0.98, 0.01]), positions)
        assert wrong > 4.0

    def test_log_loss_is_finite_on_zero_probability(self):
        positions = np.array([1.0, 2.0])
        assert np.isfinite(winner_log_loss(np.array([0.0, 1.0]), positions))

    def test_brier_perfect_is_zero(self):
        positions = np.array([1.0, 2.0, 3.0])
        assert winner_brier(np.array([1.0, 0.0, 0.0]), positions) == pytest.approx(0.0)


class TestBootstrap:
    def test_mean_is_recovered(self):
        values = np.array([1.0, 0.0, 1.0, 0.0])
        mean, lo, hi = bootstrap_ci(values, n_boot=2000)
        assert mean == pytest.approx(0.5)
        assert lo < mean < hi

    def test_interval_narrows_with_more_races(self):
        rng = np.random.default_rng(0)
        small = rng.binomial(1, 0.5, 20).astype(float)
        large = rng.binomial(1, 0.5, 2000).astype(float)
        _, lo_s, hi_s = bootstrap_ci(small, n_boot=2000)
        _, lo_l, hi_l = bootstrap_ci(large, n_boot=2000)
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_ignores_nans(self):
        mean, _, _ = bootstrap_ci(np.array([1.0, np.nan, 1.0]), n_boot=500)
        assert mean == pytest.approx(1.0)

    def test_empty_is_nan(self):
        mean, lo, hi = bootstrap_ci(np.array([]))
        assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


class TestPairedComparison:
    def test_detects_a_real_difference(self):
        a = np.ones(100)
        b = np.zeros(100)
        result = paired_bootstrap_delta(a, b, n_boot=2000)
        assert result["delta"] == pytest.approx(1.0)
        assert result["lower"] > 0
        assert result["p_better"] == pytest.approx(1.0)

    def test_identical_models_are_not_distinguishable(self):
        rng = np.random.default_rng(1)
        a = rng.binomial(1, 0.5, 100).astype(float)
        result = paired_bootstrap_delta(a, a.copy(), n_boot=2000)
        assert result["delta"] == pytest.approx(0.0)
        assert result["lower"] <= 0 <= result["upper"]

    def test_pairs_only_shared_races(self):
        a = np.array([1.0, np.nan, 1.0])
        b = np.array([0.0, 1.0, 0.0])
        result = paired_bootstrap_delta(a, b, n_boot=500)
        assert result["n_races"] == 2


class TestSummarise:
    def test_produces_one_row_per_model_with_intervals(self):
        per_race = pd.DataFrame(
            {
                "model": ["a"] * 5 + ["b"] * 5,
                "top1_accuracy": [1.0, 0, 1.0, 0, 1.0] + [0.0] * 5,
                "podium_hit_rate": [0.5] * 10,
                "spearman_rho": [0.3] * 10,
                "ndcg_at_10": [0.7] * 10,
                "winner_log_loss": [1.0] * 10,
                "winner_brier": [0.1] * 10,
            }
        )
        table = summarise(per_race)
        assert len(table) == 2
        assert table.iloc[0]["model"] == "a"  # sorted by top-1 descending
        assert "top1_accuracy_lo" in table.columns
