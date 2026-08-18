"""Tests for the model layer.

The Plackett-Luce gradient check is the important one here: an analytic gradient
that disagrees with the function it claims to differentiate produces a model
that trains to the wrong place and still looks like it worked.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from f1predict.models.base import FunctionBaseline, _softmax
from f1predict.models.choice import PlackettLuceModel
from f1predict.models.ranker import relevance_from_position


def make_field(n_drivers: int = 6, n_races: int = 12, seed: int = 0) -> pd.DataFrame:
    """Synthetic races where grid position genuinely predicts the finish."""
    rng = np.random.default_rng(seed)
    rows = []
    for rnd in range(1, n_races + 1):
        grid = rng.permutation(np.arange(1, n_drivers + 1))
        noise = rng.normal(0, 0.6, n_drivers)
        finish = np.argsort(np.argsort(grid + noise)) + 1
        for i in range(n_drivers):
            rows.append(
                {
                    "season": 2025,
                    "round": rnd,
                    "race_date": f"2025-01-{rnd:02d}",
                    "driver_id": f"d{i}",
                    "constructor_id": f"c{i % 3}",
                    "circuit_id": "spa",
                    "driver_nationality": "British",
                    "grid": float(grid[i]),
                    "finish_position": float(finish[i]),
                    "is_winner": int(finish[i] == 1),
                    "driver_points_before": float(rng.integers(0, 100)),
                    "driver_form_5": float(rng.uniform(1, 10)),
                    "driver_age": 27.0,
                }
            )
    return pd.DataFrame(rows)


class TestSoftmax:
    def test_sums_to_one(self):
        assert _softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)

    def test_is_shift_invariant(self):
        a = _softmax(np.array([1.0, 2.0, 3.0]))
        b = _softmax(np.array([101.0, 102.0, 103.0]))
        np.testing.assert_allclose(a, b)

    def test_handles_large_values_without_overflow(self):
        out = _softmax(np.array([1e5, 1e5 + 1]))
        assert np.isfinite(out).all()
        assert out.sum() == pytest.approx(1.0)

    def test_nans_do_not_win(self):
        out = _softmax(np.array([1.0, np.nan, 5.0]))
        assert out.sum() == pytest.approx(1.0)
        assert np.argmax(out) == 2

    def test_all_nan_falls_back_to_uniform(self):
        out = _softmax(np.array([np.nan, np.nan]))
        np.testing.assert_allclose(out, [0.5, 0.5])


class TestFunctionBaseline:
    def test_pole_sitter_baseline_picks_the_front_row(self):
        race = pd.DataFrame({"grid": [3.0, 1.0, 2.0]})
        model = FunctionBaseline("pole", "grid", higher_is_better=False)
        assert int(np.argmax(model.predict_scores(race))) == 1

    def test_missing_column_is_survivable(self):
        model = FunctionBaseline("x", "not_there")
        scores = model.predict_scores(pd.DataFrame({"grid": [1.0, 2.0]}))
        assert len(scores) == 2

    def test_probabilities_sum_to_one(self):
        race = pd.DataFrame({"grid": [1.0, 2.0, 3.0]})
        model = FunctionBaseline("pole", "grid", higher_is_better=False)
        assert model.predict_proba(race).sum() == pytest.approx(1.0)


class TestRelevance:
    def test_winner_gets_top_relevance(self):
        assert relevance_from_position(np.array([1.0]))[0] == 20

    def test_relevance_decreases_down_the_order(self):
        rel = relevance_from_position(np.array([1.0, 2.0, 3.0]))
        assert rel[0] > rel[1] > rel[2]

    def test_back_of_field_floors_at_zero(self):
        assert relevance_from_position(np.array([25.0]))[0] == 0

    def test_nan_position_is_zero(self):
        assert relevance_from_position(np.array([np.nan]))[0] == 0


class TestPlackettLuceGradient:
    """The analytic gradient must match finite differences."""

    @staticmethod
    def _toy_races(seed: int = 0):
        rng = np.random.default_rng(seed)
        races = []
        for _ in range(4):
            n = rng.integers(4, 8)
            X = rng.normal(size=(n, 3))
            depth = min(3, n - 1)
            races.append((X, depth))
        return races

    def test_gradient_matches_finite_differences(self):
        races = self._toy_races()
        beta = np.array([0.3, -0.7, 0.2])

        def f(b):
            return PlackettLuceModel._negative_log_likelihood(b, races, alpha=0.5)[0]

        analytic = PlackettLuceModel._negative_log_likelihood(beta, races, alpha=0.5)[1]
        numeric = approx_fprime(beta, f, 1e-6)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)

    def test_gradient_matches_at_zero(self):
        races = self._toy_races(seed=3)

        def f(b):
            return PlackettLuceModel._negative_log_likelihood(b, races, alpha=1.0)[0]

        beta = np.zeros(3)
        analytic = PlackettLuceModel._negative_log_likelihood(beta, races, alpha=1.0)[1]
        np.testing.assert_allclose(analytic, approx_fprime(beta, f, 1e-6), atol=1e-5)

    def test_l2_penalty_shrinks_coefficients(self):
        data = make_field()
        loose = PlackettLuceModel(alpha=0.01).fit(data)
        tight = PlackettLuceModel(alpha=100.0).fit(data)
        assert np.abs(tight.beta).sum() < np.abs(loose.beta).sum()


class TestPlackettLuceLearns:
    def test_recovers_that_a_better_grid_slot_helps(self):
        data = make_field(seed=5)
        model = PlackettLuceModel(alpha=0.1).fit(data)
        coefficients = pd.Series(model.beta, index=model.encoder.feature_names)
        # Lower grid number is better, so its coefficient must be negative.
        assert coefficients["grid"] < 0

    def test_probabilities_sum_to_one_over_the_field(self):
        data = make_field(seed=7)
        model = PlackettLuceModel().fit(data)
        race = data[data["round"] == 1]
        assert model.predict_proba(race).sum() == pytest.approx(1.0)

    def test_choice_set_is_not_truncated_by_depth(self):
        """depth caps modelled steps, not the field competing at each step."""
        data = make_field(n_drivers=8, seed=11)
        shallow = PlackettLuceModel(alpha=0.1, depth=2).fit(data)
        deep = PlackettLuceModel(alpha=0.1, depth=7).fit(data)
        # Both must still see all 8 drivers; a truncated choice set would make
        # the two fits agree trivially on the front of the grid.
        assert shallow.beta.shape == deep.beta.shape
        assert not np.allclose(shallow.beta, deep.beta)
