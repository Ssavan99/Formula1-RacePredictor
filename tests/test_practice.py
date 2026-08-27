"""Tests for the practice-pace join.

Regression coverage for a bug where re-attaching practice features onto a
frame that already carried them produced pandas `_x`/`_y` suffix columns
instead of overwriting -- columns the leak guard then rejected outright.
"""

import numpy as np
import pandas as pd

from f1predict.data.contracts import select_features
from f1predict.data.practice import attach_practice

FEATURE_COLUMNS = ["practice_best_gap", "practice_long_run_gap", "practice_laps"]


def make_races() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025],
            "round": [1, 1],
            "driver_id": ["max_verstappen", "norris"],
        }
    )


def make_practice() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025],
            "round": [1, 1],
            "driver_code": ["VER", "NOR"],
            "practice_best_gap": [0.0, 0.312],
            "practice_long_run_gap": [0.0, 0.201],
            "practice_laps": [24, 22],
        }
    )


class TestAttachPractice:
    def test_attaches_clean_columns(self):
        out = attach_practice(make_races(), make_practice())
        assert list(out["practice_best_gap"]) == [0.0, 0.312]
        assert not any(c.endswith(("_x", "_y")) for c in out.columns)

    def test_idempotent_on_a_frame_that_already_has_the_columns(self):
        """Calling attach_practice twice must not produce _x/_y duplicates."""
        once = attach_practice(make_races(), make_practice())
        twice = attach_practice(once, make_practice())

        assert not any(c.endswith(("_x", "_y")) for c in twice.columns)
        for column in FEATURE_COLUMNS:
            assert column in twice.columns
        assert list(twice["practice_best_gap"]) == [0.0, 0.312]

    def test_result_survives_the_leak_guard(self):
        """The attached columns are registered POST_QUALI, not unregistered."""
        out = attach_practice(make_races(), make_practice())
        selected = select_features(out, "post_quali")
        assert "practice_best_gap" in selected.columns

    def test_empty_practice_table_fills_nan_without_duplicating(self):
        out = attach_practice(make_races(), pd.DataFrame())
        assert out["practice_best_gap"].isna().all()

        twice = attach_practice(out, pd.DataFrame())
        assert not any(c.endswith(("_x", "_y")) for c in twice.columns)
        assert np.isnan(twice["practice_best_gap"]).all()
