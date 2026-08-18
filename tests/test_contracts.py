"""Tests for the leak guard.

The first test is the regression test for the defect that motivated this module:
the original modelling code dropped only ['driver', 'podium', 'points'] and so
fed the race's own finishing status into the model.
"""

import pandas as pd
import pytest

from f1predict.data.contracts import (
    Availability,
    LeakageError,
    assert_pre_race,
    availability_of,
    feature_columns,
    select_features,
)


ORIGINAL_LEAKY_COLUMNS = [
    "status_Finished",
    "status_Illness",
    "status_Incident",
    "status_Mechanical Issue",
]


class TestTheOriginalLeak:
    """The exact defect found in the 2023 pipeline must not be expressible."""

    @pytest.mark.parametrize("column", ORIGINAL_LEAKY_COLUMNS)
    def test_race_status_is_post_race(self, column):
        assert availability_of(column) is Availability.POST_RACE

    @pytest.mark.parametrize("view", ["pre_quali", "post_quali"])
    def test_status_columns_never_selected_as_features(self, view):
        columns = ["driver_age", "grid", *ORIGINAL_LEAKY_COLUMNS]
        selected = feature_columns(columns, view)
        assert not set(selected) & set(ORIGINAL_LEAKY_COLUMNS)

    @pytest.mark.parametrize("view", ["pre_quali", "post_quali"])
    def test_status_columns_rejected_by_the_gate(self, view):
        """Selection excludes them; validation refuses them outright."""
        df = pd.DataFrame(columns=["driver_age", *ORIGINAL_LEAKY_COLUMNS])
        with pytest.raises(LeakageError, match="POST_RACE"):
            assert_pre_race(df, view)

    def test_points_scored_in_race_is_post_race(self):
        assert availability_of("points") is Availability.POST_RACE

    def test_reproducing_the_original_drop_list_now_fails(self):
        """`drop(['driver', 'podium', 'points'])` used to leave the leak in."""
        full = pd.DataFrame(
            columns=[
                "driver_id",
                "podium",
                "points",
                "grid",
                "driver_age",
                *ORIGINAL_LEAKY_COLUMNS,
            ]
        )
        as_the_original_did = full.drop(columns=["driver_id", "podium", "points"])
        with pytest.raises(LeakageError):
            assert_pre_race(as_the_original_did, "post_quali")


class TestViews:
    def test_grid_excluded_before_qualifying(self):
        assert "grid" not in feature_columns(["driver_age", "grid"], "pre_quali")

    def test_grid_included_after_qualifying(self):
        assert "grid" in feature_columns(["driver_age", "grid"], "post_quali")

    def test_quali_derived_features_excluded_pre_weekend(self):
        cols = ["quali_time_seconds", "quali_gap_to_pole", "driver_age"]
        assert feature_columns(cols, "pre_quali") == ["driver_age"]

    def test_unknown_view_rejected(self):
        with pytest.raises(ValueError, match="unknown view"):
            feature_columns(["driver_age"], "race_day")


class TestFailsClosed:
    """An unregistered column is rejected rather than admitted."""

    def test_unregistered_column_raises(self):
        with pytest.raises(LeakageError, match="Unregistered column"):
            feature_columns(["driver_age", "mystery_feature"], "post_quali")

    def test_error_names_the_offending_column(self):
        with pytest.raises(LeakageError, match="mystery_feature"):
            feature_columns(["mystery_feature"], "post_quali")


class TestPrefixResolution:
    def test_one_hot_constructor_is_pre_quali(self):
        assert availability_of("constructor_ferrari") is Availability.PRE_QUALI

    def test_exact_match_beats_prefix(self, monkeypatch):
        """An exact registry entry must win over a matching prefix rule.

        Tested with an injected conflict rather than a real column: today no
        registered column disagrees with its prefix, so a real example would
        pass for the wrong reason and stop protecting anything.
        """
        from f1predict.data import contracts

        monkeypatch.setitem(
            contracts.REGISTRY, "status_special_case", Availability.PRE_QUALI
        )
        # The `status_` prefix says POST_RACE; the exact entry must override it.
        assert availability_of("status_special_case") is Availability.PRE_QUALI
        assert availability_of("status_anything_else") is Availability.POST_RACE

    def test_constructor_points_is_not_read_as_a_one_hot_column(self):
        assert availability_of("constructor_points_before") is Availability.PRE_QUALI

    def test_unknown_status_column_still_banned_by_prefix(self):
        assert availability_of("status_Disqualified") is Availability.POST_RACE

    def test_circuit_one_hot_is_pre_quali(self):
        assert availability_of("circuit_id_monza") is Availability.PRE_QUALI


class TestAssertPreRace:
    def test_target_cannot_be_its_own_feature(self):
        df = pd.DataFrame(columns=["driver_age", "is_winner"])
        with pytest.raises(LeakageError, match="TARGET"):
            assert_pre_race(df, "post_quali")

    def test_identifiers_must_be_dropped_before_fitting(self):
        df = pd.DataFrame(columns=["driver_age", "driver_id"])
        with pytest.raises(LeakageError, match="IDENTIFIER"):
            assert_pre_race(df, "post_quali")

    def test_clean_matrix_passes(self):
        df = pd.DataFrame(columns=["driver_age", "driver_form_5", "grid"])
        assert_pre_race(df, "post_quali")  # must not raise

    def test_grid_in_pre_quali_matrix_is_rejected(self):
        df = pd.DataFrame(columns=["driver_age", "grid"])
        with pytest.raises(LeakageError, match="not available in view"):
            assert_pre_race(df, "pre_quali")


class TestSelectFeatures:
    def test_selects_and_strips_identifiers_and_targets(self):
        df = pd.DataFrame(
            {
                "season": [2026],
                "driver_id": ["norris"],
                "driver_age": [26],
                "grid": [1],
                "is_winner": [1],
                "points": [25.0],
                "status_Finished": [1.0],
            }
        )
        out = select_features(df, "post_quali")
        assert set(out.columns) == {"driver_age", "grid"}

    def test_pre_quali_view_drops_grid(self):
        df = pd.DataFrame(
            {"driver_id": ["norris"], "driver_age": [26], "grid": [1], "points": [25.0]}
        )
        out = select_features(df, "pre_quali")
        assert set(out.columns) == {"driver_age"}

    def test_extra_drop_respected(self):
        df = pd.DataFrame({"driver_age": [26], "grid": [1]})
        out = select_features(df, "post_quali", extra_drop=["grid"])
        assert set(out.columns) == {"driver_age"}
