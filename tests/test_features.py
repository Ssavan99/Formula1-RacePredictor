"""Tests for row-level leakage in derived features.

The contracts module stops a post-race *column* reaching a model. These tests
stop a pre-race column carrying post-race *information* -- a rolling mean that
includes the race it is meant to predict. That bug is invisible to a column-name
check, so it needs its own tests.
"""

import numpy as np
import pandas as pd
import pytest

from f1predict.data.features import (
    _parse_lap_time,
    add_championship_state,
    add_form_features,
    add_targets,
    flatten_qualifying,
    flatten_results,
)


def make_races(n_rounds: int = 3, drivers=("alice", "bob")) -> pd.DataFrame:
    """Synthetic season: `alice` always wins, `bob` always second."""
    rows = []
    for rnd in range(1, n_rounds + 1):
        for position, driver in enumerate(drivers, start=1):
            rows.append(
                {
                    "season": 2025,
                    "round": rnd,
                    "race_date": f"2025-0{rnd}-01",
                    "race_time_utc": "14:00:00Z",
                    "race_name": f"GP {rnd}",
                    "circuit_id": "silverstone",
                    "circuit_lat": 52.0,
                    "circuit_lon": -1.0,
                    "driver_id": driver,
                    "driver_nationality": "British",
                    "driver_dob": "1995-01-01",
                    "constructor_id": f"team_{driver}",
                    "grid": position,
                    "finish_position": position,
                    "position_text": str(position),
                    "points": 25.0 if position == 1 else 18.0,
                    "status": "Finished",
                    "laps": 50,
                }
            )
    return pd.DataFrame(rows)


class TestChampionshipStateIsShifted:
    def test_first_round_has_no_prior_points(self):
        df = add_championship_state(make_races())
        first = df[df["round"] == 1]
        assert (first["driver_points_before"] == 0).all()
        assert (first["driver_wins_before"] == 0).all()

    def test_second_round_sees_only_the_first(self):
        df = add_championship_state(make_races())
        alice = df[(df["round"] == 2) & (df["driver_id"] == "alice")].iloc[0]
        assert alice["driver_points_before"] == 25.0
        assert alice["driver_wins_before"] == 1.0

    def test_points_before_equals_independently_summed_prior_rounds(self):
        """Checked against a separate computation, not against itself.

        Asserting `points_before < points_before + points` is a tautology
        whenever points > 0, and would pass against a fully leaky
        implementation. This sums the prior rounds directly instead.
        """
        df = add_championship_state(make_races(n_rounds=5))
        raw = make_races(n_rounds=5)
        for _, row in df.iterrows():
            prior = raw[
                (raw["driver_id"] == row["driver_id"])
                & (raw["round"] < row["round"])
            ]["points"].sum()
            assert row["driver_points_before"] == pytest.approx(prior)

    def test_constructor_points_shifted_too(self):
        df = add_championship_state(make_races())
        first = df[df["round"] == 1]
        assert (first["constructor_points_before"] == 0).all()
        third = df[(df["round"] == 3) & (df["driver_id"] == "alice")].iloc[0]
        assert third["constructor_points_before"] == 50.0

    def test_sprint_points_counted_when_present(self):
        raw = make_races(n_rounds=2)
        raw["sprint_points"] = 8.0
        df = add_championship_state(raw)
        alice = df[(df["round"] == 2) & (df["driver_id"] == "alice")].iloc[0]
        assert alice["driver_points_before"] == 33.0  # 25 race + 8 sprint


class TestFormFeaturesAreShifted:
    def test_first_appearance_has_no_form(self):
        df = add_form_features(add_championship_state(make_races()))
        first = df[df["round"] == 1]
        assert first["driver_form_3"].isna().all()
        assert first["driver_circuit_mean_finish"].isna().all()

    def test_form_reflects_only_prior_races(self):
        df = add_form_features(add_championship_state(make_races(n_rounds=3)))
        alice = df[(df["round"] == 3) & (df["driver_id"] == "alice")].iloc[0]
        assert alice["driver_form_3"] == 1.0  # won rounds 1 and 2

    def test_career_starts_excludes_current_race(self):
        df = add_form_features(add_championship_state(make_races(n_rounds=3)))
        alice = df[df["driver_id"] == "alice"].sort_values("round")
        assert alice["driver_career_starts"].tolist() == [0.0, 1.0, 2.0]

    def test_form_is_never_equal_to_current_finish_for_a_changing_driver(self):
        """A driver whose result changes must not have form tracking it exactly."""
        raw = make_races(n_rounds=4)
        # Make bob's finishing position alternate so a leak would be visible.
        mask = raw["driver_id"] == "bob"
        raw.loc[mask, "finish_position"] = [2, 5, 2, 5]
        df = add_form_features(add_championship_state(raw))
        bob = df[df["driver_id"] == "bob"].sort_values("round")
        # Round 2 form must reflect round 1 only (2.0), not round 2's own 5.
        assert bob.iloc[1]["driver_form_3"] == 2.0

    def test_driver_age_is_positive_and_plausible(self):
        df = add_form_features(add_championship_state(make_races()))
        assert df["driver_age"].between(15, 60).all()


class TestTargets:
    def test_winner_and_podium_flags(self):
        df = add_targets(make_races())
        assert df[df["finish_position"] == 1]["is_winner"].eq(1).all()
        assert df[df["finish_position"] == 2]["is_winner"].eq(0).all()
        assert df[df["finish_position"] <= 3]["is_podium"].eq(1).all()


class TestLapTimeParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1:23.456", 83.456),
            ("0:59.999", 59.999),
            ("83.456", 83.456),
            ("2:00.000", 120.0),
        ],
    )
    def test_parses_known_formats(self, text, expected):
        assert _parse_lap_time(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", [None, "", "   ", "not-a-time", "1:xx"])
    def test_returns_none_on_junk(self, text):
        assert _parse_lap_time(text) is None


class TestFlattening:
    def test_flatten_results_shape(self):
        payload = [
            {
                "season": "2026",
                "round": "1",
                "date": "2026-03-08",
                "time": "05:00:00Z",
                "raceName": "Australian Grand Prix",
                "Circuit": {
                    "circuitId": "albert_park",
                    "Location": {"lat": "-37.8497", "long": "144.968"},
                },
                "Results": [
                    {
                        "position": "1",
                        "positionText": "1",
                        "points": "25",
                        "grid": "1",
                        "laps": "58",
                        "status": "Finished",
                        "Driver": {
                            "driverId": "norris",
                            "nationality": "British",
                            "dateOfBirth": "1999-11-13",
                        },
                        "Constructor": {"constructorId": "mclaren"},
                    }
                ],
            }
        ]
        df = flatten_results(payload)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["driver_id"] == "norris"
        assert row["finish_position"] == 1
        assert row["circuit_lat"] == pytest.approx(-37.8497)

    def test_qualifying_gap_to_pole_is_within_session(self):
        payload = [
            {
                "season": "2026",
                "round": "1",
                "QualifyingResults": [
                    {"position": "1", "Q3": "1:20.000", "Driver": {"driverId": "a"}},
                    {"position": "2", "Q3": "1:20.500", "Driver": {"driverId": "b"}},
                ],
            }
        ]
        df = flatten_qualifying(payload)
        assert df[df["driver_id"] == "a"]["quali_gap_to_pole"].iloc[0] == pytest.approx(0.0)
        assert df[df["driver_id"] == "b"]["quali_gap_to_pole"].iloc[0] == pytest.approx(0.5)
        assert df[df["driver_id"] == "a"]["is_pole"].iloc[0] == 1

    def test_qualifying_falls_back_through_sessions(self):
        """A driver knocked out in Q1 still gets a time."""
        payload = [
            {
                "season": "2026",
                "round": "1",
                "QualifyingResults": [
                    {
                        "position": "15",
                        "Q1": "1:22.000",
                        "Q2": "",
                        "Driver": {"driverId": "c"},
                    }
                ],
            }
        ]
        df = flatten_qualifying(payload)
        assert df.iloc[0]["quali_time_seconds"] == pytest.approx(82.0)
