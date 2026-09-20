"""Deterministic nflverse-shaped play-by-play scoring-event tests."""

import unittest

import pandas as pd

from fandromeda.pbp_scoring import (
    EVENT_COLUMNS,
    available_event_fields,
    summarize_pbp_scoring_events,
)


class PlayByPlayScoringTests(unittest.TestCase):
    def test_reconstructs_player_events_without_scoring_them_twice(self):
        plays = pd.DataFrame(
            [
                {
                    "season": 2025, "week": 3, "season_type": "REG",
                    "touchdown": 1, "yards_gained": 45, "pass_touchdown": 1,
                    "rush_touchdown": 0, "passer_player_id": "qb", "receiver_player_id": "wr",
                },
                {
                    "season": 2025, "week": 3, "season_type": "REG",
                    "touchdown": 1, "yards_gained": 42, "rush_touchdown": 1,
                    "pass_touchdown": 0, "rusher_player_id": "rb",
                },
                {
                    "season": 2025, "week": 3, "season_type": "REG",
                    "returner_player_id": "returner", "return_yards": 55,
                    "return_touchdown": 1,
                },
                {
                    "season": 2025, "week": 3, "season_type": "REG",
                    "kicker_player_id": "k", "extra_point_result": "good",
                },
                {
                    "season": 2025, "week": 3, "season_type": "REG",
                    "kicker_player_id": "k", "field_goal_result": "made",
                    "kick_distance": 47,
                },
                # Postseason events cannot contaminate regular-season scoring.
                {
                    "season": 2025, "week": 19, "season_type": "POST",
                    "touchdown": 1, "yards_gained": 80, "pass_touchdown": 1,
                    "passer_player_id": "qb", "receiver_player_id": "wr",
                },
            ]
        )

        result = summarize_pbp_scoring_events(plays).set_index("player_id")
        self.assertEqual(result.loc["qb", "passing_tds_40_plus"], 1.0)
        self.assertEqual(result.loc["wr", "receiving_tds_40_plus"], 1.0)
        self.assertEqual(result.loc["rb", "rushing_tds_40_plus"], 1.0)
        self.assertEqual(result.loc["returner", "return_yards"], 55.0)
        self.assertEqual(result.loc["returner", "return_tds"], 1.0)
        self.assertEqual(result.loc["k", "extra_points_made"], 1.0)
        self.assertEqual(result.loc["k", "field_goal_yards_made"], 47.0)
        self.assertEqual(set(EVENT_COLUMNS).issubset(result.columns), True)
        availability = available_event_fields(plays)
        self.assertTrue(availability["passing_tds_40_plus"])
        self.assertFalse(availability["offensive_fumble_return_tds"])

    def test_requires_season_and_week(self):
        with self.assertRaisesRegex(ValueError, "season"):
            summarize_pbp_scoring_events(pd.DataFrame({"week": [1]}))


if __name__ == "__main__":
    unittest.main()
