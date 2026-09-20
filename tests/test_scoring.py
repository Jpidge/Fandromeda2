"""Exact scoring tests for FANDROMEDA's authoritative league rules."""

import unittest

import pandas as pd

from fandromeda.scoring import (
    SCORING_VERSION,
    defense_points_allowed,
    offensive_breakdown,
    score_defense_week,
    score_offensive_frame,
)


class LeagueScoringTests(unittest.TestCase):
    def test_core_non_ppr_offensive_stat_line(self):
        result = offensive_breakdown(
            {
                "passing_yards": 250,
                "passing_tds": 2,
                "interceptions": 1,
                "rushing_yards": 20,
                "rushing_tds": 1,
                "receptions": 12,
                "receiving_yards": 40,
                "fumbles_lost": 1,
                "two_point": 1,
            }
        )
        self.assertEqual(result["fantasy_points"], 32.0)
        self.assertEqual(result["touchdown_points"], 18.0)

    def test_event_specific_rules_are_scored_when_supplied(self):
        result = offensive_breakdown(
            {
                "passing_tds": 1,
                "passing_tds_40_plus": 1,
                "rushing_tds_40_plus": 1,
                "receiving_tds_40_plus": 1,
                "return_yards": 75,
                "return_tds": 1,
                "offensive_fumble_return_tds": 1,
                "extra_points_made": 2,
                "field_goal_yards_made": 87,
            }
        )
        self.assertAlmostEqual(result["fantasy_points"], 34.7)
        self.assertEqual(result["touchdown_points"], 21.0)

    def test_frame_scoring_matches_single_stat_line_scoring(self):
        frame = pd.DataFrame(
            [
                {"passing_yards": 125, "passing_tds": 1},
                {
                    "rushing_yards": 80,
                    "receiving_yards": 20,
                    "receiving_tds": 1,
                },
            ]
        )
        scored = score_offensive_frame(frame)
        self.assertEqual(scored["fantasy_points"].tolist(), [11.0, 16.0])
        self.assertEqual(scored["touchdown_points"].tolist(), [6.0, 6.0])

    def test_defense_points_allowed_tiers(self):
        expected = {
            0: 10.0, 1: 7.0, 6: 7.0, 7: 4.0, 13: 4.0, 14: 1.0,
            20: 1.0, 21: 0.0, 27: 0.0, 28: -1.0, 34: -1.0, 35: -4.0,
        }
        for allowed, points in expected.items():
            self.assertEqual(defense_points_allowed(allowed), points)

    def test_complete_defense_stat_line(self):
        score = score_defense_week(
            {
                "sacks": 3,
                "interceptions": 2,
                "fumble_recoveries": 1,
                "defense_tds": 1,
                "safeties": 1,
                "blocked_kicks": 1,
                "return_yards": 50,
                "return_tds": 1,
                "extra_point_returns": 1,
                "points_allowed": 10,
            }
        )
        self.assertEqual(score, 33.0)

    def test_scoring_version_is_explicit(self):
        self.assertEqual(SCORING_VERSION, "ffc_yahoo_v1")


if __name__ == "__main__":
    unittest.main()
