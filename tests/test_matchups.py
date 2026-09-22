import unittest
import pandas as pd

from fandromeda.matchups import build_defensive_history, attach_defensive_history


class MatchupFeatureTests(unittest.TestCase):
    def test_target_week_is_excluded_from_defensive_history(self):
        stats = pd.DataFrame([
            {"season": 2026, "week": 1, "opponent_team": "KC", "position_group": "QB", "fantasy_points": 10},
            {"season": 2026, "week": 2, "opponent_team": "KC", "position_group": "QB", "fantasy_points": 20},
            {"season": 2026, "week": 3, "opponent_team": "KC", "position_group": "QB", "fantasy_points": 999},
        ])
        history = build_defensive_history(stats, target_week=3)
        self.assertEqual(history.loc[0, "points_allowed"], 30)
        features = attach_defensive_history(pd.DataFrame([{
            "season": 2026, "team": "BUF", "opponent": "KC", "position_group": "QB",
        }]), history, target_week=3)
        self.assertEqual(features.loc[0, "opponent_points_allowed_per_game"], 15)

    def test_future_history_is_rejected_when_supplied(self):
        history = pd.DataFrame([{"season": 2026, "defense_team": "KC", "position_group": "QB",
                                 "points_allowed_per_game": 20, "week": 4}])
        with self.assertRaises(ValueError):
            attach_defensive_history(pd.DataFrame([{"season": 2026, "team": "BUF", "opponent": "KC", "position_group": "QB"}]), history, target_week=3)

    def test_league_scoring_takes_precedence(self):
        stats = pd.DataFrame([{"season": 2026, "week": 1, "opponent": "KC",
                               "position": "QB", "fantasy_points": 10,
                               "fantasy_points_std": 20}])
        history = build_defensive_history(stats, target_week=2)
        self.assertEqual(history.loc[0, "points_allowed"], 20)

    def test_project_canonical_aliases_are_supported(self):
        stats = pd.DataFrame([
            {"season": 2026, "week": 1, "opponent": "KC", "position": "QB", "fantasy_points_std": 12},
        ])
        history = build_defensive_history(stats, target_week=2)
        self.assertEqual(history.loc[0, "points_allowed"], 12)


if __name__ == "__main__":
    unittest.main()
