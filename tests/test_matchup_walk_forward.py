import unittest
from unittest.mock import patch
import pandas as pd
import gptindex as app


class MatchupWalkForwardTests(unittest.TestCase):
    def test_target_schedule_opponent_and_season_replace_stale_history(self):
        stats = pd.DataFrame([
            dict(player_id="p", season=2024, week=1, team="BUF", opponent="MIA", position="QB", fantasy_points_std=10),
            dict(player_id="q", season=2024, week=1, team="BAL", opponent="KC", position="QB", fantasy_points_std=30),
            dict(player_id="p", season=2024, week=2, team="BUF", opponent="KC", position="QB", fantasy_points_std=999),
        ])
        features = stats.iloc[[0]].copy()
        features["season"] = 2023  # carry-over identity must not drive the join
        schedule = pd.DataFrame([dict(season=2024, week=2, home_team="BUF", away_team="KC")])
        with patch.object(app, "load_cached", return_value=stats), \
             patch.object(app, "standardize_player_stats", side_effect=lambda df, *_: df), \
             patch.object(app, "load_scoring_event_adjustments", return_value=pd.DataFrame()), \
             patch.object(app, "load_snaps", return_value=pd.DataFrame()), \
             patch.object(app, "merge_usage", side_effect=lambda df, *_: df), \
             patch.object(app, "load_prior_season_carryover", return_value=pd.DataFrame()), \
             patch.object(app, "load_schedule", return_value=schedule), \
             patch.object(app, "build_feature_history", return_value=stats.iloc[:2]), \
             patch.object(app, "add_player_features", return_value=features):
            result = app.build_walk_forward_training([2024], include_matchup_features=True)
        self.assertEqual(result.iloc[0].opponent, "KC")
        self.assertEqual(result.iloc[0].opponent_points_allowed_per_game, 30)
        self.assertEqual(result.iloc[0].target_next_week, 999)
