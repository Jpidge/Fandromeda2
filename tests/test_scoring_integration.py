"""Integration coverage for attaching PBP event adjustments to weekly stats."""

import unittest

import pandas as pd

from fandromeda.pbp_scoring import EVENT_COLUMNS
from gptindex import standardize_player_stats


class ScoringIntegrationTests(unittest.TestCase):
    def test_pbp_events_are_scored_by_the_canonical_engine(self):
        weekly = pd.DataFrame(
            {
                "player_id": ["qb"],
                "player_name": ["Example QB"],
                "season": [2025],
                "week": [3],
                "position": ["QB"],
                "recent_team": ["NE"],
                "passing_yards": [250],
                "passing_tds": [1],
            }
        )
        events = pd.DataFrame(
            {
                "season": [2025],
                "week": [3],
                "player_id": ["qb"],
                **{
                    event: [1.0 if event == "passing_tds_40_plus" else 0.0]
                    for event in EVENT_COLUMNS
                },
            }
        )

        scored = standardize_player_stats(weekly, events)
        # 250 passing yards / 25 + six-point TD + 40+ TD bonus.
        self.assertEqual(scored.loc[0, "fantasy_points_std"], 17.0)
        self.assertEqual(scored.loc[0, "td_points"], 7.0)


if __name__ == "__main__":
    unittest.main()
