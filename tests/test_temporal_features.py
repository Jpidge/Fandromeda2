"""Regression tests for Fandromeda's pregame feature contract."""

import unittest

import pandas as pd

from gptindex import (
    add_player_features,
    build_feature_history,
    prepare_prior_season_carryover,
)


class TemporalFeatureTests(unittest.TestCase):
    """A Week W forecast must include W-1 and exclude W."""

    @staticmethod
    def player_history(include_target_week: bool) -> pd.DataFrame:
        weeks = list(range(1, 7 if include_target_week else 6))
        return pd.DataFrame(
            {
                "player_id": ["test-player"] * len(weeks),
                "season": [2026] * len(weeks),
                "week": weeks,
                # Week 6 is intentionally extreme: if it leaks, this test
                # fails plainly rather than only by a tiny rounding amount.
                "fantasy_points_std": [
                    1.0, 2.0, 3.0, 4.0, 5.0, 999.0
                ][: len(weeks)],
                "targets": weeks,
                "carries": weeks,
                "receiving_yards": [10.0 * week for week in weeks],
                "rushing_yards": [5.0 * week for week in weeks],
                "receptions": weeks,
                "offense_snaps": [20.0 + week for week in weeks],
                "offense_pct": [40.0 + week for week in weeks],
                "td_points": [0.0] * len(weeks),
                "red_zone_targets": [0.0] * len(weeks),
                "red_zone_carries": [0.0] * len(weeks),
            }
        )

    def test_week_six_features_include_week_five_not_week_six(self):
        full_history = self.player_history(include_target_week=True)
        prior_only = self.player_history(include_target_week=False)

        full_features = add_player_features(full_history, target_week=6)
        prior_features = add_player_features(prior_only, target_week=6)

        self.assertEqual(len(full_features), 1)
        self.assertEqual(int(full_features.iloc[0]["week"]), 5)
        self.assertEqual(full_features.iloc[0]["fantasy_points_last"], 5.0)
        self.assertEqual(full_features.iloc[0]["fantasy_points_roll2"], 4.5)

        # The target-week value (999) cannot affect a Week 6 forecast.
        for column in [
            "fantasy_points_last",
            "fantasy_points_roll2",
            "fantasy_points_roll3",
            "fantasy_points_roll5",
            "fantasy_points_ewma",
        ]:
            self.assertAlmostEqual(
                full_features.iloc[0][column],
                prior_features.iloc[0][column],
            )

    def test_prior_season_carryover_is_shared_pregame_history(self):
        prior = self.player_history(include_target_week=True)
        prior["season"] = 2025
        prior["week"] = [13, 14, 15, 16, 17, 18]
        prior["fantasy_points_std"] = [1, 2, 3, 4, 5, 18]

        carryover = prepare_prior_season_carryover(prior, 2026)
        self.assertEqual(carryover["week"].tolist(), [-3, -2, -1])
        self.assertEqual(carryover["source_week"].tolist(), [16, 17, 18])
        self.assertEqual(carryover["season"].tolist(), [2026, 2026, 2026])

        current = self.player_history(include_target_week=False).iloc[0:0]
        current["season"] = 2026
        historical = build_feature_history(
            current,
            carryover,
            target_week=1,
        )
        features = add_player_features(historical, target_week=1)

        # A Week 1 forecast sees the player's final prior-season game.
        self.assertEqual(features.iloc[0]["fantasy_points_last"], 18.0)


if __name__ == "__main__":
    unittest.main()
