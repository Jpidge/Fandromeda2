import unittest
import pandas as pd

from fandromeda.benchmarks import schedule_kickoffs, validate_pregame_captures, attach_actual_points, comparison_metrics


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.schedule = pd.DataFrame([{
            "season": 2026, "week": 3, "game_type": "REG", "gameday": "2026-09-20",
            "gametime": "13:00", "away_team": "KC", "home_team": "BUF", "game_id": "g1",
        }])

    def test_schedule_expands_each_game_to_team_kickoffs(self):
        result = schedule_kickoffs(self.schedule, season=2026, week=3)
        self.assertEqual(result.team.tolist(), ["KC", "BUF"])
        self.assertEqual(result.kickoff_utc.iloc[0], "2026-09-20T17:00:00Z")

    def test_pregame_is_allowed_and_post_kickoff_is_rejected(self):
        records = pd.DataFrame([{
            "player_name": "Example", "team": "KC", "captured_at_utc": "2026-09-20T16:59:00Z",
            "yahoo_projection": 10.0,
        }])
        self.assertEqual(validate_pregame_captures(records, schedule_kickoffs(self.schedule, season=2026, week=3)).kickoff_status.iloc[0], "pregame")
        records.loc[0, "captured_at_utc"] = "2026-09-20T17:00:00Z"
        with self.assertRaisesRegex(ValueError, "Post-kickoff"):
            validate_pregame_captures(records, schedule_kickoffs(self.schedule, season=2026, week=3))

    def test_missing_capture_time_is_not_benchmark_eligible(self):
        records = pd.DataFrame([{"player_name": "Example", "team": "KC", "captured_at_utc": None, "yahoo_projection": 10.0}])
        result = validate_pregame_captures(records, schedule_kickoffs(self.schedule, season=2026, week=3))
        self.assertEqual(result.kickoff_status.iloc[0], "missing_kickoff")

    def test_actual_join_and_metrics_use_only_finalized_pregame_rows(self):
        kickoffs = schedule_kickoffs(self.schedule, season=2026, week=3)
        records = pd.DataFrame([
            {"player_id": "p1", "season": 2026, "target_week": 3, "team": "KC", "player_name": "A", "captured_at_utc": "2026-09-20T16:00:00Z", "final_projection": 10.0, "yahoo_projection": 12.0},
            {"player_id": "p2", "season": 2026, "target_week": 3, "team": "BUF", "player_name": "B", "captured_at_utc": "2026-09-20T16:00:00Z", "final_projection": 20.0, "yahoo_projection": 18.0},
        ])
        safe = validate_pregame_captures(records, kickoffs)
        actuals = pd.DataFrame({"player_id": ["p1"], "season": [2026], "target_week": [3], "actual_points": [11.0]})
        joined = attach_actual_points(safe, actuals)
        metrics = comparison_metrics(joined)
        self.assertEqual(metrics.examples.tolist(), [1, 1])
        self.assertEqual(metrics.mae.tolist(), [1.0, 1.0])
        self.assertEqual(joined.actual_status.tolist(), ["finalized", "pending"])

    def test_actual_and_record_duplicates_are_rejected(self):
        base = pd.DataFrame({"player_id": ["p1"], "season": [2026], "target_week": [3], "actual_points": [1]})
        with self.assertRaises(ValueError):
            attach_actual_points(pd.concat([base.assign(final_projection=1), base.assign(final_projection=2)]), base)
        with self.assertRaises(ValueError):
            attach_actual_points(base.assign(final_projection=1), pd.concat([base, base]))


if __name__ == "__main__":
    unittest.main()
