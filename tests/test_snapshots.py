"""Regression tests for append-only prediction snapshots."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from fandromeda.snapshots import (
    SNAPSHOT_COLUMNS,
    build_league_snapshot_input,
    load_snapshot_index,
    save_prediction_snapshot,
)


def example_projections() -> pd.DataFrame:
    return pd.DataFrame([{
        "player_id": "00-0034857",
        "player_name": "Patrick Mahomes",
        "position": "QB",
        "team": "KC",
        "baseline_projection": 20.0,
        "learned_weight_projection": 21.0,
        "ml_projection": 20.5,
        "projection": 21.0,
        "range_low": 16.0,
        "range_high": 26.0,
        "confidence": 61.0,
        "classification": "Stable",
        "depth_role": "starter",
        "depth_adjustment": "",
    }])


class PredictionSnapshotTests(unittest.TestCase):
    def test_new_schema_appends_without_rewriting_legacy_index_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, _ = save_prediction_snapshot(
                example_projections(), root, season=2026, target_week=3, metadata={},
            )
            legacy = manifest.read_bytes().replace(b"prediction_snapshot_v2", b"prediction_snapshot_v1")
            manifest.write_bytes(legacy)
            save_prediction_snapshot(
                example_projections(), root, season=2026, target_week=3, metadata={},
            )
            self.assertTrue(manifest.read_bytes().startswith(legacy))
            self.assertEqual(load_snapshot_index(root)["schema_version"].tolist(),
                             ["prediction_snapshot_v1", "prediction_snapshot_v2"])

    def test_league_scope_excludes_free_agents_and_retains_roster_context(self):
        roster = pd.DataFrame([
            {"manager": "Team A", "slot": "QB", "player": "Patrick Mahomes",
             "position": "QB", "team": "KC", "nflverse_player_id": "00-0034857"},
            {"manager": "Team A", "slot": "DEF", "player": "Patriots",
             "position": "DEF", "team": "NE", "nflverse_player_id": None},
            {"manager": "Team B", "slot": "BN", "player": "Unresolved Player",
             "position": "RB", "team": "BUF", "nflverse_player_id": None},
        ])
        free_agent = example_projections().assign(player_id="free-agent", projection=30)
        # Blank IDs must never match unresolved roster entries.
        unknown = example_projections().assign(player_id="", projection=99)
        pool = pd.concat([example_projections(), free_agent, unknown], ignore_index=True)
        result = build_league_snapshot_input(roster, pool)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["player_id"].tolist(), ["00-0034857", "DEF:NE", ""])
        self.assertEqual(result["forecast_status"].tolist(),
                         ["available", "forecast_unavailable", "identity_unresolved"])
        self.assertEqual(result.loc[0, "projection"], 21)
        self.assertTrue(result.loc[1:, "projection"].isna().all())
        with TemporaryDirectory() as directory:
            path, _, _ = save_prediction_snapshot(
                result, Path(directory), season=2026, target_week=3, metadata={},
            )
            saved = pd.read_parquet(path)
            self.assertEqual(saved["fantasy_manager"].tolist(), ["Team A", "Team A", "Team B"])
            self.assertEqual(saved["roster_slot"].tolist(), ["QB", "DEF", "BN"])
            self.assertTrue(saved["snapshot_scope"].eq("league_roster").all())

    def test_duplicate_forecast_ids_cannot_multiply_roster_entries(self):
        roster = pd.DataFrame([{
            "manager": "Team A", "slot": "QB", "player": "Patrick Mahomes",
            "position": "QB", "team": "KC", "nflverse_player_id": "00-0034857",
        }])
        pool = pd.concat([example_projections(), example_projections()], ignore_index=True)
        with self.assertRaises(pd.errors.MergeError):
            build_league_snapshot_input(roster, pool)

    def test_each_save_creates_an_immutable_file_and_appends_the_manifest(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "snapshots"
            metadata = {
                "app_version": "test",
                "feature_version": "test_features",
                "scoring_version": "test_scoring",
            }
            first_path, manifest_path, first_id = save_prediction_snapshot(
                example_projections(), root, season=2026, target_week=3, metadata=metadata
            )
            first_bytes = first_path.read_bytes()
            second_path, _, second_id = save_prediction_snapshot(
                example_projections(), root, season=2026, target_week=3, metadata=metadata
            )

            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_bytes, first_path.read_bytes())
            self.assertTrue(manifest_path.exists())
            index = load_snapshot_index(root)
            self.assertEqual(index["snapshot_id"].tolist(), [first_id, second_id])
            self.assertEqual(index["row_count"].tolist(), [1, 1])

    def test_snapshot_keeps_only_pregame_values_and_empty_future_outcomes(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path, _, _ = save_prediction_snapshot(
                example_projections(), root, season=2026, target_week=3,
                metadata={"app_version": "test", "feature_version": "test", "scoring_version": "test"},
            )
            saved = pd.read_parquet(path)
            self.assertEqual(saved.columns.tolist(), SNAPSHOT_COLUMNS)
            self.assertEqual(saved.loc[0, "final_projection"], 21.0)
            self.assertTrue(pd.isna(saved.loc[0, "yahoo_projection"]))
            self.assertTrue(pd.isna(saved.loc[0, "actual_points"]))

    def test_snapshot_rejects_a_projection_without_player_identity(self):
        with TemporaryDirectory() as temporary_directory:
            without_identity = example_projections().drop(columns=["player_id"])
            with self.assertRaisesRegex(ValueError, "player_id"):
                save_prediction_snapshot(
                    without_identity, Path(temporary_directory), season=2026,
                    target_week=3, metadata={},
                )


if __name__ == "__main__":
    unittest.main()
