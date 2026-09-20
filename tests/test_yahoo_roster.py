"""Safety tests for local Yahoo roster capture persistence."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fandromeda.yahoo_roster import save_roster_capture
from gptindex import build_player_match_table, normalize_name, parse_yahoo_roster


class YahooRosterCaptureTests(unittest.TestCase):
    def test_capture_archives_and_backs_up_existing_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster_path = root / "data" / "my_roster.txt"
            exports = root / "data" / "yahoo_exports"
            roster_path.parent.mkdir(parents=True)
            roster_path.write_text("old roster", encoding="utf-8")
            visible_text = "\n".join(
                [
                    "Example Team \ue001",
                    "Pos",
                    "Player",
                    "QB",
                    "Example Quarterback",
                    "TB - QB",
                    "RB",
                    "Example Running Back",
                    "KC - RB",
                    "WR",
                    "Example Receiver",
                    "BUF - WR",
                    "TE",
                    "Example Tight End",
                    "BAL - TE",
                    "K",
                    "Example Kicker",
                    "GB - K",
                    "DEF",
                    "Example Defense",
                    "NE - DEF",
                    "BN",
                    "Example Bench One",
                    "LAR - RB",
                    "BN",
                    "Example Bench Two",
                    "SF - WR",
                ]
            )

            captured, backup = save_roster_capture(
                visible_text,
                roster_path,
                exports,
            )

            self.assertTrue(captured.exists())
            self.assertIsNotNone(backup)
            self.assertEqual(backup.read_text(encoding="utf-8"), "old roster")
            self.assertEqual(
                roster_path.read_text(encoding="utf-8"),
                visible_text + "\n",
            )

    def test_capture_refuses_suspiciously_short_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "too little"):
                save_roster_capture("not a roster", root / "roster.txt", root)

    def test_capture_refuses_advertisement_text_without_replacing_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster_path = root / "roster.txt"
            roster_path.write_text("known good roster", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "did not resemble"):
                save_roster_capture(
                    "Sponsored Ad\nShop on Amazon\n" * 20,
                    roster_path,
                    root / "exports",
                )

            self.assertEqual(roster_path.read_text(encoding="utf-8"), "known good roster")

    def test_whole_page_copy_strips_yahoo_name_labels_before_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            roster_path = Path(directory) / "roster.txt"
            roster_path.write_text(
                "\n".join(
                    [
                        "Example Team \ue001",
                        "Pos\tPlayer",
                        "QB",
                        "Baker MayfieldVideo ForecastPlayer Note",
                        "TB - QB",
                        "TE",
                        "Brock BowersDPlayer Note",
                        "LV - TE",
                    ]
                ),
                encoding="utf-8",
            )

            roster = parse_yahoo_roster(roster_path)

            self.assertEqual(roster["player"].tolist(), ["Baker Mayfield", "Brock BowersD"])
            self.assertEqual(roster["position"].tolist(), ["QB", "TE"])

    def test_status_suffixes_match_the_canonical_player_name(self):
        rows = [
            ("Michael Pittman Jr.O", "PIT", "WR", "Michael Pittman Jr."),
            ("Josh JacobsCEL", "GB", "RB", "J. Jacobs"),
            ("Jordyn TysonIR-R", "NO", "WR", "Jordyn Tyson"),
            ("Zach CharbonnetPUP-R", "SEA", "RB", "Zach Charbonnet"),
            ("Tyler LoopP", "BAL", "K", "Tyler Loop"),
        ]
        yahoo = pd.DataFrame(
            [
                {
                    "player": copied_name,
                    "team": team,
                    "position": position,
                    "slot": "BN",
                    "player_normalized": normalize_name(copied_name),
                }
                for copied_name, team, position, _ in rows
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "player_id": f"player-{index}",
                    "display_name": canonical_name,
                    "team": team,
                    "position": position,
                    "name_normalized": normalize_name(canonical_name),
                }
                for index, (_, team, position, canonical_name) in enumerate(rows)
            ]
        )

        matched = build_player_match_table(yahoo, players)

        self.assertEqual((matched["nflverse_player_id"] != "").sum(), len(rows))
        self.assertEqual(matched["player"].tolist(), [row[3] for row in rows])
        self.assertNotIn("unmatched", matched["match_method"].tolist())


if __name__ == "__main__":
    unittest.main()
