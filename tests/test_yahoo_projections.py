"""Safety checks for raw Yahoo numeric-projection capture validation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import hashlib
import pandas as pd

from fandromeda.yahoo_projections import (
    _is_yahoo_projection_text, parse_yahoo_matchup, reconcile_yahoo_projections,
    import_yahoo_projection_files,
)

FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_week3_matchup.txt"


class YahooProjectionParserTests(unittest.TestCase):
    def setUp(self):
        self.text = FIXTURE.read_text(encoding="utf-8")
        self.parsed = parse_yahoo_matchup(self.text, week=3)
        self.roster = self.parsed.rename(columns={"player_name": "player"}).copy()
        self.roster["nflverse_player_id"] = [f"test-{i}" for i in range(len(self.roster))]

    def test_actual_sample_both_sides_bench_ir_and_empty_slot(self):
        by_name = self.parsed.set_index("player_name")
        self.assertEqual(len(by_name), 31)
        for name, value in [("Justin Herbert", 18.11), ("Dak Prescott", 21.98),
                            ("Cowboys", 10.45), ("Chargers", 9), ("Jordan Love", 19.48),
                            ("A.J. Brown", 0.07), ("Puka Nacua", 11.32), ("Zay Flowers", 10.30)]:
            self.assertAlmostEqual(by_name.loc[name, "yahoo_projection"], value)
        self.assertEqual(by_name.loc["A.J. Brown", "slot"], "IR")
        self.assertEqual(self.parsed.slot.eq("BN").sum(), 12)
        totals = self.parsed.loc[self.parsed.table.eq("starters")].groupby("manager").yahoo_projection.sum()
        self.assertAlmostEqual(totals["Purple Reign"], 103.22)
        self.assertAlmostEqual(totals["Belichick Yo'self"], 97.50)

    def test_points_are_not_confused_with_actuals_and_missing_is_not_zero(self):
        text = self.text.replace("18.11\n–\nQB\n–\n21.98", "0.00\n88.50\nQB\n99.00\n–")
        parsed = parse_yahoo_matchup(text, week=3).set_index("player_name")
        self.assertEqual(parsed.loc["Justin Herbert", "yahoo_projection"], 0)
        self.assertTrue(pd.isna(parsed.loc["Dak Prescott", "yahoo_projection"]))

    def test_wrong_week_league_hidden_bench_and_broken_columns_fail(self):
        cases = [self.text.replace("Week 3:", "Week 2:"),
                 self.text.replace("ID# 893771", "ID# 123456"),
                 self.text.replace("Hide Bench Players", "Show Bench Players"),
                 self.text.replace("18.11\n–\nQB", "18.11\nQB"),
                 self.text.replace("18.11", "not-a-number", 1),
                 self.text.replace("Sun 12:00 pm @", "Final W 24-17 @", 1)]
        for text in cases:
            with self.subTest(text=text[:50]):
                with self.assertRaises(ValueError):
                    parse_yahoo_matchup(text, week=3)

    def test_reconciliation_reports_partial_coverage_and_preserves_zero(self):
        subset = self.parsed.iloc[:1].copy()
        subset["yahoo_projection"] = 0.0
        coverage, extra, summary = reconcile_yahoo_projections(subset, self.roster)
        self.assertEqual(summary["missing_entries"], 30)
        self.assertFalse(summary["complete_roster_coverage"])
        self.assertFalse(summary["benchmark_eligible"])
        self.assertEqual(coverage.loc[0, "yahoo_projection"], 0)
        self.assertTrue(coverage.loc[1:, "yahoo_projection"].isna().all())
        self.assertTrue(extra.empty)

    def test_full_coverage_and_exact_canonical_alias(self):
        self.roster["nflverse_name"] = self.roster.player
        self.roster.loc[0, "player"] = "Truncated Name"
        self.roster.loc[30, "player"] = "A.J. BrownIR"
        coverage, extra, summary = reconcile_yahoo_projections(self.parsed, self.roster)
        self.assertTrue(summary["complete_roster_coverage"])
        self.assertEqual(summary["numeric_projections"], 31)
        self.assertTrue(extra.empty)
        self.assertEqual(coverage.loc[coverage.position.eq("DEF"), "player_id"].tolist(), ["DEF:DAL", "DEF:LAC"])

    def test_duplicate_and_ambiguous_names_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate capture"):
            reconcile_yahoo_projections(pd.concat([self.parsed, self.parsed]), self.roster)
        with self.assertRaisesRegex(ValueError, "Ambiguous/duplicate"):
            reconcile_yahoo_projections(self.parsed, pd.concat([self.roster, self.roster.iloc[:1]]))

    def test_wrong_manager_is_an_explicit_coverage_gap(self):
        changed = self.parsed.copy()
        changed.loc[0, "manager"] = "Different Team"
        _, extra, summary = reconcile_yahoo_projections(changed, self.roster)
        self.assertEqual(summary["missing_entries"], 1)
        self.assertEqual(len(extra), 1)
        self.assertFalse(summary["complete_roster_coverage"])

    def test_imports_are_separate_and_do_not_invent_capture_timestamps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, summary = import_yahoo_projection_files([FIXTURE], self.roster, root, season=2026, week=3)
            before = {p.name: p.read_bytes() for p in first.iterdir()}
            second, _ = import_yahoo_projection_files([FIXTURE], self.roster, root, season=2026, week=3)
            self.assertNotEqual(first, second)
            self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
            self.assertEqual((first / "source_01.txt").read_bytes(), FIXTURE.read_bytes())
            self.assertTrue(pd.read_csv(first / "yahoo_projections.csv").captured_at_utc.isna().all())
            self.assertFalse(json.loads((first / "summary.json").read_text())["benchmark_eligible"])

    def test_capture_provenance_is_checked_and_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture.txt"
            source.write_bytes(FIXTURE.read_bytes())
            provenance = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "season": 2026, "week": 3, "captured_at_utc": "2026-09-20T12:00:00Z"}
            sidecar = source.with_suffix(".txt.json")
            sidecar.write_text(json.dumps(provenance))
            batch, summary = import_yahoo_projection_files([source], self.roster, root / "imports", season=2026, week=3)
            self.assertEqual(summary["timestamped_entries"], 31)
            self.assertTrue((batch / "source_01.txt.json").exists())
            self.assertFalse(summary["benchmark_eligible"])
            provenance["sha256"] = "wrong"
            sidecar.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "provenance"):
                import_yahoo_projection_files([source], self.roster, root / "imports", season=2026, week=3)


class YahooProjectionCaptureTests(unittest.TestCase):
    def test_accepts_a_matchup_table_with_projection_headers_and_players(self):
        rows = "\n".join(
            [
                f"Example Player {number}\nKC - RB\n{10 + number / 10:.1f}"
                for number in range(8)
            ]
        )
        text = f"Stats\nPlayer\nProj\nFan Pts\n{rows}"
        self.assertTrue(_is_yahoo_projection_text(text))

    def test_rejects_the_starting_roster_page_without_numeric_headers(self):
        rows = "\n".join(
            [f"Example Player {number}\nKC - RB" for number in range(12)]
        )
        self.assertFalse(_is_yahoo_projection_text(f"Pos\nPlayer\n{rows}"))

    def test_rejects_an_ad_or_navigation_shell(self):
        self.assertFalse(
            _is_yahoo_projection_text("Yahoo Sports\nPlayers\nProj\nFan Pts")
        )


if __name__ == "__main__":
    unittest.main()
