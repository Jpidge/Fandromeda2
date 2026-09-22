import tempfile
import unittest
from pathlib import Path
import pandas as pd
from fandromeda.validation_reports import save_validation_report, load_validation_reports


class ValidationReportTests(unittest.TestCase):
    def test_repeat_runs_preserve_results_and_fold_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            report = pd.DataFrame([{"model": "example", "mae": 4.0}])
            first = save_validation_report(report, directory, [2022, 2023], 2024)
            original = first.read_bytes()
            second = save_validation_report(report, directory, [2022, 2023], 2024)
            self.assertNotEqual(first, second)
            self.assertEqual(original, first.read_bytes())
            saved = load_validation_reports(directory)
            self.assertEqual(len(saved), 2)
            self.assertEqual(set(saved.holdout_season), {2024})
            self.assertEqual(set(saved.training_seasons), {"2022,2023"})
            self.assertNotIn("holdout_season", report)

    def test_empty_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(load_validation_reports(Path(directory)).empty)
