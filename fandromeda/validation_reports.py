"""Retain completed experiments and review them without training again."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


def save_validation_report(report, output, train_seasons, holdout_season,
                           source="computed", precision="full"):
    archive = Path(output) / "validation_runs"
    archive.mkdir(parents=True, exist_ok=True)
    saved = report.copy()
    saved["holdout_season"] = holdout_season
    saved["training_seasons"] = ",".join(map(str, train_seasons))
    saved["experiment"] = "defensive_matchup_opponent_v2"
    saved["source"] = source
    saved["precision"] = precision
    saved["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = archive / f"holdout_{holdout_season}_{uuid4().hex}.csv"
    saved.to_csv(path, index=False, mode="x")
    return path


def load_validation_reports(output):
    paths = sorted((Path(output) / "validation_runs").glob("*.csv"))
    if not paths:
        return pd.DataFrame()
    reports = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    reports["review_status"] = reports["experiment"].map(
        lambda name: "INVALID matchup evidence: own-team join" if name == "defensive_matchup_candidate"
        else "Research only; not approved for production"
    )
    return reports
