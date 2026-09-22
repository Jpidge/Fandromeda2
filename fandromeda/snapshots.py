"""Append-only records of the forecasts FANDROMEDA actually published.

The live ``data/output/projections.csv`` file is deliberately replaceable: it
always represents the most recent run.  This module preserves a separate,
immutable copy per run so later evaluation can compare a forecast made before
kickoff with the actual result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import pandas as pd


SNAPSHOT_SCHEMA_VERSION = "prediction_snapshot_v2"
SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "captured_at_utc",
    "schema_version",
    "app_version",
    "feature_version",
    "scoring_version",
    "primary_projection_model",
    "season",
    "target_week",
    "player_id",
    "player_name",
    "position",
    "team",
    "snapshot_scope",
    "fantasy_manager",
    "roster_slot",
    "forecast_status",
    "baseline_projection",
    "learned_weight_projection",
    "ml_projection",
    "final_projection",
    "range_low",
    "range_high",
    "confidence",
    "classification",
    "depth_role",
    "depth_adjustment",
    "yahoo_projection",
    "actual_points",
    "actual_finalized_at_utc",
]
MANIFEST_COLUMNS = [
    "snapshot_id",
    "captured_at_utc",
    "schema_version",
    "season",
    "target_week",
    "row_count",
    "snapshot_path",
    "app_version",
    "feature_version",
    "scoring_version",
    "primary_projection_model",
]


def _column(frame: pd.DataFrame, name: str, default: object = pd.NA) -> pd.Series:
    """Return a column aligned to ``frame`` without inventing source values."""

    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _require_columns(projections: pd.DataFrame) -> None:
    required = {"player_id", "player_name", "position", "team", "projection"}
    missing = sorted(required.difference(projections.columns))
    if missing:
        raise ValueError(
            "Cannot create a prediction snapshot; projection data is missing "
            f"required columns: {', '.join(missing)}"
        )


def build_league_snapshot_input(
    roster: pd.DataFrame, projections: pd.DataFrame,
) -> pd.DataFrame:
    """Retain every league roster entry, never the full NFL forecast pool.

    The caller supplies normalized NFL team abbreviations. Defenses receive
    team-unit IDs; unresolved players remain unresolved rather than matching
    another missing ID. Unavailable forecasts stay null, including D/ST.
    """
    required = {"manager", "slot", "player", "position", "team",
                "nflverse_player_id"}
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Snapshot roster is missing columns: {sorted(missing)}")
    _require_columns(projections)
    league = roster.reset_index(drop=True).copy()
    league["player_id"] = league["nflverse_player_id"].fillna("").astype(str).str.strip()
    defense = league["position"].eq("DEF")
    teams = league["team"].fillna("").astype(str).str.strip()
    league.loc[defense & teams.ne(""), "player_id"] = "DEF:" + teams[defense & teams.ne("")]
    league = league.rename(columns={"player": "player_name", "manager": "fantasy_manager",
                                    "slot": "roster_slot"})
    values = projections.copy()
    values["player_id"] = values["player_id"].fillna("").astype(str).str.strip()
    values = values.loc[values["player_id"].ne("")]
    # Roster identity and membership are authoritative at capture time.
    values = values.drop(columns=[c for c in values.columns
                                 if c in league.columns and c != "player_id"])
    league = league.merge(values, on="player_id", how="left", validate="many_to_one")
    league["snapshot_scope"] = "league_roster"
    league["forecast_status"] = "available"
    league.loc[pd.to_numeric(league["projection"], errors="coerce").isna(),
               "forecast_status"] = "forecast_unavailable"
    league.loc[league["player_id"].eq(""), "forecast_status"] = "identity_unresolved"
    return league


def build_snapshot_frame(
    projections: pd.DataFrame,
    *,
    snapshot_id: str,
    captured_at_utc: str,
    season: int,
    target_week: int,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Build the stable, evaluation-ready schema for one published run."""

    _require_columns(projections)
    frame = pd.DataFrame(index=projections.index)
    frame["snapshot_id"] = snapshot_id
    frame["captured_at_utc"] = captured_at_utc
    frame["schema_version"] = SNAPSHOT_SCHEMA_VERSION
    frame["app_version"] = str(metadata.get("app_version", "unknown"))
    frame["feature_version"] = str(metadata.get("feature_version", "unknown"))
    frame["scoring_version"] = str(metadata.get("scoring_version", "unknown"))
    frame["primary_projection_model"] = str(
        metadata.get("primary_projection_model", "unknown")
    )
    frame["season"] = int(season)
    frame["target_week"] = int(target_week)

    for column in ["player_id", "player_name", "position", "team", "classification",
                   "depth_role", "depth_adjustment", "fantasy_manager", "roster_slot",
                   "forecast_status"]:
        frame[column] = _column(projections, column, "").fillna("").astype(str)
    frame["snapshot_scope"] = _column(projections, "snapshot_scope", "nfl_projection_pool")

    for source, destination in [
        ("baseline_projection", "baseline_projection"),
        ("learned_weight_projection", "learned_weight_projection"),
        ("ml_projection", "ml_projection"),
        ("projection", "final_projection"),
        ("range_low", "range_low"),
        ("range_high", "range_high"),
        ("confidence", "confidence"),
    ]:
        frame[destination] = pd.to_numeric(
            _column(projections, source), errors="coerce"
        )

    # These arrive only in later evaluation milestones.  Their absence is an
    # intentional null, never a made-up zero or a value copied from hindsight.
    frame["yahoo_projection"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    frame["actual_points"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    frame["actual_finalized_at_utc"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[SNAPSHOT_COLUMNS].reset_index(drop=True)


def save_prediction_snapshot(
    projections: pd.DataFrame,
    snapshot_root: Path,
    *,
    season: int,
    target_week: int,
    metadata: Mapping[str, object],
) -> tuple[Path, Path, str]:
    """Persist one immutable Parquet snapshot and append its manifest record.

    A fresh UUID is part of every file name, so two runs in the same second
    cannot overwrite each other.  Existing snapshot files are never opened
    for writing.
    """

    captured = datetime.now(timezone.utc)
    captured_at_utc = captured.isoformat(timespec="seconds").replace("+00:00", "Z")
    snapshot_id = f"{captured.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}"
    predictions_dir = snapshot_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = predictions_dir / f"{snapshot_id}.parquet"
    manifest_path = snapshot_root / "snapshot_index.csv"

    frame = build_snapshot_frame(
        projections,
        snapshot_id=snapshot_id,
        captured_at_utc=captured_at_utc,
        season=season,
        target_week=target_week,
        metadata=metadata,
    )
    # Write then atomically publish the completed file; this never touches an
    # earlier snapshot if a run stops midway through writing.
    temporary_path = snapshot_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(snapshot_path)

    manifest_row = pd.DataFrame([{
        "snapshot_id": snapshot_id,
        "captured_at_utc": captured_at_utc,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "season": int(season),
        "target_week": int(target_week),
        "row_count": int(len(frame)),
        "snapshot_path": snapshot_path.relative_to(snapshot_root).as_posix(),
        "app_version": str(metadata.get("app_version", "unknown")),
        "feature_version": str(metadata.get("feature_version", "unknown")),
        "scoring_version": str(metadata.get("scoring_version", "unknown")),
        "primary_projection_model": str(
            metadata.get("primary_projection_model", "unknown")
        ),
    }], columns=MANIFEST_COLUMNS)
    manifest_row.to_csv(
        manifest_path,
        mode="a",
        header=not manifest_path.exists(),
        index=False,
        lineterminator="\n",
    )
    return snapshot_path, manifest_path, snapshot_id


def load_snapshot_index(snapshot_root: Path) -> pd.DataFrame:
    """Read the human-inspectable snapshot manifest without loading Parquet."""

    manifest_path = snapshot_root / "snapshot_index.csv"
    if not manifest_path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(manifest_path)
