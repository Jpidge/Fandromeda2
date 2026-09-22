"""Kickoff-safe Yahoo/FANDROMEDA comparison records."""

from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import uuid

import pandas as pd


BENCHMARK_SCHEMA_VERSION = "benchmark_v1"
_EASTERN = ZoneInfo("America/New_York")


def schedule_kickoffs(schedule: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    """Return one UTC kickoff per team for a regular-season schedule."""
    required = {"season", "week", "game_type", "gameday", "gametime", "away_team", "home_team", "game_id"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule is missing columns: {sorted(missing)}")
    games = schedule[(schedule.season == season) & (schedule.week == week)
                     & schedule.game_type.astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
    rows = []
    for row in games.itertuples():
        day = pd.to_datetime(row.gameday, errors="coerce")
        if pd.isna(day):
            raise ValueError(f"Invalid schedule date: {row.gameday!r}")
        raw_time = str(row.gametime).strip()
        try:
            clock = datetime.strptime(raw_time, "%H:%M").time()
        except ValueError as exc:
            raise ValueError(f"Invalid schedule kickoff time: {raw_time!r}") from exc
        local = datetime.combine(day.date(), clock, tzinfo=_EASTERN)
        kickoff = local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        for team in (row.away_team, row.home_team):
            rows.append({"season": season, "week": week, "team": str(team).upper(),
                         "game_id": row.game_id, "kickoff_utc": kickoff})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No regular-season games found for {season} Week {week}.")
    if result.team.duplicated().any():
        raise ValueError("Schedule contains duplicate team kickoffs.")
    return result


def validate_pregame_captures(records: pd.DataFrame, kickoffs: pd.DataFrame) -> pd.DataFrame:
    """Annotate each projection and reject captures after that team's kickoff."""
    required = {"player_name", "team", "captured_at_utc", "yahoo_projection"}
    if not required.issubset(records.columns):
        raise ValueError(f"Capture records are missing columns: {sorted(required - set(records.columns))}")
    result = records.copy()
    result["team"] = result.team.astype(str).str.upper()
    result = result.merge(kickoffs[["team", "kickoff_utc"]], on="team", how="left", validate="many_to_one")
    result["capture_time"] = pd.to_datetime(result.captured_at_utc, utc=True, errors="coerce")
    result["kickoff_time"] = pd.to_datetime(result.kickoff_utc, utc=True, errors="coerce")
    result["kickoff_status"] = "missing_kickoff"
    known = result.capture_time.notna() & result.kickoff_time.notna()
    result.loc[known, "kickoff_status"] = "pregame"
    result.loc[known & (result.capture_time >= result.kickoff_time), "kickoff_status"] = "post_kickoff"
    if (result.kickoff_status == "post_kickoff").any():
        names = result.loc[result.kickoff_status == "post_kickoff", "player_name"].tolist()
        raise ValueError(f"Post-kickoff Yahoo captures cannot enter the benchmark: {names}")
    return result


def build_comparison_records(fandromeda: pd.DataFrame, yahoo: pd.DataFrame,
                             kickoffs: pd.DataFrame) -> pd.DataFrame:
    """Join same-player projections and preserve only kickoff-safe rows."""
    required_f = {"player_id", "player_name", "team", "final_projection", "captured_at_utc"}
    required_y = {"player_id", "player_name", "team", "yahoo_projection", "captured_at_utc"}
    if not required_f.issubset(fandromeda.columns) or not required_y.issubset(yahoo.columns):
        raise ValueError("Both forecasts need stable identity, projection, and capture time fields.")
    keys = ["player_id", "team", "season", "target_week"]
    if not {"season", "target_week"}.issubset(fandromeda.columns) or not {"season", "target_week"}.issubset(yahoo.columns):
        raise ValueError("Both forecasts require season and target_week identity.")
    for frame in (fandromeda, yahoo):
        if frame[keys].isna().any().any():
            raise ValueError("Forecast identity must not be missing.")
        if frame[keys].astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
            raise ValueError("Forecast identity must not be empty.")
        scope = kickoffs[["season", "week"]].drop_duplicates()
        if len(scope) != 1 or not ((frame.season == scope.iloc[0].season) & (frame.target_week == scope.iloc[0].week)).all():
            raise ValueError("Forecast season/week does not match kickoff schedule.")
    ours = validate_pregame_captures(
        fandromeda.rename(columns={"final_projection": "yahoo_projection"}), kickoffs,
    ).rename(columns={"yahoo_projection": "final_projection"})
    ours = ours.loc[ours.kickoff_status == "pregame", fandromeda.columns]
    safe = validate_pregame_captures(yahoo, kickoffs)
    safe = safe[safe.kickoff_status == "pregame"].copy()
    merged = ours.merge(safe, on=keys, suffixes=("_fandromeda", "_yahoo"), validate="one_to_one")
    merged["fandromeda_error"] = merged["actual_points"] - merged["final_projection"] if "actual_points" in merged else pd.NA
    merged["yahoo_error"] = merged["actual_points"] - merged["yahoo_projection"] if "actual_points" in merged else pd.NA
    merged["our_absolute_error"] = merged["fandromeda_error"].abs()
    merged["yahoo_absolute_error"] = merged["yahoo_error"].abs()
    merged["benchmark_schema_version"] = BENCHMARK_SCHEMA_VERSION
    return merged


def attach_actual_points(records: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Attach finalized canonical actuals by stable player/week identity."""
    required = {"player_id", "season", "target_week", "actual_points"}
    if not required.issubset(actuals.columns):
        raise ValueError(f"Actuals are missing columns: {sorted(required - set(actuals.columns))}")
    keys = ["player_id", "season", "target_week"]
    if records.duplicated(keys).any() or actuals.duplicated(keys).any():
        raise ValueError("Comparison records and actuals must have unique player-week identities.")
    result = records.drop(columns=["actual_points"], errors="ignore").merge(
        actuals[keys + ["actual_points"]], on=keys, how="left", validate="one_to_one"
    )
    result["actual_status"] = result.actual_points.notna().map({True: "finalized", False: "pending"})
    return result


def comparison_metrics(records: pd.DataFrame) -> pd.DataFrame:
    """Calculate measured errors only for finalized, kickoff-safe rows."""
    required = {"actual_points", "final_projection", "yahoo_projection", "kickoff_status"}
    if not required.issubset(records.columns):
        raise ValueError(f"Comparison records are missing columns: {sorted(required - set(records.columns))}")
    eligible = records[(records.kickoff_status == "pregame") & records.actual_points.notna()]
    eligible = eligible.copy()
    numeric = ["actual_points", "final_projection", "yahoo_projection"]
    eligible[numeric] = eligible[numeric].apply(pd.to_numeric, errors="coerce")
    eligible = eligible.replace([float("inf"), -float("inf")], float("nan")).dropna(subset=numeric)
    rows = []
    for label, column in [("FANDROMEDA", "final_projection"), ("Yahoo", "yahoo_projection")]:
        error = eligible.actual_points - eligible[column]
        absolute = error.abs()
        rows.append({"model": label, "examples": int(error.notna().sum()),
                     "mae": float(absolute.mean()), "rmse": float((error.pow(2).mean()) ** 0.5),
                     "bias": float(error.mean()), "median_absolute_error": float(absolute.median()),
                     "within_2_points": float((absolute <= 2).mean()),
                     "within_3_points": float((absolute <= 3).mean()),
                     "within_5_points": float((absolute <= 5).mean())})
    if not rows or rows[0]["examples"] == 0:
        return pd.DataFrame(columns=["model", "examples", "mae", "rmse", "bias", "median_absolute_error",
                                     "within_2_points", "within_3_points", "within_5_points"])
    return pd.DataFrame(rows)


def save_comparison_records(records: pd.DataFrame, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}_{uuid.uuid4().hex[:12]}.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    records.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path
