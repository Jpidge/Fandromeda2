"""Leakage-safe opponent matchup features for research experiments."""

from __future__ import annotations

import pandas as pd


POSITION_GROUPS = ("QB", "RB", "WR", "TE")


def build_defensive_history(player_stats: pd.DataFrame, *, target_week: int) -> pd.DataFrame:
    """Aggregate what each defense allowed before ``target_week``.

    The source is offensive player output keyed by ``opponent_team``. This
    avoids inventing defensive totals and makes the temporal boundary clear:
    target-week games are excluded before aggregation.
    """
    frame = player_stats.copy()
    if "opponent_team" not in frame.columns and "opponent" in frame.columns:
        frame = frame.rename(columns={"opponent": "opponent_team"})
    if "position_group" not in frame.columns and "position" in frame.columns:
        frame["position_group"] = frame["position"]
    if "fantasy_points_std" in frame.columns:
        frame["fantasy_points"] = frame["fantasy_points_std"]
    required = {"season", "week", "opponent_team", "position_group", "fantasy_points"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Player stats are missing columns: {sorted(missing)}")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame[frame.week < target_week].copy()
    frame["position_group"] = frame.position_group.astype(str).str.upper()
    frame = frame[frame.position_group.isin(POSITION_GROUPS)]
    frame["fantasy_points"] = pd.to_numeric(frame.fantasy_points, errors="coerce").fillna(0.0)
    grouped = frame.groupby(["season", "opponent_team", "position_group"], as_index=False).agg(
        games=("week", "nunique"), points_allowed=("fantasy_points", "sum"),
    )
    grouped["points_allowed_per_game"] = grouped["points_allowed"] / grouped["games"].clip(lower=1)
    return grouped.rename(columns={"opponent_team": "defense_team"})


def attach_defensive_history(features: pd.DataFrame, defense_history: pd.DataFrame,
                             *, target_week: int) -> pd.DataFrame:
    """Attach prior defensive points allowed by the player's position."""
    required = {"season", "opponent", "position_group"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature rows are missing columns: {sorted(missing)}")
    result = features.copy()
    result["position_group"] = result.position_group.astype(str).str.upper()
    defense = defense_history.copy()
    if "week" in defense.columns and (defense.week >= target_week).any():
        raise ValueError("Defensive history contains target-week or future games.")
    result = result.merge(
        defense[["season", "defense_team", "position_group", "points_allowed_per_game"]],
        left_on=["season", "opponent", "position_group"],
        right_on=["season", "defense_team", "position_group"], how="left", validate="many_to_one",
    ).drop(columns=["defense_team"])
    return result.rename(columns={"points_allowed_per_game": "opponent_points_allowed_per_game"})
