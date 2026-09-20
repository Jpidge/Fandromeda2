"""Play-by-play event reconstruction for the canonical scoring engine.

The output is intentionally a compact player-week table rather than an
alternate fantasy-point calculation.  ``fandromeda.scoring`` remains the only
place that converts event counts into league points.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


EVENT_COLUMNS = (
    "passing_tds_40_plus",
    "rushing_tds_40_plus",
    "receiving_tds_40_plus",
    "return_yards",
    "return_tds",
    "offensive_fumble_return_tds",
    "extra_points_made",
    "field_goal_yards_made",
)


def available_event_fields(pbp: pd.DataFrame) -> dict[str, bool]:
    """Report which scoring events the supplied PBP schema can reconstruct."""

    has_returner = any(
        column in pbp.columns
        for column in (
            "returner_player_id",
            "kickoff_returner_player_id",
            "punt_returner_player_id",
        )
    )
    return {
        "passing_tds_40_plus": {
            "touchdown", "yards_gained", "pass_touchdown", "passer_player_id"
        }.issubset(pbp.columns),
        "rushing_tds_40_plus": {
            "touchdown", "yards_gained", "rush_touchdown", "rusher_player_id"
        }.issubset(pbp.columns),
        "receiving_tds_40_plus": {
            "touchdown", "yards_gained", "pass_touchdown", "receiver_player_id"
        }.issubset(pbp.columns),
        "return_yards": has_returner and "return_yards" in pbp.columns,
        "return_tds": has_returner and "return_touchdown" in pbp.columns,
        "offensive_fumble_return_tds": {
            "fumble_recovery_1_player_id", "fumble_return_touchdown"
        }.issubset(pbp.columns),
        "extra_points_made": {
            "kicker_player_id", "extra_point_result"
        }.issubset(pbp.columns),
        "field_goal_yards_made": {
            "kicker_player_id", "field_goal_result", "kick_distance"
        }.issubset(pbp.columns),
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _id_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _actor_events(
    frame: pd.DataFrame,
    actor_columns: Iterable[str],
    values: pd.Series,
    event_column: str,
) -> pd.DataFrame:
    """Create event rows only when the actor ID and event value are known."""

    actor_column = _id_column(frame, actor_columns)
    if actor_column is None:
        return pd.DataFrame(columns=["season", "week", "player_id", event_column])

    result = frame[["season", "week", actor_column]].copy()
    result = result.rename(columns={actor_column: "player_id"})
    result[event_column] = values
    result["player_id"] = result["player_id"].astype("string").str.strip()
    result = result[
        result["player_id"].notna()
        & result["player_id"].ne("")
        & result["player_id"].ne("<NA>")
        & result[event_column].ne(0)
    ]
    return result[["season", "week", "player_id", event_column]]


def summarize_pbp_scoring_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """Build player-week event fields unavailable in weekly player totals.

    The function accepts nflverse/nflfastR-shaped play-by-play data. It keeps
    regular-season plays only when ``season_type`` is provided. Any event
    whose required actor field is absent is *not* inferred; callers can audit
    the available columns before using the resulting adjustments in scoring.
    """

    required = {"season", "week"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(
            "Play-by-play data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = pbp.copy()
    if "season_type" in frame.columns:
        regular = frame["season_type"].astype("string").str.upper()
        frame = frame[regular.isin({"REG", "REGULAR"})].copy()

    for column in ("season", "week"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["season", "week"])

    long_touchdown = (
        _numeric(frame, "touchdown").eq(1)
        & _numeric(frame, "yards_gained").ge(40)
    ).astype(float)
    events = [
        _actor_events(
            frame,
            ("passer_player_id",),
            long_touchdown * _numeric(frame, "pass_touchdown"),
            "passing_tds_40_plus",
        ),
        _actor_events(
            frame,
            ("rusher_player_id",),
            long_touchdown * _numeric(frame, "rush_touchdown"),
            "rushing_tds_40_plus",
        ),
        _actor_events(
            frame,
            ("receiver_player_id",),
            long_touchdown * _numeric(frame, "pass_touchdown"),
            "receiving_tds_40_plus",
        ),
        _actor_events(
            frame,
            ("returner_player_id", "kickoff_returner_player_id", "punt_returner_player_id"),
            _numeric(frame, "return_yards"),
            "return_yards",
        ),
        _actor_events(
            frame,
            ("returner_player_id", "kickoff_returner_player_id", "punt_returner_player_id"),
            _numeric(frame, "return_touchdown"),
            "return_tds",
        ),
        _actor_events(
            frame,
            ("fumble_recovery_1_player_id",),
            _numeric(frame, "fumble_return_touchdown"),
            "offensive_fumble_return_tds",
        ),
        _actor_events(
            frame,
            ("kicker_player_id",),
            frame.get("extra_point_result", pd.Series("", index=frame.index))
            .astype("string")
            .str.lower()
            .eq("good")
            .astype(float),
            "extra_points_made",
        ),
        _actor_events(
            frame,
            ("kicker_player_id",),
            _numeric(frame, "kick_distance")
            * frame.get("field_goal_result", pd.Series("", index=frame.index))
            .astype("string")
            .str.lower()
            .eq("made")
            .astype(float),
            "field_goal_yards_made",
        ),
    ]
    combined = pd.concat(events, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["season", "week", "player_id", *EVENT_COLUMNS])

    result = combined.groupby(
        ["season", "week", "player_id"], as_index=False
    ).sum(numeric_only=True)
    for column in EVENT_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
    return result[["season", "week", "player_id", *EVENT_COLUMNS]].sort_values(
        ["season", "week", "player_id"]
    ).reset_index(drop=True)
