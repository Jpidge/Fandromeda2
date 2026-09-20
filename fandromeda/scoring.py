"""Canonical Fantasy Football Coalition scoring rules.

This module deliberately separates league rules from data reconstruction.
The functions can score every configured rule when supplied the necessary
stat-line fields. Current weekly NFLverse player data supplies the core
offensive fields; play-by-play enrichment in the next phase will supply the
long-touchdown, return, and other event-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


SCORING_VERSION = "ffc_yahoo_v1"


@dataclass(frozen=True)
class LeagueScoring:
    """Authoritative Yahoo rules for Fantasy Football Coalition."""

    passing_yard: float = 1.0 / 25.0
    passing_touchdown: float = 6.0
    interception: float = -2.0
    rushing_yard: float = 1.0 / 10.0
    rushing_touchdown: float = 6.0
    reception: float = 0.0
    receiving_yard: float = 1.0 / 10.0
    receiving_touchdown: float = 6.0
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0
    long_touchdown_bonus: float = 1.0
    return_yard: float = 1.0 / 25.0
    return_touchdown: float = 6.0
    offensive_fumble_return_touchdown: float = 6.0
    extra_point_made: float = 1.0
    field_goal_yard: float = 1.0 / 10.0
    defense_sack: float = 1.0
    defense_interception: float = 2.0
    defense_fumble_recovery: float = 2.0
    defense_touchdown: float = 6.0
    defense_safety: float = 2.0
    defense_blocked_kick: float = 2.0
    defense_extra_point_return: float = 2.0


FFC_SCORING = LeagueScoring()


def _number(stat_line: Mapping[str, object], field: str) -> float:
    """Return a numeric event count, treating omitted optional events as zero."""

    value = stat_line.get(field, 0.0)
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _series(frame: pd.DataFrame, field: str) -> pd.Series:
    """Return a numeric event series aligned to ``frame``.

    Event-specific fields are optional until play-by-play reconstruction is
    connected. Their absence means the weekly source cannot yet contribute
    that rule, rather than inventing an observed event.
    """

    if field not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").fillna(0.0)


def offensive_breakdown(
    stat_line: Mapping[str, object],
    rules: LeagueScoring = FFC_SCORING,
) -> dict[str, float]:
    """Score one offensive/K stat line using every configured rule.

    Extra event fields are ``passing_tds_40_plus``, ``rushing_tds_40_plus``,
    ``receiving_tds_40_plus``, ``return_yards``, ``return_tds``,
    ``offensive_fumble_return_tds``, ``extra_points_made``, and
    ``field_goal_yards_made``.
    """

    touchdown_points = (
        _number(stat_line, "passing_tds") * rules.passing_touchdown
        + _number(stat_line, "rushing_tds") * rules.rushing_touchdown
        + _number(stat_line, "receiving_tds") * rules.receiving_touchdown
        + _number(stat_line, "passing_tds_40_plus") * rules.long_touchdown_bonus
        + _number(stat_line, "rushing_tds_40_plus") * rules.long_touchdown_bonus
        + _number(stat_line, "receiving_tds_40_plus") * rules.long_touchdown_bonus
        + _number(stat_line, "return_tds") * rules.return_touchdown
        + _number(stat_line, "offensive_fumble_return_tds")
        * rules.offensive_fumble_return_touchdown
    )
    total = (
        _number(stat_line, "passing_yards") * rules.passing_yard
        + _number(stat_line, "interceptions") * rules.interception
        + _number(stat_line, "rushing_yards") * rules.rushing_yard
        + _number(stat_line, "receptions") * rules.reception
        + _number(stat_line, "receiving_yards") * rules.receiving_yard
        + _number(stat_line, "fumbles_lost") * rules.fumble_lost
        + _number(stat_line, "two_point") * rules.two_point_conversion
        + _number(stat_line, "return_yards") * rules.return_yard
        + _number(stat_line, "extra_points_made") * rules.extra_point_made
        + _number(stat_line, "field_goal_yards_made") * rules.field_goal_yard
        + touchdown_points
    )
    return {"fantasy_points": total, "touchdown_points": touchdown_points}


def score_offensive_frame(
    frame: pd.DataFrame,
    rules: LeagueScoring = FFC_SCORING,
) -> pd.DataFrame:
    """Vectorized offensive/K scoring with a transparent component breakdown."""

    passing_td = _series(frame, "passing_tds") * rules.passing_touchdown
    rushing_td = _series(frame, "rushing_tds") * rules.rushing_touchdown
    receiving_td = _series(frame, "receiving_tds") * rules.receiving_touchdown
    long_td_bonus = (
        _series(frame, "passing_tds_40_plus")
        + _series(frame, "rushing_tds_40_plus")
        + _series(frame, "receiving_tds_40_plus")
    ) * rules.long_touchdown_bonus
    return_td = _series(frame, "return_tds") * rules.return_touchdown
    fumble_return_td = (
        _series(frame, "offensive_fumble_return_tds")
        * rules.offensive_fumble_return_touchdown
    )
    touchdown_points = (
        passing_td + rushing_td + receiving_td + long_td_bonus + return_td
        + fumble_return_td
    )
    total = (
        _series(frame, "passing_yards") * rules.passing_yard
        + _series(frame, "interceptions") * rules.interception
        + _series(frame, "rushing_yards") * rules.rushing_yard
        + _series(frame, "receptions") * rules.reception
        + _series(frame, "receiving_yards") * rules.receiving_yard
        + _series(frame, "fumbles_lost") * rules.fumble_lost
        + _series(frame, "two_point") * rules.two_point_conversion
        + _series(frame, "return_yards") * rules.return_yard
        + _series(frame, "extra_points_made") * rules.extra_point_made
        + _series(frame, "field_goal_yards_made") * rules.field_goal_yard
        + touchdown_points
    )
    return pd.DataFrame(
        {"fantasy_points": total, "touchdown_points": touchdown_points},
        index=frame.index,
    )


def defense_points_allowed(points_allowed: float) -> float:
    """Return the exact configured D/ST points-allowed tier."""

    if points_allowed <= 0:
        return 10.0
    if points_allowed <= 6:
        return 7.0
    if points_allowed <= 13:
        return 4.0
    if points_allowed <= 20:
        return 1.0
    if points_allowed <= 27:
        return 0.0
    if points_allowed <= 34:
        return -1.0
    return -4.0


def score_defense_week(
    stat_line: Mapping[str, object],
    rules: LeagueScoring = FFC_SCORING,
) -> float:
    """Score one D/ST stat line using the league's complete defensive rules."""

    return (
        _number(stat_line, "sacks") * rules.defense_sack
        + _number(stat_line, "interceptions") * rules.defense_interception
        + _number(stat_line, "fumble_recoveries") * rules.defense_fumble_recovery
        + _number(stat_line, "defense_tds") * rules.defense_touchdown
        + _number(stat_line, "safeties") * rules.defense_safety
        + _number(stat_line, "blocked_kicks") * rules.defense_blocked_kick
        + _number(stat_line, "return_yards") * rules.return_yard
        + _number(stat_line, "return_tds") * rules.return_touchdown
        + _number(stat_line, "extra_point_returns")
        * rules.defense_extra_point_return
        + defense_points_allowed(_number(stat_line, "points_allowed"))
    )
