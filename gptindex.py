#!/usr/bin/env python3
"""
Fantasy Command Center - Beta v1
================================

Personal fantasy football forecasting / dashboard system.

Primary data source:
    nflreadpy / nflverse

Current league defaults:
    12 teams
    Custom Yahoo league scoring
    1 QB
    2 RB
    2 WR
    1 TE
    1 K
    1 DEF
    5 bench

INPUT:
    data/my_roster.txt

Paste your Yahoo roster table into that file.

OUTPUT:
    fantasy_dashboard.html
    data/output/*.csv

INSTALL:
    pip install nflreadpy pandas numpy pyarrow rapidfuzz lxml

OPTIONAL ML:
    pip install xgboost

EXAMPLES:
    python index.py
    python index.py --season 2026
    python index.py --season 2026 --week 4
    python index.py --roster data/my_roster.txt
    python index.py --train-ml
    python index.py --backtest

The program is designed to degrade gracefully if a particular
nflreadpy dataset/column isn't available.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import unicodedata
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:
    print("\nERROR: nflreadpy is not installed.")
    print("Run: pip install nflreadpy pandas numpy pyarrow rapidfuzz lxml")
    sys.exit(1)

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Fandromeda2"
WORDMARK_PATH = Path(
    r"C:\Users\jtpag\Documents\Python\ClaudeDash\fantasy_tool"
    r"\fantasy_tool\tools\wordmark_paths.svg"
)

DEFAULT_SEASON = 2026

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

ROSTER_FILE = DATA_DIR / "my_roster.txt"
HTML_OUTPUT = Path("index.html")
LEARNED_WEIGHT_MODEL_PATH = CACHE_DIR / "learned_weight_model.json"
LEARNED_WEIGHT_REPORT_PATH = OUTPUT_DIR / "learned_metric_weights.csv"

# YOUR LEAGUE
LEAGUE_TEAMS = 12
ROSTER_SLOTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "K": 1,
    "DEF": 1,
    "BENCH": 5,
}

# CUSTOM YAHOO LEAGUE SCORING
SCORING = {
    "pass_yd": 0.04,
    "pass_td": 6.0,
    "int": -2.0,
    "rush_yd": 0.10,
    "rush_td": 6.0,
    "rec": 0.0,
    "rec_yd": 0.10,
    "rec_td": 6.0,
    "fum_lost": -2.0,
    "two_pt": 2.0,
    # Return/long-TD settings are retained for the upcoming play-by-play
    # scoring module; weekly player totals alone do not identify 40+ yard TDs.
    "return_yd": 0.04,
    "return_td": 6.0,
    "pass_td_40_bonus": 1.0,
    "rush_td_40_bonus": 1.0,
    "rec_td_40_bonus": 1.0,
}

# Recency weighting.
EWMA_ALPHA = 0.45

# Minimum acceptable fuzzy name match.
FUZZY_MATCH_THRESHOLD = 90

# Positions we care about for offensive fantasy projections.
OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE"}

# Canonical GSIS IDs for rare cases where Yahoo's full display name cannot
# be resolved from the NFLverse display-name field. Keep this list small and
# explicit so every exceptional match is auditable.
PLAYER_ID_OVERRIDES = {
    "justin jefferson": ("00-0036322", "Justin Jefferson", "MIN", "WR"),
    "marvin harrison": ("00-0039988", "Marvin Harrison Jr.", "ARI", "WR"),
    "aaron jones": ("00-0033293", "Aaron Jones", "MIN", "RB"),
    "devonta smith": ("00-0036912", "DeVonta Smith", "PHI", "WR"),
}

# ============================================================
# GENERAL UTILITIES
# ============================================================


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def clean_text(value) -> str:
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\xa0", " ")
    value = value.replace("’", "'")
    value = value.replace("`", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_name(name: str) -> str:
    """
    Normalize player names aggressively enough to handle
    Yahoo/NFLverse formatting differences without destroying
    identity.
    """
    name = clean_text(name).lower()

    # Remove common fantasy-site decorations.
    name = re.sub(r"\[[^\]]*\]", " ", name)
    name = re.sub(r"\([^)]*\)", " ", name)

    # Remove suffixes.
    name = re.sub(
        r"\b(jr|sr|ii|iii|iv|v)\.?\b",
        " ",
        name,
    )

    # Normalize apostrophes.
    name = name.replace("'", "")

    # Remove punctuation.
    name = re.sub(r"[^a-z0-9 ]", " ", name)

    # Collapse spaces.
    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_team(team: str) -> str:
    team = clean_text(team).upper()

    aliases = {
        "JAC": "JAX",
        "LA": "LAR",
        "STL": "LAR",
        "SD": "LAC",
        "OAK": "LV",
        "LV": "LV",
        "WSH": "WAS",
        "WAS": "WAS",
        "NE": "NE",
        "NWE": "NE",
        "TB": "TB",
        "TAM": "TB",
        "SF": "SF",
        "SFO": "SF",
        "KC": "KC",
        "KAN": "KC",
        "GB": "GB",
        "GNB": "GB",
        "NO": "NO",
        "NOR": "NO",
        "CAR": "CAR",
        "ATL": "ATL",
        "CHI": "CHI",
        "CIN": "CIN",
        "CLE": "CLE",
        "DAL": "DAL",
        "DEN": "DEN",
        "DET": "DET",
        "HOU": "HOU",
        "IND": "IND",
        "MIA": "MIA",
        "MIN": "MIN",
        "NYG": "NYG",
        "NYJ": "NYJ",
        "PHI": "PHI",
        "PIT": "PIT",
        "SEA": "SEA",
        "TEN": "TEN",
        "ARI": "ARI",
        "BAL": "BAL",
        "BUF": "BUF",
    }

    return aliases.get(team, team)


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}

    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]

    return None


def rename_if_present(
    df: pd.DataFrame,
    aliases: Dict[str, List[str]],
) -> pd.DataFrame:

    rename_map = {}

    for target, candidates in aliases.items():
        col = first_existing(df, candidates)

        if col and col != target:
            rename_map[col] = target

    return df.rename(columns=rename_map)


def to_pandas(obj) -> pd.DataFrame:
    """
    nflreadpy returns Polars DataFrames.
    """
    if obj is None:
        return pd.DataFrame()

    if isinstance(obj, pd.DataFrame):
        return obj

    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()

    return pd.DataFrame(obj)


# ============================================================
# CACHING
# ============================================================


def cache_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return CACHE_DIR / f"{safe}.parquet"


def load_cached(
    name: str,
    loader,
    force: bool = False,
) -> pd.DataFrame:

    path = cache_path(name)

    if path.exists() and not force:
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    print(f"Downloading: {name}")

    try:
        data = to_pandas(loader())
    except Exception as exc:
        print(f"  WARNING: unable to load {name}: {exc}")
        return pd.DataFrame()

    if not data.empty:
        try:
            data.to_parquet(path, index=False)
        except Exception as exc:
            print(f"  WARNING: unable to cache {name}: {exc}")

    return data


# ============================================================
# YAHOO ROSTER PARSER
# ============================================================


def parse_yahoo_roster(path: Path) -> pd.DataFrame:
    """
    Handles:
      - HTML copied from browser
      - TSV copied from Yahoo
      - CSV
      - whitespace-delimited fallback
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Roster file not found: {path}\n"
            f"Create it and paste your Yahoo roster table into it."
        )

    raw = path.read_text(encoding="utf-8", errors="ignore")

    if not raw.strip():
        raise ValueError(f"{path} is empty.")

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    if "<table" in raw.lower():
        try:
            tables = pd.read_html(raw)

            if tables:
                # Prefer table containing player/name terminology.
                chosen = tables[0]

                for table in tables:
                    cols = " ".join(str(c).lower() for c in table.columns)
                    if any(
                        x in cols
                        for x in ["player", "name", "position"]
                    ):
                        chosen = table
                        break

                return standardize_roster_columns(chosen)
        except Exception:
            pass

    # --------------------------------------------------------
    # TSV / CSV
    # --------------------------------------------------------

    lines = [line for line in raw.splitlines() if line.strip()]

    first = lines[0]

    if "\t" in first:
        try:
            df = pd.read_csv(
                pd.io.common.StringIO(raw),
                sep="\t",
            )
            return standardize_roster_columns(df)
        except Exception:
            pass

    if "," in first:
        try:
            df = pd.read_csv(
                pd.io.common.StringIO(raw)
            )
            return standardize_roster_columns(df)
        except Exception:
            pass

    # --------------------------------------------------------
    # Try generic whitespace parser
    # --------------------------------------------------------

    try:
        df = pd.read_csv(
            pd.io.common.StringIO(raw),
            sep=r"\s{2,}",
            engine="python",
        )

        if len(df.columns) > 1:
            return standardize_roster_columns(df)
    except Exception:
        pass

    # --------------------------------------------------------
    # Last resort:
    # one player per line.
    # --------------------------------------------------------

    # Yahoo's visual roster copy puts each field on a separate line:
    # roster slot, player name, optional labels, then "TEAM - POSITION".
    # Parse that layout before using a generic fallback.
    records = []
    roster_slots = {
        "QB", "RB", "WR", "TE", "K", "DEF", "D/ST", "DST",
        "W/R/T", "W/R", "BN", "BENCH", "IR",
    }
    current_manager = ""

    for i, raw_line in enumerate(lines):
        line = clean_text(raw_line)

        # Yahoo appends a private-use icon to fantasy-team headings.  The
        # particular code point varies by copy/paste source, so accept its
        # private-use range instead of one hard-coded glyph.
        manager_match = re.fullmatch(
            r"(.*?)(?:\s*[\uE000-\uF8FF])+",
            line,
        )
        if manager_match:
            current_manager = clean_text(manager_match.group(1))
            continue

        slot = line.upper()
        if slot not in roster_slots:
            continue

        # The next nonblank line after a roster slot is the display name.
        player = ""
        for candidate in lines[i + 1:]:
            candidate = clean_text(candidate)
            if candidate:
                player = candidate
                break

        if not player:
            continue

        position = slot
        team = ""
        for candidate in lines[i + 1:]:
            candidate = clean_text(candidate)
            if (
                candidate.upper() in roster_slots
                or re.search(r"[\uE000-\uF8FF]", candidate)
            ):
                break
            team_match = re.fullmatch(
                r"([A-Za-z]{2,3})\s*-\s*(QB|RB|WR|TE|K|DEF|D/ST|DST)",
                candidate,
                re.I,
            )
            if team_match:
                team = team_match.group(1)
                position = team_match.group(2).upper()
                break

        records.append(
            {
                "manager": current_manager,
                "player": player,
                "team": team,
                "position": position,
                "slot": slot,
            }
        )

    if not records:
        raise ValueError(
            "Could not parse Yahoo roster.\n"
            "Paste the roster table including column headers."
        )

    # Keep the fallback parser on the same normalized schema as HTML/CSV/TSV
    # inputs.  The matching stage always expects ``player_normalized``.
    return standardize_roster_columns(pd.DataFrame(records))


def standardize_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        clean_text(c).lower().replace(" ", "_")
        for c in df.columns
    ]

    aliases = {
        "manager": [
            "manager",
            "team_manager",
            "owner",
            "fantasy_team",
            "fantasy_manager",
        ],
        "player": [
            "player",
            "player_name",
            "name",
            "full_name",
        ],
        "team": [
            "team",
            "nfl_team",
            "nfl",
            "tm",
        ],
        "position": [
            "position",
            "pos",
        ],
        "slot": [
            "slot",
            "roster_slot",
            "status",
            "lineup_slot",
        ],
    }

    df = rename_if_present(df, aliases)

    # If player column wasn't obvious, look for likely column.
    if "player" not in df.columns:
        candidates = [
            c
            for c in df.columns
            if "player" in c or "name" in c
        ]

        if candidates:
            df = df.rename(columns={candidates[0]: "player"})

    if "player" not in df.columns:
        raise ValueError(
            "Yahoo roster parser could not find a player column.\n"
            f"Detected columns: {list(df.columns)}"
        )

    for col in ["manager", "team", "position", "slot"]:
        if col not in df.columns:
            df[col] = ""

    for col in df.columns:
        df[col] = df[col].map(clean_text)

    df["position"] = (
        df["position"]
        .str.upper()
        .replace({"D/ST": "DEF", "DST": "DEF", "BN": "BENCH"})
    )

    df["slot"] = df["slot"].str.upper()

    df["player_normalized"] = df["player"].map(normalize_name)

    return df[
        [
            "manager",
            "player",
            "team",
            "position",
            "slot",
            "player_normalized",
        ]
    ].drop_duplicates()


# ============================================================
# NFLVERSE PLAYER MASTER
# ============================================================


def load_player_master(season: int) -> pd.DataFrame:

    players = load_cached(
        "players_master",
        lambda: nfl.load_players(),
    )

    if players.empty:
        raise RuntimeError("NFLverse player data could not be loaded.")

    aliases = {
        "player_id": [
            "gsis_id",
            "player_id",
            "id",
        ],
        "display_name": [
            "display_name",
            "full_name",
            "player_name",
            "name",
        ],
        "position": [
            "position",
            "position_group",
        ],
        "team": [
            "recent_team",
            "team",
        ],
    }

    players = rename_if_present(players, aliases)

    if "player_id" not in players:
        raise RuntimeError(
            "NFLverse player data does not contain a usable player ID."
        )

    if "display_name" not in players:
        players["display_name"] = ""

    if "position" not in players:
        players["position"] = ""

    if "team" not in players:
        players["team"] = ""

    players["display_name"] = players["display_name"].map(clean_text)
    players["position"] = players["position"].map(clean_text).str.upper()
    players["team"] = players["team"].map(normalize_team)

    players["name_normalized"] = players["display_name"].map(
        normalize_name
    )

    return players


# ============================================================
# PLAYER MATCHING
# ============================================================


def build_player_match_table(
    yahoo: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:

    players_by_name = {}
    players_by_last_name = {}

    for _, row in players.iterrows():
        key = row["name_normalized"]

        if key:
            players_by_name.setdefault(key, []).append(row)
            players_by_last_name.setdefault(
                key.split()[-1], []
            ).append(row)

    all_names = list(players_by_name.keys())

    matches = []

    for _, row in yahoo.iterrows():

        yahoo_name = row["player"]
        normalized = row["player_normalized"]
        yahoo_team = normalize_team(row["team"])
        yahoo_position = clean_text(row.get("position", "")).upper()

        # D/ST entries are teams, not individual NFLverse players.  Keep
        # them on the roster but do not send them through player matching.
        if yahoo_position in {"DEF", "D/ST", "DST"}:
            record = row.to_dict()
            record["nflverse_player_id"] = ""
            record["nflverse_name"] = ""
            record["nflverse_team"] = yahoo_team
            record["nflverse_position"] = "DEF"
            record["match_method"] = "team_defense"
            record["match_confidence"] = 100.0
            matches.append(record)
            continue

        override = PLAYER_ID_OVERRIDES.get(normalized)
        if override:
            player_id, display_name, team, position = override
            record = row.to_dict()
            record["nflverse_player_id"] = player_id
            record["nflverse_name"] = display_name
            record["nflverse_team"] = team
            record["nflverse_position"] = position
            record["match_method"] = "explicit_override"
            record["match_confidence"] = 100.0
            matches.append(record)
            continue

        matched = None
        method = "unmatched"
        confidence = 0.0

        # ----------------------------------------------------
        # Exact normalized name
        # ----------------------------------------------------

        candidates = players_by_name.get(normalized, [])

        if len(candidates) == 1:
            matched = candidates[0]
            method = "exact"
            confidence = 100.0

        elif len(candidates) > 1 and yahoo_team:
            team_candidates = [
                candidate
                for candidate in candidates
                if normalize_team(candidate.get("team", ""))
                == yahoo_team
            ]

            if len(team_candidates) == 1:
                matched = team_candidates[0]
                method = "exact_name_team"
                confidence = 100.0

        # Yahoo commonly uses full first names where NFLverse stores an
        # initial (for example, "Justin Jefferson" vs "J.Jefferson").
        # Only accept this match when surname, initial, NFL team, and
        # position jointly identify one player.
        if matched is None and normalized:
            yahoo_parts = normalized.split()

            if len(yahoo_parts) >= 2:
                surname_candidates = [
                    candidate
                    for candidate in players_by_last_name.get(
                        yahoo_parts[-1], []
                    )
                    if candidate["name_normalized"].split()[0][0]
                    == yahoo_parts[0][0]
                ]

                if yahoo_position:
                    surname_candidates = [
                        candidate
                        for candidate in surname_candidates
                        if clean_text(candidate.get("position", "")).upper()
                        == yahoo_position
                    ]

                position_candidates = surname_candidates

                if yahoo_team:
                    team_candidates = [
                        candidate
                        for candidate in position_candidates
                        if normalize_team(candidate.get("team", ""))
                        == yahoo_team
                    ]
                else:
                    team_candidates = position_candidates

                if len(team_candidates) == 1:
                    matched = team_candidates[0]
                    method = "surname_initial_team_position"
                    confidence = 98.0
                elif len(position_candidates) == 1:
                    # NFLverse's recent-team field can lag an offseason move.
                    matched = position_candidates[0]
                    method = "surname_initial_position"
                    confidence = 94.0

        # Last-name matching is a final safe fallback for NFLverse display
        # names that omit or vary the first name. Position must still agree.
        if matched is None and normalized and yahoo_position:
            last_name = normalized.split()[-1]
            last_name_candidates = [
                candidate
                for candidates in players_by_name.values()
                for candidate in candidates
                if last_name in candidate["name_normalized"].split()
                and clean_text(candidate.get("position", "")).upper()
                == yahoo_position
            ]

            if len(last_name_candidates) == 1:
                matched = last_name_candidates[0]
                method = "unique_last_name_position"
                confidence = 90.0

        # ----------------------------------------------------
        # Fuzzy matching
        # ----------------------------------------------------

        if matched is None and RAPIDFUZZ_AVAILABLE and normalized:
            result = process.extractOne(
                normalized,
                all_names,
                scorer=fuzz.ratio,
            )

            if result:
                best_name, score, _ = result

                if score >= FUZZY_MATCH_THRESHOLD:
                    candidate_rows = players_by_name.get(
                        best_name,
                        [],
                    )

                    # Prefer same-team candidate.
                    if yahoo_team:
                        team_candidates = [
                            candidate
                            for candidate in candidate_rows
                            if normalize_team(
                                candidate.get("team", "")
                            )
                            == yahoo_team
                        ]

                        if len(team_candidates) == 1:
                            matched = team_candidates[0]
                            method = "fuzzy_name_team"
                            confidence = score

                    if matched is None and len(candidate_rows) == 1:
                        matched = candidate_rows[0]
                        method = "fuzzy_name"
                        confidence = score

        record = row.to_dict()

        if matched is not None:
            record["nflverse_player_id"] = matched["player_id"]
            record["nflverse_name"] = matched["display_name"]
            record["nflverse_team"] = matched.get("team", "")
            record["nflverse_position"] = matched.get("position", "")
        else:
            record["nflverse_player_id"] = ""
            record["nflverse_name"] = ""
            record["nflverse_team"] = ""
            record["nflverse_position"] = ""

        record["match_method"] = method
        record["match_confidence"] = round(confidence, 1)

        matches.append(record)

    return pd.DataFrame(matches)


# ============================================================
# PLAYER STATS
# ============================================================


def standardize_player_stats(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    aliases = {
        "player_id": [
            "player_id",
            "gsis_id",
            "id",
        ],
        "player_name": [
            "player_name",
            "display_name",
            "full_name",
            "name",
        ],
        "season": [
            "season",
        ],
        "week": [
            "week",
        ],
        "position": [
            "position",
            "position_group",
        ],
        "team": [
            "recent_team",
            "team",
            "posteam",
        ],
        "opponent": [
            "opponent_team",
            "opponent",
            "defteam",
        ],
        "fantasy_points": [
            "fantasy_points",
            "fantasy_points_ppr",
            "fantasy_points_half_ppr",
        ],
        "passing_yards": [
            "passing_yards",
            "pass_yds",
            "passing_yd",
        ],
        "passing_attempts": [
            "passing_attempts",
            "pass_attempts",
            "attempts",
        ],
        "passing_tds": [
            "passing_tds",
            "pass_td",
        ],
        "interceptions": [
            "interceptions",
            "interception",
            "ints",
        ],
        "rushing_yards": [
            "rushing_yards",
            "rush_yds",
        ],
        "rushing_tds": [
            "rushing_tds",
            "rush_td",
        ],
        "carries": [
            "carries",
            "rush_attempts",
            "rushing_attempts",
        ],
        "receiving_yards": [
            "receiving_yards",
            "rec_yds",
        ],
        "receiving_tds": [
            "receiving_tds",
            "rec_td",
        ],
        "targets": [
            "targets",
            "target",
        ],
        "receptions": [
            "receptions",
            "rec",
        ],
        "fumbles_lost": [
            "rushing_fumbles_lost",
            "receiving_fumbles_lost",
            "fumbles_lost",
        ],
        "two_point": [
            "two_point_conversions",
            "two_point_conversions_pass",
            "two_point_conversions_rush",
            "two_point_conversions_rec",
        ],
    }

    df = rename_if_present(df, aliases)

    required_defaults = {
        "passing_yards": 0,
        "passing_attempts": 0,
        "passing_tds": 0,
        "interceptions": 0,
        "rushing_yards": 0,
        "rushing_tds": 0,
        "carries": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "targets": 0,
        "receptions": 0,
        "fumbles_lost": 0,
        "two_point": 0,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    if "player_id" not in df.columns:
        raise RuntimeError(
            "Player stats do not contain player IDs."
        )

    for col in [
        "season",
        "week",
        "passing_yards",
        "passing_attempts",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "carries",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "receptions",
        "fumbles_lost",
        "two_point",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    df["team"] = (
        df["team"].map(normalize_team)
        if "team" in df.columns
        else ""
    )

    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(normalize_team)
    else:
        df["opponent"] = ""

    # If nflverse already supplies fantasy_points,
    # keep it only if it appears to be standard scoring.
    # We calculate our own points to guarantee league scoring.
    df["fantasy_points_std"] = (
        df["passing_yards"] * SCORING["pass_yd"]
        + df["passing_tds"] * SCORING["pass_td"]
        + df["interceptions"] * SCORING["int"]
        + df["rushing_yards"] * SCORING["rush_yd"]
        + df["rushing_tds"] * SCORING["rush_td"]
        + df["receptions"] * SCORING["rec"]
        + df["receiving_yards"] * SCORING["rec_yd"]
        + df["receiving_tds"] * SCORING["rec_td"]
        + df["fumbles_lost"] * SCORING["fum_lost"]
        + df["two_point"] * SCORING["two_pt"]
    )

    df["td_points"] = (
        df["passing_tds"] * SCORING["pass_td"]
        + df["rushing_tds"] * SCORING["rush_td"]
        + df["receiving_tds"] * SCORING["rec_td"]
    )

    df["non_td_fantasy_points"] = (
        df["fantasy_points_std"] - df["td_points"]
    )

    return df


def load_current_stats(
    season: int,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load player-week stats, optionally refreshing a live season cache."""

    raw = load_cached(
        f"player_stats_{season}",
        lambda: nfl.load_player_stats(
            season,
            summary_level="week",
        ),
        force=refresh,
    )

    return standardize_player_stats(raw)


def load_current_injury_status(
    season: int,
    target_week: int,
) -> pd.DataFrame:
    """Load the newest official-style injury status for each player."""

    raw = load_cached(
        f"injuries_{season}",
        lambda: nfl.load_injuries(season),
        # Injury reports change during the week, so never reuse a stale cache.
        force=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=["player_id", "injury_flag", "injury_note"])

    aliases = {
        "player_id": ["gsis_id", "player_id", "nfl_id"],
        "week": ["week"],
        "status": ["report_status", "game_status", "status"],
        "injury": [
            "report_primary_injury",
            "injury_description",
            "injury",
        ],
        "updated": ["date_modified", "report_date", "date"],
    }
    injuries = rename_if_present(to_pandas(raw), aliases)
    if "player_id" not in injuries or "status" not in injuries:
        return pd.DataFrame(columns=["player_id", "injury_flag", "injury_note"])

    for column in ["week", "status", "injury", "updated"]:
        if column not in injuries:
            injuries[column] = "" if column != "week" else np.nan

    injuries["week"] = pd.to_numeric(injuries["week"], errors="coerce")
    injuries = injuries[injuries["week"] <= target_week].copy()
    injuries["status"] = injuries["status"].map(clean_text).str.upper()
    injuries["injury"] = injuries["injury"].map(clean_text)
    injuries["updated"] = pd.to_datetime(injuries["updated"], errors="coerce")

    def flag_for_status(status: str) -> str:
        if status == "IR" or any(
            term in status
            for term in ["INJURED RESERVE", "RESERVE/INJURED", " IR"]
        ):
            return "IR"
        if status == "OUT":
            return "O"
        return ""

    injuries["injury_flag"] = injuries["status"].map(flag_for_status)
    injuries = injuries[injuries["injury_flag"] != ""].copy()
    if injuries.empty:
        return pd.DataFrame(columns=["player_id", "injury_flag", "injury_note"])

    injuries["injury_note"] = injuries.apply(
        lambda row: " · ".join(
            part for part in [row["status"].title(), row["injury"]] if part
        ),
        axis=1,
    )
    return (
        injuries
        .sort_values(["player_id", "week", "updated"])
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "injury_flag", "injury_note"]]
    )


# ============================================================
# SNAP COUNTS
# ============================================================


def standardize_snaps(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    aliases = {
        "player_id": [
            "player_id",
            "gsis_id",
        ],
        "season": ["season"],
        "week": ["week"],
        "team": [
            "team",
            "recent_team",
        ],
        "offense_snaps": [
            "offense_snaps",
            "offensive_snaps",
            "off_snap",
        ],
        "offense_pct": [
            "offense_pct",
            "offense_snap_pct",
            "offensive_pct",
        ],
    }

    df = rename_if_present(df, aliases)

    for col in [
        "offense_snaps",
        "offense_pct",
    ]:
        if col not in df:
            df[col] = np.nan

    for col in [
        "season",
        "week",
        "offense_snaps",
        "offense_pct",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["team"] = df.get(
        "team",
        pd.Series(index=df.index, dtype=object),
    ).map(normalize_team)

    return df


def load_snaps(season: int) -> pd.DataFrame:

    raw = load_cached(
        f"snap_counts_{season}",
        lambda: nfl.load_snap_counts(season),
    )

    return standardize_snaps(raw)


# ============================================================
# DEPTH CHARTS
# ============================================================


def load_depth_charts(season: int) -> pd.DataFrame:

    raw = load_cached(
        f"depth_charts_{season}",
        lambda: nfl.load_depth_charts(season),
    )

    if raw.empty:
        return raw

    df = raw.copy()

    df = rename_if_present(
        df,
        {
            "player_id": [
                "gsis_id",
                "player_id",
            ],
            "season": ["season"],
            "week": ["week"],
            "team": [
                "team",
                "club_code",
            ],
            "position": [
                "position",
                "pos",
            ],
            "depth": [
                "depth",
                "depth_position",
                "depth_order",
            ],
        },
    )

    if "team" in df:
        df["team"] = df["team"].map(normalize_team)

    return df


# ============================================================
# INJURIES
# ============================================================


def load_injuries(season: int) -> pd.DataFrame:

    try:
        raw = load_cached(
            f"injuries_{season}",
            lambda: nfl.load_injuries(season),
        )
    except Exception as exc:
        print(f"  Injury data unavailable: {exc}")
        return pd.DataFrame()

    if raw.empty:
        return raw

    df = raw.copy()

    df = rename_if_present(
        df,
        {
            "player_id": [
                "gsis_id",
                "player_id",
            ],
            "season": ["season"],
            "week": ["week"],
            "status": [
                "game_status",
                "status",
            ],
            "practice_status": [
                "practice_status",
            ],
            "team": [
                "team",
                "team_abbr",
            ],
        },
    )

    return df


# ============================================================
# SCHEDULES
# ============================================================


def load_schedule(season: int) -> pd.DataFrame:

    raw = load_cached(
        f"schedules_{season}",
        lambda: nfl.load_schedules(season),
    )

    if raw.empty:
        return raw

    df = raw.copy()

    df = rename_if_present(
        df,
        {
            "season": ["season"],
            "week": ["week"],
            "home_team": [
                "home_team",
                "home",
            ],
            "away_team": [
                "away_team",
                "away",
            ],
            "home_score": [
                "home_score",
                "home_score_final",
            ],
            "away_score": [
                "away_score",
                "away_score_final",
            ],
        },
    )

    for col in ["home_team", "away_team"]:
        if col in df:
            df[col] = df[col].map(normalize_team)

    return df


# ============================================================
# MERGE USAGE DATA
# ============================================================


def merge_usage(
    stats: pd.DataFrame,
    snaps: pd.DataFrame,
) -> pd.DataFrame:

    df = stats.copy()

    required_keys = {"player_id", "season", "week"}

    # NFLverse snap counts are sometimes keyed by PFR player ID rather than
    # the GSIS/player ID used by weekly player stats.  Do not attempt an
    # unsafe name-based merge; retain projections and leave snap fields blank.
    if snaps.empty or not required_keys.issubset(snaps.columns):
        df["offense_snaps"] = np.nan
        df["offense_pct"] = np.nan
        return df

    snap_cols = [
        c
        for c in [
            "player_id",
            "season",
            "week",
            "offense_snaps",
            "offense_pct",
        ]
        if c in snaps.columns
    ]

    snaps2 = snaps[snap_cols].copy()

    # Avoid duplicate rows if source contains multiple records.
    snaps2 = (
        snaps2
        .groupby(
            ["player_id", "season", "week"],
            as_index=False,
        )
        .agg(
            {
                "offense_snaps": "max",
                "offense_pct": "max",
            }
        )
    )

    df = df.merge(
        snaps2,
        on=["player_id", "season", "week"],
        how="left",
        suffixes=("", "_snap"),
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def ewma(series: pd.Series, alpha: float = EWMA_ALPHA):
    return series.ewm(
        alpha=alpha,
        adjust=False,
        min_periods=1,
    ).mean()


def safe_ratio(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")

    return np.where(
        b.abs() > 1e-9,
        a / b,
        np.nan,
    )


def add_player_features(
    historical: pd.DataFrame,
    target_week: int,
) -> pd.DataFrame:

    """
    Creates a player-level feature table using ONLY games
    before target_week.

    This is the key anti-leakage rule.
    """

    df = historical.copy()

    df = df[
        pd.to_numeric(df["week"], errors="coerce")
        < target_week
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(
        ["player_id", "season", "week"]
    )

    # --------------------------------------------------------
    # Base per-game opportunity metrics
    # --------------------------------------------------------

    df["target_share_proxy"] = np.nan
    df["carry_share_proxy"] = np.nan

    # We calculate shares within the player's own history where
    # team-level denominators aren't available.
    # Targets/carries remain the primary opportunity measures.

    df["fantasy_per_target"] = safe_ratio(
        df["fantasy_points_std"],
        df["targets"],
    )

    df["fantasy_per_carry"] = safe_ratio(
        df["fantasy_points_std"],
        df["carries"],
    )

    df["td_rate"] = safe_ratio(
        df["td_points"],
        df["fantasy_points_std"].abs(),
    )

    grouped = df.groupby(
        ["player_id", "season"],
        group_keys=False,
    )

    metric_columns = [
        "fantasy_points_std",
        "targets",
        "carries",
        "receiving_yards",
        "rushing_yards",
        "receptions",
        "offense_snaps",
        "offense_pct",
        "td_points",
        "red_zone_targets",
        "red_zone_carries",
    ]

    # Make optional columns.
    for col in [
        "red_zone_targets",
        "red_zone_carries",
    ]:
        if col not in df:
            df[col] = 0

    # --------------------------------------------------------
    # Lag all features so the current target week can never
    # leak into its own prediction.
    # --------------------------------------------------------

    for col in metric_columns:
        if col not in df:
            df[col] = 0

        shifted = grouped[col].shift(1)

        df[f"{col}_last"] = shifted

        df[f"{col}_roll2"] = (
            shifted
            .groupby(
                [df["player_id"], df["season"]]
            )
            .rolling(2, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        df[f"{col}_roll3"] = (
            shifted
            .groupby(
                [df["player_id"], df["season"]]
            )
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        df[f"{col}_roll5"] = (
            shifted
            .groupby(
                [df["player_id"], df["season"]]
            )
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        # EWMA with the current game excluded.
        df[f"{col}_ewma"] = (
            shifted
            .groupby(
                [df["player_id"], df["season"]]
            )
            .transform(
                lambda s: s.ewm(
                    alpha=EWMA_ALPHA,
                    adjust=False,
                    min_periods=1,
                ).mean()
            )
        )

    # The scoring column is named ``fantasy_points_std``, while the
    # projection code uses the shorter ``fantasy_points`` feature names.
    # Publish those aliases once, after every lagged feature is created.
    for suffix in ["last", "roll2", "roll3", "roll5", "ewma"]:
        df[f"fantasy_points_{suffix}"] = (
            df[f"fantasy_points_std_{suffix}"]
        )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    def momentum(col: str):
        recent = df[f"{col}_roll2"]
        prior = (
            grouped[col]
            .shift(3)
            .groupby(
                [df["player_id"], df["season"]]
            )
            .rolling(2, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        return recent - prior

    for col in [
        "fantasy_points_std",
        "targets",
        "carries",
        "offense_pct",
        "offense_snaps",
        "red_zone_targets",
        "red_zone_carries",
    ]:
        df[f"{col}_momentum"] = momentum(col)

    # --------------------------------------------------------
    # Opportunity score
    # --------------------------------------------------------

    target_component = (
        df["targets_ewma"].fillna(0) * 1.0
    )

    carry_component = (
        df["carries_ewma"].fillna(0) * 0.55
    )

    snap_component = (
        df["offense_pct_ewma"].fillna(0) * 0.06
    )

    rz_component = (
        df["red_zone_targets_ewma"].fillna(0) * 1.4
        + df["red_zone_carries_ewma"].fillna(0) * 1.1
    )

    df["opportunity_score"] = (
        target_component
        + carry_component
        + snap_component
        + rz_component
    )

    # Opportunity momentum.
    df["opportunity_momentum"] = (
        df["targets_momentum"].fillna(0)
        + 0.55 * df["carries_momentum"].fillna(0)
        + 0.03 * df["offense_pct_momentum"].fillna(0)
        + 0.5 * df["red_zone_targets_momentum"].fillna(0)
        + 0.4 * df["red_zone_carries_momentum"].fillna(0)
    )

    # --------------------------------------------------------
    # Sustainability
    # --------------------------------------------------------

    opportunity = df["opportunity_score"].fillna(0)
    production = df["fantasy_points_ewma"].fillna(0)

    # Production relative to opportunity.
    df["efficiency_score"] = safe_ratio(
        production,
        opportunity,
    )

    # TD dependency.
    df["td_dependency"] = safe_ratio(
        df["td_points_ewma"],
        production.abs(),
    )

    # Higher = more concerning.
    df["sustainability_risk"] = (
        df["td_dependency"].fillna(0) * 0.65
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    df["production_momentum"] = (
        df["fantasy_points_std_momentum"]
    )

    df["trend_gap"] = (
        df["opportunity_momentum"].fillna(0)
        - df["production_momentum"].fillna(0)
    )

    # --------------------------------------------------------
    # Final row per player.
    # --------------------------------------------------------

    # Retain sample size so early-season estimates are not presented with a
    # misleadingly precise confidence percentage.
    df["games_played"] = df.groupby(
        ["player_id", "season"]
    ).cumcount() + 1

    latest = (
        df.sort_values(
            ["player_id", "season", "week"]
        )
        .groupby(
            ["player_id", "season"],
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    return latest


# ============================================================
# SIMPLE BASELINE PROJECTION
# ============================================================


def project_baseline(features: pd.DataFrame) -> pd.DataFrame:

    if features.empty:
        return features

    df = features.copy()

    # Main production signal.
    ewma_score = df["fantasy_points_ewma"].fillna(0)
    last3 = df["fantasy_points_roll3"].fillna(ewma_score)
    last5 = df["fantasy_points_roll5"].fillna(ewma_score)
    season = df["fantasy_points_ewma"].fillna(ewma_score)

    base = (
        0.45 * ewma_score
        + 0.25 * last3
        + 0.15 * last5
        + 0.15 * season
    )

    # --------------------------------------------------------
    # Opportunity adjustment
    # --------------------------------------------------------

    opp_momentum = (
        df["opportunity_momentum"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Keep this modest. We don't want opportunity momentum
    # to overwhelm actual production.
    opportunity_adjustment = np.clip(
        opp_momentum * 0.35,
        -2.5,
        2.5,
    )

    # --------------------------------------------------------
    # Sustainability penalty
    # --------------------------------------------------------

    td_dependency = (
        df["td_dependency"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Penalize extreme TD dependency.
    td_penalty = np.clip(
        (td_dependency - 0.45) * 2.0,
        0,
        2.0,
    )

    df["baseline_projection"] = np.maximum(
        0,
        base
        + opportunity_adjustment
        - td_penalty,
    )

    # --------------------------------------------------------
    # Range / confidence
    # --------------------------------------------------------

    rolling_values = []

    for _, row in df.iterrows():
        values = [
            safe_float(row.get("fantasy_points_last")),
            safe_float(row.get("fantasy_points_roll2")),
            safe_float(row.get("fantasy_points_roll3")),
            safe_float(row.get("fantasy_points_roll5")),
            safe_float(row.get("fantasy_points_ewma")),
        ]

        values = [
            x for x in values
            if np.isfinite(x)
        ]

        if len(values) >= 2:
            rolling_values.append(np.std(values))
        else:
            rolling_values.append(4.0)

    df["volatility"] = rolling_values

    df["range_low"] = np.maximum(
        0,
        df["baseline_projection"]
        - 1.15 * df["volatility"],
    )

    df["range_high"] = (
        df["baseline_projection"]
        + 1.15 * df["volatility"]
    )

    # Confidence starts at 50 and is adjusted by:
    # consistency, sample size, and opportunity.
    df["confidence"] = 50.0

    df["confidence"] += np.clip(
        12 - df["volatility"].fillna(8),
        -10,
        10,
    )

    df["confidence"] += np.clip(
        df["opportunity_score"].fillna(0) * 1.0,
        0,
        15,
    )

    df["confidence"] = df["confidence"].clip(
        20,
        90,
    )

    # Two games do not provide enough evidence for a meaningful confidence
    # percentage. The dashboard labels these estimates "Early season".
    df.loc[
        pd.to_numeric(df.get("games_played", 0), errors="coerce") < 3,
        "confidence",
    ] = np.nan

    # --------------------------------------------------------
    # Player classification
    # --------------------------------------------------------

    classifications = []

    for _, row in df.iterrows():

        opportunity_momentum = safe_float(
            row.get("opportunity_momentum"),
            0,
        )

        production_momentum = safe_float(
            row.get("production_momentum"),
            0,
        )

        td_dependency = safe_float(
            row.get("td_dependency"),
            0,
        )

        trend_gap = safe_float(
            row.get("trend_gap"),
            0,
        )

        if opportunity_momentum > 2 and trend_gap > 1:
            label = "Emerging opportunity"

        elif (
            td_dependency > 0.55
            and production_momentum > 0
        ):
            label = "TD-dependent production"

        elif (
            production_momentum < -1
            and opportunity_momentum < -1
        ):
            label = "Usage decline"

        elif (
            production_momentum > 1
            and opportunity_momentum > 1
        ):
            label = "Production + usage rising"

        elif (
            production_momentum > 1
            and opportunity_momentum < -1
        ):
            label = "Efficiency / regression watch"

        else:
            label = "Stable"

        classifications.append(label)

    df["classification"] = classifications

    return df


# ============================================================
# SCHEDULE / MATCHUP FEATURES
# ============================================================


def add_matchup_features(
    projections: pd.DataFrame,
    schedule: pd.DataFrame,
    target_week: int,
) -> pd.DataFrame:

    if projections.empty or schedule.empty:
        projections["opponent"] = projections.get(
            "opponent",
            "",
        )
        return projections

    df = projections.copy()

    sched = schedule.copy()

    sched["week"] = pd.to_numeric(
        sched["week"],
        errors="coerce",
    )

    week_games = sched[
        sched["week"] == target_week
    ].copy()

    opponent_map = {}

    for _, game in week_games.iterrows():
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        if home and away:
            opponent_map[home] = away
            opponent_map[away] = home

    df["opponent"] = (
        df["team"]
        .map(opponent_map)
        .fillna("")
    )

    # Game environment if scores/lines happen to be present.
    df["game_total_proxy"] = np.nan
    df["team_implied_points_proxy"] = np.nan

    if (
        "home_score" in week_games.columns
        and "away_score" in week_games.columns
    ):
        # Scores are not pregame information, so DO NOT use them
        # for current projections.
        pass

    return df


# ============================================================
# WHY EXPLANATIONS
# ============================================================


def generate_why(row: pd.Series) -> List[str]:

    reasons = []

    proj = safe_float(
        row.get("baseline_projection"),
        0,
    )

    opportunity = safe_float(
        row.get("opportunity_score"),
        0,
    )

    opp_momentum = safe_float(
        row.get("opportunity_momentum"),
        0,
    )

    production_momentum = safe_float(
        row.get("production_momentum"),
        0,
    )

    td_dependency = safe_float(
        row.get("td_dependency"),
        0,
    )

    trend_gap = safe_float(
        row.get("trend_gap"),
        0,
    )

    if opportunity >= 12:
        reasons.append("Strong recent opportunity")

    elif opportunity >= 7:
        reasons.append("Healthy recent opportunity")

    elif opportunity < 3:
        reasons.append("Limited recent opportunity")

    if opp_momentum >= 2:
        reasons.append("Role is trending upward")

    elif opp_momentum <= -2:
        reasons.append("Role is trending downward")

    if trend_gap >= 2:
        reasons.append(
            "Opportunity rising faster than production"
        )

    if production_momentum >= 2:
        reasons.append(
            "Recent production is improving"
        )

    elif production_momentum <= -2:
        reasons.append(
            "Recent production is declining"
        )

    if td_dependency >= 0.55:
        reasons.append(
            "Production has significant TD dependence"
        )

    if not reasons:
        reasons.append("Projection primarily follows recent baseline")

    return reasons[:4]


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["why"] = df.apply(
        generate_why,
        axis=1,
    )

    return df


# ============================================================
# WAIVER WIRE
# ============================================================


def build_waiver_pool(
    projections: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:

    if projections.empty:
        return projections

    roster_ids = set(
        roster["nflverse_player_id"]
        .astype(str)
        .replace("", np.nan)
        .dropna()
    )

    waiver = projections[
        ~projections["player_id"]
        .astype(str)
        .isin(roster_ids)
    ].copy()

    waiver = waiver[
        waiver["position"].isin(
            OFFENSIVE_POSITIONS
        )
    ]

    # Waiver priority isn't a "ranking" of all players.
    # It's a discovery score designed around upside.
    projection_values = waiver.get(
        "projection",
        waiver["baseline_projection"],
    )
    waiver["waiver_discovery_score"] = (
        projection_values.fillna(0)
        + np.clip(
            waiver["opportunity_momentum"].fillna(0),
            -3,
            5,
        )
        + np.clip(
            waiver["trend_gap"].fillna(0),
            -2,
            5,
        )
    )

    return waiver.sort_values(
        "waiver_discovery_score",
        ascending=False,
    )


# ============================================================
# MICRO SPARKLINES
# ============================================================


def sparkline(
    values: Iterable,
    width: int = 86,
    height: int = 18,
    prior_season_points: int = 0,
) -> str:

    vals = []

    for value in values:
        try:
            value = float(value)

            if np.isfinite(value):
                vals.append(value)
        except Exception:
            pass

    if len(vals) < 2:
        return ""

    low = min(vals)
    high = max(vals)

    if high == low:
        high = low + 1

    points = []
    vertical_padding = 3.0

    for i, value in enumerate(vals):
        x = (
            i
            * width
            / max(1, len(vals) - 1)
        )

        y = vertical_padding + (height - 2 * vertical_padding) * (1 - (
            (value - low)
            / (high - low)
        ))

        points.append((x, y))

    # Convert the data points into a gentle Catmull-Rom-style cubic Bézier
    # path. It passes through every observed point while avoiding the hard
    # corners of an SVG polyline.
    paths = []
    for i in range(len(points) - 1):
        previous = points[max(0, i - 1)]
        start = points[i]
        end = points[i + 1]
        following = points[min(len(points) - 1, i + 2)]
        control_1 = (
            start[0] + (end[0] - previous[0]) / 6,
            start[1] + (end[1] - previous[1]) / 6,
        )
        control_2 = (
            end[0] - (following[0] - start[0]) / 6,
            end[1] - (following[1] - start[1]) / 6,
        )
        path = (
            f"M {start[0]:.1f} {start[1]:.1f} "
            f"C {control_1[0]:.1f} {control_1[1]:.1f}, "
            f"{control_2[0]:.1f} {control_2[1]:.1f}, "
            f"{end[0]:.1f} {end[1]:.1f}"
        )
        stroke = "#8792a6" if i < prior_season_points - 1 else "currentColor"
        paths.append(
            f'<path d="{path}" fill="none" stroke="{stroke}" '
            f'stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )

    return (
        f'<svg class="spark" '
        f'viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none">'
        f'{"".join(paths)}'
        f'</svg>'
    )


def player_sparklines(
    history: pd.DataFrame,
    player_id: str,
    current_season: int,
) -> Tuple[str, str]:

    if history.empty:
        return "", ""

    h = history[
        history["player_id"].astype(str)
        == str(player_id)
    ].sort_values("week").tail(6)

    if h.empty:
        return "", ""

    production = h["fantasy_points_std"].tolist()

    source_season_values = h.get(
        "source_season",
        h.get("season", pd.Series(current_season, index=h.index)),
    )
    source_seasons = pd.to_numeric(
        source_season_values,
        errors="coerce",
    ).fillna(current_season)
    prior_season_points = int((source_seasons < current_season).sum())

    position = clean_text(h.get("position", pd.Series("", index=h.index)).iloc[-1]).upper()
    targets = h.get("targets", pd.Series(0, index=h.index)).fillna(0)
    carries = h.get("carries", pd.Series(0, index=h.index)).fillna(0)
    snap_share = h.get(
        "offense_pct",
        pd.Series(0, index=h.index),
    ).fillna(0)

    if position == "QB":
        # QB opportunity is driven by passing volume and rushing involvement,
        # not targets, which are not a quarterback opportunity measure.
        passing_attempts = h.get(
            "passing_attempts",
            pd.Series(0, index=h.index),
        ).fillna(0)
        opportunity = (
            passing_attempts
            + 0.50 * carries
            + 0.03 * snap_share
        ).tolist()
    elif position == "RB":
        opportunity = (
            carries
            + 0.75 * targets
            + 0.03 * snap_share
        ).tolist()
    else:
        # WR/TE opportunity prioritizes targets, with carries as a smaller
        # supplementary component for designed touches.
        opportunity = (
            targets
            + 0.55 * carries
            + 0.03 * snap_share
        ).tolist()

    prod = sparkline(production, prior_season_points=prior_season_points)
    opp = sparkline(opportunity, prior_season_points=prior_season_points)

    return prod, opp


# ============================================================
# HTML DASHBOARD
# ============================================================


def html_escape(value) -> str:
    return html.escape(clean_text(value))


def fmt(value, decimals=1):
    try:
        if not np.isfinite(float(value)):
            return "—"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def signal_tooltip(row: pd.Series) -> str:
    """Explain the row's signal with its current component values."""
    signal = clean_text(row.get("classification", "Stable"))
    opp = safe_float(row.get("opportunity_momentum"), 0)
    production = safe_float(row.get("production_momentum"), 0)
    gap = safe_float(row.get("trend_gap"), 0)
    td_dependency = safe_float(row.get("td_dependency"), 0) * 100

    definitions = {
        "Emerging opportunity": (
            "Opportunity is rising faster than production."
        ),
        "TD-dependent production": (
            "Recent scoring relies heavily on touchdowns."
        ),
        "Usage decline": (
            "Both opportunity and production are declining."
        ),
        "Production + usage rising": (
            "Both opportunity and production are improving."
        ),
        "Efficiency / regression watch": (
            "Production is rising while opportunity is falling."
        ),
        "Stable": "No material positive or negative trend signal.",
    }

    return (
        f"{signal}: {definitions.get(signal, definitions['Stable'])} "
        f"Opportunity momentum = {opp:.1f}; production momentum = "
        f"{production:.1f}; trend gap (opportunity minus production) = "
        f"{gap:.1f}; TD dependence = {td_dependency:.0f}%."
    )


def confidence_tooltip(row: pd.Series) -> str:
    games = safe_float(row.get("games_played"), 0)
    volatility = safe_float(row.get("volatility"), 0)
    opportunity = safe_float(row.get("opportunity_score"), 0)
    confidence = safe_float(row.get("confidence"), np.nan)

    if games < 3:
        return (
            f"Early season: only {games:.0f} included games. Confidence is "
            "withheld until at least three games are available."
        )

    consistency = np.clip(12 - volatility, -10, 10)
    opportunity_bonus = np.clip(opportunity, 0, 15)
    raw_confidence = 50 + consistency + opportunity_bonus
    if raw_confidence > 90:
        bound_explanation = " Bound at 90%."
    elif raw_confidence < 20:
        bound_explanation = " Bound at 20%."
    else:
        bound_explanation = ""
    return (
        f"Raw confidence calculation: 50 + consistency ({consistency:.1f}) "
        f"+ opportunity ({opportunity_bonus:.1f}) = {raw_confidence:.1f}%. "
        f"{bound_explanation} "
        f"Volatility is {volatility:.1f}; opportunity score is "
        f"{opportunity:.1f}; {games:.0f} games are included."
    )


def signal_chip_class(signal: str) -> str:
    return {
        "Emerging opportunity": "chip-teal",
        "Production + usage rising": "chip-blue",
        "TD-dependent production": "chip-violet",
        "Efficiency / regression watch": "chip-rose",
        "Usage decline": "chip-gold",
        "Stable": "chip-gray",
    }.get(clean_text(signal), "chip-gray")


def signal_chips_html(row: pd.Series) -> str:
    """Render independent, readable trend chips rather than one compound label."""
    opportunity = safe_float(row.get("opportunity_momentum"), 0)
    production = safe_float(row.get("production_momentum"), 0)
    td_dependency = safe_float(row.get("td_dependency"), 0)

    chips = []
    if production > 1:
        chips.append(("Production rising", "chip-blue"))
    elif production < -1:
        chips.append(("Production falling", "chip-rose"))

    if opportunity > 1:
        chips.append(("Usage rising", "chip-teal"))
    elif opportunity < -1:
        chips.append(("Usage falling", "chip-gold"))

    if td_dependency > 0.55:
        chips.append(("TD risk", "chip-violet"))

    if not chips:
        chips.append(("Stable", "chip-gray"))

    tooltip = html.escape(signal_tooltip(row))
    chips_html = "".join(
        f'<span class="signal signal-chip {chip_class}" tabindex="0" '
        f'data-tooltip="{tooltip}">{html.escape(label)}</span>'
        for label, chip_class in chips
    )
    return f'<span class="signal-chips">{chips_html}</span>'


def build_html(
    roster: pd.DataFrame,
    projections: pd.DataFrame,
    history: pd.DataFrame,
    injury_status: pd.DataFrame,
    waiver: pd.DataFrame,
    unmatched: pd.DataFrame,
    season: int,
    target_week: int,
) -> str:

    now = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    try:
        wordmark_svg = WORDMARK_PATH.read_text(encoding="utf-8")
    except OSError:
        wordmark_svg = ""

    roster_rows = []

    fantasy_teams = sorted(
        manager
        for manager in roster["manager"].dropna().map(clean_text).unique()
        if manager
    )

    team_options = "".join(
        f'<option value="{html_escape(manager)}">'
        f'{html_escape(manager)}</option>'
        for manager in fantasy_teams
    )
    team_picker_options = "".join(
        f'<button class="team-picker-option" type="button" '
        f'data-team="{html_escape(manager)}">'
        f'{html_escape(manager)}</button>'
        for manager in fantasy_teams
    )

    roster_proj = roster.merge(
        projections[
            [
                "player_id",
                "projection",
                "baseline_projection",
                "range_low",
                "range_high",
                "confidence",
                "games_played",
                "classification",
                "opportunity_momentum",
                "production_momentum",
                "trend_gap",
                "td_dependency",
                "volatility",
                "fantasy_points_ewma",
                "fantasy_points_roll3",
                "fantasy_points_roll5",
                "opportunity_score",
                "targets_ewma",
                "carries_ewma",
                "why",
            ]
        ],
        left_on="nflverse_player_id",
        right_on="player_id",
        how="left",
    )

    recent_actual_avg = {}
    recent_actual_values = {}
    injury_by_player = {}
    if not injury_status.empty:
        injury_by_player = {
            str(row["player_id"]): row
            for _, row in injury_status.iterrows()
        }
    if not history.empty and {"player_id", "week", "fantasy_points_std"}.issubset(history.columns):
        recent_actual = (
            history
            .sort_values(["player_id", "week"])
            .groupby("player_id", group_keys=False)
            .tail(3)
            .copy()
        )
        recent_actual_avg = (
            recent_actual
            .groupby("player_id")["fantasy_points_std"]
            .mean()
            .to_dict()
        )
        recent_actual_values = {
            str(player_id): group.to_dict("records")
            for player_id, group in recent_actual.groupby("player_id")
        }

    # Keep the source order from Yahoo. It is used as the default dashboard
    # order and remains available after a visitor sorts the Roster column.
    for yahoo_order, (_, row) in enumerate(roster_proj.iterrows()):

        games_played = safe_float(row.get("games_played"), 0)
        if games_played < 3:
            confidence_display = "Early season"
        else:
            confidence_display = f"{fmt(row.get('confidence'), 0)}%"

        signal_description = html_escape(signal_tooltip(row))
        chips_html = signal_chips_html(row)
        projection = safe_float(row.get("projection"))
        if not np.isfinite(projection):
            projection = safe_float(row.get("baseline_projection"))
        range_low = safe_float(row.get("range_low"))
        range_high = safe_float(row.get("range_high"))

        if all(np.isfinite(value) for value in [
            projection,
            range_low,
            range_high,
        ]):
            projection_display = (
                f"{projection:.1f} (± {((range_high - range_low) / 2):.1f})"
            )
        else:
            projection_display = fmt(projection)

        player_id = clean_text(row.get("nflverse_player_id", ""))
        injury = injury_by_player.get(player_id)
        injury_badge = ""
        if injury is not None:
            flag = clean_text(injury.get("injury_flag", ""))
            note = html_escape(injury.get("injury_note", flag))
            badge_class = "injury-ir" if flag == "IR" else "injury-out"
            injury_badge = (
                f'<span class="injury-flag {badge_class}" tabindex="0" '
                f'data-tooltip="{note}">{html.escape(flag)}</span>'
            )
        production_spark, opportunity_spark = player_sparklines(
            history,
            player_id,
            season,
        )
        spark_tooltip = html_escape(
            "Recent six included games. Pts is actual league-scoring "
            "fantasy points (a quick consistency view); Opp is a position-aware "
            "opportunity trend. It uses passing attempts and rushing involvement "
            "for QBs, carries and targets for RBs, and targets for WRs/TEs. "
            "Gray is last-season carry-over; color is the current season."
        )
        form_display = (
            f'<span class="spark-pair" tabindex="0" '
            f'data-tooltip="{spark_tooltip}">'
            f'<span class="spark-row spark-production"><span>Pts</span>'
            f'{production_spark or "—"}</span>'
            f'<span class="spark-row spark-opportunity"><span>Opp</span>'
            f'{opportunity_spark or "—"}</span></span>'
        )
        last_three_games = recent_actual_values.get(player_id, [])
        if last_three_games:
            week_scores = []
            for game in last_three_games:
                source_week = safe_float(
                    game.get("source_week", game.get("week")),
                    np.nan,
                )
                source_season = safe_float(
                    game.get("source_season", game.get("season")),
                    np.nan,
                )
                week_label = (
                    f"Wk {source_week:.0f}"
                    if np.isfinite(source_week)
                    else "Week"
                )
                if np.isfinite(source_season) and int(source_season) != season:
                    week_label += f" ({source_season:.0f})"
                week_scores.append(
                    f"{week_label}: "
                    f"{safe_float(game.get('fantasy_points_std')):.1f}"
                )
            average_tooltip = html_escape(
                " → ".join(week_scores)
            )
        else:
            average_tooltip = "No included game scores are available yet."

        roster_rows.append(
            f"""
            <tr class="roster-player"
                 data-fantasy-team="{html_escape(row.get('manager', ''))}"
                 data-yahoo-order="{yahoo_order}">
                <td>
                    <button class="player-link" type="button"
                        data-name="{html_escape(row.get('player', ''))}"
                        data-position="{html_escape(row.get('position', ''))}"
                        data-team="{html_escape(row.get('team', ''))}"
                        data-projection="{projection_display}"
                        data-confidence="{confidence_display}"
                        data-signal="{html_escape(row.get('classification', '—'))}"
                        data-fantasy-ewma="{fmt(row.get('fantasy_points_ewma'))}"
                        data-fantasy-roll3="{fmt(row.get('fantasy_points_roll3'))}"
                        data-fantasy-roll5="{fmt(row.get('fantasy_points_roll5'))}"
                        data-opportunity="{fmt(row.get('opportunity_score'))}"
                        data-targets="{fmt(row.get('targets_ewma'))}"
                        data-carries="{fmt(row.get('carries_ewma'))}"
                        data-opp-momentum="{fmt(row.get('opportunity_momentum'))}"
                        data-production-momentum="{fmt(row.get('production_momentum'))}"
                        data-trend-gap="{fmt(row.get('trend_gap'))}"
                        data-td-dependency="{fmt(row.get('td_dependency'))}">
                        {html_escape(row.get("player", ""))}
                    </button>
                    {injury_badge}
                    <span class="player-team">
                        ({html_escape(row.get("team", ""))} —
                        {html_escape(row.get("position", ""))})
                    </span>
                </td>
                <td>{html_escape(row.get("slot", ""))}</td>
                <td class="projection-col">{projection_display}</td>
                <td>
                    <span class="hover-value" tabindex="0"
                        data-tooltip="{average_tooltip}">
                        {fmt(recent_actual_avg.get(player_id))}
                    </span>
                </td>
                <td class="trend-cell">
                    {chips_html}
                </td>
                <td class="spark-cell">{form_display}</td>
            </tr>
            """
        )

    waiver_rows = []

    for _, row in waiver.head(30).iterrows():

        signal_description = html_escape(signal_tooltip(row))
        chips_html = signal_chips_html(row)
        waiver_team = clean_text(row.get("team", "")) or "—"

        waiver_rows.append(
            f"""
            <tr>
                <td>
                    <strong>
                        {html_escape(row.get("player_name", ""))}
                    </strong>
                    <span class="player-team">
                        ({html_escape(waiver_team)} —
                        {html_escape(row.get("position", ""))})
                    </span>
                </td>

                <td>
                    {fmt(row.get("waiver_discovery_score"))}
                </td>

                <td>
                    {fmt(row.get("projection"))}
                </td>

                <td>
                    {fmt(row.get("opportunity_momentum"))}
                </td>

                <td class="trend-cell">
                    {chips_html}
                </td>
            </tr>
            """
        )

    unmatched_html = ""

    if not unmatched.empty:

        items = []

        for _, row in unmatched.iterrows():
            items.append(
                f"""
                <li>
                    <strong>
                        {html_escape(row.get("player", ""))}
                    </strong>
                    —
                    Yahoo team:
                    {html_escape(row.get("team", ""))}
                </li>
                """
            )

        unmatched_html = f"""
        <section>
            <h2>Roster Mapping Needs Review</h2>
            <p>
                These Yahoo players could not be confidently
                matched to NFLverse.
            </p>
            <ul>
                {''.join(items)}
            </ul>
        </section>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">

<title>{APP_NAME}</title>

<style>

:root {{
    color-scheme: dark;
    --bg: #090a16;
    --card: #11152a;
    --card2: #191f3a;
    --text: #e7effa;
    --muted: #98a6bd;
    --border: #2a3557;
    --accent: #8a6cff;
    --cyan: #2ee6d6;
    --magenta: #f06bb8;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background:
        radial-gradient(circle at 10% -10%, #27204f 0, transparent 35%),
        radial-gradient(circle at 95% 0%, #123d52 0, transparent 28%),
        var(--bg);
    background-attachment: fixed;
    color: var(--text);
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}

header {{
    padding: 22px 0;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
}}

header h1 {{
    margin: 0 0 5px;
    font-size: 25px;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 0;
    height: 52px;
    /* Offset the SVG's built-in left-side viewBox whitespace. */
    margin-left: -39px;
}}

.brand svg {{
    width: 300px;
    height: 52px;
}}

.brand-suffix {{
    color: var(--cyan);
    font-family: Rajdhani, "Segoe UI", sans-serif;
    /* The source SVG includes generous right-side whitespace. */
    margin-left: -47px;
    font-size: 48px;
    font-weight: 700;
    line-height: 1;
}}

header p {{
    margin: 0;
    color: var(--muted);
    font-size: 12px;
}}

.header-status {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.refresh-button {{
    padding: 5px 8px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--card2);
    color: var(--cyan);
    font: inherit;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
}}

.refresh-button:hover {{
    border-color: var(--cyan);
    background: #173141;
}}

main {{
    max-width: 1100px;
    margin: auto;
    padding: 16px;
}}

section {{
    margin-bottom: 28px;
}}

h2 {{
    font-size: 20px;
    margin-bottom: 12px;
}}

.summary {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
}}

.summary-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 15px;
}}

.summary-value {{
    font-size: 23px;
    font-weight: 700;
}}

.muted {{
    color: var(--muted);
    font-size: 13px;
}}

.player-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 11px 13px;
    margin-bottom: 8px;
}}

.player-main {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}}

.player-identity {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px 12px;
    min-width: 0;
}}

.player-name {{
    font-size: 17px;
    font-weight: 700;
}}

.projection {{
    font-size: 27px;
    font-weight: 800;
    white-space: nowrap;
}}

.metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}}

.metrics span {{
    border: 1px solid var(--border);
    background: var(--card2);
    padding: 3px 6px;
    border-radius: 8px;
    color: var(--muted);
    font-size: 12px;
}}

.why {{
    margin-top: 7px;
    color: #cdd3dc;
    font-size: 13px;
    line-height: 1.5;
}}

.section-heading {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}}

.team-filter {{
    color: var(--muted);
    font-size: 13px;
}}

.team-picker {{
    display: inline-block;
    position: relative;
    margin-left: 6px;
}}

.team-native-select {{
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
}}

.team-picker-button {{
    min-width: 126px;
    padding: 7px 30px 7px 9px;
    color: var(--text);
    background-color: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    font: inherit;
    text-align: left;
    cursor: pointer;
}}

.team-picker-button::after {{
    content: "⌄";
    position: absolute;
    right: 10px;
    color: var(--muted);
}}

.team-picker-button:hover,
.team-picker-button:focus {{
    border-color: var(--cyan);
    outline: none;
}}

.team-picker-menu {{
    position: absolute;
    top: calc(100% + 5px);
    right: 0;
    z-index: 50;
    min-width: 100%;
    max-height: 300px;
    overflow-y: auto;
    padding: 4px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 12px 28px #0009;
}}

.team-picker-option {{
    display: block;
    width: 100%;
    padding: 7px 9px;
    border: 0;
    border-radius: 5px;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 12px;
    text-align: left;
    white-space: nowrap;
    cursor: pointer;
}}

.team-picker-option:hover,
.team-picker-option:focus {{
    background: #26365d;
    color: var(--cyan);
    outline: none;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
}}

th,
td {{
    padding: 9px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    font-size: 13px;
}}

th {{
    color: var(--muted);
    white-space: nowrap;
}}

.table-wrap {{
    overflow-x: auto;
}}

.tabs {{
    display: flex;
    gap: 8px;
    margin: 4px 0 18px;
}}

.tab-button {{
    color: var(--muted);
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    cursor: pointer;
}}

.tab-button.active {{
    color: var(--text);
    background: var(--accent);
    border-color: var(--accent);
}}

.tab-panel[hidden] {{
    display: none;
}}

th[data-sort] {{
    cursor: pointer;
    user-select: none;
}}

th[data-sort]:hover {{
    color: var(--text);
}}

.signal,
.confidence {{
    cursor: help;
}}

[data-tooltip] {{
    cursor: help;
    text-decoration-line: underline;
    text-decoration-style: dotted;
    text-decoration-color: #7182a6;
    text-decoration-thickness: 1px;
    text-underline-offset: 3px;
}}

.signal-chip[data-tooltip],
.spark-pair[data-tooltip],
.injury-flag[data-tooltip] {{
    text-decoration: none;
}}

.signal-chip {{
    display: inline-block;
    justify-self: start;
    padding: 3px 7px;
    border: 1px solid currentColor;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
}}

.signal-chips {{
    display: inline-grid;
    grid-template-columns: repeat(2, max-content);
    gap: 5px;
}}

.trend-cell {{
    width: 1%;
    white-space: nowrap;
}}

table.sortable th:first-child,
table.sortable td:first-child {{
    width: 32%;
}}

.spark-cell {{
    width: 116px;
}}

.spark-pair {{
    display: inline-grid;
    gap: 2px;
    cursor: help;
}}

.spark-row {{
    display: grid;
    grid-template-columns: 23px 86px;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
}}

.spark {{
    display: block;
    width: 86px;
    height: 18px;
}}

.spark-production {{ color: #69a9ff; }}
.spark-opportunity {{ color: #31e6c5; }}

.chip-gold {{ color: #f4c95d; background: #f4c95d1c; }}
.chip-teal {{ color: #31e6c5; background: #31e6c51c; }}
.chip-blue {{ color: #69a9ff; background: #69a9ff1c; }}
.chip-rose {{ color: #f58bc5; background: #f58bc51c; }}
.chip-violet {{ color: #ba9cff; background: #ba9cff1c; }}
.chip-gray {{ color: #aab4c5; background: #aab4c51c; }}

.player-team {{
    display: block;
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
}}

.injury-flag {{
    display: inline-flex;
    width: 16px;
    height: 16px;
    align-items: center;
    justify-content: center;
    margin-left: 4px;
    border-radius: 3px;
    color: #0b1020;
    font-size: 9px;
    font-weight: 900;
    line-height: 1;
    text-decoration: none;
    vertical-align: middle;
}}

.injury-out {{ background: #f06b78; }}
.injury-ir {{ background: #f4c95d; }}

.projection-col,
.confidence-col {{
    white-space: nowrap;
}}

#hover-tooltip {{
    position: fixed;
    z-index: 100;
    width: min(340px, calc(100vw - 24px));
    padding: 9px 10px;
    border: 1px solid #52658e;
    border-radius: 8px;
    background: #101a31;
    color: var(--text);
    box-shadow: 0 12px 28px #0009;
    font-size: 12px;
    line-height: 1.4;
    pointer-events: none;
}}

.player-link {{
    padding: 0;
    border: 0;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-weight: 700;
    text-align: left;
    white-space: nowrap;
    cursor: pointer;
}}

.player-link:hover {{
    color: var(--cyan);
}}

dialog {{
    width: min(620px, calc(100vw - 28px));
    border: 1px solid #52658e;
    border-radius: 14px;
    background: #111a32;
    color: var(--text);
    box-shadow: 0 24px 70px #000c;
    padding: 20px;
}}

dialog::backdrop {{
    background: #02040bb8;
}}

.detail-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 14px;
}}

.detail-grid div {{
    padding: 9px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--card2);
}}

.detail-grid span {{
    display: block;
    color: var(--muted);
    font-size: 11px;
}}

.dialog-close {{
    float: right;
    color: var(--text);
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 5px 9px;
    cursor: pointer;
}}

.signal-guide {{
    margin-top: 16px;
    padding: 12px;
    border: 1px solid var(--border);
    background: var(--card);
    border-radius: 10px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}}

.warning {{
    background: #271e12;
    border: 1px solid #584020;
    border-radius: 12px;
    padding: 15px;
}}

footer {{
    color: var(--muted);
    font-size: 12px;
    padding: 20px 0;
}}

@media (max-width: 600px) {{

    main {{
        padding: 11px;
    }}

    header {{
        padding: 18px 0;
    }}

    .projection {{
        font-size: 23px;
    }}

    th,
    td {{
        padding: 7px;
        font-size: 12px;
    }}

    /* Keep the final trend view on-screen while the wide roster table scrolls. */
    .table-wrap {{
        -webkit-overflow-scrolling: touch;
    }}

    .roster-table th.spark-col,
    .roster-table td.spark-cell {{
        position: sticky;
        right: 0;
        background: var(--card);
        box-shadow: -8px 0 12px rgba(5, 8, 20, 0.5);
    }}

    .roster-table th.spark-col {{
        z-index: 3;
    }}

    .roster-table td.spark-cell {{
        z-index: 2;
    }}

}}

</style>
</head>

<body>

<header>
    <h1 class="brand" aria-label="{APP_NAME}">
        {wordmark_svg}
        <span class="brand-suffix">2</span>
    </h1>
    <div class="header-status">
        <p>
            {season} · Week {target_week} ·
            League scoring · Generated {now}
        </p>
        <button class="refresh-button" type="button" id="refresh-dashboard">
            Load latest
        </button>
    </div>
</header>

<main>

<div class="tabs" role="tablist">
    <button class="tab-button active" data-tab="roster-panel">Roster</button>
    <button class="tab-button" data-tab="waivers-panel">Waivers</button>
    <button class="tab-button" data-tab="guide-panel">Model guide</button>
</div>

<section id="roster-panel" class="tab-panel">
    <div class="section-heading">
        <h2 id="roster-title">All teams' rosters</h2>

        <label class="team-filter" for="fantasy-team-filter">
            Fantasy team
            <span class="team-picker" id="team-picker">
                <select class="team-native-select" id="fantasy-team-filter" tabindex="-1" aria-hidden="true">
                    <option value="">All teams</option>
                    {team_options}
                </select>
                <button class="team-picker-button" type="button"
                    id="team-picker-button" aria-haspopup="listbox"
                    aria-expanded="false">All teams</button>
                <span class="team-picker-menu" id="team-picker-menu"
                    role="listbox" hidden>
                    <button class="team-picker-option" type="button" data-team="">All teams</button>
                    {team_picker_options}
                </span>
            </span>
        </label>
    </div>

    <p class="muted">
        <strong>Early-season context:</strong> during the first three weeks,
        each player's final three recorded games from last season are included
        in rolling history until enough current-season games are available.
    </p>

    <div class="table-wrap">
        <table class="sortable roster-table">
            <thead>
                <tr>
                    <th data-sort="text">Player</th>
                    <th data-sort="roster" data-direction="yahoo"
                        title="Click to cycle: Yahoo roster order, ascending, descending">Roster</th>
                    <th class="projection-col" data-sort="number" title="Fantasy-point projection; the value in parentheses is the player-specific plus/minus estimate">Proj. Pts.</th>
                    <th data-sort="number" title="Average actual league-scoring fantasy points over the latest three included games">3-Wk Avg</th>
                    <th data-sort="text">Trend signals</th>
                    <th class="spark-col" title="Recent six included games: actual fantasy points for consistency and weighted opportunity for role trend">6-Wk Trend</th>
                </tr>
            </thead>
            <tbody>
                {''.join(roster_rows)}
            </tbody>
        </table>
    </div>

    <div class="signal-guide">
        <strong>Signal guide — 6 signals.</strong><br>
        Emerging opportunity: opportunity momentum &gt; 2 and trend gap &gt; 1.<br>
        TD-dependent production: TD dependence &gt; 55% with positive production momentum.<br>
        Usage decline: opportunity and production momentum both below −1.<br>
        Production + usage rising: both momentum measures above 1.<br>
        Efficiency / regression watch: production momentum above 1 while opportunity momentum is below −1.<br>
        Stable: none of the conditions above. Hover a row's signal for its actual component values.
    </div>
</section>

<section id="waivers-panel" class="tab-panel" hidden>
    <h2>Waiver Wire — Discovery View</h2>

    <p class="muted">
        <strong>WDS (Waiver Discovery Score)</strong> = projected points +
        opportunity momentum + trend gap. The list is sorted by WDS by
        default: a higher score highlights stronger projected production and
        a possible role-growth case; it is not a complete player-value ranking.
    </p>

    <div class="table-wrap">
        <table class="sortable waiver-table">
            <thead>
                <tr>
                    <th data-sort="text">Player</th>
                    <th data-sort="number" title="Waiver Discovery Score: projected points plus capped opportunity momentum and trend gap">WDS</th>
                    <th data-sort="number">Proj. Pts.</th>
                    <th data-sort="number">Opp trend</th>
                    <th data-sort="text">Trend signals</th>
                </tr>
            </thead>

            <tbody>
                {''.join(waiver_rows)}
            </tbody>
        </table>
    </div>
</section>

<section id="guide-panel" class="tab-panel" hidden>
    <h2>Model guide</h2>

    <div class="signal-guide">
        <strong>Machine Learned-weight projection</strong><br>
        When trained weights are available, Proj. Pts. uses a transparent regularized model trained on historical NFLverse player-weeks instead of the hand-set baseline mix. It learns how much each pre-game metric should raise or lower the next-week projection, including separate QB/RB/WR/TE adjustments. Run <code>python gptindex.py --learn-weights</code> to train or refresh it. The saved weight report ranks the learned metric weights; a future direct-ML projection will appear separately for comparison.<br><br>
        <strong>Scoring and projections</strong><br>
        Core league scoring: passing yards ÷ 25, passing touchdowns × 6, interceptions × −2, rushing/receiving yards ÷ 10, rushing/receiving touchdowns × 6, receptions × 0, fumbles lost × −2, and two-point conversions × 2. Return yards score 1 point per 25 yards, and 40+ yard passing/rushing/receiving touchdowns receive +1. Those return and 40+ touchdown details require play-by-play data and are not yet included in the weekly-player projection inputs.<br><br>
        <strong>Baseline projection</strong><br>
        45% exponentially weighted fantasy-point average + 25% three-game rolling average + 15% five-game rolling average + 15% season weighted average, then a capped opportunity-momentum adjustment and TD-dependence penalty.<br><br>
        <strong>Carry-over history</strong><br>
        The last three games from the prior season are included before Week 1 so established players have useful rolling history. Current-season games remain the newest and most important observations.<br><br>
        <strong>3-Wk Avg and recent form</strong><br>
        3-Wk Avg is the average of actual fantasy points from a player's latest three included games. The compact Pts and Opp sparklines show the latest six included games: Pts is actual fantasy scoring for consistency, while Opp is position-aware opportunity — passing attempts and rushing involvement for QBs, carries and targets for RBs, and targets for WRs/TEs.<br><br>
        <strong>Opportunity score</strong><br>
        Targets × 1.00 + carries × 0.55 + offensive snap percentage × 0.06 + red-zone targets × 1.40 + red-zone carries × 1.10. It is a role/usage measure, not a fantasy-point projection.<br><br>
        <strong>Momentum</strong><br>
        Recent two-game average minus the preceding two-game average. Opportunity momentum uses the opportunity components; production momentum uses fantasy points.<br><br>
        <strong>Trend gap</strong><br>
        Opportunity momentum − production momentum. A positive value means role growth is outpacing scoring; a negative value means scoring is outpacing role growth.<br><br>
        <strong>TD dependence</strong><br>
        Touchdown fantasy points ÷ total fantasy points, using the exponentially weighted averages. Higher values imply greater risk of regression when touchdowns slow down.<br><br>
        <strong>Projection (±)</strong><br>
        Weekly fantasy-point projection followed by a player-specific plus/minus estimate. The estimate is 1.15 × the model's recent-volatility measure; it is a planning guide, not a guarantee or formal confidence interval.<br><br>
        <strong>Confidence</strong><br>
        An internal 20–90 consistency score, not a probability. It starts at 50, then adds capped consistency (12 − volatility) and capped opportunity; it is shown in player details rather than the main table and is withheld as “Early season” until at least three included games are available.
    </div>

    <div class="signal-guide">
        <strong>Model philosophy</strong><br>
        <strong>Production</strong> — recent fantasy scoring is the foundation.<br>
        <strong>Opportunity</strong> — targets, carries, snap share, and red-zone work can reveal changes before fantasy production catches up.<br>
        <strong>Sustainability</strong> — touchdown dependence and production relative to opportunity flag possible regression.<br>
        <strong>Leakage protection</strong> — features use only games before the week being projected.
    </div>
</section>

{unmatched_html}

<footer>
    Fantasy Command Center beta v1
</footer>

</main>

<div id="hover-tooltip" hidden></div>

<dialog id="player-detail">
    <button class="dialog-close" type="button" id="detail-close">Close</button>
    <h2 id="detail-name"></h2>
    <p class="muted" id="detail-subtitle"></p>
    <div class="detail-grid" id="detail-grid"></div>
</dialog>

<script>
const refreshButton = document.getElementById("refresh-dashboard");
refreshButton.addEventListener("click", function () {{
    const freshUrl = new URL(window.location.href);
    freshUrl.searchParams.set("v", Date.now().toString());
    window.location.replace(freshUrl.toString());
}});

const teamSelect = document.getElementById("fantasy-team-filter");
const rosterTitle = document.getElementById("roster-title");
const teamPicker = document.getElementById("team-picker");
const teamPickerButton = document.getElementById("team-picker-button");
const teamPickerMenu = document.getElementById("team-picker-menu");

function closeTeamPicker() {{
    teamPickerMenu.hidden = true;
    teamPickerButton.setAttribute("aria-expanded", "false");
}}

teamPickerButton.addEventListener("click", function () {{
    const willOpen = teamPickerMenu.hidden;
    teamPickerMenu.hidden = !willOpen;
    teamPickerButton.setAttribute("aria-expanded", String(willOpen));
}});

document.querySelectorAll(".team-picker-option").forEach(function (option) {{
    option.addEventListener("click", function () {{
        teamSelect.value = option.dataset.team;
        teamPickerButton.textContent = option.textContent;
        closeTeamPicker();
        teamSelect.dispatchEvent(new Event("change"));
    }});
}});

document.addEventListener("click", function (event) {{
    if (!teamPicker.contains(event.target)) closeTeamPicker();
}});

document.addEventListener("keydown", function (event) {{
    if (event.key === "Escape") closeTeamPicker();
}});

teamSelect.addEventListener("change", function () {{
    const selected = this.value;
    rosterTitle.textContent = selected ? `${{selected}}'s roster` : "All teams' rosters";

    document.querySelectorAll(".roster-player[data-fantasy-team]").forEach(
        function (card) {{
            card.hidden = selected !== "" &&
                card.dataset.fantasyTeam !== selected;
        }}
    );
}});

const hoverTooltip = document.getElementById("hover-tooltip");

function positionTooltip(element) {{
    const rect = element.getBoundingClientRect();
    const left = Math.max(12, Math.min(rect.left, window.innerWidth - 352));
    const top = Math.max(12, Math.min(rect.bottom + 8, window.innerHeight - 140));
    hoverTooltip.style.left = `${{left}}px`;
    hoverTooltip.style.top = `${{top}}px`;
}}

document.querySelectorAll("[data-tooltip]").forEach(function (element) {{
    function show() {{
        hoverTooltip.textContent = element.dataset.tooltip;
        hoverTooltip.hidden = false;
        positionTooltip(element);
    }}
    function hide() {{ hoverTooltip.hidden = true; }}

    element.addEventListener("mouseenter", show);
    element.addEventListener("mousemove", function () {{ positionTooltip(element); }});
    element.addEventListener("mouseleave", hide);
    element.addEventListener("focus", show);
    element.addEventListener("blur", hide);
}});

document.querySelectorAll(".tab-button").forEach(function (button) {{
    button.addEventListener("click", function () {{
        const target = this.dataset.tab;

        document.querySelectorAll(".tab-panel").forEach(function (panel) {{
            panel.hidden = panel.id !== target;
        }});

        document.querySelectorAll(".tab-button").forEach(function (tab) {{
            tab.classList.toggle("active", tab === button);
        }});
    }});
}});

document.querySelectorAll("table.sortable th[data-sort]").forEach(
    function (header) {{
        header.addEventListener("click", function () {{
            const table = header.closest("table");
            const body = table.tBodies[0];
            const index = Array.prototype.indexOf.call(
                header.parentElement.children,
                header
            );
            const type = header.dataset.sort;
            const rows = Array.from(body.rows);
            const rosterMode = type === "roster";
            const currentMode = header.dataset.direction || "yahoo";
            const nextMode = rosterMode
                ? (currentMode === "yahoo" ? "asc" :
                    currentMode === "asc" ? "desc" : "yahoo")
                : (currentMode === "asc" ? "desc" : "asc");
            const direction = nextMode === "desc" ? -1 : 1;

            function value(row) {{
                if (rosterMode && nextMode === "yahoo") {{
                    return Number(row.dataset.yahooOrder || 0);
                }}
                const text = row.cells[index].innerText.trim();
                if (type === "number") {{
                    const match = text.match(/-?[0-9]+([.][0-9]+)?/);
                    const number = match ? parseFloat(match[0]) : NaN;
                    return Number.isFinite(number) ? number : -Infinity;
                }}
                if (type === "range") {{
                    const number = parseFloat(text);
                    return Number.isFinite(number) ? number : -Infinity;
                }}
                return text.toLocaleLowerCase();
            }}

            rows.sort(function (left, right) {{
                const a = value(left);
                const b = value(right);
                if (a < b) return -1 * direction;
                if (a > b) return 1 * direction;
                return 0;
            }});

            rows.forEach(function (row) {{ body.appendChild(row); }});
            table.querySelectorAll("th[data-sort]").forEach(function (cell) {{
                cell.dataset.direction = "";
            }});
            header.dataset.direction = nextMode;
        }});
    }}
);

const detailDialog = document.getElementById("player-detail");
const detailName = document.getElementById("detail-name");
const detailSubtitle = document.getElementById("detail-subtitle");
const detailGrid = document.getElementById("detail-grid");

document.querySelectorAll(".player-link").forEach(function (button) {{
    button.addEventListener("click", function () {{
        const data = button.dataset;
        detailName.textContent = data.name;
        detailSubtitle.textContent =
            `${{data.position}} · ${{data.team}} · ${{data.signal}}`;

        const metrics = [
            ["Projection", data.projection],
            ["Confidence", data.confidence],
            ["Fantasy EWMA", data.fantasyEwma],
            ["Fantasy rolling 3", data.fantasyRoll3],
            ["Fantasy rolling 5", data.fantasyRoll5],
            ["Opportunity score", data.opportunity],
            ["Targets EWMA", data.targets],
            ["Carries EWMA", data.carries],
            ["Opportunity momentum", data.oppMomentum],
            ["Production momentum", data.productionMomentum],
            ["Trend gap", data.trendGap],
            ["TD dependence", data.tdDependency],
        ];

        detailGrid.replaceChildren();
        metrics.forEach(function ([label, value]) {{
            const item = document.createElement("div");
            const caption = document.createElement("span");
            caption.textContent = label;
            item.append(caption, document.createTextNode(value || "—"));
            detailGrid.append(item);
        }});

        detailDialog.showModal();
    }});
}});

document.getElementById("detail-close").addEventListener(
    "click",
    function () {{ detailDialog.close(); }}
);
</script>

</body>
</html>
"""


# ============================================================
# CSV EXPORTS
# ============================================================


def export_csvs(
    roster: pd.DataFrame,
    projections: pd.DataFrame,
    waiver: pd.DataFrame,
) -> None:

    roster.to_csv(
        OUTPUT_DIR / "roster_mapping.csv",
        index=False,
    )

    projections.to_csv(
        OUTPUT_DIR / "projections.csv",
        index=False,
    )

    waiver.to_csv(
        OUTPUT_DIR / "waiver_candidates.csv",
        index=False,
    )


# ============================================================
# OPTIONAL MACHINE LEARNING
# ============================================================


# These are deliberately limited to pre-game, player-level metrics. The
# learned-weight model is a transparent, regularized linear model: unlike a
# tree model, its coefficients can be shown and audited in the dashboard.
LEARNED_WEIGHT_FEATURES = [
    "fantasy_points_last",
    "fantasy_points_roll2",
    "fantasy_points_roll3",
    "fantasy_points_roll5",
    "fantasy_points_ewma",
    "targets_last",
    "targets_roll2",
    "targets_roll3",
    "targets_roll5",
    "targets_ewma",
    "carries_last",
    "carries_roll2",
    "carries_roll3",
    "carries_roll5",
    "carries_ewma",
    "receiving_yards_ewma",
    "rushing_yards_ewma",
    "offense_snaps_ewma",
    "offense_pct_ewma",
    "td_points_ewma",
    "opportunity_score",
    "opportunity_momentum",
    "production_momentum",
    "trend_gap",
    "td_dependency",
]


def train_learned_weight_model(
    train_df: pd.DataFrame,
) -> Tuple[Optional[Dict], pd.DataFrame]:
    """Fit auditable ridge-regression weights from historical player-weeks."""

    if train_df.empty or "target_next_week" not in train_df:
        return None, pd.DataFrame()

    features = [
        col for col in LEARNED_WEIGHT_FEATURES
        if col in train_df.columns
    ]

    train = train_df[
        train_df["target_next_week"].notna()
    ].copy()

    if len(features) < 5 or len(train) < 500:
        print("Not enough historical examples to learn metric weights.")
        return None, pd.DataFrame()

    raw_features = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    means = raw_features.mean()
    stds = raw_features.std(ddof=0).replace(0, 1.0).fillna(1.0)
    standardized = ((raw_features - means) / stds).to_numpy(dtype=float)

    # Position effects let the model establish separate QB/RB/WR/TE scoring
    # baselines without obscuring the numeric metric weights.
    train_positions = (
        train.get("position", pd.Series("UNK", index=train.index))
        .fillna("UNK")
        .astype(str)
        .str.upper()
    )
    positions = sorted(
        position for position in train_positions.unique()
        if position in OFFENSIVE_POSITIONS
    )
    if not positions:
        positions = sorted(OFFENSIVE_POSITIONS)

    position_matrix = np.column_stack([
        (train_positions == position).astype(float).to_numpy()
        for position in positions
    ])
    design = np.column_stack([
        np.ones(len(train)),
        standardized,
        position_matrix,
    ])
    target = train["target_next_week"].astype(float).to_numpy()

    # Ridge regularization stabilizes weights for overlapping metrics such as
    # rolling averages and EWMA values. The intercept is intentionally not
    # regularized.
    penalty = np.eye(design.shape[1]) * 12.0
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            design.T @ design + penalty,
            design.T @ target,
            rcond=None,
        )[0]

    feature_weights = coefficients[1:1 + len(features)]
    position_weights = coefficients[1 + len(features):]
    report = pd.DataFrame({
        "kind": "metric",
        "name": features,
        "learned_standardized_weight": feature_weights,
    })
    position_report = pd.DataFrame({
        "kind": "position_adjustment",
        "name": positions,
        "learned_standardized_weight": position_weights,
    })
    report = pd.concat([report, position_report], ignore_index=True)
    report["absolute_weight"] = report[
        "learned_standardized_weight"
    ].abs()
    report["direction"] = np.where(
        report["learned_standardized_weight"] >= 0,
        "raises projection",
        "lowers projection",
    )
    report = report.sort_values("absolute_weight", ascending=False)

    model = {
        "version": 1,
        "model_type": "ridge_regression",
        "training_examples": int(len(train)),
        "features": features,
        "feature_means": {key: float(value) for key, value in means.items()},
        "feature_stds": {key: float(value) for key, value in stds.items()},
        "feature_weights": {
            feature: float(weight)
            for feature, weight in zip(features, feature_weights)
        },
        "intercept": float(coefficients[0]),
        "position_weights": {
            position: float(weight)
            for position, weight in zip(positions, position_weights)
        },
    }
    return model, report


def save_learned_weight_model(model: Dict) -> None:
    LEARNED_WEIGHT_MODEL_PATH.write_text(
        json.dumps(model, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_learned_weight_model() -> Optional[Dict]:
    if not LEARNED_WEIGHT_MODEL_PATH.exists():
        return None
    try:
        model = json.loads(
            LEARNED_WEIGHT_MODEL_PATH.read_text(encoding="utf-8")
        )
        required = {
            "features", "feature_means", "feature_stds",
            "feature_weights", "intercept", "position_weights",
        }
        return model if required.issubset(model) else None
    except (OSError, ValueError, TypeError):
        return None


def apply_learned_weight_model(
    model: Dict,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a saved learned-weight model to the current player features."""

    if current_df.empty:
        return current_df

    features = model["features"]
    current = current_df.copy()
    raw_features = pd.DataFrame({
        feature: pd.to_numeric(
            current.get(feature, pd.Series(0.0, index=current.index)),
            errors="coerce",
        )
        for feature in features
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    standardized = np.column_stack([
        (raw_features[feature].to_numpy(dtype=float)
         - float(model["feature_means"][feature]))
        / max(float(model["feature_stds"][feature]), 1e-9)
        for feature in features
    ])
    weights = np.array([
        float(model["feature_weights"][feature])
        for feature in features
    ])
    prediction = float(model["intercept"]) + standardized @ weights

    positions = (
        current.get("position", pd.Series("UNK", index=current.index))
        .fillna("UNK")
        .astype(str)
        .str.upper()
    )
    for position, weight in model["position_weights"].items():
        prediction += (positions == position).to_numpy() * float(weight)

    current["learned_weight_projection"] = np.maximum(0.0, prediction)
    return current


def try_ml_projection(
    train_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:

    """
    Optional XGBoost model.

    It is deliberately optional for beta v1.

    If xgboost isn't installed, baseline projections continue
    working.
    """

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print(
            "\nXGBoost not installed — using baseline model."
        )
        print(
            "Optional: pip install xgboost"
        )
        return None

    if train_df.empty or current_df.empty:
        return None

    feature_candidates = [
        "fantasy_points_last",
        "fantasy_points_roll2",
        "fantasy_points_roll3",
        "fantasy_points_roll5",
        "fantasy_points_ewma",

        "targets_last",
        "targets_roll2",
        "targets_roll3",
        "targets_roll5",
        "targets_ewma",

        "carries_last",
        "carries_roll2",
        "carries_roll3",
        "carries_roll5",
        "carries_ewma",

        "receiving_yards_ewma",
        "rushing_yards_ewma",

        "offense_snaps_ewma",
        "offense_pct_ewma",

        "td_points_ewma",

        "opportunity_score",
        "opportunity_momentum",
        "production_momentum",
        "trend_gap",
        "td_dependency",
    ]

    features = [
        col
        for col in feature_candidates
        if col in train_df.columns
        and col in current_df.columns
    ]

    if len(features) < 5:
        return None

    train = train_df.copy()
    current = current_df.copy()

    train = train[
        train["target_next_week"].notna()
    ].copy()

    if len(train) < 500:
        print(
            "Not enough training examples for ML yet."
        )
        return None

    X = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = train["target_next_week"]

    model = XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X, y)

    X_current = (
        current[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    current["ml_projection"] = model.predict(
        X_current
    )

    return current


# ============================================================
# BUILD HISTORICAL TRAINING SET
# ============================================================


def build_walk_forward_training(
    seasons: List[int],
) -> pd.DataFrame:

    """
    Build leakage-safe player-week examples.

    For each player-week:

        features = weeks BEFORE target week
        target   = fantasy points in target week

    Example:

        Week 5 prediction
        uses Weeks 1-4 only.

    """

    all_examples = []

    for season in seasons:

        print(
            f"Building historical training data: {season}"
        )

        stats = load_cached(
            f"player_stats_{season}",
            lambda s=season:
                nfl.load_player_stats(
                    s,
                    summary_level="week",
                ),
        )

        stats = standardize_player_stats(stats)

        if stats.empty:
            continue

        snaps = load_snaps(season)

        stats = merge_usage(
            stats,
            snaps,
        )

        # Need at least several weeks to create useful
        # historical examples.
        weeks = sorted(
            pd.to_numeric(
                stats["week"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )

        for target_week in weeks:

            if target_week <= 1:
                continue

            previous = stats[
                stats["week"] < target_week
            ].copy()

            target = stats[
                stats["week"] == target_week
            ][
                [
                    "player_id",
                    "fantasy_points_std",
                ]
            ].copy()

            if previous.empty or target.empty:
                continue

            features = add_player_features(
                previous,
                target_week,
            )

            if features.empty:
                continue

            features = features.merge(
                target,
                on="player_id",
                how="inner",
                suffixes=(
                    "",
                    "_target",
                ),
            )

            features["target_next_week"] = (
                features["fantasy_points_std_target"]
            )

            features["season"] = season
            features["target_week"] = target_week

            all_examples.append(features)

    if not all_examples:
        return pd.DataFrame()

    return pd.concat(
        all_examples,
        ignore_index=True,
    )


# ============================================================
# BACKTEST
# ============================================================


def run_backtest(
    seasons: List[int],
) -> pd.DataFrame:

    """
    Evaluate the baseline projection against historical
    next-week outcomes.

    This is intentionally simple in beta v1.
    """

    results = []

    for season in seasons:

        print(
            f"Backtesting baseline: {season}"
        )

        stats = load_cached(
            f"player_stats_{season}",
            lambda s=season:
                nfl.load_player_stats(
                    s,
                    summary_level="week",
                ),
        )

        stats = standardize_player_stats(stats)

        if stats.empty:
            continue

        snaps = load_snaps(season)

        stats = merge_usage(
            stats,
            snaps,
        )

        weeks = sorted(
            pd.to_numeric(
                stats["week"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )

        for target_week in weeks:

            if target_week <= 1:
                continue

            previous = stats[
                stats["week"] < target_week
            ]

            target = stats[
                stats["week"] == target_week
            ][
                [
                    "player_id",
                    "fantasy_points_std",
                ]
            ]

            features = add_player_features(
                previous,
                target_week,
            )

            if features.empty:
                continue

            projections = project_baseline(
                features
            )

            merged = projections.merge(
                target,
                on="player_id",
                how="inner",
                suffixes=(
                    "",
                    "_actual",
                ),
            )

            if merged.empty:
                continue

            merged["error"] = (
                merged["baseline_projection"]
                - merged["fantasy_points_std_actual"]
            )

            merged["abs_error"] = (
                merged["error"].abs()
            )

            merged["within_3"] = (
                merged["abs_error"] <= 3
            )

            merged["within_5"] = (
                merged["abs_error"] <= 5
            )

            merged["season"] = season
            merged["target_week"] = target_week

            results.append(
                merged[
                    [
                        "season",
                        "target_week",
                        "player_id",
                        "baseline_projection",
                        "fantasy_points_std_actual",
                        "error",
                        "abs_error",
                        "within_3",
                        "within_5",
                    ]
                ]
            )

    if not results:
        return pd.DataFrame()

    result = pd.concat(
        results,
        ignore_index=True,
    )

    result.to_csv(
        OUTPUT_DIR / "backtest_results.csv",
        index=False,
    )

    return result


def print_backtest_summary(
    results: pd.DataFrame,
) -> None:

    if results.empty:
        print(
            "No backtest results available."
        )
        return

    mae = results["abs_error"].mean()
    median_error = results["error"].median()
    within3 = results["within_3"].mean()
    within5 = results["within_5"].mean()

    print("\n==============================")
    print("BACKTEST SUMMARY")
    print("==============================")

    print(
        f"Examples:       {len(results):,}"
    )

    print(
        f"MAE:            {mae:.2f}"
    )

    print(
        f"Median error:   {median_error:+.2f}"
    )

    print(
        f"Within ±3:      {within3:.1%}"
    )

    print(
        f"Within ±5:      {within5:.1%}"
    )


# ============================================================
# CURRENT PROJECTION PIPELINE
# ============================================================


def build_current_projections(
    season: int,
    target_week: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print(
        f"\nLoading {season} player statistics..."
    )

    stats = load_current_stats(season)

    if stats.empty:
        raise RuntimeError(
            "No player statistics were loaded."
        )

    print(
        f"Loaded {len(stats):,} player-week rows."
    )

    snaps = load_snaps(season)

    stats = merge_usage(
        stats,
        snaps,
    )
    stats["source_season"] = stats["season"]
    stats["source_week"] = stats["week"]

    # Use the final three weeks of last season as carry-over context. They
    # receive negative week numbers so they sort immediately before Week 1
    # and participate in the same rolling windows without leaking future
    # games from the current season.
    prior_stats = load_current_stats(season - 1)
    prior_tail = pd.DataFrame()

    if not prior_stats.empty:
        # Retain each player's own final three games, not merely the league's
        # final three calendar weeks. This preserves history for players who
        # missed a late regular-season or playoff game.
        prior_tail = (
            prior_stats
            .sort_values(["player_id", "week"])
            .groupby("player_id", group_keys=False)
            .tail(3)
            .copy()
        )
        prior_tail["source_season"] = prior_tail["season"]
        prior_tail["source_week"] = prior_tail["week"]
        prior_tail["week"] = (
            prior_tail.groupby("player_id").cumcount()
            - prior_tail.groupby("player_id")["player_id"].transform("size")
        )
        prior_tail["season"] = season

        print(
            f"Blending {len(prior_tail):,} player-week rows from "
            f"each player's final three games of {season - 1}."
        )

    # Only information available before the target week, plus the retained
    # prior-season carry-over context.
    historical = stats[
        stats["week"] < target_week
    ].copy()

    if not prior_tail.empty:
        historical = pd.concat(
            [prior_tail, historical],
            ignore_index=True,
            sort=False,
        )

    if historical.empty:
        raise RuntimeError(
            f"No historical games found before Week {target_week}."
        )

    print(
        f"Generating leakage-safe features "
        f"from {len(historical):,} prior rows..."
    )

    features = add_player_features(
        historical,
        target_week,
    )

    projections = project_baseline(
        features
    )

    schedule = load_schedule(season)

    projections = add_matchup_features(
        projections,
        schedule,
        target_week,
    )

    projections = add_explanations(
        projections
    )

    # Attach player display names.
    players = load_player_master(season)

    player_lookup = players[
        [
            "player_id",
            "display_name",
            "position",
            "team",
        ]
    ].drop_duplicates(
        "player_id"
    )

    projections = projections.merge(
        player_lookup,
        on="player_id",
        how="left",
        suffixes=("", "_master"),
    )

    # Prefer master data.
    projections["player_name"] = (
        projections["display_name"]
        .fillna(
            projections.get(
                "player_name",
                "",
            )
        )
    )

    projections["position"] = (
        projections["position_master"]
        .fillna(
            projections.get(
                "position",
                "",
            )
        )
    )

    master_team = projections["team_master"].replace("", np.nan)
    stat_team = projections.get("team", pd.Series("", index=projections.index))
    projections["team"] = master_team.fillna(stat_team).fillna("")

    return historical, projections


# ============================================================
# COMMAND LINE
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=APP_NAME
    )

    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help="NFL season",
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Week to project. "
            "Defaults to one week after the latest "
            "completed week in the data."
        ),
    )

    parser.add_argument(
        "--roster",
        type=str,
        default=str(ROSTER_FILE),
        help="Yahoo roster text file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(HTML_OUTPUT),
        help="HTML dashboard output",
    )

    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Build historical training set and train XGBoost",
    )

    parser.add_argument(
        "--learn-weights",
        action="store_true",
        help=(
            "Train and save transparent learned metric weights from "
            "historical NFLverse data"
        ),
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run historical baseline backtest",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached files",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():

    args = parse_args()

    ensure_directories()

    print()
    print("=" * 60)
    print(APP_NAME)
    print("Beta v1")
    print("=" * 60)

    print(
        f"Season:          {args.season}"
    )

    print(
        "Scoring:         Standard"
    )

    print(
        f"League teams:    {LEAGUE_TEAMS}"
    )

    # --------------------------------------------------------
    # Parse Yahoo roster.
    # --------------------------------------------------------

    roster_path = Path(args.roster)

    print(
        f"\nReading Yahoo roster: {roster_path}"
    )

    yahoo = parse_yahoo_roster(
        roster_path
    )

    print(
        f"Roster rows: {len(yahoo)}"
    )

    # --------------------------------------------------------
    # Player master + matching.
    # --------------------------------------------------------

    print(
        "\nLoading NFLverse player master..."
    )

    players = load_player_master(
        args.season
    )

    print(
        f"NFLverse players: {len(players):,}"
    )

    print(
        "\nMatching Yahoo players to NFLverse..."
    )

    roster = build_player_match_table(
        yahoo,
        players,
    )

    is_team_defense = roster["position"].isin(
        ["DEF", "D/ST", "DST"]
    )
    individual_roster = roster[~is_team_defense].copy()

    matched_count = (
        individual_roster["nflverse_player_id"]
        .astype(str)
        .str.len()
        .gt(0)
        .sum()
    )

    print(
        f"Matched: {matched_count}/{len(individual_roster)} "
        f"individual players"
    )

    if is_team_defense.any():
        print(
            f"Team defenses excluded from player matching: "
            f"{is_team_defense.sum()}"
        )

    unmatched = individual_roster[
        individual_roster["nflverse_player_id"]
        .astype(str)
        .str.len()
        .eq(0)
    ].copy()

    if not unmatched.empty:
        print(
            "\nWARNING:"
        )

        for _, row in unmatched.iterrows():
            print(
                f"  UNMATCHED: "
                f"{row['player']} "
                f"({row['team']})"
            )

    # --------------------------------------------------------
    # Load current season stats.
    # --------------------------------------------------------

    stats = load_current_stats(
        args.season,
        refresh=True,
    )

    if stats.empty:
        raise RuntimeError(
            "Unable to load current season stats."
        )

    # --------------------------------------------------------
    # Determine target week.
    # --------------------------------------------------------

    completed_weeks = sorted(
        pd.to_numeric(
            stats["week"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    if args.week is not None:
        target_week = args.week
    else:
        latest_week = max(
            completed_weeks
        )

        target_week = latest_week + 1

    print(
        f"\nProjection target: Week {target_week}"
    )

    print("\nLoading current injury statuses...")
    injury_status = load_current_injury_status(
        args.season,
        target_week,
    )
    print(f"Active Out/IR flags: {len(injury_status):,}")

    # --------------------------------------------------------
    # Build projections.
    # --------------------------------------------------------

    history, projections = build_current_projections(
        args.season,
        target_week,
    )

    # --------------------------------------------------------
    # Optional ML.
    # --------------------------------------------------------

    if args.train_ml:

        print(
            "\nBuilding historical ML training set..."
        )

        # Beta choice:
        # use the previous several seasons.
        training_seasons = list(
            range(
                max(1999, args.season - 8),
                args.season,
            )
        )

        training = build_walk_forward_training(
            training_seasons
        )

        if not training.empty:

            print(
                f"Training examples: "
                f"{len(training):,}"
            )

            ml_result = try_ml_projection(
                training,
                projections,
            )

            if ml_result is not None:

                projections = ml_result

                # Blend ML and transparent baseline.
                projections["projection"] = (
                    0.70
                    * projections["ml_projection"]
                    + 0.30
                    * projections["baseline_projection"]
                )

                print(
                    "\nML projection enabled."
                )

            else:
                projections["projection"] = (
                    projections[
                        "baseline_projection"
                    ]
                )

        else:
            projections["projection"] = (
                projections[
                    "baseline_projection"
                ]
            )

    else:

        projections["projection"] = (
            projections[
                "baseline_projection"
            ]
        )

    # Keep baseline as primary displayed projection
    # unless ML was explicitly enabled.
    if not args.train_ml:
        projections["projection"] = (
            projections[
                "baseline_projection"
            ]
        )

    # --------------------------------------------------------
    # Transparent learned-weight projection.
    # --------------------------------------------------------

    learned_weight_model = None

    if args.learn_weights:
        print("\nTraining transparent learned metric weights...")
        training_seasons = list(
            range(
                max(1999, args.season - 8),
                args.season,
            )
        )
        training = build_walk_forward_training(training_seasons)
        learned_weight_model, weight_report = train_learned_weight_model(
            training
        )

        if learned_weight_model is not None:
            save_learned_weight_model(learned_weight_model)
            weight_report.to_csv(
                LEARNED_WEIGHT_REPORT_PATH,
                index=False,
            )
            print(
                f"Learned {len(learned_weight_model['features'])} metric "
                f"weights from {learned_weight_model['training_examples']:,} "
                "historical player-weeks."
            )
            print(
                f"Weight report: {LEARNED_WEIGHT_REPORT_PATH.resolve()}"
            )
        else:
            print("Learned weights unavailable; keeping baseline projection.")

    if learned_weight_model is None:
        learned_weight_model = load_learned_weight_model()
        if learned_weight_model is not None:
            print("Using saved learned metric weights.")

    if learned_weight_model is not None:
        try:
            projections = apply_learned_weight_model(
                learned_weight_model,
                projections,
            )
            projections["projection"] = projections[
                "learned_weight_projection"
            ]
            print("Learned-weight projection enabled.")
        except (KeyError, TypeError, ValueError) as exc:
            print(
                "WARNING: saved learned weights could not be applied; "
                f"using baseline projection. ({exc})"
            )

    # --------------------------------------------------------
    # Waiver pool.
    # --------------------------------------------------------

    waiver = build_waiver_pool(
        projections,
        roster,
    )

    # --------------------------------------------------------
    # Export CSVs.
    # --------------------------------------------------------

    export_csvs(
        roster,
        projections,
        waiver,
    )

    # --------------------------------------------------------
    # HTML dashboard.
    # --------------------------------------------------------

    dashboard_html = build_html(
        roster,
        projections,
        history,
        injury_status,
        waiver,
        unmatched,
        args.season,
        target_week,
    )

    output_path = Path(
        args.output
    )

    output_path.write_text(
        dashboard_html,
        encoding="utf-8",
    )

    try:
        dashboard_path = output_path.resolve()
        dashboard_url = (
            dashboard_path.as_uri()
            + f"?generated={pd.Timestamp.now().value}"
        )
        opened = webbrowser.open_new_tab(dashboard_url)
        if not opened and os.name == "nt":
            os.startfile(str(dashboard_path))
        elif not opened:
            webbrowser.open(dashboard_path.as_uri())
    except Exception as exc:
        print(f"WARNING: unable to open dashboard automatically: {exc}")

    # --------------------------------------------------------
    # Optional backtest.
    # --------------------------------------------------------

    if args.backtest:

        # Use recent historical seasons initially.
        backtest_seasons = list(
            range(
                max(1999, args.season - 5),
                args.season,
            )
        )

        results = run_backtest(
            backtest_seasons
        )

        print_backtest_summary(
            results
        )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROJECTION SUMMARY")
    print("=" * 60)

    top = projections.sort_values(
        "projection",
        ascending=False,
    )

    display_columns = [
        "player_name",
        "position",
        "team",
        "projection",
        "range_low",
        "range_high",
        "confidence",
        "classification",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in top.columns
    ]

    print(
        top[display_columns]
        .head(20)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)

    print(
        f"Dashboard: {output_path.resolve()}"
    )

    print(
        f"Roster mapping: "
        f"{(OUTPUT_DIR / 'roster_mapping.csv').resolve()}"
    )

    print(
        f"Projections: "
        f"{(OUTPUT_DIR / 'projections.csv').resolve()}"
    )

    print(
        f"Waiver candidates: "
        f"{(OUTPUT_DIR / 'waiver_candidates.csv').resolve()}"
    )

    if not unmatched.empty:

        print(
            "\nACTION NEEDED:"
        )

        print(
            "Some Yahoo names could not be confidently "
            "matched to NFLverse."
        )

        print(
            "Those mappings should be reviewed before "
            "trusting the waiver/projection results."
        )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()
