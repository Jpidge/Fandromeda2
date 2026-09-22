"""User-directed capture of Yahoo's week-specific projection table.

This deliberately keeps the raw capture separate from the working roster.
The first verified capture establishes Yahoo's current text layout before the
numeric parser is allowed to feed benchmark snapshots.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import unicodedata
from uuid import uuid4
from collections import Counter

import pandas as pd

from fandromeda.yahoo_roster import (
    _native_copy_yahoo_window,
    _open_yahoo_roster_tab,
)


PARSER_VERSION = "yahoo_matchup_clipboard_v1"
TABLE_HEADER = ["Stats", "Player", "Proj", "Fan Pts", "Pos",
                "Fan Pts", "Proj", "Player", "Stats"]
SLOTS = {"QB", "RB", "WR", "TE", "W/R/T", "K", "DEF", "BN", "IR"}


def _clean_player_name(value: str) -> str:
    name = re.split(r"Video Forecast|New Player Note|No new player Notes|Player Note",
                    value, maxsplit=1, flags=re.I)[0].strip()
    # Only attached status suffixes, not arbitrary trailing name characters.
    return re.sub(r"(?<=[a-z.])(?:IR-R|PUP-R|IR|PUP|CEL|Q|O|D|P)$", "", name).strip()


def _name_key(value: object) -> str:
    name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"\b(?:jr|sr|ii|iii|iv|v)\.?\b", "", name)
    return re.sub(r"[^a-z0-9]", "", name)


def _team_key(value: object) -> str:
    team = str(value).strip().upper()
    return {"JAC": "JAX", "WSH": "WAS", "NWE": "NE", "TAM": "TB",
            "SFO": "SF", "KAN": "KC", "GNB": "GB", "NOR": "NO",
            "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV"}.get(team, team)


def _number(value: str) -> float | None:
    if value in {"–", "—", "-"}:
        return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        raise ValueError(f"Unexpected Yahoo points value: {value!r}")
    return float(value)


def parse_yahoo_matchup(text: str, *, week: int, league_id: str = "893771") -> pd.DataFrame:
    """Parse the verified two-sided, expanded pregame matchup clipboard layout.

    This deliberately fails on changed/in-game layouts rather than confusing
    actual statistics with projections. Season and capture provenance belong
    to the import record, not to a guess from a page without a season label.
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines() if line.strip()]
    if not re.search(rf"\(ID#\s*{re.escape(str(league_id))}\)", text):
        raise ValueError("Yahoo league ID is missing or does not match.")
    displayed = re.search(r"(?m)^Week (\d+):", text)
    if displayed is None or int(displayed[1]) != week:
        raise ValueError("Yahoo displayed week does not match the requested week.")
    starts = [i for i in range(len(lines)) if lines[i:i + 9] == TABLE_HEADER]
    if len(starts) != 2 or "Hide Bench Players" not in lines:
        raise ValueError("Expected starter and expanded bench tables; show bench players before copying.")
    header = lines[:starts[0]]
    managers = [header[i - 2] for i, line in enumerate(header)
                if i >= 2 and re.fullmatch(r"\d+-\d+-\d+\s*\|.*", line)]
    if len(managers) != 2 or managers[0] == managers[1]:
        raise ValueError("Could not identify exactly two distinct matchup managers.")

    def player_at(index: int):
        if lines[index] == "(Empty)":
            return None, index + 1
        if index + 2 >= len(lines):
            raise ValueError("Truncated Yahoo player block.")
        team_position = re.fullmatch(r"([A-Za-z]{2,4})\s*-\s*(QB|RB|WR|TE|K|DEF)", lines[index + 1])
        if team_position is None:
            raise ValueError(f"Unexpected Yahoo player block near {lines[index]!r}")
        matchup = lines[index + 2]
        if not re.match(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b|^Bye\b", matchup):
            raise ValueError("Only the verified pregame layout is supported; game may have started.")
        return {"player_name": _clean_player_name(lines[index]),
                "team": _team_key(team_position[1]), "position": team_position[2],
                "game_label": matchup}, index + 3

    rows = []
    for table_index, start in enumerate(starts):
        index = start + 9
        paired_rows = 0
        table_slots = []
        while index < len(lines):
            # The only permitted end is a verified totals row.
            if "TOTAL" in lines[index:index + 5]:
                total_pos = lines.index("TOTAL", index, min(index + 5, len(lines)))
                expected_before = 2 if table_index == 0 else 1
                expected_after = 2 if table_index == 0 else 1
                if total_pos - index != expected_before:
                    raise ValueError("Unexpected Yahoo total-row layout.")
                if len(lines[total_pos + 1:total_pos + 1 + expected_after]) != expected_after:
                    raise ValueError("Truncated Yahoo totals row.")
                for value in lines[index:total_pos] + lines[total_pos + 1:total_pos + 1 + expected_after]:
                    _number(value)
                break
            left, index = player_at(index)
            if index + 5 >= len(lines):
                raise ValueError("Truncated Yahoo score cells.")
            left_projection, left_actual, slot, right_actual, right_projection = lines[index:index + 5]
            if slot not in SLOTS or (table_index == 0) != (slot not in {"BN", "IR"}):
                raise ValueError("Unexpected roster slot or shifted Yahoo score columns.")
            table_slots.append(slot)
            # Validate both actual columns without storing them as forecasts.
            _number(left_actual)
            _number(right_actual)
            left_value, right_value = _number(left_projection), _number(right_projection)
            right, index = player_at(index + 5)
            for side, player, value in [(0, left, left_value), (1, right, right_value)]:
                if player is not None:
                    rows.append({**player, "manager": managers[side], "slot": slot,
                                 "yahoo_projection": value, "week": week,
                                 "league_id": str(league_id), "table": "starters" if table_index == 0 else "bench"})
            paired_rows += 1
        else:
            raise ValueError("Yahoo table is missing its totals row.")
        if (table_index == 0 and paired_rows != 9) or (table_index == 1 and not 1 <= paired_rows <= 7):
            raise ValueError("Unexpected roster table size for this league.")
        if table_index == 0 and Counter(table_slots) != Counter(["QB", "RB", "RB", "WR", "WR", "TE", "W/R/T", "K", "DEF"]):
            raise ValueError("Starter slots do not match the league roster rules.")
    result = pd.DataFrame(rows)
    player_blocks = sum(bool(re.fullmatch(r"[A-Za-z]{2,4}\s*-\s*(QB|RB|WR|TE|K|DEF)", line)) for line in lines)
    if len(result) != player_blocks:
        raise ValueError("Some Yahoo player blocks were not parsed; refusing partial table output.")
    keys = result.apply(lambda r: (r.manager, r.position, _name_key(r.player_name)), axis=1)
    if keys.duplicated().any():
        raise ValueError("Duplicate Yahoo player entries in matchup capture.")
    return result


def reconcile_yahoo_projections(captured: pd.DataFrame, roster: pd.DataFrame):
    """Exact, conservative roster reconciliation with explicit coverage gaps."""
    required = {"manager", "player", "position", "team", "slot", "nflverse_player_id"}
    if not required.issubset(roster.columns):
        raise ValueError(f"Roster mapping is missing: {sorted(required - set(roster.columns))}")
    roster = roster.fillna("").reset_index(drop=True)
    if captured.empty:
        raise ValueError("No Yahoo projections were captured.")
    if captured["week"].nunique() != 1 or captured["league_id"].nunique() != 1:
        raise ValueError("Cannot combine different Yahoo weeks or leagues.")
    def key(name, position, team):
        return (position, _team_key(team) if position == "DEF" else _name_key(_clean_player_name(str(name))))
    roster_keys = [key(r.player, r.position, r.team) for r in roster.itertuples()]
    if len(set(roster_keys)) != len(roster_keys):
        raise ValueError("Ambiguous/duplicate roster identities; resolve before importing Yahoo projections.")
    lookup = dict(zip(roster_keys, range(len(roster))))
    # Existing roster parsing can leave a decorated/truncated display name;
    # the previously resolved canonical name is an exact alias, not a fuzzy
    # guess. Refuse aliases that would point at two roster entries.
    for index, row in roster.iterrows():
        canonical = row.get("nflverse_name", "")
        if canonical and row.position != "DEF":
            alias = key(canonical, row.position, row.team)
            if alias in lookup and lookup[alias] != index:
                raise ValueError("Ambiguous canonical roster name; resolve before importing.")
            lookup[alias] = index
    found = {}
    unexpected = []
    for record in captured.to_dict("records"):
        identity = key(record["player_name"], record["position"], record["team"])
        index = lookup.get(identity)
        if index is None or _name_key(roster.iloc[index]["manager"]) != _name_key(record["manager"]):
            unexpected.append(record)
            continue
        if index in found:
            raise ValueError(f"Duplicate capture for {record['player_name']}; do not combine repeated matchups.")
        found[index] = record
    coverage = []
    for index, row in roster.iterrows():
        record = found.get(index)
        player_id = ("DEF:" + _team_key(row.team)) if row.position == "DEF" else row.nflverse_player_id
        status = "not_captured" if record is None else (
            "projection_unavailable" if pd.isna(record["yahoo_projection"]) else "matched")
        coverage.append({"manager": row.manager, "player_name": row.player,
                         "position": row.position, "team": _team_key(row.team),
                         "roster_slot": row.slot, "player_id": player_id,
                         "identity_resolved": bool(player_id), "coverage_status": status,
                         "yahoo_projection": None if record is None else record["yahoo_projection"],
                         "yahoo_slot": None if record is None else record["slot"],
                         "team_mismatch": bool(record and _team_key(row.team) != record["team"]),
                         "source_file": None if record is None else record.get("source_file"),
                         "captured_at_utc": None if record is None else record.get("captured_at_utc")})
    coverage = pd.DataFrame(coverage)
    summary = {"roster_entries": len(roster), "captured_entries": len(captured),
               "matched_entries": len(found), "numeric_projections": int(coverage.yahoo_projection.notna().sum()),
               "missing_entries": len(roster) - len(found), "unexpected_entries": len(unexpected),
               "roster_managers": int(roster.manager.nunique()),
               "captured_managers": int(captured.manager.nunique()),
               "unresolved_identities": int((~coverage.identity_resolved).sum()),
               "complete_roster_coverage": len(found) == len(roster) and not unexpected,
               "benchmark_eligible": False,
               "benchmark_status": "pending_capture_cutoff_and_kickoff_validation"}
    return coverage, pd.DataFrame(unexpected), summary


def import_yahoo_projection_files(paths: list[Path], roster: pd.DataFrame,
                                  output_root: Path, *, season: int, week: int):
    """Save an append-only import batch, raw evidence and a coverage report.

    File modification time and import time are never used as capture time.
    Untimestamped samples can exercise parsing but cannot become benchmarks.
    """
    inputs, frames = [], []
    for path in paths:
        raw = path.read_bytes()
        frame = parse_yahoo_matchup(raw.decode("utf-8-sig"), week=week)
        source_hash = hashlib.sha256(raw).hexdigest()
        frame["source_file"] = path.name
        frame["captured_at_utc"] = None
        sidecar = path.with_suffix(path.suffix + ".json")
        provenance = None
        if sidecar.exists():
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            if (provenance.get("sha256") != source_hash or
                    provenance.get("season") != season or provenance.get("week") != week):
                raise ValueError(f"Capture provenance does not match {path.name}.")
            timestamp = datetime.fromisoformat(provenance["captured_at_utc"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None or timestamp > datetime.now(timezone.utc):
                raise ValueError("Capture timestamp must have a timezone and cannot be in the future.")
            frame["captured_at_utc"] = timestamp.astimezone(timezone.utc).isoformat()
        inputs.append((path.name, raw, source_hash, provenance))
        frames.append(frame)
    if not frames:
        raise ValueError("Supply at least one matchup text file.")
    captured = pd.concat(frames, ignore_index=True)
    coverage, unexpected, summary = reconcile_yahoo_projections(captured, roster)
    now = datetime.now(timezone.utc)
    batch = output_root / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}"
    batch.mkdir(parents=True, exist_ok=False)
    sources = []
    for number, (name, raw, digest, provenance) in enumerate(inputs, 1):
        filename = f"source_{number:02d}.txt"
        (batch / filename).write_bytes(raw)
        sources.append({"original_name": name, "archived_file": filename, "sha256": digest})
        if provenance is not None:
            (batch / (filename + ".json")).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    roster.to_csv(batch / "roster_mapping_at_import.csv", index=False)
    captured.to_csv(batch / "yahoo_projections.csv", index=False)
    coverage.to_csv(batch / "roster_coverage.csv", index=False)
    if not unexpected.empty:
        unexpected.to_csv(batch / "unexpected_entries.csv", index=False)
    summary.update({"parser_version": PARSER_VERSION, "season": season, "week": week,
                    "imported_at_utc": now.isoformat(), "sources": sources,
                    "timestamped_entries": int(captured.captured_at_utc.notna().sum())})
    (batch / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return batch, summary


def _is_yahoo_projection_text(text: str) -> bool:
    normalized = text.replace("\r\n", "\n")
    has_projection_header = "Proj" in normalized and "Fan Pts" in normalized
    has_player_rows = normalized.count(" - QB") + normalized.count(" - RB")
    has_player_rows += normalized.count(" - WR") + normalized.count(" - TE")
    has_player_rows += normalized.count(" - K") + normalized.count(" - DEF")
    return has_projection_header and has_player_rows >= 8


def capture_yahoo_projection_sample(
    matchup_url: str,
    export_directory: Path,
    *,
    attach_port: int,
    native_copy: bool = True,
    confirm: bool = True,
    season: int = 2026,
    week: int | None = None,
) -> tuple[Path, str]:
    """Capture one Yahoo matchup projection table for parser development.

    No credentials are read or stored, and no roster file is replaced.
    """

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run: python -m pip install playwright"
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(
            f"http://127.0.0.1:{attach_port}"
        )
        if not browser.contexts:
            raise RuntimeError("No Chrome browsing context is available to attach.")
        context = browser.contexts[0]
        page = _open_yahoo_roster_tab(context, matchup_url)
        print("\nAttached to Chrome and opened the requested Yahoo matchup week.")
        try:
            bench = page.get_by_text("Show Bench Players", exact=True)
            if bench.count() == 1:
                bench.click()
        except Exception:
            # Native-copy users can expand the bench themselves; the strict
            # parser below rejects a capture with hidden bench players.
            pass
        if confirm:
            input(
                "When both rosters, their Proj columns, and bench/IR are visible, "
                "press Enter here to capture them: "
            )
        else:
            page.wait_for_timeout(3_000)

        text = ""
        source = ""
        try:
            text = page.evaluate(
                "() => document.body ? document.body.innerText : ''"
            )
            source = "Yahoo matchup document text"
        except Exception:
            text = ""

        if not _is_yahoo_projection_text(text):
            if not native_copy:
                raise RuntimeError(
                    "Yahoo did not expose a projection-shaped matchup table. "
                    "Retry with --yahoo-native-copy."
                )
            print("Yahoo did not expose the matchup table to the browser connection.")
            print("Copying the visible Yahoo matchup with native Windows input...")
            text = _native_copy_yahoo_window()
            source = "native Windows Ctrl+A/Ctrl+C clipboard copy"

        if not _is_yahoo_projection_text(text):
            raise RuntimeError(
                "The captured page did not contain Yahoo Proj/Fan Pts headers "
                "and enough player rows. No projection capture was saved."
            )

        if week is None:
            raise ValueError("An explicit target week is required to validate projection capture.")
        parse_yahoo_matchup(text, week=week)

    export_directory.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(timezone.utc)
    stamp = captured_at.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12]
    capture_path = export_directory / f"yahoo_projection_sample_{stamp}.txt"
    cleaned = text.replace("\r\n", "\n").strip() + "\n"
    with capture_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(cleaned)
    capture_path.with_suffix(".txt.json").write_text(json.dumps({
        "captured_at_utc": captured_at.isoformat(), "season": season, "week": week,
        "source_url": matchup_url, "capture_method": source,
        "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    return capture_path, source
