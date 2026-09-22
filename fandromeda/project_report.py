"""Generate a creator-only, human-readable FANDROMEDA project report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import html
import json

import pandas as pd
from fandromeda.validation_reports import load_validation_reports


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="muted">No measured records available.</p>'
    return frame.to_html(index=False, classes="data", border=0, float_format=lambda v: f"{v:.3f}")


def build_project_report(project_root: Path) -> str:
    output = project_root / "data" / "output"
    snapshots = project_root / "data" / "snapshots"
    holdout_path = output / "ml_holdout_validation.csv"
    holdout = pd.read_csv(holdout_path) if holdout_path.exists() else pd.DataFrame()
    archived = load_validation_reports(output)
    if not archived.empty:
        holdout = archived[archived["position"] == "Offense (QB/RB/WR/TE)"]
    index_path = snapshots / "snapshot_index.csv"
    snapshot_index = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
    imports = []
    import_root = project_root / "data" / "yahoo_projection_exports" / "imports"
    for summary in import_root.glob("*/summary.json"):
        try:
            imports.append(json.loads(summary.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    imports_frame = pd.DataFrame([{
        "imported_at_utc": x.get("imported_at_utc", ""),
        "season": x.get("season", ""), "week": x.get("week", ""),
        "roster_entries": x.get("roster_entries", ""),
        "matched_entries": x.get("matched_entries", ""),
        "numeric_projections": x.get("numeric_projections", ""),
        "missing_entries": x.get("missing_entries", ""),
        "benchmark_eligible": x.get("benchmark_eligible", False),
    } for x in imports])
    latest = snapshot_index.tail(8) if not snapshot_index.empty else snapshot_index
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    version_file = project_root / "VERSION_HISTORY.md"
    version_history = version_file.read_text(encoding="utf-8") if version_file.exists() else "Version history unavailable."
    version_lines = [line.strip() for line in version_history.splitlines() if line.startswith("## ")]
    next_steps_file = project_root / "NEXT_STEPS.md"
    next_steps = next_steps_file.read_text(encoding="utf-8") if next_steps_file.exists() else "Next steps unavailable."
    next_steps_lines = [line[2:].strip() for line in next_steps.splitlines() if line.startswith("## ")]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FANDROMEDA Project Lab</title><style>
body{{margin:0;background:#101522;color:#e8edf7;font:15px system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}}
main{{max-width:1180px;margin:auto;padding:28px 20px 60px}} h1{{margin:0 0 6px;font-size:30px}}
h2{{margin-top:32px;border-bottom:1px solid #33405c;padding-bottom:8px}} .muted{{color:#9ba8c1}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}}
.card{{background:#192238;border:1px solid #33405c;border-radius:10px;padding:16px}} .card b{{display:block;font-size:25px}}
.data{{width:100%;border-collapse:collapse;background:#192238;border-radius:8px;overflow:hidden}} .data th,.data td{{padding:8px 10px;border-bottom:1px solid #33405c;text-align:left}} .data th{{color:#b8c8e6;background:#202d47}}
.status{{padding:3px 8px;border-radius:99px;background:#51401a;color:#ffe08a}} code{{color:#b7d5ff}}
</style></head><body><main><p class="muted">Creator-only research view · generated {html.escape(generated)}</p>
<h1>FANDROMEDA Project Lab</h1><p class="muted">Validation, provenance, benchmark readiness, and historical model evidence. This page is separate from the fantasy-manager dashboard.</p>
<div class="cards"><div class="card"><b>{len(snapshot_index):,}</b>saved snapshots</div><div class="card"><b>{int(snapshot_index.iloc[-1].row_count) if not snapshot_index.empty else 0:,}</b>latest snapshot rows</div><div class="card"><b>{len(imports):,}</b>Yahoo imports</div><div class="card"><b>{"READY" if imports and all(x.get("complete_roster_coverage") for x in imports) else "PENDING"}</b>benchmark coverage</div></div>
<h2>Current milestone status</h2><p><span class="status">PENDING</span> Actual finalized points and per-player kickoff validation are required before publishing FANDROMEDA-versus-Yahoo results.</p>
<h2>Version history</h2><p>{html.escape(" · ".join(version_lines))}</p>
<h2>Historical ML holdout</h2>{_table(holdout)}
<h2>Train/test boundary</h2><p>Each fold trains on seasons before its labeled holdout year. The holdout is never passed to fitting. The offensive QB/RB/WR/TE aggregate is the primary benchmark.</p>
<p class="status">Release review: legacy defensive_matchup_candidate results joined the player's own defense, not the upcoming opponent, and preferred generic scoring. They are retained for audit but are invalid evidence about opponent strength. The corrected experiment has not been evaluated; do not rerun historical training just to view this report.</p>
<h2>Feature coverage</h2><ul><li>The defensive matchup candidate now uses prior-week fantasy points allowed by position in walk-forward holdout experiments; it is not yet approved for live projections.</li><li>Kickoff and punt return yards/touchdowns are reconstructed from play-by-play for historical actual scoring when player IDs are available.</li><li>Return opportunity is not yet forecast as a model feature. Offensive fumble-return touchdowns remain unavailable where the source cannot assign player credit.</li></ul>
<h2>Next milestones</h2><p>{html.escape(" · ".join(next_steps_lines))}</p><p class="muted">Full sequencing and validation gates are documented in NEXT_STEPS.md.</p>
<h2>Yahoo capture and reconciliation</h2>{_table(imports_frame)}
<h2>Prediction snapshot history</h2>{_table(latest)}
<h2>Interpretation guardrails</h2><ul><li>Only finalized actual points belong in benchmark metrics.</li><li>Yahoo and FANDROMEDA captures must both precede each player’s kickoff.</li><li>Legacy snapshot schemas remain preserved and are not silently combined with newer records.</li><li>Missing data stays missing; it is not treated as zero unless the scoring rule says zero.</li></ul>
</main></body></html>"""


def write_project_report(project_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_project_report(project_root), encoding="utf-8")
    return output_path
