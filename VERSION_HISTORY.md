# FANDROMEDA version history

Version changes are recorded here before they are treated as releases. A
version bump means the application behavior or published artifacts changed; it
does not by itself mean a GitHub release or deployment has happened.

## v0.6.2-beta — 2026-09-21

- Preserve inline Yahoo injury markers in roster captures.
- Display Yahoo Out, Questionable, Doubtful, and related roster statuses on
  dashboard player rows, including Puka Nacua's current Out status.
- Added regression coverage for inline status parsing; no historical retraining
  is required.

## v0.6.1-beta — 2026-09-21

- Publish the accumulated Project Lab, Yahoo import, roster snapshot, and
  bench styling improvements from v0.6.0-beta.
- Archive each validation fold separately and review saved results with
  --validation-report without retraining.
- Correct the research matchup join to the target-week opponent and league
  scoring. Earlier experiments are retained but flagged as invalid matchup
  evidence. The corrected candidate is unvalidated and not promoted to live use.
- Check both forecast capture times and matching player-week identity before
  benchmark comparisons; compare only rows with both numeric forecasts.
- Keep generated dashboards, private roster exports, and local research
  archives on the owner's computer rather than publishing them in Git.
- No historical retraining is required for this reporting/tooling release.

## v0.6.0-beta — 2026-09-21

- Added creator-only Project Lab report generation.
- Added Yahoo matchup parsing, roster reconciliation, capture provenance, and
  full Week 3 league coverage checks.
- Added kickoff-safe benchmark foundations and comparison metrics.
- Corrected normal prediction snapshots to retain the league roster rather than
  the full NFL projection pool.
- Dashboard footer now reports v0.6.0-beta and bench rows have a subtle tint.

## v0.5.0-beta — 2026-09-20

- Added append-only prediction snapshots and initial Yahoo projection capture.
- Preserved the prior temporal, scoring, and train/serve fixes.

## Version-change policy

Every future version bump should be announced to the project owner in the same
working session, recorded here with date and user-visible changes, reflected in
the dashboard footer, and included in the creator Project Lab report.
