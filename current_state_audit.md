# FANDROMEDA current-state audit

Date: 2026-09-19

## Current implementation

- `gptindex.py` is the working single-file application and `index.py` is its
  lightweight launcher.
- The app parses copied Yahoo roster text, resolves most players to NFLverse
  IDs, and treats team defenses separately from individual-player matching.
- Cached NFLverse inputs include weekly player stats (2018–2026), player
  master data, schedules, snap counts, current injuries, and depth charts.
- The dashboard renders roster and waiver views, player details, signal
  explanations, injury/depth flags, and separate primary and direct-ML
  projection columns.
- Forecast layers are a transparent baseline, a learned-weight ridge model,
  and a direct XGBoost model. When saved learned weights are available, the
  learned-weight value is the displayed `Proj. Pts.`; direct ML remains a
  comparison value.
- Existing outputs include roster mapping, current projections, waiver
  candidates, learned metric weights, and ML holdout validation.

## Confirmed temporal finding

The suspected over-lag was confirmed.

`add_player_features` correctly removes the target week, but it previously
shifted every metric again before returning the final historical row. As a
result, a Week 6 forecast used data through Week 4 rather than the available
Week 5 data. This was not target leakage, but it made live and historical
forecasts one game unnecessarily stale.

The feature contract has now been corrected:

- A Week W forecast includes all information through Week W-1.
- Week W remains excluded.
- Recent rolling values and EWMAs end at Week W-1.
- Momentum compares W-2/W-1 with W-4/W-3.

`tests/test_temporal_features.py` was added as a deterministic regression
test. Its Week 6 target contains an intentionally extreme Week 6 value and
asserts that the resulting features still end with Week 5.

## Train/serve consistency

Resolved and validated on 2026-09-19.

Live forecasts and historical walk-forward training now call the same shared
feature-history path. Each player's final three prior-season games, including
available snap usage, are attached before Week 1 as negative-week carry-over
context. The holdout now trains and tests with the same early-season history
structure that the live dashboard uses.

Saved learned-weight and direct-ML artifacts from before this change must be
retrained before they are used as current live forecasts.

## Canonical scoring engine — phase 1

Implemented and locally verified on 2026-09-19 with
``python -m unittest discover -s tests -v``.

`fandromeda/scoring.py` is now the single source of league rules. It provides
formulas and regression tests for core non-PPR offense, 40+ yard TD bonuses, return scoring,
offensive fumble-return TDs, kicker PAT/field-goal-yard scoring, and every
D/ST points-allowed tier. `standardize_player_stats` now uses that module for
the currently available weekly offensive inputs.

The remaining phase-2 work is data reconstruction, not scoring-rule design:
play-by-play is required to populate long-TD bonuses, return events, and
offensive fumble-return TDs; separate event aggregation is required for K/DEF.
Until then, current historical ML evaluation remains an offensive-model
benchmark using the core weekly fields.

Play-by-play event caches for 2018–2026 have now been built and validated at
the schema level. Available events are joined before canonical actual scoring.
The nflreadpy schema does not expose a reliable player credit for offensive
fumble-return touchdowns, so that rare rule remains explicitly unavailable
rather than being silently assigned as zero. Cached ML artifacts trained on
the former targets are stale and must be regenerated before use.
Model files now include a scoring-version marker and are rejected when their
target definition does not match the active canonical scoring engine.

The revised 2025 holdout and both saved ML artifacts were regenerated after
the PBP integration. Their current training size is 135,913 player-weeks and
their scoring marker is ``ffc_yahoo_v1``. Team-defense event reconstruction
and dedicated K/DEF projection models remain separate future work; offensive
and kicker player scoring is now sourced through the canonical engine where
the available NFLverse events permit it.

## Prediction snapshot store

Implemented and locally verified on 2026-09-20. Every normal projection run
now writes one new immutable Parquet file in
`data/snapshots/predictions/` and appends one readable record to
`data/snapshots/snapshot_index.csv`. The mutable `data/output/projections.csv`
remains the live dashboard export and is not used as history.

Each snapshot stores the UTC capture time, season/week, player identity,
baseline / learned-weight / direct-ML / final displayed projections, range,
classification, depth context, and the app, feature, scoring, and primary
model versions. `yahoo_projection` and `actual_points` are intentionally null
at capture time; later evaluation work may populate comparisons without
rewriting the original forecast values. A unique ID prevents same-second runs
from replacing one another. `python index.py --list-snapshots` prints the
human-readable index without loading every Parquet file.

Cached NFLverse files are source-data caches, not prediction records. Numeric
Yahoo projections are not yet ingested.

## Validation status

### 2026-09-21 continuation — snapshot population correction

The original v1 writer saved the full NFL projection pool (2,176 rows in the
first snapshot), not the league roster. Normal runs now build v2 snapshots
from roster membership, retaining fantasy manager, slot, and missing forecast
status. The current roster export contains 171 individual players and 13 D/ST
entries across 12 managers. Team defenses use separate `DEF:<team>` IDs;
unresolved players cannot join to blank projection IDs. Old snapshot files
and manifest rows are preserved. Full-pool live/waiver exports are unchanged.

Temporal/scoring work and the 10 p.m. roster automation are confirmed complete
by the project owner. Yahoo numeric projection capture remains a raw-text
prototype; it has no numeric parser or benchmark ingestion yet. Snapshot
kickoff eligibility and trained-model provenance remain future benchmark work.

Verified after correction: all 25 tests pass, including six snapshot tests.
A temporary real-roster Parquet roundtrip retained 184 entries, with 169
forecasts and 15 unavailable (13 D/ST plus two WRs). Existing snapshot and
manifest hashes were unchanged. The production writer will use roster scope
on its next normal run; no synthetic production capture was created.

A real Week 3 Yahoo matchup clipboard sample was captured on 2026-09-21
with bench and IR visible (31 entries across two managers), saved under
`data/yahoo_projection_exports/` with layout notes. This is a parser-development
sample only, not a kickoff-certified benchmark observation.

The first aggregate 2025 holdout was useful for exercising the evaluation
pipeline, but it was produced before the over-lag correction. It must not be
used as the post-fix model benchmark. Preserve it in `testing_history.txt` as
pre-correction context, then rerun validation after the temporal regression
test passes.

## Safe milestone order

1. Retrain and save live ML artifacts using the corrected feature pipeline.
2. Build and test one canonical league-scoring engine.
3. Add versioned, append-only pre-kickoff prediction snapshots.
4. Add a reliable, policy-compliant Yahoo-projection capture workflow.
5. Compare Fandromeda and Yahoo fairly, then evaluate lineup decisions.
6. Research matchup, role, and uncertainty features one at a time against
   locked temporal benchmarks.
