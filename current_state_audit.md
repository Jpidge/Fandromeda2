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

## Snapshot and Yahoo benchmark status

No immutable production-prediction snapshot store exists yet. Cached NFLverse
files are source-data caches, not records of what Fandromeda predicted at a
specific pre-kickoff time. Numeric Yahoo projections are not yet ingested.

## Validation status

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
