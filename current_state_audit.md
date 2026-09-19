# Fandromeda current-state audit

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

## Confirmed train/serve skew

Live early-season forecasts blend each player's final three games from the
prior season. Historical walk-forward training currently does not use that
same carry-over history. This is the next temporal-consistency item to fix,
after the Week W-1 test is run and the corrected holdout baseline is saved.

## Scoring audit

The current player scoring calculation correctly implements core non-PPR
offensive scoring: passing/rushing/receiving yardage, six-point passing TDs,
interceptions, fumbles lost, and two-point conversions.

It does not yet reconstruct rules requiring play-by-play or dedicated K/DEF
data: 40+ yard TD bonuses, return yards/return TDs, offensive fumble-return
TDs, exact kicker field-goal-yard scoring, or team-defense scoring tiers.
Therefore the current historical holdout measures offensive-model behavior,
not yet complete Fantasy Football Coalition Yahoo scoring.

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

1. Run the temporal regression test and regenerate the corrected 2025
   overall/by-position holdout benchmark.
2. Make train and live feature construction consistent for prior-season
   carry-over history; rerun the same benchmark.
3. Build and test one canonical league-scoring engine.
4. Add versioned, append-only pre-kickoff prediction snapshots.
5. Add a reliable, policy-compliant Yahoo-projection capture workflow.
6. Compare Fandromeda and Yahoo fairly, then evaluate lineup decisions.
7. Research matchup, role, and uncertainty features one at a time against
   locked temporal benchmarks.
