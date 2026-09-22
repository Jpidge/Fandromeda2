# FANDROMEDA next engineering milestones

The fantasy-manager dashboard remains decision-focused. Research and
validation details belong in Project Lab.

## Milestone A — defensive matchup feature research

1. Build leakage-safe opponent defensive history from games completed before
   each target player-week.
2. Start with stable, interpretable measures: fantasy points allowed by
   position, passing yards/attempts allowed, rushing yards/attempts allowed,
   sacks, interceptions, and defensive touchdowns.
3. Keep the target-week schedule opponent and home/away fields separate from
   the historical strength fields.
4. Add deterministic tests proving Week W uses defensive games through W-1
   only, and that the target game cannot affect its own matchup feature.
5. Run the same 2025 holdout against the locked current benchmark.
6. Keep the feature group only if the improvement is consistent across more
   than one temporal fold and does not create meaningful position bias.

No matchup feature should use final target-week scores, target-week defensive
stats, or post-kickoff information.

## Milestone B — return-feature research

1. Audit kickoff and punt returner identity coverage by season.
2. Preserve separate kickoff-return and punt-return counts, yards, and
   touchdowns in the event table while retaining combined league scoring.
3. Add pregame history features such as return attempts, return yards per
   game, and recent return-role continuity only when the source identifies the
   returner reliably.
4. Test that return events change finalized historical actuals but never enter
   a target-week feature row.
5. Compare a return-aware model against the locked benchmark; do not enable it
   in production based on one season or a tiny special-teams sample.

## Milestone C — benchmark publication

1. Capture Week 4 Yahoo and FANDROMEDA projections Thursday before kickoff.
2. Validate every row against its team kickoff in UTC.
3. Join finalized canonical actual points after the week is complete.
4. Publish FANDROMEDA versus Yahoo metrics in Project Lab only when all rows
   have comparable capture timing and finalized actuals.

## Current known boundary

Opponent identity is already attached to live projections. Opponent defensive
strength is not yet a predictive feature. Return yards and touchdowns can be
reconstructed for historical scoring from NFLverse play-by-play, but return
opportunity is not yet forecast. Offensive fumble-return touchdowns remain
unavailable when player credit cannot be assigned reliably.
