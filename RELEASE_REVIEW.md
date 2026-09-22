# Release review — 2026-09-21

Status: hold. Version remains v0.6.0-beta; no commit or push performed.

The defensive matchup experiment used own-team defense and generic scoring.
Legacy fold results remain archived, but are not evidence for or against the
intended opponent feature. The research implementation now resolves opponents
from the target-week schedule and prefers league scoring. A focused integration
test checks the join and temporal boundary. All 46 short tests pass.

Before the next research evaluation, add paired feature-off/on model evaluation
using shared historical construction and separate result provenance. Do not ask
the owner to rerun the unchanged 2024/2025 jobs. This research work need not delay
a separately scoped dashboard/report release once its artifacts are reviewed.

Pending release packaging: inspect all untracked modules and tests, decide
which generated dashboards/league data belong in Git, and review staged diff.
Existing snapshots, archives, and uncommitted work must remain preserved.
Generated index.html currently includes pre-existing trailing whitespace.

Purple Reign is Yahoo team 10. Week 3 roster membership was verified on Yahoo;
Corum flex / Nacua bench-Out changes were copied into the saved roster with a
backup first. Other managers' saved rows were not refreshed in this pass.
The dashboard uses cached statistics; this is not a fresh injury/news feed.
