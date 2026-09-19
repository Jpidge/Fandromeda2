# FANDROMEDA
Cosmic Fantasy Analytics & Waiver Intelligence Hub

## Quick Start
1. Update Roster: Update data/my_roster.txt with current league roster (copy paste table from Yahoo)
2. Generate dashboard: Run python index.py

## Features
* **Interactive Dashboard:** Filter by team, click headers to sort, click names for detailed stats, hover or tap on values for detailed explanations.

* **Machine Learned Points Projection:*
The XGBoost ML model trained on historical, leakage-safe player-weeks to predict next-week fantasy points directly. It considers the same pre-game history plus position indicators; it is a comparison forecast and does not replace Proj. Pts. A dash means the direct model has not been trained or is unavailable on this computer.

* **Machine Learned Weight Values:*
When trained weights are available, Proj. Pts. uses a transparent regularized model trained on historical NFLverse player-weeks instead of the hand-set baseline mix. It learns how much each pre-game metric should raise or lower the next-week projection, including separate QB/RB/WR/TE adjustments. Run python gptindex.py --learn-weights to train or refresh it. The saved weight report ranks the learned metric weights; it remains separate from Machine Learned Points Projection.

* **Projection (±):*
Weekly fantasy-point projection followed by a player-specific plus/minus estimate. The estimate is 1.15 × the model's recent-volatility measure; it is a planning guide, not a guarantee or formal confidence interval.

* **Scoring and projections:*
Core league scoring: passing yards ÷ 25, passing touchdowns × 6, interceptions × −2, rushing/receiving yards ÷ 10, rushing/receiving touchdowns × 6, receptions × 0, fumbles lost × −2, and two-point conversions × 2. Return yards score 1 point per 25 yards, and 40+ yard passing/rushing/receiving touchdowns receive +1. Those return and 40+ touchdown details require play-by-play data and are not yet included in the weekly-player projection inputs.

* **Baseline projection:*
45% exponentially weighted fantasy-point average + 25% three-game rolling average + 15% five-game rolling average + 15% season weighted average, then a capped opportunity-momentum adjustment and TD-dependence penalty.

* **Carry-over history:*
The last three games from the prior season are included before Week 1 so established players have useful rolling history. Current-season games remain the newest and most important observations.

* **3-Wk Avg and recent form:*
3-Wk Avg is the average of actual fantasy points from a player's latest three included games. The compact Pts and Opp sparklines show the latest six included games: Pts is actual fantasy scoring for consistency, while Opp is position-aware opportunity — passing attempts and rushing involvement for QBs, carries and targets for RBs, and targets for WRs/TEs.

* **Opportunity score:*
Targets × 1.00 + carries × 0.55 + offensive snap percentage × 0.06 + red-zone targets × 1.40 + red-zone carries × 1.10. It is a role/usage measure, not a fantasy-point projection.

* **Momentum:*
The latest two-game average minus the preceding two-game average. For example, a 10-point average in the earlier two games and a 15-point average in the latest two produces +5 production momentum. Opportunity momentum uses role/usage components; production momentum uses fantasy points. It reacts quickly to a recent change, while the 6-Wk Trend lines show the broader shape and consistency of the player’s recent history.

* **Trend gap:*
Opportunity momentum − production momentum. A positive value means role growth is outpacing scoring; a negative value means scoring is outpacing role growth.

* **TD dependence:*
Touchdown fantasy points ÷ total fantasy points, using the exponentially weighted averages. Higher values imply greater risk of regression when touchdowns slow down.

* **Depth-chart context:*
Live depth-chart rank and Out/IR teammate status help identify real starters, healthy backups, and temporary replacements. A healthy QB2 is projected for 0 unless the QB1 is unavailable; RB/WR/TE depth status is context only because those players often have planned roles. A TEMP flag means recent usage is likely injury-driven; it remains valid for the next matchup while the higher-ranked teammate is unavailable, then resets automatically when that player returns. IR is treated as season-ending only when the feed explicitly says so.

* **Confidence:*
An internal 20–90 consistency score, not a probability. It starts at 50, then adds capped consistency (12 − volatility) and capped opportunity; it is shown in player details rather than the main table and is withheld as “Early season” until at least three included games are available.

## Security Note
Keep API credentials out of tracking. Ensure `.env` is listed in `.gitignore`.
