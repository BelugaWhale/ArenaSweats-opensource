# High-Elo Rating Analysis

This player-centric analysis uses player-games from daily top-quarter Challenger members. A qualifying three-player stack contributes three observations.

Daily top-quarter membership uses the latest completed regional leaderboard before the game date. It means ranks 1-50 in 200-slot Challenger regions, ranks 1-25 in 100-slot regions, and ranks 1-13 in 50-slot regions. Membership moves daily and uses no later games. Exact second-by-second pre-game regional rank is not retained.

The first season week, July 28 through August 3, is excluded. Analysis begins August 4. ME is omitted because only three post-cutoff days remain.

## Run

Replace every `{region}` token in `extract.sql` with a trusted region code, export one CSV per region with `clickhouse-client`, then run:

```bash
/home/m/projects/.venv311/bin/python validations/highelo_rating_sim/highelo_rating_sim.py \
  /path/to/highelo_players_*.csv.gz \
  --output validations/highelo_rating_sim/output
```

Each regional panel shows the minimum number of most-prevalent exact combinations needed to cover at least 80% of that region's player-games. The y-axis is mean direct rating change.
