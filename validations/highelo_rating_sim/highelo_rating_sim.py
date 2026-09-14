#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Analyze daily top-quarter Challenger player-games.")
parser.add_argument("inputs", nargs="+", help="CSV or CSV.gz files produced by extract.sql")
parser.add_argument("--output", type=Path, required=True, help="Directory for CSV, Markdown, and PNG output")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

required = {
    "region", "player_hash", "game_id", "game_date", "game_ts", "placing", "pregame_rating",
    "rating_change", "repeated_teammates", "other_gm_plus_teammates",
    "is_top_quarter_challenger", "team_player_count",
}
observations = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
missing = required - set(observations.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")
if observations[list(required)].isna().any().any():
    raise RuntimeError("Input contains null required values")
if observations.duplicated(["region", "player_hash", "game_id"]).any():
    raise RuntimeError("Input contains duplicate regional player-game rows")
if not observations["team_player_count"].eq(3).all():
    raise RuntimeError("Expected three-player teams")
if not observations["placing"].between(1, 6).all():
    raise RuntimeError("Expected placements from 1 through 6")
if not observations["repeated_teammates"].between(0, 2).all() or not observations["other_gm_plus_teammates"].between(0, 2).all():
    raise RuntimeError("Repeated-teammate and other-GM+ counts must be 0, 1, or 2")
if not observations["is_top_quarter_challenger"].isin([0, 1]).all():
    raise RuntimeError("Top-quarter flag must be zero or one")
observations = observations[observations["is_top_quarter_challenger"].eq(1)].copy()
observations = observations[(pd.to_datetime(observations["game_date"]) >= "2026-08-04") & observations["region"].ne("me")].copy()
if observations.empty:
    raise RuntimeError("Input contains no daily top-quarter Challenger player-games")

observations["combination"] = "R" + observations["repeated_teammates"].astype(str) + " / GM+" + observations["other_gm_plus_teammates"].astype(str)
combination_order = [f"R{repeated} / GM+{gm_plus}" for repeated in range(3) for gm_plus in range(3)]
colors = dict(zip(combination_order, plt.get_cmap("tab10").colors))

summary = observations.groupby("combination", observed=True).agg(
    player_games=("game_id", "size"),
    players=("player_hash", "nunique"),
    mean_placing=("placing", "mean"),
    first_place_pct=("placing", lambda values: values.eq(1).mean() * 100),
    mean_rating_change=("rating_change", "mean"),
).reindex(combination_order).reset_index()
summary["player_game_share_pct"] = summary["player_games"] / len(observations) * 100
summary.to_csv(args.output / "top_quarter_prevalence.csv", index=False)

regional = observations.groupby(["region", "combination"], observed=True).agg(
    player_games=("game_id", "size"),
    players=("player_hash", "nunique"),
).reset_index()
regional["regional_player_game_share_pct"] = regional["player_games"] / regional.groupby("region", observed=True)["player_games"].transform("sum") * 100
regional = regional.sort_values(["region", "player_games", "combination"], ascending=[True, False, True])
regional["prevalence_rank"] = regional.groupby("region", observed=True).cumcount() + 1
regional["cumulative_share_before_pct"] = regional.groupby("region", observed=True)["regional_player_game_share_pct"].cumsum() - regional["regional_player_game_share_pct"]
regional["shown_in_chart"] = regional["cumulative_share_before_pct"] < 80
regional.to_csv(args.output / "top_quarter_regional_prevalence.csv", index=False)

shown = regional[regional["shown_in_chart"]]
placement = observations.merge(
    shown[["region", "combination", "player_games", "regional_player_game_share_pct", "prevalence_rank"]],
    on=["region", "combination"], validate="many_to_one",
).groupby(["region", "combination", "player_games", "regional_player_game_share_pct", "prevalence_rank", "placing"], observed=True).agg(
    observations=("game_id", "size"),
    mean_rating_change=("rating_change", "mean"),
).reset_index()
placement.to_csv(args.output / "regional_shown_combination_placement_summary.csv", index=False)

regions = sorted(observations["region"].unique())
fig, axes = plt.subplots(5, 3, figsize=(18, 24))
for axis, region in zip(axes.flat, regions):
    region_lines = placement[placement["region"] == region].sort_values("prevalence_rank")
    for row in region_lines[["combination", "player_games", "regional_player_game_share_pct", "prevalence_rank"]].drop_duplicates().itertuples(index=False):
        line = region_lines[region_lines["combination"] == row.combination].sort_values("placing")
        axis.plot(
            line["placing"], line["mean_rating_change"], marker="o", linewidth=1.8,
            color=colors[row.combination],
            label=f"{row.combination}: {row.regional_player_game_share_pct:.1f}% (n={int(row.player_games):,})",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title(f"{region.upper()} — {observations.loc[observations['region'] == region, 'game_id'].nunique():,} games")
    axis.set_xticks(range(1, 7))
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7.1)
for axis in axes.flat[len(regions):]:
    axis.set_visible(False)
fig.suptitle("Daily top 25% of Challenger: direct rating change after the first season week", fontsize=18, y=0.995)
fig.supxlabel("Team placement")
fig.supylabel("Mean direct rating change")
fig.text(0.5, 0.006, "One observation per player-game. Each panel shows the minimum most-prevalent combinations covering at least 80% of that region. R = repeated teammates; GM+ = other GM+ teammates.", ha="center", fontsize=10)
fig.tight_layout(rect=[0.025, 0.025, 1, 0.985])
fig.savefig(args.output / "top_quarter_challenger_rating_change.png", dpi=180, bbox_inches="tight")
plt.close(fig)

summary_lines = "\n".join(
    f"| {row.combination} | {int(row.player_games):,} | {row.player_game_share_pct:.2f}% | {row.mean_placing:.2f} | {row.first_place_pct:.2f}% |"
    for row in summary.itertuples(index=False)
)
report = f"""# Daily Top-Quarter Challenger Play Patterns

Generated from production match history through {pd.to_datetime(observations['game_ts'], utc=True).max().strftime('%Y-%m-%d')}.

- {len(observations):,} player-games from {observations['player_hash'].nunique():,} players in {observations[['region', 'game_id']].drop_duplicates().shape[0]:,} games across {observations['region'].nunique()} regions.
- One observation is one player in one game. A qualifying three-player stack contributes three observations.
- Membership uses the latest completed regional leaderboard before the game date: ranks 1-50 of 200 Challenger slots, 1-25 of 100, or 1-13 of 50.
- Membership updates daily and uses no later games; exact second-by-second regional rank is not retained.
- The first season week (July 28 through August 3) is excluded; analyzed games begin August 4. ME is omitted because only three post-cutoff days remain.
- Each chart panel shows the minimum number of most-prevalent combinations whose cumulative regional share reaches at least 80%.

| Combination | Player-games | Share | Mean place | First place |
|---|---:|---:|---:|---:|
{summary_lines}

The chart shows mean direct rating change. Repeated teammates use the selected player's production detector value. Other GM+ teammates are that player's teammates whose pre-game tier was Grandmaster or Challenger.
"""
(args.output / "REPORT.md").write_text(report, encoding="utf-8")
print(report)
