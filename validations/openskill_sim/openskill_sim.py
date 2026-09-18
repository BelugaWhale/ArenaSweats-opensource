#!/usr/bin/env python3
import argparse
import importlib
import json
import logging
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ranking_algorithm as ranking_algo

from ranking_algorithm import (
    FRESH_GAP_SATURATION,
    FRESH_GAP_TRIGGER,
    UNBALANCED_PAIR_RATIO_ALPHA,
    apply_teammate_gap_penalty,
    calculate_teammate_gap_modifiers,
    calculate_rating,
    check_for_unbalanced_lobby,
    instantiate_rating_model,
    process_game_ratings,
    _teammate_penalty_scale_gap_pct,
    _unbalanced_grace_reduction_pct,
)

try:
    from rich.console import Console
    from rich.table import Table

    _USE_RICH = True
    _console = Console()
except Exception:
    _USE_RICH = False
    _console = None


def _simple_rating(mu: float, sigma: float) -> float:
    return (mu - 3.0 * sigma) * 75.0


def _clone_rating(model, rating_obj):
    return model.rating(mu=rating_obj.mu, sigma=rating_obj.sigma)


def _live_rating_delta(before_rating, after_rating) -> int:
    return int(calculate_rating(after_rating) - calculate_rating(before_rating))


def _build_arena_format(teams, placings, arena_format_data):
    if arena_format_data:
        team_count = int(arena_format_data["team_count"])
        team_size = int(arena_format_data["team_size"])
        player_count = int(arena_format_data["player_count"])
        placement_count = int(arena_format_data["placement_count"])
        tophalf_cutoff = int(arena_format_data["tophalf_cutoff"])
        name = str(arena_format_data.get("name") or f"{team_size}x{team_count}")
    else:
        team_count = len(teams)
        sizes = {len(team) for team in teams}
        if len(sizes) != 1:
            raise ValueError(f"Inconsistent team sizes: {sorted(sizes)}")
        team_size = next(iter(sizes))
        player_count = team_count * team_size
        placement_count = team_count
        tophalf_cutoff = team_count // 2
        name = f"{team_size}x{team_count}"
    if len(placings) != placement_count or sorted(placings) != list(range(1, placement_count + 1)):
        raise ValueError(f"Placings must be a permutation of [1..{placement_count}]")
    return {
        "name": name,
        "team_count": team_count,
        "team_size": team_size,
        "player_count": player_count,
        "placement_count": placement_count,
        "tophalf_cutoff": tophalf_cutoff,
    }


def _team_stats(ratings):
    mu_sum = sum(r.mu for r in ratings)
    sigma_rms = math.sqrt(sum(r.sigma * r.sigma for r in ratings))
    return mu_sum, sigma_rms, _simple_rating(mu_sum, sigma_rms)


def _team_label(team, names_map):
    return " + ".join(names_map.get(pid) or pid for pid in team)


def _render_table(title, headers, rows):
    if _USE_RICH:
        table = Table(title=title, show_lines=False)
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        _console.print(table)
        return
    print(f"\n{title}")
    print("=" * 120)
    print(" | ".join(headers))
    print("-" * 120)
    for row in rows:
        print(" | ".join(str(cell) for cell in row))


parser = argparse.ArgumentParser(
    description="Run OpenSkill validation sim using either a local JSON input or a direct ClickHouse game pull."
)
parser.add_argument("--input", help="Path to sim input JSON.")
parser.add_argument("input_positional", nargs="?", help=argparse.SUPPRESS)
parser.add_argument("--game-id", help="Target game_id to pull from ClickHouse.")
parser.add_argument("--region", help="Region for ClickHouse tables.")
parser.add_argument("--ch-prefix", help="Optional env prefix override.")
parser.add_argument("--save-input", help="Optional path to save the fetched ClickHouse game in sim-input JSON format.")
parser.add_argument("--export-report", help="Optional path to write a machine-readable JSON report for UI consumers.")
parser.add_argument("--no-charts", action="store_true", help="Disable chart rendering and experiment plots.")
args = parser.parse_args()

if args.input and args.input_positional:
    raise ValueError("Provide input only once: use either --input or positional input path.")
input_arg = args.input or args.input_positional
if input_arg and (args.game_id or args.region or args.ch_prefix):
    raise ValueError("Cannot combine --input with --game-id/--region/--ch-prefix.")
if args.ch_prefix and not (args.game_id and args.region):
    raise ValueError("--ch-prefix requires --game-id and --region.")
if args.save_input and not (args.game_id and args.region):
    raise ValueError("--save-input requires --game-id and --region.")

input_path = None
game_id = None
region = None
if args.game_id or args.region:
    if not args.game_id or not args.region:
        raise ValueError("Both --game-id and --region are required when running from ClickHouse.")
    game_id = args.game_id.strip()
    region = args.region.strip().lower()
    if not game_id:
        raise ValueError("game_id cannot be empty.")
    if not region:
        raise ValueError("region cannot be empty.")
    private_ch_loader = importlib.import_module("openskill_sim_ch_private")
    if not hasattr(private_ch_loader, "load_sim_input_from_clickhouse"):
        raise RuntimeError("openskill_sim_ch_private.py is missing load_sim_input_from_clickhouse(game_id, region, ch_prefix).")
    if not hasattr(private_ch_loader, "REGION_TO_CH_PREFIX"):
        raise RuntimeError("openskill_sim_ch_private.py is missing REGION_TO_CH_PREFIX.")
    ch_prefix = (args.ch_prefix or private_ch_loader.REGION_TO_CH_PREFIX.get(region, "")).strip().upper()
    if not ch_prefix:
        raise ValueError(f"Unknown region '{region}'.")
    data = private_ch_loader.load_sim_input_from_clickhouse(game_id=game_id, region=region, ch_prefix=ch_prefix)
    if args.save_input:
        os.makedirs(os.path.dirname(args.save_input) or ".", exist_ok=True)
        with open(args.save_input, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"Wrote sim input JSON: {args.save_input}")
    print(f"Using game_id={game_id} region={region} from ClickHouse\n")
else:
    input_path = input_arg or os.path.join(_SCRIPT_DIR, os.pardir, "sim_inputs", "sim_inputs_28.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Sim input not found: {input_path}")
    print(f"Using input: {input_path}\n")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

players = data["players"]
teams = data["teams"]
placings = data["placings"]
targets = data.get("targets") or {}
target_games = data.get("target_games")
if not target_games:
    raise ValueError("target_games with modifiers are required in the sim input.")

arena_format = _build_arena_format(teams, placings, data.get("arena_format"))
player_ids = [str(player["id"]) for player in players]
player_id_set = set(player_ids)
if len(players) != arena_format["player_count"]:
    raise ValueError(f"Expected {arena_format['player_count']} players, got {len(players)}")
if len(set(player_ids)) != len(player_ids):
    raise ValueError("Player IDs must be unique.")
if len(teams) != arena_format["team_count"]:
    raise ValueError(f"Expected {arena_format['team_count']} teams, got {len(teams)}")
for index, team in enumerate(teams, start=1):
    if len(team) != arena_format["team_size"]:
        raise ValueError(
            f"Team index {index} must have exactly {arena_format['team_size']} players, got {len(team)}"
        )
used_ids = {str(pid) for team in teams for pid in team}
if used_ids != player_id_set:
    raise ValueError("Teams must reference the player IDs exactly once.")

required_modifier_keys = {"team_gap_pct", "team_gap_scale", "unbalanced_reduction_pct", "is_gm"}
names_map = {str(player["id"]): player.get("name", "") for player in players}
if data.get("recent_teammate_repeat_by_pid") is not None:
    raise ValueError("Legacy recent_teammate_repeat_by_pid input cannot identify which teammate repeated; reload the game.")
repeated_teammate_ids_by_pid = data.get("repeated_teammate_ids_by_pid")
if repeated_teammate_ids_by_pid is not None:
    repeated_teammate_ids_by_pid = {
        str(pid): {str(teammate_id) for teammate_id in teammate_ids}
        for pid, teammate_ids in repeated_teammate_ids_by_pid.items()
    }

gm_set = set()
target_by_pid = {}
afk_pids = set()
afk_protected_pids = set()
for target_game in target_games:
    pid = str(target_game.get("player_id") or "")
    if not pid:
        raise ValueError("Each target_games entry must include player_id.")
    missing = required_modifier_keys - set(target_game.keys())
    if missing:
        raise ValueError(f"Missing modifier fields for player {pid}: {sorted(missing)}")
    target_by_pid[pid] = target_game
    if not isinstance(target_game["is_gm"], bool):
        raise ValueError(f"is_gm must be an explicit boolean for player {pid}.")
    if int(target_game.get("afk_penalty_applied", 0)) == 1:
        afk_pids.add(pid)
    if int(target_game.get("afk_protection_applied", 0)) == 1:
        afk_protected_pids.add(pid)
    if target_game["is_gm"]:
        gm_set.add(pid)
if set(target_by_pid.keys()) != player_id_set:
    raise ValueError("target_games must include every player exactly once.")
if repeated_teammate_ids_by_pid is not None and set(repeated_teammate_ids_by_pid.keys()) != player_id_set:
    raise ValueError("repeated_teammate_ids_by_pid must include every player when provided.")

model = instantiate_rating_model()
before_ratings = {}
for player in players:
    mu = float(player.get("mu", 25.0))
    sigma = float(player.get("sigma", 25.0 / 3.0))
    before_ratings[str(player["id"])] = model.rating(mu=mu, sigma=sigma)

placing_with_team = list(zip(placings, [[str(pid) for pid in team] for team in teams]))
placing_with_team.sort(key=lambda item: item[0])
team_order_ids = [team for _, team in placing_with_team]
teams_ratings = [[before_ratings[pid] for pid in team] for team in team_order_ids]
gm_team_any = []
gm_team_unbalanced_eligible = []
for team in team_order_ids:
    gm_count = sum(1 for pid in team if pid in gm_set)
    gm_team_any.append(gm_count >= 1)
    gm_team_unbalanced_eligible.append(gm_count >= min(2, arena_format["team_size"]))

gap_pct_by_pid, gap_scale_by_pid, unbalanced_grace_blocked_by_team, _, _ = calculate_teammate_gap_modifiers(
    teams_ratings,
    gm_team_any,
    team_order_ids,
    repeated_teammate_ids_by_pid,
)
gm_team_unbalanced_eligible = [
    eligible and not unbalanced_grace_blocked_by_team[team_index]
    for team_index, eligible in enumerate(gm_team_unbalanced_eligible)
]

ranks = list(range(arena_format["team_count"]))
baseline_rated = model.rate(teams_ratings, ranks=ranks)
baseline_after_ratings = dict(before_ratings)
for team_index, team in enumerate(team_order_ids):
    for player_index, pid in enumerate(team):
        baseline_after_ratings[pid] = baseline_rated[team_index][player_index]

ranking_logger = logging.getLogger("openskill_sim.ranking")
ranking_logger.setLevel(logging.WARNING)
if not ranking_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    ranking_logger.addHandler(handler)

players_for_process = []
for placing, team in placing_with_team:
    for pid in team:
        players_for_process.append((pid, placing))

player_ratings_input = {
    pid: model.rating(mu=rating.mu, sigma=rating.sigma)
    for pid, rating in before_ratings.items()
}
source_game_id = str(data.get("source_game_id", "simulation_game"))
success, production_map, production_modifiers = process_game_ratings(
    model,
    players_for_process,
    source_game_id,
    player_ratings_input,
    ranking_logger,
    gm_set if gm_set else set(),
    arena_format=arena_format,
    afk_pids=afk_pids or None,
    afk_protected_pids=afk_protected_pids or None,
    repeated_teammate_ids_by_pid=repeated_teammate_ids_by_pid,
)
if not success:
    raise RuntimeError("Ranking algorithm pipeline failed for the simulated game.")
production_after_ratings = {pid: production_map[pid] for pid in before_ratings}

print("\nTeams & Placings (1 = best)")
print("=" * 40)
for placing, team in placing_with_team:
    print(f"{placing:>2} | {_team_label(team, names_map)}")

print("\nPer-Player Rating Changes")
print("=" * 70)
for pid in sorted(before_ratings.keys()):
    before = before_ratings[pid]
    after = production_after_ratings[pid]
    print(
        f"{names_map.get(pid) or pid:36} | "
        f"mu {before.mu:7.3f} -> {after.mu:7.3f} | "
        f"sigma {before.sigma:6.3f} -> {after.sigma:6.3f} | "
        f"rating {_live_rating_delta(before, after):+d}"
    )

target_comparison_rows = []
mu_rmse = None
sigma_rmse = None
if targets:
    mu_sq_err = []
    sigma_sq_err = []
    for pid in sorted(production_after_ratings.keys()):
        if pid not in targets:
            continue
        calc_rating_obj = production_after_ratings[pid]
        target_mu = float(targets[pid]["mu"])
        target_sigma = float(targets[pid]["sigma"])
        mu_err = calc_rating_obj.mu - target_mu
        sigma_err = calc_rating_obj.sigma - target_sigma
        target_comparison_rows.append(
            {
                "player": names_map.get(pid) or pid,
                "mu_calc": calc_rating_obj.mu,
                "mu_target": target_mu,
                "mu_err": mu_err,
                "sigma_calc": calc_rating_obj.sigma,
                "sigma_target": target_sigma,
                "sigma_err": sigma_err,
            }
        )
        mu_sq_err.append(mu_err * mu_err)
        sigma_sq_err.append(sigma_err * sigma_err)
    if mu_sq_err:
        mu_rmse = math.sqrt(sum(mu_sq_err) / len(mu_sq_err))
    if sigma_sq_err:
        sigma_rmse = math.sqrt(sum(sigma_sq_err) / len(sigma_sq_err))

headers_req = [
    "placing",
    "player",
    "pregame_player_stats",
    "pregame_team_stats",
    "postgame_team_stats",
    "postgame_player_stats",
    "rating_change",
]
rows_req = []
pre_team_stats_by_place = {}
baseline_team_stats_by_place = {}
for placing, team in placing_with_team:
    pre_team = [before_ratings[pid] for pid in team]
    post_team = [baseline_after_ratings[pid] for pid in team]
    pre_team_stats_by_place[placing] = _team_stats(pre_team)
    baseline_team_stats_by_place[placing] = _team_stats(post_team)
    for player_index, pid in enumerate(team):
        before = before_ratings[pid]
        after = baseline_after_ratings[pid]
        pre_team_str = ""
        post_team_str = ""
        if player_index == 0:
            pre_mu_sum, pre_sigma_rms, pre_team_rating = pre_team_stats_by_place[placing]
            post_mu_sum, post_sigma_rms, post_team_rating = baseline_team_stats_by_place[placing]
            pre_team_str = f"{pre_mu_sum:.2f} {pre_sigma_rms:.2f} ({pre_team_rating:.2f})"
            post_team_str = f"{post_mu_sum:.2f} {post_sigma_rms:.2f} ({post_team_rating:.2f})"
        rows_req.append(
            [
                str(placing),
                names_map.get(pid) or pid,
                f"{before.mu:.2f} {before.sigma:.2f} ({calculate_rating(before)})",
                pre_team_str,
                post_team_str,
                f"{after.mu:.2f} {after.sigma:.2f} ({calculate_rating(after)})",
                f"{_live_rating_delta(before, after):+d}",
            ]
        )
_render_table("Requested Summary Table (ordered by placing)", headers_req, rows_req)

baseline_rating_change = {
    pid: _live_rating_delta(before_ratings[pid], baseline_after_ratings[pid])
    for pid in before_ratings
}


def run_pipeline(apply_unbalanced=False, apply_gap_penalty=False, override_ratings=None):
    base_teams = []
    for team in team_order_ids:
        current_team = []
        for pid in team:
            current_team.append(_clone_rating(model, override_ratings.get(pid, before_ratings[pid]) if override_ratings else before_ratings[pid]))
        base_teams.append(current_team)
    adjusted_teams = None
    if apply_unbalanced:
        adjusted_teams, _ = check_for_unbalanced_lobby(
            model,
            base_teams,
            logger=None,
            gm_team_eligible_mask=gm_team_unbalanced_eligible,
        )
    rate_input_final = adjusted_teams if adjusted_teams is not None else base_teams
    rated_teams = model.rate(rate_input_final, ranks=ranks)
    new_teams_local = []
    for team_index in range(len(base_teams)):
        original_team = base_teams[team_index]
        adjusted_team = rate_input_final[team_index]
        rated_team = rated_teams[team_index]
        final_team = []
        for player_index in range(len(original_team)):
            original = original_team[player_index]
            adjusted = adjusted_team[player_index]
            rated = rated_team[player_index]
            final_team.append(
                model.rating(
                    mu=original.mu + (rated.mu - adjusted.mu),
                    sigma=original.sigma + (rated.sigma - adjusted.sigma),
                )
            )
        new_teams_local.append(final_team)
    if apply_gap_penalty:
        apply_teammate_gap_penalty(
            model,
            base_teams,
            new_teams_local,
            team_order_ids,
            gap_scale_by_pid,
        )
    after_local = dict(before_ratings)
    for team_index, team in enumerate(team_order_ids):
        for player_index, pid in enumerate(team):
            after_local[pid] = new_teams_local[team_index][player_index]
    return after_local


def run_unbalanced_only(alpha_value):
    previous_alpha = ranking_algo.UNBALANCED_PAIR_RATIO_ALPHA
    ranking_algo.UNBALANCED_PAIR_RATIO_ALPHA = alpha_value
    try:
        adjusted_teams, reductions = check_for_unbalanced_lobby(
            model,
            teams_ratings,
            logger=None,
            gm_team_eligible_mask=gm_team_unbalanced_eligible,
        )
    finally:
        ranking_algo.UNBALANCED_PAIR_RATIO_ALPHA = previous_alpha
    if adjusted_teams is None:
        adjusted_teams = teams_ratings
        reductions = [0.0] * len(teams_ratings)
    rated_teams = model.rate(adjusted_teams, ranks=ranks)
    after_local = dict(before_ratings)
    for team_index, team in enumerate(team_order_ids):
        for player_index, pid in enumerate(team):
            original = teams_ratings[team_index][player_index]
            adjusted = adjusted_teams[team_index][player_index]
            rated = rated_teams[team_index][player_index]
            after_local[pid] = model.rating(
                mu=original.mu + (rated.mu - adjusted.mu),
                sigma=original.sigma + (rated.sigma - adjusted.sigma),
            )
    return after_local, reductions


gap_new_teams = [[_clone_rating(model, rating) for rating in team] for team in baseline_rated]
apply_teammate_gap_penalty(
    model,
    teams_ratings,
    gap_new_teams,
    team_order_ids,
    gap_scale_by_pid,
)
gap_after = dict(before_ratings)
for team_index, team in enumerate(team_order_ids):
    for player_index, pid in enumerate(team):
        gap_after[pid] = gap_new_teams[team_index][player_index]

headers_gap = [
    "placing",
    "player",
    "pregame_player_stats",
    "pregame_team_stats",
    "postgame_team_stats",
    "postgame_player_stats",
    "rating_change",
    "rating_change_diff",
    "team_gap_repeat",
    "team_gap_norepeat",
]
rows_gap = []
gap_rating_change_by_pid = {}
for team_index, (placing, team) in enumerate(placing_with_team):
    post_gap_team_stats = _team_stats([gap_after[pid] for pid in team])
    for player_index, pid in enumerate(team):
        before = before_ratings[pid]
        after = gap_after[pid]
        repeat_text = ""
        norepeat_text = ""
        if pid in gap_pct_by_pid and pid in gap_scale_by_pid:
            gap_pct = gap_pct_by_pid[pid]
            repeat_scale = _teammate_penalty_scale_gap_pct(gap_pct)
            norepeat_scale = _teammate_penalty_scale_gap_pct(gap_pct, FRESH_GAP_TRIGGER, FRESH_GAP_SATURATION)
            repeat_text = f"{gap_pct * 100:.1f}% (scale: {repeat_scale * 100:.1f}%)"
            norepeat_text = f"{gap_pct * 100:.1f}% (scale: {norepeat_scale * 100:.1f}%)"
        pre_team_str = ""
        post_team_str = ""
        if player_index == 0:
            pre_mu_sum, pre_sigma_rms, pre_team_rating = pre_team_stats_by_place[placing]
            post_mu_sum, post_sigma_rms, post_team_rating = post_gap_team_stats
            pre_team_str = f"{pre_mu_sum:.2f} {pre_sigma_rms:.2f} ({pre_team_rating:.2f})"
            post_team_str = f"{post_mu_sum:.2f} {post_sigma_rms:.2f} ({post_team_rating:.2f})"
        delta_rating = _live_rating_delta(before, after)
        gap_rating_change_by_pid[pid] = delta_rating
        rows_gap.append(
            [
                str(placing),
                names_map.get(pid) or pid,
                f"{before.mu:.2f} {before.sigma:.2f} ({calculate_rating(before)})",
                pre_team_str,
                post_team_str,
                f"{after.mu:.2f} {after.sigma:.2f} ({calculate_rating(after)})",
                f"{delta_rating:+d}",
                f"{delta_rating - baseline_rating_change[pid]:+d}",
                repeat_text,
                norepeat_text,
            ]
        )
_render_table("Gap-Penalty Summary Table (ordered by placing)", headers_gap, rows_gap)

ub_alpha_current = float(UNBALANCED_PAIR_RATIO_ALPHA)
ub_alpha_zero = 0.0
ub_after_alpha0, ub_reductions_alpha0 = run_unbalanced_only(ub_alpha_zero)
ub_after, ub_reductions = run_unbalanced_only(ub_alpha_current)

team_mu_sum_by_place = {}
for placing, team in placing_with_team:
    team_mu_sum_by_place[placing] = sum(before_ratings[pid].mu for pid in team)
sorted_team_mu_sums = sorted(team_mu_sum_by_place.values())
mid = len(sorted_team_mu_sums) // 2
median_team_mu_value = (
    (sorted_team_mu_sums[mid - 1] + sorted_team_mu_sums[mid]) / 2.0
    if len(sorted_team_mu_sums) % 2 == 0
    else sorted_team_mu_sums[mid]
)

headers_ub = [
    "placing",
    "player",
    "pregame_player_stats",
    "pregame_team_stats",
    "postgame_team_stats",
    "postgame_player_stats",
    f"rating_change_base (+ub_a{ub_alpha_current:g}_diff)",
    f"lobby_diff_a{ub_alpha_zero:g}",
    f"lobby_diff_a{ub_alpha_current:g}",
]
headers_ub_alpha_compare = [
    "placing",
    "team",
    "base_gap",
    "spread_ratio",
    "effective_gap_a0",
    f"effective_gap_a{ub_alpha_current:g}",
    "reduction_a0",
    f"reduction_a{ub_alpha_current:g}",
]
rows_ub = []
rows_ub_alpha_compare = []
ub_rating_change_by_pid = {}
for team_index, (placing, team) in enumerate(placing_with_team):
    team_before = teams_ratings[team_index]
    team_mu_sum = team_mu_sum_by_place[placing]
    base_gap_pct = 0.0
    if gm_team_unbalanced_eligible[team_index] and median_team_mu_value > 0.0:
        base_gap_pct = max(0.0, (team_mu_sum - median_team_mu_value) / median_team_mu_value)
    spread_ratio = ranking_algo._unbalanced_team_ratio_scale(team_before, alpha=ub_alpha_current)
    effective_gap_a0 = base_gap_pct
    effective_gap_current = base_gap_pct * spread_ratio
    reduction_a0 = ub_reductions_alpha0[team_index]
    reduction_current = ub_reductions[team_index]
    rows_ub_alpha_compare.append(
        [
            str(placing),
            _team_label(team, names_map),
            f"{base_gap_pct * 100:.2f}%",
            f"{spread_ratio:.4f}",
            f"{effective_gap_a0 * 100:.2f}%",
            f"{effective_gap_current * 100:.2f}%",
            f"{reduction_a0 * 100:.2f}%",
            f"{reduction_current * 100:.2f}%",
        ]
    )
    ub_pre_team_stats = _team_stats(
        [
            model.rating(
                mu=before_ratings[pid].mu * (1.0 - reduction_current),
                sigma=before_ratings[pid].sigma,
            )
            for pid in team
        ]
    )
    ub_post_team_stats = _team_stats([ub_after[pid] for pid in team])
    lobby_diff_alpha0 = ""
    lobby_diff_alpha_current = ""
    if gm_team_unbalanced_eligible[team_index] and base_gap_pct > 0.0:
        lobby_diff_alpha0 = f"{effective_gap_a0 * 100:.1f}% (scale: {(1.0 - reduction_a0) * 100:.1f}%)"
        lobby_diff_alpha_current = f"{effective_gap_current * 100:.1f}% (scale: {(1.0 - reduction_current) * 100:.1f}%)"
    for player_index, pid in enumerate(team):
        before = before_ratings[pid]
        after = ub_after[pid]
        pre_team_str = ""
        post_team_str = ""
        if player_index == 0:
            pre_mu_sum, pre_sigma_rms, pre_team_rating = ub_pre_team_stats
            post_mu_sum, post_sigma_rms, post_team_rating = ub_post_team_stats
            pre_team_str = f"{pre_mu_sum:.2f} {pre_sigma_rms:.2f} ({pre_team_rating:.2f})"
            post_team_str = f"{post_mu_sum:.2f} {post_sigma_rms:.2f} ({post_team_rating:.2f})"
        delta_rating = _live_rating_delta(before, after)
        ub_rating_change_by_pid[pid] = delta_rating
        rows_ub.append(
            [
                str(placing),
                names_map.get(pid) or pid,
                f"{before.mu:.2f} {before.sigma:.2f} ({calculate_rating(before)})",
                pre_team_str,
                post_team_str,
                f"{after.mu:.2f} {after.sigma:.2f} ({calculate_rating(after)})",
                f"{baseline_rating_change[pid]:+d} ({delta_rating - baseline_rating_change[pid]:+d})",
                lobby_diff_alpha0,
                lobby_diff_alpha_current,
            ]
        )
_render_table("Unbalanced-Lobby Summary Table (ordered by placing)", headers_ub, rows_ub)
_render_table("Unbalanced-Lobby Alpha Comparison (team-level)", headers_ub_alpha_compare, rows_ub_alpha_compare)

stack_after_unbalanced = run_pipeline(apply_unbalanced=True, apply_gap_penalty=False)
stack_after_gap = run_pipeline(apply_unbalanced=True, apply_gap_penalty=True)
headers_combo = [
    "placing",
    "player",
    "rating_change_base",
    "unbalanced_grace_effect",
    "rating_change_after_unbalanced",
    "team_gap_effect",
    "rating_change_after_gap",
    "final_rating_change",
]
rows_combo = []
for placing, team in placing_with_team:
    for pid in team:
        base_change = baseline_rating_change[pid]
        unbalanced_change = _live_rating_delta(before_ratings[pid], stack_after_unbalanced[pid])
        final_change = _live_rating_delta(before_ratings[pid], stack_after_gap[pid])
        rows_combo.append(
            [
                str(placing),
                names_map.get(pid) or pid,
                f"{base_change:+d}",
                f"{unbalanced_change - base_change:+d}",
                f"{unbalanced_change:+d}",
                f"{final_change - unbalanced_change:+d}",
                f"{final_change:+d}",
                f"{final_change:+d}",
            ]
        )
_render_table("Stacked-Penalty Summary Table (ordered by placing)", headers_combo, rows_combo)

headers_validation = [
    "placing",
    "player",
    "recorded_rating_change",
    "sim_rating_change",
    "recorded_team_gap_pct",
    "sim_team_gap_pct",
    "recorded_team_gap_scale",
    "sim_team_gap_scale",
    "recorded_unbalanced_reduction_pct",
    "sim_unbalanced_reduction_pct",
]
rows_validation = []
for placing, team in placing_with_team:
    for pid in team:
        target_game = target_by_pid[pid]
        modifier = production_modifiers.get(pid, {})
        rows_validation.append(
            [
                str(placing),
                names_map.get(pid) or pid,
                f"{int(round(float(target_game['rating_change']))):+d}",
                f"{_live_rating_delta(before_ratings[pid], production_after_ratings[pid]):+d}",
                f"{float(target_game['team_gap_pct']):.6f}",
                f"{float(modifier.get('gap_pct', 0.0)):.6f}",
                f"{float(target_game['team_gap_scale']):.6f}",
                f"{float(modifier.get('gap_scale', 1.0)):.6f}",
                f"{float(target_game['unbalanced_reduction_pct']):.6f}",
                f"{float(modifier.get('unbalanced_reduction_pct', 0.0)):.6f}",
            ]
        )
_render_table("Matchhistory Modifier Validation", headers_validation, rows_validation)

teams_placings_rows = [{"place": placing, "players": list(team)} for placing, team in placing_with_team]
per_player_changes_rows = []
for pid in sorted(before_ratings.keys()):
    before = before_ratings[pid]
    after = production_after_ratings[pid]
    per_player_changes_rows.append(
        {
            "player_name": names_map.get(pid) or pid,
            "mu_before": before.mu,
            "mu_after": after.mu,
            "delta_mu": after.mu - before.mu,
            "sigma_before": before.sigma,
            "sigma_after": after.sigma,
            "delta_sigma": after.sigma - before.sigma,
        }
    )

gap_curve_xs = list(range(101))
gap_curve_ys = [_teammate_penalty_scale_gap_pct(x / 100.0) * 100.0 for x in gap_curve_xs]
report_payload = {
    "meta": {
        "source_game_id": source_game_id,
        "input_mode": "clickhouse" if (args.game_id or args.region) else "file",
        "input_path": input_path,
        "region": region,
        "arena_format": arena_format,
        "mu_rmse": mu_rmse,
        "sigma_rmse": sigma_rmse,
        "median_team_mu": median_team_mu_value,
        "recent_teammate_repeat_context": repeated_teammate_ids_by_pid is not None,
        "unbalanced_pair_ratio_alpha": ub_alpha_current,
        "unbalanced_constant": float(ranking_algo.UNBALANCED_TEAM_MU_REDUCTION),
        "unbalanced_3v3_breakpoint": float(ranking_algo.UNBALANCED_3V3_GRACE_BREAKPOINT),
        "unbalanced_3v3_tail_slope": float(ranking_algo.UNBALANCED_3V3_GRACE_TAIL_SLOPE),
        "sigma_floor": float(ranking_algo.SIGMA_FLOOR),
    },
    "teams_placings": teams_placings_rows,
    "per_player_changes": per_player_changes_rows,
    "target_comparison": target_comparison_rows,
    "tables": {
        "requested_summary": {"headers": headers_req, "rows": rows_req},
        "gap_penalty_summary": {"headers": headers_gap, "rows": rows_gap},
        "unbalanced_lobby_summary": {"headers": headers_ub, "rows": rows_ub},
        "unbalanced_lobby_alpha_comparison": {"headers": headers_ub_alpha_compare, "rows": rows_ub_alpha_compare},
        "stacked_penalty_summary": {"headers": headers_combo, "rows": rows_combo},
        "modifier_validation_summary": {"headers": headers_validation, "rows": rows_validation},
    },
    "charts": {
        "gap_penalty_curve": {
            "x_pct": gap_curve_xs,
            "y_multiplier_pct": gap_curve_ys,
        }
    },
}

if args.export_report:
    os.makedirs(os.path.dirname(args.export_report) or ".", exist_ok=True)
    with open(args.export_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
        f.write("\n")
    print(f"Wrote report JSON: {args.export_report}")

if not args.no_charts:
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(gap_curve_xs, gap_curve_ys)
        plt.title("Gap-Penalty Scaling Curve")
        plt.xlabel("Relative teammate mu gap (%)")
        plt.ylabel("Low-impact multiplier (%)")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True)
        plt.show()
    except Exception as exc:
        print(f"Could not render gap-penalty chart: {exc}")

if not args.no_charts and arena_format["name"] == "2x8":
    try:
        import openskill_sim_charts as experiments

        experiments.run_experiment_1(model, players, teams, placings)
        experiments.run_experiment_2(model, players, teams, placings)
    except ImportError as exc:
        print(f"\nNote: Could not import openskill_sim_charts module: {exc}")
    except Exception as exc:
        print(f"\nError running experiments 1/2: {exc}")
elif not args.no_charts:
    print("\nSkipping legacy 2v2 chart experiments for non-2x8 input.")

print("\nDone.")
