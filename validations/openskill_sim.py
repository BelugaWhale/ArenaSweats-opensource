#!/usr/bin/env python3
import argparse
import importlib
import json
import math
import sys
import os
import logging
from datetime import datetime, timezone

# Ensure Unicode table labels print reliably on Windows/non-UTF8 default consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make repository root importable when running this file directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ranking_algorithm as ranking_algo

# Import production helpers/constants so the sim mirrors real logic
from ranking_algorithm import (
    apply_teammate_gap_penalty,
    _teammate_penalty_scale,
    _unbalanced_pair_ratio_scale,
    UNBALANCED_TEAM_MU_REDUCTION,
    UNBALANCED_PAIR_RATIO_ALPHA,
    check_for_unbalanced_lobby,
    process_game_ratings,
    apply_sigma_cap as apply_sigma_cap_algo,
    instantiate_rating_model,
    calculate_rating,
)

try:
    from openskill.models import ThurstoneMostellerFull
except Exception as e:
    print("ERROR: This script requires the 'openskill' package.")
    print("Install with: pip install openskill")
    print(f"Import error: {e}")
    sys.exit(1)

# Pretty tables (optional)
_USE_RICH = True
try:
    from rich.console import Console
    from rich.table import Table

    _console = Console()
except Exception:
    _USE_RICH = False

# Inactivity decay constants (from the newer sim)
SIGMA_DECAY_CLAMP = 6.0
DECAY_GRACE_DAYS = 7
DECAY_FACTOR = 0.99  # 1% rating decay per day


def _simple_rating(mu: float, sigma: float) -> float:
    return (mu - 3.0 * sigma) * 75.0


def _clone_rating(model, rating_obj):
    return model.rating(mu=rating_obj.mu, sigma=rating_obj.sigma)


def _live_rating_delta(before_rating, after_rating) -> int:
    return int(calculate_rating(after_rating) - calculate_rating(before_rating))


# ----------------------------------------------------------------------
# INPUTS
# ----------------------------------------------------------------------
region_to_ch_prefix = {
    "euw": "DBCH",
    "br": "DBCH",
    "sea": "DBCH",
    "ru": "DBCH",
    "me": "DBCH",
    "na": "SCRAPECH",
    "vn": "SCRAPECH",
    "las": "SCRAPECH",
    "tr": "SCRAPECH",
    "oce": "SCRAPECH",
    "kr": "SPLITCH",
    "eune": "SPLITCH",
    "lan": "SPLITCH",
    "tw": "SPLITCH",
    "jp": "SPLITCH",
}

parser = argparse.ArgumentParser(
    description="Run OpenSkill validation sim using either a local JSON input or a direct ClickHouse game pull."
)
parser.add_argument("--input", help="Path to sim input JSON. Defaults to validations/sim_inputs/sim_inputs_28.json.")
parser.add_argument("input_positional", nargs="?", help=argparse.SUPPRESS)
parser.add_argument("--game-id", help="Target game_id to pull from ClickHouse.")
parser.add_argument("--region", help="Region for ClickHouse tables (for example: euw, na, oce).")
parser.add_argument("--ch-prefix", help="Optional env prefix override (for example: DBCH, SCRAPECH, SPLITCH).")
parser.add_argument("--save-input", help="Optional path to save the fetched ClickHouse game in sim-input JSON format.")
parser.add_argument("--export-report", help="Optional path to write a machine-readable JSON report for UI consumers.")
parser.add_argument("--no-charts", action="store_true", help="Disable chart rendering and experiment plots.")
args = parser.parse_args()
if args.input and args.input_positional:
    raise ValueError("Provide input only once: use either --input or positional input path.")
input_arg = args.input or args.input_positional

if input_arg and (args.game_id or args.region or args.ch_prefix):
    raise ValueError("Cannot combine --input with --game-id/--region/--ch-prefix.")
if args.ch_prefix and not (args.game_id or args.region):
    raise ValueError("--ch-prefix requires --game-id and --region.")
if args.save_input and not (args.game_id or args.region):
    raise ValueError("--save-input requires --game-id and --region.")

if args.game_id or args.region:
    if not args.game_id or not args.region:
        raise ValueError("Both --game-id and --region are required when running from ClickHouse.")

    game_id = args.game_id.strip()
    if not game_id:
        raise ValueError("game_id cannot be empty.")
    region = args.region.strip().lower()
    if not region:
        raise ValueError("Region cannot be empty.")

    ch_prefix = (args.ch_prefix or region_to_ch_prefix.get(region, "")).strip().upper()
    if not ch_prefix:
        known_regions = ", ".join(sorted(region_to_ch_prefix.keys()))
        raise ValueError(f"Unknown region '{region}'. Known regions: {known_regions}")

    try:
        private_ch_loader = importlib.import_module("openskill_sim_ch_private")
    except Exception as e:
        raise RuntimeError(
            "ClickHouse mode requires local private module validations/openskill_sim_ch_private.py. "
            "That module must expose load_sim_input_from_clickhouse(game_id, region, ch_prefix). "
            f"Import error: {e}"
        ) from e

    if not hasattr(private_ch_loader, "load_sim_input_from_clickhouse"):
        raise RuntimeError(
            "Module openskill_sim_ch_private is missing load_sim_input_from_clickhouse(game_id, region, ch_prefix)."
        )

    data = private_ch_loader.load_sim_input_from_clickhouse(game_id=game_id, region=region, ch_prefix=ch_prefix)
    if not isinstance(data, dict):
        raise TypeError(
            "load_sim_input_from_clickhouse(game_id, region, ch_prefix) must return a dict compatible with sim input JSON."
        )

    if args.save_input:
        output_dir = os.path.dirname(args.save_input) or "."
        os.makedirs(output_dir, exist_ok=True)
        with open(args.save_input, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"Wrote sim input JSON: {args.save_input}")

    print(f"Using game_id={game_id} region={region} from ClickHouse\n")
else:
    input_path = input_arg or os.path.join(_SCRIPT_DIR, "sim_inputs", "sim_inputs_28.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Sim input not found: {input_path}. "
            "Provide --game-id/--region for direct ClickHouse mode or pass --input."
        )
    print(f"Using input: {input_path}\n")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)


# ----------------------------------------------------------------------
# INPUT VALIDATION
# ----------------------------------------------------------------------
players = data["players"]
names_map = {p.get("id"): p.get("name", "") for p in players}
teams = data["teams"]
placings = data["placings"]
targets = data.get("targets") or {}
target_games = data.get("target_games")
if not target_games:
    raise ValueError("target_games with modifiers are required in the sim input.")

required_modifier_keys = {"sigma_cap_scale", "team_gap_pct", "team_gap_scale", "unbalanced_reduction_pct"}
gm_set = set()
target_game_ids = set()
for tg in target_games:
    pid = tg.get("player_id")
    if not pid:
        raise ValueError("Each target_games entry must include player_id.")
    missing = required_modifier_keys - set(tg.keys())
    if missing:
        raise ValueError(f"Missing modifier fields for player {pid}: {sorted(missing)}")
    sigma_cap_scale = float(tg["sigma_cap_scale"])
    team_gap_pct = float(tg["team_gap_pct"])
    team_gap_scale = float(tg["team_gap_scale"])
    unbalanced_reduction_pct = float(tg["unbalanced_reduction_pct"])
    is_default_sigma = math.isclose(sigma_cap_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    is_default_gap_pct = math.isclose(team_gap_pct, 0.0, rel_tol=1e-9, abs_tol=1e-9)
    is_default_gap_scale = math.isclose(team_gap_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    is_default_unbalanced = math.isclose(unbalanced_reduction_pct, 0.0, rel_tol=1e-9, abs_tol=1e-9)
    target_game_ids.add(pid)
    if not (is_default_sigma and is_default_gap_pct and is_default_gap_scale and is_default_unbalanced):
        gm_set.add(pid)

if target_game_ids != {p["id"] for p in players}:
    raise ValueError("target_games must include modifiers for all 16 players.")

gm_mask_provided = True

# Validate inputs
if len(players) != 16:
    raise ValueError(f"Expected 16 players, got {len(players)}")
if len(teams) != 8:
    raise ValueError(f"Expected 8 teams, got {len(teams)}")
for i, t in enumerate(teams, start=1):
    if len(t) != 2:
        raise ValueError(f"Team index {i} must have exactly 2 players, got {len(t)}")
if sorted(placings) != [1, 2, 3, 4, 5, 6, 7, 8]:
    raise ValueError("Placings must be a permutation of [1..8] (1=best, 8=worst)")

ids = [p["id"] for p in players]
if len(set(ids)) != 16:
    raise ValueError("Player IDs must be unique (16 unique IDs required).")
used = {pid for team in teams for pid in team}
if used != set(ids):
    missing = set(ids) - used
    extra = used - set(ids)
    details = []
    if missing:
        details.append(f"missing in teams: {sorted(missing)}")
    if extra:
        details.append(f"unknown ids in teams: {sorted(extra)}")
    raise ValueError("Teams must reference the 16 players exactly; " + ", ".join(details))

# ----------------------------------------------------------------------
# BASELINE (NO SPECIAL PENALTIES UNLESS RECORDED IN INPUTS)
# ----------------------------------------------------------------------
model = instantiate_rating_model()

before_ratings = {}
for p in players:
    mu = float(p.get("mu", 25.0))
    sigma = float(p.get("sigma", 25.0 / 3.0))
    before_ratings[p["id"]] = model.rating(mu=mu, sigma=sigma)

placing_with_team = list(zip(placings, teams))
placing_with_team.sort(key=lambda x: x[0])

teams_ratings = []
team_order_ids = []
gm_team_any = []
gm_team_both = []
for placing, team in placing_with_team:
    r0 = before_ratings[team[0]]
    r1 = before_ratings[team[1]]
    teams_ratings.append([r0, r1])
    team_order_ids.append(team)
    gm_count = sum(1 for pid in team if pid in gm_set)
    gm_team_any.append(gm_count >= 1)
    gm_team_both.append(gm_count == 2)

ranks = list(range(8))

new_teams = model.rate(teams_ratings, ranks=ranks)
baseline_after_ratings = dict(before_ratings)
for idx, new_pair in enumerate(new_teams):
    pid0, pid1 = team_order_ids[idx]
    baseline_after_ratings[pid0] = new_pair[0]
    baseline_after_ratings[pid1] = new_pair[1]

ranking_logger = logging.getLogger("openskill_sim.ranking")
ranking_logger.setLevel(logging.WARNING)
if not ranking_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    ranking_logger.addHandler(handler)

players_for_process = []
for idx, place in enumerate(placings):
    team = teams[idx]
    for pid in team:
        players_for_process.append((pid, place))

player_ratings_input = {}
for pid, rating in before_ratings.items():
    player_ratings_input[pid] = model.rating(mu=rating.mu, sigma=rating.sigma)

gm_for_process = gm_set if gm_set else set()
game_id = data.get("source_game_id", "simulation_game")
game_ts_raw = data.get("game_ts")
if not game_ts_raw and target_games:
    game_ts_raw = target_games[0].get("game_ts")
if game_ts_raw:
    try:
        game_date = datetime.fromisoformat(game_ts_raw)
    except Exception:
        game_date = datetime.now(timezone.utc)
    else:
        if game_date.tzinfo is None:
            game_date = game_date.replace(tzinfo=timezone.utc)
else:
    game_date = datetime.now(timezone.utc)

success, production_map, _ = process_game_ratings(
    model,
    players_for_process,
    game_id,
    player_ratings_input,
    ranking_logger,
    game_date,
    gm_for_process,
)
if not success:
    raise RuntimeError("Ranking algorithm pipeline failed for the simulated game.")
production_after_ratings = {pid: production_map[pid] for pid in before_ratings}

# ----------------------------------------------------------------------
# BASELINE OUTPUTS
# ----------------------------------------------------------------------
print("\nTeams & Placings (1 = best)")
print("=" * 30)
print("Place | Player A | Player B")
print("------|----------|----------")
for place, team in placing_with_team:
    print(f"{place:5} | {team[0]:8} | {team[1]:8}")

print("\nPer-Player Rating Changes")
print("=" * 45)
print("Player | mu_before | mu_after | Δmu      | sigma_before | sigma_after | Δsigma")
print("-------|-----------|----------|----------|--------------|-------------|--------")
for pid in sorted(before_ratings.keys()):
    b = before_ratings[pid]
    a = production_after_ratings[pid]
    print(
        f"{pid:6} | {b.mu:9.4f} | {a.mu:8.4f} | {a.mu - b.mu:8.5f} | "
        f"{b.sigma:12.4f} | {a.sigma:11.4f} | {a.sigma - b.sigma:7.5f}"
    )

# Target comparison if provided
target_comparison_rows = []
mu_rmse = None
s_rmse = None
if targets:
    print("\nComparison vs. Provided Target End Ratings")
    print("=" * 50)
    print("Player | mu_calc | mu_target | mu_err   | sigma_calc | sigma_target | sigma_err")
    print("-------|---------|-----------|----------|------------|--------------|----------")
    mu_sq_err = []
    s_sq_err = []
    for pid in sorted(production_after_ratings.keys()):
        a = production_after_ratings[pid]
        t = targets.get(pid)
        if not t:
            continue
        t_mu = float(t["mu"])
        if t_mu == 0.0:
            continue
        mu_err = a.mu - t_mu
        s_err = a.sigma - float(t["sigma"])
        target_comparison_rows.append(
            {
                "player": pid,
                "mu_calc": a.mu,
                "mu_target": t_mu,
                "mu_err": mu_err,
                "sigma_calc": a.sigma,
                "sigma_target": float(t["sigma"]),
                "sigma_err": s_err,
            }
        )
        mu_sq_err.append(mu_err**2)
        s_sq_err.append(s_err**2)
        print(
            f"{pid:6} | {a.mu:7.4f} | {t_mu:9.4f} | {mu_err:8.5f} | "
            f"{a.sigma:10.4f} | {float(t['sigma']):12.4f} | {s_err:9.5f}"
        )

    if mu_sq_err or s_sq_err:
        mu_rmse = math.sqrt(sum(mu_sq_err) / max(1, len(mu_sq_err)))
        s_rmse = math.sqrt(sum(s_sq_err) / max(1, len(s_sq_err)))
        print(f"\nRMSE: mu={mu_rmse:.5f}  sigma={s_rmse:.5f}")

# ----------------------------------------------------------------------
# REQUESTED SUMMARY TABLE (BASELINE)
# ----------------------------------------------------------------------
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

pre_team = {}
post_team = {}

for place, team in placing_with_team:
    b0 = before_ratings[team[0]]
    b1 = before_ratings[team[1]]
    a0 = baseline_after_ratings[team[0]]
    a1 = baseline_after_ratings[team[1]]

    pre_mu_sum = b0.mu + b1.mu
    pre_sig_sum = math.hypot(b0.sigma, b1.sigma)
    pre_rate_sum = _simple_rating(pre_mu_sum, pre_sig_sum)

    post_mu_sum = a0.mu + a1.mu
    post_sig_sum = math.hypot(a0.sigma, a1.sigma)
    post_rate_sum = _simple_rating(post_mu_sum, post_sig_sum)

    pre_team[place] = (pre_mu_sum, pre_sig_sum, pre_rate_sum)
    post_team[place] = (post_mu_sum, post_sig_sum, post_rate_sum)

    for idx, pid in enumerate(team):
        name = names_map.get(pid) or pid
        b = before_ratings[pid]
        a = baseline_after_ratings[pid]

        pre_player_rating = calculate_rating(b)
        post_player_rating = calculate_rating(a)
        delta_rating = post_player_rating - pre_player_rating

        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating})"
        post_player_str = f"{a.mu:.2f} {a.sigma:.2f} ({post_player_rating})"

        if idx == 0:
            pre_team_str = f"{pre_mu_sum:.2f} {pre_sig_sum:.2f} ({pre_rate_sum:.2f})"
            post_team_str = f"{post_mu_sum:.2f} {post_sig_sum:.2f} ({post_rate_sum:.2f})"
        else:
            pre_team_str = ""
            post_team_str = ""

        rows_req.append(
            [
                f"{place}",
                f"{name}",
                pre_player_str,
                pre_team_str,
                post_team_str,
                post_player_str,
                f"{delta_rating:+d}",
            ]
        )

if _USE_RICH:
    table_req = Table(title="Requested Summary Table (ordered by placing)", show_lines=False)
    for h in headers_req:
        table_req.add_column(h)
    for r in rows_req:
        table_req.add_row(*r)
    _console.print(table_req)
else:
    print("\nRequested Summary Table (ordered by placing)")
    print("=" * 120)
    print(
        "placing | player                                    | pregame_player_stats        | "
        "pregame_team_stats           | postgame_team_stats          | postgame_player_stats       | rating_change"
    )
    print("-" * 120)
    for r in rows_req:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:26} | {r[3]:26} | {r[4]:26} | {r[5]:26} | {r[6]:>13}"
        )

# Baseline rating_change per player (used for comparisons)
baseline_rating_change = {
    pid: _live_rating_delta(before_ratings[pid], baseline_after_ratings[pid])
    for pid in before_ratings
}


# ----------------------------------------------------------------------
# PIPELINE RUNNER (sigma cap -> unbalanced lobby -> gap penalty)
# ----------------------------------------------------------------------
def run_pipeline(apply_sigma_cap=False, apply_unbalanced=False, apply_gap_penalty=False, override_ratings=None):
    base_teams = []
    current_gm_any = []
    current_gm_both = []

    for idx, (pid0, pid1) in enumerate(team_order_ids):
        r0 = override_ratings.get(pid0, before_ratings[pid0]) if override_ratings else before_ratings[pid0]
        r1 = override_ratings.get(pid1, before_ratings[pid1]) if override_ratings else before_ratings[pid1]
        current_gm_any.append(gm_team_any[idx])
        current_gm_both.append(gm_team_both[idx])
        base_teams.append([_clone_rating(model, r0), _clone_rating(model, r1)])

    if apply_sigma_cap:
        rate_input, _ = apply_sigma_cap_algo(
            model,
            base_teams,
            current_gm_any,
            team_order_ids,
            logger=None,
        )
    else:
        rate_input = []
        for team_pair in base_teams:
            rate_input.append(
                [
                    _clone_rating(model, team_pair[0]),
                    _clone_rating(model, team_pair[1]),
                ]
            )

    adjusted_teams = None
    if apply_unbalanced:
        adjusted_teams, _ = check_for_unbalanced_lobby(
            model,
            rate_input,
            logger=None,
            gm_team_both_mask=current_gm_both if gm_mask_provided else None,
        )
    else:
        adjusted_teams = None
    rate_input_final = adjusted_teams if adjusted_teams is not None else rate_input

    rated_teams = model.rate(rate_input_final, ranks=ranks)

    new_teams_local = []
    for team_idx in range(len(rate_input_final)):
        orig_team = rate_input[team_idx]
        old_final = rate_input_final[team_idx]
        new_from_rate = rated_teams[team_idx]

        final_team = []
        for p_idx in range(len(orig_team)):
            orig = orig_team[p_idx]
            old_adj = old_final[p_idx]
            new_adj = new_from_rate[p_idx]
            delta_mu = new_adj.mu - old_adj.mu
            delta_sigma = new_adj.sigma - old_adj.sigma
            final_team.append(model.rating(mu=orig.mu + delta_mu, sigma=orig.sigma + delta_sigma))
        new_teams_local.append(final_team)

    if apply_gap_penalty:
        apply_teammate_gap_penalty(
            model,
            rate_input,
            new_teams_local,
            logger=None,
            gm_team_any=current_gm_any if gm_mask_provided else None,
        )

    after_local = dict(before_ratings)
    for idx, pair in enumerate(new_teams_local):
        pid0, pid1 = team_order_ids[idx]
        after_local[pid0] = pair[0]
        after_local[pid1] = pair[1]
    return after_local

def run_unbalanced_only(alpha_value):
    prev_alpha = ranking_algo.UNBALANCED_PAIR_RATIO_ALPHA
    ranking_algo.UNBALANCED_PAIR_RATIO_ALPHA = alpha_value
    try:
        ub_rate_input_local, ub_reductions_local = check_for_unbalanced_lobby(
            model,
            teams_ratings,
            logger=None,
            gm_team_both_mask=gm_team_both if gm_mask_provided else None,
        )
    finally:
        ranking_algo.UNBALANCED_PAIR_RATIO_ALPHA = prev_alpha

    if ub_rate_input_local is None:
        ub_new_teams_local = new_teams
        ub_reductions_local = [0.0] * len(team_order_ids)
    else:
        ub_rated_from_adjusted_local = model.rate(ub_rate_input_local, ranks=ranks)
        ub_new_teams_local = []

        for team_idx in range(len(teams_ratings)):
            orig_team = teams_ratings[team_idx]
            adj_old_team = ub_rate_input_local[team_idx]
            adj_new_team = ub_rated_from_adjusted_local[team_idx]

            final_team = []
            for p_idx in range(len(orig_team)):
                orig = orig_team[p_idx]
                old_adj = adj_old_team[p_idx]
                new_adj = adj_new_team[p_idx]
                delta_mu = new_adj.mu - old_adj.mu
                delta_sigma = new_adj.sigma - old_adj.sigma
                final_team.append(model.rating(mu=orig.mu + delta_mu, sigma=orig.sigma + delta_sigma))
            ub_new_teams_local.append(final_team)
        if ub_reductions_local is None:
            ub_reductions_local = [0.0] * len(team_order_ids)

    ub_after_local = dict(before_ratings)
    for idx, pair in enumerate(ub_new_teams_local):
        pid0, pid1 = team_order_ids[idx]
        ub_after_local[pid0] = pair[0]
        ub_after_local[pid1] = pair[1]
    return ub_after_local, ub_reductions_local


# ----------------------------------------------------------------------
# SIGMA-CAP summary table (ratio-preserving, exact team-sigma target)
# ----------------------------------------------------------------------
base_mu_delta_by_pid = {pid: baseline_after_ratings[pid].mu - before_ratings[pid].mu for pid in before_ratings}
base_team_mu_delta_by_place = {place: sum(base_mu_delta_by_pid[pid] for pid in team) for place, team in placing_with_team}

scaled_input_by_pid = {}
scaled_pre_team = {}

sigma_cap_base_pairs = []
for pid0, pid1 in team_order_ids:
    sigma_cap_base_pairs.append([_clone_rating(model, before_ratings[pid0]), _clone_rating(model, before_ratings[pid1])])

scaled_pairs, sigma_scale_by_pid = apply_sigma_cap_algo(
    model,
    sigma_cap_base_pairs,
    gm_team_any,
    team_order_ids,
    logger=None,
)

for idx, team in enumerate(team_order_ids):
    place = placing_with_team[idx][0]
    p0, p1 = team
    c0, c1 = scaled_pairs[idx]
    scaled_input_by_pid[p0] = c0
    scaled_input_by_pid[p1] = c1

    mu_sum = before_ratings[p0].mu + before_ratings[p1].mu
    sig_rms = math.hypot(c0.sigma, c1.sigma)
    team_rt = _simple_rating(mu_sum, sig_rms)
    scaled_pre_team[place] = (mu_sum, sig_rms, team_rt)

scaled_pairs_for_rate = [
    [
        _clone_rating(model, pair[0]),
        _clone_rating(model, pair[1]),
    ]
    for pair in scaled_pairs
]

scaled_new_pairs = model.rate(scaled_pairs_for_rate, ranks=ranks)

scaled_after = dict(before_ratings)
for idx, new_pair in enumerate(scaled_new_pairs):
    pid0, pid1 = team_order_ids[idx]
    scaled_after[pid0] = new_pair[0]
    scaled_after[pid1] = new_pair[1]

scaled_post_team = {}
for place, team in placing_with_team:
    a0 = scaled_after[team[0]]
    a1 = scaled_after[team[1]]
    mu_sum = a0.mu + a1.mu
    sig_rms = math.hypot(a0.sigma, a1.sigma)
    team_rt = _simple_rating(mu_sum, sig_rms)
    scaled_post_team[place] = (mu_sum, sig_rms, team_rt)

team_scale = {}
for place, team in placing_with_team:
    base_team_dmu = base_team_mu_delta_by_place[place]
    new_team_dmu = sum(scaled_after[pid].mu - scaled_input_by_pid[pid].mu for pid in team)

    if abs(base_team_dmu) < 1e-12:
        raise ValueError(f"Baseline team Δμ ~ 0 at placing {place}; cannot preserve split under scaling.")

    team_scale[place] = new_team_dmu / base_team_dmu

final_mu = {}
final_sigma = {}
for place, team in placing_with_team:
    r_team = team_scale[place]
    for pid in team:
        final_mu[pid] = before_ratings[pid].mu + r_team * base_mu_delta_by_pid[pid]
        final_sigma_delta = scaled_after[pid].sigma - scaled_input_by_pid[pid].sigma
        final_sigma[pid] = before_ratings[pid].sigma + final_sigma_delta

headers_sigma = [
    "placing",
    "player",
    "pregame_player_stats",
    "pregame_team_stats",
    "postgame_team_stats",
    "postgame_player_stats",
    "rating_change",
    "rating_change_diff",
    "sigma_cap_scale",
]
rows_sigma = []
sigma_rating_change_by_pid = {}
sigma_rating_diff_by_pid = {}

for place, team in placing_with_team:
    for idx, pid in enumerate(team):
        name = names_map.get(pid) or pid

        b = before_ratings[pid]
        pre_player_rating = calculate_rating(b)
        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating})"

        mu_f = final_mu[pid]
        sg_f = final_sigma[pid]
        post_player_rating = calculate_rating(model.rating(mu=mu_f, sigma=sg_f))
        post_player_str = f"{mu_f:.2f} {sg_f:.2f} ({post_player_rating})"

        delta_rating = post_player_rating - pre_player_rating
        delta_diff = delta_rating - baseline_rating_change[pid]
        sigma_rating_change_by_pid[pid] = delta_rating
        sigma_rating_diff_by_pid[pid] = delta_diff

        if idx == 0:
            mu_t, sg_t, rt_t = scaled_pre_team[place]
            pre_team_str = f"{mu_t:.2f} {sg_t:.2f} ({rt_t:.2f})"
            mu_u, sg_u, rt_u = scaled_post_team[place]
            sigma_team_str = f"{mu_u:.2f} {sg_u:.2f} ({rt_u:.2f})"
        else:
            pre_team_str = ""
            sigma_team_str = ""

        scale_val = sigma_scale_by_pid.get(pid, 1.0)

        rows_sigma.append(
            [
                f"{place}",
                f"{name}",
                pre_player_str,
                pre_team_str,
                sigma_team_str,
                post_player_str,
                f"{delta_rating:+d}",
                f"{delta_diff:+d}",
                f"{scale_val:.4f}",
            ]
        )

if _USE_RICH:
    table_sigma = Table(title="SIGMA-CAP summary table (ordered by placing)", show_lines=False)
    for h in headers_sigma:
        table_sigma.add_column(h)
    for r in rows_sigma:
        table_sigma.add_row(*r)
    _console.print(table_sigma)
else:
    print("\nSIGMA-CAP summary table (ordered by placing)")
    print("=" * 220)
    print(
        "placing | player                                    | pregame_player_stats        | pregame_team_stats           | "
        "postgame_team_stats            | postgame_player_stats         | rating_change | rating_change_diff | sigma_cap_scale"
    )
    print("-" * 220)
    for r in rows_sigma:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:26} | {r[3]:26} | {r[4]:26} | {r[5]:26} | {r[6]:>23} | {r[7]:>18} | {r[8]:>13}"
        )

# ----------------------------------------------------------------------
# GAP-PENALTY summary table (ordered by placing)
# ----------------------------------------------------------------------
gap_new_teams = [pair.copy() for pair in new_teams]
apply_teammate_gap_penalty(
    model,
    teams_ratings,
    gap_new_teams,
    logger=None,
    gm_team_any=gm_team_any if gm_mask_provided else None,
    team_player_ids=team_order_ids if gm_mask_provided else None,
)

gap_after = dict(before_ratings)
for idx, pair in enumerate(gap_new_teams):
    pid0, pid1 = team_order_ids[idx]
    gap_after[pid0] = pair[0]
    gap_after[pid1] = pair[1]

gap_post_team = {}
for place, team in placing_with_team:
    a0 = gap_after[team[0]]
    a1 = gap_after[team[1]]
    mu_sum = a0.mu + a1.mu
    sig_rms = math.hypot(a0.sigma, a1.sigma)
    team_rt = _simple_rating(mu_sum, sig_rms)
    gap_post_team[place] = (mu_sum, sig_rms, team_rt)

headers_gap = [
    "placing",
    "player",
    "pregame_player_stats",
    "pregame_team_stats",
    "postgame_team_stats",
    "postgame_player_stats",
    "rating_change",
    "rating_change_diff",
    "mu_gap",
]
rows_gap = []
gap_rating_change_by_pid = {}
gap_rating_diff_by_pid = {}

for place, team in placing_with_team:
    r0 = before_ratings[team[0]]
    r1 = before_ratings[team[1]]
    show_gap = gm_team_any[place - 1] if gm_mask_provided else True
    if show_gap:
        if r0.mu >= r1.mu:
            mu_hi, mu_lo = r0.mu, r1.mu
        else:
            mu_hi, mu_lo = r1.mu, r0.mu
        if mu_hi > 0.0:
            gap_pct_val = max(0.0, min(1.0, 1.0 - (mu_lo / mu_hi)))
            scale_val = _teammate_penalty_scale(gap_pct_val)
        else:
            gap_pct_val = 0.0
            scale_val = 1.0
    else:
        gap_pct_val = None
        scale_val = None

    for idx, pid in enumerate(team):
        name = names_map.get(pid) or pid

        b = before_ratings[pid]
        pre_player_rating = calculate_rating(b)
        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating})"

        a_gap = gap_after[pid]
        post_player_gap_rating = calculate_rating(a_gap)
        post_player_gap_str = f"{a_gap.mu:.2f} {a_gap.sigma:.2f} ({post_player_gap_rating})"

        delta_rating = post_player_gap_rating - pre_player_rating
        delta_diff = delta_rating - baseline_rating_change[pid]
        gap_rating_change_by_pid[pid] = delta_rating
        gap_rating_diff_by_pid[pid] = delta_diff

        if idx == 0:
            mu_pre_t, sg_pre_t, rt_pre_t = pre_team[place]
            pre_team_str = f"{mu_pre_t:.2f} {sg_pre_t:.2f} ({rt_pre_t:.2f})"
            mu_post_t, sg_post_t, rt_post_t = post_team[place]
            post_team_base_str = f"{mu_post_t:.2f} {sg_post_t:.2f} ({rt_post_t:.2f})"
            mu_gap_str = f"{gap_pct_val*100:.1f}% (scale: {scale_val*100:.1f}%)" if show_gap and gap_pct_val is not None else ""
        else:
            pre_team_str = ""
            post_team_base_str = ""
            mu_gap_str = ""

        rows_gap.append(
            [
                f"{place}",
                f"{name}",
                pre_player_str,
                pre_team_str,
                post_team_base_str,
                post_player_gap_str,
                f"{delta_rating:+d}",
                f"{delta_diff:+d}",
                mu_gap_str,
            ]
        )

if _USE_RICH:
    table_gap = Table(title="GAP-PENALTY summary table (ordered by placing)", show_lines=False)
    for h in headers_gap:
        table_gap.add_column(h)
    for r in rows_gap:
        table_gap.add_row(*r)
    _console.print(table_gap)
else:
    print("\nGAP-PENALTY summary table (ordered by placing)")
    print("=" * 220)
    print(
        "placing | player | pregame_player_stats | pregame_team_stats | "
        "postgame_team_stats | postgame_player_stats | rating_change | rating_change_diff | mu_gap"
    )
    print("-" * 220)
    for r in rows_gap:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:26} | {r[3]:26} | {r[4]:26} | "
            f"{r[5]:26} | {r[6]:>13} | {r[7]:>18} | {r[8]:>20}"
        )

# ----------------------------------------------------------------------
# UNBALANCED-LOBBY summary table (ordered by placing)
# ----------------------------------------------------------------------
ub_alpha_current = float(UNBALANCED_PAIR_RATIO_ALPHA)
ub_alpha_zero = 0.0
_ub_after_alpha0, ub_reductions_alpha0 = run_unbalanced_only(ub_alpha_zero)
ub_after, ub_reductions = run_unbalanced_only(ub_alpha_current)

ub_post_team = {}
for place, team in placing_with_team:
    a0 = ub_after[team[0]]
    a1 = ub_after[team[1]]
    mu_sum = a0.mu + a1.mu
    sig_rms = math.hypot(a0.sigma, a1.sigma)
    team_rt = _simple_rating(mu_sum, sig_rms)
    ub_post_team[place] = (mu_sum, sig_rms, team_rt)

team_mu_sum_by_place = {}
for place, team in placing_with_team:
    b0 = before_ratings[team[0]]
    b1 = before_ratings[team[1]]
    team_mu_sum_by_place[place] = b0.mu + b1.mu

sorted_team_mu_sums = sorted(team_mu_sum_by_place.values(), reverse=True)
if sorted_team_mu_sums:
    mid = len(sorted_team_mu_sums) // 2
    if len(sorted_team_mu_sums) % 2 == 1:
        median_team_mu_value = sorted_team_mu_sums[mid]
    else:
        median_team_mu_value = (sorted_team_mu_sums[mid - 1] + sorted_team_mu_sums[mid]) / 2.0
else:
    median_team_mu_value = 0.0

ub_pre_team = {}
for idx, (place, team) in enumerate(placing_with_team):
    reduction_pct = ub_reductions[idx] if ub_reductions and idx < len(ub_reductions) else 0.0
    mu_sum = before_ratings[team[0]].mu + before_ratings[team[1]].mu
    adj_mu_sum = mu_sum * (1.0 - reduction_pct)
    sig_rms = math.hypot(before_ratings[team[0]].sigma, before_ratings[team[1]].sigma)
    team_rt = _simple_rating(adj_mu_sum, sig_rms)
    ub_pre_team[place] = (adj_mu_sum, sig_rms, team_rt)

team_base_gap_by_place = {}
team_pair_ratio_by_place = {}
team_effective_gap_alpha0_by_place = {}
team_effective_gap_alpha_current_by_place = {}
team_reduction_alpha0_by_place = {}
team_reduction_alpha_current_by_place = {}
for idx, (place, team) in enumerate(placing_with_team):
    show_lobby = gm_team_both[place - 1] if gm_mask_provided else True
    team_mu_sum = team_mu_sum_by_place[place]
    if show_lobby and median_team_mu_value > 0.0:
        base_gap_pct = max(0.0, (team_mu_sum - median_team_mu_value) / median_team_mu_value)
    else:
        base_gap_pct = 0.0
    pair_ratio_scale_current = _unbalanced_pair_ratio_scale(teams_ratings[idx], alpha=ub_alpha_current)
    pair_ratio_value = 0.0
    if teams_ratings[idx][0].mu >= teams_ratings[idx][1].mu:
        mu_hi = teams_ratings[idx][0].mu
        mu_lo = teams_ratings[idx][1].mu
    else:
        mu_hi = teams_ratings[idx][1].mu
        mu_lo = teams_ratings[idx][0].mu
    if mu_hi > 0.0:
        pair_ratio_value = mu_lo / mu_hi

    effective_gap_alpha0 = base_gap_pct
    effective_gap_alpha_current = base_gap_pct * pair_ratio_scale_current

    team_base_gap_by_place[place] = base_gap_pct
    team_pair_ratio_by_place[place] = pair_ratio_value
    team_effective_gap_alpha0_by_place[place] = effective_gap_alpha0
    team_effective_gap_alpha_current_by_place[place] = effective_gap_alpha_current
    team_reduction_alpha0_by_place[place] = ub_reductions_alpha0[idx] if idx < len(ub_reductions_alpha0) else 0.0
    team_reduction_alpha_current_by_place[place] = ub_reductions[idx] if idx < len(ub_reductions) else 0.0

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
rows_ub = []
ub_rating_change_by_pid = {}
ub_rating_diff_by_pid = {}

for place, team in placing_with_team:
    show_lobby = gm_team_both[place - 1] if gm_mask_provided else True
    if show_lobby and team_base_gap_by_place[place] > 0.0:
        scale_alpha0 = 1.0 - team_reduction_alpha0_by_place[place]
        scale_alpha_current = 1.0 - team_reduction_alpha_current_by_place[place]
        lobby_diff_alpha0_str = (
            f"{team_base_gap_by_place[place] * 100:.1f}% "
            f"(scale: {scale_alpha0 * 100:.1f}%)"
        )
        lobby_diff_alpha_current_str = (
            f"{team_effective_gap_alpha_current_by_place[place] * 100:.1f}% "
            f"(scale: {scale_alpha_current * 100:.1f}%)"
        )
    else:
        lobby_diff_alpha0_str = ""
        lobby_diff_alpha_current_str = ""

    for idx, pid in enumerate(team):
        name = names_map.get(pid) or pid

        b = before_ratings[pid]
        pre_player_rating = calculate_rating(b)
        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating})"

        a_ub = ub_after[pid]
        post_player_ub_rating = calculate_rating(a_ub)
        post_player_ub_str = f"{a_ub.mu:.2f} {a_ub.sigma:.2f} ({post_player_ub_rating})"

        delta_rating = post_player_ub_rating - pre_player_rating
        delta_diff = delta_rating - baseline_rating_change[pid]
        ub_rating_change_by_pid[pid] = delta_rating
        ub_rating_diff_by_pid[pid] = delta_diff

        if idx == 0:
            mu_pre_t, sg_pre_t, rt_pre_t = ub_pre_team[place]
            pre_team_str = f"{mu_pre_t:.2f} {sg_pre_t:.2f} ({rt_pre_t:.2f})"

            mu_post_t, sg_post_t, rt_post_t = ub_post_team[place]
            post_team_str = f"{mu_post_t:.2f} {sg_post_t:.2f} ({rt_post_t:.2f})"
        else:
            pre_team_str = ""
            post_team_str = ""

        rows_ub.append(
            [
                f"{place}",
                f"{name}",
                pre_player_str,
                pre_team_str,
                post_team_str,
                post_player_ub_str,
                f"{baseline_rating_change[pid]:+d} ({delta_diff:+d})",
                lobby_diff_alpha0_str,
                lobby_diff_alpha_current_str,
            ]
        )

headers_ub_alpha_compare = [
    "placing",
    "team",
    "base_gap",
    "pair_ratio",
    "effective_gap_a0",
    f"effective_gap_a{ub_alpha_current:g}",
    "reduction_a0",
    f"reduction_a{ub_alpha_current:g}",
]
rows_ub_alpha_compare = []
for place, team in placing_with_team:
    team_name = f"{names_map.get(team[0]) or team[0]} + {names_map.get(team[1]) or team[1]}"
    rows_ub_alpha_compare.append(
        [
            f"{place}",
            team_name,
            f"{team_base_gap_by_place[place] * 100:.2f}%",
            f"{team_pair_ratio_by_place[place]:.4f}",
            f"{team_effective_gap_alpha0_by_place[place] * 100:.2f}%",
            f"{team_effective_gap_alpha_current_by_place[place] * 100:.2f}%",
            f"{team_reduction_alpha0_by_place[place] * 100:.2f}%",
            f"{team_reduction_alpha_current_by_place[place] * 100:.2f}%",
        ]
    )

if _USE_RICH:
    table_ub = Table(
        title=(
            f"UNBALANCED-LOBBY summary table (ordered by placing)\n"
            f"median team mu: {median_team_mu_value:.2f}, alpha={ub_alpha_current:g}"
        ),
        show_lines=False,
    )
    for h in headers_ub:
        table_ub.add_column(h)
    for r in rows_ub:
        table_ub.add_row(*r)
    _console.print(table_ub)
    table_ub_alpha_compare = Table(
        title=(
            "UNBALANCED-LOBBY alpha comparison (team-level)\n"
            f"alpha0={ub_alpha_zero:g}, alpha_current={ub_alpha_current:g}"
        ),
        show_lines=False,
    )
    for h in headers_ub_alpha_compare:
        table_ub_alpha_compare.add_column(h)
    for r in rows_ub_alpha_compare:
        table_ub_alpha_compare.add_row(*r)
    _console.print(table_ub_alpha_compare)
else:
    print("\nUNBALANCED-LOBBY summary table (ordered by placing)")
    print("=" * 220)
    print(f"median team mu: {median_team_mu_value:.2f}, alpha={ub_alpha_current:g}")
    print(
        "placing | player | pregame_player_stats | pregame_team_stats | "
        "postgame_team_stats | postgame_player_stats | "
        f"rating_change_base (+ub_a{ub_alpha_current:g}_diff) | "
        f"lobby_diff_a{ub_alpha_zero:g} | lobby_diff_a{ub_alpha_current:g}"
    )
    print("-" * 220)
    for r in rows_ub:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:26} | {r[3]:26} | {r[4]:26} | "
            f"{r[5]:26} | {r[6]:>18} | {r[7]:>24} | {r[8]:>24}"
        )
    print("\nUNBALANCED-LOBBY alpha comparison (team-level)")
    print("=" * 180)
    print(f"alpha0={ub_alpha_zero:g}, alpha_current={ub_alpha_current:g}")
    print(
        "placing | team | base_gap | pair_ratio | effective_gap_a0 | "
        f"effective_gap_a{ub_alpha_current:g} | reduction_a0 | reduction_a{ub_alpha_current:g}"
    )
    print("-" * 180)
    for r in rows_ub_alpha_compare:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:>10} | {r[3]:>10} | {r[4]:>16} | "
            f"{r[5]:>18} | {r[6]:>12} | {r[7]:>18}"
        )

# ----------------------------------------------------------------------
# COMBINED STACKED-PENALTY table (sigma-cap -> unbalanced lobby -> gap penalty)
# ----------------------------------------------------------------------
final_rating_change_by_pid = {
    pid: _live_rating_delta(before_ratings[pid], production_after_ratings[pid])
    for pid in before_ratings
}

headers_combo = [
    "placing",
    "player",
    "rating_change_base",
    "sigma_cap_effect",
    "rating_change_after_sigma",
    "unbalanced_grace_effect",
    "rating_change_after_unbalanced",
    "team_gap_effect",
    "rating_change_after_gap",
    "final_rating_change",
]
rows_combo = []

for place, team in placing_with_team:
    for pid in team:
        name = names_map.get(pid) or pid

        base_change = baseline_rating_change[pid]
        sigma_change = sigma_rating_change_by_pid[pid]
        sigma_penalty = sigma_rating_diff_by_pid[pid]
        unbalanced_change = ub_rating_change_by_pid[pid]
        unbalanced_penalty = ub_rating_diff_by_pid[pid]
        gap_change = gap_rating_change_by_pid[pid]
        gap_penalty = gap_rating_diff_by_pid[pid]
        final_change = final_rating_change_by_pid[pid]

        rows_combo.append(
            [
                f"{place}",
                f"{name}",
                f"{base_change:+d}",
                f"{sigma_penalty:+d}",
                f"{sigma_change:+d}",
                f"{unbalanced_penalty:+d}",
                f"{unbalanced_change:+d}",
                f"{gap_penalty:+d}",
                f"{gap_change:+d}",
                f"{final_change:+d}",
            ]
        )

if _USE_RICH:
    combo_table = Table(title="STACKED-PENALTY summary table (ordered by placing)", show_lines=False)
    for h in headers_combo:
        combo_table.add_column(h)
    for r in rows_combo:
        combo_table.add_row(*r)
    _console.print(combo_table)
else:
    print("\nSTACKED-PENALTY summary table (ordered by placing)")
    print("=" * 220)
    print(
        "placing | player | rating_change_base | sigma_cap_effect | rating_change_after_sigma | "
        "unbalanced_grace_effect | rating_change_after_unbalanced | team_gap_effect | rating_change_after_gap | final_rating_change"
    )
    print("-" * 220)
    for r in rows_combo:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:>18} | {r[3]:>16} | {r[4]:>25} | "
            f"{r[5]:>23} | {r[6]:>29} | {r[7]:>16} | {r[8]:>23} | {r[9]:>19}"
        )

teams_placings_rows = []
for place, team in placing_with_team:
    teams_placings_rows.append({"place": place, "player_a": team[0], "player_b": team[1]})

per_player_changes_rows = []
for pid in sorted(before_ratings.keys()):
    b = before_ratings[pid]
    a = production_after_ratings[pid]
    per_player_changes_rows.append(
        {
            "player_name": names_map.get(pid) or pid,
            "mu_before": b.mu,
            "mu_after": a.mu,
            "delta_mu": a.mu - b.mu,
            "sigma_before": b.sigma,
            "sigma_after": a.sigma,
            "delta_sigma": a.sigma - b.sigma,
        }
    )

gap_curve_xs = list(range(101))
gap_curve_ys = [_teammate_penalty_scale(x / 100.0) * 100.0 for x in gap_curve_xs]

report_payload = {
    "meta": {
        "source_game_id": game_id,
        "input_mode": "clickhouse" if (args.game_id or args.region) else "file",
        "input_path": None if (args.game_id or args.region) else input_path,
        "region": region if (args.game_id or args.region) else None,
        "mu_rmse": mu_rmse,
        "sigma_rmse": s_rmse,
        "median_team_mu": median_team_mu_value,
        "unbalanced_pair_ratio_alpha": ub_alpha_current,
    },
    "teams_placings": teams_placings_rows,
    "per_player_changes": per_player_changes_rows,
    "target_comparison": target_comparison_rows,
    "tables": {
        "requested_summary": {"headers": headers_req, "rows": rows_req},
        "sigma_cap_summary": {"headers": headers_sigma, "rows": rows_sigma},
        "gap_penalty_summary": {"headers": headers_gap, "rows": rows_gap},
        "unbalanced_lobby_summary": {"headers": headers_ub, "rows": rows_ub},
        "unbalanced_lobby_alpha_comparison": {"headers": headers_ub_alpha_compare, "rows": rows_ub_alpha_compare},
        "stacked_penalty_summary": {"headers": headers_combo, "rows": rows_combo},
    },
    "charts": {
        "gap_penalty_curve": {
            "x_pct": gap_curve_xs,
            "y_multiplier_pct": gap_curve_ys,
        }
    },
}

if args.export_report:
    output_dir = os.path.dirname(args.export_report) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.export_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
        f.write("\n")
    print(f"Wrote report JSON: {args.export_report}")

# ----------------------------------------------------------------------
# GAP-PENALTY CURVE chart
# ----------------------------------------------------------------------
if not args.no_charts:
    try:
        import matplotlib.pyplot as plt

        xs = gap_curve_xs
        ys = gap_curve_ys

        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys)
        plt.title("GAP-PENALTY Scaling Curve")
        plt.xlabel("Relative teammate μ gap (%)")
        plt.ylabel("Low-impact multiplier (%)")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Could not render GAP-PENALTY chart: {e}")

# ----------------------------------------------------------------------
# Experiments 1 & 2 (Sigma & Mu impact curves)
# ----------------------------------------------------------------------
if not args.no_charts:
    try:
        import openskill_sim_charts as experiments

        experiments.run_experiment_1(model, players, teams, placings)
        experiments.run_experiment_2(model, players, teams, placings)

    except ImportError as e:
        print(f"\nNote: Could not import openskill_sim_charts module: {e}")
        print("Experiments 1 and 2 will be skipped.")
    except Exception as e:
        print(f"\nError running experiments 1/2: {e}")

# ----------------------------------------------------------------------
# EXPERIMENT 3: DECAY & RECOVERY ANALYSIS (Player 8)
# ----------------------------------------------------------------------
TARGET_ID = "p8"

if not args.no_charts:
    print(f"\nRunning Experiment 3: Inactivity Decay Analysis for {TARGET_ID}...")

if not args.no_charts and TARGET_ID not in before_ratings:
    print(f"Skipping Experiment 3: Player {TARGET_ID} not found in input.")
if not args.no_charts and TARGET_ID in before_ratings:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping Experiment 3: matplotlib not available ({e})")
    else:
        p_base = before_ratings[TARGET_ID]

        # Use full production-ish pipeline as the baseline for comparison
        prod_after = run_pipeline(apply_sigma_cap=True, apply_unbalanced=True, apply_gap_penalty=True)
        prod_rating_change = {
            pid: _simple_rating(prod_after[pid].mu, prod_after[pid].sigma)
            - _simple_rating(before_ratings[pid].mu, before_ratings[pid].sigma)
            for pid in before_ratings
        }
        base_change = prod_rating_change[TARGET_ID]

        if abs(base_change) < 0.01:
            print(f"Baseline change for {TARGET_ID} is too small ({base_change:.4f}) to calculate multipliers.")
        else:
            results_days = []
            results_peak = []
            results_avg = []
            results_games = []

            for day in range(31):
                if day <= DECAY_GRACE_DAYS:
                    start_sigma = p_base.sigma
                else:
                    base_r = _simple_rating(p_base.mu, p_base.sigma)
                    decayed_r = base_r * (DECAY_FACTOR ** (day - DECAY_GRACE_DAYS))
                    calc_sigma = (p_base.mu - (decayed_r / 75.0)) / 3.0
                    start_sigma = min(calc_sigma, SIGMA_DECAY_CLAMP)

                p_decayed = model.rating(mu=p_base.mu, sigma=start_sigma)

                after_decay = run_pipeline(
                    apply_sigma_cap=True,
                    apply_unbalanced=True,
                    apply_gap_penalty=True,
                    override_ratings={TARGET_ID: p_decayed},
                )

                p_after = after_decay[TARGET_ID]
                change_decayed = _simple_rating(p_after.mu, p_after.sigma) - _simple_rating(
                    p_decayed.mu, p_decayed.sigma
                )
                peak_multiplier = change_decayed / base_change

                curr_sigma = start_sigma
                recovery_games = 0
                total_mult = 0.0

                while curr_sigma > p_base.sigma + 0.01:
                    recovery_games += 1
                    p_step_in = model.rating(mu=p_base.mu, sigma=curr_sigma)

                    after_step = run_pipeline(
                        apply_sigma_cap=True,
                        apply_unbalanced=True,
                        apply_gap_penalty=True,
                        override_ratings={TARGET_ID: p_step_in},
                    )
                    p_step_out = after_step[TARGET_ID]

                    step_change = _simple_rating(p_step_out.mu, p_step_out.sigma) - _simple_rating(
                        p_step_in.mu, p_step_in.sigma
                    )
                    step_mult = step_change / base_change
                    total_mult += step_mult

                    curr_sigma = p_step_out.sigma

                    if recovery_games > 50:
                        break

                avg_multiplier = (total_mult / recovery_games) if recovery_games > 0 else 1.0

                results_days.append(day)
                results_peak.append(peak_multiplier)
                results_avg.append(avg_multiplier)
                results_games.append(recovery_games)

            plt.figure(figsize=(10, 6))
            plt.plot(results_days, results_peak, label="Peak Multiplier (1st Game)", color="#d62728", marker="o", markersize=4)
            plt.plot(results_days, results_avg, label="Avg Multiplier (Recovery)", color="#1f77b4", marker="s", markersize=4, linestyle="--")

            last_g = -1
            for day, val, games in zip(results_days, results_avg, results_games):
                if day > DECAY_GRACE_DAYS and (games != last_g or day % 5 == 0):
                    plt.annotate(
                        f"{games}g",
                        (day, val),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                        color="#1f77b4",
                        fontweight="bold",
                    )
                    last_g = games

            plt.title(
                f"Inactivity Decay Analysis: {names_map.get(TARGET_ID, TARGET_ID)}\n"
                f"(Base μ={p_base.mu:.1f}, σ={p_base.sigma:.2f})"
            )
            plt.xlabel("Days Inactive")
            plt.ylabel("Rating Change Multiplier (vs Base)")
            plt.axvline(x=DECAY_GRACE_DAYS, color="gray", linestyle=":", label="Grace Period Ends")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper left")
            plt.ylim(bottom=0.9)
            plt.tight_layout()
            plt.show()

print("\nDone.")
