#!/usr/bin/env python3
import json
import math
import sys
import os
from datetime import datetime, timezone

# Make repository root importable when running this file directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import production helpers/constants so the sim mirrors real logic
from ranking_algorithm import (
    apply_teammate_gap_penalty,
    _teammate_penalty_scale,
    UNBALANCED_LOBBY_THRESHOLD,
    UNBALANCED_TEAM_MU_REDUCTION,
    check_for_unbalanced_lobby,
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


# ----------------------------------------------------------------------
# INPUTS & VALIDATION
# ----------------------------------------------------------------------
default_path = os.path.join(_SCRIPT_DIR, "sim_inputs", "sim_inputs_18.json")
if len(sys.argv) >= 2:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
elif os.path.exists(default_path):
    print(f"Using default input: {default_path}\n")
    with open(default_path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    print("No input file provided; running tiny demo instead.\n")
    players = [{"id": f"p{i+1}"} for i in range(16)]
    teams = [[f"p{1+i*2}", f"p{2+i*2}"] for i in range(8)]
    placings = [1, 2, 3, 4, 5, 6, 7, 8]
    data = {"players": players, "teams": teams, "placings": placings}

players = data["players"]
teams = data["teams"]
placings = data["placings"]
targets = data.get("targets") or {}
target_games = data.get("target_games") or []
gm_players = data.get("gm_players") or []
gm_set = set(gm_players)
gm_rating_threshold = data.get("gm_rating_threshold")
gm_mask_provided = False
modifiers_by_pid = {}
if target_games:
    for tg in target_games:
        pid = tg.get("player_id")
        if pid:
            modifiers_by_pid[pid] = tg
modifier_keys = {"sigma_cap_scale", "team_gap_pct", "team_gap_scale", "unbalanced_reduction_pct"}
use_recorded_modifiers = bool(modifiers_by_pid) and all(modifier_keys.issubset(set(tg.keys())) for tg in target_games)

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
model = ThurstoneMostellerFull(beta=(25 / 6) * 4, tau=(25 / 300))

before_ratings = {}
for p in players:
    mu = float(p.get("mu", 25.0))
    sigma = float(p.get("sigma", 25.0 / 3.0))
    before_ratings[p["id"]] = model.rating(mu=mu, sigma=sigma)

# Derive GM mask from rating threshold if provided
if gm_rating_threshold is not None:
    try:
        threshold = float(gm_rating_threshold)
        for pid, r in before_ratings.items():
            simple_rating = _simple_rating(r.mu, r.sigma)
            if simple_rating >= threshold:
                gm_set.add(pid)
    except Exception as e:
        print(f"Warning: could not parse gm_rating_threshold '{gm_rating_threshold}': {e}")
gm_mask_provided = bool(gm_set)

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

after_ratings = {}
if use_recorded_modifiers:
    # Replay recorded sigma cap / unbalanced / gap modifiers when present in the input.
    rate_input_final = []
    for team in team_order_ids:
        r0 = before_ratings[team[0]]
        r1 = before_ratings[team[1]]
        m0 = modifiers_by_pid.get(team[0], {})
        m1 = modifiers_by_pid.get(team[1], {})
        s0 = m0.get("sigma_cap_scale", 1.0)
        s1 = m1.get("sigma_cap_scale", 1.0)
        s0 = 1.0 if s0 is None else float(s0)
        s1 = 1.0 if s1 is None else float(s1)
        base0 = model.rating(mu=r0.mu, sigma=r0.sigma * s0)
        base1 = model.rating(mu=r1.mu, sigma=r1.sigma * s1)
        red = m0.get("unbalanced_reduction_pct")
        if red is None:
            red = m1.get("unbalanced_reduction_pct")
        red = 0.0 if red is None else float(red)
        if red != 0.0:
            rate_input_final.append([
                model.rating(mu=base0.mu * (1.0 - red), sigma=base0.sigma),
                model.rating(mu=base1.mu * (1.0 - red), sigma=base1.sigma),
            ])
        else:
            rate_input_final.append([base0, base1])

    rated_teams = model.rate(rate_input_final, ranks=ranks)

    new_teams_local = []
    for idx in range(len(rate_input_final)):
        orig_team = [before_ratings[team_order_ids[idx][0]], before_ratings[team_order_ids[idx][1]]]
        old_final = rate_input_final[idx]
        new_from_rate = rated_teams[idx]

        final_team = []
        for p_idx in range(len(orig_team)):
            orig = orig_team[p_idx]
            delta_mu = new_from_rate[p_idx].mu - old_final[p_idx].mu
            delta_sigma = new_from_rate[p_idx].sigma - old_final[p_idx].sigma
            final_team.append(model.rating(mu=orig.mu + delta_mu, sigma=orig.sigma + delta_sigma))
        new_teams_local.append(final_team)

    after_ratings = dict(before_ratings)
    for idx, pair in enumerate(new_teams_local):
        pid0, pid1 = team_order_ids[idx]
        m0 = modifiers_by_pid.get(pid0, {})
        m1 = modifiers_by_pid.get(pid1, {})
        g0 = m0.get("team_gap_scale", 1.0)
        g1 = m1.get("team_gap_scale", 1.0)
        g0 = 1.0 if g0 is None else float(g0)
        g1 = 1.0 if g1 is None else float(g1)

        if g0 != 1.0:
            after_ratings[pid0] = model.rating(
                mu=before_ratings[pid0].mu + (pair[0].mu - before_ratings[pid0].mu) * g0,
                sigma=before_ratings[pid0].sigma + (pair[0].sigma - before_ratings[pid0].sigma) * g0,
            )
        else:
            after_ratings[pid0] = pair[0]

        if g1 != 1.0:
            after_ratings[pid1] = model.rating(
                mu=before_ratings[pid1].mu + (pair[1].mu - before_ratings[pid1].mu) * g1,
                sigma=before_ratings[pid1].sigma + (pair[1].sigma - before_ratings[pid1].sigma) * g1,
            )
        else:
            after_ratings[pid1] = pair[1]
else:
    new_teams = model.rate(teams_ratings, ranks=ranks)
    after_ratings = dict(before_ratings)
    for idx, new_pair in enumerate(new_teams):
        pid0, pid1 = team_order_ids[idx]
        after_ratings[pid0] = new_pair[0]
        after_ratings[pid1] = new_pair[1]

names_map = {p.get("id"): p.get("name", "") for p in players}

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
    a = after_ratings[pid]
    print(
        f"{pid:6} | {b.mu:9.4f} | {a.mu:8.4f} | {a.mu - b.mu:8.5f} | "
        f"{b.sigma:12.4f} | {a.sigma:11.4f} | {a.sigma - b.sigma:7.5f}"
    )

# Target comparison if provided
if targets:
    print("\nComparison vs. Provided Target End Ratings")
    print("=" * 50)
    print("Player | mu_calc | mu_target | mu_err   | sigma_calc | sigma_target | sigma_err")
    print("-------|---------|-----------|----------|------------|--------------|----------")
    mu_sq_err = []
    s_sq_err = []
    for pid in sorted(after_ratings.keys()):
        a = after_ratings[pid]
        t = targets.get(pid)
        if not t:
            continue
        mu_err = a.mu - float(t["mu"])
        s_err = a.sigma - float(t["sigma"])
        mu_sq_err.append(mu_err**2)
        s_sq_err.append(s_err**2)
        print(
            f"{pid:6} | {a.mu:7.4f} | {float(t['mu']):9.4f} | {mu_err:8.5f} | "
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
    a0 = after_ratings[team[0]]
    a1 = after_ratings[team[1]]

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
        a = after_ratings[pid]

        pre_player_rating = _simple_rating(b.mu, b.sigma)
        post_player_rating = _simple_rating(a.mu, a.sigma)
        delta_rating = post_player_rating - pre_player_rating

        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating:.2f})"
        post_player_str = f"{a.mu:.2f} {a.sigma:.2f} ({post_player_rating:.2f})"

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
                f"{delta_rating:+.2f}",
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
    pid: _simple_rating(after_ratings[pid].mu, after_ratings[pid].sigma)
    - _simple_rating(before_ratings[pid].mu, before_ratings[pid].sigma)
    for pid in before_ratings
}


# ----------------------------------------------------------------------
# PIPELINE RUNNER (sigma cap -> unbalanced lobby -> gap penalty)
# ----------------------------------------------------------------------
def run_pipeline(apply_sigma_cap=False, apply_unbalanced=False, apply_gap_penalty=False, override_ratings=None):
    rate_input = []
    current_gm_any = []
    current_gm_both = []

    # Build input teams, optionally overriding starting ratings
    for idx, (pid0, pid1) in enumerate(team_order_ids):
        r0 = override_ratings.get(pid0, before_ratings[pid0]) if override_ratings else before_ratings[pid0]
        r1 = override_ratings.get(pid1, before_ratings[pid1]) if override_ratings else before_ratings[pid1]
        current_gm_any.append(gm_team_any[idx])
        current_gm_both.append(gm_team_both[idx])

        if apply_sigma_cap and current_gm_any[idx]:
            if r0.mu >= r1.mu:
                s_high = r0.sigma
                s_low = r1.sigma
            else:
                s_high = r1.sigma
                s_low = r0.sigma
            if s_low <= s_high:
                k = 1.0
            else:
                current_team_sigma = math.hypot(r0.sigma, r1.sigma)
                target_team_sigma = math.hypot(s_high, s_high)
                k = target_team_sigma / current_team_sigma if current_team_sigma > 0 else 1.0
            rate_input.append(
                [
                    model.rating(mu=r0.mu, sigma=r0.sigma * k),
                    model.rating(mu=r1.mu, sigma=r1.sigma * k),
                ]
            )
        else:
            rate_input.append([_clone_rating(model, r0), _clone_rating(model, r1)])

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


# ----------------------------------------------------------------------
# SIGMA-CAP summary table (ratio-preserving, exact team-sigma target)
# ----------------------------------------------------------------------
def _team_sigma(r0, r1):
    return math.hypot(r0.sigma, r1.sigma)


def _target_team_sigma_for_cap(r0, r1):
    if r0.mu > r1.mu:
        s_high_mu = r0.sigma
        s_low_mu = r1.sigma
    elif r1.mu > r0.mu:
        s_high_mu = r1.sigma
        s_low_mu = r0.sigma
    else:
        return _team_sigma(r0, r1)
    s_low_mu_capped = s_low_mu if s_low_mu <= s_high_mu else s_high_mu
    return math.hypot(s_high_mu, s_low_mu_capped)


base_mu_delta_by_pid = {pid: after_ratings[pid].mu - before_ratings[pid].mu for pid in before_ratings}
base_team_mu_delta_by_place = {place: sum(base_mu_delta_by_pid[pid] for pid in team) for place, team in placing_with_team}

scaled_input_by_pid = {}
scaled_pairs = []
scaled_pre_team = {}

for place, team in placing_with_team:
    p0, p1 = team
    r0 = before_ratings[p0]
    r1 = before_ratings[p1]

    current_team_sigma = _team_sigma(r0, r1)
    T = _target_team_sigma_for_cap(r0, r1)

    if current_team_sigma <= 0.0:
        raise ValueError(f"Non-positive team sigma at placing {place}.")

    k = T / current_team_sigma

    c0 = model.rating(mu=r0.mu, sigma=r0.sigma * k)
    c1 = model.rating(mu=r1.mu, sigma=r1.sigma * k)

    scaled_pairs.append([c0, c1])
    scaled_input_by_pid[p0] = c0
    scaled_input_by_pid[p1] = c1

    mu_sum = r0.mu + r1.mu
    sig_rms = math.hypot(c0.sigma, c1.sigma)
    team_rt = _simple_rating(mu_sum, sig_rms)
    scaled_pre_team[place] = (mu_sum, sig_rms, team_rt)

scaled_new_pairs = model.rate(scaled_pairs, ranks=ranks)

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
    "rating_change_difference",
]
rows_sigma = []

for place, team in placing_with_team:
    for idx, pid in enumerate(team):
        name = names_map.get(pid) or pid

        b = before_ratings[pid]
        pre_player_rating = _simple_rating(b.mu, b.sigma)
        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating:.2f})"

        mu_f = final_mu[pid]
        sg_f = final_sigma[pid]
        post_player_rating = _simple_rating(mu_f, sg_f)
        post_player_str = f"{mu_f:.2f} {sg_f:.2f} ({post_player_rating:.2f})"

        delta_rating = post_player_rating - pre_player_rating
        delta_diff = delta_rating - baseline_rating_change[pid]

        if idx == 0:
            mu_t, sg_t, rt_t = scaled_pre_team[place]
            pre_team_str = f"{mu_t:.2f} {sg_t:.2f} ({rt_t:.2f})"
            mu_u, sg_u, rt_u = scaled_post_team[place]
            post_team_str = f"{mu_u:.2f} {sg_u:.2f} ({rt_u:.2f})"
        else:
            pre_team_str = ""
            post_team_str = ""

        rows_sigma.append(
            [
                f"{place}",
                f"{name}",
                pre_player_str,
                pre_team_str,
                post_team_str,
                post_player_str,
                f"{delta_rating:+.2f}",
                f"{delta_diff:+.2f}",
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
    print("=" * 150)
    print(
        "placing | player                                    | pregame_player_stats        | pregame_team_stats           | "
        "postgame_team_stats          | postgame_player_stats       | rating_change | rating_change_difference"
    )
    print("-" * 150)
    for r in rows_sigma:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:26} | {r[3]:26} | {r[4]:26} | {r[5]:26} | {r[6]:>13} | {r[7]:>24}"
        )

# ----------------------------------------------------------------------
# GAP-PENALTY summary table (ordered by placing)
# ----------------------------------------------------------------------
gap_new_teams = [pair.copy() for pair in new_teams]
apply_teammate_gap_penalty(model, teams_ratings, gap_new_teams, logger=None)

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
    "postgame_player_stats_gap",
    "rating_change",
    "rating_change_difference",
    "gap_pct",
    "scale_used",
]
rows_gap = []

for place, team in placing_with_team:
    r0 = before_ratings[team[0]]
    r1 = before_ratings[team[1]]
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

    for idx, pid in enumerate(team):
        name = names_map.get(pid) or pid

        b = before_ratings[pid]
        pre_player_rating = _simple_rating(b.mu, b.sigma)
        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating:.2f})"

        a_base = after_ratings[pid]
        post_player_base_rating = _simple_rating(a_base.mu, a_base.sigma)
        post_player_base_str = f"{a_base.mu:.2f} {a_base.sigma:.2f} ({post_player_base_rating:.2f})"

        a_gap = gap_after[pid]
        post_player_gap_rating = _simple_rating(a_gap.mu, a_gap.sigma)
        post_player_gap_str = f"{a_gap.mu:.2f} {a_gap.sigma:.2f} ({post_player_gap_rating:.2f})"

        delta_rating = post_player_gap_rating - pre_player_rating
        delta_diff = delta_rating - baseline_rating_change[pid]

        if idx == 0:
            mu_pre_t, sg_pre_t, rt_pre_t = pre_team[place]
            pre_team_str = f"{mu_pre_t:.2f} {sg_pre_t:.2f} ({rt_pre_t:.2f})"
            mu_post_t, sg_post_t, rt_post_t = post_team[place]
            post_team_base_str = f"{mu_post_t:.2f} {sg_post_t:.2f} ({rt_post_t:.2f})"
            gap_str = f"{gap_pct_val*100:.1f}%"
            scale_str = f"{scale_val*100:.1f}%"
        else:
            pre_team_str = ""
            post_team_base_str = ""
            gap_str = ""
            scale_str = ""

        rows_gap.append(
            [
                f"{place}",
                f"{name}",
                pre_player_str,
                pre_team_str,
                post_team_base_str,
                post_player_base_str,
                post_player_gap_str,
                f"{delta_rating:+.2f}",
                f"{delta_diff:+.2f}",
                gap_str,
                scale_str,
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
        "postgame_team_stats | postgame_player_stats | postgame_player_stats_gap | "
        "rating_change | rating_change_difference | gap_pct | scale_used"
    )
    print("-" * 220)
    for r in rows_gap:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:26} | {r[3]:26} | {r[4]:26} | "
            f"{r[5]:26} | {r[6]:26} | {r[7]:>13} | {r[8]:>24} | {r[9]:>7} | {r[10]:>10}"
        )

# ----------------------------------------------------------------------
# UNBALANCED-LOBBY summary table (ordered by placing)
# ----------------------------------------------------------------------
ub_rate_input, _ = check_for_unbalanced_lobby(
    model,
    teams_ratings,
    logger=None,
    gm_team_both_mask=gm_team_both if gm_mask_provided else None,
)

if ub_rate_input is None:
    ub_new_teams = new_teams
else:
    ub_rated_from_adjusted = model.rate(ub_rate_input, ranks=ranks)
    ub_new_teams = []

    for team_idx in range(len(teams_ratings)):
        orig_team = teams_ratings[team_idx]
        adj_old_team = ub_rate_input[team_idx]
        adj_new_team = ub_rated_from_adjusted[team_idx]

        final_team = []
        for p_idx in range(len(orig_team)):
            orig = orig_team[p_idx]
            old_adj = adj_old_team[p_idx]
            new_adj = adj_new_team[p_idx]

            delta_mu = new_adj.mu - old_adj.mu
            delta_sigma = new_adj.sigma - old_adj.sigma

            final_mu = orig.mu + delta_mu
            final_sigma = orig.sigma + delta_sigma

            final_team.append(model.rating(mu=final_mu, sigma=final_sigma))

        ub_new_teams.append(final_team)

ub_after = dict(before_ratings)
for idx, pair in enumerate(ub_new_teams):
    pid0, pid1 = team_order_ids[idx]
    ub_after[pid0] = pair[0]
    ub_after[pid1] = pair[1]

ub_post_team = {}
for place, team in placing_with_team:
    a0 = ub_after[team[0]]
    a1 = ub_after[team[1]]
    mu_sum = a0.mu + a1.mu
    sig_rms = math.hypot(a0.sigma, a1.sigma)
    team_rt = _simple_rating(mu_sum, sig_rms)
    ub_post_team[place] = (mu_sum, sig_rms, team_rt)

team_avg_mu_by_place = {}
for place, team in placing_with_team:
    b0 = before_ratings[team[0]]
    b1 = before_ratings[team[1]]
    team_avg_mu_by_place[place] = (b0.mu + b1.mu) / 2.0

lobby_avg_team_mu = sum(team_avg_mu_by_place.values()) / len(team_avg_mu_by_place)

headers_ub = [
    "placing",
    "player",
    "pregame_player_stats",
    "pregame_team_stats",
    "postgame_team_stats_base",
    "postgame_player_stats_base",
    "postgame_player_stats_unbalanced",
    "rating_change",
    "rating_change_difference",
    "lobby_diff_pct",
    "mu_reduction",
]
rows_ub = []

for place, team in placing_with_team:
    avg_mu = team_avg_mu_by_place[place]
    if lobby_avg_team_mu > 0.0:
        diff_pct = (avg_mu - lobby_avg_team_mu) / lobby_avg_team_mu
    else:
        diff_pct = 0.0

    is_unbalanced_team = lobby_avg_team_mu > 0.0 and (diff_pct >= UNBALANCED_LOBBY_THRESHOLD)

    if is_unbalanced_team:
        lobby_diff_str = f"{diff_pct * 100:.1f}%"
        mu_red_str = f"{UNBALANCED_TEAM_MU_REDUCTION * 100:.1f}%"
    else:
        lobby_diff_str = ""
        mu_red_str = ""

    for idx, pid in enumerate(team):
        name = names_map.get(pid) or pid

        b = before_ratings[pid]
        pre_player_rating = _simple_rating(b.mu, b.sigma)
        pre_player_str = f"{b.mu:.2f} {b.sigma:.2f} ({pre_player_rating:.2f})"

        a_base = after_ratings[pid]
        post_player_base_rating = _simple_rating(a_base.mu, a_base.sigma)
        post_player_base_str = f"{a_base.mu:.2f} {a_base.sigma:.2f} ({post_player_base_rating:.2f})"

        a_ub = ub_after[pid]
        post_player_ub_rating = _simple_rating(a_ub.mu, a_ub.sigma)
        post_player_ub_str = f"{a_ub.mu:.2f} {a_ub.sigma:.2f} ({post_player_ub_rating:.2f})"

        delta_rating = post_player_ub_rating - pre_player_rating
        delta_diff = delta_rating - baseline_rating_change[pid]

        if idx == 0:
            mu_pre_t, sg_pre_t, rt_pre_t = pre_team[place]
            pre_team_str = f"{mu_pre_t:.2f} {sg_pre_t:.2f} ({rt_pre_t:.2f})"

            mu_post_base_t, sg_post_base_t, rt_post_base_t = post_team[place]
            post_team_base_str = f"{mu_post_base_t:.2f} {sg_post_base_t:.2f} ({rt_post_base_t:.2f})"
        else:
            pre_team_str = ""
            post_team_base_str = ""

        rows_ub.append(
            [
                f"{place}",
                f"{name}",
                pre_player_str,
                pre_team_str,
                post_team_base_str,
                post_player_base_str,
                post_player_ub_str,
                f"{delta_rating:+.2f}",
                f"{delta_diff:+.2f}",
                lobby_diff_str,
                mu_red_str,
            ]
        )

if _USE_RICH:
    table_ub = Table(title="UNBALANCED-LOBBY summary table (ordered by placing)", show_lines=False)
    for h in headers_ub:
        table_ub.add_column(h)
    for r in rows_ub:
        table_ub.add_row(*r)
    _console.print(table_ub)
else:
    print("\nUNBALANCED-LOBBY summary table (ordered by placing)")
    print("=" * 220)
    print(
        "placing | player | pregame_player_stats | pregame_team_stats | "
        "postgame_team_stats_base | postgame_player_stats_base | "
        "postgame_player_stats_unbalanced | rating_change | rating_change_difference | lobby_diff_pct | mu_reduction"
    )
    print("-" * 220)
    for r in rows_ub:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:26} | {r[3]:26} | {r[4]:26} | "
            f"{r[5]:26} | {r[6]:26} | {r[7]:>13} | {r[8]:>24} | {r[9]:>13} | {r[10]:>11}"
        )

# ----------------------------------------------------------------------
# COMBINED STACKED-PENALTY table (sigma-cap -> unbalanced lobby -> gap penalty)
# ----------------------------------------------------------------------
sigma_stage_after = run_pipeline(apply_sigma_cap=True, apply_unbalanced=False, apply_gap_penalty=False)
unbalanced_stage_after = run_pipeline(apply_sigma_cap=True, apply_unbalanced=True, apply_gap_penalty=False)
gap_stage_after = run_pipeline(apply_sigma_cap=True, apply_unbalanced=True, apply_gap_penalty=True)

sigma_stage_change = {
    pid: _simple_rating(sigma_stage_after[pid].mu, sigma_stage_after[pid].sigma)
    - _simple_rating(before_ratings[pid].mu, before_ratings[pid].sigma)
    for pid in before_ratings
}
unbalanced_stage_change = {
    pid: _simple_rating(unbalanced_stage_after[pid].mu, unbalanced_stage_after[pid].sigma)
    - _simple_rating(before_ratings[pid].mu, before_ratings[pid].sigma)
    for pid in before_ratings
}
gap_stage_change = {
    pid: _simple_rating(gap_stage_after[pid].mu, gap_stage_after[pid].sigma)
    - _simple_rating(before_ratings[pid].mu, before_ratings[pid].sigma)
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
    "final_rating_change",
]
rows_combo = []

for place, team in placing_with_team:
    for pid in team:
        name = names_map.get(pid) or pid

        base_change = baseline_rating_change[pid]
        sigma_change = sigma_stage_change[pid]
        unbalanced_change = unbalanced_stage_change[pid]
        gap_change = gap_stage_change[pid]

        sigma_penalty = sigma_change - base_change
        unbalanced_penalty = unbalanced_change - sigma_change
        gap_penalty = gap_change - unbalanced_change

        rows_combo.append(
            [
                f"{place}",
                f"{name}",
                f"{base_change:+.2f}",
                f"{sigma_penalty:+.2f}",
                f"{sigma_change:+.2f}",
                f"{unbalanced_penalty:+.2f}",
                f"{unbalanced_change:+.2f}",
                f"{gap_penalty:+.2f}",
                f"{gap_change:+.2f}",
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
    print("=" * 180)
    print(
        "placing | player | rating_change_base | sigma_cap_effect | rating_change_after_sigma | "
        "unbalanced_grace_effect | rating_change_after_unbalanced | team_gap_effect | final_rating_change"
    )
    print("-" * 180)
    for r in rows_combo:
        print(
            f"{r[0]:7} | {r[1]:42} | {r[2]:>18} | {r[3]:>16} | {r[4]:>25} | "
            f"{r[5]:>23} | {r[6]:>29} | {r[7]:>16} | {r[8]:>19}"
        )

# ----------------------------------------------------------------------
# GAP-PENALTY CURVE chart
# ----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    xs = list(range(101))
    ys = [_teammate_penalty_scale(x / 100.0) * 100.0 for x in xs]

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

print(f"\nRunning Experiment 3: Inactivity Decay Analysis for {TARGET_ID}...")

if TARGET_ID not in before_ratings:
    print(f"Skipping Experiment 3: Player {TARGET_ID} not found in input.")
else:
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
