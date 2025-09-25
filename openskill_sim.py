#!/usr/bin/env python3
"""
OpenSkill Single-Game Simulator (8 teams of 2 players)

- Uses ThurstoneMostellerFull from openskill to update ratings for one ranked match.
- Exact settings per user spec:
    beta = (25/6) * 4
    tau  = (25/300)
- NO ramp-up modifier and NO anti-boosting adjustments (plain model.rate).

Inputs
------
Provide a JSON file path as the first CLI arg with the following shape:

{
  "players": [
    {"id": "p1", "mu": 25.0, "sigma": 8.3333333333},
    {"id": "p2", "mu": 25.0, "sigma": 8.3333333333},
    ...
    (must include 16 players total)
  ],
  "teams": [
    ["p1","p2"],
    ["p3","p4"],
    ...
    (8 inner lists total, each with 2 player ids)
  ],
  "placings": [1,2,3,4,5,6,7,8],  # 1 is best, 8 is worst
  "targets": {                     # optional; "end" ratings you want to compare to
    "p1": {"mu": 27.1, "sigma": 8.01},
    "p2": {"mu": 24.2, "sigma": 8.55},
    ...
  }
}

If you don't pass a file, a tiny demo will run.

Outputs
-------
- Pretty tables showing team placings.
- Per-player before/after mu, sigma, and deltas.
- If targets are provided: error vs. target and summary stats.

Usage
-----
pip install openskill
python openskill_sim.py input.json

"""

from __future__ import annotations

import json
import math
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    from openskill.models import ThurstoneMostellerFull
except Exception as e:
    print("ERROR: This script requires the 'openskill' package.")
    print("Install with: pip install openskill")
    print(f"Import error: {e}")
    sys.exit(1)


# ------------------------- Model Setup -------------------------

def instantiate_rating_model() -> ThurstoneMostellerFull:
    # Per user spec
    return ThurstoneMostellerFull(beta=(25/6) * 4, tau=(25/300))


# ------------------------- Data Types -------------------------

@dataclass
class RatingView:
    mu: float
    sigma: float

    @classmethod
    def from_openskill(cls, r) -> "RatingView":
        return cls(mu=float(r.mu), sigma=float(r.sigma))


# ------------------------- Core Logic -------------------------

def validate_inputs(players: List[Dict], teams: List[List[str]], placings: List[int]) -> None:
    if len(players) != 16:
        raise ValueError(f"Expected 16 players, got {len(players)}")
    if len(teams) != 8:
        raise ValueError(f"Expected 8 teams, got {len(teams)}")
    for i, t in enumerate(teams, start=1):
        if len(t) != 2:
            raise ValueError(f"Team index {i} must have exactly 2 players, got {len(t)}")
    if sorted(placings) != [1,2,3,4,5,6,7,8]:
        raise ValueError("Placings must be a permutation of [1..8] (1=best, 8=worst)")
    # Ensure player ids unique and used
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


def build_player_ratings(model: ThurstoneMostellerFull, players: List[Dict]) -> Dict[str, object]:
    """Return id -> openskill Rating (object)"""
    out = {}
    for p in players:
        mu = float(p.get("mu", 25.0))
        sigma = float(p.get("sigma", 25.0/3.0))
        out[p["id"]] = model.rating(mu=mu, sigma=sigma)
    return out


def rate_one_game(model: ThurstoneMostellerFull,
                  teams: List[List[str]],
                  placings: List[int],
                  player_ratings: Dict[str, object]) -> Dict[str, object]:
    """Run model.rate for one ranked match and return updated ratings by player id."""
    # Build teams aligned to rank order (rank 0 is best). placings are 1..8, so we sort by placing asc.
    placing_with_team = list(zip(placings, teams))
    placing_with_team.sort(key=lambda x: x[0])  # lowest placing first

    teams_ratings = []
    team_order_ids = []  # parallel list for mapping back
    for placing, team in placing_with_team:
        r0 = player_ratings[team[0]]
        r1 = player_ratings[team[1]]
        teams_ratings.append([r0, r1])
        team_order_ids.append(team)

    ranks = list(range(8))  # [0..7]

    new_teams = model.rate(teams_ratings, ranks=ranks)

    # Map back to players in the same sorted order
    updated = dict(player_ratings)  # copy
    for idx, new_pair in enumerate(new_teams):
        pid0, pid1 = team_order_ids[idx]
        updated[pid0] = new_pair[0]
        updated[pid1] = new_pair[1]

    return updated


# ------------------------- Reporting -------------------------

def fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def print_team_table(teams: List[List[str]], placings: List[int]) -> None:
    print_header("Teams & Placings (1 = best)")
    rows = [("Place", "Player A", "Player B")]
    placing_with_team = list(zip(placings, teams))
    placing_with_team.sort(key=lambda x: x[0])
    for place, team in placing_with_team:
        rows.append((str(place), team[0], team[1]))
    _print_table(rows)


def print_player_changes(before: Dict[str, RatingView],
                         after: Dict[str, RatingView],
                         names: Dict[str, str] = None) -> None:
    print_header("Per-Player Rating Changes")
    rows = [("Player", "Name", "mu_before", "mu_after", "Δmu", "sigma_before", "sigma_after", "Δsigma")]
    for pid in sorted(before.keys()):
        b = before[pid]
        a = after[pid]
        nm = names.get(pid) if names else ""
        rows.append((
            pid,
            nm,
            fmt(b.mu), fmt(a.mu), fmt(a.mu - b.mu, 5),
            fmt(b.sigma), fmt(a.sigma), fmt(a.sigma - b.sigma, 5)
        ))
    _print_table(rows)


def print_target_comparison(after: Dict[str, RatingView],
                            targets: Dict[str, Dict[str, float]]) -> None:
    print_header("Comparison vs. Provided Target End Ratings")
    rows = [("Player", "mu_calc", "mu_target", "mu_err", "sigma_calc", "sigma_target", "sigma_err")]
    mu_sq_err = []
    s_sq_err  = []
    for pid in sorted(after.keys()):
        a = after[pid]
        t = targets.get(pid)
        if not t:
            continue
        mu_err = a.mu - float(t["mu"])
        s_err  = a.sigma - float(t["sigma"])
        mu_sq_err.append(mu_err ** 2)
        s_sq_err.append(s_err ** 2)
        rows.append((
            pid,
            fmt(a.mu), fmt(float(t["mu"])), fmt(mu_err, 5),
            fmt(a.sigma), fmt(float(t["sigma"])), fmt(s_err, 5)
        ))
    _print_table(rows)
    if mu_sq_err or s_sq_err:
        mu_rmse = math.sqrt(sum(mu_sq_err) / max(1, len(mu_sq_err)))
        s_rmse  = math.sqrt(sum(s_sq_err)  / max(1, len(s_sq_err)))
        print(f"\nRMSE: mu={fmt(mu_rmse,5)}  sigma={fmt(s_rmse,5)}")


def _print_table(rows: List[Tuple[str, ...]]) -> None:
    # Minimal, dependency-free fixed-width table
    widths = [0] * len(rows[0])
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(r):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r))

    sep = "-+-".join("-"*w for w in widths)
    print(fmt_row(rows[0]))
    print(sep)
    for r in rows[1:]:
        print(fmt_row(r))



def print_target_games(target_games: List[Dict], players: List[Dict], before: Dict[str, RatingView]) -> None:
    if not target_games:
        return
    print_header("Target Games Summary (metadata provided by user)")
    id_to_name = {p.get("id"): p.get("name","") for p in players}
    rows = [("Player", "Name", "Game ID", "Placing", "RatingΔ", "Target pre μ", "Our pre μ", "Δμ (our - target)", "Target pre σ", "Our pre σ", "Δσ (our - target)")]
    for tg in target_games:
        pid = tg.get("player_id","")
        nm  = id_to_name.get(pid, "")
        gid = tg.get("game_id","")
        placing = tg.get("placing","")
        rchg = tg.get("rating_change","")
        t_mu = tg.get("pregame_mu", None)
        t_s  = tg.get("pregame_sigma", None)
        b = before.get(pid)
        if b is None:
            continue
        our_mu = b.mu
        our_s  = b.sigma
        dmu = (our_mu - t_mu) if t_mu is not None else None
        ds  = (our_s - t_s)  if t_s  is not None else None
        rows.append((
            pid, nm, gid, str(placing), str(rchg),
            fmt(t_mu) if t_mu is not None else "",
            fmt(our_mu),
            fmt(dmu,5) if dmu is not None else "",
            fmt(t_s) if t_s is not None else "",
            fmt(our_s),
            fmt(ds,5) if ds is not None else ""
        ))
    _print_table(rows)

# ------------------------- I/O Helpers -------------------------

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_view_map(d: Dict[str, object]) -> Dict[str, RatingView]:
    return {k: RatingView.from_openskill(v) for k, v in d.items()}


# ------------------------- Demo -------------------------

def demo_payload() -> dict:
    # 16 players, default mu/sigma
    players = [{"id": f"p{i+1}"} for i in range(16)]
    teams = [[f"p{1+i*2}", f"p{2+i*2}"] for i in range(8)]
    placings = [1,2,3,4,5,6,7,8]
    return {"players": players, "teams": teams, "placings": placings}


# ------------------------- Main -------------------------

def main():
    default_path = "sim_inputs.json"
    if len(sys.argv) >= 2:
        data = load_json(sys.argv[1])
    elif os.path.exists(default_path):
        print(f"Using default input: {default_path}\n")
        data = load_json(default_path)
    else:
        print("No input file provided; running tiny demo instead.\n")
        data = demo_payload()

    # Quick input summary
    num_players = len(data.get("players", []))
    mu_vals = [float(p.get("mu", 25.0)) for p in data.get("players", [])]
    s_vals  = [float(p.get("sigma", 25.0/3.0)) for p in data.get("players", [])]
    has_names = any("name" in p for p in data.get("players", []))
    if mu_vals and s_vals:
        print(f"Loaded {num_players} players. "
              f"mu: min={min(mu_vals):.4f} max={max(mu_vals):.4f} | "
              f"sigma: min={min(s_vals):.4f} max={max(s_vals):.4f} | "
              f"names_present={has_names}\\n")

    players  = data["players"]
    teams    = data["teams"]
    placings = data["placings"]
    targets  = data.get("targets") or {}

    validate_inputs(players, teams, placings)

    model = instantiate_rating_model()

    before = build_player_ratings(model, players)
    before_view = to_view_map(before)

    after = rate_one_game(model, teams, placings, before)
    after_view = to_view_map(after)

    # names map
    names_map = {p.get("id"): p.get("name","") for p in players}

    print_team_table(teams, placings)
    print_player_changes(before_view, after_view, names_map)

    # Target metadata comparison (pregame consistency and game info)
    print_target_games(data.get("target_games", []), players, before_view)

    if targets:
        print_target_comparison(after_view, targets)


if __name__ == "__main__":
    main()
