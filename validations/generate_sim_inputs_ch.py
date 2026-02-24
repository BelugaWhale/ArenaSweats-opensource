#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

from clickhouse_connect import get_client
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Build sim input JSON from ClickHouse matchhistory for a single game.")
parser.add_argument("--game-id", help="Target game_id to export (prompts if omitted).")
parser.add_argument("--output", "-o", help="Output path for JSON (defaults to validations/sim_inputs/sim_inputs_<game_id>.json).")
parser.add_argument("--region", help="Region prefix for ClickHouse tables (e.g. euw). Prompts if omitted.")
parser.add_argument("--ch-prefix", default="DBCH", help="Env prefix for ClickHouse config (default: DBCH).")
args = parser.parse_args()

game_id = args.game_id or input("Enter game_id: ").strip()
if not game_id:
    raise SystemExit("game_id is required")
region = (args.region or "").strip().lower() or input("Enter region: ").strip().lower()
if not region:
    raise SystemExit("region is required")
ch_prefix = args.ch_prefix.upper()
mh_table = f"{region}_player_matchhistory"
results_table = f"{region}_player_results"

host = os.getenv(f"{ch_prefix}_HOST")
scheme = (os.getenv(f"{ch_prefix}_SCHEME") or "").lower()
sslmode = (os.getenv(f"{ch_prefix}_SSLMODE") or "").lower()
secure = scheme == "https" or sslmode in {"require", "verify-ca", "verify-full"}
port = int(os.getenv(f"{ch_prefix}_PORT", "8443" if secure else "8123"))

if not host:
    raise SystemExit(f"Missing {ch_prefix}_HOST")

client = get_client(
    host=host,
    port=port,
    username=os.getenv(f"{ch_prefix}_USER", "default"),
    password=os.getenv(f"{ch_prefix}_PASSWORD", ""),
    database=os.getenv(f"{ch_prefix}_NAME") or None,
    secure=secure,
    verify=(os.getenv(f"{ch_prefix}_SSL_VERIFY") or str(secure)).lower() in {"1", "true", "yes", "y", "on"},
    ca_cert=os.getenv(f"{ch_prefix}_SSLROOTCERT") or None,
    client_cert=os.getenv(f"{ch_prefix}_SSLCERT") or None,
    client_cert_key=os.getenv(f"{ch_prefix}_SSLKEY") or None,
    connect_timeout=float(os.getenv(f"{ch_prefix}_CONNECT_TIMEOUT", "8")),
    send_receive_timeout=float(os.getenv(f"{ch_prefix}_RW_TIMEOUT", "30")),
)

query = f"""
WITH current_game AS (
    SELECT player_hash AS player_hash, game_ts AS game_ts
    FROM {mh_table}
    WHERE game_id = {{game_id:String}}
),
next_game AS (
    SELECT
        pr.player_hash AS player_hash,
        argMin(pr.game_id, pr.game_ts) AS next_game_id
    FROM {results_table} AS pr
    INNER JOIN current_game AS cg
        ON pr.player_hash = cg.player_hash
    WHERE pr.game_ts > cg.game_ts
    GROUP BY pr.player_hash
)
SELECT
    curr.player_hash AS player_hash,
    curr.player_name AS player_name,
    curr.game_id AS game_id,
    curr.game_ts AS game_ts,
    curr.placing AS placing,
    
    -- Current Game Stats (from matchhistory)
    curr.rating_change AS rating_change,
    curr.avg_worse_opp_rating AS avg_worse_opp_rating,
    curr.avg_better_opp_rating AS avg_better_opp_rating,
    curr.pregame_mu AS pregame_mu,
    curr.pregame_sigma AS pregame_sigma,
    
    -- Config Params
    curr.sigma_cap_scale AS sigma_cap_scale,
    curr.team_gap_pct AS team_gap_pct,
    curr.team_gap_scale AS team_gap_scale,
    curr.unbalanced_reduction_pct AS unbalanced_reduction_pct,
    
    -- Next Game Stats (Target)
    mh_next.pregame_mu AS next_pregame_mu,
    mh_next.pregame_sigma AS next_pregame_sigma

FROM {mh_table} AS curr

-- Join Next Game Stats
LEFT JOIN next_game AS ng
    ON curr.player_hash = ng.player_hash
LEFT JOIN {mh_table} AS mh_next
    ON ng.player_hash = mh_next.player_hash AND ng.next_game_id = mh_next.game_id

WHERE curr.game_id = {{game_id:String}}
"""

res = client.query(query, parameters={"game_id": game_id})
rows = res.result_rows

if len(rows) != 16:
    print(f"Warning: Expected 16 rows for game {game_id}, got {len(rows)}. Proceeding anyway.")

# Sort by Placing, then Hash to ensure consistent p1..p16 assignment
rows_sorted = sorted(rows, key=lambda r: (int(r[4]), int(r[0])))

players = []
teams_map = {}
targets = {}
target_games = []

for idx, row in enumerate(rows_sorted, start=1):
    pid = f"p{idx}"
    
    # Unpack row based on SQL SELECT order
    player_hash = row[0]
    name = row[1] or f"Unknown#{player_hash}" # Fallback if join fails
    gid = row[2]
    gts = row[3]
    placing = int(row[4])
    
    rating_change = row[5]
    avg_worse = row[6]
    avg_better = row[7]
    pre_mu = row[8]
    pre_sigma = row[9]
    
    sigma_cap_scale = row[10]
    team_gap_pct = row[11]
    team_gap_scale = row[12]
    unbalanced_reduction_pct = row[13]
    
    next_pre_mu = row[14]
    next_pre_sigma = row[15]

    if pre_mu is None or pre_sigma is None:
        print(f"Warning: Missing pregame ratings for {name} ({pid}). using default 25/8.33")
        pre_mu = 25.0
        pre_sigma = 8.333

    # Build Player object
    players.append({
        "id": pid,
        "name": name,
        "mu": float(pre_mu),
        "sigma": float(pre_sigma),
        "pre_source": "mh",
        "team_placing": placing,
    })

    # Group teams
    teams_map.setdefault(placing, []).append(pid)

    # Set validation target from next game stats if available
    if next_pre_mu is not None and next_pre_sigma is not None:
        targets[pid] = {
            "mu": float(next_pre_mu), 
            "sigma": float(next_pre_sigma)
        }

    # Build target_games entry with extended config fields
    target_games.append({
        "player_id": pid,
        "game_id": gid,
        "game_ts": gts.isoformat() if hasattr(gts, "isoformat") else gts,
        "placing": placing,
        "rating_change": rating_change,
        "avg_worse_opp_rating": avg_worse,
        "avg_better_opp_rating": avg_better,
        "pregame_mu": float(pre_mu),
        "pregame_sigma": float(pre_sigma),
        
        # New Config Fields
        "sigma_cap_scale": float(sigma_cap_scale) if sigma_cap_scale is not None else None,
        "team_gap_pct": float(team_gap_pct) if team_gap_pct is not None else None,
        "team_gap_scale": float(team_gap_scale) if team_gap_scale is not None else None,
        "unbalanced_reduction_pct": float(unbalanced_reduction_pct) if unbalanced_reduction_pct is not None else None,
        
        "postgame_mu": float(next_pre_mu) if next_pre_mu is not None else None,
        "postgame_sigma": float(next_pre_sigma) if next_pre_sigma is not None else None,
        "next_pregame_mu": float(next_pre_mu) if next_pre_mu is not None else None,
        "next_pregame_sigma": float(next_pre_sigma) if next_pre_sigma is not None else None,
        "post_source": "next_game_pregame" if (next_pre_mu is not None) else None,
    })

placings = sorted(teams_map.keys())
teams = []
for p in placings:
    members = teams_map.get(p, [])
    if len(members) != 2:
        print(f"Warning: Placing {p} has {len(members)} players, expected 2.")
    teams.append(members)

default_output = Path(__file__).parent / "sim_inputs" / f"sim_inputs_{game_id}.json"
sim_inputs_dir = default_output.parent
if args.output:
    output_path = Path(args.output)
else:
    sim_inputs_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for path in sim_inputs_dir.glob("sim_inputs_*.json"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            num = int(suffix)
            if num > max_idx:
                max_idx = num
    output_path = sim_inputs_dir / f"sim_inputs_{max_idx + 1}.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

data = {
    "meta": {
        "note": "Teams formed by grouping two players sharing the same placing. Placings 1..8 map directly.",
        "rating_pref": f"Used pre_mu/pre_sigma from {mh_table}; targets derived from next-game pre-ratings."
    },
    "source_game_id": game_id,
    "players": players,
    "teams": teams,
    "placings": placings,
    "targets": targets,
    "target_games": target_games,
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"Wrote {output_path}")
