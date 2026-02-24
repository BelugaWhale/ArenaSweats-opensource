#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from clickhouse_connect import get_client
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

parser = argparse.ArgumentParser(description="Plot sigma vs total_games per region from ClickHouse player_rankhistory.")
parser.add_argument("--regions", help="Comma-separated regions (default: all 15).")
parser.add_argument("--min-games", type=int, default=0, help="Minimum total_games to include (default: 0).")
parser.add_argument("--max-games", type=int, default=300, help="Maximum total_games to include (default: 300).")
parser.add_argument("--gp-bin", type=int, default=5, help="total_games bin size for clustering (default: 5).")
parser.add_argument("--output", "-o", help="Output HTML path (default: validations/plots/sigma_games_by_region.html).")
args = parser.parse_args()

db_to_regions = {
    "DB": ["euw", "br", "sea", "ru", "me"],
    "SCRAPE": ["na", "vn", "las", "tr", "oce"],
    "SPLIT": ["kr", "eune", "lan", "tw", "jp"],
}
region_to_alias = {}
for alias, regions_for_alias in db_to_regions.items():
    for region in regions_for_alias:
        region_to_alias[region] = alias

regions = [r for alias in db_to_regions.values() for r in alias]
if args.regions:
    regions = [r.strip().lower() for r in args.regions.split(",") if r.strip()]
if not regions:
    raise SystemExit("No regions provided.")
if args.gp_bin <= 0:
    raise SystemExit("--gp-bin must be > 0")
if args.min_games < 0:
    raise SystemExit("--min-games must be >= 0")
if args.max_games <= 0:
    raise SystemExit("--max-games must be > 0")
if args.max_games <= args.min_games:
    raise SystemExit("--max-games must be greater than --min-games")
for region in regions:
    if region not in region_to_alias:
        raise SystemExit(f"Unknown region: {region}")

ch_clients = {}
def get_client_for_alias(alias):
    if alias in ch_clients:
        return ch_clients[alias]
    prefix = f"{alias}CH"
    host = os.getenv(f"{prefix}_HOST")
    scheme = (os.getenv(f"{prefix}_SCHEME") or "").lower()
    sslmode = (os.getenv(f"{prefix}_SSLMODE") or "").lower()
    secure = scheme == "https" or sslmode in {"require", "verify-ca", "verify-full"}
    port = int(os.getenv(f"{prefix}_PORT", "8443" if secure else "8123"))
    if not host:
        raise SystemExit(f"Missing {prefix}_HOST")
    client = get_client(
        host=host,
        port=port,
        username=os.getenv(f"{prefix}_USER", "default"),
        password=os.getenv(f"{prefix}_PASSWORD", ""),
        database=os.getenv(f"{prefix}_NAME") or None,
        secure=secure,
        verify=(os.getenv(f"{prefix}_SSL_VERIFY") or str(secure)).lower() in {"1", "true", "yes", "y", "on"},
        ca_cert=os.getenv(f"{prefix}_SSLROOTCERT") or None,
        client_cert=os.getenv(f"{prefix}_SSLCERT") or None,
        client_cert_key=os.getenv(f"{prefix}_SSLKEY") or None,
        connect_timeout=float(os.getenv(f"{prefix}_CONNECT_TIMEOUT", "8")),
        send_receive_timeout=float(os.getenv(f"{prefix}_RW_TIMEOUT", "30")),
    )
    ch_clients[alias] = client
    return client

fig = go.Figure()

for region in regions:
    alias = region_to_alias[region]
    client = get_client_for_alias(alias)
    table = f"{region}_player_rankhistory"
    query = f"""
    SELECT
        intDiv(total_games, {{gp_bin:Int32}}) * {{gp_bin:Int32}} AS total_games_bin,
        avg(sigma) AS sigma_avg,
        count() AS player_count
    FROM {table}
    WHERE total_games >= {{min_games:Int32}} AND total_games <= {{max_games:Int32}} AND sigma IS NOT NULL
    GROUP BY total_games_bin
    ORDER BY total_games_bin
    """
    rows = client.query(query, parameters={"gp_bin": args.gp_bin, "min_games": args.min_games, "max_games": args.max_games}).result_rows
    if not rows:
        raise SystemExit(f"No rows returned for region {region}.")
    x_vals = [int(r[0]) for r in rows]
    y_vals = [float(r[1]) for r in rows]
    counts = [int(r[2]) for r in rows]
    sizes = [max(4.0, min(20.0, (c ** 0.5))) for c in counts]
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        name=region.upper(),
        marker={"size": sizes, "opacity": 0.7},
        line={"width": 2},
        customdata=counts,
    hovertemplate="total_games=%{x}<br>sigma=%{y:.3f}<br>rows=%{customdata}<extra></extra>",
    ))

fig.update_layout(
    title="Sigma vs Games Played (Clustered by Games Played Bin)",
    xaxis_title="games_played (binned)",
    yaxis_title="sigma (avg per bin)",
    yaxis={"rangemode": "tozero"},
    legend_title="Region (click to toggle)",
    template="plotly_white",
)

output_path = Path(args.output) if args.output else Path(__file__).parent / "plots" / "sigma_games_by_region.html"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(output_path), include_plotlyjs="cdn")

print(f"Wrote {output_path}")
