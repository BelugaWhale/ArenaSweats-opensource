#!/usr/bin/env python3
import math
import traceback
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from ranking_algorithm import instantiate_rating_model

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


DEFAULT_DAYS = 45
DEFAULT_LIMIT_GAMES = 2000
DEFAULT_MIN_TEAMMATE_MU = 20.0


def _quantile(values, q):
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    sorted_vals = sorted(float(v) for v in values)
    idx = int(round((len(sorted_vals) - 1) * q))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return float(sorted_vals[idx])


def _group_by_placing(players):
    groups = {}
    for player in players:
        placing = int(player["placing"])
        groups.setdefault(placing, []).append(player)
    for placing in range(1, 9):
        if len(groups.get(placing, [])) != 2:
            raise ValueError(f"Expected two players at placing {placing}, got {len(groups.get(placing, []))}")
    return groups


def _to_point_delta(mu_old, sigma_old, mu_new, sigma_new):
    before = (mu_old - (3.0 * sigma_old)) * 75.0
    after = (mu_new - (3.0 * sigma_new)) * 75.0
    return int(round(after - before))


def _simulate_game(game, model, use_grace):
    groups = _group_by_placing(game["players"])
    ranks = list(range(8))
    tau = float(model.tau)

    teams_orig = []
    teams_old_adj = []
    ordered_players = []
    grace_team_mask = []
    for placing in range(1, 9):
        team_players = sorted(groups[placing], key=lambda row: int(row["player_hash"]))
        ordered_players.append(team_players)
        team_reduction = 0.0
        if use_grace:
            team_reduction = max(float(team_players[0]["unbalanced_reduction_pct"]), float(team_players[1]["unbalanced_reduction_pct"]))
        grace_team_mask.append(team_reduction > 0.0)

        orig_team = []
        old_adj_team = []
        for player in team_players:
            orig_rating = model.rating(mu=float(player["pregame_mu"]), sigma=float(player["pregame_sigma"]))
            old_adj_rating = model.rating(mu=float(player["pregame_mu"]) * (1.0 - team_reduction), sigma=float(player["pregame_sigma"]))
            orig_team.append(orig_rating)
            old_adj_team.append(old_adj_rating)
        teams_orig.append(orig_team)
        teams_old_adj.append(old_adj_team)

    rated_teams = model.rate(teams_old_adj, ranks=ranks)

    final_by_hash = {}
    grace_by_hash = {}
    for team_idx in range(8):
        team_players = ordered_players[team_idx]
        orig_team = teams_orig[team_idx]
        old_adj_team = teams_old_adj[team_idx]
        new_adj_team = rated_teams[team_idx]

        provisional_team = []
        for idx in range(2):
            orig = orig_team[idx]
            old_adj = old_adj_team[idx]
            new_adj = new_adj_team[idx]
            delta_mu = new_adj.mu - old_adj.mu
            delta_sigma = new_adj.sigma - old_adj.sigma
            provisional_team.append([orig.mu + delta_mu, orig.sigma + delta_sigma])

        high_idx = 0 if float(team_players[0]["pregame_mu"]) >= float(team_players[1]["pregame_mu"]) else 1
        high_scale = float(team_players[high_idx]["team_gap_scale"])
        if high_scale < 1.0:
            high_orig = orig_team[high_idx]
            sigma_prior = math.sqrt((high_orig.sigma * high_orig.sigma) + (tau * tau))
            sigma_delta = provisional_team[high_idx][1] - sigma_prior
            provisional_team[high_idx][0] = high_orig.mu + ((provisional_team[high_idx][0] - high_orig.mu) * high_scale)
            provisional_team[high_idx][1] = sigma_prior + (sigma_delta * high_scale)

        for idx in range(2):
            player = team_players[idx]
            player_hash = int(player["player_hash"])
            mu_old = float(player["pregame_mu"])
            sigma_old = float(player["pregame_sigma"])
            mu_new = provisional_team[idx][0]
            sigma_new = provisional_team[idx][1]
            final_by_hash[player_hash] = {
                "delta_points": _to_point_delta(mu_old, sigma_old, mu_new, sigma_new),
                "placing": int(player["placing"]),
                "pregame_mu": mu_old,
            }
            grace_by_hash[player_hash] = grace_team_mask[team_idx]

    return final_by_hash, grace_by_hash


def _safety_floor_place(player_mu, teammate_mu, gm_cutoff_mu, diamond_cutoff_mu, min_teammate_mu, enable_top2_high_teammate):
    if player_mu < gm_cutoff_mu:
        return 0
    if teammate_mu < min_teammate_mu:
        return 0
    if teammate_mu < diamond_cutoff_mu:
        return 4
    if teammate_mu < gm_cutoff_mu:
        return 3
    if enable_top2_high_teammate:
        return 2
    return 0


def _apply_zero_sum_safety(deltas, floors):
    adjusted = {pid: float(value) for pid, value in deltas.items()}
    needs = {}
    for pid, floor in floors.items():
        if floor <= 0:
            continue
        value = adjusted[pid]
        if value < 0.0:
            needs[pid] = -value

    if not needs:
        return adjusted, 0.0, 0.0, {}

    total_need = sum(needs.values())
    donors = {pid: value for pid, value in adjusted.items() if value > 0.0}
    total_donor = sum(donors.values())
    transfer_total = min(total_need, total_donor)
    if transfer_total <= 0.0:
        return adjusted, total_need, transfer_total, {}

    receiver_scale = transfer_total / total_need
    for pid, need in needs.items():
        adjusted[pid] = adjusted[pid] + (need * receiver_scale)

    donor_burden = {}
    donor_scale = transfer_total / total_donor
    for pid, amount in donors.items():
        donated = amount * donor_scale
        adjusted[pid] = adjusted[pid] - donated
        donor_burden[pid] = donated

    return adjusted, total_need, transfer_total, donor_burden


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Grace vs Solo Safety")

    controls = ttk.Frame(frame, style="Panel.TFrame")
    controls.pack(fill="x", padx=10, pady=(10, 6))

    ttk.Label(controls, text="Region").grid(row=0, column=0, padx=(0, 8), pady=6, sticky="w")
    region_var = tk.StringVar(value="euw")
    region_combo = ttk.Combobox(
        controls,
        textvariable=region_var,
        values=sorted(region_to_ch_prefix.keys()),
        width=8,
        state="readonly",
    )
    region_combo.grid(row=0, column=1, padx=(0, 14), pady=6, sticky="w")

    ttk.Label(controls, text="Season").grid(row=0, column=2, padx=(0, 8), pady=6, sticky="w")
    season_var = tk.StringVar(value="live")
    season_combo = ttk.Combobox(
        controls,
        textvariable=season_var,
        values=["live", "2025season3"],
        width=12,
        state="readonly",
    )
    season_combo.grid(row=0, column=3, padx=(0, 14), pady=6, sticky="w")

    ttk.Label(controls, text="Days").grid(row=0, column=4, padx=(0, 8), pady=6, sticky="w")
    days_var = tk.StringVar(value=str(DEFAULT_DAYS))
    days_spin = ttk.Spinbox(controls, from_=1, to=3650, increment=1, textvariable=days_var, width=8)
    days_spin.grid(row=0, column=5, padx=(0, 14), pady=6, sticky="w")

    ttk.Label(controls, text="Games").grid(row=0, column=6, padx=(0, 8), pady=6, sticky="w")
    limit_games_var = tk.StringVar(value=str(DEFAULT_LIMIT_GAMES))
    limit_games_spin = ttk.Spinbox(controls, from_=10, to=5000, increment=10, textvariable=limit_games_var, width=8)
    limit_games_spin.grid(row=0, column=7, padx=(0, 14), pady=6, sticky="w")

    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=8, padx=(0, 8), pady=6, sticky="w")

    policy_controls = ttk.Frame(frame, style="Panel.TFrame")
    policy_controls.pack(fill="x", padx=10, pady=(0, 6))

    ttk.Label(policy_controls, text="Safety baseline").grid(row=0, column=0, padx=(0, 8), pady=6, sticky="w")
    baseline_var = tk.StringVar(value="observed")
    baseline_combo = ttk.Combobox(
        policy_controls,
        textvariable=baseline_var,
        values=["no_grace", "observed"],
        width=12,
        state="readonly",
    )
    baseline_combo.grid(row=0, column=1, padx=(0, 14), pady=6, sticky="w")

    ttk.Label(policy_controls, text="Min teammate mu").grid(row=0, column=2, padx=(0, 8), pady=6, sticky="w")
    min_teammate_mu_var = tk.StringVar(value=f"{DEFAULT_MIN_TEAMMATE_MU:.1f}")
    min_teammate_mu_entry = ttk.Entry(policy_controls, textvariable=min_teammate_mu_var, width=8)
    min_teammate_mu_entry.grid(row=0, column=3, padx=(0, 14), pady=6, sticky="w")

    top2_high_teammate_var = tk.BooleanVar(value=False)
    top2_high_teammate_toggle = ttk.Checkbutton(
        policy_controls,
        text="Enable top2 safety when teammate >= GM cutoff",
        variable=top2_high_teammate_var,
    )
    top2_high_teammate_toggle.grid(row=0, column=4, padx=(0, 14), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Chart 1-2 use macro inflation signals: grace distribution/exposure, grace-mass, mu-suppression, and observed delta contrasts. "
            "Chart 3-4 apply a zero-sum solo safety policy (GM+ protagonist only): teammate<Diamond -> top4 safety, "
            "Diamond<=teammate<GM -> top3 safety, optional top2 safety when teammate>=GM."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Ready. Press Regenerate.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return {
            "frame": frame,
            "run_query": lambda: set_status("matplotlib unavailable"),
            "tab_text": "Experiment - Grace vs Solo Safety",
        }

    figure = Figure(figsize=(14.5, 8.0), dpi=100)
    gs = figure.add_gridspec(2, 2, left=0.05, right=0.985, top=0.92, bottom=0.09, wspace=0.28, hspace=0.36)
    ax_grace_place = figure.add_subplot(gs[0, 0])
    ax_grace_net = figure.add_subplot(gs[0, 1])
    ax_safety_rates = figure.add_subplot(gs[1, 0])
    ax_donor_burden = figure.add_subplot(gs[1, 1])

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    chart_state = {"dataset": None}

    def _draw_placeholder():
        for ax in [ax_grace_place, ax_grace_net, ax_safety_rates, ax_donor_burden]:
            ax.clear()
            ax.text(0.5, 0.5, "Press Regenerate to load data.", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.2)
        ax_grace_place.set_title("Grace effect by placing")
        ax_grace_net.set_title("Grace net point impact")
        ax_safety_rates.set_title("Safety outcomes")
        ax_donor_burden.set_title("Donor burden by placing")
        canvas.draw_idle()

    def _run_backtest():
        dataset = chart_state["dataset"]
        if dataset is None:
            return None

        gm_cutoff_mu = float(dataset["cutoffs"]["gm_plus_min_mu"])
        diamond_cutoff_mu = float(dataset["cutoffs"]["diamond_plus_min_mu"])
        min_teammate_mu = float(min_teammate_mu_var.get().strip())
        if min_teammate_mu <= 0.0:
            raise ValueError("Min teammate mu must be > 0")

        games = dataset["games"]
        if not games:
            raise ValueError("No games in dataset")

        grace_delta_by_place_sum = {placing: 0.0 for placing in range(1, 9)}
        grace_delta_by_place_count = {placing: 0 for placing in range(1, 9)}
        nongrace_delta_by_place_sum = {placing: 0.0 for placing in range(1, 9)}
        nongrace_delta_by_place_count = {placing: 0 for placing in range(1, 9)}
        grace_team_delta_sum = 0.0
        grace_team_delta_count = 0
        nongrace_team_delta_sum = 0.0
        nongrace_team_delta_count = 0
        grace_team_count = 0
        total_team_count = 0
        games_with_grace = 0
        grace_mass_sum = 0.0
        mu_suppression_sum = 0.0
        reduction_positive = []
        game_delta_with_grace_sum = 0.0
        game_delta_with_grace_count = 0
        game_delta_without_grace_sum = 0.0
        game_delta_without_grace_count = 0

        protected_cases = 0
        protected_neg_before = 0
        protected_neg_after = 0
        protected_nonneg_after = 0
        donor_burden_by_place = {placing: 0.0 for placing in range(1, 9)}
        per_game_net_drift_abs_sum = 0.0
        mean_by_placing_before_sum = {placing: 0.0 for placing in range(1, 9)}
        mean_by_placing_before_count = {placing: 0 for placing in range(1, 9)}
        mean_by_placing_after_sum = {placing: 0.0 for placing in range(1, 9)}
        mean_by_placing_after_count = {placing: 0 for placing in range(1, 9)}

        baseline_mode = baseline_var.get().strip().lower()
        if baseline_mode not in {"no_grace", "observed"}:
            raise ValueError(f"Unsupported safety baseline: {baseline_mode}")
        model = instantiate_rating_model() if baseline_mode == "no_grace" else None

        for game in games:
            players = game["players"]
            groups = _group_by_placing(players)

            observed_deltas = {int(row["player_hash"]): float(row["rating_change"]) for row in players}
            no_grace_replay_deltas = None
            if baseline_mode == "no_grace":
                simulated_no_grace, _ = _simulate_game(game, model, use_grace=False)
                no_grace_replay_deltas = {pid: float(payload["delta_points"]) for pid, payload in simulated_no_grace.items()}

            game_has_grace = False
            game_delta_sum = 0.0
            for placing in range(1, 9):
                team_players = sorted(groups[placing], key=lambda row: int(row["player_hash"]))
                team_reduction = max(float(team_players[0]["unbalanced_reduction_pct"]), float(team_players[1]["unbalanced_reduction_pct"]))
                team_mu_sum = float(team_players[0]["pregame_mu"]) + float(team_players[1]["pregame_mu"])
                team_delta = float(team_players[0]["rating_change"]) + float(team_players[1]["rating_change"])
                game_delta_sum += team_delta
                total_team_count += 1

                if team_reduction > 0.0:
                    game_has_grace = True
                    grace_team_count += 1
                    reduction_positive.append(team_reduction)
                    grace_mass_sum += team_reduction
                    mu_suppression_sum += team_reduction * team_mu_sum
                    grace_team_delta_sum += team_delta
                    grace_team_delta_count += 1
                    grace_delta_by_place_sum[placing] += team_delta
                    grace_delta_by_place_count[placing] += 1
                else:
                    nongrace_team_delta_sum += team_delta
                    nongrace_team_delta_count += 1
                    nongrace_delta_by_place_sum[placing] += team_delta
                    nongrace_delta_by_place_count[placing] += 1

            if game_has_grace:
                games_with_grace += 1
                game_delta_with_grace_sum += game_delta_sum
                game_delta_with_grace_count += 1
            else:
                game_delta_without_grace_sum += game_delta_sum
                game_delta_without_grace_count += 1

            if baseline_mode == "observed":
                baseline_deltas = observed_deltas
            else:
                baseline_deltas = no_grace_replay_deltas

            floors = {}
            for placing in range(1, 9):
                team_players = sorted(groups[placing], key=lambda row: int(row["player_hash"]))
                first = team_players[0]
                second = team_players[1]
                first_mu = float(first["pregame_mu"])
                second_mu = float(second["pregame_mu"])
                first_floor = _safety_floor_place(
                    player_mu=first_mu,
                    teammate_mu=second_mu,
                    gm_cutoff_mu=gm_cutoff_mu,
                    diamond_cutoff_mu=diamond_cutoff_mu,
                    min_teammate_mu=min_teammate_mu,
                    enable_top2_high_teammate=bool(top2_high_teammate_var.get()),
                )
                second_floor = _safety_floor_place(
                    player_mu=second_mu,
                    teammate_mu=first_mu,
                    gm_cutoff_mu=gm_cutoff_mu,
                    diamond_cutoff_mu=diamond_cutoff_mu,
                    min_teammate_mu=min_teammate_mu,
                    enable_top2_high_teammate=bool(top2_high_teammate_var.get()),
                )
                floors[int(first["player_hash"])] = first_floor
                floors[int(second["player_hash"])] = second_floor

            adjusted, _need, _transfer, donor_burden_by_pid = _apply_zero_sum_safety(baseline_deltas, floors)
            pre_sum = sum(float(value) for value in baseline_deltas.values())
            post_sum = sum(float(value) for value in adjusted.values())
            per_game_net_drift_abs_sum += abs(post_sum - pre_sum)

            for placing in range(1, 9):
                team_players = groups[placing]
                for player in team_players:
                    pid = int(player["player_hash"])
                    before = float(baseline_deltas[pid])
                    after = float(adjusted[pid])
                    mean_by_placing_before_sum[placing] += before
                    mean_by_placing_before_count[placing] += 1
                    mean_by_placing_after_sum[placing] += after
                    mean_by_placing_after_count[placing] += 1
                    floor_place = int(floors[pid])
                    if floor_place > 0 and placing <= floor_place:
                        protected_cases += 1
                        if before < 0.0:
                            protected_neg_before += 1
                        if after < 0.0:
                            protected_neg_after += 1
                        else:
                            protected_nonneg_after += 1
                    if pid in donor_burden_by_pid:
                        donor_burden_by_place[placing] += float(donor_burden_by_pid[pid])

        avg_grace_diff_by_place = []
        for placing in range(1, 9):
            grace_count = grace_delta_by_place_count[placing]
            nongrace_count = nongrace_delta_by_place_count[placing]
            grace_avg = (grace_delta_by_place_sum[placing] / grace_count) if grace_count > 0 else 0.0
            nongrace_avg = (nongrace_delta_by_place_sum[placing] / nongrace_count) if nongrace_count > 0 else 0.0
            avg_grace_diff_by_place.append(grace_avg - nongrace_avg)

        mean_before_by_place = []
        mean_after_by_place = []
        for placing in range(1, 9):
            count_before = mean_by_placing_before_count[placing]
            count_after = mean_by_placing_after_count[placing]
            mean_before_by_place.append((mean_by_placing_before_sum[placing] / count_before) if count_before > 0 else 0.0)
            mean_after_by_place.append((mean_by_placing_after_sum[placing] / count_after) if count_after > 0 else 0.0)

        ordering_before_ok = True
        ordering_after_ok = True
        for idx in range(7):
            if mean_before_by_place[idx] < mean_before_by_place[idx + 1]:
                ordering_before_ok = False
            if mean_after_by_place[idx] < mean_after_by_place[idx + 1]:
                ordering_after_ok = False

        loaded_games = int(dataset["meta"]["loaded_games"])
        protected_neg_rate_before = (protected_neg_before / protected_cases) if protected_cases > 0 else 0.0
        protected_neg_rate_after = (protected_neg_after / protected_cases) if protected_cases > 0 else 0.0
        protected_nonneg_rate_after = (protected_nonneg_after / protected_cases) if protected_cases > 0 else 0.0
        share_games_with_grace = (games_with_grace / loaded_games) if loaded_games > 0 else 0.0
        share_teams_with_grace = (grace_team_count / total_team_count) if total_team_count > 0 else 0.0
        avg_grace_pct_given = (grace_mass_sum / grace_team_count) if grace_team_count > 0 else 0.0
        avg_grace_mass_per_game = grace_mass_sum / loaded_games
        avg_mu_suppression_per_game = mu_suppression_sum / loaded_games
        avg_team_delta_grace = (grace_team_delta_sum / grace_team_delta_count) if grace_team_delta_count > 0 else 0.0
        avg_team_delta_nongrace = (nongrace_team_delta_sum / nongrace_team_delta_count) if nongrace_team_delta_count > 0 else 0.0
        avg_game_delta_with_grace = (game_delta_with_grace_sum / game_delta_with_grace_count) if game_delta_with_grace_count > 0 else 0.0
        avg_game_delta_without_grace = (game_delta_without_grace_sum / game_delta_without_grace_count) if game_delta_without_grace_count > 0 else 0.0

        return {
            "loaded_games": loaded_games,
            "gm_cutoff_mu": gm_cutoff_mu,
            "diamond_cutoff_mu": diamond_cutoff_mu,
            "share_games_with_grace": share_games_with_grace,
            "share_teams_with_grace": share_teams_with_grace,
            "avg_grace_pct_given": avg_grace_pct_given,
            "grace_p50": _quantile(reduction_positive, 0.50),
            "grace_p75": _quantile(reduction_positive, 0.75),
            "grace_p90": _quantile(reduction_positive, 0.90),
            "grace_p95": _quantile(reduction_positive, 0.95),
            "grace_p99": _quantile(reduction_positive, 0.99),
            "grace_mass_per_game": avg_grace_mass_per_game,
            "mu_suppression_per_game": avg_mu_suppression_per_game,
            "avg_team_delta_grace": avg_team_delta_grace,
            "avg_team_delta_nongrace": avg_team_delta_nongrace,
            "avg_game_delta_with_grace": avg_game_delta_with_grace,
            "avg_game_delta_without_grace": avg_game_delta_without_grace,
            "avg_grace_diff_by_place": avg_grace_diff_by_place,
            "protected_cases": protected_cases,
            "protected_neg_rate_before": protected_neg_rate_before,
            "protected_neg_rate_after": protected_neg_rate_after,
            "protected_nonneg_rate_after": protected_nonneg_rate_after,
            "donor_burden_by_place": [donor_burden_by_place[placing] / loaded_games for placing in range(1, 9)],
            "mean_before_by_place": mean_before_by_place,
            "mean_after_by_place": mean_after_by_place,
            "ordering_before_ok": ordering_before_ok,
            "ordering_after_ok": ordering_after_ok,
            "avg_per_game_net_drift_abs": per_game_net_drift_abs_sum / loaded_games,
            "baseline_mode": baseline_mode,
            "min_teammate_mu": min_teammate_mu,
            "top2_high_teammate": bool(top2_high_teammate_var.get()),
        }

    def _draw_chart():
        result = _run_backtest()
        if result is None:
            _draw_placeholder()
            return

        places = [1, 2, 3, 4, 5, 6, 7, 8]

        ax_grace_place.clear()
        ax_grace_place.bar(places, result["avg_grace_diff_by_place"], color="#2b6ea8")
        ax_grace_place.axhline(0.0, color="#666666", linewidth=1.0)
        ax_grace_place.set_title("Observed Delta Gap By Placing (Grace Teams - Non-Grace Teams)")
        ax_grace_place.set_xlabel("placing")
        ax_grace_place.set_ylabel("avg team rating_change gap")
        ax_grace_place.set_xticks(places)
        ax_grace_place.grid(alpha=0.2)

        ax_grace_net.clear()
        net_labels = ["Game share\nwith grace %", "Avg grace\nwhen given %", "Mu suppression\nper game"]
        net_values = [
            result["share_games_with_grace"] * 100.0,
            result["avg_grace_pct_given"] * 100.0,
            result["mu_suppression_per_game"],
        ]
        ax_grace_net.bar(net_labels, net_values, color=["#2a7f62", "#c13b20", "#7a4ca4"])
        ax_grace_net.axhline(0.0, color="#666666", linewidth=1.0)
        ax_grace_net.set_title("Macro Grace Exposure / Strength")
        ax_grace_net.set_ylabel("mixed units")
        ax_grace_net.grid(alpha=0.2)

        ax_safety_rates.clear()
        safety_labels = ["neg rate before", "neg rate after", "non-neg after"]
        safety_values = [
            result["protected_neg_rate_before"] * 100.0,
            result["protected_neg_rate_after"] * 100.0,
            result["protected_nonneg_rate_after"] * 100.0,
        ]
        ax_safety_rates.bar(safety_labels, safety_values, color=["#c13b20", "#2b6ea8", "#2a7f62"])
        ax_safety_rates.set_ylim(0.0, 100.0)
        ax_safety_rates.set_title("Protected Outcome Rates (Zero-Sum Safety)")
        ax_safety_rates.set_ylabel("share of protected rows (%)")
        ax_safety_rates.grid(alpha=0.2)

        ax_donor_burden.clear()
        ax_donor_burden.bar(places, result["donor_burden_by_place"], color="#805d3f")
        ax_donor_burden.set_title("Donor Burden By Placing (Per Game)")
        ax_donor_burden.set_xlabel("placing")
        ax_donor_burden.set_ylabel("donated points per game")
        ax_donor_burden.set_xticks(places)
        ax_donor_burden.grid(alpha=0.2)

        figure.suptitle(
            "Unbalanced Grace Inflation + Solo Safety Backtest | "
            f"games={result['loaded_games']} baseline={result['baseline_mode']}",
            fontsize=11,
        )
        canvas.draw_idle()

        summary_var.set(
            f"gm_cutoff_mu={result['gm_cutoff_mu']:.3f} | diamond_cutoff_mu={result['diamond_cutoff_mu']:.3f} | "
            f"games_with_grace={result['share_games_with_grace'] * 100.0:.2f}% | "
            f"teams_with_grace={result['share_teams_with_grace'] * 100.0:.2f}% | "
            f"avg_grace={result['avg_grace_pct_given'] * 100.0:.3f}% | "
            f"grace_p95={result['grace_p95'] * 100.0:.3f}% | "
            f"mu_suppression/game={result['mu_suppression_per_game']:.3f} | "
            f"game_delta_gap={(result['avg_game_delta_with_grace'] - result['avg_game_delta_without_grace']):.3f} | "
            f"protected_cases={result['protected_cases']:,} | "
            f"protected_neg_before={result['protected_neg_rate_before'] * 100.0:.2f}% -> after={result['protected_neg_rate_after'] * 100.0:.2f}% | "
            f"net_drift_abs/game={result['avg_per_game_net_drift_abs']:.6f} | "
            f"ordering_before_ok={result['ordering_before_ok']} ordering_after_ok={result['ordering_after_ok']}"
        )
        log_console(
            "[INFO] Grace/Solo safety backtest "
            f"games={result['loaded_games']} baseline={result['baseline_mode']} "
            f"min_teammate_mu={result['min_teammate_mu']:.2f} top2_high_teammate={result['top2_high_teammate']} "
            f"games_with_grace={result['share_games_with_grace']:.4f} "
            f"avg_grace_given={result['avg_grace_pct_given']:.6f} "
            f"grace_p95={result['grace_p95']:.6f} "
            f"mu_suppression_per_game={result['mu_suppression_per_game']:.6f} "
            f"avg_game_delta_with_grace={result['avg_game_delta_with_grace']:.4f} "
            f"avg_game_delta_without_grace={result['avg_game_delta_without_grace']:.4f} "
            f"protected_neg_before={result['protected_neg_rate_before']:.4f} "
            f"protected_neg_after={result['protected_neg_rate_after']:.4f} "
            f"net_drift_abs={result['avg_per_game_net_drift_abs']:.8f} "
            f"ordering_before={result['ordering_before_ok']} ordering_after={result['ordering_after_ok']}"
        )

    def run_query():
        try:
            region = region_var.get().strip().lower()
            if region not in region_to_ch_prefix:
                raise ValueError(f"Unknown region: {region}")
            season = season_var.get().strip().lower()
            days = int(days_var.get().strip())
            limit_games = int(limit_games_var.get().strip())
            if days <= 0:
                raise ValueError("Days must be > 0")
            if limit_games <= 0:
                raise ValueError("Games must be > 0")

            set_status(f"Loading grace/safety dataset for {region}...")
            log_console(
                "[INFO] Grace/Solo safety query "
                f"region={region} season={season} days={days} limit_games={limit_games}"
            )
            chart_state["dataset"] = private_ch.load_unbalanced_inflation_solo_safety_dataset(
                region=region,
                ch_prefix=region_to_ch_prefix[region],
                season=season,
                days=days,
                limit_games=limit_games,
            )
            _draw_chart()
            set_status(
                f"Loaded grace/safety dataset for {region} "
                f"({chart_state['dataset']['meta']['loaded_games']:,} games)"
            )
        except Exception as exc:
            log_console(f"[ERROR] Grace/Solo safety experiment failed: {exc}")
            log_console(traceback.format_exc().rstrip())
            messagebox.showerror("Grace/Solo Safety Experiment Failed", str(exc))
            set_status("Grace/Solo safety experiment failed")

    def _run_if_ready():
        if chart_state["dataset"] is None:
            return
        try:
            _draw_chart()
            set_status("Grace/Solo safety chart updated")
        except Exception as exc:
            log_console(f"[ERROR] Grace/Solo safety redraw failed: {exc}")
            log_console(traceback.format_exc().rstrip())
            messagebox.showerror("Grace/Solo Safety", str(exc))
            set_status("Grace/Solo safety redraw failed")

    regenerate_btn.configure(command=run_query)
    baseline_combo.bind("<<ComboboxSelected>>", lambda _event: _run_if_ready())
    top2_high_teammate_toggle.configure(command=_run_if_ready)
    min_teammate_mu_entry.bind("<Return>", lambda _event: _run_if_ready())
    region_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("dataset", None),
            summary_var.set("Ready. Press Regenerate."),
            _draw_placeholder(),
        ),
    )
    season_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("dataset", None),
            summary_var.set("Ready. Press Regenerate."),
            _draw_placeholder(),
        ),
    )

    _draw_placeholder()
    return {
        "frame": frame,
        "run_query": run_query,
        "tab_text": "Experiment - Grace vs Solo Safety",
    }
