#!/usr/bin/env python3
import tkinter as tk
import traceback
from tkinter import messagebox
from tkinter import ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


DEFAULT_DAYS = 45
DEFAULT_LIMIT_GAMES = 2000
PENALTY_WEIGHT_BY_PLACING = {5: 1.0, 6: 1.2, 7: 1.4, 8: 1.6}


def _is_diamond_or_higher(league_rank):
    normalized = str(league_rank or "").strip().lower().replace(" ", "")
    if not normalized:
        return False
    if normalized.startswith("challenger"):
        return True
    if normalized.startswith("grandmaster"):
        return True
    if normalized.startswith("master"):
        return True
    if normalized.startswith("diamond"):
        return True
    return False


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Emerald-or-Below Boxplot")

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
    limit_games_spin = ttk.Spinbox(controls, from_=10, to=10000, increment=10, textvariable=limit_games_var, width=8)
    limit_games_spin.grid(row=0, column=7, padx=(0, 14), pady=6, sticky="w")

    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=8, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Two charts: baseline and adjusted. Filter keeps teams with exactly one Diamond/Master/Grand Master/"
            "Challenger player plus one Emerald-or-below player, then uses only the Emerald-or-below rating_change. "
            "Adjustment floors negative deltas at places 3/4 to 0 and applies a calibrated heavier loss multiplier at 5-8."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Ready. Press Regenerate to load box-plot data.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return {
            "frame": frame,
            "run_query": lambda: set_status("matplotlib unavailable"),
            "tab_text": "Experiment - Emerald-or-Below Boxplot",
        }

    figure = Figure(figsize=(14.8, 6.2), dpi=100)
    ax_baseline = figure.add_subplot(1, 2, 1)
    ax_adjusted = figure.add_subplot(1, 2, 2, sharey=ax_baseline)

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    chart_state = {
        "placing_to_deltas_baseline": None,
        "placing_to_deltas_adjusted": None,
        "meta": None,
    }

    def _draw_placeholder():
        ax_baseline.clear()
        ax_adjusted.clear()
        ax_baseline.text(
            0.5,
            0.5,
            "Press Regenerate to load baseline data.",
            ha="center",
            va="center",
            transform=ax_baseline.transAxes,
        )
        ax_adjusted.text(
            0.5,
            0.5,
            "Press Regenerate to load adjusted data.",
            ha="center",
            va="center",
            transform=ax_adjusted.transAxes,
        )
        for axis in [ax_baseline, ax_adjusted]:
            axis.set_xlabel("placing")
            axis.set_ylabel("rating_change")
            axis.set_xticks(list(range(1, 9)))
            axis.set_xlim(0.5, 8.5)
            axis.grid(alpha=0.2, axis="y")
            axis.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
        ax_baseline.set_title("Baseline: Emerald-or-below Rating Change")
        ax_adjusted.set_title("Adjusted: 3rd/4th Floor + 5th-8th Heavier Loss")
        canvas.draw_idle()

    def _draw_chart():
        placing_to_deltas_baseline = chart_state["placing_to_deltas_baseline"]
        placing_to_deltas_adjusted = chart_state["placing_to_deltas_adjusted"]
        if placing_to_deltas_baseline is None or placing_to_deltas_adjusted is None:
            _draw_placeholder()
            return

        positions = list(range(1, 9))
        baseline_positions = [placing for placing in positions if placing_to_deltas_baseline[placing]]
        baseline_series = [placing_to_deltas_baseline[placing] for placing in baseline_positions]
        adjusted_positions = [placing for placing in positions if placing_to_deltas_adjusted[placing]]
        adjusted_series = [placing_to_deltas_adjusted[placing] for placing in adjusted_positions]

        ax_baseline.clear()
        if baseline_positions:
            baseline_boxplot = ax_baseline.boxplot(
                baseline_series,
                positions=baseline_positions,
                widths=0.62,
                patch_artist=True,
                showmeans=True,
                meanline=False,
            )
            for patch in baseline_boxplot["boxes"]:
                patch.set_facecolor("#6ca6d9")
                patch.set_alpha(0.55)
            for median in baseline_boxplot["medians"]:
                median.set_color("#19324a")
                median.set_linewidth(1.8)
            for mean in baseline_boxplot["means"]:
                mean.set_marker("o")
                mean.set_markerfacecolor("#b0352d")
                mean.set_markeredgecolor("#b0352d")
                mean.set_markersize(4.0)
        else:
            ax_baseline.text(0.5, 0.5, "No matching baseline rows.", ha="center", va="center", transform=ax_baseline.transAxes)

        ax_adjusted.clear()
        if adjusted_positions:
            adjusted_boxplot = ax_adjusted.boxplot(
                adjusted_series,
                positions=adjusted_positions,
                widths=0.62,
                patch_artist=True,
                showmeans=True,
                meanline=False,
            )
            for patch in adjusted_boxplot["boxes"]:
                patch.set_facecolor("#d9876c")
                patch.set_alpha(0.55)
            for median in adjusted_boxplot["medians"]:
                median.set_color("#4a1f19")
                median.set_linewidth(1.8)
            for mean in adjusted_boxplot["means"]:
                mean.set_marker("o")
                mean.set_markerfacecolor("#1f5c3f")
                mean.set_markeredgecolor("#1f5c3f")
                mean.set_markersize(4.0)
        else:
            ax_adjusted.text(0.5, 0.5, "No matching adjusted rows.", ha="center", va="center", transform=ax_adjusted.transAxes)

        all_values = []
        for placing in positions:
            all_values.extend(placing_to_deltas_baseline[placing])
            all_values.extend(placing_to_deltas_adjusted[placing])
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            if y_min == y_max:
                y_pad = 1.0
            else:
                y_pad = (y_max - y_min) * 0.08
            y_limits = (y_min - y_pad, y_max + y_pad)
        else:
            y_limits = (-1.0, 1.0)

        for axis in [ax_baseline, ax_adjusted]:
            axis.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
            axis.set_xlabel("placing")
            axis.set_ylabel("rating_change")
            axis.set_xticks(positions)
            axis.set_xlim(0.5, 8.5)
            axis.set_ylim(y_limits)
            axis.grid(alpha=0.2, axis="y")

        ax_baseline.set_title("Baseline: Emerald-or-below Rating Change")
        ax_adjusted.set_title("Adjusted: 3rd/4th Floor + 5th-8th Heavier Loss")
        ax_adjusted.text(
            0.02,
            0.98,
            (
                f"net cohort delta: {chart_state['meta']['net_drift']:+.6f}\n"
                f"floor gain (3/4): +{chart_state['meta']['floor_boost']:.6f}\n"
                f"extra loss (5-8): -{chart_state['meta']['extra_loss_5_to_8']:.6f}\n"
                f"holdout net delta: {chart_state['meta']['holdout_net_drift']:+.6f}\n"
                f"holdout split: {chart_state['meta']['holdout_train_count']} train / {chart_state['meta']['holdout_test_count']} test\n"
                f"holdout alpha: {chart_state['meta']['holdout_alpha']:.6f}"
            ),
            transform=ax_adjusted.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "#f0f2f5", "alpha": 0.85, "edgecolor": "#b7bec7", "boxstyle": "round,pad=0.3"},
        )
        ax_adjusted.text(
            0.98,
            0.98,
            (
                f"p5 mean: {chart_state['meta']['place_mean_baseline'][5]:+.3f} -> {chart_state['meta']['place_mean_adjusted'][5]:+.3f} "
                f"(d={chart_state['meta']['place_mean_adjusted'][5] - chart_state['meta']['place_mean_baseline'][5]:+.3f})\n"
                f"p6 mean: {chart_state['meta']['place_mean_baseline'][6]:+.3f} -> {chart_state['meta']['place_mean_adjusted'][6]:+.3f} "
                f"(d={chart_state['meta']['place_mean_adjusted'][6] - chart_state['meta']['place_mean_baseline'][6]:+.3f})\n"
                f"p7 mean: {chart_state['meta']['place_mean_baseline'][7]:+.3f} -> {chart_state['meta']['place_mean_adjusted'][7]:+.3f} "
                f"(d={chart_state['meta']['place_mean_adjusted'][7] - chart_state['meta']['place_mean_baseline'][7]:+.3f})\n"
                f"p8 mean: {chart_state['meta']['place_mean_baseline'][8]:+.3f} -> {chart_state['meta']['place_mean_adjusted'][8]:+.3f} "
                f"(d={chart_state['meta']['place_mean_adjusted'][8] - chart_state['meta']['place_mean_baseline'][8]:+.3f})"
            ),
            transform=ax_adjusted.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox={"facecolor": "#f0f2f5", "alpha": 0.85, "edgecolor": "#b7bec7", "boxstyle": "round,pad=0.3"},
        )

        counts_summary = " | ".join([f"p{placing}:{len(placing_to_deltas_baseline[placing])}" for placing in positions])
        summary_var.set(
            f"matched_teams={chart_state['meta']['matched_teams']:,} | matched_games={chart_state['meta']['matched_games']:,} | "
            f"loaded_games={chart_state['meta']['loaded_games']:,} | "
            f"m5={chart_state['meta']['multiplier_by_placing'][5]:.4f} "
            f"m6={chart_state['meta']['multiplier_by_placing'][6]:.4f} "
            f"m7={chart_state['meta']['multiplier_by_placing'][7]:.4f} "
            f"m8={chart_state['meta']['multiplier_by_placing'][8]:.4f} | "
            f"floored(3/4)={chart_state['meta']['floored_count']:,} | "
            f"floor_gain={chart_state['meta']['floor_boost']:.6f} | "
            f"extra_loss_5to8={chart_state['meta']['extra_loss_5_to_8']:.6f} | "
            f"net_drift={chart_state['meta']['net_drift']:+.6f} | {counts_summary}"
        )
        figure.suptitle(
            f"region={chart_state['meta']['region']} season={chart_state['meta']['season']} "
            f"days={chart_state['meta']['days']} limit_games={chart_state['meta']['limit_games']}",
            fontsize=11,
        )
        canvas.draw_idle()

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

            set_status(f"Loading Emerald-or-below baseline/adjusted box-plot data for {region}...")
            log_console(
                "[INFO] Emerald-or-below boxplot query "
                f"region={region} season={season} days={days} limit_games={limit_games} "
                "filter=teams_with_exactly_one_non_diamond_plus"
            )
            dataset = private_ch.load_unbalanced_inflation_solo_safety_dataset(
                region=region,
                ch_prefix=region_to_ch_prefix[region],
                season=season,
                days=days,
                limit_games=limit_games,
            )

            placing_to_deltas_baseline = {placing: [] for placing in range(1, 9)}
            matched_teams = 0
            matched_games = 0
            matched_records = []
            for game in dataset["games"]:
                placing_groups = {placing: [] for placing in range(1, 9)}
                for player in game["players"]:
                    placing_groups[int(player["placing"])].append(player)

                game_matched = False
                for placing in range(1, 9):
                    team_players = placing_groups[placing]
                    if len(team_players) != 2:
                        raise ValueError(
                            f"Game {game['game_id']} placing {placing} has {len(team_players)} players (expected 2)"
                        )
                    first_is_high = _is_diamond_or_higher(team_players[0].get("league_rank"))
                    second_is_high = _is_diamond_or_higher(team_players[1].get("league_rank"))
                    if (1 if first_is_high else 0) + (1 if second_is_high else 0) != 1:
                        continue

                    low_tier_player = team_players[0] if not first_is_high else team_players[1]
                    low_tier_delta = float(low_tier_player["rating_change"])
                    placing_to_deltas_baseline[placing].append(low_tier_delta)
                    matched_records.append(
                        {
                            "game_ts": str(game["game_ts"]),
                            "placing": placing,
                            "delta": low_tier_delta,
                        }
                    )
                    matched_teams += 1
                    game_matched = True

                if game_matched:
                    matched_games += 1

            if matched_teams == 0:
                raise ValueError("No teams matched filter: exactly one Diamond+-or-higher player and one Emerald-or-below player")

            floor_boost = 0.0
            floored_count = 0
            weighted_penalty_mass = 0.0
            for placing in [3, 4]:
                for delta in placing_to_deltas_baseline[placing]:
                    if delta < 0.0:
                        floor_boost += -delta
                        floored_count += 1
            for placing in [5, 6, 7, 8]:
                weight = PENALTY_WEIGHT_BY_PLACING[placing]
                for delta in placing_to_deltas_baseline[placing]:
                    if delta < 0.0:
                        weighted_penalty_mass += (-delta) * weight

            if floor_boost > 0.0 and weighted_penalty_mass <= 0.0:
                raise ValueError(
                    "Cannot rebalance floor at places 3/4 because there is no negative loss mass at places 5-8"
                )

            penalty_alpha = 0.0
            if floor_boost > 0.0:
                penalty_alpha = floor_boost / weighted_penalty_mass

            multiplier_by_placing = {}
            for placing in [5, 6, 7, 8]:
                multiplier_by_placing[placing] = 1.0 + (penalty_alpha * PENALTY_WEIGHT_BY_PLACING[placing])

            placing_to_deltas_adjusted = {placing: [] for placing in range(1, 9)}
            baseline_total = 0.0
            adjusted_total = 0.0
            extra_loss_5_to_8 = 0.0
            place_mean_baseline = {}
            place_mean_adjusted = {}
            for placing in range(1, 9):
                for delta in placing_to_deltas_baseline[placing]:
                    adjusted = delta
                    if placing in [3, 4] and delta < 0.0:
                        adjusted = 0.0
                    if placing in [5, 6, 7, 8] and delta < 0.0:
                        adjusted = delta * multiplier_by_placing[placing]
                        extra_loss_5_to_8 += (delta - adjusted)
                    placing_to_deltas_adjusted[placing].append(adjusted)
                    baseline_total += delta
                    adjusted_total += adjusted

                place_values_baseline = placing_to_deltas_baseline[placing]
                place_values_adjusted = placing_to_deltas_adjusted[placing]
                if place_values_baseline:
                    place_mean_baseline[placing] = sum(place_values_baseline) / len(place_values_baseline)
                else:
                    place_mean_baseline[placing] = 0.0
                if place_values_adjusted:
                    place_mean_adjusted[placing] = sum(place_values_adjusted) / len(place_values_adjusted)
                else:
                    place_mean_adjusted[placing] = 0.0

            net_drift = adjusted_total - baseline_total

            holdout_net_drift = 0.0
            holdout_train_count = 0
            holdout_test_count = 0
            holdout_alpha = 0.0
            matched_records.sort(key=lambda row: row["game_ts"])
            split_idx = int(len(matched_records) * 0.7)
            if split_idx > 0 and split_idx < len(matched_records):
                holdout_train_count = split_idx
                holdout_test_count = len(matched_records) - split_idx
                train_records = matched_records[:split_idx]
                test_records = matched_records[split_idx:]

                holdout_floor_boost = 0.0
                holdout_weighted_penalty_mass = 0.0
                for row in train_records:
                    placing = int(row["placing"])
                    delta = float(row["delta"])
                    if placing in [3, 4] and delta < 0.0:
                        holdout_floor_boost += -delta
                    if placing in [5, 6, 7, 8] and delta < 0.0:
                        holdout_weighted_penalty_mass += (-delta) * PENALTY_WEIGHT_BY_PLACING[placing]

                if holdout_floor_boost > 0.0 and holdout_weighted_penalty_mass <= 0.0:
                    raise ValueError(
                        "Cannot compute holdout multiplier: train split has floor gain at 3/4 but no negative loss mass at 5-8"
                    )

                if holdout_floor_boost > 0.0:
                    holdout_alpha = holdout_floor_boost / holdout_weighted_penalty_mass

                holdout_multiplier_by_placing = {}
                for placing in [5, 6, 7, 8]:
                    holdout_multiplier_by_placing[placing] = 1.0 + (holdout_alpha * PENALTY_WEIGHT_BY_PLACING[placing])

                holdout_baseline_total = 0.0
                holdout_adjusted_total = 0.0
                for row in test_records:
                    placing = int(row["placing"])
                    delta = float(row["delta"])
                    adjusted = delta
                    if placing in [3, 4] and delta < 0.0:
                        adjusted = 0.0
                    if placing in [5, 6, 7, 8] and delta < 0.0:
                        adjusted = delta * holdout_multiplier_by_placing[placing]
                    holdout_baseline_total += delta
                    holdout_adjusted_total += adjusted
                holdout_net_drift = holdout_adjusted_total - holdout_baseline_total

            log_console(
                "[INFO] Emerald-or-below formula "
                f"floor_boost={floor_boost:.6f} penalty_alpha={penalty_alpha:.6f} "
                f"m5={multiplier_by_placing[5]:.6f} m6={multiplier_by_placing[6]:.6f} "
                f"m7={multiplier_by_placing[7]:.6f} m8={multiplier_by_placing[8]:.6f} "
                f"net_drift={net_drift:.6f} holdout_net_drift={holdout_net_drift:.6f} "
                f"holdout_train={holdout_train_count} holdout_test={holdout_test_count} holdout_alpha={holdout_alpha:.6f}"
            )

            chart_state["placing_to_deltas_baseline"] = placing_to_deltas_baseline
            chart_state["placing_to_deltas_adjusted"] = placing_to_deltas_adjusted
            chart_state["meta"] = {
                "region": region,
                "season": season,
                "days": days,
                "limit_games": limit_games,
                "loaded_games": int(dataset["meta"]["loaded_games"]),
                "matched_games": matched_games,
                "matched_teams": matched_teams,
                "floored_count": floored_count,
                "floor_boost": floor_boost,
                "extra_loss_5_to_8": extra_loss_5_to_8,
                "penalty_alpha": penalty_alpha,
                "multiplier_by_placing": multiplier_by_placing,
                "place_mean_baseline": place_mean_baseline,
                "place_mean_adjusted": place_mean_adjusted,
                "net_drift": net_drift,
                "holdout_net_drift": holdout_net_drift,
                "holdout_train_count": holdout_train_count,
                "holdout_test_count": holdout_test_count,
                "holdout_alpha": holdout_alpha,
            }
            _draw_chart()
            set_status(
                f"Loaded Emerald-or-below baseline/adjusted boxplots for {region} "
                f"({matched_teams:,} matched teams across {matched_games:,} games)"
            )
        except Exception as exc:
            log_console(f"[ERROR] Emerald-or-below boxplot experiment failed: {exc}")
            log_console(traceback.format_exc().rstrip())
            messagebox.showerror("Emerald-or-below Boxplot Experiment Failed", str(exc))
            set_status("Emerald-or-below boxplot experiment failed")

    regenerate_btn.configure(command=run_query)
    region_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("placing_to_deltas_baseline", None),
            chart_state.__setitem__("placing_to_deltas_adjusted", None),
            chart_state.__setitem__("meta", None),
            _draw_placeholder(),
            summary_var.set("Ready. Press Regenerate to load box-plot data."),
        ),
    )
    season_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("placing_to_deltas_baseline", None),
            chart_state.__setitem__("placing_to_deltas_adjusted", None),
            chart_state.__setitem__("meta", None),
            _draw_placeholder(),
            summary_var.set("Ready. Press Regenerate to load box-plot data."),
        ),
    )
    _draw_placeholder()

    return {
        "frame": frame,
        "run_query": run_query,
        "tab_text": "Experiment - Emerald-or-Below Boxplot",
    }
