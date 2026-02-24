#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from ranking_algorithm import PENALTY_MIN_MULTIPLIER

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


DEFAULT_DAYS = 180
DEFAULT_MIN_GAMES = 1
DEFAULT_MU_BIN_SIZE = 2.5
DEFAULT_GAP_BIN_SIZE = 0.02
DISPLAY_MU_MIN = 15.0
DISPLAY_MU_MAX = 80.0
DISPLAY_GAP_MIN = 0.0
DISPLAY_GAP_MAX = 1.0
POLICY_PREVIEW_PLAYER_MUS = [60.0, 40.0]
GM_CUTOFF_MU_FORMULA_OFFSET = 3.0 * 2.25
# Legacy relative-gap curve constants used for experiment charting.
GAP_TRIGGER = 0.10
GAP_SATURATION = 0.55


def _curve_scale(gap_pct):
    if gap_pct <= GAP_TRIGGER:
        return 1.0
    if gap_pct >= GAP_SATURATION:
        return PENALTY_MIN_MULTIPLIER
    progress = (gap_pct - GAP_TRIGGER) / (GAP_SATURATION - GAP_TRIGGER)
    return 1.0 - (1.0 - PENALTY_MIN_MULTIPLIER) * progress


def _curve_scale_from_zero(gap_pct):
    if gap_pct <= 0.0:
        return 1.0
    if gap_pct >= GAP_SATURATION:
        return PENALTY_MIN_MULTIPLIER
    progress = gap_pct / GAP_SATURATION
    return 1.0 - (1.0 - PENALTY_MIN_MULTIPLIER) * progress


def _gap_current(mu_high, mu_low):
    if mu_high <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (mu_low / mu_high)))


def _gap_gm_cutoff_switch(mu_high, mu_low, gm_cutoff_mu):
    if mu_high <= 0.0 or gm_cutoff_mu <= 0.0:
        return 0.0
    if mu_low < gm_cutoff_mu:
        return max(0.0, min(1.0, 1.0 - (mu_low / gm_cutoff_mu)))
    return max(0.0, min(1.0, 1.0 - (mu_low / mu_high)))


def _build_fixed_bins(start, stop, step):
    values = []
    current = start
    while current <= stop + (step * 0.5):
        values.append(round(current, 6))
        current += step
    return values


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Team Gap Population")

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
    protagonist_only_var = tk.BooleanVar(value=True)
    protagonist_only_toggle = ttk.Checkbutton(controls, text="Protagonist only", variable=protagonist_only_var)
    protagonist_only_toggle.grid(row=0, column=4, padx=(0, 14), pady=6, sticky="w")
    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=5, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Cohort is GM+ / Challenger. If 'Protagonist only' is unticked, query uses all rows with team_gap_pct > 0 "
            "excluding rows where both player_mu and teammate_mu are above gm_cutoff_mu. "
            f"Population cutoff uses GM+ min rating: cutoff_mu = (gm_plus_min_rating / 75) - {GM_CUTOFF_MU_FORMULA_OFFSET:.2f}. "
            f"Curve: trigger={GAP_TRIGGER:.2f}, saturation={GAP_SATURATION:.2f}, min_scale={PENALTY_MIN_MULTIPLIER:.2f}."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="No dataset loaded.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return frame

    content_notebook = ttk.Notebook(frame)
    content_notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    policy_tab = ttk.Frame(content_notebook, style="Panel.TFrame")
    data_tab = ttk.Frame(content_notebook, style="Panel.TFrame")
    content_notebook.add(policy_tab, text="Policy (No Query)")
    content_notebook.add(data_tab, text="Population (Query)")

    policy_figure = Figure(figsize=(13.5, 4.8), dpi=100)
    policy_gs = policy_figure.add_gridspec(1, 1, left=0.06, right=0.98, top=0.90, bottom=0.14)
    ax_policy = policy_figure.add_subplot(policy_gs[0, 0])
    policy_canvas = FigureCanvasTkAgg(policy_figure, master=policy_tab)
    policy_canvas_widget = policy_canvas.get_tk_widget()
    policy_canvas_widget.pack(fill="both", expand=True)

    figure = Figure(figsize=(21.0, 10.2), dpi=100)
    gs = figure.add_gridspec(2, 5, left=0.03, right=0.99, top=0.92, bottom=0.08, wspace=0.28, hspace=0.35)
    ax_gap_top = figure.add_subplot(gs[0, 0])
    ax_mu_teammate_top = figure.add_subplot(gs[0, 1])
    ax_mu_gap_top = figure.add_subplot(gs[0, 2])
    ax_rel_mu_teammate_top = figure.add_subplot(gs[0, 3])
    ax_rel_mu_gap_top = figure.add_subplot(gs[0, 4])
    ax_gap_bottom = figure.add_subplot(gs[1, 0])
    ax_mu_teammate_bottom = figure.add_subplot(gs[1, 1])
    ax_mu_gap_bottom = figure.add_subplot(gs[1, 2])
    ax_rel_mu_teammate_bottom = figure.add_subplot(gs[1, 3])
    ax_rel_mu_gap_bottom = figure.add_subplot(gs[1, 4])

    canvas = FigureCanvasTkAgg(figure, master=data_tab)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)
    chart_state = {
        "colorbar_mu_gap_top": None,
        "colorbar_mu_gap_bottom": None,
        "colorbar_mu_teammate_top": None,
        "colorbar_mu_teammate_bottom": None,
        "colorbar_rel_mu_teammate_top": None,
        "colorbar_rel_mu_teammate_bottom": None,
        "colorbar_rel_mu_gap_top": None,
        "colorbar_rel_mu_gap_bottom": None,
        "current_dataset": None,
        "previous_dataset": None,
        "policy_gm_cutoff_mu": 45.0,
    }

    def _draw_policy_previews_only():
        ax_policy.set_axis_on()
        ax_policy.clear()
        team_gap_values = _build_fixed_bins(0.0, 1.0, 0.005)
        team_gap_scale_values = [_curve_scale(gap_value) for gap_value in team_gap_values]
        gm_cutoff_mu = float(chart_state["policy_gm_cutoff_mu"])
        teammate_mu_values = [gm_cutoff_mu * (1.0 - gap_value) for gap_value in team_gap_values]
        gm_cutoff_gap_values = [max(0.0, min(1.0, 1.0 - (teammate_mu / gm_cutoff_mu))) for teammate_mu in teammate_mu_values]
        gm_cutoff_scale_values = [_curve_scale_from_zero(gap_value) for gap_value in gm_cutoff_gap_values]
        ax_policy.plot(
            team_gap_values,
            team_gap_scale_values,
            linewidth=2.0,
            label=f"Line 1 current curve (player_mu={POLICY_PREVIEW_PLAYER_MUS[0]:.0f})",
        )
        ax_policy.plot(
            team_gap_values,
            team_gap_scale_values,
            linewidth=2.0,
            linestyle="--",
            label=f"Line 2 current curve (player_mu={POLICY_PREVIEW_PLAYER_MUS[1]:.0f})",
        )
        ax_policy.plot(
            gm_cutoff_gap_values,
            gm_cutoff_scale_values,
            linewidth=2.0,
            linestyle="-.",
            label=f"Line 3 gm_cutoff curve (gm_cutoff_mu={gm_cutoff_mu:.2f})",
        )
        ax_policy.axvline(GAP_TRIGGER, color="#2a7f62", linestyle="--", linewidth=1.4, label="trigger")
        ax_policy.axvline(GAP_SATURATION, color="#6b2a7f", linestyle="--", linewidth=1.4, label="saturation")
        ax_policy.set_title("Policy (No Query): Team Gap vs Team Gap Scale")
        ax_policy.set_xlabel("team_gap_pct")
        ax_policy.set_ylabel("team_gap_scale")
        ax_policy.set_xlim(0.0, 1.0)
        ax_policy.set_ylim(0.0, 1.02)
        ax_policy.grid(alpha=0.2)
        ax_policy.legend(loc="best", fontsize=7)
        policy_canvas.draw_idle()

    def _heatmap_matrix(rows, x_axis, y_axis, x_key, y_key):
        if not x_axis or not y_axis:
            return [[0.0]]
        x_index = {value: idx for idx, value in enumerate(x_axis)}
        y_index = {value: idx for idx, value in enumerate(y_axis)}
        raw = [[0.0 for _ in x_axis] for _ in y_axis]
        for row in rows:
            yi = y_index.get(row[y_key])
            xi = x_index.get(row[x_key])
            if yi is None or xi is None:
                continue
            raw[yi][xi] = float(row["game_count"])
        normalized = []
        for row in raw:
            total = sum(row)
            if total <= 0.0:
                normalized.append([0.0 for _ in row])
            else:
                normalized.append([value / total for value in row])
        return normalized

    def _ticks_for_axis(axis_values):
        if not axis_values:
            return [0]
        step = max(1, len(axis_values) // 8)
        return list(range(0, len(axis_values), step))

    def _draw_row(
        dataset,
        ax_gap,
        ax_mu_teammate,
        ax_mu_gap,
        ax_rel_mu_teammate,
        ax_rel_mu_gap,
        gap_x_max,
        mu_bins_axis,
        gap_bins_axis,
        rel_player_mu_axis,
        rel_teammate_mu_axis,
        rel_gap_axis,
        colorbar_mu_teammate_key,
        colorbar_mu_gap_key,
        colorbar_rel_mu_teammate_key,
        colorbar_rel_mu_gap_key,
    ):
        ax_gap.set_axis_on()
        ax_mu_teammate.set_axis_on()
        ax_mu_gap.set_axis_on()
        ax_rel_mu_teammate.set_axis_on()
        ax_rel_mu_gap.set_axis_on()
        ax_gap.clear()
        ax_mu_teammate.clear()
        ax_mu_gap.clear()
        ax_rel_mu_teammate.clear()
        ax_rel_mu_gap.clear()

        gap_distribution = dataset["gap_distribution"]
        x_gap = [row["gap_bin_start"] for row in gap_distribution]
        y_count = [float(row["game_count"]) for row in gap_distribution]
        max_count = max(y_count) if y_count else 1.0
        y_hist_norm = [value / max_count for value in y_count]
        y_scale_mid = [_curve_scale(x) for x in x_gap]
        ax_gap.bar(
            x_gap,
            y_hist_norm,
            width=dataset["meta"]["gap_bin_size"] * 0.9,
            color="#8cb9df",
            alpha=0.45,
            label="gap histogram (normalized)",
        )
        ax_gap.plot(x_gap, y_scale_mid, color="#c22e2e", linewidth=2.0, label="current curve")
        ax_gap.axvline(GAP_TRIGGER, color="#2a7f62", linestyle="--", linewidth=1.4, label="trigger")
        ax_gap.axvline(GAP_SATURATION, color="#6b2a7f", linestyle="--", linewidth=1.4, label="saturation")
        ax_gap.set_title("Gap Histogram + Curve")
        ax_gap.set_xlabel("team_gap_pct")
        ax_gap.set_ylabel("normalized")
        ax_gap.set_xlim(0.0, gap_x_max)
        ax_gap.set_ylim(0.0, 1.02)
        ax_gap.grid(alpha=0.2)
        ax_gap.legend(loc="best", fontsize=7)

        mu_teammate_grid_abs = _heatmap_matrix(
            dataset["player_vs_teammate_bins"],
            mu_bins_axis,
            mu_bins_axis,
            x_key="teammate_mu_bin_start",
            y_key="player_mu_bin_start",
        )
        mu_teammate_abs_image = ax_mu_teammate.imshow(
            mu_teammate_grid_abs, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis"
        )
        ax_mu_teammate.set_title("Player Mu vs Teammate Mu (Row-Normalized)")
        ax_mu_teammate.set_xlabel("teammate_mu_bin")
        ax_mu_teammate.set_ylabel("player_mu_bin")
        mu_ticks = _ticks_for_axis(mu_bins_axis)
        ax_mu_teammate.set_xticks(mu_ticks)
        ax_mu_teammate.set_xticklabels([f"{mu_bins_axis[idx]:.1f}" for idx in mu_ticks], rotation=45, ha="right")
        ax_mu_teammate.set_yticks(mu_ticks)
        ax_mu_teammate.set_yticklabels([f"{mu_bins_axis[idx]:.1f}" for idx in mu_ticks])
        if chart_state[colorbar_mu_teammate_key] is not None:
            chart_state[colorbar_mu_teammate_key].remove()
        chart_state[colorbar_mu_teammate_key] = figure.colorbar(
            mu_teammate_abs_image, ax=ax_mu_teammate, fraction=0.046, pad=0.04, label="row proportion"
        )

        mu_gap_grid_abs = _heatmap_matrix(
            dataset["player_vs_gap_bins"],
            gap_bins_axis,
            mu_bins_axis,
            x_key="team_gap_bin_start",
            y_key="player_mu_bin_start",
        )
        mu_gap_abs_image = ax_mu_gap.imshow(mu_gap_grid_abs, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax_mu_gap.set_title("Player Mu vs Team Gap Density (Row-Normalized)")
        ax_mu_gap.set_xlabel("team_gap_bin")
        ax_mu_gap.set_ylabel("player_mu_bin")
        gap_ticks = _ticks_for_axis(gap_bins_axis)
        ax_mu_gap.set_xticks(gap_ticks)
        ax_mu_gap.set_xticklabels([f"{gap_bins_axis[idx]:.2f}" for idx in gap_ticks], rotation=45, ha="right")
        ax_mu_gap.set_yticks(mu_ticks)
        ax_mu_gap.set_yticklabels([f"{mu_bins_axis[idx]:.1f}" for idx in mu_ticks])
        if chart_state[colorbar_mu_gap_key] is not None:
            chart_state[colorbar_mu_gap_key].remove()
        chart_state[colorbar_mu_gap_key] = figure.colorbar(
            mu_gap_abs_image, ax=ax_mu_gap, fraction=0.046, pad=0.04, label="row proportion"
        )

        rel_teammate_grid = _heatmap_matrix(
            dataset["player_vs_teammate_rel_gm_bins"],
            rel_teammate_mu_axis,
            rel_player_mu_axis,
            x_key="gm_cutoff_minus_teammate_mu_bin_start",
            y_key="player_mu_above_gm_cutoff_bin_start",
        )
        rel_teammate_image = ax_rel_mu_teammate.imshow(rel_teammate_grid, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax_rel_mu_teammate.set_title("Mu Above GM Cutoff vs (GM Cutoff - Teammate Mu) (Row-Normalized)")
        ax_rel_mu_teammate.set_xlabel("gm_cutoff_mu - teammate_mu")
        ax_rel_mu_teammate.set_ylabel("player_mu - gm_cutoff_mu")
        rel_teammate_ticks = _ticks_for_axis(rel_teammate_mu_axis)
        rel_player_ticks = _ticks_for_axis(rel_player_mu_axis)
        ax_rel_mu_teammate.set_xticks(rel_teammate_ticks)
        ax_rel_mu_teammate.set_xticklabels([f"{rel_teammate_mu_axis[idx]:.1f}" for idx in rel_teammate_ticks], rotation=45, ha="right")
        ax_rel_mu_teammate.set_yticks(rel_player_ticks)
        ax_rel_mu_teammate.set_yticklabels([f"{rel_player_mu_axis[idx]:.1f}" for idx in rel_player_ticks])
        if chart_state[colorbar_rel_mu_teammate_key] is not None:
            chart_state[colorbar_rel_mu_teammate_key].remove()
        chart_state[colorbar_rel_mu_teammate_key] = figure.colorbar(
            rel_teammate_image, ax=ax_rel_mu_teammate, fraction=0.046, pad=0.04, label="row proportion"
        )

        rel_gap_grid = _heatmap_matrix(
            dataset["player_vs_gap_rel_gm_bins"],
            rel_gap_axis,
            rel_player_mu_axis,
            x_key="team_gap_using_gm_cutoff_bin_start",
            y_key="player_mu_above_gm_cutoff_bin_start",
        )
        rel_gap_image = ax_rel_mu_gap.imshow(rel_gap_grid, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax_rel_mu_gap.set_title("Mu Above GM Cutoff vs Team Gap Using GM Cutoff (Row-Normalized)")
        ax_rel_mu_gap.set_xlabel("1 - (teammate_mu / gm_cutoff_mu)")
        ax_rel_mu_gap.set_ylabel("player_mu - gm_cutoff_mu")
        rel_gap_ticks = _ticks_for_axis(rel_gap_axis)
        ax_rel_mu_gap.set_xticks(rel_gap_ticks)
        ax_rel_mu_gap.set_xticklabels([f"{rel_gap_axis[idx]:.2f}" for idx in rel_gap_ticks], rotation=45, ha="right")
        ax_rel_mu_gap.set_yticks(rel_player_ticks)
        ax_rel_mu_gap.set_yticklabels([f"{rel_player_mu_axis[idx]:.1f}" for idx in rel_player_ticks])
        if chart_state[colorbar_rel_mu_gap_key] is not None:
            chart_state[colorbar_rel_mu_gap_key].remove()
        chart_state[colorbar_rel_mu_gap_key] = figure.colorbar(
            rel_gap_image, ax=ax_rel_mu_gap, fraction=0.046, pad=0.04, label="row proportion"
        )

    def _draw_all():
        current_dataset = chart_state["current_dataset"]
        previous_dataset = chart_state["previous_dataset"]
        if current_dataset is None:
            return

        datasets = [current_dataset]
        if previous_dataset is not None:
            datasets.append(previous_dataset)

        gap_x_max = 1.0
        mu_bins_axis = _build_fixed_bins(DISPLAY_MU_MIN, DISPLAY_MU_MAX, DEFAULT_MU_BIN_SIZE)
        gap_bins_axis = _build_fixed_bins(DISPLAY_GAP_MIN, DISPLAY_GAP_MAX, DEFAULT_GAP_BIN_SIZE)
        rel_player_mu_axis = sorted(
            {
                float(row["player_mu_above_gm_cutoff_bin_start"])
                for ds in datasets
                for row in ds["player_vs_teammate_rel_gm_bins"]
            }
            | {
                float(row["player_mu_above_gm_cutoff_bin_start"])
                for ds in datasets
                for row in ds["player_vs_gap_rel_gm_bins"]
            }
        )
        rel_teammate_mu_axis = sorted(
            {
                float(row["gm_cutoff_minus_teammate_mu_bin_start"])
                for ds in datasets
                for row in ds["player_vs_teammate_rel_gm_bins"]
            }
        )
        rel_gap_axis = sorted(
            {
                float(row["team_gap_using_gm_cutoff_bin_start"])
                for ds in datasets
                for row in ds["player_vs_gap_rel_gm_bins"]
            }
        )
        if not rel_player_mu_axis:
            rel_player_mu_axis = [0.0]
        if not rel_teammate_mu_axis:
            rel_teammate_mu_axis = [0.0]
        if not rel_gap_axis:
            rel_gap_axis = [0.0]

        for ds in datasets:
            if ds["gap_distribution"]:
                gap_x_max = max(gap_x_max, max(row["gap_bin_start"] for row in ds["gap_distribution"]) + ds["meta"]["gap_bin_size"])

        _draw_row(
            current_dataset,
            ax_gap_top,
            ax_mu_teammate_top,
            ax_mu_gap_top,
            ax_rel_mu_teammate_top,
            ax_rel_mu_gap_top,
            gap_x_max,
            mu_bins_axis,
            gap_bins_axis,
            rel_player_mu_axis,
            rel_teammate_mu_axis,
            rel_gap_axis,
            "colorbar_mu_teammate_top",
            "colorbar_mu_gap_top",
            "colorbar_rel_mu_teammate_top",
            "colorbar_rel_mu_gap_top",
        )

        if previous_dataset is not None:
            _draw_row(
                previous_dataset,
                ax_gap_bottom,
                ax_mu_teammate_bottom,
                ax_mu_gap_bottom,
                ax_rel_mu_teammate_bottom,
                ax_rel_mu_gap_bottom,
                gap_x_max,
                mu_bins_axis,
                gap_bins_axis,
                rel_player_mu_axis,
                rel_teammate_mu_axis,
                rel_gap_axis,
                "colorbar_mu_teammate_bottom",
                "colorbar_mu_gap_bottom",
                "colorbar_rel_mu_teammate_bottom",
                "colorbar_rel_mu_gap_bottom",
            )
        else:
            ax_gap_bottom.clear()
            ax_mu_teammate_bottom.clear()
            ax_mu_gap_bottom.clear()
            ax_rel_mu_teammate_bottom.clear()
            ax_rel_mu_gap_bottom.clear()
            ax_gap_bottom.text(0.5, 0.5, "Previous chart will appear after next regenerate.", ha="center", va="center")
            ax_gap_bottom.set_axis_off()
            ax_mu_teammate_bottom.set_axis_off()
            ax_mu_gap_bottom.set_axis_off()
            ax_rel_mu_teammate_bottom.set_axis_off()
            ax_rel_mu_gap_bottom.set_axis_off()
            for colorbar_key in [
                "colorbar_mu_teammate_bottom",
                "colorbar_mu_gap_bottom",
                "colorbar_rel_mu_teammate_bottom",
                "colorbar_rel_mu_gap_bottom",
            ]:
                if chart_state[colorbar_key] is not None:
                    chart_state[colorbar_key].remove()
                    chart_state[colorbar_key] = None

        if current_dataset["meta"]["require_protagonist"]:
            current_scope = "protagonist_only"
        elif current_dataset["meta"]["require_positive_gap"]:
            current_scope = "all_players_team_gap_gt_0_excl_both_above_cutoff"
        else:
            current_scope = "all_players"
        title = (
            f"GM+ protagonist team-gap profile | current: {current_dataset['meta']['region']} / {current_dataset['meta']['season']} "
            f"| scope={current_scope} "
            f"| rows={current_dataset['summary']['sample_count']:,} | "
            f"gm+_min_rating={current_dataset['summary']['gm_cutoff_rating']:.2f} | "
            f"cutoff_mu=(rating/75 - {GM_CUTOFF_MU_FORMULA_OFFSET:.2f})={current_dataset['summary']['gm_cutoff_mu']:.2f}"
        )
        figure.suptitle(title, fontsize=11)
        canvas.draw_idle()

    def _refresh_policy_only():
        _draw_policy_previews_only()

    def run_query():
        try:
            region = region_var.get().strip().lower()
            if region not in region_to_ch_prefix:
                raise ValueError(f"Unknown region: {region}")
            season = season_var.get().strip().lower()
            protagonist_only = bool(protagonist_only_var.get())
            require_positive_gap = not protagonist_only
            _refresh_policy_only()
            if content_notebook.index(content_notebook.select()) == 0:
                set_status("Policy tab active: skipped database query")
                return

            set_status(f"Loading team-gap charts for {region}...")
            log_console(
                "[INFO] Team-gap population query "
                f"region={region} season={season} days={DEFAULT_DAYS} min_games={DEFAULT_MIN_GAMES} "
                f"mu_bin={DEFAULT_MU_BIN_SIZE} gap_bin={DEFAULT_GAP_BIN_SIZE} protagonist_only={protagonist_only} "
                f"require_positive_gap={require_positive_gap}"
            )
            dataset = private_ch.load_team_gap_population_dataset(
                region=region,
                ch_prefix=region_to_ch_prefix[region],
                season=season,
                days=DEFAULT_DAYS,
                min_games=DEFAULT_MIN_GAMES,
                mu_bin_size=DEFAULT_MU_BIN_SIZE,
                gap_bin_size=DEFAULT_GAP_BIN_SIZE,
                rank_filter=["Grand Master", "Challenger"],
                require_protagonist=protagonist_only,
                require_positive_gap=require_positive_gap,
            )
            previous_dataset = chart_state["current_dataset"]
            chart_state["previous_dataset"] = previous_dataset
            chart_state["current_dataset"] = dataset
            summary = dataset["summary"]
            chart_state["policy_gm_cutoff_mu"] = float(summary["gm_cutoff_mu"])
            _draw_policy_previews_only()
            log_console(
                "[INFO] GM+ cutoff from player_ranks "
                f"min_rating={summary['gm_cutoff_rating']:.2f} "
                f"-> cutoff_mu=(rating/75 - {GM_CUTOFF_MU_FORMULA_OFFSET:.2f})={summary['gm_cutoff_mu']:.2f}"
            )
            scope_label = "protagonist_only" if protagonist_only else "all_players_team_gap_gt_0_excl_both_above_cutoff"
            summary_var.set(
                f"scope={scope_label} | rows={summary['sample_count']:,} | gap_p50={summary['gap_p50']:.3f} | "
                f"gap_p75={summary['gap_p75']:.3f} | gap_p90={summary['gap_p90']:.3f} | gap_p95={summary['gap_p95']:.3f} | "
                f"gm+_min_rating={summary['gm_cutoff_rating']:.2f} | "
                f"cutoff_mu=(rating/75 - {GM_CUTOFF_MU_FORMULA_OFFSET:.2f})={summary['gm_cutoff_mu']:.2f} | "
                f"mu_above_gm_p50={summary['player_mu_above_gm_cutoff_p50']:.2f} | "
                f"gm_cutoff_minus_teammate_p50={summary['gm_cutoff_minus_teammate_mu_p50']:.2f} | "
                f"gap_using_gm_cutoff_p50={summary['team_gap_using_gm_cutoff_p50']:.2f} | "
                f"player_mu_range=[{summary['min_player_mu']:.1f}, {summary['max_player_mu']:.1f}] | "
                f"player_mu>=60={summary['share_player_mu_ge_60'] * 100.0:.2f}% | "
                f"player_mu>=70={summary['share_player_mu_ge_70'] * 100.0:.2f}% | "
                f"<trigger={summary['share_below_trigger'] * 100.0:.1f}% | linear={summary['share_in_linear_band'] * 100.0:.1f}% | "
                f">=saturation={summary['share_at_or_above_saturation'] * 100.0:.1f}%"
            )
            _draw_all()
            set_status(f"Loaded team-gap charts for {region} ({summary['sample_count']:,} rows)")
        except Exception as exc:
            messagebox.showerror("Team Gap Experiment Failed", str(exc))
            set_status("Team gap experiment failed")

    def refresh_policy_only():
        try:
            _refresh_policy_only()
            set_status("Updated policy preview charts")
        except Exception as exc:
            messagebox.showerror("Policy Preview Failed", str(exc))
            set_status("Policy preview failed")

    regenerate_btn.configure(command=run_query)
    region_combo.bind("<<ComboboxSelected>>", lambda _event: run_query())
    season_combo.bind("<<ComboboxSelected>>", lambda _event: run_query())
    protagonist_only_toggle.configure(command=run_query)
    _draw_policy_previews_only()
    return {
        "frame": frame,
        "run_query": run_query,
        "tab_text": "Experiment - Team Gap Population",
    }
