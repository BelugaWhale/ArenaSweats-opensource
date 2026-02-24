#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


DEFAULT_DAYS = 180
DEFAULT_MIN_GAMES = 1
DEFAULT_PERCENTILE_STEP = 1.0
DEFAULT_PERCENTILE_BIN_SIZE = 1.0
DEFAULT_DIFF_BIN_SIZE = 1.0
DEFAULT_MU_BUCKET_SIZE = 0.25
DEFAULT_ROW2_Y_AXIS_MIN_PLAYER_RELATIVE_PCT = 60.0

def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Team Gap Percentiles")

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

    generate_btn = ttk.Button(controls, text="Generate")
    generate_btn.grid(row=0, column=2, padx=(0, 8), pady=6, sticky="w")

    require_protagonist_var = tk.BooleanVar(value=True)
    protagonist_only_checkbox = ttk.Checkbutton(
        controls,
        text="Protagonist Only",
        variable=require_protagonist_var,
    )
    protagonist_only_checkbox.grid(row=0, column=3, padx=(8, 0), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Builds regional mu distribution from latest player mu, then maps GM+ "
            "team_gap>0 rows into linear relative scale space. "
            "Scale uses min(mu)->0% and max(mu)->100% with binned heatmaps."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    progress_wrap = ttk.Frame(frame, style="Panel.TFrame")
    progress_wrap.pack(fill="x", padx=10, pady=(0, 6))
    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = ttk.Progressbar(
        progress_wrap,
        mode="determinate",
        maximum=100.0,
        variable=progress_var,
        style="Loading.Horizontal.TProgressbar",
    )
    progress_bar.pack(fill="x", expand=True, side="left")
    progress_text_var = tk.StringVar(value="Idle")
    progress_text = ttk.Label(progress_wrap, textvariable=progress_text_var, style="Sub.TLabel")
    progress_text.pack(side="left", padx=(10, 0))

    summary_var = tk.StringVar(value="No dataset loaded.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return frame

    figure = Figure(figsize=(19.2, 10.8), dpi=100)
    grid = figure.add_gridspec(2, 6)
    ax_distribution = figure.add_subplot(grid[0, 0:2])
    ax_distribution_secondary = ax_distribution.twinx()
    ax_player_vs_diff = figure.add_subplot(grid[0, 2:4])
    ax_player_vs_teammate = figure.add_subplot(grid[0, 4:6])
    ax_scatter_current = figure.add_subplot(grid[1, 0:2])
    ax_scatter_new = figure.add_subplot(grid[1, 2:4])
    ax_teammate_mu_range = figure.add_subplot(grid[1, 4:6])
    figure.subplots_adjust(top=0.92, hspace=0.35, wspace=0.38)

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    chart_state = {
        "dataset": None,
        "protagonist_dataset": None,
        "colorbar_diff": None,
        "colorbar_teammate": None,
    }

    def _heatmap_matrix(rows, x_axis, y_axis, x_key, y_key):
        if not x_axis or not y_axis:
            return [[0.0]]
        x_index = {value: idx for idx, value in enumerate(x_axis)}
        y_index = {value: idx for idx, value in enumerate(y_axis)}
        raw = [[0.0 for _ in x_axis] for _ in y_axis]
        for row in rows:
            y_idx = y_index.get(float(row[y_key]))
            x_idx = x_index.get(float(row[x_key]))
            if y_idx is None or x_idx is None:
                continue
            raw[y_idx][x_idx] = float(row["game_count"])
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

    def _axis_from_rows(rows, key):
        values = sorted({float(row[key]) for row in rows})
        if values:
            return values
        return [0.0]

    def _draw_all():
        dataset = chart_state["dataset"]
        if dataset is None:
            return

        ax_distribution.clear()
        ax_distribution_secondary.clear()
        ax_player_vs_diff.clear()
        ax_player_vs_teammate.clear()
        ax_scatter_current.clear()
        ax_scatter_new.clear()
        ax_teammate_mu_range.clear()

        mu_distribution = dataset.get("mu_distribution_bins") or []
        checkpoint_rows = dataset.get("scale_line_points") or []
        mu_x = [row["mu_bucket_start"] + (dataset["meta"]["mu_bucket_size"] / 2.0) for row in mu_distribution]
        mu_y_share = [float(row["player_count"]) / float(dataset["summary"]["player_population_count"]) for row in mu_distribution]
        ax_distribution.bar(
            mu_x,
            mu_y_share,
            width=dataset["meta"]["mu_bucket_size"] * 0.9,
            color="#8cb9df",
            alpha=0.5,
            label="mu distribution",
        )
        checkpoint_mu = [float(row["mu"]) for row in checkpoint_rows]
        checkpoint_pct = [float(row["relative_pct"]) for row in checkpoint_rows]
        ax_distribution_secondary.plot(
            checkpoint_mu,
            checkpoint_pct,
            color="#c22e2e",
            linewidth=2.0,
            label="mu->relative scale",
        )
        ax_distribution.set_title("Mu Distribution + Relative Scale Line")
        ax_distribution.set_xlabel("mu")
        ax_distribution.set_ylabel("population share")
        ax_distribution_secondary.set_ylabel("relative scale (%)")
        ax_distribution.grid(alpha=0.2)
        ax_distribution_secondary.set_ylim(0.0, 100.0)
        distribution_handles, distribution_labels = ax_distribution.get_legend_handles_labels()
        curve_handles, curve_labels = ax_distribution_secondary.get_legend_handles_labels()
        ax_distribution.legend(distribution_handles + curve_handles, distribution_labels + curve_labels, loc="best", fontsize=8)

        diff_rows = dataset.get("player_vs_team_percentile_diff_bins") or []
        teammate_rows = dataset.get("player_vs_teammate_percentile_bins") or []
        relative_points = dataset.get("relative_points") or []
        if not diff_rows and not teammate_rows and not relative_points:
            raise ValueError("No binned rows returned")
        if not diff_rows:
            raise ValueError("No player_vs_team_percentile_diff bins available")
        gap_axis = _axis_from_rows(diff_rows, "team_percentile_diff_bin_start")
        player_axis_for_diff = _axis_from_rows(diff_rows, "player_percentile_bin_start")
        diff_matrix = _heatmap_matrix(
            diff_rows,
            gap_axis,
            player_axis_for_diff,
            x_key="team_percentile_diff_bin_start",
            y_key="player_percentile_bin_start",
        )
        diff_img = ax_player_vs_diff.imshow(
            diff_matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis"
        )
        ax_player_vs_diff.set_title("Player Relative Scale vs Team Relative Gap (Row-Normalized)")
        ax_player_vs_diff.set_xlabel("team_relative_gap_pct (max(player - teammate, 0))")
        ax_player_vs_diff.set_ylabel("player_relative_pct")
        gap_ticks = _ticks_for_axis(gap_axis)
        player_diff_ticks = _ticks_for_axis(player_axis_for_diff)
        ax_player_vs_diff.set_xticks(gap_ticks)
        ax_player_vs_diff.set_xticklabels([f"{gap_axis[idx]:g}" for idx in gap_ticks], rotation=45, ha="right")
        ax_player_vs_diff.set_yticks(player_diff_ticks)
        ax_player_vs_diff.set_yticklabels([f"{player_axis_for_diff[idx]:g}" for idx in player_diff_ticks])
        if chart_state["colorbar_diff"] is not None:
            chart_state["colorbar_diff"].remove()
        chart_state["colorbar_diff"] = figure.colorbar(
            diff_img, ax=ax_player_vs_diff, fraction=0.046, pad=0.04, label="row proportion"
        )

        if not teammate_rows:
            raise ValueError("No player_vs_teammate_percentile bins available")
        player_axis = _axis_from_rows(teammate_rows, "player_percentile_bin_start")
        teammate_axis = _axis_from_rows(teammate_rows, "teammate_percentile_bin_start")
        teammate_matrix = _heatmap_matrix(
            teammate_rows,
            teammate_axis,
            player_axis,
            x_key="teammate_percentile_bin_start",
            y_key="player_percentile_bin_start",
        )
        teammate_img = ax_player_vs_teammate.imshow(
            teammate_matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis"
        )
        ax_player_vs_teammate.set_title("Player Relative Scale vs Teammate Relative Scale (Row-Normalized)")
        ax_player_vs_teammate.set_xlabel("teammate_relative_pct")
        ax_player_vs_teammate.set_ylabel("player_relative_pct")
        teammate_ticks = _ticks_for_axis(teammate_axis)
        player_ticks = _ticks_for_axis(player_axis)
        ax_player_vs_teammate.set_xticks(teammate_ticks)
        ax_player_vs_teammate.set_xticklabels([f"{teammate_axis[idx]:g}" for idx in teammate_ticks], rotation=45, ha="right")
        ax_player_vs_teammate.set_yticks(player_ticks)
        ax_player_vs_teammate.set_yticklabels([f"{player_axis[idx]:g}" for idx in player_ticks])
        if chart_state["colorbar_teammate"] is not None:
            chart_state["colorbar_teammate"].remove()
        chart_state["colorbar_teammate"] = figure.colorbar(
            teammate_img, ax=ax_player_vs_teammate, fraction=0.046, pad=0.04, label="row proportion"
        )

        if not relative_points:
            raise ValueError("No relative_points rows available")
        current_gap_points = []
        player_current_points = []
        new_gap_points = []
        player_new_points = []
        for row in relative_points:
            game_count = int(row["game_count"])
            if game_count <= 0:
                continue
            current_gap = float(row["current_system_gap_pct"])
            player_relative = float(row["player_relative_pct"])
            new_gap = float(row["team_relative_gap_pct"])
            current_gap_points.extend([current_gap] * game_count)
            player_current_points.extend([player_relative] * game_count)
            new_gap_points.extend([new_gap] * game_count)
            player_new_points.extend([player_relative] * game_count)
        if not current_gap_points:
            raise ValueError("No scatter points available after expansion")

        ax_scatter_current.scatter(
            current_gap_points,
            player_current_points,
            s=5,
            alpha=0.18,
            c="#5db2e8",
            edgecolors="none",
            rasterized=True,
        )
        ax_scatter_current.set_title("All Points: Player Relative vs Current Team Gap System")
        ax_scatter_current.set_xlabel("current_team_gap_pct (1 - teammate_mu/player_mu)")
        ax_scatter_current.set_ylabel("player_relative_pct")
        ax_scatter_current.set_ylim(DEFAULT_ROW2_Y_AXIS_MIN_PLAYER_RELATIVE_PCT, 100.0)
        ax_scatter_current.set_xlim(left=0.0)
        ax_scatter_current.grid(alpha=0.2)

        ax_scatter_new.scatter(
            new_gap_points,
            player_new_points,
            s=5,
            alpha=0.18,
            c="#f2903d",
            edgecolors="none",
            rasterized=True,
        )
        ax_scatter_new.set_title("All Points: Player Relative vs New Relative Gap")
        ax_scatter_new.set_xlabel("new_gap_pct (player_relative_pct - teammate_relative_pct)")
        ax_scatter_new.set_ylabel("player_relative_pct")
        ax_scatter_new.set_ylim(DEFAULT_ROW2_Y_AXIS_MIN_PLAYER_RELATIVE_PCT, 100.0)
        ax_scatter_new.set_xlim(left=0.0)
        ax_scatter_new.grid(alpha=0.2)

        protagonist_dataset = chart_state.get("protagonist_dataset") or dataset
        protagonist_relative_points = protagonist_dataset.get("relative_points") or []
        teammate_mu_values = []
        teammate_mu_weights = []
        for row in protagonist_relative_points:
            game_count = int(row["game_count"])
            if game_count <= 0:
                continue
            if float(row["player_mu"]) < 50.0:
                continue
            teammate_mu_values.append(float(row["teammate_mu"]))
            teammate_mu_weights.append(float(game_count))
        if not teammate_mu_values:
            raise ValueError("No rows available for teammate mu range chart with player_mu >= 50")

        min_teammate_mu = min(teammate_mu_values)
        max_teammate_mu = max(teammate_mu_values)
        bin_count = min(60, max(12, int((max_teammate_mu - min_teammate_mu) / 0.4)))
        ax_teammate_mu_range.hist(
            teammate_mu_values,
            bins=bin_count,
            weights=teammate_mu_weights,
            color="#7dcf8a",
            alpha=0.78,
            edgecolor="#2f5f39",
            linewidth=0.35,
        )

        sorted_pairs = sorted(zip(teammate_mu_values, teammate_mu_weights), key=lambda item: item[0])
        total_weight = sum(teammate_mu_weights)
        if total_weight <= 0.0:
            raise ValueError("Invalid total weight for teammate mu range chart")
        quantile_targets = [0.10, 0.25, 0.50, 0.75, 0.90]
        quantile_values = {}
        running_weight = 0.0
        quantile_index = 0
        for mu_value, mu_weight in sorted_pairs:
            running_weight += mu_weight
            while quantile_index < len(quantile_targets) and (running_weight / total_weight) >= quantile_targets[quantile_index]:
                quantile_values[quantile_targets[quantile_index]] = mu_value
                quantile_index += 1
        while quantile_index < len(quantile_targets):
            quantile_values[quantile_targets[quantile_index]] = sorted_pairs[-1][0]
            quantile_index += 1

        ax_teammate_mu_range.axvline(min_teammate_mu, color="#1f2f1f", linestyle="--", linewidth=1.1, label="min/max")
        ax_teammate_mu_range.axvline(max_teammate_mu, color="#1f2f1f", linestyle="--", linewidth=1.1)
        ax_teammate_mu_range.axvline(quantile_values[0.50], color="#ce2f2f", linewidth=1.4, label="p50")
        ax_teammate_mu_range.axvline(quantile_values[0.25], color="#6a7076", linestyle=":", linewidth=1.2, label="p25/p75")
        ax_teammate_mu_range.axvline(quantile_values[0.75], color="#6a7076", linestyle=":", linewidth=1.2)
        ax_teammate_mu_range.axvline(quantile_values[0.10], color="#7f8790", linestyle="-.", linewidth=1.0, label="p10/p90")
        ax_teammate_mu_range.axvline(quantile_values[0.90], color="#7f8790", linestyle="-.", linewidth=1.0)
        ax_teammate_mu_range.set_title("Teammate Mu Range | GM+ Protagonists with Player Mu >= 50")
        ax_teammate_mu_range.set_xlabel("teammate_mu")
        ax_teammate_mu_range.set_ylabel("weighted game count")
        ax_teammate_mu_range.grid(alpha=0.2)
        ax_teammate_mu_range.legend(loc="best", fontsize=7)

        stats_text = (
            f"min={min_teammate_mu:.2f}  p10={quantile_values[0.10]:.2f}  p25={quantile_values[0.25]:.2f}\n"
            f"p50={quantile_values[0.50]:.2f}  p75={quantile_values[0.75]:.2f}  p90={quantile_values[0.90]:.2f}  max={max_teammate_mu:.2f}"
        )
        ax_teammate_mu_range.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_teammate_mu_range.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f5fff1", "alpha": 0.75, "edgecolor": "#9fb89a"},
        )

        summary = dataset["summary"]
        summary_var.set(
            f"rows={summary['source_pair_rows']:,} | mu_pairs={summary['pair_bucket_rows']:,} | "
            f"population_players={summary['player_population_count']:,} | "
            f"avg_player_rel={summary['avg_player_relative_pct']:.2f} | "
            f"avg_teammate_rel={summary['avg_teammate_relative_pct']:.2f} | "
            f"avg_gap_rel={summary['avg_team_relative_gap_pct']:.2f} | "
            f"player_rel_range=[{summary['min_player_relative_pct']:.2f}, {summary['max_player_relative_pct']:.2f}] | "
            f"scale_mu=[{summary['scale_floor_mu']:.2f}, {summary['scale_ceiling_mu']:.2f}] | "
            f"population_mu_range=[{summary['min_population_mu']:.2f}, {summary['max_population_mu']:.2f}]"
        )
        figure.suptitle(
            f"Team Gap Percentiles | region={dataset['meta']['region']} season={dataset['meta']['season']} "
            f"| days={dataset['meta']['days']} min_games={dataset['meta']['min_games']} "
            f"| protagonist={dataset['meta']['require_protagonist']} "
            "| linear relative scale from min(mu) to max(mu)",
            fontsize=11,
        )
        canvas.draw_idle()

    def _set_progress(percent, text):
        progress_var.set(float(percent))
        progress_text_var.set(text)
        frame.update_idletasks()

    def run_query():
        try:
            region = region_var.get().strip().lower()
            require_protagonist = bool(require_protagonist_var.get())
            if region not in region_to_ch_prefix:
                raise ValueError(f"Unknown region: {region}")
            _set_progress(0.0, "Starting...")
            set_status(f"Loading team gap relative-scale charts for {region}...")

            def _progress_callback(step, total, text):
                if total <= 0:
                    percent = 0.0
                else:
                    percent = (float(step) / float(total)) * 100.0
                _set_progress(percent, text)
                set_status(text)

            log_console(
                "[INFO] Team gap relative-scale query "
                f"region={region} days={DEFAULT_DAYS} min_games={DEFAULT_MIN_GAMES} "
                f"percentile_step={DEFAULT_PERCENTILE_STEP} percentile_bin={DEFAULT_PERCENTILE_BIN_SIZE} "
                f"diff_bin={DEFAULT_DIFF_BIN_SIZE} mu_bucket={DEFAULT_MU_BUCKET_SIZE} "
                f"scope=gm+ team_gap>0 require_protagonist={require_protagonist}"
            )
            dataset = private_ch.load_team_gap_percentiles_dataset(
                region=region,
                ch_prefix=region_to_ch_prefix[region],
                season="live",
                days=DEFAULT_DAYS,
                min_games=DEFAULT_MIN_GAMES,
                rank_filter=["Grand Master", "Challenger"],
                require_protagonist=require_protagonist,
                require_positive_gap=True,
                percentile_step=DEFAULT_PERCENTILE_STEP,
                percentile_bin_size=DEFAULT_PERCENTILE_BIN_SIZE,
                diff_bin_size=DEFAULT_DIFF_BIN_SIZE,
                mu_bucket_size=DEFAULT_MU_BUCKET_SIZE,
                progress_callback=_progress_callback,
            )
            protagonist_dataset = dataset
            if not require_protagonist:
                _set_progress(0.0, "Loading protagonist-only subset for teammate mu range...")
                set_status(f"Loading teammate mu range subset for {region}...")
                log_console(
                    "[INFO] Team gap relative-scale secondary query for teammate mu range "
                    f"region={region} days={DEFAULT_DAYS} min_games={DEFAULT_MIN_GAMES} "
                    "scope=gm+ team_gap>0 require_protagonist=True"
                )
                protagonist_dataset = private_ch.load_team_gap_percentiles_dataset(
                    region=region,
                    ch_prefix=region_to_ch_prefix[region],
                    season="live",
                    days=DEFAULT_DAYS,
                    min_games=DEFAULT_MIN_GAMES,
                    rank_filter=["Grand Master", "Challenger"],
                    require_protagonist=True,
                    require_positive_gap=True,
                    percentile_step=DEFAULT_PERCENTILE_STEP,
                    percentile_bin_size=DEFAULT_PERCENTILE_BIN_SIZE,
                    diff_bin_size=DEFAULT_DIFF_BIN_SIZE,
                    mu_bucket_size=DEFAULT_MU_BUCKET_SIZE,
                    progress_callback=_progress_callback,
                )
            chart_state["dataset"] = dataset
            chart_state["protagonist_dataset"] = protagonist_dataset
            _draw_all()
            _set_progress(100.0, "Completed")
            set_status(
                f"Loaded team gap relative-scale charts for {region} ({dataset['summary']['source_pair_rows']:,} rows, protagonist={require_protagonist})"
            )
        except Exception as exc:
            messagebox.showerror("Team Gap Percentiles Experiment Failed", str(exc))
            _set_progress(0.0, "Failed")
            set_status("Team gap percentiles experiment failed")

    generate_btn.configure(command=run_query)
    return {
        "frame": frame,
        "run_query": run_query,
        "tab_text": "Experiment - Team Gap Percentiles",
    }
