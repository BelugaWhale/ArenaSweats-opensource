#!/usr/bin/env python3
from datetime import datetime
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


LOCKED_WINDOW_START_UTC = "2026-01-07 03:00:00"
LOCKED_WINDOW_END_UTC = None
LOCKED_TOP_N = 10
TARGET_REGIONS = ["na", "oce", "euw", "tr", "lan"]


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Lobby Quality")

    controls = ttk.Frame(frame, style="Panel.TFrame")
    controls.pack(fill="x", padx=10, pady=(10, 6))

    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=0, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Top 10 players per region (from player_ranks), then player_matchhistory filtered to placing=1 "
            f"within UTC window [{LOCKED_WINDOW_START_UTC}, latest). "
            "Plots daily avg_worse_opp_rating as time-series lines by region."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Ready. Press Regenerate to load lobby-quality time series.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return {
            "frame": frame,
            "run_query": lambda: set_status("matplotlib is unavailable for lobby quality chart"),
            "tab_text": "Experiment - Lobby Quality",
        }

    figure = Figure(figsize=(14.2, 7.2), dpi=100)
    ax_series = figure.add_subplot(1, 1, 1)

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    chart_state = {"datasets": []}

    def _draw_placeholder():
        ax_series.clear()
        ax_series.text(
            0.5,
            0.5,
            "Press Regenerate to load lobby-quality data.",
            ha="center",
            va="center",
            transform=ax_series.transAxes,
        )
        ax_series.set_title("Daily avg_worse_opp_rating by region")
        ax_series.set_xlabel("time (UTC day)")
        ax_series.set_ylabel("avg_worse_opp_rating")
        ax_series.grid(alpha=0.2)
        canvas.draw_idle()

    def _draw_chart():
        datasets = chart_state["datasets"]
        if not datasets:
            _draw_placeholder()
            return

        global_first_place_rows = 0
        global_weighted_sum = 0.0
        global_players_with_data = 0
        total_time_points = 0

        ax_series.clear()
        for dataset in datasets:
            region = str(dataset["meta"]["region"])
            summary = dataset["summary"]
            weighted_avg = summary["avg_worse_opp_rating_weighted"]
            first_place_rows_total = int(summary["first_place_rows_total"])
            if weighted_avg is not None and first_place_rows_total > 0:
                global_first_place_rows += first_place_rows_total
                global_weighted_sum += float(weighted_avg) * first_place_rows_total
            global_players_with_data += int(summary["players_with_data_count"])
            series = dataset.get("time_series") or []
            if not series:
                raise ValueError(f"No time-series rows found for region '{region}'")
            x_values = []
            y_values = []
            region_first_place_rows = 0
            for row in series:
                day_start_utc = str(row["day_start_utc"])
                avg_worse_opp_rating = row["avg_worse_opp_rating"]
                first_place_rows = int(row["first_place_rows"])
                if not day_start_utc:
                    raise ValueError(f"Missing day_start_utc in time-series for region '{region}'")
                if avg_worse_opp_rating is None:
                    raise ValueError(f"Missing avg_worse_opp_rating in time-series for region '{region}'")
                x_values.append(datetime.fromisoformat(day_start_utc.replace("Z", "+00:00")))
                y_values.append(float(avg_worse_opp_rating))
                region_first_place_rows += first_place_rows
            total_time_points += len(x_values)
            ax_series.plot(
                x_values,
                y_values,
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                label=f"{region} (n={region_first_place_rows:,})",
            )

        if total_time_points <= 0:
            raise ValueError("No time-series points found for lobby quality chart")
        if global_first_place_rows <= 0:
            raise ValueError("No first-place rows found across all regions for lobby quality chart")

        global_weighted_avg = global_weighted_sum / global_first_place_rows
        summary_var.set(
            f"regions={len(datasets)} | top_n={LOCKED_TOP_N} | players_with_data={global_players_with_data} | "
            f"first_place_rows={global_first_place_rows:,} | weighted_avg_worse_opp={global_weighted_avg:.2f} | "
            f"time_points={total_time_points:,}"
        )
        window_label = f"[{LOCKED_WINDOW_START_UTC}, latest)"
        if LOCKED_WINDOW_END_UTC:
            window_label = f"[{LOCKED_WINDOW_START_UTC}, {LOCKED_WINDOW_END_UTC})"
        ax_series.set_title("Daily avg_worse_opp_rating | top-10 by player_ranks, placing=1 only")
        ax_series.set_xlabel("time (UTC day)")
        ax_series.set_ylabel("avg_worse_opp_rating")
        ax_series.grid(alpha=0.2)
        ax_series.legend(loc="best", fontsize=9)
        figure.suptitle(
            "Lobby Quality | window_utc=" f"{window_label} | placing=1",
            fontsize=11,
        )
        figure.autofmt_xdate(rotation=30, ha="right")
        canvas.draw_idle()

    def run_query():
        try:
            sorted_regions = [region for region in TARGET_REGIONS if region in region_to_ch_prefix]
            if len(sorted_regions) != len(TARGET_REGIONS):
                missing_regions = sorted(set(TARGET_REGIONS) - set(sorted_regions))
                raise ValueError(f"Missing target regions in region_to_ch_prefix: {missing_regions}")
            datasets = []
            for idx, region in enumerate(sorted_regions, start=1):
                set_status(f"Loading lobby quality {idx}/{len(sorted_regions)} ({region})...")
                window_end_text = LOCKED_WINDOW_END_UTC if LOCKED_WINDOW_END_UTC else "latest"
                log_console(
                    "[INFO] Lobby quality query "
                    f"region={region} season=live top_n={LOCKED_TOP_N} "
                    f"window_start_utc={LOCKED_WINDOW_START_UTC} window_end_utc={window_end_text} "
                    "filter=placing=1"
                )
                datasets.append(
                    private_ch.load_lobby_quality_dataset(
                        region=region,
                        ch_prefix=region_to_ch_prefix[region],
                        season="live",
                        top_n=LOCKED_TOP_N,
                        window_start_utc=LOCKED_WINDOW_START_UTC,
                        window_end_utc=LOCKED_WINDOW_END_UTC,
                    )
                )
            chart_state["datasets"] = datasets
            _draw_chart()
            total_rows = sum(dataset["summary"]["first_place_rows_total"] for dataset in datasets)
            set_status(
                f"Loaded lobby quality for {len(datasets)} regions "
                f"({total_rows:,} first-place rows)"
            )
        except Exception as exc:
            log_console(f"[ERROR] Lobby quality experiment failed: {exc}")
            log_console(traceback.format_exc().rstrip())
            messagebox.showerror("Lobby Quality Experiment Failed", str(exc))
            set_status("Lobby quality experiment failed")

    regenerate_btn.configure(command=run_query)
    _draw_placeholder()

    return {
        "frame": frame,
        "run_query": run_query,
        "tab_text": "Experiment - Lobby Quality",
    }
