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


DEFAULT_DAYS = 180
DEFAULT_DAMAGE_BIN_SIZE = 1000


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - AFK Damage")

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

    ttk.Label(controls, text="Bin Size").grid(row=0, column=6, padx=(0, 8), pady=6, sticky="w")
    damage_bin_var = tk.StringVar(value=str(DEFAULT_DAMAGE_BIN_SIZE))
    damage_bin_spin = ttk.Spinbox(controls, from_=1, to=50000, increment=100, textvariable=damage_bin_var, width=8)
    damage_bin_spin.grid(row=0, column=7, padx=(0, 14), pady=6, sticky="w")

    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=8, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Filters rows where kills=0 and assists=0, then shows a histogram of total_damage_dealt_to_champions "
            "for the selected region/season and time window."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Ready. Press Regenerate to load AFK damage distribution.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return {
            "frame": frame,
            "run_query": lambda: set_status("matplotlib is unavailable for AFK damage chart"),
            "tab_text": "Experiment - AFK Damage",
        }

    figure = Figure(figsize=(13.8, 6.2), dpi=100)
    ax = figure.add_subplot(1, 1, 1)

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    chart_state = {
        "dataset": None,
    }

    def _draw_placeholder():
        ax.clear()
        ax.text(0.5, 0.5, "Press Regenerate to load AFK histogram data.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AFK Damage Dealt Histogram")
        ax.set_xlabel("total_damage_dealt_to_champions")
        ax.set_ylabel("game_count")
        ax.grid(alpha=0.2)
        canvas.draw_idle()

    def _draw_chart():
        dataset = chart_state["dataset"]
        if dataset is None:
            _draw_placeholder()
            return

        histogram_rows = dataset.get("histogram") or []
        histogram_rows.sort(key=lambda row: row["damage_bin_start"])
        bin_size = float(dataset["meta"]["damage_bin_size"])
        x_bins = [float(row["damage_bin_start"]) for row in histogram_rows]
        y_counts = [int(row["game_count"]) for row in histogram_rows]
        summary = dataset["summary"]
        zero_damage_pct = (summary["zero_damage_count"] / summary["sample_count"]) * 100.0

        ax.clear()
        if histogram_rows:
            ax.bar(
                x_bins,
                y_counts,
                width=bin_size * 0.95,
                align="edge",
                color="#245f8c",
                edgecolor="#173c58",
                linewidth=0.5,
            )
            ax.axvline(summary["damage_p50"], color="#2a7f62", linewidth=1.8, linestyle="--", label=f"p50={summary['damage_p50']:.0f}")
            ax.axvline(summary["damage_p90"], color="#a06c00", linewidth=1.8, linestyle="--", label=f"p90={summary['damage_p90']:.0f}")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No histogram bins returned.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AFK Damage Dealt Histogram")
        ax.set_xlabel("total_damage_dealt_to_champions")
        ax.set_ylabel("game_count")
        ax.grid(alpha=0.2)
        ax.set_xlim(left=0.0)

        summary_var.set(
            f"rows={summary['sample_count']:,} | min={summary['min_damage']:,} | max={summary['max_damage']:,} | "
            f"avg={summary['avg_damage']:.1f} | p50={summary['damage_p50']:.1f} | p90={summary['damage_p90']:.1f} | "
            f"p99={summary['damage_p99']:.1f} | zero_damage={summary['zero_damage_count']:,} ({zero_damage_pct:.2f}%)"
        )
        figure.suptitle(
            f"AFK Damage | region={dataset['meta']['region']} season={dataset['meta']['season']} "
            f"days={dataset['meta']['days']} bin={bin_size:.0f}",
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
            damage_bin_size = int(damage_bin_var.get().strip())
            if days <= 0:
                raise ValueError("Days must be > 0")
            if damage_bin_size <= 0:
                raise ValueError("Bin Size must be > 0")

            set_status(f"Loading AFK damage histogram for {region}...")
            log_console(
                "[INFO] AFK damage histogram query "
                f"region={region} season={season} days={days} damage_bin_size={damage_bin_size} "
                "filter=kills=0,assists=0"
            )
            chart_state["dataset"] = private_ch.load_afk_damage_histogram_dataset(
                region=region,
                ch_prefix=region_to_ch_prefix[region],
                season=season,
                days=days,
                damage_bin_size=damage_bin_size,
            )
            _draw_chart()
            set_status(
                f"Loaded AFK damage histogram for {region} "
                f"({chart_state['dataset']['summary']['sample_count']:,} rows)"
            )
        except Exception as exc:
            log_console(f"[ERROR] AFK damage experiment failed: {exc}")
            log_console(traceback.format_exc().rstrip())
            messagebox.showerror("AFK Damage Experiment Failed", str(exc))
            set_status("AFK damage experiment failed")

    regenerate_btn.configure(command=run_query)
    region_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("dataset", None),
            _draw_placeholder(),
            summary_var.set("Ready. Press Regenerate to load AFK damage distribution."),
        ),
    )
    season_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("dataset", None),
            _draw_placeholder(),
            summary_var.set("Ready. Press Regenerate to load AFK damage distribution."),
        ),
    )
    _draw_placeholder()

    return {
        "frame": frame,
        "run_query": run_query,
        "tab_text": "Experiment - AFK Damage",
    }
