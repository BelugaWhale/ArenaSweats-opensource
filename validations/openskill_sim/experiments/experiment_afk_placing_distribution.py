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
DEFAULT_MAX_DAMAGE = 5000


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - AFK Placings")

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

    ttk.Label(controls, text="Max Damage").grid(row=0, column=6, padx=(0, 8), pady=6, sticky="w")
    max_damage_var = tk.StringVar(value=str(DEFAULT_MAX_DAMAGE))
    max_damage_spin = ttk.Spinbox(controls, from_=1, to=50000, increment=500, textvariable=max_damage_var, width=8)
    max_damage_spin.grid(row=0, column=7, padx=(0, 14), pady=6, sticky="w")

    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=8, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Filters rows where kills=0, assists=0, damage<max_damage, then shows a placing distribution "
            "as % of filtered players for the selected region/season and time window."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Ready. Press Regenerate to load placing distribution.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return {
            "frame": frame,
            "run_query": lambda: set_status("matplotlib is unavailable for AFK placing chart"),
            "tab_text": "Experiment - AFK Placings",
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
        ax.text(0.5, 0.5, "Press Regenerate to load AFK placing distribution.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AFK Placing Distribution")
        ax.set_xlabel("placing")
        ax.set_ylabel("frequency (%)")
        ax.grid(alpha=0.2, axis="y")
        canvas.draw_idle()

    def _draw_chart():
        dataset = chart_state["dataset"]
        if dataset is None:
            _draw_placeholder()
            return

        distribution_rows = dataset.get("distribution") or []
        summary = dataset["summary"]
        sample_count = summary["sample_count"]
        placements = list(range(1, 9))
        counts_map = {int(row["placing"]): int(row["placing_count"]) for row in distribution_rows}
        counts = [counts_map.get(placing, 0) for placing in placements]
        percentages = [(count / sample_count) * 100.0 if sample_count else 0.0 for count in counts]

        ax.clear()
        if sample_count > 0:
            ax.bar(
                placements,
                percentages,
                width=0.8,
                color="#245f8c",
                edgecolor="#173c58",
                linewidth=0.6,
            )
        else:
            ax.text(0.5, 0.5, "No rows returned.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AFK Placing Distribution")
        ax.set_xlabel("placing")
        ax.set_ylabel("frequency (%)")
        ax.set_xticks(placements)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.2, axis="y")

        summary_var.set(
            f"rows={summary['sample_count']:,} | games={summary['game_count']:,} | "
            f"placing_range={summary['min_placing']}-{summary['max_placing']} | "
            f"max_damage<{dataset['meta']['max_damage']:,}"
        )
        figure.suptitle(
            f"AFK Placings | region={dataset['meta']['region']} season={dataset['meta']['season']} "
            f"days={dataset['meta']['days']} max_damage<{dataset['meta']['max_damage']:,}",
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
            max_damage = int(max_damage_var.get().strip())
            if days <= 0:
                raise ValueError("Days must be > 0")
            if max_damage <= 0:
                raise ValueError("Max Damage must be > 0")

            set_status(f"Loading AFK placing distribution for {region}...")
            log_console(
                "[INFO] AFK placing distribution query "
                f"region={region} season={season} days={days} max_damage<{max_damage} "
                "filter=kills=0,assists=0"
            )
            chart_state["dataset"] = private_ch.load_afk_placing_distribution_dataset(
                region=region,
                ch_prefix=region_to_ch_prefix[region],
                season=season,
                days=days,
                max_damage=max_damage,
            )
            _draw_chart()
            set_status(
                f"Loaded AFK placing distribution for {region} "
                f"({chart_state['dataset']['summary']['sample_count']:,} rows)"
            )
        except Exception as exc:
            log_console(f"[ERROR] AFK placing experiment failed: {exc}")
            log_console(traceback.format_exc().rstrip())
            messagebox.showerror("AFK Placing Experiment Failed", str(exc))
            set_status("AFK placing experiment failed")

    regenerate_btn.configure(command=run_query)
    region_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("dataset", None),
            _draw_placeholder(),
            summary_var.set("Ready. Press Regenerate to load placing distribution."),
        ),
    )
    season_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (
            chart_state.__setitem__("dataset", None),
            _draw_placeholder(),
            summary_var.set("Ready. Press Regenerate to load placing distribution."),
        ),
    )
    _draw_placeholder()

    return {
        "frame": frame,
        "run_query": run_query,
        "tab_text": "Experiment - AFK Placings",
    }
