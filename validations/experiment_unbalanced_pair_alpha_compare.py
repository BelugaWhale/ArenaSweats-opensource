#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


COMPARE_ALPHA = 3.0
BASE_GRACE_SCENARIOS = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18]


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Unbalanced Alpha")

    controls = ttk.Frame(frame, style="Panel.TFrame")
    controls.pack(fill="x", padx=10, pady=(10, 6))

    ttk.Label(controls, text="Alpha").grid(row=0, column=0, padx=(0, 8), pady=6, sticky="w")
    alpha_var = tk.StringVar(value=f"{COMPARE_ALPHA:.1f}")
    alpha_entry = ttk.Entry(controls, textvariable=alpha_var, width=8, state="readonly")
    alpha_entry.grid(row=0, column=1, padx=(0, 14), pady=6, sticky="w")

    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=2, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Formula-only visualization (no data query). "
            "For each baseline grace g in {3,6,9,12,15,18}%, plotted line is: adjusted_grace = g * ((1-gap)^alpha), "
            "where gap is Team gap (%) = 1 - (mu_low / mu_high)."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Ready to render formula lines.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return frame

    figure = Figure(figsize=(13.8, 6.2), dpi=100)
    ax = figure.add_subplot(1, 1, 1)

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def draw_chart():
        set_status("Rendering formula lines...")
        x_gap = [value / 100.0 for value in range(0, 51, 1)]
        x_percent = [value * 100.0 for value in x_gap]

        ax.clear()
        colors = ["#1d4f8f", "#2a7f62", "#a06c00", "#7a2f9a", "#c22e2e", "#5d4a3d"]
        for idx, base_grace in enumerate(BASE_GRACE_SCENARIOS):
            y_percent = [base_grace * ((1.0 - gap) ** COMPARE_ALPHA) * 100.0 for gap in x_gap]
            ax.plot(
                x_percent,
                y_percent,
                color=colors[idx % len(colors)],
                linewidth=2.0,
                label=f"base grace {base_grace * 100.0:.0f}%",
            )

        ax.set_title(f"Adjusted Grace By Team Gap (alpha={COMPARE_ALPHA:.0f})")
        ax.set_xlabel("Team gap (%)")
        ax.set_ylabel("Adjusted grace (%)")
        ax.grid(alpha=0.2)
        ax.set_xlim(0.0, 50.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left", fontsize=8)

        sample_gap = 0.05
        sample_text = []
        for base_grace in BASE_GRACE_SCENARIOS:
            adjusted = base_grace * ((1.0 - sample_gap) ** COMPARE_ALPHA) * 100.0
            sample_text.append(f"{base_grace * 100.0:.0f}%->{adjusted:.4f}%")
        summary_var.set(
            f"At team gap=5% with alpha={COMPARE_ALPHA:.0f}: " + " | ".join(sample_text)
        )

        figure.suptitle("Grace Scenarios Using adjusted_grace = grace * ((1-gap)^alpha)", fontsize=11)
        canvas.draw_idle()
        log_console("[INFO] Rendered formula-only alpha scenario chart")
        set_status("Formula chart ready")

    regenerate_btn.configure(command=draw_chart)
    draw_chart()

    return {
        "frame": frame,
        "run_query": draw_chart,
        "tab_text": "Experiment - Unbalanced Alpha",
    }
