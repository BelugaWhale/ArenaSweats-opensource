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


DEFAULT_MU_HIGH = 40.0
DEFAULT_TRIGGER_LOW_MU_RATIO = 0.90
DEFAULT_SATURATION_LOW_MU = 20.0
# Legacy relative-gap curve constants used for experiment charting.
GAP_TRIGGER = 0.10
GAP_SATURATION = 0.55


def _old_scale(mu_high, mu_low):
    if mu_high <= 0.0:
        raise ValueError(f"mu_high must be > 0, got {mu_high}")
    gap_pct = max(0.0, min(1.0, 1.0 - (mu_low / mu_high)))
    if gap_pct <= GAP_TRIGGER:
        return 1.0
    if gap_pct >= GAP_SATURATION:
        return PENALTY_MIN_MULTIPLIER
    progress = (gap_pct - GAP_TRIGGER) / (GAP_SATURATION - GAP_TRIGGER)
    return 1.0 - (1.0 - PENALTY_MIN_MULTIPLIER) * progress


def _new_scale(mu_high, mu_low, trigger_low_mu_ratio, saturation_low_mu):
    if mu_high <= 0.0:
        raise ValueError(f"mu_high must be > 0, got {mu_high}")
    trigger_low_mu = mu_high * trigger_low_mu_ratio
    if mu_low >= trigger_low_mu:
        return 1.0
    if mu_low <= saturation_low_mu:
        return PENALTY_MIN_MULTIPLIER
    if trigger_low_mu <= saturation_low_mu:
        return PENALTY_MIN_MULTIPLIER
    progress = (trigger_low_mu - mu_low) / (trigger_low_mu - saturation_low_mu)
    return max(PENALTY_MIN_MULTIPLIER, min(1.0, 1.0 - (1.0 - PENALTY_MIN_MULTIPLIER) * progress))


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Team Gap Curve")

    controls = ttk.Frame(frame, style="Panel.TFrame")
    controls.pack(fill="x", padx=10, pady=(10, 6))

    ttk.Label(controls, text="Main Player mu (mu_high)").grid(row=0, column=0, padx=(0, 8), pady=6, sticky="w")
    mu_high_var = tk.StringVar(value=f"{DEFAULT_MU_HIGH:.2f}")
    mu_high_entry = ttk.Entry(controls, textvariable=mu_high_var, width=10)
    mu_high_entry.grid(row=0, column=1, padx=(0, 14), pady=6, sticky="w")

    ttk.Label(controls, text="Trigger low-mu ratio").grid(row=0, column=2, padx=(0, 8), pady=6, sticky="w")
    trigger_ratio_var = tk.StringVar(value=f"{DEFAULT_TRIGGER_LOW_MU_RATIO:.2f}")
    trigger_ratio_entry = ttk.Entry(controls, textvariable=trigger_ratio_var, width=8)
    trigger_ratio_entry.grid(row=0, column=3, padx=(0, 14), pady=6, sticky="w")

    ttk.Label(controls, text="Saturation teammate mu").grid(row=0, column=4, padx=(0, 8), pady=6, sticky="w")
    saturation_low_mu_var = tk.StringVar(value=f"{DEFAULT_SATURATION_LOW_MU:.2f}")
    saturation_low_mu_entry = ttk.Entry(controls, textvariable=saturation_low_mu_var, width=8)
    saturation_low_mu_entry.grid(row=0, column=5, padx=(0, 14), pady=6, sticky="w")

    redraw_btn = ttk.Button(controls, text="Redraw")
    redraw_btn.grid(row=0, column=6, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Compares old and new teammate-gap penalty curves at a chosen mu_high. "
            "Old system saturates by relative gap; new system saturates when teammate mu reaches fixed low-mu floor."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Ready.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return frame

    figure = Figure(figsize=(14.5, 6.8), dpi=100)
    axis = figure.add_subplot(1, 1, 1)
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _to_float(value_text, name):
        value = float(value_text)
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value

    def _draw():
        mu_high = _to_float(mu_high_var.get().strip(), "mu_high")
        trigger_ratio = float(trigger_ratio_var.get().strip())
        saturation_low_mu = float(saturation_low_mu_var.get().strip())
        if trigger_ratio <= 0.0 or trigger_ratio >= 1.0:
            raise ValueError(f"trigger ratio must be in (0, 1), got {trigger_ratio}")
        if saturation_low_mu < 0.0:
            raise ValueError(f"saturation teammate mu must be >= 0, got {saturation_low_mu}")

        old_trigger_low_mu = mu_high * (1.0 - GAP_TRIGGER)
        old_saturation_low_mu = mu_high * (1.0 - GAP_SATURATION)
        new_trigger_low_mu = mu_high * trigger_ratio
        x_max = max(mu_high * 1.1, saturation_low_mu + 10.0, 40.0)
        points = 480
        step = x_max / points
        mu_low_values = [idx * step for idx in range(points + 1)]
        old_scales = [_old_scale(mu_high, mu_low) for mu_low in mu_low_values]
        new_scales = [_new_scale(mu_high, mu_low, trigger_ratio, saturation_low_mu) for mu_low in mu_low_values]

        axis.clear()
        axis.plot(mu_low_values, old_scales, color="#2f5a9a", linewidth=2.4, label="Old scale (relative-gap saturation)")
        axis.plot(mu_low_values, new_scales, color="#c13b20", linewidth=2.4, label="New scale (teammate-mu saturation)")
        axis.fill_between(mu_low_values, old_scales, new_scales, color="#f4bb6a", alpha=0.18, label="Delta region")
        axis.axvline(old_trigger_low_mu, color="#2f5a9a", linestyle="--", linewidth=1.3, alpha=0.8)
        axis.axvline(old_saturation_low_mu, color="#2f5a9a", linestyle=":", linewidth=1.3, alpha=0.8)
        axis.axvline(new_trigger_low_mu, color="#c13b20", linestyle="--", linewidth=1.3, alpha=0.8)
        axis.axvline(saturation_low_mu, color="#c13b20", linestyle=":", linewidth=1.3, alpha=0.8)
        axis.set_title(
            "Team Gap Scale Curve Comparison (Old vs New)\n"
            f"mu_high={mu_high:.2f}, old_sat_low_mu={old_saturation_low_mu:.2f}, new_sat_low_mu={saturation_low_mu:.2f}"
        )
        axis.set_xlabel("Teammate mu (mu_low)")
        axis.set_ylabel("team_gap_scale")
        axis.set_xlim(0.0, x_max)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=8)
        canvas.draw_idle()

        summary_var.set(
            "Old trigger/sat mu_low: "
            f"{old_trigger_low_mu:.2f}/{old_saturation_low_mu:.2f} | "
            "New trigger/sat mu_low: "
            f"{new_trigger_low_mu:.2f}/{saturation_low_mu:.2f} | "
            f"min_scale={PENALTY_MIN_MULTIPLIER:.2f}"
        )
        set_status("Team gap curve chart updated")
        log_console(
            "[INFO] Team gap curve redraw: "
            f"mu_high={mu_high:.4f}, trigger_ratio={trigger_ratio:.4f}, saturation_low_mu={saturation_low_mu:.4f}"
        )

    def _redraw_from_ui():
        try:
            _draw()
        except Exception as exc:
            messagebox.showerror("Team Gap Curve", str(exc))
            set_status("Team gap curve redraw failed")

    redraw_btn.configure(command=_redraw_from_ui)
    mu_high_entry.bind("<Return>", lambda _event: _redraw_from_ui())
    trigger_ratio_entry.bind("<Return>", lambda _event: _redraw_from_ui())
    saturation_low_mu_entry.bind("<Return>", lambda _event: _redraw_from_ui())
    _redraw_from_ui()
    return frame
