#!/usr/bin/env python3
import math
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from ranking_algorithm import UNBALANCED_TEAM_MU_REDUCTION, UNBALANCED_PAIR_RATIO_ALPHA

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


DEFAULT_DAYS = 180
DEFAULT_GAP_BIN_SIZE = 0.02
DEFAULT_MU_BIN_SIZE = 2.5
ASYMPTOTE_Y = 0.20
FORMULA_CURVE_SCALE_BY_QUANTILE = {"p90": 0.85, "p95": 0.95, "p99": 1.00}
FORMULA_GAP_MAX = 1.50
FORMULA_Y_MAX = 0.30
FORMULA_GAP_STEP = 0.001


def _linear_reduction_pct(base_gap_pct):
    return UNBALANCED_TEAM_MU_REDUCTION * base_gap_pct


def _smooth_tapered_reduction_pct(base_gap_pct, curve_scale):
    if curve_scale <= 0.0 or curve_scale > 1.0:
        raise ValueError("curve_scale must be in (0, 1]")
    if base_gap_pct <= 0.0:
        return 0.0
    linear_scaled_gap = (UNBALANCED_TEAM_MU_REDUCTION / ASYMPTOTE_Y) * base_gap_pct
    return ASYMPTOTE_Y * math.tanh(linear_scaled_gap * curve_scale)


def install_tab(experiment_notebook, private_ch, region_to_ch_prefix, set_status, log_console):
    frame = ttk.Frame(experiment_notebook, style="Panel.TFrame")
    experiment_notebook.add(frame, text="Experiment - Unbalanced Lobby Grace")

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

    ttk.Label(controls, text="Adaptive cap").grid(row=0, column=4, padx=(0, 8), pady=6, sticky="w")
    cap_quantile_var = tk.StringVar(value="p95")
    cap_quantile_combo = ttk.Combobox(
        controls,
        textvariable=cap_quantile_var,
        values=["p90", "p95", "p99"],
        width=8,
        state="readonly",
    )
    cap_quantile_combo.grid(row=0, column=5, padx=(0, 14), pady=6, sticky="w")

    regenerate_btn = ttk.Button(controls, text="Regenerate")
    regenerate_btn.grid(row=0, column=6, padx=(0, 8), pady=6, sticky="w")

    subtitle = ttk.Label(
        frame,
        text=(
            "Chart 1 is formula-only and renders instantly. Chart 2 (higher_player_mu bins) loads from query on Regenerate. "
            f"Linear reduction: {UNBALANCED_TEAM_MU_REDUCTION:.2f} * gap. "
            f"Alternative (green): one smooth asymptotic curve toward y={ASYMPTOTE_Y:.2f}, always <= red."
        ),
        style="Sub.TLabel",
    )
    subtitle.pack(anchor="w", padx=10, pady=(0, 6))

    summary_var = tk.StringVar(value="Chart 1 ready. Press Regenerate to load chart 2 data.")
    summary_label = ttk.Label(frame, textvariable=summary_var, style="Sub.TLabel")
    summary_label.pack(anchor="w", padx=10, pady=(0, 6))

    if not _MATPLOTLIB_AVAILABLE:
        ttk.Label(frame, text="matplotlib is unavailable, cannot render charts.", style="Sub.TLabel").pack(
            anchor="w", padx=10, pady=10
        )
        return frame

    figure = Figure(figsize=(13.8, 5.8), dpi=100)
    ax_delta = figure.add_subplot(1, 2, 1)
    ax_mu = figure.add_subplot(1, 2, 2)

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    chart_state = {
        "dataset": None,
    }

    def _draw_formula_chart():
        cap_key = cap_quantile_var.get().strip().lower()
        if cap_key not in FORMULA_CURVE_SCALE_BY_QUANTILE:
            raise ValueError(f"Unsupported cap quantile: {cap_key}")
        curve_scale = FORMULA_CURVE_SCALE_BY_QUANTILE[cap_key]
        x_max = FORMULA_GAP_MAX
        x_count = int(x_max / FORMULA_GAP_STEP) + 1
        gap_x = [idx * FORMULA_GAP_STEP for idx in range(x_count)]
        delta_linear = [_linear_reduction_pct(gap) for gap in gap_x]
        delta_log = [_smooth_tapered_reduction_pct(gap, curve_scale) for gap in gap_x]

        ax_delta.clear()
        ax_delta.plot(gap_x, delta_linear, color="#c22e2e", linewidth=2.2, label="CURRENT (linear)")
        ax_delta.plot(gap_x, delta_log, color="#217a4a", linewidth=2.2, label="NEW")
        ax_delta.set_title("Current vs New - Unbalanced Lobby Grace")
        ax_delta.set_xlabel("Team Skill - Lobby Avg Skill (%)")
        ax_delta.set_ylabel("Unbalanced Lobby Grace given")
        ax_delta.set_xlim(0.0, x_max)
        ax_delta.set_ylim(0.0, FORMULA_Y_MAX)
        ax_delta.grid(alpha=0.2)
        ax_delta.legend(loc="best", fontsize=8)

        if chart_state["dataset"] is None:
            summary_var.set(
                f"curve_scale({cap_key})={curve_scale:.2f} | "
                f"green_at_10%={_smooth_tapered_reduction_pct(0.10, curve_scale):.4f} | "
                "press Regenerate for chart 2 data"
            )
        figure.suptitle(
            f"Unbalanced Lobby Grace | mu_reduction={UNBALANCED_TEAM_MU_REDUCTION:.2f} pair_ratio_alpha={UNBALANCED_PAIR_RATIO_ALPHA:.2f}",
            fontsize=11,
        )
        canvas.draw_idle()

    def _draw_mu_placeholder():
        ax_mu.clear()
        ax_mu.text(0.5, 0.5, "Press Regenerate to load chart 2 from DB.", ha="center", va="center", transform=ax_mu.transAxes)
        ax_mu.set_title("2) Higher Rated Player Mu (Binned) -> Grace")
        ax_mu.set_xlabel("higher_player_mu_bin")
        ax_mu.set_ylabel("grace_given (reduction_pct)")
        ax_mu.grid(alpha=0.2)
        canvas.draw_idle()

    def _draw_mu_chart():
        dataset = chart_state["dataset"]
        if dataset is None:
            _draw_mu_placeholder()
            return

        mu_rows = dataset.get("mu_profile") or []
        mu_rows.sort(key=lambda row: row["player_mu_bin_start"])
        mu_x = [row["player_mu_bin_start"] + (dataset["meta"]["mu_bin_size"] / 2.0) for row in mu_rows]
        mu_recorded_avg = [float(row["avg_recorded_grace_pct"]) for row in mu_rows]
        mu_recorded_min = [float(row["min_recorded_grace_pct"]) for row in mu_rows]
        mu_recorded_max = [float(row["max_recorded_grace_pct"]) for row in mu_rows]
        summary = dataset["summary"]
        cap_key = cap_quantile_var.get().strip().lower()
        if cap_key not in FORMULA_CURVE_SCALE_BY_QUANTILE:
            raise ValueError(f"Unsupported cap quantile: {cap_key}")
        curve_scale = FORMULA_CURVE_SCALE_BY_QUANTILE[cap_key]

        summary_var.set(
            f"rows={summary['sample_count']:,} | positive={summary['positive_gap_count']:,} | "
            f"pair_ratio_avg={summary['avg_pair_ratio']:.3f} | p90={summary['gap_p90']:.3f} | "
            f"p95={summary['gap_p95']:.3f} | p99={summary['gap_p99']:.3f} | "
            f"curve_scale({cap_key})={curve_scale:.2f}"
        )

        ax_mu.clear()
        if mu_x:
            ax_mu.fill_between(mu_x, mu_recorded_min, mu_recorded_max, color="#c22e2e", alpha=0.15, label="current min-max")
            ax_mu.plot(mu_x, mu_recorded_avg, color="#c22e2e", linewidth=2.0, label="current avg")
            ax_mu.legend(loc="best", fontsize=7)
        else:
            ax_mu.text(0.5, 0.5, "No mu profile rows returned.", ha="center", va="center", transform=ax_mu.transAxes)
        ax_mu.set_title("2) Higher Rated Player Mu (Binned) -> Grace")
        ax_mu.set_xlabel("higher_player_mu_bin")
        ax_mu.set_ylabel("grace_given (reduction_pct)")
        ax_mu.set_xlim(left=45.0)
        ax_mu.grid(alpha=0.2)

        figure.suptitle(
            f"Unbalanced Lobby Grace | region={dataset['meta']['region']} season={dataset['meta']['season']} "
            f"| mu_reduction={UNBALANCED_TEAM_MU_REDUCTION:.2f} pair_ratio_alpha={UNBALANCED_PAIR_RATIO_ALPHA:.2f}",
            fontsize=11,
        )
        canvas.draw_idle()

    def run_query():
        try:
            region = region_var.get().strip().lower()
            if region not in region_to_ch_prefix:
                raise ValueError(f"Unknown region: {region}")
            season = season_var.get().strip().lower()

            set_status(f"Loading unbalanced grace chart 2 for {region}...")
            log_console(
                "[INFO] Unbalanced grace query "
                f"region={region} season={season} days={DEFAULT_DAYS} "
                f"gap_bin={DEFAULT_GAP_BIN_SIZE} mu_bin={DEFAULT_MU_BIN_SIZE} "
                "filter=unbalanced_reduction_pct>0"
            )
            dataset = private_ch.load_unbalanced_lobby_grace_dataset(
                region=region,
                ch_prefix=region_to_ch_prefix[region],
                season=season,
                days=DEFAULT_DAYS,
                gap_bin_size=DEFAULT_GAP_BIN_SIZE,
                mu_bin_size=DEFAULT_MU_BIN_SIZE,
            )
            chart_state["dataset"] = dataset
            _draw_mu_chart()
            set_status(f"Loaded unbalanced grace chart 2 for {region} ({dataset['summary']['sample_count']:,} rows)")
        except Exception as exc:
            messagebox.showerror("Unbalanced Grace Experiment Failed", str(exc))
            set_status("Unbalanced grace experiment failed")

    regenerate_btn.configure(command=run_query)
    region_combo.bind("<<ComboboxSelected>>", lambda _event: (chart_state.__setitem__("dataset", None), _draw_mu_placeholder(), _draw_formula_chart(), set_status("Press Regenerate to load chart 2")))
    season_combo.bind("<<ComboboxSelected>>", lambda _event: (chart_state.__setitem__("dataset", None), _draw_mu_placeholder(), _draw_formula_chart(), set_status("Press Regenerate to load chart 2")))
    cap_quantile_combo.bind(
        "<<ComboboxSelected>>",
        lambda _event: (_draw_formula_chart(), _draw_mu_chart() if chart_state["dataset"] is not None else None),
    )
    _draw_mu_placeholder()
    _draw_formula_chart()
    return {
        "frame": frame,
        "run_query": _draw_formula_chart,
        "tab_text": "Experiment - Unbalanced Lobby Grace",
    }
