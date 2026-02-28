#!/usr/bin/env python3
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ranking_algorithm import instantiate_rating_model, calculate_rating
from experiments.experiment_team_gap_population import install_tab as install_team_gap_population_tab
from experiments.experiment_team_gap_percentiles import install_tab as install_team_gap_percentiles_tab
from experiments.experiment_team_gap_curve_compare import install_tab as install_team_gap_curve_compare_tab
from experiments.experiment_unbalanced_lobby_grace import install_tab as install_unbalanced_lobby_grace_tab
from experiments.experiment_unbalanced_pair_alpha_compare import install_tab as install_unbalanced_pair_alpha_compare_tab
from experiments.experiment_afk_damage_histogram import install_tab as install_afk_damage_histogram_tab
from experiments.experiment_afk_placing_distribution import install_tab as install_afk_placing_distribution_tab
from experiments.experiment_unbalanced_inflation_solo_safety import install_tab as install_unbalanced_inflation_solo_safety_tab
from experiments.experiment_emerald_or_below_placing_boxplot import install_tab as install_emerald_or_below_placing_boxplot_tab

REGION_TO_CH_PREFIX = {
    "euw": "DBCH",
    "br": "DBCH",
    "sea": "DBCH",
    "ru": "DBCH",
    "me": "DBCH",
    "na": "SCRAPECH",
    "vn": "SCRAPECH",
    "las": "SCRAPECH",
    "tr": "SCRAPECH",
    "oce": "SCRAPECH",
    "kr": "SPLITCH",
    "eune": "SPLITCH",
    "lan": "SPLITCH",
    "tw": "SPLITCH",
    "jp": "SPLITCH",
}

private_ch = importlib.import_module("openskill_sim_ch_private")
if not hasattr(private_ch, "list_player_games_exact"):
    raise RuntimeError(
        "Missing list_player_games_exact(player_name, region, ch_prefix, limit, season='live') "
        "in openskill_sim_ch_private.py"
    )
if not hasattr(private_ch, "load_sim_input_from_clickhouse"):
    raise RuntimeError(
        "Missing load_sim_input_from_clickhouse(game_id, region, ch_prefix, season='live') "
        "in openskill_sim_ch_private.py"
    )
if not hasattr(private_ch, "load_unbalanced_lobby_grace_dataset"):
    raise RuntimeError(
        "Missing load_unbalanced_lobby_grace_dataset(region, ch_prefix, season='live', ...) "
        "in openskill_sim_ch_private.py"
    )
if not hasattr(private_ch, "load_unbalanced_alpha_compare_dataset"):
    raise RuntimeError(
        "Missing load_unbalanced_alpha_compare_dataset(region, ch_prefix, season='live', ...) "
        "in openskill_sim_ch_private.py"
    )
if not hasattr(private_ch, "load_team_gap_percentiles_dataset"):
    raise RuntimeError(
        "Missing load_team_gap_percentiles_dataset(region, ch_prefix, season='live', ...) "
        "in openskill_sim_ch_private.py"
    )
if not hasattr(private_ch, "load_afk_damage_histogram_dataset"):
    raise RuntimeError(
        "Missing load_afk_damage_histogram_dataset(region, ch_prefix, season='live', ...) "
        "in openskill_sim_ch_private.py"
    )
if not hasattr(private_ch, "load_afk_placing_distribution_dataset"):
    raise RuntimeError(
        "Missing load_afk_placing_distribution_dataset(region, ch_prefix, season='live', ...) "
        "in openskill_sim_ch_private.py"
    )
if not hasattr(private_ch, "load_unbalanced_inflation_solo_safety_dataset"):
    raise RuntimeError(
        "Missing load_unbalanced_inflation_solo_safety_dataset(region, ch_prefix, season='live', ...) "
        "in openskill_sim_ch_private.py"
    )

root = tk.Tk()
root.title("ArenaSweats OpenSkill Debugger")
root.geometry("1780x980")
root.configure(bg="#0f141b")

style = ttk.Style()
style.theme_use("clam")
style.configure("Main.TFrame", background="#0f141b")
style.configure("Panel.TFrame", background="#17212d")
style.configure("Header.TLabel", background="#0f141b", foreground="#f3f7ff", font=("Segoe UI", 22, "bold"))
style.configure("Sub.TLabel", background="#0f141b", foreground="#c2d1e4", font=("Segoe UI", 12))
style.configure("Status.TLabel", background="#0f141b", foreground="#8ce8bb", font=("Segoe UI", 12, "bold"))
style.configure("TLabel", background="#17212d", foreground="#e9f0fa", font=("Segoe UI", 12))
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8, background="#24415e", foreground="#f3f7ff", borderwidth=0)
style.map("TButton", background=[("active", "#2f557c"), ("disabled", "#1c2f42")], foreground=[("active", "#ffffff"), ("disabled", "#8ea2ba")])
style.configure("TEntry", padding=6, fieldbackground="#101823", foreground="#f0f6ff", insertcolor="#ffffff", borderwidth=0)
style.configure("TSpinbox", padding=6, fieldbackground="#101823", foreground="#f0f6ff", borderwidth=0)
style.configure("TCombobox", padding=6, fieldbackground="#101823", foreground="#f0f6ff", borderwidth=0, arrowsize=16)
style.map(
    "TCombobox",
    fieldbackground=[("readonly", "#101823"), ("disabled", "#0f141b")],
    foreground=[("readonly", "#f0f6ff"), ("disabled", "#8ea2ba")],
    selectbackground=[("readonly", "#2c4b69")],
    selectforeground=[("readonly", "#ffffff")],
    background=[("readonly", "#101823"), ("disabled", "#0f141b")],
)
style.configure("Treeview", font=("Segoe UI", 11), rowheight=30, fieldbackground="#111a26", background="#111a26", foreground="#f0f6ff")
style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"), background="#26445f", foreground="#f3f8ff")
style.map("Treeview", background=[("selected", "#355e84")], foreground=[("selected", "#ffffff")])
style.configure("TNotebook", background="#17212d", borderwidth=0)
style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=(12, 8), background="#24415e", foreground="#dbe7f6")
style.map("TNotebook.Tab", background=[("selected", "#17212d")], foreground=[("selected", "#ffffff")])
style.configure("Loading.Horizontal.TProgressbar", troughcolor="#111a26", background="#55c7f7", borderwidth=0, lightcolor="#55c7f7", darkcolor="#55c7f7")
root.option_add("*TCombobox*Listbox*Background", "#101823")
root.option_add("*TCombobox*Listbox*Foreground", "#f0f6ff")
root.option_add("*TCombobox*Listbox*selectBackground", "#2c4b69")
root.option_add("*TCombobox*Listbox*selectForeground", "#ffffff")
root.option_add("*TCombobox*Listbox*Font", ("Segoe UI", 11))

state = {
    "games": [],
    "game_index": -1,
    "games_sort_state": {},
    "games_all_columns": [],
    "games_visible_columns": ["game_ts", "placing", "rating_change", "champion"],
    "match_history_expanded": False,
    "player_name": "",
    "region": "euw",
    "season": "live",
    "tmp_dir": None,
    "input_path": None,
    "report_path": None,
    "is_loading": False,
}

EXPERIMENT_SIGMA_PLAYER = "Gwrach y Rhibyn#ofn"
EXPERIMENT_SIGMA_OVERRIDE = 2.32
EXPERIMENT_SIGMA_PLAYERS = [
    {"placing": 1, "player": "Gwrach y Rhibyn#ofn", "mu": 26.25, "sigma": 4.24},
    {"placing": 1, "player": "Kibbylol#EUW", "mu": 47.20, "sigma": 2.32},
    {"placing": 2, "player": "DydyGalax#EUW", "mu": 32.08, "sigma": 2.86},
    {"placing": 2, "player": "masteryik01#EUW", "mu": 35.73, "sigma": 2.23},
    {"placing": 3, "player": "happyprov3#EUW", "mu": 37.43, "sigma": 2.16},
    {"placing": 3, "player": "ItsTr0piCs#DADDY", "mu": 34.46, "sigma": 2.61},
    {"placing": 4, "player": "MrChakz#EUW", "mu": 43.11, "sigma": 2.32},
    {"placing": 4, "player": "Windwave#EUW", "mu": 36.77, "sigma": 2.37},
    {"placing": 5, "player": "Gorgu#4MMC", "mu": 31.30, "sigma": 2.24},
    {"placing": 5, "player": "kijke je wukong#EUW", "mu": 39.89, "sigma": 2.23},
    {"placing": 6, "player": "OfCourseYou#EUW", "mu": 30.64, "sigma": 2.73},
    {"placing": 6, "player": "FraSin#EUW", "mu": 36.70, "sigma": 2.35},
    {"placing": 7, "player": "Morify#death", "mu": 42.52, "sigma": 2.19},
    {"placing": 7, "player": "P\u00e2tes au saumon#9299", "mu": 35.79, "sigma": 2.37},
    {"placing": 8, "player": "zabimarou#EUW", "mu": 33.18, "sigma": 2.33},
    {"placing": 8, "player": "sazarene#EUW", "mu": 39.19, "sigma": 2.36},
]

table_views = {}
games_context_menu = None
table_context_menu = None
active_table_key = None

main = ttk.Frame(root, style="Main.TFrame")
main.pack(fill="both", expand=True, padx=14, pady=12)

header = ttk.Frame(main, style="Main.TFrame")
header.pack(fill="x", pady=(0, 10))

title = ttk.Label(header, text="OpenSkill Game Debugger", style="Header.TLabel")
title.pack(anchor="w")
sub = ttk.Label(
    header,
    text="Exact player lookup -> match history picker -> game simulation tables and chart",
    style="Sub.TLabel",
)
sub.pack(anchor="w")
open_experiments_btn = ttk.Button(header, text="Open Experiments")
open_experiments_btn.pack(anchor="e", pady=(6, 0))

controls = ttk.Frame(main, style="Panel.TFrame")
controls.pack(fill="x", pady=(0, 10))
controls.columnconfigure(1, weight=1)

player_label = ttk.Label(controls, text="Player Name")
player_label.grid(row=0, column=0, padx=8, pady=8, sticky="w")
player_var = tk.StringVar()
player_entry = ttk.Entry(controls, textvariable=player_var)
player_entry.grid(row=0, column=1, padx=8, pady=8, sticky="ew")

region_label = ttk.Label(controls, text="Region")
region_label.grid(row=0, column=2, padx=(14, 8), pady=8, sticky="w")
region_var = tk.StringVar(value="euw")
region_combo = ttk.Combobox(controls, textvariable=region_var, values=sorted(REGION_TO_CH_PREFIX.keys()), state="readonly", width=8)
region_combo.grid(row=0, column=3, padx=8, pady=8, sticky="w")

limit_label = ttk.Label(controls, text="Games")
limit_label.grid(row=0, column=4, padx=(14, 8), pady=8, sticky="w")
limit_var = tk.StringVar(value="100")
limit_spin = ttk.Spinbox(controls, from_=1, to=1000, increment=1, textvariable=limit_var, width=7)
limit_spin.grid(row=0, column=5, padx=8, pady=8, sticky="w")

season_label = ttk.Label(controls, text="Season")
season_label.grid(row=0, column=6, padx=(14, 8), pady=8, sticky="w")
season_var = tk.StringVar(value="live")
season_combo = ttk.Combobox(controls, textvariable=season_var, values=["live", "2025season3"], state="readonly", width=12)
season_combo.grid(row=0, column=7, padx=8, pady=8, sticky="w")

search_btn = ttk.Button(controls, text="Search")
search_btn.grid(row=0, column=8, padx=(14, 8), pady=8, sticky="w")

prev_btn = ttk.Button(controls, text="Previous")
prev_btn.grid(row=0, column=9, padx=8, pady=8, sticky="w")

next_btn = ttk.Button(controls, text="Next")
next_btn.grid(row=0, column=10, padx=8, pady=8, sticky="w")

status_wrap = ttk.Frame(main, style="Main.TFrame")
status_wrap.pack(fill="x", pady=(0, 10))
status_var = tk.StringVar(value="Ready")
status = ttk.Label(status_wrap, textvariable=status_var, style="Status.TLabel")
status.pack(side="left")
loading_text_var = tk.StringVar(value="")
loading_text = ttk.Label(status_wrap, textvariable=loading_text_var, style="Sub.TLabel")
loading_spinner = ttk.Progressbar(status_wrap, mode="indeterminate", length=140, style="Loading.Horizontal.TProgressbar")

split = ttk.Panedwindow(main, orient="horizontal")
split.pack(fill="both", expand=True)

left_panel = ttk.Frame(split, style="Panel.TFrame")
right_panel = ttk.Frame(split, style="Panel.TFrame")
split.add(left_panel, weight=1)
split.add(right_panel, weight=3)

left_header = ttk.Frame(left_panel, style="Panel.TFrame")
left_header.pack(fill="x", padx=10, pady=(10, 8))
left_title = ttk.Label(left_header, text="Match History")
left_title.pack(side="left")
expand_match_history_btn = ttk.Button(left_header, text="Expand >>")
expand_match_history_btn.pack(side="right")

games_tree = ttk.Treeview(
    left_panel,
    columns=("game_ts", "placing", "rating_change", "champion"),
    show="headings",
    selectmode="browse",
)
games_tree.heading("game_ts", text="Time", command=lambda: sort_games("game_ts"))
games_tree.heading("placing", text="Place", command=lambda: sort_games("placing"))
games_tree.heading("rating_change", text="Delta", command=lambda: sort_games("rating_change"))
games_tree.heading("champion", text="Champion", command=lambda: sort_games("champion"))
games_tree.column("game_ts", width=240, anchor="w")
games_tree.column("placing", width=60, anchor="center")
games_tree.column("rating_change", width=80, anchor="center")
games_tree.column("champion", width=180, anchor="w")
games_tree.tag_configure("negative", foreground="#ff9da5")
games_tree.tag_configure("positive", foreground="#90e4b7")
games_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

games_scroll = ttk.Scrollbar(games_tree, orient="vertical", command=games_tree.yview)
games_scroll_x = ttk.Scrollbar(left_panel, orient="horizontal", command=games_tree.xview)
games_tree.configure(yscrollcommand=games_scroll.set, xscrollcommand=games_scroll_x.set)
games_scroll.pack(side="right", fill="y")
games_scroll_x.pack(fill="x", padx=10, pady=(0, 8))

notebook = ttk.Notebook(right_panel)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

experiment_main = ttk.Frame(root, style="Main.TFrame")
experiment_header = ttk.Frame(experiment_main, style="Main.TFrame")
experiment_header.pack(fill="x", pady=(0, 10))
experiment_back_btn = ttk.Button(experiment_header, text="Back To Debugger")
experiment_back_btn.pack(side="left")
experiment_title = ttk.Label(experiment_header, text="Experiment Workspace", style="Header.TLabel")
experiment_title.pack(anchor="w", padx=(12, 0))
experiment_sub = ttk.Label(
    experiment_main,
    text="One-off experiments isolated from search/matchhistory flow.",
    style="Sub.TLabel",
)
experiment_sub.pack(anchor="w", pady=(0, 8))
experiment_notebook = ttk.Notebook(experiment_main)
experiment_notebook.pack(fill="both", expand=True)
experiment_sigma_tab = ttk.Frame(experiment_notebook, style="Panel.TFrame")
experiment_notebook.add(experiment_sigma_tab, text="Experiment - Sigma Cap")
experiment_run_btn = ttk.Button(experiment_sigma_tab, text="Run Experiment")
experiment_run_btn.pack(anchor="w", padx=10, pady=(10, 8))
experiment_note_var = tk.StringVar(
    value=(
        "Scenario B override: "
        f"{EXPERIMENT_SIGMA_PLAYER} pregame sigma -> {EXPERIMENT_SIGMA_OVERRIDE:.2f}"
    )
)
experiment_note = ttk.Label(experiment_sigma_tab, textvariable=experiment_note_var, style="Sub.TLabel")
experiment_note.pack(anchor="w", padx=10, pady=(0, 8))
experiment_tables_split = ttk.Panedwindow(experiment_sigma_tab, orient="vertical")
experiment_tables_split.pack(fill="both", expand=True, padx=10, pady=(0, 10))
experiment_base_panel = ttk.Frame(experiment_tables_split, style="Panel.TFrame")
experiment_mod_panel = ttk.Frame(experiment_tables_split, style="Panel.TFrame")
experiment_tables_split.add(experiment_base_panel, weight=1)
experiment_tables_split.add(experiment_mod_panel, weight=1)
experiment_base_title = ttk.Label(experiment_base_panel, text="Baseline (No Sigma Cap / Gap / Unbalanced)", style="Header.TLabel")
experiment_base_title.pack(anchor="w", padx=8, pady=(8, 4))
experiment_mod_title = ttk.Label(experiment_mod_panel, text="Modified Input (Gwrach Sigma Forced To 2.32)", style="Header.TLabel")
experiment_mod_title.pack(anchor="w", padx=8, pady=(8, 4))
experiment_base_tree = ttk.Treeview(experiment_base_panel, show="headings")
experiment_mod_tree = ttk.Treeview(experiment_mod_panel, show="headings")
experiment_base_tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
experiment_mod_tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
experiment_base_scroll = ttk.Scrollbar(experiment_base_tree, orient="vertical", command=experiment_base_tree.yview)
experiment_base_tree.configure(yscrollcommand=experiment_base_scroll.set)
experiment_base_scroll.pack(side="right", fill="y")
experiment_mod_scroll = ttk.Scrollbar(experiment_mod_tree, orient="vertical", command=experiment_mod_tree.yview)
experiment_mod_tree.configure(yscrollcommand=experiment_mod_scroll.set)
experiment_mod_scroll.pack(side="right", fill="y")
experiment_base_tree.tag_configure("negative", foreground="#ff9da5")
experiment_base_tree.tag_configure("positive", foreground="#90e4b7")
experiment_mod_tree.tag_configure("negative", foreground="#ff9da5")
experiment_mod_tree.tag_configure("positive", foreground="#90e4b7")
experiment_mod_tree.tag_configure("changed_player", foreground="#ff4d4f")
team_gap_experiment = install_team_gap_population_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
team_gap_percentiles_experiment = install_team_gap_percentiles_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
team_gap_curve_experiment = install_team_gap_curve_compare_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
unbalanced_lobby_experiment = install_unbalanced_lobby_grace_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
unbalanced_pair_alpha_experiment = install_unbalanced_pair_alpha_compare_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
afk_damage_experiment = install_afk_damage_histogram_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
afk_placing_experiment = install_afk_placing_distribution_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
unbalanced_inflation_solo_safety_experiment = install_unbalanced_inflation_solo_safety_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)
emerald_or_below_placing_boxplot_experiment = install_emerald_or_below_placing_boxplot_tab(
    experiment_notebook=experiment_notebook,
    private_ch=private_ch,
    region_to_ch_prefix=REGION_TO_CH_PREFIX,
    set_status=lambda text: (status_var.set(text), root.update_idletasks()),
    log_console=lambda message: sys.stdout.write(f"{message}\n"),
)


def set_status(text):
    status_var.set(text)
    root.update_idletasks()


def copy_to_clipboard(payload):
    root.clipboard_clear()
    root.clipboard_append(payload)
    root.update()


def row_dict_to_text(row_dict, ordered_keys=None):
    keys = ordered_keys if ordered_keys else list(row_dict.keys())
    return "\n".join(f"{key}: {format_display_value(row_dict.get(key, ''))}" for key in keys)


def row_values_to_text(headers, row_values):
    return "\n".join(f"{header}: {row_values[idx] if idx < len(row_values) else ''}" for idx, header in enumerate(headers))


def game_sort_value(game, key):
    if key == "placing":
        raw = game.get("placing")
        return raw if isinstance(raw, int) else int(raw or 0)
    if key == "rating_change":
        raw = game.get("rating_change")
        return raw if isinstance(raw, (int, float)) else float(raw or 0.0)
    if key == "game_ts":
        return str(game.get("game_ts") or "")
    if key == "champion":
        return str(game.get("champion") or "").lower()
    return str(game.get(key) or "").lower()


def parse_numeric_value(raw):
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    numeric_text = re.sub(r"[^0-9.+-]", "", text)
    if numeric_text in {"", "+", "-", ".", "+.", "-."}:
        return None
    try:
        return float(numeric_text)
    except ValueError:
        return None


def classify_change_tag(headers, row_values):
    negative_found = False
    positive_found = False
    for idx, header in enumerate(headers):
        header_key = str(header).lower()
        if "delta" not in header_key and "change" not in header_key and "penalty" not in header_key and "diff" not in header_key:
            continue
        numeric = parse_numeric_value(row_values[idx] if idx < len(row_values) else None)
        if numeric is None:
            continue
        if numeric < 0:
            negative_found = True
        if numeric > 0:
            positive_found = True
    if negative_found:
        return "negative"
    if positive_found:
        return "positive"
    return ""


def ordered_game_columns():
    ordered = []
    hidden_in_expanded = {"team_gap_pct", "pregame_mu", "pregame_sigma", "pregame_rating", "game_id", "game_date", "player_name", "player_hash"}
    for key in [
        "game_ts",
        "placing",
        "rating_change",
        "champion",
        "avg_worse_opp_rating",
        "avg_better_opp_rating",
        "sigma_cap_scale",
        "team_gap_scale",
        "unbalanced_reduction_pct",
    ]:
        if key not in ordered:
            ordered.append(key)
    for game in state["games"]:
        for key in game.keys():
            if key in hidden_in_expanded:
                continue
            if key not in ordered:
                ordered.append(key)
    return ordered


def configure_games_columns():
    if state["match_history_expanded"]:
        visible = list(state["games_all_columns"]) if state["games_all_columns"] else ["game_ts", "placing", "rating_change", "champion"]
    else:
        visible = ["game_ts", "placing", "rating_change", "champion"]
    state["games_visible_columns"] = visible
    games_tree["columns"] = visible
    for column_key in visible:
        games_tree.heading(column_key, text=column_key, command=lambda c=column_key: sort_games(c))
        width = 180
        anchor = "w"
        if column_key == "game_ts":
            width = 220
        if column_key == "game_date":
            width = 120
        if column_key == "placing":
            width = 80
            anchor = "center"
        if column_key == "rating_change":
            width = 110
            anchor = "center"
        if column_key in {"avg_worse_opp_rating", "avg_better_opp_rating"}:
            width = 175
            anchor = "center"
        if column_key in {"sigma_cap_scale", "team_gap_pct", "team_gap_scale", "unbalanced_reduction_pct", "pregame_mu", "pregame_sigma", "pregame_rating"}:
            width = 160
            anchor = "center"
        if column_key.endswith("_id") or column_key in {"game_id", "puuid"}:
            width = 230
        games_tree.column(column_key, width=width, minwidth=90, stretch=False, anchor=anchor)


def set_match_history_split_position():
    root.update_idletasks()
    total_width = split.winfo_width()
    if total_width <= 0:
        return
    target = int(total_width * 0.74) if state["match_history_expanded"] else int(total_width * 0.34)
    try:
        split.sashpos(0, target)
    except tk.TclError:
        return


def toggle_match_history_expand():
    state["match_history_expanded"] = not state["match_history_expanded"]
    expand_match_history_btn.configure(text="<< Compact" if state["match_history_expanded"] else "Expand >>")
    configure_games_columns()
    render_games_tree()
    set_match_history_split_position()


def render_games_tree():
    games_tree.delete(*games_tree.get_children())
    visible = state["games_visible_columns"]
    for idx, game in enumerate(state["games"]):
        row_values = [format_display_value(game.get(column_key, "")) for column_key in visible]
        row_tag = classify_change_tag(visible, row_values)
        games_tree.insert(
            "",
            "end",
            iid=str(idx),
            values=tuple(row_values),
            tags=(row_tag,) if row_tag else (),
        )


def copy_selected_game_row():
    selected = games_tree.selection()
    if not selected:
        raise RuntimeError("Select a match history row first")
    idx = int(selected[0])
    payload = row_dict_to_text(state["games"][idx])
    copy_to_clipboard(payload)
    log_console(f"[INFO] Copied match history row index={idx} to clipboard")
    set_status("Copied match history row")


def copy_games_table():
    if not state["games"]:
        raise RuntimeError("No match history rows to copy")
    ordered_keys = []
    for game in state["games"]:
        for key in game.keys():
            if key not in ordered_keys:
                ordered_keys.append(key)
    header = "\t".join(ordered_keys)
    rows = ["\t".join(format_display_value(game.get(key, "")) for key in ordered_keys) for game in state["games"]]
    copy_to_clipboard("\n".join([header] + rows))
    log_console(f"[INFO] Copied match history table rows={len(rows)} to clipboard")
    set_status("Copied match history table")


def on_games_right_click(event):
    row_id = games_tree.identify_row(event.y)
    if row_id:
        games_tree.selection_set(row_id)
        games_tree.focus(row_id)
    if games_context_menu is not None:
        games_context_menu.tk_popup(event.x_root, event.y_root)
        games_context_menu.grab_release()


def sort_games(column_key):
    current = state["games_sort_state"].get(column_key, "none")
    reverse = current == "asc"
    state["games"].sort(key=lambda g: game_sort_value(g, column_key), reverse=reverse)
    state["games_sort_state"] = {column_key: "desc" if reverse else "asc"}
    render_games_tree()
    state["game_index"] = -1


def to_sort_value(raw):
    text = str(raw)
    numeric_text = re.sub(r"[^0-9.+-]", "", text)
    if numeric_text in {"", "+", "-", ".", "+.", "-."}:
        return text.lower()
    try:
        return float(numeric_text)
    except ValueError:
        return text.lower()


def format_display_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, str):
        return value
    return str(value)


def build_experiment_requested_rows(players_input):
    model = instantiate_rating_model()
    grouped_by_place = {}
    for row in players_input:
        grouped_by_place.setdefault(int(row["placing"]), []).append(row)
    ordered_places = sorted(grouped_by_place.keys())
    if ordered_places != [1, 2, 3, 4, 5, 6, 7, 8]:
        raise ValueError("Experiment input must contain places 1..8 exactly once each as teams.")
    teams = []
    for place in ordered_places:
        members = grouped_by_place[place]
        if len(members) != 2:
            raise ValueError(f"Experiment place {place} does not have exactly 2 players.")
        teams.append((place, members))

    before_ratings = {}
    for _, members in teams:
        for member in members:
            before_ratings[member["player"]] = model.rating(mu=float(member["mu"]), sigma=float(member["sigma"]))

    team_rate_input = []
    for _, members in teams:
        team_rate_input.append([before_ratings[members[0]["player"]], before_ratings[members[1]["player"]]])
    rated_teams = model.rate(team_rate_input, ranks=list(range(8)))

    after_ratings = dict(before_ratings)
    for idx, (_, members) in enumerate(teams):
        after_ratings[members[0]["player"]] = rated_teams[idx][0]
        after_ratings[members[1]["player"]] = rated_teams[idx][1]

    headers = [
        "placing",
        "player",
        "pregame_player_stats",
        "pregame_team_stats",
        "postgame_team_stats",
        "postgame_player_stats",
        "rating_change",
    ]
    rows = []
    for place, members in teams:
        b0 = before_ratings[members[0]["player"]]
        b1 = before_ratings[members[1]["player"]]
        a0 = after_ratings[members[0]["player"]]
        a1 = after_ratings[members[1]["player"]]
        pre_team_mu = b0.mu + b1.mu
        pre_team_sigma = math.hypot(b0.sigma, b1.sigma)
        pre_team_rating = (pre_team_mu - 3.0 * pre_team_sigma) * 75.0
        post_team_mu = a0.mu + a1.mu
        post_team_sigma = math.hypot(a0.sigma, a1.sigma)
        post_team_rating = (post_team_mu - 3.0 * post_team_sigma) * 75.0
        for idx, member in enumerate(members):
            before = before_ratings[member["player"]]
            after = after_ratings[member["player"]]
            before_display = calculate_rating(before)
            after_display = calculate_rating(after)
            pre_player_str = f"{before.mu:.2f} {before.sigma:.2f} ({before_display})"
            post_player_str = f"{after.mu:.2f} {after.sigma:.2f} ({after_display})"
            pre_team_str = f"{pre_team_mu:.2f} {pre_team_sigma:.2f} ({pre_team_rating:.2f})" if idx == 0 else ""
            post_team_str = f"{post_team_mu:.2f} {post_team_sigma:.2f} ({post_team_rating:.2f})" if idx == 0 else ""
            rows.append(
                [
                    str(place),
                    member["player"],
                    pre_player_str,
                    pre_team_str,
                    post_team_str,
                    post_player_str,
                    f"{(after_display - before_display):+d}",
                ]
            )
    return headers, rows


def render_experiment_tree(tree, headers, rows, highlight_player=None):
    tree.delete(*tree.get_children())
    tree["columns"] = headers
    for header in headers:
        tree.heading(header, text=header)
        tree.column(header, width=210, anchor="w")
    for idx, row in enumerate(rows):
        values = [format_display_value(cell) for cell in row]
        row_tag = classify_change_tag(headers, values)
        if highlight_player and len(values) > 1 and values[1] == highlight_player:
            tags = ("changed_player", row_tag) if row_tag else ("changed_player",)
        else:
            tags = (row_tag,) if row_tag else ()
        tree.insert("", "end", iid=f"exp_{id(tree)}_{idx}", values=values, tags=tags)


def run_sigma_cap_experiment():
    base_players = [dict(row) for row in EXPERIMENT_SIGMA_PLAYERS]
    modified_players = [dict(row) for row in EXPERIMENT_SIGMA_PLAYERS]
    changed = False
    for row in modified_players:
        if row["player"] == EXPERIMENT_SIGMA_PLAYER:
            row["sigma"] = EXPERIMENT_SIGMA_OVERRIDE
            changed = True
            break
    if not changed:
        raise ValueError(f"Experiment player not found: {EXPERIMENT_SIGMA_PLAYER}")

    base_headers, base_rows = build_experiment_requested_rows(base_players)
    mod_headers, mod_rows = build_experiment_requested_rows(modified_players)
    render_experiment_tree(experiment_base_tree, base_headers, base_rows)
    render_experiment_tree(experiment_mod_tree, mod_headers, mod_rows, highlight_player=EXPERIMENT_SIGMA_PLAYER)


def apply_table_filter(table_key):
    view = table_views[table_key]
    tree = view["tree"]
    query = view["filter_var"].get().strip().lower()
    tree.delete(*tree.get_children())
    rows = view["rows"]
    if query:
        rows = [row for row in rows if query in " | ".join(str(cell).lower() for cell in row)]
    for idx, row in enumerate(rows):
        row_tag = classify_change_tag(view["headers"], row)
        tree.insert("", "end", iid=f"{table_key}_{idx}", values=row, tags=(row_tag,) if row_tag else ())


def copy_selected_table_row(table_key):
    view = table_views[table_key]
    tree = view["tree"]
    selected = tree.selection()
    if not selected:
        raise RuntimeError("Select a table row first")
    row_values = list(tree.item(selected[0], "values"))
    payload = row_values_to_text(view["headers"], row_values)
    copy_to_clipboard(payload)
    log_console(f"[INFO] Copied row from table={table_key} to clipboard")
    set_status(f"Copied row from {table_key}")


def copy_table(table_key):
    view = table_views[table_key]
    headers = view["headers"]
    if not headers:
        raise RuntimeError(f"No headers in table {table_key}")
    rows = view["rows"]
    text_rows = ["\t".join(headers)] + ["\t".join(str(cell) for cell in row) for row in rows]
    copy_to_clipboard("\n".join(text_rows))
    log_console(f"[INFO] Copied table={table_key} rows={len(rows)} to clipboard")
    set_status(f"Copied table {table_key}")


def on_table_right_click(event, table_key):
    tree = table_views[table_key]["tree"]
    row_id = tree.identify_row(event.y)
    if row_id:
        tree.selection_set(row_id)
        tree.focus(row_id)
    global active_table_key
    active_table_key = table_key
    if table_context_menu is not None:
        table_context_menu.tk_popup(event.x_root, event.y_root)
        table_context_menu.grab_release()


def copy_active_table_row():
    if not active_table_key:
        raise RuntimeError("No active table selected")
    copy_selected_table_row(active_table_key)


def copy_active_table():
    if not active_table_key:
        raise RuntimeError("No active table selected")
    copy_table(active_table_key)


def sort_table(table_key, col_name):
    view = table_views[table_key]
    col_idx = view["headers"].index(col_name)
    current = view["sort_state"].get(col_name, "none")
    reverse = current == "asc"
    view["rows"].sort(key=lambda row: to_sort_value(row[col_idx]), reverse=reverse)
    view["sort_state"][col_name] = "asc" if not reverse else "desc"
    apply_table_filter(table_key)


def render_table(table_key, headers, rows):
    view = table_views[table_key]
    tree = view["tree"]
    view["headers"] = list(headers)
    view["rows"] = [[format_display_value(cell) for cell in row] for row in rows]
    view["sort_state"] = {}
    tree.delete(*tree.get_children())
    tree["columns"] = headers
    for header in headers:
        tree.heading(header, text=header, command=lambda c=header: sort_table(table_key, c))
        tree.column(header, width=180, anchor="w")
    tree.tag_configure("negative", foreground="#ff9da5")
    tree.tag_configure("positive", foreground="#90e4b7")
    apply_table_filter(table_key)


def make_table_tab(tab_key, title_text):
    frame = ttk.Frame(notebook, style="Panel.TFrame")
    notebook.add(frame, text=title_text)

    filter_wrap = ttk.Frame(frame, style="Panel.TFrame")
    filter_wrap.pack(fill="x", padx=8, pady=(8, 4))
    ttk.Label(filter_wrap, text="Filter:").pack(side="left")
    filter_var = tk.StringVar()
    filter_entry = ttk.Entry(filter_wrap, textvariable=filter_var)
    filter_entry.pack(side="left", fill="x", expand=True, padx=8)
    clear_btn = ttk.Button(filter_wrap, text="Clear", command=lambda: (filter_var.set(""), apply_table_filter(tab_key)))
    clear_btn.pack(side="left")

    tree = ttk.Treeview(frame, show="headings")
    tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    yscroll = ttk.Scrollbar(tree, orient="vertical", command=tree.yview)
    xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    yscroll.pack(side="right", fill="y")
    xscroll.pack(fill="x", padx=8, pady=(0, 8))

    table_views[tab_key] = {
        "tree": tree,
        "filter_var": filter_var,
        "headers": [],
        "rows": [],
        "sort_state": {},
    }
    filter_var.trace_add("write", lambda *_: apply_table_filter(tab_key))
    tree.bind("<Button-3>", lambda event, key=tab_key: on_table_right_click(event, key))
    tree.bind("<Button-2>", lambda event, key=tab_key: on_table_right_click(event, key))


make_table_tab("players", "Players")
make_table_tab("requested_summary", "Requested")
make_table_tab("sigma_cap_summary", "Sigma Cap")
make_table_tab("gap_penalty_summary", "Gap Penalty")
make_table_tab("unbalanced_lobby_summary", "Unbalanced")
make_table_tab("stacked_penalty_summary", "Stacked")

charts_tab = ttk.Frame(notebook, style="Panel.TFrame")
notebook.add(charts_tab, text="Charts")
charts_note = ttk.Label(
    charts_tab,
    text="Select this tab to open full matplotlib charts in popup windows.",
)
charts_note.pack(anchor="w", padx=10, pady=10)

log_tab = ttk.Frame(notebook, style="Panel.TFrame")
notebook.add(log_tab, text="Run Log")
log_text = tk.Text(log_tab, bg="#111a26", fg="#e4eefb", insertbackground="#ffffff", font=("Segoe UI", 11), wrap="word")
log_text.pack(fill="both", expand=True, padx=8, pady=8)
games_context_menu = tk.Menu(root, tearoff=0)
games_context_menu.add_command(label="Copy Row (All Fields)", command=copy_selected_game_row)
games_context_menu.add_command(label="Copy Match History Table", command=copy_games_table)
table_context_menu = tk.Menu(root, tearoff=0)
table_context_menu.add_command(label="Copy Row (All Fields)", command=copy_active_table_row)
table_context_menu.add_command(label="Copy Table", command=copy_active_table)


def log_console(message, is_error=False):
    stream = sys.stderr if is_error else sys.stdout
    stream.write(f"{message}\n")
    stream.flush()


def log_exception(context, exc):
    log_console(f"[ERROR] {context}: {exc}", is_error=True)
    tb = traceback.format_exc()
    if tb and tb.strip() != "NoneType: None":
        log_console(tb.rstrip(), is_error=True)


def render_report(report):
    player_headers = ["player_name", "mu_before", "mu_after", "delta_mu", "sigma_before", "sigma_after", "delta_sigma"]
    player_rows = [
        [
            row.get("player_name", row.get("player")),
            row.get("mu_before"),
            row.get("mu_after"),
            row.get("delta_mu"),
            row.get("sigma_before"),
            row.get("sigma_after"),
            row.get("delta_sigma"),
        ]
        for row in report.get("per_player_changes", [])
    ]
    render_table("players", player_headers, player_rows)

    for key in [
        "requested_summary",
        "sigma_cap_summary",
        "gap_penalty_summary",
        "unbalanced_lobby_summary",
        "stacked_penalty_summary",
    ]:
        block = report.get("tables", {}).get(key, {})
        render_table(key, block.get("headers", []), block.get("rows", []))



def set_loading_state(is_loading, context_text=""):
    state["is_loading"] = is_loading
    loading_text_var.set(context_text if is_loading else "")
    if is_loading:
        loading_text.pack(side="right", padx=(8, 8))
        loading_spinner.pack(side="right")
        loading_spinner.start(10)
    else:
        loading_spinner.stop()
        loading_spinner.pack_forget()
        loading_text.pack_forget()
    for widget in [player_entry, region_combo, limit_spin, season_combo, search_btn, prev_btn, next_btn, expand_match_history_btn]:
        if is_loading:
            widget.state(["disabled"])
        else:
            widget.state(["!disabled"])
    if is_loading:
        games_tree.state(["disabled"])
    else:
        games_tree.state(["!disabled"])


def run_sim_for_game(game_id):
    region = state["region"]
    season = state["season"]
    ch_prefix = REGION_TO_CH_PREFIX[region]
    log_console(
        f"[INFO] Loading sim input from ClickHouse game_id={game_id} region={region} season={season} ch_prefix={ch_prefix}"
    )
    sim_input = private_ch.load_sim_input_from_clickhouse(game_id=game_id, region=region, ch_prefix=ch_prefix, season=season)

    if state["tmp_dir"] is None:
        state["tmp_dir"] = tempfile.mkdtemp(prefix="openskill_sim_app_")
    state["input_path"] = os.path.join(state["tmp_dir"], f"sim_input_{game_id}.json")
    state["report_path"] = os.path.join(state["tmp_dir"], f"report_{game_id}.json")

    with open(state["input_path"], "w", encoding="utf-8") as f:
        json.dump(sim_input, f, indent=2)
        f.write("\n")

    sim_script = os.path.join(_SCRIPT_DIR, "openskill_sim.py")
    cmd = [
        sys.executable,
        sim_script,
        "--input",
        state["input_path"],
        "--export-report",
        state["report_path"],
        "--no-charts",
    ]
    log_console(f"[INFO] Running sim command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_text.delete("1.0", "end")
    log_text.insert("1.0", combined.strip())
    if proc.stdout:
        log_console("[SIM STDOUT]")
        log_console(proc.stdout.rstrip())
    if proc.stderr:
        log_console("[SIM STDERR]", is_error=True)
        log_console(proc.stderr.rstrip(), is_error=True)
    if proc.returncode != 0:
        log_console(f"[ERROR] Sim command failed with code {proc.returncode}", is_error=True)
        raise RuntimeError(f"openskill_sim.py failed (code {proc.returncode}). See Run Log tab for details.")

    with open(state["report_path"], "r", encoding="utf-8") as f:
        report = json.load(f)
    log_console(f"[INFO] Loaded report: {state['report_path']}")
    render_report(report)



def select_game_index(index):
    if not state["games"]:
        return
    if index < 0 or index >= len(state["games"]):
        return
    state["game_index"] = index
    game = state["games"][index]
    games_tree.selection_set(str(index))
    games_tree.focus(str(index))
    games_tree.see(str(index))
    set_loading_state(True, f"Loading game {game['game_id']}...")
    try:
        set_status(f"Loading game {index + 1}/{len(state['games'])}: {game['game_id']}")
        run_sim_for_game(game["game_id"])
        set_status(f"Loaded game {game['game_id']} for {state['player_name']} [{state['region']}] ({state['season']})")
    finally:
        set_loading_state(False)



def on_game_select(_event=None):
    if state["is_loading"]:
        return
    selected = games_tree.selection()
    if not selected:
        return
    idx = int(selected[0])
    if idx == state["game_index"]:
        return
    try:
        select_game_index(idx)
    except Exception as exc:
        log_exception("on_game_select failed", exc)
        messagebox.showerror("Load Failed", str(exc))
        set_status("Load failed")



def search_games(_event=None):
    try:
        player_name = player_var.get().strip()
        if not player_name:
            raise ValueError("Player name is required")
        region = region_var.get().strip().lower()
        if region not in REGION_TO_CH_PREFIX:
            raise ValueError(f"Unknown region: {region}")
        season = season_var.get().strip().lower()
        if season not in {"live", "2025season3"}:
            raise ValueError(f"Unknown season: {season}")
        try:
            limit = int(limit_var.get().strip())
        except ValueError as exc:
            raise ValueError("Games limit must be an integer") from exc
        if limit <= 0:
            raise ValueError("Games limit must be > 0")

        state["player_name"] = player_name
        state["region"] = region
        state["season"] = season

        set_status(f"Fetching games for {player_name} [{region}] ({season})...")
        ch_prefix = REGION_TO_CH_PREFIX[region]
        log_console(
            f"[INFO] Searching games player_name={player_name} region={region} season={season} limit={limit}"
        )
        games = private_ch.list_player_games_exact(
            player_name=player_name,
            region=region,
            ch_prefix=ch_prefix,
            limit=limit,
            season=season,
        )

        state["games"] = games
        state["games_all_columns"] = ordered_game_columns()
        state["game_index"] = -1
        state["games_sort_state"] = {}
        configure_games_columns()
        render_games_tree()
        set_match_history_split_position()

        if not games:
            raise ValueError("No games returned")
        select_game_index(0)
    except Exception as exc:
        log_exception("search_games failed", exc)
        messagebox.showerror("Search Failed", str(exc))
        set_status("Search failed")



def prev_game():
    if state["is_loading"]:
        return
    if state["game_index"] <= 0:
        return
    try:
        select_game_index(state["game_index"] - 1)
    except Exception as exc:
        log_exception("prev_game failed", exc)
        messagebox.showerror("Load Failed", str(exc))
        set_status("Load failed")



def next_game():
    if state["is_loading"]:
        return
    if state["game_index"] < 0 or state["game_index"] >= len(state["games"]) - 1:
        return
    try:
        select_game_index(state["game_index"] + 1)
    except Exception as exc:
        log_exception("next_game failed", exc)
        messagebox.showerror("Load Failed", str(exc))
        set_status("Load failed")



def open_full_charts():
    if not state["input_path"] or not os.path.exists(state["input_path"]):
        messagebox.showerror("Missing Input", "Load a game first before opening charts")
        return
    sim_script = os.path.join(_SCRIPT_DIR, "openskill_sim.py")
    cmd = [sys.executable, sim_script, "--input", state["input_path"]]
    log_console(f"[INFO] Launching full charts command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


def on_tab_changed(_event=None):
    selected = notebook.select()
    if not selected:
        return
    tab_text = notebook.tab(selected, "text")
    if tab_text == "Charts":
        open_full_charts()


def open_experiments_view():
    try:
        run_sigma_cap_experiment()
        main.pack_forget()
        experiment_main.pack(fill="both", expand=True, padx=14, pady=12)
        selected = experiment_notebook.select()
        if not selected:
            return
        selected_text = experiment_notebook.tab(selected, "text")
        if team_gap_experiment and selected_text == team_gap_experiment["tab_text"]:
            team_gap_experiment["run_query"]()
        if unbalanced_lobby_experiment and selected_text == unbalanced_lobby_experiment["tab_text"]:
            unbalanced_lobby_experiment["run_query"]()
        if afk_damage_experiment and selected_text == afk_damage_experiment["tab_text"]:
            afk_damage_experiment["run_query"]()
        if afk_placing_experiment and selected_text == afk_placing_experiment["tab_text"]:
            afk_placing_experiment["run_query"]()
        if emerald_or_below_placing_boxplot_experiment and selected_text == emerald_or_below_placing_boxplot_experiment["tab_text"]:
            emerald_or_below_placing_boxplot_experiment["run_query"]()
    except Exception as exc:
        log_exception("open_experiments_view failed", exc)
        messagebox.showerror("Experiment Failed", str(exc))


def close_experiments_view():
    experiment_main.pack_forget()
    main.pack(fill="both", expand=True, padx=14, pady=12)


def on_experiment_tab_changed(_event=None):
    selected = experiment_notebook.select()
    if not selected:
        return
    if not experiment_main.winfo_ismapped():
        return
    selected_text = experiment_notebook.tab(selected, "text")
    try:
        if team_gap_experiment and selected_text == team_gap_experiment["tab_text"]:
            team_gap_experiment["run_query"]()
        if unbalanced_lobby_experiment and selected_text == unbalanced_lobby_experiment["tab_text"]:
            unbalanced_lobby_experiment["run_query"]()
        if afk_damage_experiment and selected_text == afk_damage_experiment["tab_text"]:
            afk_damage_experiment["run_query"]()
        if afk_placing_experiment and selected_text == afk_placing_experiment["tab_text"]:
            afk_placing_experiment["run_query"]()
        if emerald_or_below_placing_boxplot_experiment and selected_text == emerald_or_below_placing_boxplot_experiment["tab_text"]:
            emerald_or_below_placing_boxplot_experiment["run_query"]()
    except Exception as exc:
        log_exception("on_experiment_tab_changed failed", exc)
        messagebox.showerror("Experiment Failed", str(exc))


def on_tk_exception(exc_type, exc_value, exc_tb):
    log_console("[ERROR] Unhandled Tkinter callback exception", is_error=True)
    log_console("".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip(), is_error=True)
    messagebox.showerror("Unhandled Error", str(exc_value))


root.report_callback_exception = on_tk_exception

search_btn.configure(command=search_games)
prev_btn.configure(command=prev_game)
next_btn.configure(command=next_game)
expand_match_history_btn.configure(command=toggle_match_history_expand)
open_experiments_btn.configure(command=open_experiments_view)
experiment_back_btn.configure(command=close_experiments_view)
experiment_run_btn.configure(command=run_sigma_cap_experiment)
player_entry.bind("<Return>", search_games)
games_tree.bind("<<TreeviewSelect>>", on_game_select)
games_tree.bind("<Button-3>", on_games_right_click)
games_tree.bind("<Button-2>", on_games_right_click)
notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
experiment_notebook.bind("<<NotebookTabChanged>>", on_experiment_tab_changed)

configure_games_columns()
set_match_history_split_position()
player_entry.focus_set()
root.mainloop()
