#!/usr/bin/env python3
import importlib
import json
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

root = tk.Tk()
root.title("ArenaSweats OpenSkill Debugger")
root.geometry("1600x930")
root.configure(bg="#111418")

style = ttk.Style()
style.theme_use("clam")
style.configure("Main.TFrame", background="#111418")
style.configure("Panel.TFrame", background="#1b2129")
style.configure("Header.TLabel", background="#111418", foreground="#f2f5f8", font=("Segoe UI", 18, "bold"))
style.configure("Sub.TLabel", background="#111418", foreground="#9db0c7", font=("Segoe UI", 10))
style.configure("Status.TLabel", background="#111418", foreground="#7de3b6", font=("Segoe UI", 10))
style.configure("TLabel", background="#1b2129", foreground="#e6ebf2", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, background="#2a3340", foreground="#f2f5f8")
style.map("TButton", background=[("active", "#3a4656")], foreground=[("active", "#ffffff")])
style.configure("TEntry", padding=5, fieldbackground="#151b23", foreground="#e6ebf2")
style.configure("TCombobox", fieldbackground="#151b23", foreground="#e6ebf2")
style.configure("Treeview", font=("Consolas", 10), rowheight=26, fieldbackground="#151b23", background="#151b23", foreground="#e6ebf2")
style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#273343", foreground="#f2f5f8")
style.map("Treeview", background=[("selected", "#385170")], foreground=[("selected", "#ffffff")])
style.configure("TNotebook", background="#1b2129", borderwidth=0)
style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(10, 6), background="#273343", foreground="#d9e3ef")
style.map("TNotebook.Tab", background=[("selected", "#1b2129")], foreground=[("selected", "#ffffff")])

state = {
    "games": [],
    "game_index": -1,
    "games_sort_state": {},
    "player_name": "",
    "region": "euw",
    "season": "live",
    "tmp_dir": None,
    "input_path": None,
    "report_path": None,
}

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

status_var = tk.StringVar(value="Ready")
status = ttk.Label(main, textvariable=status_var, style="Status.TLabel")
status.pack(fill="x", pady=(0, 10))

split = ttk.Panedwindow(main, orient="horizontal")
split.pack(fill="both", expand=True)

left_panel = ttk.Frame(split, style="Panel.TFrame")
right_panel = ttk.Frame(split, style="Panel.TFrame")
split.add(left_panel, weight=1)
split.add(right_panel, weight=3)

left_title = ttk.Label(left_panel, text="Match History", style="Header.TLabel")
left_title.pack(anchor="w", padx=10, pady=(10, 8))

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
games_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

games_scroll = ttk.Scrollbar(games_tree, orient="vertical", command=games_tree.yview)
games_tree.configure(yscrollcommand=games_scroll.set)
games_scroll.pack(side="right", fill="y")

notebook = ttk.Notebook(right_panel)
notebook.pack(fill="both", expand=True, padx=10, pady=10)


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


def render_games_tree():
    games_tree.delete(*games_tree.get_children())
    for idx, game in enumerate(state["games"]):
        games_tree.insert(
            "",
            "end",
            iid=str(idx),
            values=(
                format_display_value(game.get("game_ts", "")),
                format_display_value(game.get("placing", "")),
                format_display_value(game.get("rating_change", "")),
                format_display_value(game.get("champion", "")),
            ),
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
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+\.\d+", stripped):
            return f"{float(stripped):.2f}"
        return value
    return str(value)


def apply_table_filter(table_key):
    view = table_views[table_key]
    tree = view["tree"]
    query = view["filter_var"].get().strip().lower()
    tree.delete(*tree.get_children())
    rows = view["rows"]
    if query:
        rows = [row for row in rows if query in " | ".join(str(cell).lower() for cell in row)]
    for idx, row in enumerate(rows):
        tree.insert("", "end", iid=f"{table_key}_{idx}", values=row)


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
    tree.configure(yscrollcommand=yscroll.set)
    yscroll.pack(side="right", fill="y")

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
log_text = tk.Text(log_tab, bg="#151b23", fg="#d9e3ef", insertbackground="#ffffff", font=("Consolas", 10), wrap="word")
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
    set_status(f"Loading game {index + 1}/{len(state['games'])}: {game['game_id']}")
    run_sim_for_game(game["game_id"])
    set_status(f"Loaded game {game['game_id']} for {state['player_name']} [{state['region']}] ({state['season']})")



def on_game_select(_event=None):
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
        state["game_index"] = -1
        state["games_sort_state"] = {}
        render_games_tree()

        if not games:
            raise ValueError("No games returned")
        select_game_index(0)
    except Exception as exc:
        log_exception("search_games failed", exc)
        messagebox.showerror("Search Failed", str(exc))
        set_status("Search failed")



def prev_game():
    if state["game_index"] <= 0:
        return
    try:
        select_game_index(state["game_index"] - 1)
    except Exception as exc:
        log_exception("prev_game failed", exc)
        messagebox.showerror("Load Failed", str(exc))
        set_status("Load failed")



def next_game():
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


def on_tk_exception(exc_type, exc_value, exc_tb):
    log_console("[ERROR] Unhandled Tkinter callback exception", is_error=True)
    log_console("".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip(), is_error=True)
    messagebox.showerror("Unhandled Error", str(exc_value))


root.report_callback_exception = on_tk_exception

search_btn.configure(command=search_games)
prev_btn.configure(command=prev_game)
next_btn.configure(command=next_game)
player_entry.bind("<Return>", search_games)
games_tree.bind("<<TreeviewSelect>>", on_game_select)
games_tree.bind("<Button-3>", on_games_right_click)
games_tree.bind("<Button-2>", on_games_right_click)
notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

player_entry.focus_set()
root.mainloop()
