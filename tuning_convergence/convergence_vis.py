import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.widgets import Slider
# --- 1. Load Data ---
# Load the new CSV file provided
df = pd.read_csv('tuning_convergence/dataset.csv')
df.rename(columns={'label': 'target'}, inplace=True)
df['target'] = df['target'].fillna(0)
df['target'] = df['target'].astype(int)
# --- 2. Data Preparation (Initial) ---
sigma_spread = 8.33 - 1.5  # Fixed: 6.83, initial to steady-state sigma
df['diff_sigma'] = np.abs(df['player_sigma'] - df['teammate_sigma']) / sigma_spread
# --- 3. Identify Top Players and Create Mappings ---
all_players = pd.concat([df['player_name'], df['teammate_name']])
top_5_players = all_players.value_counts().nlargest(5).index.tolist()
markers = ['^', 'D', '*', 's', 'p']  # Triangle, Diamond, Star, Square, Pentagon
marker_map = {player: marker for player, marker in zip(top_5_players, markers)}
# --- 4. Generate Grid Data ---
d_sigma = np.linspace(0, 1, 100)
d_mu = np.linspace(0, 1, 100)
grid_sigma, grid_mu = np.meshgrid(d_sigma, d_mu)
grid_data = pd.DataFrame({'diff_mu': grid_mu.ravel(), 'diff_sigma': grid_sigma.ravel()})
# --- Bias Calculation Function ---
def calculate_bias_vec(data, mu_threshold, sigma_threshold, mu_scale_end, bias_min, bias_max, sigma_penalty):
    d_mu = data['diff_mu'].values
    d_sigma = data['diff_sigma'].values
    bias = np.zeros_like(d_mu)
    mask_sigma_only = (d_mu < mu_threshold) & (d_sigma >= sigma_threshold)
    bias[mask_sigma_only] = sigma_penalty
    mask_mu = d_mu >= mu_threshold
    denominator = mu_scale_end - mu_threshold
    fraction = np.zeros_like(d_mu[mask_mu])
    if denominator > 0:
        fraction = np.minimum(1.0, (d_mu[mask_mu] - mu_threshold) / denominator)
    bias[mask_mu] = bias_min + fraction * (bias_max - bias_min)
    return bias
# --- Initial Parameters ---
initial_mu_threshold = 0.6
initial_sigma_threshold = 0.5
initial_mu_scale_end = 0.8
initial_bias_min = 0.75
initial_bias_max = 0.95
initial_min_mu_diff = 30
initial_sigma_penalty = 0.8
# --- Create Figure and Axes ---
fig, ax = plt.subplots(figsize=(16, 12))
plt.subplots_adjust(bottom=0.40, right=0.85)  # Leave more space for sliders
# Colorbar (created after initial plot)
cbar = None
# --- Function to Plot Contents ---
def plot_contents(min_mu_diff, mu_threshold, sigma_threshold, mu_scale_end, bias_min, bias_max, sigma_penalty):
    global cbar
    ax.cla()  # Clear axis for replot
   
    # Recompute diff_mu based on current min_mu_diff
    diff_mu_list = []
    for idx, row in df.iterrows():
        mu1 = row['player_mu']
        mu2 = row['teammate_mu']
        weaker_mu = min(mu1, mu2)
        stronger_mu = max(mu1, mu2)
        abs_mu_diff = stronger_mu - weaker_mu
        if abs_mu_diff < min_mu_diff:
            diff_mu = 0.0
        else:
            diff_mu = 1 - (abs(weaker_mu) / abs_mu_diff)
            diff_mu = min(1.0, max(0.0, diff_mu))
        diff_mu_list.append(diff_mu)
    df['diff_mu'] = diff_mu_list
   
    # Compute bias grid
    bias = calculate_bias_vec(
        grid_data, mu_threshold, sigma_threshold, mu_scale_end, bias_min, bias_max, sigma_penalty
    )
    bias_grid = bias.reshape(100, 100)
   
    # Plot image
    global im
    im = ax.imshow(
        bias_grid, cmap="viridis", extent=[0, 1, 0, 1], origin='lower',
        aspect='auto', interpolation='bilinear', vmin=0
    )
    im.set_clim(0, np.max(bias_grid))
   
    # Recreate colorbar if first time
    if cbar is None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Bias Value', rotation=270, labelpad=15)
    else:
        cbar.update_normal(im)  # Update existing colorbar
   
    # Scatter points
    for idx, row in df.iterrows():
        color = 'lime' if row['target'] == 1 else 'red'
        player1 = row['player_name']
        player2 = row['teammate_name']
        marker = 'o'
        size = 30
        # Prioritize player_name for marker if both are top players
        if player1 in marker_map:
            marker = marker_map[player1]
            size = 80
        elif player2 in marker_map:
            marker = marker_map[player2]
            size = 80
        ax.scatter(
            row['diff_sigma'], row['diff_mu'],
            marker=marker, color=color,
            edgecolor='white', s=size, alpha=0.9, linewidths=0.5
        )
   
    # --- Create and Position Custom Legend ---
    legend_elements = []
    for player, marker in marker_map.items():
        legend_elements.append(mlines.Line2D([], [], color='grey', marker=marker, linestyle='None',
                                             markersize=10, label=player))
    legend_elements.append(mlines.Line2D([], [], color='white', marker='', linestyle='None', label=''))
    legend_elements.append(mlines.Line2D([], [], color='lime', marker='o', linestyle='None',
                                         markersize=10, label='Boosted match'))
    legend_elements.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                         markersize=10, label='Standard match'))
    ax.legend(handles=legend_elements, title="Legend", loc='upper right')
   
    # Titles and labels
    ax.set_title('Calculated Bias with Detailed Data Overlay (New Dataset)', fontsize=16)
    ax.set_xlabel('diff_sigma (Normalized Sigma Difference)', fontsize=12)
    ax.set_ylabel('diff_mu (Normalized Mu Difference)', fontsize=12)
# Initial plot
plot_contents(initial_min_mu_diff, initial_mu_threshold, initial_sigma_threshold,
              initial_mu_scale_end, initial_bias_min, initial_bias_max, initial_sigma_penalty)
# --- Add Sliders ---
axcolor = 'lightgoldenrodyellow'
ax_sigma_thresh = plt.axes([0.1, 0.35, 0.65, 0.03], facecolor=axcolor)
ax_sigma_penalty = plt.axes([0.1, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_mu_thresh = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_mu_end = plt.axes([0.1, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_bias_min = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_bias_max = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_min_mu = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
s_sigma_thresh = Slider(ax_sigma_thresh, 'Sigma Threshold', 0.0, 1.0, valinit=initial_sigma_threshold)
s_sigma_penalty = Slider(ax_sigma_penalty, 'Sigma Penalty', 0.0, 1.0, valinit=initial_sigma_penalty)
s_mu_thresh = Slider(ax_mu_thresh, 'Mu Threshold', 0.0, 1.0, valinit=initial_mu_threshold)
s_mu_end = Slider(ax_mu_end, 'Mu Scale End', 0.0, 1.0, valinit=initial_mu_scale_end)
s_bias_min = Slider(ax_bias_min, 'Bias Min', 0.0, 1.0, valinit=initial_bias_min)
s_bias_max = Slider(ax_bias_max, 'Bias Max', 0.0, 1.0, valinit=initial_bias_max)
s_min_mu = Slider(ax_min_mu, 'Min Mu Diff', 0.0, 50.0, valinit=initial_min_mu_diff)
# --- Update Function ---
def update(val):
    sigma_threshold = s_sigma_thresh.val
    sigma_penalty = s_sigma_penalty.val
    mu_threshold = s_mu_thresh.val
    mu_scale_end = s_mu_end.val
    bias_min = s_bias_min.val
    bias_max = s_bias_max.val
    min_mu_diff = s_min_mu.val
    plot_contents(min_mu_diff, mu_threshold, sigma_threshold,
                  mu_scale_end, bias_min, bias_max, sigma_penalty)
    fig.canvas.draw_idle()
# Attach update to sliders
s_sigma_thresh.on_changed(update)
s_sigma_penalty.on_changed(update)
s_mu_thresh.on_changed(update)
s_mu_end.on_changed(update)
s_bias_min.on_changed(update)
s_bias_max.on_changed(update)
s_min_mu.on_changed(update)
# Adjust layout
fig.tight_layout(rect=[0, 0.40, 0.85, 1])  # Make space for sliders and legend
plt.show()