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
# --- 2. Data Preparation ---
mu_spread = df['mu_spread'].iloc[0]
sigma_spread = df['player_sigma'].max() - df['player_sigma'].min()
df['diff_mu'] = np.abs(df['player_mu'] - df['teammate_mu']) / mu_spread
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
def calculate_bias_vec_balanced(data, blend_p, steepness, midpoint, convergence_strength, max_deviation, skip_threshold):
    gap_score = blend_p * data['diff_mu'] + (1 - blend_p) * data['diff_sigma']
    sigmoid_val = 1 / (1 + np.exp(-steepness * (gap_score - midpoint)))
    bias = convergence_strength * sigmoid_val
    bias[bias < skip_threshold] = 0
    bias = np.minimum(bias, max_deviation)  # Cap at max_deviation
    return bias
# --- Initial Parameters ---
initial_blend_p = 0.75
initial_steepness = 10.0
initial_midpoint = 0.4
initial_convergence_strength = 1.0
initial_max_deviation = 0.95
initial_skip_threshold = 0.2
# --- Create Figure and Axes ---
fig, ax = plt.subplots(figsize=(16, 12))
plt.subplots_adjust(bottom=0.35, right=0.85)  # Leave space for sliders and colorbar/legend
# Initial bias grid
initial_bias = calculate_bias_vec_balanced(
    grid_data, initial_blend_p, initial_steepness, initial_midpoint, initial_convergence_strength, initial_max_deviation, initial_skip_threshold
)
bias_grid = initial_bias.values.reshape(100, 100)
im = ax.imshow(
    bias_grid, cmap="viridis", extent=[0, 1, 0, 1], origin='lower',
    aspect='auto', interpolation='bilinear', vmin=0
)
# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Bias Value', rotation=270, labelpad=15)
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
# --- Add Sliders ---
axcolor = 'lightgoldenrodyellow'
ax_blend = plt.axes([0.1, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_steep = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_mid = plt.axes([0.1, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_conv = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_maxdev = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_skip = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
s_blend = Slider(ax_blend, 'Blend Parameter', 0.0, 1.0, valinit=initial_blend_p)
s_steep = Slider(ax_steep, 'Steepness', 0.0, 50.0, valinit=initial_steepness)
s_mid = Slider(ax_mid, 'Midpoint', 0.0, 1.0, valinit=initial_midpoint)
s_conv = Slider(ax_conv, 'Convergence Strength', 0.0, 5.0, valinit=initial_convergence_strength)
s_maxdev = Slider(ax_maxdev, 'Max Deviation', 0.0, 5.0, valinit=initial_max_deviation)
s_skip = Slider(ax_skip, 'Skip Threshold', 0.0, 3.0, valinit=initial_skip_threshold)
# --- Update Function ---
def update(val):
    blend_p = s_blend.val
    steepness = s_steep.val
    midpoint = s_mid.val
    convergence_strength = s_conv.val
    max_deviation = s_maxdev.val
    skip_threshold = s_skip.val
    new_bias = calculate_bias_vec_balanced(
        grid_data, blend_p, steepness, midpoint, convergence_strength, max_deviation, skip_threshold
    )
    new_bias_grid = new_bias.values.reshape(100, 100)
    im.set_data(new_bias_grid)
    im.set_clim(0, np.max(new_bias_grid))
    fig.canvas.draw_idle()
# Attach update to sliders
s_blend.on_changed(update)
s_steep.on_changed(update)
s_mid.on_changed(update)
s_conv.on_changed(update)
s_maxdev.on_changed(update)
s_skip.on_changed(update)
# Adjust layout
fig.tight_layout(rect=[0, 0.35, 0.85, 1])  # Make space for sliders and legend
plt.show()