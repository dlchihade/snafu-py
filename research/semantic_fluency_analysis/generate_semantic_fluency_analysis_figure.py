#!/usr/bin/env python3
"""
Generate publication-ready semantic fluency analysis figure for Movement Disorders journal.
Four panels: A. Total vs Unique Words, B. Repetition Rate, C. Distribution, D. Most Common Animals
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, linregress

# Load fluency data
data_path = Path('data/fluency_data.csv')
if not data_path.exists():
    data_path = Path('research/semantic_fluency_analysis/data/fluency_data.csv')

if not data_path.exists():
    raise FileNotFoundError(f"Could not find fluency_data.csv. Tried: {data_path}")

print(f"Loading fluency data from {data_path}...")
data = pd.read_csv(data_path)

# Prepare merged_df_all with items column grouped by participant
print("Preparing data...")
merged_df_all = data.groupby('ID')['Item'].apply(list).reset_index()
merged_df_all.columns = ['participant', 'items']

n_participants = len(merged_df_all)
print(f"Loaded data for {n_participants} participants")

# 1. Prepare data for plots
subject_summary = merged_df_all.groupby('participant').agg(
    total_words=('items', lambda x: len(x.iloc[0])),
    unique_words=('items', lambda x: len(set(x.iloc[0])))
).reset_index()

subject_summary['repetitions'] = subject_summary['total_words'] - subject_summary['unique_words']
subject_summary['repetition_rate'] = subject_summary['repetitions'] / subject_summary['total_words']
subject_summary_top_rep = subject_summary.sort_values('repetition_rate', ascending=False).head(7)  # Top 7

# Data for Subplot C: Distribution of Total Words
total_words_distribution = subject_summary['total_words']
mean_total_words = total_words_distribution.mean()
sd_total_words = total_words_distribution.std(ddof=1)
median_total_words = total_words_distribution.median()

# Data for Subplot D: Most Common Animals (Top 8)
all_animals = [item for sublist in merged_df_all['items'] for item in sublist]
animal_counts = pd.Series(all_animals).value_counts().reset_index()
animal_counts.columns = ['Animal', 'Count']
animal_counts['Percentage'] = (animal_counts['Count'] / n_participants) * 100
animal_counts_top_8 = animal_counts.head(8)

# 2. Journal specifications: Movement Disorders
# Figure width: 183mm (full page) = 7.2 inches
# Font: Arial or Helvetica
# Font sizes: 8-10pt axis labels, 6-8pt tick labels

plt.style.use('default')
# Match comprehensive figure font style exactly - use same setup as setup_nature_style()
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

# Convert 183mm to inches: 183mm / 25.4 = 7.2 inches
# Use similar proportions to comprehensive figure but for 2x2 grid
fig_width_mm = 183
fig_width_inches = fig_width_mm / 25.4
# For 2x2, make it taller but maintain good proportions
fig_height_inches = fig_width_inches * 1.1

# Match comprehensive figure spacing: more generous margins and spacing
fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.45,
                      left=0.08, right=0.96, top=0.92, bottom=0.15)

# ========== Panel A: Scatter Plot - Total vs. Unique Words ==========
ax1 = fig.add_subplot(gs[0, 0])

# Calculate regression and R²
x_vals = subject_summary['total_words'].values
y_vals = subject_summary['unique_words'].values
mask_valid = np.isfinite(x_vals) & np.isfinite(y_vals)
x_clean = x_vals[mask_valid]
y_clean = y_vals[mask_valid]

if len(x_clean) > 2:
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    r_squared = r_value ** 2
    # Calculate 95% CI for slope
    n = len(x_clean)
    t_crit = stats.t.ppf(0.975, n - 2)
    slope_ci_lower = slope - t_crit * std_err
    slope_ci_upper = slope + t_crit * std_err
else:
    slope, intercept, r_value, p_value, r_squared = 0, 0, 0, 1, 0
    slope_ci_lower, slope_ci_upper = 0, 0

# Scatter plot with smaller, semi-transparent points
# Use colorblind-friendly colormap (viridis instead of YlOrRd)
scatter = ax1.scatter(subject_summary['total_words'],
                      subject_summary['unique_words'],
                      c=subject_summary['repetition_rate'],
                      cmap='viridis', s=40, alpha=0.6, edgecolors='black', linewidth=0.3)

# Add diagonal reference line
max_val = subject_summary['total_words'].max()
ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.2, linewidth=0.8, label='No repetitions')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
cbar.set_label('Repetition Rate', fontsize=8)
cbar.ax.tick_params(labelsize=7, pad=2)

# Highlight outliers with smaller, non-overlapping labels
top_rep = subject_summary.nlargest(3, 'repetition_rate')
label_positions = {}  # Track positions to avoid overlap
for idx, row in top_rep.iterrows():
    x_pos = row['total_words']
    y_pos = row['unique_words']
    
    # Adjust position to avoid overlap
    offset_x, offset_y = 3, 3
    for existing_pos in label_positions.values():
        if abs(x_pos - existing_pos[0]) < 5 and abs(y_pos - existing_pos[1]) < 5:
            offset_x += 8
            offset_y += 8
    
    ax1.annotate(row['participant'],
                 (x_pos, y_pos),
                 xytext=(offset_x, offset_y), textcoords='offset points',
                 fontsize=6, alpha=0.8,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, 
                          edgecolor='gray', linewidth=0.5),
                 family='Arial')
    label_positions[row['participant']] = (x_pos, y_pos)

ax1.set_title('A. Relationship Between Total and Unique Words Generated', pad=12, fontsize=8)
ax1.set_xlabel('Total Words Generated (n)', fontsize=8)
ax1.set_ylabel('Unique Words Generated (n)', fontsize=8)
ax1.set_xlim(0, max_val + 2)
ax1.set_ylim(0, max_val + 2)

# Fix tick label alignment
ax1.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax1.tick_params(axis='x', which='major', labelrotation=0)
ax1.tick_params(axis='y', which='major', labelrotation=0)

# Very light grid
ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Add text box with regression statistics in upper left (avoid overlap with data)
textstr = f'R² = {r_squared:.3f}\n'
textstr += f'p = {p_value:.3f}'

props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9, pad=0.4)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=7,
         verticalalignment='top', ha='left', bbox=props)

# ========== Panel B: Repetition Rate by Subject (Top 7) ==========
ax2 = fig.add_subplot(gs[0, 1])

# Uniform blue gradient (colorblind-friendly)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(subject_summary_top_rep)))

bars = ax2.bar(range(len(subject_summary_top_rep)),
                subject_summary_top_rep['repetition_rate'],
                color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

ax2.set_title('B. Repetition Rate by Subject (Top 7)', pad=12, fontsize=8)
ax2.set_xlabel('Subject ID (Sorted by Rate)', fontsize=8)
ax2.set_ylabel('Repetition Rate (proportion)', fontsize=8)
ax2.set_ylim(0, subject_summary_top_rep['repetition_rate'].max() * 1.2)
ax2.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)
ax2.set_xticks(range(len(subject_summary_top_rep)))
ax2.set_xticklabels(subject_summary_top_rep['participant'], rotation=90, ha='right', va='top', fontsize=6)
# Fix y-axis tick alignment
ax2.tick_params(axis='y', which='major', labelsize=7, pad=2)
ax2.tick_params(axis='x', which='major', labelsize=6, pad=3)

# Removed n display - not needed

# ========== Panel C: Distribution of Total Words ==========
ax3 = fig.add_subplot(gs[1, 0])

# Integer-width bins
bins = np.arange(int(total_words_distribution.min()), 
                 int(total_words_distribution.max()) + 2, 1)
n, bins, patches = ax3.hist(total_words_distribution, bins=bins, 
                            color='#4A90E2', alpha=0.7, edgecolor='black', linewidth=0.5)

# Add mean ± SD line (no label, will be in stats box)
ax3.axvline(mean_total_words, color='red', linestyle='--', linewidth=1.5)

ax3.set_title('C. Distribution of Total Words Generated', pad=12, fontsize=8)
ax3.set_xlabel('Total Words Generated (n)', fontsize=8)
ax3.set_ylabel('Number of Participants', fontsize=8)
# Fix tick label alignment
ax3.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax3.tick_params(axis='x', which='major', labelrotation=0)
ax3.tick_params(axis='y', which='major', labelrotation=0)
ax3.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
ax3.set_axisbelow(True)

# Add statistics box in upper left to avoid overlap with data
stats_text = f'Mean ± SD: {mean_total_words:.1f} ± {sd_total_words:.1f}\n'
stats_text += f'Median: {median_total_words:.1f}'

ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
         ha='left', va='top', fontsize=7,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', pad=0.4))

# ========== Panel D: Most Common Animals (Top 8) ==========
ax4 = fig.add_subplot(gs[1, 1])

# Colorblind-friendly gradient (use Blues)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(animal_counts_top_8)))

# Create horizontal bar chart
y_pos = np.arange(len(animal_counts_top_8))
bars = ax4.barh(y_pos, animal_counts_top_8['Count'], color=colors, 
                edgecolor='black', linewidth=0.5, alpha=0.8)

# Add both count and percentage labels with proper alignment
max_count = animal_counts_top_8['Count'].max()
for i, (idx, row) in enumerate(animal_counts_top_8.iterrows()):
    label_x = row['Count'] + max_count * 0.03
    label_text = f"{int(row['Count'])} ({row['Percentage']:.1f}%)"
    ax4.text(label_x, i, label_text,
             va='center', ha='left', fontsize=7, fontweight='bold')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(animal_counts_top_8['Animal'], fontsize=7)
ax4.set_title('D. Most Common Animals (Top 8)', pad=12, fontsize=8)
ax4.set_xlabel('Participants (n, %)', fontsize=8)
ax4.set_ylabel('Animal Name', fontsize=8)
# Fix tick label alignment
ax4.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax4.tick_params(axis='x', which='major', labelrotation=0)
ax4.tick_params(axis='y', which='major', labelrotation=0)
ax4.grid(True, alpha=0.15, axis='x', linestyle='-', linewidth=0.5)
ax4.set_axisbelow(True)
ax4.set_xlim(0, animal_counts_top_8['Count'].max() * 1.3)
ax4.invert_yaxis()

# Removed n display - not needed

# Save figure in multiple formats
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Save at exact journal dimensions with 300 DPI
base_name = 'semantic_fluency_analysis_movement_disorders'

# PNG (300 DPI)
output_png = output_dir / f'{base_name}.png'
fig.savefig(output_png, format='png', bbox_inches='tight', pad_inches=0.1, 
            facecolor='white', dpi=300, edgecolor='none')

# PDF (vector, embedded fonts)
output_pdf = output_dir / f'{base_name}.pdf'
fig.savefig(output_pdf, format='pdf', bbox_inches='tight', pad_inches=0.1,
            facecolor='white', edgecolor='none')

# EPS (vector, for journal submission)
output_eps = output_dir / f'{base_name}.eps'
fig.savefig(output_eps, format='eps', bbox_inches='tight', pad_inches=0.1,
            facecolor='white', edgecolor='none', dpi=300)

# TIFF (high-res raster, 300 DPI)
try:
    from PIL import Image
    # Convert PNG to TIFF
    img = Image.open(output_png)
    output_tiff = output_dir / f'{base_name}.tiff'
    img.save(output_tiff, format='TIFF', dpi=(300, 300), compression='tiff_lzw')
    print(f"✅ Saved: {output_tiff}")
except ImportError:
    print("⚠️  PIL not available, skipping TIFF export")

plt.close(fig)

print(f"✅ Saved publication-ready figures:")
print(f"   PNG (300 DPI): {output_png}")
print(f"   PDF (vector): {output_pdf}")
print(f"   EPS (vector): {output_eps}")
print(f"\nFigure dimensions: {fig_width_mm}mm × {fig_height_inches*25.4:.1f}mm")
print(f"Font: Arial, Axis labels: 9pt, Tick labels: 7pt")
print(f"Sample size: n = {n_participants}")
