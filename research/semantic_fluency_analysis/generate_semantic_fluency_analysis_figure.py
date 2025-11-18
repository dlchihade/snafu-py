#!/usr/bin/env python3
"""
Generate the semantic fluency analysis figure with four panels:
A. Relationship Between Total and Unique Words
B. Repetition Rate by Subject (Top 10)
C. Distribution of Total Words
D. Most Common Animals (Top 10)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load fluency data
data_path = Path('data/fluency_data.csv')
if not data_path.exists():
    data_path = Path('research/semantic_fluency_analysis/data/fluency_data.csv')

if not data_path.exists():
    raise FileNotFoundError(f"Could not find fluency_data.csv. Tried: {data_path}")

print(f"Loading fluency data from {data_path}...")
data = pd.read_csv(data_path)

# Prepare merged_df_all with items column grouped by participant
# The 'ID' column contains participant IDs, 'Item' contains the words
print("Preparing data...")
merged_df_all = data.groupby('ID')['Item'].apply(list).reset_index()
merged_df_all.columns = ['participant', 'items']

print(f"Loaded data for {len(merged_df_all)} participants")

# 1. Prepare data for plots
# Data for Subplot A: Total vs. Unique Words by Subject
subject_summary = merged_df_all.groupby('participant').agg(
    total_words=('items', lambda x: len(x.iloc[0])),
    unique_words=('items', lambda x: len(set(x.iloc[0])))
).reset_index()

# Data for Subplot B: Repetition Rate by Subject
subject_summary['repetitions'] = subject_summary['total_words'] - subject_summary['unique_words']
subject_summary['repetition_rate'] = subject_summary['repetitions'] / subject_summary['total_words']
subject_summary_top_rep = subject_summary.sort_values('repetition_rate', ascending=False).head(10)

# Data for Subplot C: Distribution of Total Words
total_words_distribution = subject_summary['total_words']
mean_total_words = total_words_distribution.mean()
median_total_words = total_words_distribution.median()

# Data for Subplot D: Most Common Animals
all_animals = [item for sublist in merged_df_all['items'] for item in sublist]
animal_counts = pd.Series(all_animals).value_counts().reset_index()
animal_counts.columns = ['Animal', 'Count']
# Calculate percentage based on number of unique participants
n_participants = len(merged_df_all['participant'].unique())
animal_counts['Percentage'] = (animal_counts['Count'] / n_participants) * 100
animal_counts_top_10 = animal_counts.head(10)

# 2. Create the figure with improved formatting
plt.style.use('default')  # Use default style for clean appearance
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Create figure with better spacing
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                      left=0.06, right=0.94, top=0.93, bottom=0.05)

# ========== Subplot A: Scatter Plot - Total vs. Unique Words ==========
ax1 = fig.add_subplot(gs[0, 0])

# Scatter plot showing relationship
scatter = ax1.scatter(subject_summary['total_words'],
                      subject_summary['unique_words'],
                      c=subject_summary['repetition_rate'],
                      cmap='YlOrRd', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add diagonal reference line (where total = unique, i.e., no repetitions)
max_val = subject_summary['total_words'].max()
ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No repetitions line')

# Add colorbar for repetition rate
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Repetition Rate', fontsize=10)

# Highlight subjects with highest repetition rates
top_rep = subject_summary.nlargest(3, 'repetition_rate')
for idx, row in top_rep.iterrows():
    ax1.annotate(row['participant'],
                 (row['total_words'], row['unique_words']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8)

ax1.set_title('A. Relationship Between Total and Unique Words', pad=10)
ax1.set_xlabel('Total Words Named')
ax1.set_ylabel('Unique Words Named')
ax1.set_xlim(0, max_val + 2)
ax1.set_ylim(0, max_val + 2)
ax1.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# Add text box with summary statistics
textstr = f'Mean repetition rate: {subject_summary["repetition_rate"].mean():.2f}\n'
textstr += f'Subjects with repetitions: {(subject_summary["repetition_rate"] > 0).sum()}/{len(subject_summary)}'
props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# ========== Subplot B: Repetition Rate by Subject (Top 10) ==========
ax2 = fig.add_subplot(gs[0, 1])

# Create color gradient
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(subject_summary_top_rep)))

bars = ax2.bar(range(len(subject_summary_top_rep)),
                subject_summary_top_rep['repetition_rate'],
                color=colors, edgecolor='none')

# Add labels for top 3 participants
for i, (idx, row) in enumerate(subject_summary_top_rep.head(3).iterrows()):
    ax2.text(i, row['repetition_rate'] + 0.005, f"{row['repetition_rate']:.2f}",
             ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(i, row['repetition_rate'] + 0.015, row['participant'],
             ha='center', va='bottom', fontsize=8)

ax2.set_title('B. Repetition Rate by Subject (Top 10)', pad=10)
ax2.set_xlabel('Subject ID (Sorted by Rate)')
ax2.set_ylabel('Repetition Rate (repetitions/total words)')
ax2.set_ylim(0, subject_summary_top_rep['repetition_rate'].max() * 1.15)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_axisbelow(True)
ax2.set_xticks(range(len(subject_summary_top_rep)))
ax2.set_xticklabels(subject_summary_top_rep['participant'], rotation=45, ha='right')

# ========== Subplot C: Distribution of Total Words ==========
ax3 = fig.add_subplot(gs[1, 0])

# Create histogram with appropriate bins
bins = np.arange(5, max(total_words_distribution) + 5, 2.5)
n, bins, patches = ax3.hist(total_words_distribution, bins=bins, 
                            color='#FF69B4', alpha=0.7, edgecolor='white', linewidth=0.8)

# Add mean and median lines
ax3.axvline(mean_total_words, color='red', linestyle='--', linewidth=1.5, 
            label=f'Mean: {mean_total_words:.1f}')
ax3.axvline(median_total_words, color='orange', linestyle='--', linewidth=1.5, 
            label=f'Median: {median_total_words:.1f}')

ax3.set_title('C. Distribution of Total Words', pad=10)
ax3.set_xlabel('Number of Animals Named')
ax3.set_ylabel('Number of Participants')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_axisbelow(True)
ax3.legend(frameon=True, fontsize=9)

# ========== Subplot D: Most Common Animals (Top 10) ==========
ax4 = fig.add_subplot(gs[1, 1])

# Create color gradient
colors = plt.cm.plasma(np.linspace(0, 0.7, len(animal_counts_top_10)))

# Create horizontal bar chart
y_pos = np.arange(len(animal_counts_top_10))
bars = ax4.barh(y_pos, animal_counts_top_10['Count'], color=colors, edgecolor='white', linewidth=0.8)

# Add percentage labels at the end of each bar
for i, (idx, row) in enumerate(animal_counts_top_10.iterrows()):
    ax4.text(row['Count'] + 0.5, i, f"{row['Percentage']:.1f}%",
             va='center', fontsize=9, fontweight='bold')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(animal_counts_top_10['Animal'])
ax4.set_title('D. Most Common Animals (Top 10)', pad=10)
ax4.set_xlabel('Number of Participants Naming Each Animal')
ax4.set_ylabel('Animal Name')
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_axisbelow(True)
ax4.set_xlim(0, animal_counts_top_10['Count'].max() * 1.15)
ax4.invert_yaxis()  # Show most common at top

# Save figure
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

output_svg = output_dir / 'semantic_fluency_analysis.svg'

plt.savefig(output_svg, format='svg', bbox_inches='tight')
plt.close(fig)

print(f"✅ Saved figure to:")
print(f"   {output_svg}")

