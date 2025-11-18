#!/usr/bin/env python3
"""
Generate two standalone distribution figures:
1. Distribution of Total Words (Panel C from semantic fluency analysis)
2. Distribution of SVF scores
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
print("Preparing data...")
merged_df_all = data.groupby('ID')['Item'].apply(list).reset_index()
merged_df_all.columns = ['participant', 'items']

# Calculate total words per participant
subject_summary = merged_df_all.groupby('participant').agg(
    total_words=('items', lambda x: len(x.iloc[0]))
).reset_index()

total_words_distribution = subject_summary['total_words']
mean_total_words = total_words_distribution.mean()
median_total_words = total_words_distribution.median()

# Calculate SVF scores from fluency data (same source as total words)
# This ensures we use the same participants for both distributions
print("Calculating SVF scores from fluency data...")
svf_scores = subject_summary['total_words'].copy()  # SVF count = total words named
mean_svf = svf_scores.mean()
median_svf = svf_scores.median()
print(f"Calculated SVF scores for {len(svf_scores)} participants")

# Set up styling
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# ========== Figure 1: Distribution of Total Words ==========
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Create histogram with appropriate bins
bins = np.arange(5, max(total_words_distribution) + 5, 2.5)
n, bins, patches = ax1.hist(total_words_distribution, bins=bins, 
                            color='#FF69B4', alpha=0.7, edgecolor='white', linewidth=0.8)

# Add mean and median lines
ax1.axvline(mean_total_words, color='red', linestyle='--', linewidth=1.5, 
            label=f'Mean: {mean_total_words:.1f}')
ax1.axvline(median_total_words, color='orange', linestyle='--', linewidth=1.5, 
            label=f'Median: {median_total_words:.1f}')

ax1.set_title('Distribution of Total Words', pad=15, fontweight='bold')
ax1.set_xlabel('Number of Animals Named', fontweight='normal')
ax1.set_ylabel('Number of Participants', fontweight='normal')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_axisbelow(True)
ax1.legend(frameon=True, fontsize=10, loc='upper right')

plt.tight_layout()
output_svg1 = output_dir / 'distribution_total_words.svg'
plt.savefig(output_svg1, format='svg', bbox_inches='tight')
plt.close(fig1)

print(f"✅ Saved distribution of total words to: {output_svg1}")

# ========== Figure 2: Distribution of SVF Scores ==========
if svf_scores is not None and len(svf_scores) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Create histogram with appropriate bins
    bins = np.arange(0, max(svf_scores) + 5, 2.5)
    n, bins, patches = ax2.hist(svf_scores, bins=bins, 
                                color='#4A90E2', alpha=0.7, edgecolor='white', linewidth=0.8)
    
    # Add mean and median lines
    ax2.axvline(mean_svf, color='red', linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_svf:.1f}')
    ax2.axvline(median_svf, color='orange', linestyle='--', linewidth=1.5, 
                label=f'Median: {median_svf:.1f}')
    
    ax2.set_title('Distribution of SVF Scores', pad=15, fontweight='bold')
    ax2.set_xlabel('SVF Count (Number of Words)', fontweight='normal')
    ax2.set_ylabel('Number of Participants', fontweight='normal')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    ax2.legend(frameon=True, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    output_svg2 = output_dir / 'distribution_svf_scores.svg'
    plt.savefig(output_svg2, format='svg', bbox_inches='tight')
    plt.close(fig2)
    
    print(f"✅ Saved distribution of SVF scores to: {output_svg2}")
else:
    print("⚠️  Could not generate SVF distribution figure - no SVF data available")

print(f"\n📊 Summary Statistics:")
print(f"   Total Words: Mean={mean_total_words:.1f}, Median={median_total_words:.1f}, N={len(total_words_distribution)}")
if svf_scores is not None:
    print(f"   SVF Scores: Mean={mean_svf:.1f}, Median={median_svf:.1f}, N={len(svf_scores)}")

