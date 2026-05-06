# How to Use EE Correlation Analysis with Your Data

## Your Data Structure

You have:
- **Metrics data**: `output/NATURE_REAL_metrics.csv`
  - Columns: `ID`, `num_switches`, `exploitation_intra_mean`, `exploration_intra_mean`, etc.
  
- **MEG data**: `data/meg_data.csv`
  - Columns: `ID`, `alpha_NET_mean`, `norm_SN_avg`, `norm_LC_avg`, etc.

## Step-by-Step Usage

### Step 1: Load Your Data

```python
import pandas as pd
from pathlib import Path

# Load your metrics
metrics_df = pd.read_csv('output/NATURE_REAL_metrics.csv')

# Load MEG data
meg_df = pd.read_csv('data/meg_data.csv')

# Merge them
merged_df = pd.merge(metrics_df, meg_df, on='ID', how='left')
```

### Step 2: Adapt Column Names

The correlation analysis code expects columns like:
- `exploitation_time`, `exploration_time`
- `exploitation_percentage`, `exploration_percentage`
- `exploitation_phases_ratio`, `exploration_phases_ratio`
- `ee_tradeoff`, `mean_phase_size`

**Your data has:**
- `exploitation_intra_mean`, `exploration_intra_mean`
- `num_switches`
- `novelty_score`
- `exploitation_coherence_ratio`, `exploration_coherence_ratio`

**Option A: Use Available Columns Directly**

```python
# Work with what you have
available_columns = [
    'exploitation_intra_mean',
    'exploration_intra_mean',
    'num_switches',
    'novelty_score',
    'exploitation_coherence_ratio',
    'exploration_coherence_ratio',
    'phase_separation_index'
]

# Filter to existing columns
ee_columns = [col for col in available_columns if col in merged_df.columns]

# Compute correlations
corr_matrix = merged_df[ee_columns].corr()
```

**Option B: Calculate Missing Metrics from Phase Data**

If you have phase data, calculate:
```python
# From phase_coherence_analysis_all_participants.csv or phase analysis results
# Calculate exploitation_time, exploration_time, etc.
```

### Step 3: Run Basic Correlation Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Select your EE columns
ee_columns = [
    'exploitation_intra_mean',
    'exploration_intra_mean',
    'num_switches',
    'novelty_score',
    'exploitation_coherence_ratio',
    'exploration_coherence_ratio'
]

# Filter to existing columns
ee_columns = [col for col in ee_columns if col in merged_df.columns]

# Compute correlation matrix
corr_matrix = merged_df[ee_columns].corr(method='pearson')

# Create heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True
)
plt.title("Correlation Matrix of EE Measures")
plt.tight_layout()
plt.show()
```

### Step 4: Find Significant Correlations

```python
from scipy.stats import pearsonr

# Calculate all pairwise correlations
results = []
for i, col1 in enumerate(ee_columns):
    for j, col2 in enumerate(ee_columns):
        if i < j:  # Avoid duplicates
            valid_data = merged_df[[col1, col2]].dropna()
            if len(valid_data) >= 5:
                corr, p = pearsonr(valid_data[col1], valid_data[col2])
                results.append({
                    'Variable1': col1,
                    'Variable2': col2,
                    'Correlation': corr,
                    'p_value': p,
                    'Significant': p < 0.05
                })

# Convert to DataFrame
corr_results = pd.DataFrame(results)
significant = corr_results[corr_results['p_value'] < 0.05].sort_values('p_value')

print("Significant Correlations:")
print(significant)
```

### Step 5: Correlations with MEG Data

```python
# MEG columns
meg_columns = ['alpha_NET_mean', 'norm_SN_avg', 'norm_LC_avg']

# Calculate EE-MEG correlations
ee_meg_results = []
for ee_col in ee_columns:
    for meg_col in meg_columns:
        if meg_col in merged_df.columns:
            valid_data = merged_df[[ee_col, meg_col]].dropna()
            if len(valid_data) >= 5:
                corr, p = pearsonr(valid_data[ee_col], valid_data[meg_col])
                ee_meg_results.append({
                    'EE_Measure': ee_col,
                    'MEG_Measure': meg_col,
                    'Correlation': corr,
                    'p_value': p,
                    'Significant': p < 0.05
                })

ee_meg_df = pd.DataFrame(ee_meg_results)
print("\nEE-MEG Correlations:")
print(ee_meg_df.sort_values('p_value'))
```

### Step 6: Create EE-MEG Heatmap

```python
# Create correlation matrix for heatmap
corr_data = merged_df[ee_columns + meg_columns].dropna()
corr_matrix_full = corr_data.corr()

# Extract EE-MEG subset
ee_meg_corr = corr_matrix_full.loc[ee_columns, meg_columns]

# Plot
plt.figure(figsize=(8, len(ee_columns) * 0.8))
sns.heatmap(
    ee_meg_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Correlations: EE Measures vs MEG Measures")
plt.tight_layout()
plt.show()
```

### Step 7: Plot Specific Relationships

```python
# Example: Exploitation vs Exploration
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=merged_df,
    x='exploitation_intra_mean',
    y='exploration_intra_mean',
    alpha=0.7
)

# Add correlation
from scipy.stats import pearsonr
valid = merged_df[['exploitation_intra_mean', 'exploration_intra_mean']].dropna()
corr, p = pearsonr(valid['exploitation_intra_mean'], valid['exploration_intra_mean'])
plt.title(f"Exploitation vs Exploration\nr = {corr:.3f}, p = {p:.3f}")
plt.show()

# Example: Alpha vs Exploitation
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=merged_df,
    x='alpha_NET_mean',
    y='exploitation_intra_mean',
    alpha=0.7
)
valid = merged_df[['alpha_NET_mean', 'exploitation_intra_mean']].dropna()
corr, p = pearsonr(valid['alpha_NET_mean'], valid['exploitation_intra_mean'])
plt.title(f"Alpha Power vs Exploitation\nr = {corr:.3f}, p = {p:.3f}")
plt.show()
```

## Complete Example Script

See `example_ee_correlation_analysis.py` for a complete working example.

## Notes

1. **Column Names**: The correlation analysis code expects specific column names. You may need to:
   - Rename your columns to match
   - Or modify the correlation functions to use your column names
   - Or calculate missing metrics from your phase data

2. **Missing Metrics**: If you need `exploitation_time`, `exploration_time`, etc., calculate them from your phase analysis results.

3. **Data Quality**: Make sure to handle missing values appropriately (`.dropna()` before correlations).






