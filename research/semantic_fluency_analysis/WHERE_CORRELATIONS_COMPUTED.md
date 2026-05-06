# WHERE CORRELATIONS ARE COMPUTED IN `example_ee_correlation_analysis.py`

## Location 1: EE-EE Correlations (Between Exploitation/Exploration Measures)

**File:** `example_ee_correlation_analysis.py`  
**Lines:** 161-165

```python
# Compute correlation matrix
corr_matrix = merged_df[available_ee_columns].corr(method='pearson')

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))
```

**What this does:**
- Computes Pearson correlations between all EE measures
- Uses pandas `.corr()` method
- Creates an 8x8 correlation matrix (all EE measures vs all EE measures)

**Columns included:**
- `exploitation_intra_mean`
- `exploration_intra_mean`
- `num_switches`
- `novelty_score`
- `exploitation_coherence_ratio`
- `exploration_coherence_ratio`
- `phase_separation_index`
- `inter_phase_mean`

---

## Location 2: EE-MEG Correlations (The Significant One!)

**File:** `example_ee_correlation_analysis.py`  
**Lines:** 212-227

```python
for ee_col in available_ee_columns:
    for meg_col in meg_columns:
        # Get valid data
        valid_data = merged_df[[ee_col, meg_col]].dropna()
        
        if len(valid_data) >= 5:  # Need at least 5 data points
            from scipy.stats import pearsonr
            corr, p = pearsonr(valid_data[ee_col], valid_data[meg_col])
            
            ee_meg_correlations.append({
                'EE_Measure': ee_col,
                'MEG_Measure': meg_col,
                'Correlation': corr,
                'p_value': p,
                'Significant': p < 0.05
            })
```

**What this does:**
- Loops through all EE measures × all MEG measures
- Uses `scipy.stats.pearsonr()` to compute correlation + p-value
- **THIS IS WHERE** the significant correlation is computed:
  - `exploitation_coherence_ratio` vs `norm_SN_avg`: r = -0.31, p = 0.036

**MEG columns:**
- `alpha_NET_mean`
- `norm_SN_avg`
- `norm_LC_avg`

---

## Location 3: EE-MEG Correlation Matrix (for Heatmap)

**File:** `example_ee_correlation_analysis.py`  
**Lines:** 238-242

```python
# Create correlation matrix
corr_data = merged_df[available_ee_columns + meg_columns].dropna()
if len(corr_data) > 0:
    ee_meg_corr_matrix = corr_data.corr()
    ee_meg_subset = ee_meg_corr_matrix.loc[available_ee_columns, meg_columns]
```

**What this does:**
- Creates a full correlation matrix (EE + MEG columns)
- Extracts the subset: EE rows × MEG columns
- Used for the heatmap visualization

---

## Location 4: Individual Relationship Plots

**File:** `example_ee_correlation_analysis.py`  
**Lines:** 278-283 (Exploitation vs Exploration)

```python
# Add correlation
from scipy.stats import pearsonr
valid_data = merged_df[['exploitation_intra_mean', 'exploration_intra_mean']].dropna()
if len(valid_data) > 0:
    corr, p = pearsonr(valid_data['exploitation_intra_mean'], valid_data['exploration_intra_mean'])
    plt.title(f"Exploitation vs Exploration Intra-Phase Means\nr = {corr:.3f}, p = {p:.3f}")
```

**Lines:** 304-305 (Alpha vs Exploitation)

```python
corr, p = pearsonr(valid_data['alpha_NET_mean'], valid_data['exploitation_intra_mean'])
plt.title(f"Alpha Power vs Exploitation Intra-Phase Mean\nr = {corr:.3f}, p = {p:.3f}")
```

**What this does:**
- Computes correlations for specific scatter plots
- Uses `scipy.stats.pearsonr()` for individual pairs

---

## Summary: All Correlation Computation Locations

| Location | Lines | Method | Purpose |
|----------|-------|--------|---------|
| **EE-EE Matrix** | 162 | `pandas .corr()` | All EE measures vs all EE measures |
| **EE-MEG Pairs** | 219 | `scipy.stats.pearsonr()` | **Each EE × Each MEG (with p-values)** |
| **EE-MEG Matrix** | 241 | `pandas .corr()` | Full matrix for heatmap |
| **Plot 1** | 282 | `scipy.stats.pearsonr()` | Exploitation vs Exploration |
| **Plot 2** | 304 | `scipy.stats.pearsonr()` | Alpha vs Exploitation |

---

## The Significant Correlation

**Found at:** Line 219 in the nested loop (lines 212-227)

When:
- `ee_col = 'exploitation_coherence_ratio'`
- `meg_col = 'norm_SN_avg'`

The code computes:
```python
corr, p = pearsonr(valid_data['exploitation_coherence_ratio'], valid_data['norm_SN_avg'])
```

Result: **r = -0.31, p = 0.036** (significant at p < 0.05)






