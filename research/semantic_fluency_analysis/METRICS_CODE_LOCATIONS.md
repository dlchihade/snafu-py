# Metrics Code Locations

This document shows where each metric in the "Exploitation vs. Exploration Metrics" figure is computed and used.

---

## Panel A: Intra-Phase Mean

### Computation Location
**File:** `phase_coherence_analysis.py`

**Exploitation Intra-Phase Mean:**
- **Lines 39-62**: Collect pairwise cosine similarities within all exploitation phases
- **Line 104**: Compute mean: `exploitation_mean = np.mean(exploitation_intra_similarities)`
- **Line 120**: Store in metrics: `'exploitation_intra_mean': exploitation_mean`

**Exploration Intra-Phase Mean:**
- **Lines 68-91**: Collect pairwise cosine similarities within all exploration phases
- **Line 138**: Compute mean: `exploration_mean = np.mean(exploration_intra_similarities)`
- **Line 154**: Store in metrics: `'exploration_intra_mean': exploration_mean`

### Usage in Figure
**File:** `create_exploit_explore_metrics_figure.py`
- **Lines 213-222**: Panel A creation
  ```python
  create_boxplot_panel(
      axes[0, 0],
      df['exploitation_intra_mean'].values,  # Line 216
      df['exploration_intra_mean'].values,    # Line 217
      'Intra-Phase Mean',
      'Cosine Similarity',
      colors,
      'A'
  )
  ```

---

## Panel B: Intra-Phase Variance

### Computation Location
**File:** `phase_coherence_analysis.py`

**Exploitation Intra-Phase Variance:**
- **Lines 39-62**: Collect pairwise cosine similarities within all exploitation phases
- **Line 107**: Compute variance: `exploitation_variance = np.var(exploitation_intra_similarities)`
- **Line 121**: Store in metrics: `'exploitation_intra_variance': exploitation_variance`

**Exploration Intra-Phase Variance:**
- **Lines 68-91**: Collect pairwise cosine similarities within all exploration phases
- **Line 141**: Compute variance: `exploration_variance = np.var(exploration_intra_similarities)`
- **Line 155**: Store in metrics: `'exploration_intra_variance': exploration_variance`

### Usage in Figure
**File:** `create_exploit_explore_metrics_figure.py`
- **Lines 224-233**: Panel B creation
  ```python
  create_boxplot_panel(
      axes[0, 1],
      df['exploitation_intra_variance'].values,  # Line 227
      df['exploration_intra_variance'].values,  # Line 228
      'Intra-Phase Variance',
      'Variance',
      colors,
      'B'
  )
  ```

---

## Panel C: Inter-Phase Mean

### Computation Location
**File:** `phase_coherence_analysis.py`

**Inter-Phase Similarities Collection:**
- **Lines 194-236**: Calculate phase centroids (mean vectors) for each phase
- **Lines 250-270**: Compute cosine similarity between all phase centroids
- **Lines 262-267**: Separate by phase type:
  - If `centroid1['type'] == 'Exploitation'`: add to `exploitation_inter_similarities`
  - If `centroid1['type'] == 'Exploration'`: add to `exploration_inter_similarities`

**Exploitation Inter-Phase Mean:**
- **Line 305**: Compute mean: `exploitation_inter_mean = np.mean(exploitation_inter_similarities)`
- **Line 345**: Store in metrics: `'inter_phase_mean_exploitation': exploitation_inter_mean`

**Exploration Inter-Phase Mean:**
- **Line 323**: Compute mean: `exploration_inter_mean = np.mean(exploration_inter_similarities)`
- **Line 349**: Store in metrics: `'inter_phase_mean_exploration': exploration_inter_mean`

### Usage in Figure
**File:** `create_exploit_explore_metrics_figure.py`
- **Lines 235-258**: Panel C creation
  ```python
  if has_separated_inter:
      create_boxplot_panel(
          axes[1, 0],
          df['inter_phase_mean_exploitation'].values,  # Line 240
          df['inter_phase_mean_exploration'].values,   # Line 241
          'Inter-Phase Mean',
          'Cosine Similarity',
          colors,
          'C'
      )
  ```

---

## Panel D: Inter-Phase Variance

### Computation Location
**File:** `phase_coherence_analysis.py`

**Exploitation Inter-Phase Variance:**
- **Line 306**: Compute variance: `exploitation_inter_variance = np.var(exploitation_inter_similarities)`
- **Line 346**: Store in metrics: `'inter_phase_variance_exploitation': exploitation_inter_variance`

**Exploration Inter-Phase Variance:**
- **Line 324**: Compute variance: `exploration_inter_variance = np.var(exploration_inter_similarities)`
- **Line 350**: Store in metrics: `'inter_phase_variance_exploration': exploration_inter_variance`

### Usage in Figure
**File:** `create_exploit_explore_metrics_figure.py`
- **Lines 260-281**: Panel D creation
  ```python
  if has_separated_inter:
      create_boxplot_panel(
          axes[1, 1],
          df['inter_phase_variance_exploitation'].values,  # Line 265
          df['inter_phase_variance_exploration'].values,   # Line 266
          'Inter-Phase Variance',
          'Variance',
          colors,
          'D'
      )
  ```

---

## Statistical Tests (t, p, d, η²)

### Computation Location
**File:** `create_exploit_explore_metrics_figure.py`

**t-statistic and p-value:**
- **Line 112**: `t_stat, p_val = ttest_ind(data1, data2)`
  - Uses `scipy.stats.ttest_ind` for independent samples t-test

**Cohen's d (effect size):**
- **Lines 59-66**: `compute_effect_size()` function
  ```python
  pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
  return (np.mean(group1) - np.mean(group2)) / pooled_std
  ```
- **Line 113**: Called as `cohen_d = compute_effect_size(data1, data2)`

**Eta-squared (η²):**
- **Lines 69-75**: `compute_eta_squared()` function
  ```python
  t_stat, _ = ttest_ind(group1, group2)
  df = n1 + n2 - 2
  eta_sq = t_stat**2 / (t_stat**2 + df)
  ```
- **Line 114**: Called as `eta_sq = compute_eta_squared(data1, data2)`

**Display:**
- **Lines 153-159**: Statistics box added to each panel
  ```python
  stats_text = f't = {t_stat:.2f}\np = {p_str}\nd = {cohen_d:.2f}\nη² = {eta_sq:.2f}'
  ax.text(stats_x_pos, stats_y_pos, stats_text, ...)
  ```

---

## Data Flow Summary

1. **Data Generation** (`phase_coherence_analysis.py`):
   - Computes all metrics per participant
   - Saves to CSV: `output/phase_coherence_analysis_all_participants.csv`

2. **Data Loading** (`create_exploit_explore_metrics_figure.py`):
   - **Line 182**: Load CSV: `df = pd.read_csv(data_path)`
   - **Lines 188-199**: Check for required columns and filter missing data

3. **Figure Creation** (`create_exploit_explore_metrics_figure.py`):
   - **Lines 213-281**: Create 4 panels using `create_boxplot_panel()` function
   - **Lines 78-159**: `create_boxplot_panel()` function:
     - Creates boxplots (lines 86-109)
     - Computes statistics (lines 112-114)
     - Adds statistics box (lines 153-159)

---

## Key Functions

### `compute_intra_phase_metrics()` 
- **Location:** `phase_coherence_analysis.py`, lines 22-167
- **Purpose:** Compute pairwise similarities within phases
- **Returns:** Dictionary with `exploitation_intra_mean`, `exploitation_intra_variance`, `exploration_intra_mean`, `exploration_intra_variance`

### `compute_inter_phase_metrics()`
- **Location:** `phase_coherence_analysis.py`, lines 169-353
- **Purpose:** Compute centroid similarities between phases
- **Returns:** Dictionary with `inter_phase_mean_exploitation`, `inter_phase_mean_exploration`, `inter_phase_variance_exploitation`, `inter_phase_variance_exploration`

### `create_boxplot_panel()`
- **Location:** `create_exploit_explore_metrics_figure.py`, lines 78-159
- **Purpose:** Create a single boxplot panel with statistical annotations
- **Parameters:** axis, data1 (exploitation), data2 (exploration), title, ylabel, colors, panel_label


