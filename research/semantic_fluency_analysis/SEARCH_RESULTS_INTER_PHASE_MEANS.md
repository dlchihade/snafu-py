# DEEP SEARCH RESULTS: Where Do 0.78 and 0.33 Come From?

## 🔍 Key Finding

**The values 0.78 and 0.33 are NOT computed in your scripts!**

### Actual Computed Values (from `output/phase_coherence_analysis_all_participants.csv`):

```
Exploitation inter-phase mean:
  Count: 55 participants
  Mean: 0.7051  ← NOT 0.78!
  Min: 0.5468
  Max: 0.8667

Exploration inter-phase mean:
  Count: 55 participants
  Mean: 0.7589  ← NOT 0.33!
  Min: 0.5766
  Max: 0.8829
```

---

## 📍 Where Inter-Phase Means ARE Computed

### 1. **`phase_coherence_analysis.py`** (Primary Script)

**Function:** `compute_inter_phase_metrics()` (Lines 244-353)

**Computation Locations:**
- **Line 305:** `exploitation_inter_mean = np.mean(exploitation_inter_similarities)`
- **Line 323:** `exploration_inter_mean = np.mean(exploration_inter_similarities)`

**Problem:** Lines 262-267 include same-type pairs (Exploitation↔Exploitation, Exploration↔Exploration) when they should only include cross-type pairs.

**Output:** Saved to `output/phase_coherence_analysis_all_participants.csv`

**Group-level statistics printed at:**
- **Line 543:** `results_df['inter_phase_mean'].mean()` (aggregate, not separated)

---

### 2. **`create_exploit_explore_metrics_figure.py`**

**Lines 240-241:** Reads the values from CSV:
```python
df['inter_phase_mean_exploitation'].values,
df['inter_phase_mean_exploration'].values,
```

**Purpose:** Creates boxplots comparing exploitation vs exploration inter-phase means

---

### 3. **Colab Notebook: `mediation_analysis.ipynb`**

**Lines 3198-3200:** Sets values to `np.nan` (NOT computed):
```python
inter_phase_mean_exploitation = np.nan
inter_phase_mean_exploration = np.nan
```

**Status:** ❌ Not implemented in the notebook

---

## 🤔 Where Do 0.78 and 0.33 Come From?

### Possibilities:

1. **Expected/Target Values from Literature**
   - These might be values from a published paper or theoretical expectation
   - Not computed from your actual data

2. **Different Analysis Method**
   - Might come from a different way of computing inter-phase similarities
   - Could be from a corrected version that only includes cross-type pairs

3. **Manual Entry/Reference**
   - Could be values you manually entered or referenced from another source
   - Might be from a different dataset or analysis

4. **Bug in Current Implementation**
   - The current code includes same-type pairs, which inflates the values
   - If fixed to only include cross-type pairs, values might be closer to 0.78/0.33

---

## 🔧 What Needs to Be Fixed

To potentially get values closer to 0.78 and 0.33, fix `phase_coherence_analysis.py`:

**Current (Lines 262-267):**
```python
if centroid1['type'] == 'Exploitation':
    exploitation_inter_similarities.append(similarity)  # Includes Exploitation↔Exploitation

if centroid1['type'] == 'Exploration':
    exploration_inter_similarities.append(similarity)  # Includes Exploration↔Exploration
```

**Should be:**
```python
if centroid1['type'] == 'Exploitation' and centroid2['type'] == 'Exploration':
    exploitation_inter_similarities.append(similarity)  # Only Exploitation↔Exploration

if centroid1['type'] == 'Exploration' and centroid2['type'] == 'Exploitation':
    exploration_inter_similarities.append(similarity)  # Only Exploration↔Exploitation
```

---

## 📊 Summary

| Value | Source | Status |
|-------|--------|--------|
| **0.78** (Exploitation) | ❓ Unknown - NOT in your scripts | Expected/Reference value? |
| **0.33** (Exploration) | ❓ Unknown - NOT in your scripts | Expected/Reference value? |
| **0.7051** (Exploitation) | ✅ Computed in `phase_coherence_analysis.py` | Actual group mean |
| **0.7589** (Exploration) | ✅ Computed in `phase_coherence_analysis.py` | Actual group mean |

---

## 🎯 Next Steps

1. **Check your paper/manuscript** - Do 0.78 and 0.33 appear there? Where did they come from?
2. **Fix the bug** in `phase_coherence_analysis.py` to only include cross-type pairs
3. **Re-run the analysis** and see if the corrected values are closer to 0.78/0.33
4. **Check other scripts** - Are there any scripts that compute these values differently?






