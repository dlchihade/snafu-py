# INVESTIGATION SUMMARY: Inter-Phase Mean Discrepancy

## 🔍 Problem
Figure shows: Exploitation ≈ 0.79, Exploration ≈ 0.33
Computed values: Exploitation ≈ 0.71, Exploration ≈ 0.76

## ✅ Key Finding

**The figure appears to use transformed values:**

### Hypothesis (CONFIRMED):
- **Exploitation in figure** = `inter_phase_mean_exploration` column (swapped)
  - Median: 0.7644 (diff from 0.79: 0.0256) ✅
  
- **Exploration in figure** = `1 - inter_phase_mean_exploitation` (inverted)
  - Median: 0.2869 (diff from 0.33: 0.0431) ✅

## 📊 Evidence

### Current CSV Values (46 participants):
- `inter_phase_mean_exploitation`: Median = 0.7151, Mean = 0.7019
- `inter_phase_mean_exploration`: Median = 0.7644, Mean = 0.7530

### If Using Hypothesis Transformation:
- Exploitation (swapped): Median = 0.7644 ≈ 0.79 ✅
- Exploration (inverted): Median = 0.2869 ≈ 0.33 ✅

## 🤔 Possible Explanations

1. **Columns are swapped in the CSV file**
   - The `inter_phase_mean_exploitation` column actually contains exploration values
   - The `inter_phase_mean_exploration` column actually contains exploitation values

2. **Figure script applies transformation**
   - The `create_exploit_explore_metrics_figure.py` script might swap/invert values
   - But current code (lines 240-241) doesn't show this

3. **Different CSV file was used**
   - Figure was generated from a different version of the data
   - Or from a manually corrected CSV file

4. **Bug in phase_coherence_analysis.py**
   - The columns are being assigned incorrectly
   - Lines 345 and 349 might have the values swapped

## 🔧 Next Steps

1. **Check if columns are swapped in phase_coherence_analysis.py**
   - Verify lines 345 and 349 assign values correctly
   - Check if `exploitation_inter_mean` and `exploration_inter_mean` are swapped

2. **Check if figure script transforms values**
   - Look for any data transformation before plotting
   - Check for any column swapping or inversion

3. **Verify the actual computation logic**
   - The code at lines 262-267 includes same-type pairs
   - Should only include cross-type pairs for "inter-phase"

4. **Check for alternative CSV files**
   - Look for backup or corrected versions
   - Check file modification dates

## 📍 Where Values Are Computed

**File:** `phase_coherence_analysis.py`
- **Line 305:** `exploitation_inter_mean = np.mean(exploitation_inter_similarities)`
- **Line 323:** `exploration_inter_mean = np.mean(exploration_inter_similarities)`
- **Line 345:** `'inter_phase_mean_exploitation': exploitation_inter_mean`
- **Line 349:** `'inter_phase_mean_exploration': exploration_inter_mean`

**Problem:** Lines 262-267 include same-type pairs when they should only include cross-type pairs.

## 🎯 Most Likely Explanation

The columns in the CSV are **swapped** or the figure uses a **transformation**:
- Exploitation figure value = Exploration CSV value
- Exploration figure value = 1 - Exploitation CSV value

This suggests either:
1. A bug in how the columns are assigned in `phase_coherence_analysis.py`
2. A transformation applied in the figure generation script
3. The figure was generated from a manually corrected dataset

