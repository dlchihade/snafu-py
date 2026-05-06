# WHERE INTER-PHASE MEANS ARE COMPUTED IN THE COLAB NOTEBOOK

## Colab Notebook Location
**URL:** https://colab.research.google.com/drive/1jwDkbF8srYnnxLwQAaTNqcHo2L0YjAaV?hl=en

## Local Copy
**File:** `output/scripts/mediation/mediation_analysis.ipynb`

---

## Key Finding

In the Colab notebook (`mediation_analysis.ipynb`), the inter-phase mean values are **NOT directly computed** in the `analyze_responses` function. Instead, they are set as placeholders:

### Lines 3198-3200:
```python
inter_phase_mean_exploitation = np.nan
inter_phase_mean_exploration = np.nan
```

These are returned as `np.nan` in the function's return dictionary (lines 3234-3236).

---

## Where They SHOULD Be Computed

Based on the notebook structure, the inter-phase means (0.78 for exploitation, 0.33 for exploration) should be computed in a function that:

1. **Calculates phase centroids** (lines 4040-4068):
   - `calculate_phase_centroids(phases)` - Creates centroids for exploitation and exploration phases

2. **Computes inter-phase similarities** (lines 4108-4119):
   - `calculate_inter_phase_similarities(phase_centroids)` - But this only computes **consecutive** phases, not all cross-type pairs

3. **Should compute cross-type similarities**:
   - Need to compute similarities between ALL Exploitation centroids ↔ ALL Exploration centroids
   - Then calculate means separately for:
     - Exploitation inter-phase mean: Mean of all Exploitation ↔ Exploration similarities
     - Exploration inter-phase mean: Mean of all Exploration ↔ Exploitation similarities

---

## The Problem

The notebook's `calculate_inter_phase_similarities` function (line 4109) only computes **consecutive** phase similarities:

```python
def calculate_inter_phase_similarities(phase_centroids):
    inter_phase_similarities = []
    for i in range(len(phase_centroids) - 1):
        if phase_centroids[i] is not None and phase_centroids[i+1] is not None:
            # Only computes i ↔ i+1 (consecutive)
            sim = cosine_similarity(phase_centroids[i][0], phase_centroids[i+1][0])
            inter_phase_similarities.append((sim, ...))
    return inter_phase_similarities
```

**This is NOT the same as computing all Exploitation ↔ Exploration cross-type similarities!**

---

## What Needs to Be Done in the Colab Notebook

To compute the inter-phase means (0.78 and 0.33), you need code that:

1. **Separates centroids by type:**
   ```python
   exploitation_centroids = [c for c in all_centroids if c['type'] == 'Exploitation']
   exploration_centroids = [c for c in all_centroids if c['type'] == 'Exploration']
   ```

2. **Computes ALL cross-type similarities:**
   ```python
   exploitation_inter_similarities = []
   exploration_inter_similarities = []
   
   for exp_centroid in exploitation_centroids:
       for expl_centroid in exploration_centroids:
           similarity = cosine_similarity(exp_centroid, expl_centroid)
           exploitation_inter_similarities.append(similarity)
           exploration_inter_similarities.append(similarity)  # Same pairs, different perspective
   ```

3. **Computes the means:**
   ```python
   exploitation_inter_mean = np.mean(exploitation_inter_similarities)  # Should be ~0.78
   exploration_inter_mean = np.mean(exploration_inter_similarities)    # Should be ~0.33
   ```

---

## Comparison with Local Script

The **local script** `phase_coherence_analysis.py` DOES compute these values, but has a bug:

- **File:** `phase_coherence_analysis.py`
- **Lines 244-267:** Collects inter-phase similarities (but includes same-type pairs - BUG!)
- **Line 305:** `exploitation_inter_mean = np.mean(exploitation_inter_similarities)`
- **Line 323:** `exploration_inter_mean = np.mean(exploration_inter_similarities)`

**The bug:** Lines 262-267 include same-type pairs when they should only include cross-type pairs.

---

## Summary

| Location | Status | Issue |
|----------|--------|-------|
| **Colab Notebook** | ❌ Not computed | Set to `np.nan` - needs implementation |
| **Local Script** | ⚠️ Computed incorrectly | Includes same-type pairs, should only be cross-type |

**To fix:** Both need to compute only Exploitation ↔ Exploration cross-type similarities, then take the mean.






