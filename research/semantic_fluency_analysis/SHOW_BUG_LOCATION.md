# WHERE THE BUG OCCURS - Exact Location

## File: `phase_coherence_analysis.py`

## Lines 262-267: The Bug

```python
# Line 250-270
for i, centroid1 in enumerate(phase_centroids):
    for j, centroid2 in enumerate(phase_centroids):
        if i != j:
            similarity = np.dot(centroid1['centroid'], centroid2['centroid'])
            
            inter_phase_similarities.append(similarity)
            
            # ⚠️ BUG HERE: Lines 262-267
            # If the first phase is exploitation, add to exploitation inter-phase
            if centroid1['type'] == 'Exploitation':  # Line 262
                exploitation_inter_similarities.append(similarity)  # Line 263
                # ❌ PROBLEM: This includes Exploitation↔Exploitation pairs!
            
            # If the first phase is exploration, add to exploration inter-phase
            if centroid1['type'] == 'Exploration':  # Line 266
                exploration_inter_similarities.append(similarity)  # Line 267
                # ❌ PROBLEM: This includes Exploration↔Exploration pairs!
```

## The Problem

**Current code only checks `centroid1['type']`, not `centroid2['type']`**

This means:
- When `centroid1` is Exploitation, it adds ALL pairs (Exploitation↔Exploitation AND Exploitation↔Exploration)
- When `centroid1` is Exploration, it adds ALL pairs (Exploration↔Exploration AND Exploration↔Exploitation)

## The Fix

**Change lines 262-267 to:**

```python
# Only include cross-type pairs (Exploitation ↔ Exploration)
if centroid1['type'] == 'Exploitation' and centroid2['type'] == 'Exploration':
    exploitation_inter_similarities.append(similarity)

if centroid1['type'] == 'Exploration' and centroid2['type'] == 'Exploitation':
    exploration_inter_similarities.append(similarity)
```

This ensures:
- ✅ Only Exploitation↔Exploration pairs are included
- ✅ No same-type pairs (Exploitation↔Exploitation or Exploration↔Exploration)
- ✅ Both lists contain the same cross-type pairs (just from different perspectives)

## Why This Matters

The text reports:
- Exploitation inter-phase mean: **0.78**
- Exploration inter-phase mean: **0.33**

These values suggest they were computed using **only cross-type pairs**. The current buggy code produces different values because it includes same-type pairs, which inflate the means.






