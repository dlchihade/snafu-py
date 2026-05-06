# WHERE SAME-TYPE PAIRS ARE INCLUDED (Lines 262-267)

## File: `phase_coherence_analysis.py`

## Location: Lines 250-270

```python
for i, centroid1 in enumerate(phase_centroids):
    for j, centroid2 in enumerate(phase_centroids):
        if i != j:  # Include all pairs (not just i < j), but track by phase type
            
            # Cosine similarity between centroids
            similarity = np.dot(centroid1['centroid'], centroid2['centroid'])
            
            inter_phase_similarities.append(similarity)
            
            # ⚠️ PROBLEM: Lines 262-267
            # If the first phase is exploitation, add to exploitation inter-phase
            if centroid1['type'] == 'Exploitation':
                exploitation_inter_similarities.append(similarity)  # Line 263
            
            # If the first phase is exploration, add to exploration inter-phase
            if centroid1['type'] == 'Exploration':
                exploration_inter_similarities.append(similarity)  # Line 267
```

## The Problem

**Lines 262-267 include ALL pairs where the first phase is of that type, including:**

### What Gets Added to `exploitation_inter_similarities`:
- ✅ Exploitation ↔ Exploration (cross-type) - **CORRECT**
- ❌ Exploitation ↔ Exploitation (same-type) - **WRONG!**

### What Gets Added to `exploration_inter_similarities`:
- ✅ Exploration ↔ Exploitation (cross-type) - **CORRECT**
- ❌ Exploration ↔ Exploration (same-type) - **WRONG!**

## Example

If you have phases: [Exploitation1, Exploration1, Exploitation2, Exploration2]

**Current code (WRONG) adds:**
- `exploitation_inter_similarities`: 
  - Exploitation1 ↔ Exploration1 ✅
  - Exploitation1 ↔ Exploitation2 ❌ (same-type!)
  - Exploitation1 ↔ Exploration2 ✅
  - Exploitation2 ↔ Exploration1 ✅
  - Exploitation2 ↔ Exploitation1 (duplicate)
  - Exploitation2 ↔ Exploration2 ✅

- `exploration_inter_similarities`:
  - Exploration1 ↔ Exploitation1 ✅
  - Exploration1 ↔ Exploration2 ❌ (same-type!)
  - Exploration1 ↔ Exploitation2 ✅
  - Exploration2 ↔ Exploitation1 ✅
  - Exploration2 ↔ Exploitation2 ✅
  - Exploration2 ↔ Exploration1 (duplicate)

## The Fix

**Should be (lines 262-267):**

```python
# Only include cross-type pairs
if centroid1['type'] == 'Exploitation' and centroid2['type'] == 'Exploration':
    exploitation_inter_similarities.append(similarity)

if centroid1['type'] == 'Exploration' and centroid2['type'] == 'Exploitation':
    exploration_inter_similarities.append(similarity)
```

This ensures:
- `exploitation_inter_similarities` only contains Exploitation ↔ Exploration pairs
- `exploration_inter_similarities` only contains Exploration ↔ Exploitation pairs
- Both lists contain the same pairs (just from different perspectives)
- No same-type pairs are included






