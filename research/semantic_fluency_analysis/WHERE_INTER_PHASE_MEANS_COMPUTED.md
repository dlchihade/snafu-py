# WHERE INTER-PHASE MEANS ARE COMPUTED

## File: `phase_coherence_analysis.py`

### Step 1: Collecting Inter-Phase Similarities (Lines 244-270)

```python
# Lines 244-248: Initialize lists
exploitation_inter_similarities = []
exploration_inter_similarities = []

# Lines 250-270: Loop through all phase pairs
for i, centroid1 in enumerate(phase_centroids):
    for j, centroid2 in enumerate(phase_centroids):
        if i != j:  # Include all pairs (not just i < j), but track by phase type
            
            # Cosine similarity between centroids
            similarity = np.dot(centroid1['centroid'], centroid2['centroid'])
            
            inter_phase_similarities.append(similarity)
            
            # ⚠️ THIS IS WHERE THE ISSUE IS:
            # If the first phase is exploitation, add to exploitation inter-phase
            if centroid1['type'] == 'Exploitation':
                exploitation_inter_similarities.append(similarity)  # Line 263
            
            # If the first phase is exploration, add to exploration inter-phase
            if centroid1['type'] == 'Exploration':
                exploration_inter_similarities.append(similarity)  # Line 267
```

**Problem:** This includes ALL pairs where the first phase is Exploitation/Exploration, including:
- Exploitation ↔ Exploitation (same-type)
- Exploitation ↔ Exploration (cross-type) ✅
- Exploration ↔ Exploration (same-type)
- Exploration ↔ Exploitation (cross-type) ✅

**Should be:** Only cross-type pairs (Exploitation ↔ Exploration)

---

### Step 2: Computing Exploitation Inter-Phase Mean (Line 305)

```python
# Lines 303-305
if exploitation_inter_similarities:
    exploitation_inter_similarities = np.array(exploitation_inter_similarities)
    exploitation_inter_mean = np.mean(exploitation_inter_similarities)  # ← HERE!
```

**This computes:** Mean of ALL similarities where first phase is Exploitation
- Includes: Exploitation ↔ Exploitation AND Exploitation ↔ Exploration
- **Expected value:** 0.78 (should only include Exploitation ↔ Exploration)

---

### Step 3: Computing Exploration Inter-Phase Mean (Line 323)

```python
# Lines 321-323
if exploration_inter_similarities:
    exploration_inter_similarities = np.array(exploration_inter_similarities)
    exploration_inter_mean = np.mean(exploration_inter_similarities)  # ← HERE!
```

**This computes:** Mean of ALL similarities where first phase is Exploration
- Includes: Exploration ↔ Exploration AND Exploration ↔ Exploitation
- **Expected value:** 0.33 (should only include Exploration ↔ Exploitation)

---

### Step 4: Return Values (Lines 345, 349)

```python
return {
    # ...
    'inter_phase_mean_exploitation': exploitation_inter_mean,  # Line 345
    # ...
    'inter_phase_mean_exploration': exploration_inter_mean,    # Line 349
}
```

---

## Summary: Exact Computation Locations

| Metric | Computation Line | Current Behavior | Expected Behavior |
|--------|----------------|------------------|-------------------|
| **Exploitation Inter-Phase Mean** | **Line 305** | Mean of all pairs where first phase is Exploitation (includes Exploitation↔Exploitation) | Mean of only Exploitation↔Exploration pairs |
| **Exploration Inter-Phase Mean** | **Line 323** | Mean of all pairs where first phase is Exploration (includes Exploration↔Exploration) | Mean of only Exploration↔Exploitation pairs |

---

## The Fix Needed

To match the expected values (0.78 and 0.33), change lines 262-267 to only include cross-type pairs:

```python
# CURRENT CODE (Lines 261-267):
if centroid1['type'] == 'Exploitation':
    exploitation_inter_similarities.append(similarity)

if centroid1['type'] == 'Exploration':
    exploration_inter_similarities.append(similarity)

# SHOULD BE:
if centroid1['type'] == 'Exploitation' and centroid2['type'] == 'Exploration':
    exploitation_inter_similarities.append(similarity)

if centroid1['type'] == 'Exploration' and centroid2['type'] == 'Exploitation':
    exploration_inter_similarities.append(similarity)
```

This ensures:
- `exploitation_inter_similarities` only contains Exploitation ↔ Exploration pairs
- `exploration_inter_similarities` only contains Exploration ↔ Exploitation pairs
- Both lists contain the same pairs (just from different perspectives)
- The means should match the expected values: 0.78 and 0.33






