# Detailed Explanation: NATURE_REAL_behavior_performance.pdf

## Overview

This figure (`NATURE_REAL_behavior_performance.pdf`) presents four panels analyzing behavioral performance metrics from semantic fluency data, with a focus on the Exploitation-Exploration (E–E) index.

---

## Panel A: Exploration Intra-Phase Mean Distribution

### What it shows:
A histogram displaying the distribution of `exploration_intra_mean` across participants.

### Mathematical Definition:

The exploration intra-phase mean (μ_exploration^intra) is calculated as:

\[
\mu_{\text{exploration}}^{\text{intra}} = \frac{1}{N_{\text{expl}}} \sum_{p \in P_{\text{expl}}} \sum_{i<j} S_{ij}^{(p)}
\]

Where:
- **S_ij^(p)** = cosine similarity between words i and j within exploration phase p
- **P_expl** = set of all exploration phases for a participant
- **N_expl** = total number of pairwise similarities across all exploration phases
- **i < j** ensures we only count each pair once (upper triangle of similarity matrix)

### Interpretation:
- **Higher values** (closer to 1.0) indicate that words within exploration phases are more semantically similar to each other
- **Lower values** indicate less semantic coherence within exploration phases
- This metric quantifies how "tightly clustered" exploration phases are semantically

### From the code:
```python
expl_intra = df['exploration_intra_mean'].to_numpy(float)
vals = expl_intra[np.isfinite(expl_intra)]
muA = float(np.mean(vals))  # Mean across participants
sdA = float(np.std(vals, ddof=1))  # Standard deviation
```

---

## Panel B: E–E Index (Exploitation-Exploration Index) Distribution

### What it shows:
A histogram displaying the distribution of the E–E index proxy across participants.

### Mathematical Definition:

The E–E index is calculated **per participant** as:

\[
\text{EE\_index}_i = \frac{\text{exploitation\_coherence\_ratio}_i}{\text{exploration\_coherence\_ratio}_i}
\]

Where the coherence ratios are defined as:

\[
\text{exploitation\_coherence\_ratio} = \frac{\mu_{\text{exploitation}}^{\text{intra}}}{\mu^{\text{inter}}}
\]

\[
\text{exploration\_coherence\_ratio} = \frac{\mu_{\text{exploration}}^{\text{intra}}}{\mu^{\text{inter}}}
\]

And μ^inter is the mean cosine similarity between different phases (inter-phase mean).

### Simplification:

Since both ratios share the same denominator (μ^inter), the E–E index simplifies to:

\[
\text{EE\_index} = \frac{\mu_{\text{exploitation}}^{\text{intra}} / \mu^{\text{inter}}}{\mu_{\text{exploration}}^{\text{intra}} / \mu^{\text{inter}}} = \frac{\mu_{\text{exploitation}}^{\text{intra}}}{\mu_{\text{exploration}}^{\text{intra}}}
\]

**The inter-phase mean cancels out!** The E–E index is simply the ratio of exploitation intra-phase mean to exploration intra-phase mean.

### From the code:
```python
explo_ratio = df['exploitation_coherence_ratio'].to_numpy(float)
expo_ratio = df['exploration_coherence_ratio'].to_numpy(float)

mask_ee = np.isfinite(explo_ratio) & np.isfinite(expo_ratio) & (expo_ratio > 0)
ee_index = (explo_ratio[mask_ee] / expo_ratio[mask_ee])
ee_index = ee_index[np.isfinite(ee_index)]

muB = float(np.mean(ee_index))  # Mean of individual EE indices
sdB = float(np.std(ee_index, ddof=1))
```

### Interpretation:
- **EE index > 1.0**: Exploitation phases have higher intra-phase similarity than exploration phases
  - Participant shows stronger semantic clustering during exploitation
- **EE index < 1.0**: Exploration phases have higher intra-phase similarity than exploitation phases
  - Participant shows stronger semantic clustering during exploration
- **EE index = 1.0**: Equal intra-phase similarity in both phase types

### Important Note on Calculation:

The figure displays the **mean of individual EE indices** (one per participant):

\[
\bar{\text{EE}} = \frac{1}{n} \sum_{i=1}^{n} \text{EE\_index}_i
\]

This is **NOT** the same as the ratio of means:

\[
\frac{\bar{\mu}_{\text{exploitation}}^{\text{intra}}}{\bar{\mu}_{\text{exploration}}^{\text{intra}}} \neq \frac{1}{n} \sum_{i=1}^{n} \frac{\mu_{\text{exploitation},i}^{\text{intra}}}{\mu_{\text{exploration},i}^{\text{intra}}}
\]

The mean of ratios and the ratio of means are generally different due to Jensen's inequality.

### Regarding the value 1.37:

If the expected mean EE index should be **1.37**, this could be:
1. The **ratio of means** (mean exploitation intra / mean exploration intra) rather than the mean of individual ratios
2. A value from a different dataset or calculation method
3. A theoretical or expected value based on prior research

To check which calculation gives 1.37, we would need to compute:
- Mean of individual ratios: `mean(EE_index_per_participant)`
- Ratio of means: `mean(exploitation_intra_mean) / mean(exploration_intra_mean)`

---

## Panel C: Exploitation-Dominant Proportion

### What it shows:
The proportion of participants with EE index > 1.0.

### Calculation:
```python
prop_exploit = float((ee_index > 1.0).sum()) / float(len(ee_index))
```

### Interpretation:
- Percentage of participants who show stronger semantic clustering during exploitation phases
- If > 50%, most participants are exploitation-dominant
- If < 50%, most participants are exploration-dominant

---

## Panel D: Summary Statistics

### What it shows:
A text box displaying:
1. **Exploration intra mean**: μ and σ from Panel A
2. **E–E index**: μ, σ, and range from Panel B
3. **SVF vs MoCA correlation**: If available
4. **E–E vs UPDRS-III correlation**: If available

### From the code:
```python
lines.append(f'Exploration intra mean: μ={muA:.2f}, σ={sdA:.2f}')
lines.append(f'E–E index (proxy): μ={muB:.2f}, σ={sdB:.2f}, range={rngB[0]:.2f}–{rngB[1]:.2f}')
```

---

## Complete Mathematical Chain

### Step 1: Compute Intra-Phase Similarities

For each phase (exploitation or exploration):
1. Extract word embeddings: {**v**₁, **v**₂, ..., **v**ₙ}
2. Compute pairwise cosine similarities: S_ij = (**v**ᵢ · **v**ⱼ) / (||**v**ᵢ|| ||**v**ⱼ||)
3. Average all pairs: μ^intra = mean(S_ij for all i < j)

### Step 2: Compute Inter-Phase Similarity

1. Compute phase centroids: **c**ₚ = mean(**v**ᵢ for all i in phase p)
2. Normalize centroids: **ĉ**ₚ = **c**ₚ / ||**c**ₚ||
3. Compute centroid similarities: S_inter(p,q) = **ĉ**ₚ · **ĉ**_q
4. Average all pairs: μ^inter = mean(S_inter(p,q) for all p < q)

### Step 3: Compute Coherence Ratios

\[
\text{exploitation\_coherence\_ratio} = \frac{\mu_{\text{exploitation}}^{\text{intra}}}{\mu^{\text{inter}}}
\]

\[
\text{exploration\_coherence\_ratio} = \frac{\mu_{\text{exploration}}^{\text{intra}}}{\mu^{\text{inter}}}
\]

### Step 4: Compute E–E Index

\[
\text{EE\_index} = \frac{\text{exploitation\_coherence\_ratio}}{\text{exploration\_coherence\_ratio}} = \frac{\mu_{\text{exploitation}}^{\text{intra}}}{\mu_{\text{exploration}}^{\text{intra}}}
\]

### Step 5: Aggregate Across Participants

\[
\bar{\text{EE}} = \frac{1}{n} \sum_{i=1}^{n} \text{EE\_index}_i
\]

---

## Key Insights

1. **The E–E index is a normalized measure**: By using the ratio of coherence ratios (which share the same denominator), it becomes a simple ratio of intra-phase means, independent of inter-phase similarity.

2. **Interpretation is relative**: An EE index of 1.5 means exploitation phases are 50% more coherent than exploration phases, not that they are absolutely coherent.

3. **Individual vs. aggregate**: The figure shows the distribution of individual EE indices, not a single aggregate value. This allows us to see variability across participants.

4. **The value 1.37**: If this is the expected mean, it suggests that on average, exploitation phases are 37% more semantically coherent than exploration phases across the sample.

---

## Code Location

The figure is generated in:
- **File**: `research/semantic_fluency_analysis/create_nature_quality_figures_real.py`
- **Function**: `fig_behavior_performance(df, colors)` (lines 1422-1523)

