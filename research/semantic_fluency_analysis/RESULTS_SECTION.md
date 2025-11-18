# Results

## Computation of Coherence Ratios

Coherence ratios quantify the degree to which words within each phase type (exploitation or exploration) are more similar to each other than they are to words in other phases. The computation involves several steps, beginning with word embedding extraction and phase classification, followed by similarity calculations at multiple levels of analysis.

### Word Embedding and Phase Classification

**Word Embeddings**: Each word in the semantic fluency sequence was represented as a 300-dimensional vector using the spaCy `en_core_web_md` model, which provides pre-trained word embeddings based on GloVe vectors trained on Common Crawl data. This model achieved 100% coverage of words in our dataset, ensuring that all words could be represented in the semantic space.

**Phase Classification**: Phases were classified as either "Exploitation" or "Exploration" based on consecutive word-pair similarities using a threshold-based algorithm. Specifically:
1. For each consecutive word pair (i, i+1) in the sequence, we computed the cosine similarity between their embedding vectors
2. A similarity threshold of τ = 0.6 was used to classify transitions:
   - If similarity > τ: the transition is classified as "Exploitation" (staying within a semantic cluster)
   - If similarity ≤ τ: the transition is classified as "Exploration" (switching between semantic domains)
3. Phases were required to have a minimum length of 2 words to be included in the analysis
4. Phase transitions occurred when the similarity crossed the threshold and the current phase met the minimum length requirement

This classification procedure segments the fluency sequence into alternating exploitation and exploration phases, where exploitation phases represent periods of semantic clustering and exploration phases represent periods of semantic switching.

### Intra-Phase Similarity Computation

For each phase (exploitation or exploration), we computed pairwise cosine similarities between all word embeddings within that phase. The mathematical procedure was as follows:

**Step 1: Similarity Matrix Construction**
For a phase containing n words with embedding vectors {**v**₁, **v**₂, ..., **v**ₙ}, we constructed an n × n similarity matrix **S** where each element Sᵢⱼ represents the cosine similarity between vectors **v**ᵢ and **v**ⱼ:

\[
S_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{||\mathbf{v}_i|| \cdot ||\mathbf{v}_j||} = \frac{\sum_{k=1}^{300} v_{ik} \cdot v_{jk}}{\sqrt{\sum_{k=1}^{300} v_{ik}^2} \cdot \sqrt{\sum_{k=1}^{300} v_{jk}^2}}
\]

where cosine similarity ranges from -1 to 1, with values closer to 1 indicating greater semantic similarity.

**Step 2: Pairwise Similarity Extraction**
To avoid double-counting and exclude self-similarities (diagonal elements, which are always 1.0), we extracted the upper triangle of the similarity matrix (excluding the diagonal). For a phase with n words, this yields n(n-1)/2 unique pairwise similarities.

**Step 3: Aggregation by Phase Type**
We calculated the mean of all pairwise similarities separately for all exploitation phases and all exploration phases:

\[
\mu_{\text{exploitation}}^{\text{intra}} = \frac{1}{N_{\text{exp}}} \sum_{p \in P_{\text{exp}}} \sum_{i<j} S_{ij}^{(p)}
\]

\[
\mu_{\text{exploration}}^{\text{intra}} = \frac{1}{N_{\text{expl}}} \sum_{p \in P_{\text{expl}}} \sum_{i<j} S_{ij}^{(p)}
\]

where P_exp and P_expl are the sets of exploitation and exploration phases, respectively, and N_exp and N_expl are the total number of pairwise similarities across all phases of each type.

This procedure yields two key metrics:
- **Exploitation intra-phase mean** (μ_exploitation^intra): The average semantic similarity between all pairs of words within exploitation phases
- **Exploration intra-phase mean** (μ_exploration^intra): The average semantic similarity between all pairs of words within exploration phases

### Inter-Phase Similarity Computation

To quantify the similarity between different phases (both within and across phase types), we used a centroid-based approach:

**Step 1: Phase Centroid Calculation**
For each phase p containing n words with embedding vectors {**v**₁, **v**₂, ..., **v**ₙ}, we computed the centroid **c**ₚ as the arithmetic mean of all vectors in that phase:

\[
\mathbf{c}_p = \frac{1}{n} \sum_{i=1}^{n} \mathbf{v}_i
\]

**Step 2: Centroid Normalization**
Each centroid was normalized to unit length to enable cosine similarity computation:

\[
\hat{\mathbf{c}}_p = \frac{\mathbf{c}_p}{||\mathbf{c}_p||} = \frac{\mathbf{c}_p}{\sqrt{\sum_{k=1}^{300} c_{pk}^2}}
\]

This normalization ensures that the centroid represents the direction (semantic theme) of the phase rather than its magnitude, which is appropriate for cosine similarity calculations.

**Step 3: Inter-Phase Similarity Calculation**
We computed cosine similarity between all unique pairs of phase centroids (i.e., between all phases, including both exploitation-exploitation, exploration-exploration, and exploitation-exploration pairs):

\[
S_{\text{inter}}(p, q) = \hat{\mathbf{c}}_p \cdot \hat{\mathbf{c}}_q = \sum_{k=1}^{300} \hat{c}_{pk} \cdot \hat{c}_{qk}
\]

Since the centroids are normalized, this dot product directly yields the cosine similarity.

**Step 4: Inter-Phase Mean**
We calculated the mean of all inter-phase similarities:

\[
\mu^{\text{inter}} = \frac{1}{N_{\text{pairs}}} \sum_{p<q} S_{\text{inter}}(p, q)
\]

where N_pairs is the total number of unique phase pairs (equal to P(P-1)/2 for P total phases).

This yields:
- **Inter-phase mean** (μ^inter): The average semantic similarity between different phases, providing a baseline measure of how distinct phases are from each other

### Coherence Ratio Calculation

The coherence ratios were computed as ratios of intra-phase to inter-phase similarity:

\[
\text{Exploitation Coherence Ratio} = \frac{\mu_{\text{exploitation}}^{\text{intra}}}{\mu^{\text{inter}}}
\]

\[
\text{Exploration Coherence Ratio} = \frac{\mu_{\text{exploration}}^{\text{intra}}}{\mu^{\text{inter}}}
\]

**Interpretation**: A coherence ratio greater than 1.0 indicates that words within a given phase type are, on average, more similar to each other than phases are to each other, reflecting coherent semantic clustering within that phase type. Higher ratios indicate stronger within-phase coherence relative to between-phase separation. For example:
- A ratio of 1.5 indicates that within-phase similarity is 50% higher than between-phase similarity
- A ratio of 0.8 indicates that within-phase similarity is lower than between-phase similarity, suggesting less coherent clustering

**Edge Cases**: If μ^inter = 0 (which would occur only if all phase centroids were orthogonal), the coherence ratios were set to 0 to avoid division by zero. In practice, this case did not occur in our data, as semantic relationships between phases were always non-zero.

### Statistical Properties

The coherence ratios provide a normalized measure that accounts for the overall semantic structure of the fluency sequence. By dividing intra-phase similarity by inter-phase similarity, we control for individual differences in the general semantic relatedness of words produced, allowing the ratios to specifically reflect the degree of phase-specific clustering. This normalization is important because participants may differ in the overall semantic similarity of their word choices, independent of their exploration-exploitation patterns.

## Neurobiological Correlates of Exploration-Exploitation Coherence

To investigate the neural mechanisms underlying exploration-exploitation patterns in semantic fluency, we examined the relationships between neurophysiological markers and coherence ratios across participants. Specifically, we assessed associations between MEG alpha power (alpha_NET_mean) and locus coeruleus (LC) neuromelanin integrity with both exploitation and exploration coherence ratios.

### MEG Alpha Power and Coherence Ratios

We first examined whether MEG alpha power, a marker of cortical inhibition and attention regulation, was associated with exploitation and exploration coherence ratios. Pearson correlation analyses revealed no significant relationships between alpha power and either coherence measure. The relationship between alpha power and exploitation coherence ratio (Figure 3A) showed a weak negative correlation (r = -0.127, p = 0.395, n = 48), which did not reach statistical significance. Similarly, the association between alpha power and exploration coherence ratio (Figure 3B) was non-significant, with a weak positive correlation (r = 0.076, p = 0.599, n = 48). These findings suggest that MEG alpha power, as measured in the current study, does not significantly predict individual differences in exploitation or exploration coherence patterns during semantic fluency performance.

### Locus Coeruleus Integrity and Coherence Ratios

We next investigated whether LC neuromelanin integrity, a structural marker of noradrenergic system function, was associated with coherence ratios. The relationship between LC integrity and exploitation coherence ratio (Figure 3C) showed a weak, non-significant negative correlation (r = -0.085, p = 0.546, n = 48). In contrast, the association between LC integrity and exploration coherence ratio (Figure 3D) revealed a moderate negative correlation that approached statistical significance (r = -0.262, p = 0.073, n = 48). This trend suggests that lower LC integrity may be associated with reduced exploration coherence, though the effect did not reach conventional significance thresholds. The pattern of results indicates that LC integrity may be more relevant to exploration than exploitation coherence, consistent with the role of the noradrenergic system in cognitive flexibility and attentional control.

### Summary

Taken together, these analyses revealed limited associations between the neurophysiological markers examined and coherence ratios. While the relationship between LC integrity and exploration coherence showed a trend toward significance, none of the correlations reached statistical significance. This pattern suggests that the coherence ratios may reflect cognitive processes that are not strongly captured by these particular neurophysiological measures, or that the relationships may be more complex and require larger sample sizes or alternative analytical approaches to detect.

