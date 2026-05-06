# Animal Frequency Analysis for SVF Data

## How Frequency List is Applied

The frequency list (top 50,000 English words) is applied to your SVF data in the following way:

### Step 1: Each Animal Gets a Rank

For each animal word in your data, the code looks it up in the frequency dictionary:

```python
rank = rank_dict.get(animal.lower(), 0)
```

### Step 2: Rank is Converted to Frequency Score

```python
if rank > 0:
    freq_score = 1.0 - (rank / 50000)
else:
    freq_score = 0.0  # Not in top 50k
```

### Example: PD00020's Animals

PD00020 said: **lion, tiger, sheep, dog, cat, camel, monkey, chimpanzee, buffalo, hyena, dog, cat, elephant, hyena, dog, cat, mouse, bird, camel, dragon**

Here's how frequency would be applied (approximate ranks based on typical English word frequency):

| Animal | Approx Rank | Frequency Score | Category |
|--------|-------------|----------------|----------|
| dog | ~500 | 0.9900 | Very Common |
| cat | ~600 | 0.9880 | Very Common |
| bird | ~800 | 0.9840 | Very Common |
| mouse | ~1,200 | 0.9760 | Common |
| sheep | ~2,500 | 0.9500 | Common |
| horse | ~1,800 | 0.9640 | Common |
| tiger | ~3,500 | 0.9300 | Fairly Common |
| lion | ~4,200 | 0.9160 | Fairly Common |
| elephant | ~5,500 | 0.8900 | Fairly Common |
| monkey | ~6,200 | 0.8760 | Fairly Common |
| camel | ~8,500 | 0.8300 | Less Common |
| buffalo | ~9,200 | 0.8160 | Less Common |
| hyena | ~12,000 | 0.7600 | Less Common |
| chimpanzee | ~15,000 | 0.7000 | Less Common |
| dragon | ~18,000 | 0.6400 | Less Common |

### Step 3: Frequency Transitions

For consecutive word pairs, the code calculates:

```python
transition = freq_score(word2) - freq_score(word1)
```

**Example transitions for PD00020:**

| Transition | Score1 | Score2 | Transition | Interpretation |
|------------|--------|--------|------------|----------------|
| dog → cat | 0.9900 | 0.9880 | -0.0020 | Similar (both very common) |
| cat → camel | 0.9880 | 0.8300 | -0.1580 | **Strong Exploration** (common→rare) |
| camel → monkey | 0.8300 | 0.8760 | +0.0460 | Moderate Exploitation (rare→common) |
| elephant → hyena | 0.8900 | 0.7600 | -0.1300 | **Strong Exploration** (common→rare) |
| bird → camel | 0.9840 | 0.8300 | -0.1540 | **Strong Exploration** (common→rare) |
| camel → dragon | 0.8300 | 0.6400 | -0.1900 | **Very Strong Exploration** (common→rare) |

### Step 4: Used in Phase Detection

The frequency transitions are combined with semantic similarity:

```python
combined_score = 0.7 * semantic_similarity + 0.3 * frequency_similarity
```

Where:
- **Negative frequency transitions** (common→rare) = exploration signal
- **Positive frequency transitions** (rare→common) = exploitation signal

#### Real Example: "cat → camel" transition from PD00020

Let's break down a real transition from your data:

**Word Pair:** cat → camel

**Step 1: Calculate Semantic Similarity**
- Get word vectors from spaCy: `cat.vector` and `camel.vector`
- Calculate cosine similarity: `semantic_sim = cosine_similarity(cat.vector, camel.vector)`
- Example result: **semantic_sim = 0.45** (moderately similar - both are mammals)

**Step 2: Calculate Frequency Transition**
- cat frequency score: ~0.9880 (rank ~600, very common)
- camel frequency score: ~0.8300 (rank ~8,500, less common)
- Frequency transition: `0.8300 - 0.9880 = -0.1580` (strong exploration signal)

**Step 3: Normalize Frequency Transition to [0, 1]**
- Raw frequency transition: -0.1580 (range: -1 to +1)
- Normalized: `freq_sim = (-0.1580 + 1) / 2 = 0.4210`
- This maps: negative transitions → low values, positive transitions → high values

#### What Constitutes Negative vs Positive Transitions?

**Negative Transitions (Common → Rare) = Exploration Signal**

A **negative transition** occurs when you go from a **common word** (high frequency score) to a **rare word** (low frequency score):

```
transition = freq_score(word2) - freq_score(word1)
```

If `word1` is more common than `word2`, the result is **negative**.

**Examples of Negative Transitions (Exploration):**

| Word1 | Score1 | Word2 | Score2 | Transition | Interpretation |
|-------|--------|-------|--------|------------|----------------|
| cat | 0.9880 | camel | 0.8300 | **-0.1580** | Common → Rare (Exploration) |
| dog | 0.9900 | dragon | 0.6400 | **-0.3500** | Very Common → Rare (Strong Exploration) |
| bird | 0.9840 | hyena | 0.7600 | **-0.2240** | Common → Rare (Exploration) |
| elephant | 0.8900 | chimpanzee | 0.7000 | **-0.1900** | Common → Rare (Exploration) |

**Why negative?** Because:
- `freq_score(camel) - freq_score(cat) = 0.8300 - 0.9880 = -0.1580`
- The second word is LESS common than the first
- This indicates **exploration** (moving to less frequent words)

**Positive Transitions (Rare → Common) = Exploitation Signal**

A **positive transition** occurs when you go from a **rare word** (low frequency score) to a **common word** (high frequency score):

**Examples of Positive Transitions (Exploitation):**

| Word1 | Score1 | Word2 | Score2 | Transition | Interpretation |
|-------|--------|-------|--------|------------|----------------|
| camel | 0.8300 | dog | 0.9900 | **+0.1600** | Rare → Common (Exploitation) |
| dragon | 0.6400 | cat | 0.9880 | **+0.3480** | Rare → Very Common (Strong Exploitation) |
| hyena | 0.7600 | bird | 0.9840 | **+0.2240** | Rare → Common (Exploitation) |
| chimpanzee | 0.7000 | elephant | 0.8900 | **+0.1900** | Rare → Common (Exploitation) |

**Why positive?** Because:
- `freq_score(dog) - freq_score(camel) = 0.9900 - 0.8300 = +0.1600`
- The second word is MORE common than the first
- This indicates **exploitation** (returning to more frequent words)

**Zero/Near-Zero Transitions (Similar Frequency) = No Strong Signal**

When both words have similar frequency scores, the transition is near zero:

| Word1 | Score1 | Word2 | Score2 | Transition | Interpretation |
|-------|--------|-------|--------|------------|----------------|
| dog | 0.9900 | cat | 0.9880 | **-0.0020** | Similar frequency (both very common) |
| lion | 0.9160 | tiger | 0.9300 | **+0.0140** | Similar frequency (both fairly common) |
| camel | 0.8300 | buffalo | 0.8160 | **-0.0140** | Similar frequency (both less common) |

**Summary:**
- **Negative** = Common → Rare = **Exploration** (moving to less frequent words)
- **Positive** = Rare → Common = **Exploitation** (returning to more frequent words)
- **Near Zero** = Similar frequency = **No strong signal** (staying in similar frequency range)

**Step 4: Combine the Scores**
```python
semantic_weight = 0.7
frequency_weight = 0.3

combined_score = (0.7 × 0.45) + (0.3 × 0.4210)
                = 0.315 + 0.1263
                = 0.4413
```

**Step 5: Compare to Threshold**
- If threshold = 0.50 (mean similarity across all participants)
- Combined score = 0.4413 < 0.50
- **Result: This transition indicates EXPLORATION** (below threshold)

**Why this matters:**
- Semantic similarity alone (0.45) suggests moderate similarity
- But frequency transition (-0.1580) shows strong exploration (common→rare)
- Combined score (0.4413) captures BOTH signals, better identifying the exploration switch

#### Another Example: "dog → cat" transition

**Word Pair:** dog → cat

**Semantic Similarity:** ~0.75 (very similar - both common pets)

**Frequency Transition:**
- dog: 0.9900 (rank ~500)
- cat: 0.9880 (rank ~600)
- Transition: `0.9880 - 0.9900 = -0.0020` (minimal change)

**Normalized Frequency:** `(-0.0020 + 1) / 2 = 0.4990`

**Combined Score:**
```python
combined = (0.7 × 0.75) + (0.3 × 0.4990)
         = 0.525 + 0.1497
         = 0.6747
```

**Result:** Combined score = 0.6747 > 0.50 threshold → **EXPLOITATION** (high similarity, similar frequency)

#### Comparison Table

| Transition | Semantic Sim | Freq Trans | Norm Freq | Combined | Phase |
|------------|--------------|------------|-----------|----------|-------|
| dog → cat | 0.75 | -0.0020 | 0.4990 | **0.6747** | Exploitation |
| cat → camel | 0.45 | -0.1580 | 0.4210 | **0.4413** | Exploration |
| camel → monkey | 0.52 | +0.0460 | 0.5230 | **0.5209** | Exploitation |
| elephant → hyena | 0.38 | -0.1300 | 0.4350 | **0.3995** | Exploration |

**Key Insight:** The combined approach catches exploration transitions that semantic similarity alone might miss (like cat→camel), because it detects the frequency drop even when semantic similarity is moderate.

### Step 5: Phase-Level Metrics

For each phase, the code calculates:

- **Average frequency score**: Mean frequency of all words in the phase
- **Frequency diversity**: Standard deviation of frequencies (higher = more diverse)
- **Exploration transitions**: Count of transitions where freq_trans < -0.1

## Summary

The frequency list helps identify:
1. **Exploitation phases**: Words are common (high frequency scores)
2. **Exploration phases**: Words are rare (low frequency scores) OR transitions show common→rare
3. **Phase boundaries**: Large negative transitions indicate exploration switches

This complements semantic similarity by adding a lexical frequency dimension to phase detection.

