# Methods Description for Figure 4 and Mediation Analysis Figures

## Overview

This document describes the statistical methods and computational approaches used to generate Figure 4 (comprehensive scatter plots) and the mediation analysis figures.

---

## Figure 4: Comprehensive Scatter Plots

### Data Preparation

**Dataset**: `final_complete_disease_severity_mediation_data.csv`
- **Sample size**: N = 46 participants with complete data
- **Variables**:
  - `exploitation_intra_mean`: Mean intra-phase cosine similarity for exploitation phases
  - `exploration_intra_mean`: Mean intra-phase cosine similarity for exploration phases
  - `num_switches`: Count of cluster/phase switches
  - `novelty_score`: Novelty score (arbitrary units)

### Statistical Methods

#### 1. Pearson Correlation Analysis

For each scatter plot, we computed:
- **Pearson correlation coefficient (r)**: Measures linear association between two continuous variables
- **P-value**: Two-tailed test of the null hypothesis that r = 0

**Formula**:
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

**Implementation**: `scipy.stats.pearsonr(x, y)`

#### 2. Linear Regression with 95% Confidence Intervals

For each scatter plot, we fitted a linear regression model:
```
y = β₀ + β₁x + ε
```

**Confidence Interval Calculation**:
- **Standard error of prediction**: 
  ```
  SE_pred = SE_res × √[1/n + (x - x̄)²/SSx]
  ```
  where:
  - `SE_res` = standard error of residuals = √(MSE)
  - `MSE` = mean squared error = Σ(residuals²)/(n-2)
  - `SSx` = sum of squared deviations of x from mean
  - `n` = sample size

- **95% Confidence Interval**:
  ```
  CI = ŷ ± t(0.975, n-2) × SE_pred
  ```
  where `t(0.975, n-2)` is the 97.5th percentile of the t-distribution with n-2 degrees of freedom

**Visualization**:
- **Solid black line**: Regression line (ŷ = β₀ + β₁x)
- **Gray shaded region**: 95% confidence interval band around the regression line
- The confidence interval represents uncertainty in the **regression line estimate**, not prediction intervals for individual observations

#### 3. Panel Descriptions

**Panel A: Exploitation vs Exploration (Cosine Similarity)**
- **X-axis**: Exploitation intra-phase mean (cosine similarity)
- **Y-axis**: Exploration intra-phase mean (cosine similarity)
- **Analysis**: Tests the relationship between exploitation and exploration coherence measures
- **Color**: Light blue markers

**Panel B: Exploitation vs Cluster Switches**
- **X-axis**: Exploitation intra-phase mean (cosine similarity)
- **Y-axis**: Cluster switches (count)
- **Analysis**: Tests whether higher exploitation coherence is associated with more phase transitions
- **Color**: Teal/green markers

**Panel C: Exploration vs Novelty**
- **X-axis**: Exploration intra-phase mean (cosine similarity)
- **Y-axis**: Novelty score (a.u.)
- **Analysis**: Tests the relationship between exploration coherence and novelty
- **Color**: Orange/red markers

### Software and Packages

- **Python 3.x**
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions (`pearsonr`, `linregress`, `t` distribution)
- **Matplotlib**: Figure generation
- **Pandas**: Data manipulation

---

## Mediation Analysis Figures

### Statistical Framework

We used **Baron & Kenny (1986) mediation analysis** with age as a covariate.

### Model Specification

**Predictor (X)**: `norm_LC_avg` - Locus coeruleus (LC) neuromelanin integrity (normalized)

**Mediator (M)**: `alpha_NET_mean` - MEG alpha power (NET mean)

**Outcome (Y)**: 
- `exploitation_coherence_ratio` (for EE metric figures)
- `SVF_count` (for SVF performance figures)

**Covariate**: `Age` - Participant age (standardized)

### Path Analysis

#### Path a: X → M (LC integrity → α-power)
```
M = β₀ + β₁X + β₂Age + ε
```
- Tests whether LC integrity predicts α-power, controlling for age

#### Path b: M → Y (α-power → Outcome)
```
Y = β₀ + β₁X + β₂M + β₃Age + ε
```
- Tests whether α-power predicts the outcome, controlling for LC integrity and age

#### Path c': Direct Effect (X → Y, controlling for M)
```
Y = β₀ + β₁X + β₂M + β₃Age + ε
```
- Direct effect of LC integrity on outcome, controlling for α-power and age

#### Path c: Total Effect (X → Y)
```
Y = β₀ + β₁X + β₂Age + ε
```
- Total effect of LC integrity on outcome, controlling for age

### Effect Decomposition

**Total Effect (c)**: The overall effect of X on Y
```
c = c' + (a × b)
```

**Direct Effect (c')**: The effect of X on Y that does not go through M

**Indirect Effect (a × b)**: The effect of X on Y that is mediated by M
```
Indirect = a × b
```

**Proportion Mediated**:
```
Proportion = (a × b) / c
```

### Statistical Inference

#### Standard Errors and P-values

All path coefficients were estimated using **Ordinary Least Squares (OLS) regression** with standard errors computed as:
```
SE(β) = √[MSE × (X'X)⁻¹]
```

P-values were computed using t-tests:
```
t = β / SE(β)
p = 2 × P(T > |t|)
```
where T follows a t-distribution with n-p degrees of freedom (p = number of parameters).

#### Bootstrap Confidence Intervals for Indirect Effect

The indirect effect (a × b) does not follow a normal distribution, so we used **bootstrap resampling** to compute confidence intervals:

1. **Bootstrap iterations**: B = 5,000
2. **Resampling**: For each iteration:
   - Randomly sample n participants with replacement
   - Re-estimate path a and path b
   - Compute indirect effect: a_boot × b_boot
3. **Confidence interval**: 95% CI = [2.5th percentile, 97.5th percentile] of bootstrap distribution

**Implementation**:
```python
for i in range(B):
    idx = rng.integers(0, n, n)  # Bootstrap sample
    Xb, Mb, Yb, Cb = Xz[idx], Mz[idx], Yz[idx], Cz[idx]
    a_b = ols([ones, Xb, Cb], Mb)[0][1]  # Path a
    b_b = ols([ones, Xb, Mb, Cb], Yb)[0][2]  # Path b
    ab_boot[i] = a_b * b_b
ci_low, ci_high = np.percentile(ab_boot, [2.5, 97.5])
```

### Standardization

All variables were **z-standardized** before analysis:
```
z = (x - μ) / σ
```
where μ is the mean and σ is the standard deviation (using Bessel's correction, ddof=1).

This standardization:
- Makes coefficients interpretable as standardized beta weights
- Does not affect statistical inference on the indirect effect
- Facilitates comparison across different scales

### Figure Components

#### Left Panel: Path Diagram
- **Rectangles**: Variables (LC integrity, α-power, Outcome)
- **Arrows**: Causal paths (a, b, c')
- **Labels**: Standardized beta coefficients (β) for each path
- **Colors**: 
  - Blue: LC integrity
  - Orange: α-power
  - Green/Purple: Outcome (varies by figure)

#### Right Panel: Effect Size Bar Chart
- **Total (c)**: Total effect of LC on outcome
- **Direct (c')**: Direct effect controlling for mediator
- **Indirect (a×b)**: Mediated effect through α-power
- **Y-axis**: 
  - SVF figure: Starts at 0, extends upward (all positive effects)
  - EE figure: Symmetric around 0 (mixed positive/negative effects)
- **Header**: Indirect effect with 95% CI, sample size, and adjustment status

### Software and Packages

- **Python 3.x**
- **NumPy**: Numerical computations and bootstrap resampling
- **SciPy**: Statistical distributions (t-distribution)
- **Matplotlib**: Figure generation
- **Pandas**: Data manipulation

---

## Data Quality and Assumptions

### Missing Data Handling

- **Listwise deletion**: Only participants with complete data on all variables (X, M, Y, Age) were included
- **Sample sizes**: 
  - Figure 4: N = 46 (complete cases)
  - Mediation analyses: N = 46 (age-adjusted)

### Zero Values in Data

**Important Note**: One participant (PD01156) has zero values for:
- `exploitation_intra_mean` = 0.0
- `num_switches` = 0
- `exploitation_coherence_ratio` = 0.0

**Reason**: This participant has **no exploitation phases** in their semantic fluency task. They produced 8 words, all classified as exploration phases. This is a legitimate data point reflecting individual differences in task performance.

**Impact on Analyses**:
- **Scatter plots**: This participant appears as a point at (0, y) in panels where exploitation_intra_mean is on the x-axis
- **Mediation analysis**: Zero values are included in the analysis; z-standardization handles the scale difference
- **Statistical validity**: The zero value is a true zero (absence of exploitation phases), not a missing value or error

**Alternative approaches considered**:
- Excluding this participant would reduce sample size from N=46 to N=45
- We chose to include this participant as it represents valid individual variation in exploitation-exploration patterns

### Statistical Assumptions

#### Linear Regression Assumptions:
1. **Linearity**: Relationship between variables is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed (for inference)

#### Mediation Analysis Assumptions:
1. **Temporal precedence**: X precedes M, M precedes Y
2. **No unmeasured confounding**: All relevant confounders are included (Age)
3. **No X-M interaction**: The effect of M on Y does not depend on X
4. **Linear relationships**: All paths are linear

### Limitations

- **Sample size**: N = 46 limits statistical power
- **Cross-sectional data**: Cannot establish causality
- **Potential confounders**: Age is controlled, but other factors may exist
- **Bootstrap CI**: May not be accurate for very small samples

---

## References

1. Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction in social psychological research: Conceptual, strategic, and statistical considerations. *Journal of Personality and Social Psychology*, 51(6), 1173-1182.

2. Hayes, A. F. (2017). *Introduction to Mediation, Moderation, and Conditional Process Analysis: A Regression-Based Approach* (2nd ed.). Guilford Press.

3. Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. *Behavior Research Methods*, 40(3), 879-891.

---

## Code References

- **Figure 4 generation**: `create_nature_quality_figures_real.py` → `fig4_comprehensive()`
- **Mediation analysis**: `mediation_figures_nature.py` → `mediation_age_adjusted()`
- **Confidence interval calculation**: `create_nature_quality_figures_real.py` → `add_line_with_ci()`

