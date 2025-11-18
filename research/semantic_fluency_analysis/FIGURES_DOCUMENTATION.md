# Figures Documentation

This document provides a comprehensive overview of all figures generated for the semantic fluency analysis study.

**Note**: All figures exclude participants with zero exploitation_intra_mean values (N=45, excluding 1 participant with no exploitation phases).

---

## All Figures (28 Total)

---

## Main Figures

### Figure 0a: Alpha Power Distribution
**File**: `output/NATURE_REAL_figure0_alpha_violin_swarm.png`

**Description**: Violin plot with swarm overlay showing the distribution of MEG alpha power (NET mean) across participants.

**Key Information**:
- Shows individual data points (swarm plot)
- Displays distribution shape (violin plot)
- Useful for assessing data distribution and outliers

---

### Figure 0b: Coherence Distribution
**File**: `output/NATURE_REAL_figure0b_coherence_violin_swarm.png`

**Description**: Violin plot with swarm overlay comparing exploitation and exploration coherence distributions.

**Key Information**:
- Compares two distributions side-by-side
- Shows individual participant values
- Helps visualize differences between exploitation and exploration phases

---

### Figure 1: Exploration vs Exploitation
**File**: `output/NATURE_REAL_figure1_exploration_exploitation.png`

**Description**: Comprehensive visualization of exploration and exploitation patterns across participants.

**Key Information**:
- Shows the relationship between exploration and exploitation metrics
- May include multiple panels showing different aspects of the E-E pattern

---

### Figure 2: Phase Coherence
**File**: `output/NATURE_REAL_figure2_phase_coherence.png`

**Description**: Distribution and comparison of phase coherence metrics.

**Panels**:
- **Panel A**: Exploitation intra-phase mean distribution
- **Panel B**: Exploration intra-phase mean distribution  
- **Panel C**: Inter-phase mean distribution
- **Panel D**: E-E index proxy distribution

**Key Information**:
- Mean values (μ) are displayed for each distribution
- Standard deviations (σ) and ranges are shown
- Sample size: N=45 (excluding zero values)

---

### Figure 3: MEG Correlations
**File**: `output/NATURE_REAL_figure3_meg_correlations.png`

**Description**: Scatter plots showing correlations between MEG alpha power and various coherence metrics.

**Panels**:
- Multiple scatter plots with regression lines
- Pearson correlation coefficients and p-values
- 95% confidence intervals around regression lines

**Key Information**:
- Tests relationships between neurophysiological (MEG) and behavioral (coherence) measures
- Sample size: N=45

---

### Figure 4: Comprehensive Scatter Plots
**File**: `output/NATURE_REAL_figure4_comprehensive.png`

**Description**: Three-panel scatter plot figure showing key relationships in the exploitation-exploration framework.

**Panels**:
- **Panel A**: Exploitation vs Exploration (cosine similarity)
- **Panel B**: Exploitation vs Cluster Switches
- **Panel C**: Exploration vs Novelty

**Statistical Information**:
- Pearson correlation coefficients (r) and p-values
- Linear regression lines with 95% confidence intervals
- Sample size displayed for each panel (n=45)

**Key Findings**:
- Tests relationships between exploitation/exploration coherence and behavioral outcomes
- Confidence intervals show uncertainty in regression estimates

---

### Behavior Performance Metrics
**File**: `output/NATURE_REAL_behavior_performance.png`

**Description**: Multi-panel figure showing behavioral performance metrics from semantic fluency tasks.

**Panels**:
- Distribution of exploration intra-phase mean
- E-E index distribution
- Other behavioral metrics

**Key Information**:
- Mean, median, and distribution statistics
- Sample size: N=45

---

### E-E Index vs MoCA
**File**: `output/NATURE_REAL_ee_vs_moca.png`

**Description**: Scatter plot comparing the E-E index with MoCA (Montreal Cognitive Assessment) scores.

**Key Information**:
- Tests relationship between exploitation-exploration patterns and cognitive function
- Pearson correlation and p-value displayed
- 95% confidence interval around regression line
- Sample size: N=32 (limited by MoCA data availability)

---

### Compiled Results
**File**: `output/NATURE_REAL_compiled_results.png`

**Description**: Composite figure combining multiple key results.

**Panels**:
- E-E index vs MoCA comparison
- LC vs Alpha plot
- Behavior Performance Metrics

---

### Participant Demographics
**File**: `output/NATURE_REAL_participant_values_table.png`

**Description**: Table or figure showing participant demographic and clinical characteristics.

**Key Information**:
- Age distribution
- Disease duration
- Hoehn & Yahr scores
- Cognitive measures (MoCA, SVF)
- Sex/Gender distribution

---

## Intermediate Figures

### Intermediate: Participant Demographics
**File**: `output/figures/intermediate/participant_demographics.png`

**Description**: Expanded demographics figure showing multiple panels of participant characteristics.

**Panels**: Age, Hoehn & Yahr, Disease Duration, Sex/Gender, MoCA, SVF count

---

### Intermediate: Age Distribution
**File**: `output/figures/intermediate/age_distribution.png`

**Description**: Histogram showing the age distribution of participants.

---

### Intermediate: Alpha Power Violin
**File**: `output/figures/intermediate/alpha_power_violin.png`

**Description**: Violin plot showing MEG alpha power distribution.

---

### Intermediate: LC vs Alpha
**File**: `output/figures/intermediate/lc_vs_alpha.png`

**Description**: Scatter plot showing relationship between LC neuromelanin integrity and MEG alpha power.

**Key Information**:
- Includes regression line with confidence intervals
- Correlation coefficient and p-value

---

### Intermediate: SVF vs EE
**File**: `output/figures/intermediate/svf_vs_ee.png`

**Description**: Scatter plot comparing SVF count with E-E index.

---

### Intermediate: SVF EE Box-Swarm
**File**: `output/figures/intermediate/svf_ee_boxswarm.png`

**Description**: Box plot with swarm overlay comparing SVF and E-E metrics.

---

## Additional Analysis Figures

### Alpha vs LC Residuals
**File**: `output/alpha_vs_LC_residuals.png`

**Description**: Scatter plot showing relationship between alpha power and LC residuals.

---

### Semantic Fluency Analysis
**File**: `output/semantic_fluency_analysis.png`

**Description**: Multi-panel figure showing semantic fluency task analysis.

**Panels**:
- Total vs Unique Words
- Repetition Rate by Subject
- Distribution of Total Words
- Most Common Animals

---

### Compiled Distribution Figures
**File**: `output/compiled_distribution_figures.png`

**Description**: Composite figure showing multiple distribution plots.

**Panels**:
- SVF Count distribution
- Exploration Intra-phase Mean distribution
- E-E Index distribution

---

### Exploit Explore Bar Chart
**File**: `output/exploit_explore_bar.png`

**Description**: Bar chart comparing exploitation and exploration metrics.

---

### Combined Mediation Exploit Panels
**File**: `output/combined_mediation_exploit_panels.png`

**Description**: Combined view of multiple mediation analysis panels.

---

### Test Grid Panels
**File**: `output/test_grid_panels.png`

**Description**: Test figure showing grid layout of panels.

---

### LC Residuals Violin Plot
**File**: `output/NATURE_REAL_violin_lc_residuals.png`

**Description**: Violin plot showing distribution of LC residuals.

---

## Mediation Analysis Figures

### Mediation: SVF (Age-adjusted)
**File**: `output/mediation_svf_age_nature.png`

**Description**: Mediation analysis showing the indirect effect of LC integrity on SVF performance through MEG alpha power, controlling for age.

**Components**:
- **Left Panel**: Path diagram showing:
  - Path a: LC integrity → α-power
  - Path b: α-power → SVF
  - Path c': Direct effect (LC → SVF, controlling for mediator)
- **Right Panel**: Bar chart showing:
  - Total effect (c)
  - Direct effect (c')
  - Indirect effect (a×b)

**Key Information**:
- Standardized beta coefficients (β) for each path
- Bootstrap 95% confidence interval for indirect effect
- Sample size: N=45 (excluding zero values)
- Y-axis starts at 0 (all positive effects)

**Statistical Details**:
- All variables z-standardized
- Bootstrap resampling: 5,000 iterations
- Age included as covariate

---

### Mediation: EE Metric (Age-adjusted)
**File**: `output/mediation_exploit_age_nature.png`

**Description**: Mediation analysis for the EE metric (exploitation_coherence_ratio) as the outcome.

**Components**:
- Same structure as SVF mediation figure
- Outcome variable: exploitation_coherence_ratio

**Key Information**:
- Sample size: N=45
- Y-axis symmetric around 0 (mixed positive/negative effects)

---

### Mediation: EE Coherence Metric (Age-adjusted)
**File**: `output/mediation_exploit_coherence_metric_nature.png`

**Description**: Mediation analysis explicitly labeled as "EE coherence metric" (same outcome as EE metric figure, but with updated labeling).

**Components**:
- Same structure as EE metric mediation figure
- Labeled as "EE coherence metric" for clarity

**Key Information**:
- Sample size: N=45
- This figure is kept separate from the "EE metric" figure for comparison

---

### Mediation: SVF (Disease Stage)
**File**: `output/mediation_svf_disease_stage_working.png`

**Description**: Mediation analysis with disease stage as an additional covariate.

**Key Information**:
- Includes age and disease stage as covariates
- Tests whether disease progression moderates the mediation pathway

---

### Mediation: EE (Disease Stage)
**File**: `output/mediation_ee_disease_stage_working.png`

**Description**: Mediation analysis for EE metric with disease stage as covariate.

**Key Information**:
- Includes age and disease stage as covariates
- Tests mediation pathway controlling for disease progression

---

### Mediation: Age vs Disease
**File**: `output/mediation_age_vs_disease.png`

**Description**: Comparison of mediation analyses with age vs disease stage as covariates.

**Key Information**:
- Shows differences between age-adjusted and disease-stage-adjusted models
- Helps assess which covariate better explains the mediation pathway

---

## Statistical Methods

### Correlation Analysis
- **Method**: Pearson correlation
- **Test**: Two-tailed test of H₀: r = 0
- **Implementation**: `scipy.stats.pearsonr`

### Linear Regression
- **Model**: y = β₀ + β₁x + ε
- **Confidence Intervals**: 95% CI using t-distribution (df = n-2)
- **Standard Error**: SE_pred = SE_res × √[1/n + (x - x̄)²/SSx]

### Mediation Analysis
- **Framework**: Baron & Kenny (1986) mediation model
- **Bootstrap**: 5,000 iterations for indirect effect CI
- **Standardization**: All variables z-standardized
- **Covariates**: Age (and disease stage where applicable)

---

## Data Quality Notes

### Zero Values Exclusion
- **Excluded**: 1 participant (PD01156) with zero exploitation_intra_mean
- **Reason**: This participant had no exploitation phases (all words classified as exploration)
- **Impact**: Sample size reduced from N=46 to N=45
- **Justification**: Zero values represent absence of exploitation phases, not missing data

### Sample Sizes by Figure
- **Figures 0-4**: N=45 (complete cases, excluding zeros)
- **E-E vs MoCA**: N=32 (limited by MoCA data availability)
- **Mediation analyses**: N=45 (age-adjusted, excluding zeros)

---

## File Formats

All figures are available in multiple formats:
- **PNG**: High-resolution raster (600 DPI)
- **PDF**: Vector format for publications
- **SVG**: Scalable vector graphics (where applicable)

---

## References

For detailed statistical methods, see:
- `METHODS_DESCRIPTION.md`: Comprehensive methods documentation
- `CONFIDENCE_INTERVAL_EXPLANATION.md`: Explanation of confidence intervals
- `EE_INDEX_EXPLANATION.md`: E-E index calculation details

---

## Figure Generation Scripts

- **Main figures**: `create_nature_quality_figures_real.py`
- **Mediation figures**: `mediation_figures_nature.py`
- **Figure 4 only**: `generate_fig4_only.py` (standalone, no spaCy dependency)

---

## Last Updated

Figures regenerated: November 2024
- Excluding participants with zero exploitation_intra_mean
- Updated sample sizes in figure annotations
- All figures use consistent styling and color schemes

