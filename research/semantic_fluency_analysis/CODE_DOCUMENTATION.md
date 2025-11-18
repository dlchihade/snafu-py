# Code Documentation

This document provides a comprehensive explanation of all code in the semantic fluency analysis project.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Modules](#core-modules)
4. [Main Scripts](#main-scripts)
5. [Figure Generation Scripts](#figure-generation-scripts)
6. [Mediation Analysis Scripts](#mediation-analysis-scripts)
7. [Data Processing Scripts](#data-processing-scripts)
8. [Utility Scripts](#utility-scripts)
9. [Workflow and Execution](#workflow-and-execution)
10. [Key Functions and Classes](#key-functions-and-classes)

---

## Project Overview

This project analyzes semantic fluency data from Parkinson's disease patients, examining exploitation-exploration patterns, phase coherence metrics, and their relationships with neurophysiological measures (MEG alpha power) and neuroimaging (LC neuromelanin integrity).

**Main Goals**:
- Compute exploitation-exploration metrics from semantic fluency data
- Analyze phase coherence (intra-phase and inter-phase similarity)
- Generate publication-quality figures
- Perform mediation analyses linking LC integrity → α-power → behavioral outcomes

---

## Project Structure

```
research/semantic_fluency_analysis/
├── src/                          # Core analysis modules
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── analyzer.py              # Main analysis engine
│   ├── pipeline.py              # Data processing pipeline
│   ├── utils.py                 # Utility functions (spaCy optimization)
│   └── visualization.py         # Visualization helpers
├── config/                      # Configuration files
│   └── config.yaml              # Main configuration
├── data/                        # Input data files
│   └── fluency_data.csv         # Semantic fluency task data
├── output/                      # Generated figures and results
├── *.py                         # Main scripts (see below)
└── *.md                         # Documentation files
```

---

## Core Modules

### `src/config.py`

**Purpose**: Configuration management using YAML files.

**Key Class**: `AnalysisConfig`
- Loads settings from `config/config.yaml`
- Manages data paths, model settings, analysis parameters
- Provides default values for missing configurations

**Usage**:
```python
from src.config import AnalysisConfig
config = AnalysisConfig.from_yaml('config/config.yaml')
```

---

### `src/analyzer.py`

**Purpose**: Main analysis engine for processing semantic fluency data.

**Key Class**: `SemanticFluencyAnalyzer`

**Main Responsibilities**:
1. **Data Loading**: Loads fluency data and MEG data from CSV files
2. **Word Embedding**: Uses spaCy to generate word embeddings
3. **Phase Detection**: Identifies exploitation and exploration phases
4. **Similarity Computation**: Calculates cosine similarities between words
5. **Metric Calculation**: Computes various coherence and behavioral metrics

**Key Methods**:
- `load_data()`: Loads and preprocesses input data
- `analyze_participant()`: Analyzes a single participant's data
- `detect_phases()`: Identifies exploitation/exploration phases
- `compute_similarities()`: Calculates pairwise cosine similarities

**Dependencies**:
- spaCy for word embeddings (`en_core_web_md` model)
- NumPy, Pandas for data manipulation
- scikit-learn for similarity calculations

---

### `src/pipeline.py`

**Purpose**: Orchestrates the complete analysis pipeline.

**Key Class**: `SemanticFluencyPipeline`

**Functionality**:
- Coordinates data loading, analysis, and result saving
- Handles batch processing of multiple participants
- Manages intermediate results and caching

---

### `src/utils.py`

**Purpose**: Utility functions, particularly for spaCy optimization.

**Key Class**: `SpacyOptimizer`

**Functionality**:
- Optimizes spaCy model loading and usage
- Caches word embeddings to avoid redundant computations
- Manages batch processing of word embeddings

**Why it exists**: spaCy model loading is expensive. This class optimizes performance by:
- Loading the model once and reusing it
- Caching computed embeddings
- Processing words in batches

---

### `src/visualization.py`

**Purpose**: Helper functions for creating visualizations.

**Functionality**:
- Common plotting utilities
- Style presets for publication-quality figures
- Reusable visualization components

---

## Main Scripts

### `main.py`

**Purpose**: Main entry point for running the complete analysis.

**What it does**:
1. Loads configuration
2. Initializes the analyzer
3. Processes all participants
4. Generates results and figures

**Usage**:
```bash
python main.py
```

---

### `phase_coherence_analysis.py`

**Purpose**: Computes phase coherence metrics (intra-phase and inter-phase similarity).

**Key Functions**:

#### `compute_intra_phase_metrics(phases, vectors, verbose=True)`
- **Input**: List of phases (exploitation/exploration), word embedding vectors
- **Output**: Dictionary with intra-phase statistics
- **What it does**:
  - For each phase, computes all pairwise cosine similarities
  - Calculates mean, variance, and standard deviation
  - Separates by phase type (exploitation vs exploration)

**Mathematical Formula**:
```
μ_exploitation^intra = (1/N_exp) × Σ(pairs in exploitation phases) S_ij
μ_exploration^intra = (1/N_expl) × Σ(pairs in exploration phases) S_ij
```

Where `S_ij` is the cosine similarity between words i and j.

#### `compute_inter_phase_metrics(phases, vectors, verbose=True)`
- **Input**: List of phases, word embedding vectors
- **Output**: Dictionary with inter-phase statistics
- **What it does**:
  - Computes centroids for exploitation and exploration phases
  - Calculates cosine similarity between centroids
  - Measures separation between phase types

#### `compute_phase_coherence_metrics_detailed(analyzer, participant_id)`
- **Purpose**: Wrapper function that computes all coherence metrics for a participant
- **Returns**: Dictionary with all metrics (intra-phase means, inter-phase mean, coherence ratios, etc.)

**Key Metrics Computed**:
- `exploitation_intra_mean`: Mean similarity within exploitation phases
- `exploration_intra_mean`: Mean similarity within exploration phases
- `inter_phase_mean`: Similarity between exploitation and exploration centroids
- `exploitation_coherence_ratio`: exploitation_intra_mean / inter_phase_mean
- `exploration_coherence_ratio`: exploration_intra_mean / inter_phase_mean
- `phase_separation_index`: Measure of how distinct the phases are

---

### `create_nature_quality_figures_real.py`

**Purpose**: Generates all publication-quality figures from real data.

**Key Functions**:

#### `setup_nature_style()`
- Configures matplotlib for publication-quality styling
- Sets Arial font, appropriate sizes, removes top/right spines
- Returns color palette (colorblind-friendly, no yellow)

#### `compute_real_metrics()`
- Runs the analyzer on all participants
- Computes coherence metrics for each participant
- Returns DataFrame with all metrics

#### `fig1_exploration_exploitation(df, colors)`
- Generates Figure 1: Exploration vs Exploitation visualization
- Shows the relationship between exploration and exploitation patterns

#### `fig2_phase_coherence(df, colors)`
- Generates Figure 2: Phase Coherence distributions
- **Panels**:
  - A: Exploitation intra-phase mean distribution
  - B: Exploration intra-phase mean distribution
  - C: Inter-phase mean distribution
  - D: E-E index proxy distribution
- Displays mean (μ), standard deviation (σ), and range for each

#### `fig3_meg_correlations(df, colors)`
- Generates Figure 3: MEG Correlations
- Scatter plots showing relationships between MEG alpha power and coherence metrics
- Includes regression lines with 95% confidence intervals
- Pearson correlation coefficients and p-values

#### `fig4_comprehensive(df, colors)`
- Generates Figure 4: Comprehensive Scatter Plots
- **Panels**:
  - A: Exploitation vs Exploration (cosine similarity)
  - B: Exploitation vs Cluster Switches
  - C: Exploration vs Novelty
- **Key Features**:
  - Excludes participants with zero exploitation_intra_mean (N=45)
  - Regression lines with 95% confidence intervals
  - Correlation statistics (r, p, n) displayed on each panel
  - Sample size shown for each panel

**Helper Function**: `add_line_with_ci(ax, x_vals, y_vals, color)`
- Adds regression line with 95% confidence interval shading
- Uses `scipy.stats.linregress` for regression
- Calculates confidence intervals using t-distribution

#### `fig_behavior_performance(df, colors)`
- Generates behavior performance metrics figure
- Shows distributions of key behavioral measures

#### `fig_ee_vs_moca(df, colors)`
- Generates E-E Index vs MoCA comparison
- Scatter plot with regression line and confidence intervals
- Tests relationship between exploitation-exploration patterns and cognitive function

**Main Function**: `main()`
- Orchestrates generation of all figures
- Saves metrics to CSV
- Calls all figure generation functions

---

### `create_intermediate_figures.py`

**Purpose**: Generates intermediate/exploratory figures.

**Key Functions**:

#### `scatter_plot(df, x_col, y_col, colors, out_name, title)`
- Creates scatter plot with regression line and 95% confidence intervals
- Used for LC vs Alpha plot (`lc_vs_alpha.pdf`)
- Calculates Pearson correlation

#### `demographics_figure(df, colors)`
- Generates participant demographics figure
- **Panels** (2x3 grid):
  - Age distribution
  - Hoehn & Yahr score distribution
  - Disease duration distribution
  - Sex/Gender distribution
  - MoCA scores distribution
  - SVF count distribution
- Shows mean/median lines and statistics

**Data Source**: Uses `try_load_and_merge_demographics()` to merge demographic data from multiple sources

---

## Figure Generation Scripts

### `generate_fig4_only.py`

**Purpose**: Standalone script to generate only Figure 4 (avoids spaCy dependency issues).

**Why it exists**: The main script requires spaCy model loading, which can cause environment issues. This script loads data directly from CSV.

**Key Difference**: Loads pre-computed metrics from `final_complete_disease_severity_mediation_data.csv` instead of computing them on-the-fly.

---

### `generate_ee_vs_moca.py`

**Purpose**: Generates E-E Index vs MoCA comparison figure.

**Key Features**:
- Merges MoCA data from multiple sources to maximize sample size
- Uses `try_load_and_merge_demographics()` with specific focus on `cognitive_measure_2` (MoCA)
- Creates scatter plot with regression and confidence intervals

---

### `generate_semantic_fluency_analysis_figure.py`

**Purpose**: Generates comprehensive semantic fluency analysis figure (4 panels).

**Panels**:
- A: Total vs Unique Words (scatter, colored by repetition rate)
- B: Repetition Rate by Subject (bar chart, top 10)
- C: Distribution of Total Words (histogram)
- D: Most Common Animals (horizontal bar chart, top 10)

**Data Source**: `data/fluency_data.csv`

---

### `generate_word_distribution_figures.py`

**Purpose**: Generates two separate distribution figures.

**Outputs**:
1. `distribution_total_words.svg`: Distribution of total words named
2. `distribution_svf_scores.svg`: Distribution of SVF scores

**Key Features**:
- Both distributions use the same data source (`fluency_data.csv`) for consistency
- Common x-axis, y-axis, and bin limits for direct comparison ("to scale")
- Colorblind-friendly colors
- Mean and median lines displayed

---

### `compile_results_figure.py`

**Purpose**: Compiles multiple existing figures into one composite figure.

**Functionality**:
- Loads existing PNG/PDF files
- Trims white margins
- Arranges in grid layout
- Adds panel labels (A, B, C)

**Figures Combined**:
- E-E Index vs MoCA
- LC vs Alpha
- Behavior Performance Metrics

---

### `compile_figures_to_pptx.py` / `compile_all_figures_complete.py`

**Purpose**: Compiles all figures into PowerPoint presentations.

**Functionality**:
- Finds all figure files (PNG, PDF, SVG)
- Creates PowerPoint with one slide per figure/format
- Displays PNG images directly
- Shows file paths for PDF/SVG files

**Output**: `output/all_figures_presentation.pptx` or `output/all_figures_all_formats_complete.pptx`

---

## Mediation Analysis Scripts

### `mediation_analysis.py`

**Purpose**: Core mediation analysis functions.

**Key Functions**:

#### `ols(X, y)`
- **Purpose**: Ordinary Least Squares regression
- **Input**: Design matrix X, outcome vector y
- **Output**: Beta coefficients, standard errors, t-values, p-values
- **Formula**: β = (X'X)⁻¹X'y

#### `mediation(df, B=5000, random_state=42)`
- **Purpose**: Performs Baron & Kenny mediation analysis
- **Paths**:
  - Path a: M ~ X + C (mediator regressed on predictor + covariate)
  - Path b: Y ~ X + M + C (outcome regressed on predictor + mediator + covariate)
  - Path c': Direct effect (coefficient of X in path b model)
  - Path c: Total effect (Y ~ X + C)
- **Bootstrap**: 5,000 iterations for indirect effect confidence interval
- **Returns**: Dictionary with all path coefficients, p-values, indirect effect, CI

---

### `mediation_figures_nature.py`

**Purpose**: Generates Nature-quality mediation figures.

**Key Functions**:

#### `mediation_age_adjusted(df, outcome_col, B=5000, seed=42)`
- **Purpose**: Performs mediation analysis with age as covariate
- **Key Features**:
  - Z-standardizes all variables
  - Excludes participants with zero values for `exploitation_coherence_ratio` (if that's the outcome)
  - Returns dictionary with all path statistics

#### `plot_mediation_nature(result, title, out_path, outcome_type='coherence')`
- **Purpose**: Creates mediation figure with path diagram and effect decomposition
- **Left Panel**: Path diagram showing:
  - Variables (LC integrity, α-power, Outcome)
  - Paths (a, b, c') with standardized beta coefficients
- **Right Panel**: Bar chart showing:
  - Total effect (c)
  - Direct effect (c')
  - Indirect effect (a×b)
  - 95% bootstrap confidence interval for indirect effect
- **Y-axis**: 
  - For SVF: Starts at 0 (all positive effects)
  - For EE: Symmetric around 0 (mixed effects)

#### `plot_mediation_nature_coherence(result, title, out_path, outcome_type='coherence')`
- **Purpose**: Same as `plot_mediation_nature` but explicitly labeled as "EE coherence metric"
- **Why separate**: Allows comparison between "EE metric" and "EE coherence metric" figures

**Main Function**: `main()`
- Runs mediation analyses for:
  - SVF (age-adjusted)
  - EE metric (age-adjusted)
  - EE coherence metric (age-adjusted)
- Generates separate figures for each

---

### `mediation_disease_stage_working.py`

**Purpose**: Mediation analysis with disease stage as additional covariate.

**Key Difference**: Includes both age and disease stage (Hoehn & Yahr score) as covariates.

---

## Data Processing Scripts

### `create_final_disease_severity_data.py`

**Purpose**: Creates final dataset with disease severity variables.

**Functionality**:
- Merges disease severity data (Hoehn & Yahr, disease duration, cognitive complaints)
- Identifies complete cases for mediation analysis
- Saves multiple versions:
  - `final_disease_severity_mediation_data.csv`: All participants
  - `final_complete_disease_severity_mediation_data.csv`: Complete cases only (N=46)

---

### `clean_final_data.py`

**Purpose**: Cleans final comprehensive dataframe by removing duplicates.

**Functionality**:
- Checks for duplicate participant IDs
- Removes duplicates (keeps first occurrence)
- Reports missing values in key variables

---

### `compile_comprehensive_data.py`

**Purpose**: Compiles comprehensive dataset from multiple sources.

**Functionality**:
- Merges data from multiple CSV files
- Handles missing values
- Creates unified dataset for analysis

---

## Utility Scripts

### `run_all_figures_single_script.py`

**Purpose**: One-stop script to generate all figures.

**Functionality**:
- Loads final dataset
- Runs all figure generation functions
- Composes combined outputs (multi-page PDF, 2x2 grid)

---

### `run_all_mediations_single_script.py`

**Purpose**: Runs all mediation analyses in one script.

**Functionality**:
- Loads data
- Runs mediation analyses for multiple outcomes
- Generates all mediation figures

---

### `print_interphase_stats.py`

**Purpose**: Prints summary statistics for inter-phase metrics.

**Functionality**:
- Loads metrics from CSV
- Computes and displays mean, median, std for key variables

---

## Workflow and Execution

### Typical Workflow

1. **Data Preparation**:
   ```bash
   python create_final_disease_severity_data.py
   python clean_final_data.py
   ```

2. **Compute Metrics**:
   ```bash
   python phase_coherence_analysis.py  # Computes coherence metrics
   ```

3. **Generate Figures**:
   ```bash
   python create_nature_quality_figures_real.py  # Main figures
   python create_intermediate_figures.py          # Intermediate figures
   python mediation_figures_nature.py             # Mediation figures
   ```

4. **Compile Results**:
   ```bash
   python compile_results_figure.py              # Composite figures
   python compile_all_figures_complete.py        # PowerPoint
   ```

### Alternative: Single Script Execution

```bash
python run_all_figures_single_script.py    # All figures
python run_all_mediations_single_script.py  # All mediations
```

---

## Key Functions and Classes

### Statistical Functions

#### Confidence Intervals for Regression Lines

**Location**: `create_nature_quality_figures_real.py` → `add_line_with_ci()`

**Formula**:
```
SE_pred = SE_res × √[1/n + (x - x̄)²/SSx]
CI = ŷ ± t(0.975, n-2) × SE_pred
```

Where:
- `SE_res` = standard error of residuals = √(MSE)
- `MSE` = mean squared error
- `SSx` = sum of squared deviations of x from mean
- `t(0.975, n-2)` = 97.5th percentile of t-distribution

#### Mediation Analysis

**Location**: `mediation_analysis.py` → `mediation()`

**Paths**:
- **Path a**: M = β₀ + β₁X + β₂C + ε
- **Path b**: Y = β₀ + β₁X + β₂M + β₃C + ε
- **Path c'**: Direct effect (β₁ from path b)
- **Path c**: Total effect (Y = β₀ + β₁X + β₂C + ε)

**Indirect Effect**: a × b

**Bootstrap CI**: 5,000 iterations, 95% CI = [2.5th, 97.5th] percentiles

---

### Data Filtering

#### Zero Value Exclusion

**Location**: `create_nature_quality_figures_real.py` → `fig4_comprehensive()`

**Code**:
```python
df_filtered = df[df['exploitation_intra_mean'] > 0].copy()
```

**Reason**: Participants with zero `exploitation_intra_mean` have no exploitation phases. This is a legitimate data point but can affect analyses.

**Impact**: Reduces sample size from N=46 to N=45

---

### Figure Styling

#### Nature Style Configuration

**Location**: `create_nature_quality_figures_real.py` → `setup_nature_style()`

**Key Settings**:
- Font: Arial
- DPI: 300 (publication quality)
- Spines: Top and right removed
- Colors: Colorblind-friendly palette (no yellow)
- Grid: Disabled

**Color Palette**:
- Primary: Black (#000000)
- Secondary: Purple (#7B4F9E)
- Accent: Light Blue (#56B4E9)
- Highlight: Teal (#009E73)
- Red: Orange-red (#D55E00)

---

## Dependencies

### Required Packages

```python
# Core
numpy
pandas
matplotlib
scipy

# NLP
spacy
en_core_web_md  # spaCy model (must download separately)

# Visualization
PIL (Pillow)
python-pptx  # For PowerPoint compilation

# Optional
mplcursors  # Interactive hover (optional)
```

### Installation

```bash
pip install numpy pandas matplotlib scipy pillow python-pptx
python -m spacy download en_core_web_md
```

---

## Configuration

### `config/config.yaml`

**Key Settings**:
- Data paths (fluency_data, meg_data)
- spaCy model name
- Analysis parameters
- Output directories

**Usage**:
```python
from src.config import AnalysisConfig
config = AnalysisConfig.from_yaml('config/config.yaml')
```

---

## Output Files

### Generated Figures

**Location**: `output/`

**Main Figures**:
- `NATURE_REAL_figure1_exploration_exploitation.png/pdf`
- `NATURE_REAL_figure2_phase_coherence.png/pdf`
- `NATURE_REAL_figure3_meg_correlations.png/pdf`
- `NATURE_REAL_figure4_comprehensive.png/pdf/svg`

**Mediation Figures**:
- `mediation_svf_age_nature.png/pdf`
- `mediation_exploit_age_nature.png/pdf`
- `mediation_exploit_coherence_metric_nature.png/pdf`

**Compiled**:
- `all_figures_all_formats_complete.pptx`

### Data Files

**Location**: Root directory

**Key Files**:
- `final_complete_disease_severity_mediation_data.csv`: Main dataset (N=46, then N=45 after filtering)
- `NATURE_REAL_metrics.csv`: Computed metrics per participant

---

## Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   - **Error**: `OSError: [E050] Can't find model 'en_core_web_md'`
   - **Solution**: `python -m spacy download en_core_web_md`

2. **Missing Data**
   - **Error**: Empty DataFrame after filtering
   - **Solution**: Check data file paths in `config/config.yaml`

3. **Zero Values**
   - **Issue**: Some participants have zero exploitation_intra_mean
   - **Solution**: Scripts automatically exclude these (see `fig4_comprehensive()`)

4. **PowerPoint PDF/SVG Display**
   - **Issue**: PDF and SVG files can't be displayed directly in PowerPoint
   - **Solution**: File paths are shown on slides; open externally

---

## Code Organization Principles

1. **Separation of Concerns**:
   - Core analysis: `src/` modules
   - Figure generation: Separate scripts
   - Data processing: Dedicated scripts

2. **Reusability**:
   - Common functions in utility modules
   - Shared styling via `setup_nature_style()`
   - Reusable statistical functions

3. **Modularity**:
   - Each figure has its own function
   - Scripts can be run independently
   - Easy to add new figures/analyses

4. **Documentation**:
   - Function docstrings explain purpose
   - Comments explain complex calculations
   - Markdown files document methods and figures

---

## Future Extensions

### Adding New Figures

1. Create function in `create_nature_quality_figures_real.py`:
   ```python
   def fig5_new_analysis(df, colors):
       # Your figure code
       pass
   ```

2. Call in `main()`:
   ```python
   fig5_new_analysis(df, colors)
   ```

3. Add to compilation script if needed

### Adding New Metrics

1. Compute in `phase_coherence_analysis.py`
2. Add to `compute_real_metrics()` output
3. Use in figure generation functions

---

## Summary

This codebase provides a complete pipeline for:
- ✅ Processing semantic fluency data
- ✅ Computing exploitation-exploration metrics
- ✅ Analyzing phase coherence
- ✅ Generating publication-quality figures
- ✅ Performing mediation analyses
- ✅ Compiling results into presentations

All code follows consistent patterns and is well-documented for easy extension and maintenance.

