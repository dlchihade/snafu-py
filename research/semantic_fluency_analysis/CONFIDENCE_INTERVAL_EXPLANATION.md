# Confidence Interval Explanation for Figure 4

## What is the Confidence Interval?

In the scatter plots of Figure 4, the **gray shaded region** around the regression line represents the **95% confidence interval** for the regression line.

## Statistical Interpretation

### What it shows:
- The confidence interval indicates the **uncertainty** in our estimate of the true relationship between the two variables
- The **true regression line** (the actual relationship in the population) has a **95% probability** of falling within this shaded band
- The width of the band reflects the **precision** of our estimate:
  - **Narrower bands** = more precise estimate (less uncertainty)
  - **Wider bands** = less precise estimate (more uncertainty)

### How it's calculated:
1. A linear regression line is fitted to the data points
2. The standard error of the regression is calculated
3. Using the t-distribution (with n-2 degrees of freedom), a 95% confidence band is constructed around the regression line
4. This accounts for:
   - Sample size (larger samples = narrower intervals)
   - Variability in the data (more scatter = wider intervals)
   - Distance from the mean of x (predictions farther from the mean are less certain)

### Visual Interpretation:
- **Solid black line**: The best-fit regression line through the data
- **Gray shaded region**: The 95% confidence interval band
- **Data points**: Individual participant measurements

### What it tells us:
- If the confidence interval is **narrow**, we can be confident that the relationship is well-estimated
- If the confidence interval is **wide**, there is more uncertainty about the exact relationship
- If the confidence interval **includes zero slope** (horizontal line), the relationship may not be statistically significant

## Example from Figure 4

### Panel A: Exploitation vs Exploration
- The confidence interval shows the uncertainty in the relationship between exploitation and exploration cosine similarity
- A wider band suggests more variability in this relationship across participants

### Panel B: Exploitation vs Cluster Switches  
- The confidence interval around the positive regression line indicates the precision of the relationship
- The band width reflects how consistently exploitation relates to cluster switches

### Panel C: Exploration vs Novelty
- The confidence interval shows uncertainty in the relationship between exploration and novelty scores
- The band helps assess the reliability of this association

## Key Takeaway

The confidence interval is a **measure of statistical uncertainty**, not a prediction interval for individual data points. It tells us how confident we can be about the **average relationship** between variables, not about where individual observations will fall.

