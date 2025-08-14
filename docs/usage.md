# Usage Guide

## Quick Start

```python
import snafu

# Load data
data = snafu.load_data("your_data.csv")

# Analyze
results = snafu.analyze(data)
```

## Data Format

SNAFU expects fluency data in CSV format with columns:
- `subject_id`: Participant identifier
- `word`: Generated word
- `position`: Word position in sequence
- `category`: Semantic category (optional)

## Analysis Methods

### Cluster Analysis
```python
clusters = snafu.analyze_clusters(data)
```

### Network Generation
```python
network = snafu.generate_network(data, method='u-invite')
```

## Output Interpretation

Explanation of analysis results...
