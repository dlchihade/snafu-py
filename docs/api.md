# API Reference

## Core Functions

### `snafu.load_data(filepath)`
Load fluency data from CSV file.

**Parameters:**
- `filepath` (str): Path to CSV file

**Returns:**
- DataFrame: Loaded data

### `snafu.analyze_clusters(data)`
Analyze semantic clusters in fluency data.

**Parameters:**
- `data` (DataFrame): Fluency data

**Returns:**
- dict: Cluster analysis results

### `snafu.generate_network(data, method='u-invite')`
Generate semantic network from fluency data.

**Parameters:**
- `data` (DataFrame): Fluency data
- `method` (str): Network generation method

**Returns:**
- NetworkX graph: Generated network

## Network Methods

- `u-invite`: U-INVITE networks
- `pathfinder`: Pathfinder networks
- `correlation`: Correlation-based networks
- `naive`: Naive random walk networks
- `conceptual`: Conceptual networks
- `first-edge`: First Edge networks
