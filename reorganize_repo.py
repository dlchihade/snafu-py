#!/usr/bin/env python3
"""
Script to reorganize the SNAFU repository for better GitHub structure.
This script creates the recommended directory structure and moves files accordingly.
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the recommended directory structure."""
    
    directories = [
        "docs",
        "examples",
        "examples/jupyter_notebooks",
        "scripts",
        "scripts/data_processing",
        "scripts/analysis", 
        "scripts/visualization",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/test_data",
        "data",
        "data/sample",
        "data/schemes",
        "data/spellfiles",
        "data/networks",
        "research",
        "research/publications",
        "research/experiments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def move_files():
    """Move existing files to their new locations."""
    
    # Move schemes to data/schemes
    if os.path.exists("schemes"):
        for file in os.listdir("schemes"):
            src = os.path.join("schemes", file)
            dst = os.path.join("data/schemes", file)
            if os.path.isfile(src):
                shutil.move(src, dst)
        os.rmdir("schemes")
        print("Moved schemes to data/schemes")
    
    # Move spellfiles to data/spellfiles
    if os.path.exists("spellfiles"):
        for file in os.listdir("spellfiles"):
            src = os.path.join("spellfiles", file)
            dst = os.path.join("data/spellfiles", file)
            if os.path.isfile(src):
                shutil.move(src, dst)
        os.rmdir("spellfiles")
        print("Moved spellfiles to data/spellfiles")
    
    # Move snet files to data/networks
    if os.path.exists("snet"):
        for file in os.listdir("snet"):
            src = os.path.join("snet", file)
            dst = os.path.join("data/networks", file)
            if os.path.isfile(src):
                shutil.move(src, dst)
        os.rmdir("snet")
        print("Moved snet files to data/networks")
    
    # Move demos to examples
    if os.path.exists("demos"):
        for file in os.listdir("demos"):
            src = os.path.join("demos", file)
            dst = os.path.join("examples", file)
            if os.path.isfile(src):
                shutil.move(src, dst)
        os.rmdir("demos")
        print("Moved demos to examples")
    
    # Move semantic_fluency_analysis to research
    if os.path.exists("semantic_fluency_analysis"):
        shutil.move("semantic_fluency_analysis", "research/semantic_fluency_analysis")
        print("Moved semantic_fluency_analysis to research/")
    
    # Move tools (keep as is, just ensure it's in the right place)
    if not os.path.exists("tools"):
        print("Tools directory already in correct location")

def create_example_files():
    """Create example files to demonstrate usage."""
    
    # Basic usage example
    basic_example = '''"""
Basic SNAFU Usage Example

This example demonstrates how to use SNAFU for basic semantic fluency analysis.
"""

import snafu
import pandas as pd

def basic_analysis_example():
    """Demonstrate basic SNAFU functionality."""
    
    # Load sample data
    data = pd.read_csv("data/sample/snafu_sample.csv")
    
    # Analyze clusters
    clusters = snafu.analyze_clusters(data)
    print("Cluster analysis completed")
    
    # Generate network
    network = snafu.generate_network(data, method='u-invite')
    print("Network generated")
    
    # Get statistics
    stats = snafu.network_statistics(network)
    print(f"Network density: {stats['density']:.3f}")
    
    return clusters, network, stats

if __name__ == "__main__":
    basic_analysis_example()
'''
    
    with open("examples/basic_usage.py", "w") as f:
        f.write(basic_example)
    
    # Network analysis example
    network_example = '''"""
Network Analysis Example

This example shows how to perform network analysis with SNAFU.
"""

import snafu
import networkx as nx
import matplotlib.pyplot as plt

def network_analysis_example():
    """Demonstrate network analysis capabilities."""
    
    # Load data
    data = snafu.load_data("data/sample/snafu_sample.csv")
    
    # Generate different types of networks
    networks = {}
    methods = ['u-invite', 'pathfinder', 'correlation']
    
    for method in methods:
        networks[method] = snafu.generate_network(data, method=method)
    
    # Compare networks
    comparison = snafu.compare_networks(networks)
    print("Network comparison completed")
    
    # Visualize networks
    for name, network in networks.items():
        plt.figure(figsize=(8, 6))
        nx.draw(network, with_labels=True, node_color='lightblue')
        plt.title(f"{name.title()} Network")
        plt.savefig(f"examples/{name}_network.png")
        plt.close()
    
    return networks, comparison

if __name__ == "__main__":
    network_analysis_example()
'''
    
    with open("examples/network_analysis.py", "w") as f:
        f.write(network_example)

def create_documentation_files():
    """Create basic documentation files."""
    
    docs = {
        "docs/installation.md": """# Installation Guide

## Prerequisites

- Python 3.5 or higher
- pip package manager

## Installation Methods

### From PyPI (Recommended)
```bash
pip install pysnafu
```

### From Source
```bash
git clone https://github.com/AusterweilLab/snafu-py.git
cd snafu-py
pip install -e .
```

## Verification

```python
import snafu
print(snafu.__version__)
```

## Troubleshooting

Common installation issues and solutions...
""",
        
        "docs/usage.md": """# Usage Guide

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
""",
        
        "docs/api.md": """# API Reference

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
"""
    }
    
    for filepath, content in docs.items():
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created documentation: {filepath}")

def main():
    """Main reorganization function."""
    
    print("Starting SNAFU repository reorganization...")
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # Move existing files
    print("\n2. Moving existing files...")
    move_files()
    
    # Create example files
    print("\n3. Creating example files...")
    create_example_files()
    
    # Create documentation
    print("\n4. Creating documentation...")
    create_documentation_files()
    
    print("\n✅ Repository reorganization completed!")
    print("\nNext steps:")
    print("1. Review the new structure")
    print("2. Update any import paths in your code")
    print("3. Test that everything still works")
    print("4. Commit the changes to git")

if __name__ == "__main__":
    main()
