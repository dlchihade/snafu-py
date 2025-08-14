"""
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
