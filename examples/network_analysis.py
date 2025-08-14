"""
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
