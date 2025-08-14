#!/usr/bin/env python3
"""
Exploratory Data Analysis: Navigating Semantic Space through Exploitation vs Exploration
=====================================================================================

This script analyzes semantic fluency data to understand how individuals navigate 
semantic space through different strategies:
- Exploitation: Staying within semantic clusters/communities
- Exploration: Jumping between different semantic clusters/communities

Author: Dietta Chihade, et al.
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import SNAFU modules
import snafu
from snafu.core import *
from snafu.netstats import *
from snafu.clustering import *
from snafu.search import *
from snafu.triadic import *
from snafu.word_properties import *

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SemanticSpaceAnalyzer:
    """Analyzer for semantic fluency data focusing on exploitation vs exploration patterns."""
    
    def __init__(self, data_path):
        """Initialize the analyzer with fluency data."""
        self.data_path = data_path
        self.data = None
        self.participant_lists = {}
        self.networks = {}
        self.clusters = {}
        self.transition_matrices = {}
        
    def load_data(self):
        """Load and preprocess the fluency data."""
        print("Loading fluency data...")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                self.data = pd.read_csv(self.data_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read CSV file with any encoding")
        
        # Clean the data
        self.data['ID'] = self.data['ID'].astype(str).str.strip("'")
        self.data['Item'] = self.data['Item'].str.strip()
        
        # Group by participant
        for participant_id, group in self.data.groupby('ID'):
            self.participant_lists[participant_id] = group['Item'].tolist()
        
        print(f"Loaded data for {len(self.participant_lists)} participants")
        return self
    
    def analyze_participant(self, participant_id):
        """Analyze a single participant's semantic navigation patterns."""
        if participant_id not in self.participant_lists:
            print(f"Participant {participant_id} not found")
            return None
            
        word_list = self.participant_lists[participant_id]
        print(f"\nAnalyzing participant {participant_id} ({len(word_list)} words)")
        
        # Create semantic network
        network = self._create_semantic_network(word_list)
        self.networks[participant_id] = network
        
        # Analyze clusters
        clusters = self._identify_clusters(network, word_list)
        self.clusters[participant_id] = clusters
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transitions(word_list, clusters)
        self.transition_matrices[participant_id] = transition_matrix
        
        # Calculate exploitation/exploration metrics
        metrics = self._calculate_navigation_metrics(word_list, clusters, transition_matrix)
        
        return {
            'participant_id': participant_id,
            'word_list': word_list,
            'network': network,
            'clusters': clusters,
            'transition_matrix': transition_matrix,
            'metrics': metrics
        }
    
    def _create_semantic_network(self, word_list):
        """Create a semantic network from word list using SNAFU."""
        try:
            # Use SNAFU's conceptual network function
            network = conceptualNetwork([word_list], numnodes=len(set(word_list)))
            return network
        except Exception as e:
            print(f"Error creating network: {e}")
            # Fallback: create simple co-occurrence network
            return self._create_simple_network(word_list)
    
    def _create_simple_network(self, word_list):
        """Create a simple co-occurrence network as fallback."""
        G = nx.Graph()
        G.add_nodes_from(word_list)
        
        # Add edges based on adjacency in the list
        for i in range(len(word_list) - 1):
            G.add_edge(word_list[i], word_list[i+1])
        
        return G
    
    def _identify_clusters(self, network, word_list):
        """Identify semantic clusters in the network."""
        try:
            # Use SNAFU's clustering functions
            clusters = findCommunities(network)
            return clusters
        except Exception as e:
            print(f"Error in clustering: {e}")
            # Fallback: use networkx community detection
            return self._simple_clustering(network)
    
    def _simple_clustering(self, network):
        """Simple clustering using networkx."""
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(network)
            return [list(comm) for comm in communities]
        except:
            # If all else fails, treat each word as its own cluster
            return [[word] for word in network.nodes()]
    
    def _calculate_transitions(self, word_list, clusters):
        """Calculate transition probabilities between clusters."""
        # Map words to clusters
        word_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for word in cluster:
                word_to_cluster[word] = i
        
        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(word_list) - 1):
            current_cluster = word_to_cluster.get(word_list[i], -1)
            next_cluster = word_to_cluster.get(word_list[i+1], -1)
            transitions[current_cluster][next_cluster] += 1
        
        # Convert to probability matrix
        transition_matrix = {}
        for from_cluster in transitions:
            total = sum(transitions[from_cluster].values())
            transition_matrix[from_cluster] = {
                to_cluster: count / total 
                for to_cluster, count in transitions[from_cluster].items()
            }
        
        return transition_matrix
    
    def _calculate_navigation_metrics(self, word_list, clusters, transition_matrix):
        """Calculate exploitation vs exploration metrics."""
        metrics = {}
        
        # 1. Cluster switching rate
        word_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for word in cluster:
                word_to_cluster[word] = i
        
        cluster_sequence = [word_to_cluster.get(word, -1) for word in word_list]
        switches = sum(1 for i in range(len(cluster_sequence)-1) 
                      if cluster_sequence[i] != cluster_sequence[i+1])
        metrics['cluster_switches'] = switches
        metrics['switch_rate'] = switches / (len(word_list) - 1) if len(word_list) > 1 else 0
        
        # 2. Average cluster size
        metrics['avg_cluster_size'] = np.mean([len(cluster) for cluster in clusters])
        
        # 3. Exploitation score (staying within clusters)
        exploitation_transitions = 0
        total_transitions = 0
        for from_cluster in transition_matrix:
            for to_cluster, prob in transition_matrix[from_cluster].items():
                total_transitions += prob
                if from_cluster == to_cluster:
                    exploitation_transitions += prob
        
        metrics['exploitation_score'] = exploitation_transitions / total_transitions if total_transitions > 0 else 0
        metrics['exploration_score'] = 1 - metrics['exploitation_score']
        
        # 4. Entropy of transitions (higher = more exploration)
        entropy = 0
        for from_cluster in transition_matrix:
            probs = list(transition_matrix[from_cluster].values())
            if probs:
                entropy += -sum(p * np.log2(p) for p in probs if p > 0)
        metrics['transition_entropy'] = entropy
        
        # 5. Network density
        if hasattr(self.networks.get(list(self.networks.keys())[0]), 'number_of_edges'):
            network = list(self.networks.values())[0]
            metrics['network_density'] = nx.density(network)
        else:
            metrics['network_density'] = 0
        
        return metrics
    
    def analyze_all_participants(self):
        """Analyze all participants and return summary statistics."""
        print("Analyzing all participants...")
        
        all_metrics = []
        for participant_id in self.participant_lists:
            result = self.analyze_participant(participant_id)
            if result:
                all_metrics.append(result['metrics'])
        
        # Create summary dataframe
        summary_df = pd.DataFrame(all_metrics)
        return summary_df
    
    def plot_navigation_patterns(self, participant_id):
        """Plot navigation patterns for a specific participant."""
        if participant_id not in self.participant_lists:
            print(f"Participant {participant_id} not found")
            return
        
        word_list = self.participant_lists[participant_id]
        clusters = self.clusters.get(participant_id, [])
        network = self.networks.get(participant_id)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Semantic Navigation Analysis - Participant {participant_id}', fontsize=16)
        
        # 1. Word sequence with cluster colors
        word_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for word in cluster:
                word_to_cluster[word] = i
        
        cluster_sequence = [word_to_cluster.get(word, -1) for word in word_list]
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        axes[0, 0].bar(range(len(word_list)), [1]*len(word_list), 
                      color=[colors[cluster_sequence[i] % len(colors)] for i in range(len(word_list))])
        axes[0, 0].set_title('Word Sequence by Cluster')
        axes[0, 0].set_xlabel('Word Position')
        axes[0, 0].set_ylabel('Cluster')
        axes[0, 0].set_xticks(range(len(word_list)))
        axes[0, 0].set_xticklabels(word_list, rotation=45, ha='right')
        
        # 2. Network visualization
        if network and hasattr(network, 'nodes'):
            pos = nx.spring_layout(network, k=1, iterations=50)
            nx.draw(network, pos, ax=axes[0, 1], 
                   node_color=[colors[word_to_cluster.get(node, 0) % len(colors)] for node in network.nodes()],
                   with_labels=True, node_size=1000, font_size=8)
            axes[0, 1].set_title('Semantic Network')
        
        # 3. Transition matrix heatmap
        if participant_id in self.transition_matrices:
            transition_matrix = self.transition_matrices[participant_id]
            if transition_matrix:
                # Convert to matrix format
                cluster_ids = sorted(set([k for d in transition_matrix.values() for k in d.keys()]))
                matrix = np.zeros((len(cluster_ids), len(cluster_ids)))
                
                for i, from_cluster in enumerate(cluster_ids):
                    for j, to_cluster in enumerate(cluster_ids):
                        matrix[i, j] = transition_matrix.get(from_cluster, {}).get(to_cluster, 0)
                
                sns.heatmap(matrix, annot=True, fmt='.2f', ax=axes[1, 0],
                           xticklabels=[f'C{i}' for i in cluster_ids],
                           yticklabels=[f'C{i}' for i in cluster_ids])
                axes[1, 0].set_title('Cluster Transition Matrix')
                axes[1, 0].set_xlabel('To Cluster')
                axes[1, 0].set_ylabel('From Cluster')
        
        # 4. Metrics summary
        if participant_id in [list(self.networks.keys())[0]]:  # Just for first participant
            metrics = self._calculate_navigation_metrics(word_list, clusters, 
                                                       self.transition_matrices.get(participant_id, {}))
            
            metric_names = ['Exploitation Score', 'Exploration Score', 'Switch Rate', 'Transition Entropy']
            metric_values = [metrics['exploitation_score'], metrics['exploration_score'], 
                           metrics['switch_rate'], metrics['transition_entropy']]
            
            bars = axes[1, 1].bar(metric_names, metric_values, color=['red', 'blue', 'green', 'orange'])
            axes[1, 1].set_title('Navigation Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_summary_statistics(self, summary_df):
        """Plot summary statistics across all participants."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Semantic Navigation Summary Statistics', fontsize=16)
        
        # 1. Exploitation vs Exploration distribution
        axes[0, 0].scatter(summary_df['exploitation_score'], summary_df['exploration_score'], alpha=0.7)
        axes[0, 0].set_xlabel('Exploitation Score')
        axes[0, 0].set_ylabel('Exploration Score')
        axes[0, 0].set_title('Exploitation vs Exploration')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Switch rate distribution
        axes[0, 1].hist(summary_df['switch_rate'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Switch Rate')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Cluster Switch Rates')
        
        # 3. Transition entropy distribution
        axes[0, 2].hist(summary_df['transition_entropy'], bins=20, alpha=0.7, color='orange')
        axes[0, 2].set_xlabel('Transition Entropy')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Transition Entropy')
        
        # 4. Average cluster size vs switch rate
        axes[1, 0].scatter(summary_df['avg_cluster_size'], summary_df['switch_rate'], alpha=0.7)
        axes[1, 0].set_xlabel('Average Cluster Size')
        axes[1, 0].set_ylabel('Switch Rate')
        axes[1, 0].set_title('Cluster Size vs Switch Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Network density vs exploitation
        axes[1, 1].scatter(summary_df['network_density'], summary_df['exploitation_score'], alpha=0.7)
        axes[1, 1].set_xlabel('Network Density')
        axes[1, 1].set_ylabel('Exploitation Score')
        axes[1, 1].set_title('Network Density vs Exploitation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Box plot of key metrics
        metrics_to_plot = ['exploitation_score', 'exploration_score', 'switch_rate']
        data_to_plot = [summary_df[metric] for metric in metrics_to_plot]
        axes[1, 2].boxplot(data_to_plot, labels=['Exploitation', 'Exploration', 'Switch Rate'])
        axes[1, 2].set_title('Distribution of Key Metrics')
        axes[1, 2].set_ylabel('Score')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(summary_df.describe())
        
        print("\n=== CORRELATION MATRIX ===")
        correlation_matrix = summary_df.corr()
        print(correlation_matrix.round(3))

def main():
    """Main analysis function."""
    print("=== SEMANTIC SPACE NAVIGATION ANALYSIS ===")
    print("Analyzing exploitation vs exploration patterns in fluency data\n")
    
    # Initialize analyzer
    analyzer = SemanticSpaceAnalyzer('fluency_data/SVF Data1.csv')
    
    # Load data
    analyzer.load_data()
    
    # Analyze all participants
    summary_df = analyzer.analyze_all_participants()
    
    # Plot summary statistics
    analyzer.plot_summary_statistics(summary_df)
    
    # Plot detailed analysis for first few participants
    participant_ids = list(analyzer.participant_lists.keys())[:3]
    for participant_id in participant_ids:
        analyzer.plot_navigation_patterns(participant_id)
    
    # Save results
    summary_df.to_csv('semantic_navigation_results.csv', index=False)
    print(f"\nResults saved to 'semantic_navigation_results.csv'")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Average exploitation score: {summary_df['exploitation_score'].mean():.3f}")
    print(f"Average exploration score: {summary_df['exploration_score'].mean():.3f}")
    print(f"Average switch rate: {summary_df['switch_rate'].mean():.3f}")
    print(f"Average transition entropy: {summary_df['transition_entropy'].mean():.3f}")
    
    # Identify participants with extreme patterns
    high_exploitation = summary_df.loc[summary_df['exploitation_score'].idxmax()]
    high_exploration = summary_df.loc[summary_df['exploration_score'].idxmax()]
    
    print(f"\nMost exploitative participant: {high_exploitation.name} (score: {high_exploitation['exploitation_score']:.3f})")
    print(f"Most explorative participant: {high_exploration.name} (score: {high_exploration['exploration_score']:.3f})")

if __name__ == "__main__":
    main()
