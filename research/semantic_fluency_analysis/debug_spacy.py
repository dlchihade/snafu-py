#!/usr/bin/env python3
"""
Debug script to isolate the numpy array comparison issue
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.utils import SpacyOptimizer
from src.analyzer import SemanticFluencyAnalyzer

def debug_spacy_issue():
    """Debug the numpy array comparison issue"""
    print("🔍 Debugging spaCy Issue")
    print("=" * 40)
    
    try:
        # Load configuration
        config = AnalysisConfig.from_yaml('config/config.yaml')
        print("✅ Configuration loaded")
        
        # Initialize analyzer
        analyzer = SemanticFluencyAnalyzer(config)
        print("✅ Analyzer initialized")
        
        # Load data
        analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        print("✅ Data loaded")
        
        # Test with one participant
        participant_id = 'PD00020'
        participant_data = analyzer.data[analyzer.data['ID'] == participant_id]
        print(f"Testing with participant: {participant_id}")
        print(f"Number of items: {len(participant_data)}")
        
        # Get items
        items = participant_data['Item'].tolist()
        print(f"Items: {items[:5]}...")  # Show first 5 items
        
        # Test spaCy optimizer directly
        print("\n🔍 Testing SpacyOptimizer directly...")
        vectors, valid_words, valid_indices = analyzer.spacy_optimizer.get_vectors_batch(items)
        print(f"Got {len(vectors)} vectors for {len(valid_words)} valid words")
        
        if len(vectors) >= 2:
            print("✅ Vectors obtained successfully")
            
            # Test similarity calculation
            print("\n🔍 Testing similarity calculation...")
            similarities = analyzer.utils.calculate_similarities_vectorized(vectors)
            print(f"Similarities shape: {similarities.shape}")
            print(f"First few similarities: {similarities[:5]}")
            
            # Test phase identification
            print("\n🔍 Testing phase identification...")
            phases = analyzer._identify_phases(
                similarities, valid_words, vectors, config.similarity_threshold
            )
            print(f"Number of phases: {len(phases)}")
            
            # Test metrics calculation
            print("\n🔍 Testing metrics calculation...")
            metrics = analyzer._calculate_metrics(phases, similarities, valid_words)
            print(f"Metrics: {metrics}")
            
            # Test clustering
            print("\n🔍 Testing clustering...")
            clustering_metrics = analyzer._calculate_clustering_metrics(valid_words, vectors)
            print(f"Clustering metrics: {clustering_metrics}")
            
            # Test frequency stats
            print("\n🔍 Testing frequency stats...")
            frequency_stats = analyzer.utils.get_frequency_stats(valid_words)
            print(f"Frequency stats: {frequency_stats}")
            
        else:
            print("❌ Not enough vectors for analysis")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_spacy_issue()
