#!/usr/bin/env python3
"""
Test what values we get if we fix the bug to only include cross-type pairs.
This will show if fixing the bug produces values closer to 0.78 and 0.33.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer

def test_fixed_calculation():
    """Test inter-phase calculation with only cross-type pairs."""
    
    config = AnalysisConfig.from_yaml('config/config.yaml')
    analyzer = SemanticFluencyAnalyzer(config)
    analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
    
    # Test with a few participants
    test_participants = ['PD00020', 'PD00048', 'PD00146', 'PD00215', 'PD00267']
    
    all_exp_cross_type = []
    all_expl_cross_type = []
    
    print("="*70)
    print("TESTING FIXED CALCULATION (Only Cross-Type Pairs)")
    print("="*70)
    
    for pid in test_participants:
        participant_data = analyzer.data[analyzer.data['ID'] == pid]
        if participant_data.empty:
            continue
        
        result = analyzer.analyze_participant(participant_data)
        phases = result['phases']
        
        # Compute phase centroids
        phase_centroids = []
        for i, phase in enumerate(phases):
            if len(phase['vectors']) > 0:
                centroid = np.mean(phase['vectors'], axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 0:
                    centroid_normalized = centroid / centroid_norm
                else:
                    centroid_normalized = centroid
                
                phase_centroids.append({
                    'type': phase['type'],
                    'centroid': centroid_normalized,
                    'phase_index': i
                })
        
        # FIXED: Only cross-type pairs
        exploitation_inter_similarities = []
        exploration_inter_similarities = []
        
        for i, centroid1 in enumerate(phase_centroids):
            for j, centroid2 in enumerate(phase_centroids):
                if i != j:
                    similarity = np.dot(centroid1['centroid'], centroid2['centroid'])
                    
                    # FIXED: Only include cross-type pairs
                    if centroid1['type'] == 'Exploitation' and centroid2['type'] == 'Exploration':
                        exploitation_inter_similarities.append(similarity)
                    
                    if centroid1['type'] == 'Exploration' and centroid2['type'] == 'Exploitation':
                        exploration_inter_similarities.append(similarity)
        
        if exploitation_inter_similarities:
            exp_mean = np.mean(exploitation_inter_similarities)
            all_exp_cross_type.append(exp_mean)
        
        if exploration_inter_similarities:
            expl_mean = np.mean(exploration_inter_similarities)
            all_expl_cross_type.append(expl_mean)
        
        print(f"\n{pid}:")
        if exploitation_inter_similarities:
            print(f"  Exploitation inter (cross-type only): {exp_mean:.4f} (n={len(exploitation_inter_similarities)})")
        if exploration_inter_similarities:
            print(f"  Exploration inter (cross-type only): {expl_mean:.4f} (n={len(exploration_inter_similarities)})")
    
    if all_exp_cross_type and all_expl_cross_type:
        print(f"\n{'='*70}")
        print("GROUP STATISTICS (Fixed - Cross-Type Only)")
        print(f"{'='*70}")
        
        exp_median = np.median(all_exp_cross_type)
        expl_median = np.median(all_expl_cross_type)
        
        print(f"\nExploitation inter-phase mean (cross-type only):")
        print(f"  Count: {len(all_exp_cross_type)}")
        print(f"  Mean: {np.mean(all_exp_cross_type):.4f}")
        print(f"  Median: {exp_median:.4f}")
        
        print(f"\nExploration inter-phase mean (cross-type only):")
        print(f"  Count: {len(all_expl_cross_type)}")
        print(f"  Mean: {np.mean(all_expl_cross_type):.4f}")
        print(f"  Median: {expl_median:.4f}")
        
        print(f"\n{'='*70}")
        print("COMPARISON WITH TEXT VALUES:")
        print(f"{'='*70}")
        print(f"Text reports: Exploitation = 0.78, Exploration = 0.33")
        print(f"\nFixed calculation:")
        print(f"  Exploitation: {exp_median:.4f} (diff from 0.78: {abs(exp_median-0.78):.4f})")
        print(f"  Exploration: {expl_median:.4f} (diff from 0.33: {abs(expl_median-0.33):.4f})")
        
        if abs(exp_median - 0.78) < 0.1 and abs(expl_median - 0.33) < 0.1:
            print(f"\n✅ FIXED CALCULATION MATCHES TEXT VALUES!")
        else:
            print(f"\n⚠️  Fixed calculation still differs from text values")

if __name__ == "__main__":
    test_fixed_calculation()






