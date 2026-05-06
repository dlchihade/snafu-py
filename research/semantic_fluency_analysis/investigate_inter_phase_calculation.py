#!/usr/bin/env python3
"""
Investigate the inter-phase calculation to understand the discrepancy.
Test different calculation methods to see which matches the figure values.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer

def analyze_single_participant(participant_id):
    """Analyze one participant to understand the inter-phase calculation."""
    
    config = AnalysisConfig.from_yaml('config/config.yaml')
    analyzer = SemanticFluencyAnalyzer(config)
    analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
    
    participant_data = analyzer.data[analyzer.data['ID'] == participant_id]
    if participant_data.empty:
        return None
    
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
    
    print(f"\n{'='*70}")
    print(f"Participant: {participant_id}")
    print(f"Number of phases: {len(phase_centroids)}")
    print(f"Phase types: {[p['type'] for p in phase_centroids]}")
    print(f"{'='*70}")
    
    # Method 1: Current implementation (includes same-type pairs)
    exploitation_inter_similarities_method1 = []
    exploration_inter_similarities_method1 = []
    
    for i, centroid1 in enumerate(phase_centroids):
        for j, centroid2 in enumerate(phase_centroids):
            if i != j:
                similarity = np.dot(centroid1['centroid'], centroid2['centroid'])
                
                if centroid1['type'] == 'Exploitation':
                    exploitation_inter_similarities_method1.append(similarity)
                if centroid1['type'] == 'Exploration':
                    exploration_inter_similarities_method1.append(similarity)
    
    # Method 2: Only cross-type pairs
    exploitation_inter_similarities_method2 = []
    exploration_inter_similarities_method2 = []
    
    for i, centroid1 in enumerate(phase_centroids):
        for j, centroid2 in enumerate(phase_centroids):
            if i != j:
                similarity = np.dot(centroid1['centroid'], centroid2['centroid'])
                
                # Only cross-type pairs
                if centroid1['type'] == 'Exploitation' and centroid2['type'] == 'Exploration':
                    exploitation_inter_similarities_method2.append(similarity)
                if centroid1['type'] == 'Exploration' and centroid2['type'] == 'Exploitation':
                    exploration_inter_similarities_method2.append(similarity)
    
    # Method 3: Only when transitioning FROM that type
    # (This might be what "inter-phase" means in the figure context)
    
    print(f"\nMethod 1 (Current - includes same-type pairs):")
    if exploitation_inter_similarities_method1:
        print(f"  Exploitation inter-phase: n={len(exploitation_inter_similarities_method1)}, mean={np.mean(exploitation_inter_similarities_method1):.4f}")
    if exploration_inter_similarities_method1:
        print(f"  Exploration inter-phase: n={len(exploration_inter_similarities_method1)}, mean={np.mean(exploration_inter_similarities_method1):.4f}")
    
    print(f"\nMethod 2 (Only cross-type pairs):")
    if exploitation_inter_similarities_method2:
        print(f"  Exploitation inter-phase: n={len(exploitation_inter_similarities_method2)}, mean={np.mean(exploitation_inter_similarities_method2):.4f}")
    if exploration_inter_similarities_method2:
        print(f"  Exploration inter-phase: n={len(exploration_inter_similarities_method2)}, mean={np.mean(exploration_inter_similarities_method2):.4f}")
    
    # Check if Method 2 values are closer to figure values (0.79, 0.33)
    if exploitation_inter_similarities_method2 and exploration_inter_similarities_method2:
        exp_mean = np.mean(exploitation_inter_similarities_method2)
        expl_mean = np.mean(exploration_inter_similarities_method2)
        print(f"\n  Comparison with figure:")
        print(f"    Figure: Exploitation=0.79, Exploration=0.33")
        print(f"    Method 2: Exploitation={exp_mean:.4f}, Exploration={expl_mean:.4f}")
        print(f"    Difference: Exploitation={abs(exp_mean-0.79):.4f}, Exploration={abs(expl_mean-0.33):.4f}")
    
    return {
        'method1': {
            'exploitation': exploitation_inter_similarities_method1,
            'exploration': exploration_inter_similarities_method1
        },
        'method2': {
            'exploitation': exploitation_inter_similarities_method2,
            'exploration': exploration_inter_similarities_method2
        }
    }

if __name__ == "__main__":
    # Test with a few participants
    test_participants = ['PD00020', 'PD00048', 'PD00146']
    
    all_results = {}
    for pid in test_participants:
        result = analyze_single_participant(pid)
        if result:
            all_results[pid] = result
    
    # Compute group statistics for Method 2
    print(f"\n{'='*70}")
    print("GROUP STATISTICS (Method 2 - Cross-type pairs only)")
    print(f"{'='*70}")
    
    all_exp_method2 = []
    all_expl_method2 = []
    
    for pid, result in all_results.items():
        all_exp_method2.extend(result['method2']['exploitation'])
        all_expl_method2.extend(result['method2']['exploration'])
    
    if all_exp_method2:
        print(f"\nExploitation inter-phase (cross-type only):")
        print(f"  Count: {len(all_exp_method2)}")
        print(f"  Mean: {np.mean(all_exp_method2):.4f}")
        print(f"  Median: {np.median(all_exp_method2):.4f}")
    
    if all_expl_method2:
        print(f"\nExploration inter-phase (cross-type only):")
        print(f"  Count: {len(all_expl_method2)}")
        print(f"  Mean: {np.mean(all_expl_method2):.4f}")
        print(f"  Median: {np.median(all_expl_method2):.4f}")






