#!/usr/bin/env python3
"""
Recompute inter-phase means for the specified participants.
Uses the participant list provided and recomputes from phase coherence data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer
from phase_coherence_analysis import compute_phase_coherence_metrics_detailed

# Participant list from the provided data
PARTICIPANT_LIST = [
    'PD00020', 'PD00048', 'PD00146', 'PD00215', 'PD00267', 'PD00457', 'PD00458',
    'PD00471', 'PD00472', 'PD00576', 'PD00583', 'PD00647', 'PD00653', 'PD00666',
    'PD00786', 'PD00849', 'PD00869', 'PD00955', 'PD00959', 'PD00999', 'PD01003',
    'PD01126', 'PD01133', 'PD01145', 'PD01146', 'PD01156', 'PD01160', 'PD01161',
    'PD01201', 'PD01223', 'PD01225', 'PD01237', 'PD01247', 'PD01270', 'PD01284',
    'PD01290', 'PD01306', 'PD01312', 'PD01319', 'PD01369', 'PD01377', 'PD01485',
    'PD01559', 'PD01623', 'PD01660', 'PD01667', 'PD01715'
]

def recompute_inter_phase_means(participant_ids):
    """Recompute inter-phase means for specified participants."""
    
    print("="*70)
    print("RECOMPUTING INTER-PHASE MEANS")
    print("="*70)
    print(f"Participants to process: {len(participant_ids)}")
    
    try:
        # Load configuration
        config = AnalysisConfig.from_yaml('config/config.yaml')
        
        # Initialize analyzer
        analyzer = SemanticFluencyAnalyzer(config)
        analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        
        results = []
        successful = 0
        
        for i, participant_id in enumerate(participant_ids, 1):
            print(f"\n[{i}/{len(participant_ids)}] Processing {participant_id}...", end=" ")
            
            try:
                # Get participant data
                participant_data = analyzer.data[analyzer.data['ID'] == participant_id]
                
                if participant_data.empty:
                    print(f"❌ Not found")
                    continue
                
                # Analyze participant
                result = analyzer.analyze_participant(participant_data)
                
                # Compute detailed phase coherence metrics
                coherence_metrics = compute_phase_coherence_metrics_detailed(
                    result['phases'], result['vectors'], result['items'], verbose=False
                )
                
                # Add participant ID
                coherence_metrics['ID'] = participant_id
                results.append(coherence_metrics)
                successful += 1
                print(f"✅")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print(f"\n{'='*70}")
        print(f"Successfully processed: {successful}/{len(participant_ids)} participants")
        print(f"{'='*70}")
        
        if results:
            # Create DataFrame
            results_df = pd.DataFrame(results)
            
            # Compute group statistics
            print(f"\n📊 GROUP STATISTICS:")
            print(f"{'='*70}")
            
            if 'inter_phase_mean_exploitation' in results_df.columns:
                exp_inter = results_df['inter_phase_mean_exploitation'].dropna()
                if len(exp_inter) > 0:
                    print(f"\nExploitation Inter-Phase Mean:")
                    print(f"  Count: {len(exp_inter)}")
                    print(f"  Mean: {exp_inter.mean():.4f}")
                    print(f"  Median: {exp_inter.median():.4f}")
                    print(f"  Std: {exp_inter.std():.4f}")
                    print(f"  Min: {exp_inter.min():.4f}")
                    print(f"  Max: {exp_inter.max():.4f}")
                    print(f"  Q1: {exp_inter.quantile(0.25):.4f}")
                    print(f"  Q3: {exp_inter.quantile(0.75):.4f}")
            
            if 'inter_phase_mean_exploration' in results_df.columns:
                expl_inter = results_df['inter_phase_mean_exploration'].dropna()
                if len(expl_inter) > 0:
                    print(f"\nExploration Inter-Phase Mean:")
                    print(f"  Count: {len(expl_inter)}")
                    print(f"  Mean: {expl_inter.mean():.4f}")
                    print(f"  Median: {expl_inter.median():.4f}")
                    print(f"  Std: {expl_inter.std():.4f}")
                    print(f"  Min: {expl_inter.min():.4f}")
                    print(f"  Max: {expl_inter.max():.4f}")
                    print(f"  Q1: {expl_inter.quantile(0.25):.4f}")
                    print(f"  Q3: {expl_inter.quantile(0.75):.4f}")
            
            # Save results
            output_file = Path('output/recomputed_inter_phase_means.csv')
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
            
            return results_df
        else:
            print("\n❌ No results to save")
            return None
            
    except Exception as e:
        print(f"\n❌ Error in main processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results_df = recompute_inter_phase_means(PARTICIPANT_LIST)
    
    if results_df is not None:
        print(f"\n✅ Recomputation complete!")
        print(f"\nTo regenerate the figure, run:")
        print(f"  python create_exploit_explore_metrics_figure.py")
        print(f"\n(After updating it to use: output/recomputed_inter_phase_means.csv)")






