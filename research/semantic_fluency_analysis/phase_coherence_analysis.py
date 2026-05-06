#!/usr/bin/env python3
"""
Inter-Phase and Intra-Phase Mean and Variance Analysis
Detailed computation with mathematical explanations
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer

def compute_intra_phase_metrics(phases: List[Dict], vectors: List[np.ndarray], verbose: bool = True) -> Dict:
    """
    Compute INTRA-PHASE mean and variance (within each phase)
    
    Mathematical approach:
    1. For each phase, compute all pairwise cosine similarities between words
    2. Calculate mean and variance of these similarities
    3. Separate by phase type (exploitation vs exploration)
    """
    
    exploitation_phases = [p for p in phases if p['type'] == 'Exploitation']
    exploration_phases = [p for p in phases if p['type'] == 'Exploration']
    
    # ============================================================================
    # EXPLOITATION PHASES - INTRA-PHASE SIMILARITIES
    # ============================================================================
    
    exploitation_intra_similarities = []
    
    for i, phase in enumerate(exploitation_phases):
        if len(phase['vectors']) >= 2:
            if verbose:
                print(f"\n🔍 Exploitation Phase {i+1}: {len(phase['vectors'])} words")
                print(f"   Words: {phase['items']}")
            
            # Step 1: Create similarity matrix for this phase
            phase_vectors = np.array(phase['vectors'])
            phase_sim_matrix = cosine_similarity(phase_vectors)
            
            if verbose:
                print(f"   Similarity Matrix Shape: {phase_sim_matrix.shape}")
                print(f"   Similarity Matrix:\n{phase_sim_matrix.round(3)}")
            
            # Step 2: Extract upper triangle (excluding diagonal) for pairwise similarities
            upper_triangle_indices = np.triu_indices_from(phase_sim_matrix, k=1)
            phase_pairwise_sims = phase_sim_matrix[upper_triangle_indices]
            
            if verbose:
                print(f"   Pairwise Similarities: {phase_pairwise_sims.round(3)}")
            
            exploitation_intra_similarities.extend(phase_pairwise_sims)
    
    # ============================================================================
    # EXPLORATION PHASES - INTRA-PHASE SIMILARITIES
    # ============================================================================
    
    exploration_intra_similarities = []
    
    for i, phase in enumerate(exploration_phases):
        if len(phase['vectors']) >= 2:
            if verbose:
                print(f"\n🔍 Exploration Phase {i+1}: {len(phase['vectors'])} words")
                print(f"   Words: {phase['items']}")
            
            # Step 1: Create similarity matrix for this phase
            phase_vectors = np.array(phase['vectors'])
            phase_sim_matrix = cosine_similarity(phase_vectors)
            
            if verbose:
                print(f"   Similarity Matrix Shape: {phase_sim_matrix.shape}")
                print(f"   Similarity Matrix:\n{phase_sim_matrix.round(3)}")
            
            # Step 2: Extract upper triangle (excluding diagonal) for pairwise similarities
            upper_triangle_indices = np.triu_indices_from(phase_sim_matrix, k=1)
            phase_pairwise_sims = phase_sim_matrix[upper_triangle_indices]
            
            if verbose:
                print(f"   Pairwise Similarities: {phase_pairwise_sims.round(3)}")
            
            exploration_intra_similarities.extend(phase_pairwise_sims)
    
    # ============================================================================
    # COMPUTE INTRA-PHASE STATISTICS
    # ============================================================================
    
    metrics = {}
    
    # Exploitation intra-phase statistics
    if exploitation_intra_similarities:
        exploitation_intra_similarities = np.array(exploitation_intra_similarities)
        
        # Mean: μ = (1/n) * Σ(x_i)
        exploitation_mean = np.mean(exploitation_intra_similarities)
        
        # Variance: σ² = (1/n) * Σ(x_i - μ)²
        exploitation_variance = np.var(exploitation_intra_similarities)
        
        # Standard deviation: σ = √σ²
        exploitation_std = np.std(exploitation_intra_similarities)
        
        if verbose:
            print(f"\n📊 EXPLOITATION INTRA-PHASE STATISTICS:")
            print(f"   Number of pairwise similarities: {len(exploitation_intra_similarities)}")
            print(f"   Mean: {exploitation_mean:.4f}")
            print(f"   Variance: {exploitation_variance:.4f}")
            print(f"   Standard Deviation: {exploitation_std:.4f}")
        
        metrics.update({
            'exploitation_intra_mean': exploitation_mean,
            'exploitation_intra_variance': exploitation_variance,
            'exploitation_intra_std': exploitation_std,
            'exploitation_intra_similarities': exploitation_intra_similarities.tolist()
        })
    else:
        metrics.update({
            'exploitation_intra_mean': 0,
            'exploitation_intra_variance': 0,
            'exploitation_intra_std': 0,
            'exploitation_intra_similarities': []
        })
    
    # Exploration intra-phase statistics
    if exploration_intra_similarities:
        exploration_intra_similarities = np.array(exploration_intra_similarities)
        
        # Mean: μ = (1/n) * Σ(x_i)
        exploration_mean = np.mean(exploration_intra_similarities)
        
        # Variance: σ² = (1/n) * Σ(x_i - μ)²
        exploration_variance = np.var(exploration_intra_similarities)
        
        # Standard deviation: σ = √σ²
        exploration_std = np.std(exploration_intra_similarities)
        
        if verbose:
            print(f"\n📊 EXPLORATION INTRA-PHASE STATISTICS:")
            print(f"   Number of pairwise similarities: {len(exploration_intra_similarities)}")
            print(f"   Mean: {exploration_mean:.4f}")
            print(f"   Variance: {exploration_variance:.4f}")
            print(f"   Standard Deviation: {exploration_std:.4f}")
        
        metrics.update({
            'exploration_intra_mean': exploration_mean,
            'exploration_intra_variance': exploration_variance,
            'exploration_intra_std': exploration_std,
            'exploration_intra_similarities': exploration_intra_similarities.tolist()
        })
    else:
        metrics.update({
            'exploration_intra_mean': 0,
            'exploration_intra_variance': 0,
            'exploration_intra_std': 0,
            'exploration_intra_similarities': []
        })
    
    return metrics

def compute_inter_phase_metrics(phases: List[Dict], vectors: List[np.ndarray], verbose: bool = True) -> Dict:
    """
    Compute INTER-PHASE mean and variance (between phases)
    
    Mathematical approach:
    1. Calculate centroid (mean vector) for each phase
    2. Compute cosine similarity between all phase centroids
    3. Calculate mean and variance of these inter-phase similarities
    """
    
    exploitation_phases = [p for p in phases if p['type'] == 'Exploitation']
    exploration_phases = [p for p in phases if p['type'] == 'Exploration']
    
    inter_phase_similarities = []
    phase_centroids = []
    
    if verbose:
        print(f"\n🔗 INTER-PHASE ANALYSIS:")
        print(f"   Number of exploitation phases: {len(exploitation_phases)}")
        print(f"   Number of exploration phases: {len(exploration_phases)}")
    
    # ============================================================================
    # CALCULATE PHASE CENTROIDS
    # ============================================================================
    
    # Exploitation phase centroids
    for i, phase in enumerate(exploitation_phases):
        if len(phase['vectors']) > 0:
            # Centroid = mean of all vectors in the phase
            phase_centroid = np.mean(phase['vectors'], axis=0)
            
            # Normalize centroid for cosine similarity
            centroid_norm = np.linalg.norm(phase_centroid)
            if centroid_norm > 0:
                phase_centroid_normalized = phase_centroid / centroid_norm
            else:
                phase_centroid_normalized = phase_centroid
            
            phase_centroids.append({
                'type': 'Exploitation',
                'phase_index': i,
                'centroid': phase_centroid_normalized,
                'words': phase['items']
            })
            
            print(f"   Exploitation Phase {i+1} Centroid Norm: {np.linalg.norm(phase_centroid_normalized):.4f}")
    
    # Exploration phase centroids
    for i, phase in enumerate(exploration_phases):
        if len(phase['vectors']) > 0:
            # Centroid = mean of all vectors in the phase
            phase_centroid = np.mean(phase['vectors'], axis=0)
            
            # Normalize centroid for cosine similarity
            centroid_norm = np.linalg.norm(phase_centroid)
            if centroid_norm > 0:
                phase_centroid_normalized = phase_centroid / centroid_norm
            else:
                phase_centroid_normalized = phase_centroid
            
            phase_centroids.append({
                'type': 'Exploration',
                'phase_index': i,
                'centroid': phase_centroid_normalized,
                'words': phase['items']
            })
            
            print(f"   Exploration Phase {i+1} Centroid Norm: {np.linalg.norm(phase_centroid_normalized):.4f}")
    
    # ============================================================================
    # COMPUTE INTER-PHASE SIMILARITIES
    # ============================================================================
    
    print(f"\n🔗 INTER-PHASE SIMILARITIES:")
    
    # Separate lists for exploitation and exploration inter-phase similarities
    # Inter-phase for exploitation: similarities between exploitation phases and ALL other phases
    # Inter-phase for exploration: similarities between exploration phases and ALL other phases
    exploitation_inter_similarities = []
    exploration_inter_similarities = []
    
    for i, centroid1 in enumerate(phase_centroids):
        for j, centroid2 in enumerate(phase_centroids):
            if i != j:  # Include all pairs (not just i < j), but track by phase type
                
                # Cosine similarity between centroids
                # cos(θ) = (a · b) / (||a|| * ||b||)
                # Since centroids are normalized: cos(θ) = a · b
                similarity = np.dot(centroid1['centroid'], centroid2['centroid'])
                
                inter_phase_similarities.append(similarity)
                
                # If the first phase is exploitation, add to exploitation inter-phase
                if centroid1['type'] == 'Exploitation':
                    exploitation_inter_similarities.append(similarity)
                
                # If the first phase is exploration, add to exploration inter-phase
                if centroid1['type'] == 'Exploration':
                    exploration_inter_similarities.append(similarity)
                
                if i < j:  # Only print once per pair
                    print(f"   {centroid1['type']} Phase {centroid1['phase_index']+1} ↔ {centroid2['type']} Phase {centroid2['phase_index']+1}: {similarity:.4f}")
    
    # ============================================================================
    # COMPUTE INTER-PHASE STATISTICS (AGGREGATE)
    # ============================================================================
    
    if inter_phase_similarities:
        inter_phase_similarities = np.array(inter_phase_similarities)
        
        # Mean: μ = (1/n) * Σ(x_i)
        inter_mean = np.mean(inter_phase_similarities)
        
        # Variance: σ² = (1/n) * Σ(x_i - μ)²
        inter_variance = np.var(inter_phase_similarities)
        
        # Standard deviation: σ = √σ²
        inter_std = np.std(inter_phase_similarities)
        
        print(f"\n📊 INTER-PHASE STATISTICS (AGGREGATE):")
        print(f"   Number of inter-phase similarities: {len(inter_phase_similarities)}")
        print(f"   Mean: {inter_mean:.4f}")
        print(f"   Variance: {inter_variance:.4f}")
        print(f"   Standard Deviation: {inter_std:.4f}")
    else:
        inter_mean = 0
        inter_variance = 0
        inter_std = 0
    
    # ============================================================================
    # COMPUTE INTER-PHASE STATISTICS BY PHASE TYPE
    # ============================================================================
    
    # Exploitation inter-phase statistics
    if exploitation_inter_similarities:
        exploitation_inter_similarities = np.array(exploitation_inter_similarities)
        exploitation_inter_mean = np.mean(exploitation_inter_similarities)
        exploitation_inter_variance = np.var(exploitation_inter_similarities)
        exploitation_inter_std = np.std(exploitation_inter_similarities)
        
        print(f"\n📊 EXPLOITATION INTER-PHASE STATISTICS:")
        print(f"   Number of similarities: {len(exploitation_inter_similarities)}")
        print(f"   Mean: {exploitation_inter_mean:.4f}")
        print(f"   Variance: {exploitation_inter_variance:.4f}")
        print(f"   Standard Deviation: {exploitation_inter_std:.4f}")
    else:
        exploitation_inter_mean = 0
        exploitation_inter_variance = 0
        exploitation_inter_std = 0
        exploitation_inter_similarities = []
    
    # Exploration inter-phase statistics
    if exploration_inter_similarities:
        exploration_inter_similarities = np.array(exploration_inter_similarities)
        exploration_inter_mean = np.mean(exploration_inter_similarities)
        exploration_inter_variance = np.var(exploration_inter_similarities)
        exploration_inter_std = np.std(exploration_inter_similarities)
        
        print(f"\n📊 EXPLORATION INTER-PHASE STATISTICS:")
        print(f"   Number of similarities: {len(exploration_inter_similarities)}")
        print(f"   Mean: {exploration_inter_mean:.4f}")
        print(f"   Variance: {exploration_inter_variance:.4f}")
        print(f"   Standard Deviation: {exploration_inter_std:.4f}")
    else:
        exploration_inter_mean = 0
        exploration_inter_variance = 0
        exploration_inter_std = 0
        exploration_inter_similarities = []
    
    return {
        # Aggregate metrics (for backward compatibility)
        'inter_phase_mean': inter_mean,
        'inter_phase_variance': inter_variance,
        'inter_phase_std': inter_std,
        'inter_phase_similarities': inter_phase_similarities.tolist() if isinstance(inter_phase_similarities, np.ndarray) else [],
        # Phase-type-specific metrics
        'inter_phase_mean_exploitation': exploitation_inter_mean,
        'inter_phase_variance_exploitation': exploitation_inter_variance,
        'inter_phase_std_exploitation': exploitation_inter_std,
        'inter_phase_similarities_exploitation': exploitation_inter_similarities.tolist() if isinstance(exploitation_inter_similarities, np.ndarray) else [],
        'inter_phase_mean_exploration': exploration_inter_mean,
        'inter_phase_variance_exploration': exploration_inter_variance,
        'inter_phase_std_exploration': exploration_inter_std,
        'inter_phase_similarities_exploration': exploration_inter_similarities.tolist() if isinstance(exploration_inter_similarities, np.ndarray) else [],
    }

def compute_phase_coherence_metrics_detailed(phases: List[Dict], vectors: List[np.ndarray], 
                                           items: List[str], verbose: bool = True) -> Dict:
    """
    Compute comprehensive phase coherence metrics with detailed explanations
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE COHERENCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Total phases: {len(phases)}")
        print(f"Total words: {len(items)}")
        print(f"Total vectors: {len(vectors)}")
    
    # ============================================================================
    # INTRA-PHASE ANALYSIS
    # ============================================================================
    
    if verbose:
        print(f"\n{'='*40}")
        print(f"INTRA-PHASE ANALYSIS (Within Phase Coherence)")
        print(f"{'='*40}")
    
    intra_metrics = compute_intra_phase_metrics(phases, vectors, verbose)
    
    # ============================================================================
    # INTER-PHASE ANALYSIS
    # ============================================================================
    
    if verbose:
        print(f"\n{'='*40}")
        print(f"INTER-PHASE ANALYSIS (Between Phase Separation)")
        print(f"{'='*40}")
    
    inter_metrics = compute_inter_phase_metrics(phases, vectors, verbose)
    
    # ============================================================================
    # COHERENCE RATIOS AND INDICES
    # ============================================================================
    
    if verbose:
        print(f"\n{'='*40}")
        print(f"COHERENCE RATIOS AND INDICES")
        print(f"{'='*40}")
    
    # Phase coherence ratios
    if inter_metrics['inter_phase_mean'] > 0:
        exploitation_coherence_ratio = intra_metrics['exploitation_intra_mean'] / inter_metrics['inter_phase_mean']
        exploration_coherence_ratio = intra_metrics['exploration_intra_mean'] / inter_metrics['inter_phase_mean']
    else:
        exploitation_coherence_ratio = 0
        exploration_coherence_ratio = 0
    
    # Phase separation index
    if (intra_metrics['exploitation_intra_mean'] > 0 and 
        intra_metrics['exploration_intra_mean'] > 0):
        phase_separation_index = (
            (intra_metrics['exploitation_intra_mean'] + intra_metrics['exploration_intra_mean']) / 2
        ) - inter_metrics['inter_phase_mean']
    else:
        phase_separation_index = 0
    
    if verbose:
        print(f"Exploitation Coherence Ratio: {exploitation_coherence_ratio:.4f}")
        print(f"   (Intra-phase mean / Inter-phase mean)")
        print(f"   ({intra_metrics['exploitation_intra_mean']:.4f} / {inter_metrics['inter_phase_mean']:.4f})")
        
        print(f"Exploration Coherence Ratio: {exploration_coherence_ratio:.4f}")
        print(f"   (Intra-phase mean / Inter-phase mean)")
        print(f"   ({intra_metrics['exploration_intra_mean']:.4f} / {inter_metrics['inter_phase_mean']:.4f})")
        
        print(f"Phase Separation Index: {phase_separation_index:.4f}")
        print(f"   (Average intra-phase mean - Inter-phase mean)")
        print(f"   ({(intra_metrics['exploitation_intra_mean'] + intra_metrics['exploration_intra_mean']) / 2:.4f} - {inter_metrics['inter_phase_mean']:.4f})")
    
    # Combine all metrics
    all_metrics = {**intra_metrics, **inter_metrics}
    all_metrics.update({
        'exploitation_coherence_ratio': exploitation_coherence_ratio,
        'exploration_coherence_ratio': exploration_coherence_ratio,
        'phase_separation_index': phase_separation_index
    })
    
    return all_metrics

def analyze_single_participant_detailed(participant_id: str, verbose: bool = False):
    """Analyze a single participant with detailed phase coherence metrics"""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DETAILED PHASE COHERENCE ANALYSIS")
        print(f"Participant: {participant_id}")
        print(f"{'='*60}")
    else:
        print(f"   Analyzing {participant_id}...", end=" ")
    
    try:
        # Load configuration
        config = AnalysisConfig.from_yaml('config/config.yaml')
        
        # Initialize analyzer
        analyzer = SemanticFluencyAnalyzer(config)
        analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        
        # Get participant data
        participant_data = analyzer.data[analyzer.data['ID'] == participant_id]
        
        if participant_data.empty:
            print(f"❌ Participant {participant_id} not found")
            return None
        
        # Analyze participant
        result = analyzer.analyze_participant(participant_data)
        
        # Compute detailed phase coherence metrics
        coherence_metrics = compute_phase_coherence_metrics_detailed(
            result['phases'], result['vectors'], result['items'], verbose
        )
        
        # Add participant ID
        coherence_metrics['ID'] = participant_id
        
        if not verbose:
            print(f"✅ ({len(result['phases'])} phases)")
        
        return coherence_metrics
        
    except Exception as e:
        print(f"❌ Error analyzing participant {participant_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run detailed phase coherence analysis for ALL participants"""
    
    print("🔬 DETAILED PHASE COHERENCE ANALYSIS")
    print("Computing Inter-Phase and Intra-Phase Mean and Variance for ALL PARTICIPANTS")
    print("=" * 60)
    
    try:
        # Load configuration
        config = AnalysisConfig.from_yaml('config/config.yaml')
        
        # Initialize analyzer
        analyzer = SemanticFluencyAnalyzer(config)
        analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        
        # Get ALL participant IDs
        all_participant_ids = analyzer.data['ID'].unique()
        print(f"📊 Found {len(all_participant_ids)} participants to analyze")
        
        detailed_results = []
        successful_analyses = 0
        
        for i, participant_id in enumerate(all_participant_ids, 1):
            print(f"\n🔄 Processing participant {i}/{len(all_participant_ids)}: {participant_id}")
            
            result = analyze_single_participant_detailed(participant_id, verbose=False)
            if result:
                detailed_results.append(result)
                successful_analyses += 1
            else:
                print(f"⚠️  Skipped participant {participant_id} due to analysis issues")
        
        print(f"\n✅ Successfully analyzed {successful_analyses}/{len(all_participant_ids)} participants")
        
    except Exception as e:
        print(f"❌ Error in main analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if detailed_results:
        # Create summary
        print(f"\n{'='*60}")
        print(f"SUMMARY OF DETAILED ANALYSIS FOR ALL PARTICIPANTS")
        print(f"{'='*60}")
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(detailed_results)
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS ({len(results_df)} participants):")
        print(f"   Exploitation Intra-Phase Mean: {results_df['exploitation_intra_mean'].mean():.4f} ± {results_df['exploitation_intra_mean'].std():.4f}")
        print(f"   Exploitation Intra-Phase Variance: {results_df['exploitation_intra_variance'].mean():.4f} ± {results_df['exploitation_intra_variance'].std():.4f}")
        print(f"   Exploration Intra-Phase Mean: {results_df['exploration_intra_mean'].mean():.4f} ± {results_df['exploration_intra_mean'].std():.4f}")
        print(f"   Exploration Intra-Phase Variance: {results_df['exploration_intra_variance'].mean():.4f} ± {results_df['exploration_intra_variance'].std():.4f}")
        print(f"   Inter-Phase Mean: {results_df['inter_phase_mean'].mean():.4f} ± {results_df['inter_phase_mean'].std():.4f}")
        print(f"   Inter-Phase Variance: {results_df['inter_phase_variance'].mean():.4f} ± {results_df['inter_phase_variance'].std():.4f}")
        print(f"   Exploitation Coherence Ratio: {results_df['exploitation_coherence_ratio'].mean():.4f} ± {results_df['exploitation_coherence_ratio'].std():.4f}")
        print(f"   Exploration Coherence Ratio: {results_df['exploration_coherence_ratio'].mean():.4f} ± {results_df['exploration_coherence_ratio'].std():.4f}")
        print(f"   Phase Separation Index: {results_df['phase_separation_index'].mean():.4f} ± {results_df['phase_separation_index'].std():.4f}")
        
        # Save detailed results to CSV
        output_file = 'output/phase_coherence_analysis_all_participants.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Show top and bottom performers
        print(f"\n🏆 TOP 5 HIGHEST EXPLOITATION COHERENCE:")
        top_exploitation = results_df.nlargest(5, 'exploitation_coherence_ratio')[['ID', 'exploitation_coherence_ratio']]
        for _, row in top_exploitation.iterrows():
            print(f"   {row['ID']}: {row['exploitation_coherence_ratio']:.4f}")
        
        print(f"\n🏆 TOP 5 HIGHEST EXPLORATION COHERENCE:")
        top_exploration = results_df.nlargest(5, 'exploration_coherence_ratio')[['ID', 'exploration_coherence_ratio']]
        for _, row in top_exploration.iterrows():
            print(f"   {row['ID']}: {row['exploration_coherence_ratio']:.4f}")
        
        print(f"\n📈 PHASE SEPARATION ANALYSIS:")
        print(f"   Best Phase Separation: {results_df['phase_separation_index'].max():.4f} (Participant {results_df.loc[results_df['phase_separation_index'].idxmax(), 'ID']})")
        print(f"   Worst Phase Separation: {results_df['phase_separation_index'].min():.4f} (Participant {results_df.loc[results_df['phase_separation_index'].idxmin(), 'ID']})")
        
    else:
        print(f"\n❌ No successful analyses completed")
    
    print(f"\n✅ Detailed phase coherence analysis completed for ALL participants!")

if __name__ == "__main__":
    main()
