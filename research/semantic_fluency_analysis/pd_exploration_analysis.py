#!/usr/bin/env python3
"""
PD Exploration Analysis: Why exploration might be higher in Parkinson's Disease patients
Comprehensive analysis of semantic fluency patterns in PD vs theoretical controls
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer

def analyze_pd_exploration_patterns():
    """
    Analyze why exploration might be higher in PD patients
    """
    
    print("🧠 PD EXPLORATION ANALYSIS")
    print("Why exploration might be higher in Parkinson's Disease patients")
    print("=" * 70)
    
    try:
        # Load configuration and data
        config = AnalysisConfig.from_yaml('config/config.yaml')
        analyzer = SemanticFluencyAnalyzer(config)
        analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        
        # Run analysis on all participants
        results_df = analyzer.analyze_all_participants()
        
        # Load MEG data for additional context
        meg_data = pd.read_csv('data/meg_data.csv')
        
        # Merge results with MEG data
        full_results = pd.merge(results_df, meg_data, on='ID', how='left')
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   Total participants: {len(full_results)}")
        print(f"   PD participants: {len(full_results[full_results['status'] == 'PD'])}")
        print(f"   Control participants: {len(full_results[full_results['status'] == 'wh'])}")
        
        # ============================================================================
        # EXPLORATION PATTERNS ANALYSIS
        # ============================================================================
        
        print(f"\n{'='*50}")
        print(f"EXPLORATION PATTERNS IN PD PATIENTS")
        print(f"{'='*50}")
        
        # Calculate exploration metrics
        exploration_stats = {
            'mean_exploration_percentage': full_results['exploration_percentage'].mean(),
            'std_exploration_percentage': full_results['exploration_percentage'].std(),
            'median_exploration_percentage': full_results['exploration_percentage'].median(),
            'mean_exploitation_percentage': full_results['exploitation_percentage'].mean(),
            'std_exploitation_percentage': full_results['exploitation_percentage'].std(),
            'median_exploitation_percentage': full_results['exploitation_percentage'].median(),
            'mean_num_switches': full_results['num_switches'].mean(),
            'std_num_switches': full_results['num_switches'].std(),
            'mean_novelty_score': full_results['novelty_score'].mean(),
            'std_novelty_score': full_results['novelty_score'].std()
        }
        
        print(f"\n📈 EXPLORATION METRICS:")
        print(f"   Mean Exploration Percentage: {exploration_stats['mean_exploration_percentage']:.2f}% ± {exploration_stats['std_exploration_percentage']:.2f}%")
        print(f"   Median Exploration Percentage: {exploration_stats['median_exploration_percentage']:.2f}%")
        print(f"   Mean Exploitation Percentage: {exploration_stats['mean_exploitation_percentage']:.2f}% ± {exploration_stats['std_exploitation_percentage']:.2f}%")
        print(f"   Median Exploitation Percentage: {exploration_stats['median_exploitation_percentage']:.2f}%")
        print(f"   Mean Number of Switches: {exploration_stats['mean_num_switches']:.2f} ± {exploration_stats['std_num_switches']:.2f}")
        print(f"   Mean Novelty Score: {exploration_stats['mean_novelty_score']:.3f} ± {exploration_stats['std_novelty_score']:.3f}")
        
        # ============================================================================
        # THEORETICAL EXPLANATIONS FOR HIGHER EXPLORATION IN PD
        # ============================================================================
        
        print(f"\n{'='*50}")
        print(f"THEORETICAL EXPLANATIONS FOR HIGHER EXPLORATION IN PD")
        print(f"{'='*50}")
        
        explanations = {
            "1. Dopaminergic Dysfunction": {
                "mechanism": "Reduced dopamine in striatum affects reward-based learning",
                "effect": "Impaired exploitation of familiar semantic clusters",
                "evidence": "PD patients show reduced reward sensitivity and impaired habit formation",
                "exploration_impact": "Higher exploration due to reduced exploitation efficiency"
            },
            "2. Executive Function Impairment": {
                "mechanism": "Frontal-striatal circuit dysfunction affects cognitive control",
                "effect": "Difficulty maintaining focused semantic search strategies",
                "evidence": "PD patients show impaired set-shifting and cognitive flexibility",
                "exploration_impact": "Increased random semantic jumping between categories"
            },
            "3. Working Memory Deficits": {
                "mechanism": "Impaired maintenance of semantic context and search strategies",
                "effect": "Difficulty staying within semantic clusters",
                "evidence": "PD patients show reduced working memory capacity",
                "exploration_impact": "Frequent switching between semantic domains"
            },
            "4. Attentional Dysfunction": {
                "mechanism": "Impaired sustained attention and selective attention",
                "effect": "Difficulty maintaining focus on semantic categories",
                "evidence": "PD patients show attentional deficits and distractibility",
                "exploration_impact": "Increased exploration due to attentional drift"
            },
            "5. Semantic Network Disruption": {
                "mechanism": "Altered semantic network connectivity and organization",
                "effect": "Less coherent semantic clusters, more diffuse associations",
                "evidence": "PD patients show semantic processing deficits",
                "exploration_impact": "Higher exploration due to semantic network fragmentation"
            },
            "6. Compensatory Strategy": {
                "mechanism": "Active compensation for exploitation deficits",
                "effect": "Deliberate use of exploration to maintain performance",
                "evidence": "PD patients may develop adaptive strategies",
                "exploration_impact": "Strategic increase in exploration to compensate"
            }
        }
        
        for i, (explanation, details) in enumerate(explanations.items(), 1):
            print(f"\n{i}. {explanation}")
            print(f"   Mechanism: {details['mechanism']}")
            print(f"   Effect: {details['effect']}")
            print(f"   Evidence: {details['evidence']}")
            print(f"   Exploration Impact: {details['exploration_impact']}")
        
        # ============================================================================
        # NEUROBIOLOGICAL BASIS
        # ============================================================================
        
        print(f"\n{'='*50}")
        print(f"NEUROBIOLOGICAL BASIS FOR EXPLORATION-EXPLOITATION IMBALANCE")
        print(f"{'='*50}")
        
        neurobiology = {
            "Dopaminergic Pathways": {
                "Mesolimbic Pathway": "Reward processing and motivation",
                "Nigrostriatal Pathway": "Motor control and habit formation", 
                "Mesocortical Pathway": "Executive function and cognitive control",
                "PD Impact": "Reduced dopamine in all pathways affects exploitation"
            },
            "Frontal-Striatal Circuits": {
                "Dorsolateral Prefrontal Cortex": "Working memory and cognitive control",
                "Orbitofrontal Cortex": "Reward evaluation and decision-making",
                "Anterior Cingulate Cortex": "Conflict monitoring and error detection",
                "PD Impact": "Impaired circuit function leads to exploration bias"
            },
            "Default Mode Network": {
                "Medial Prefrontal Cortex": "Self-referential thinking",
                "Posterior Cingulate Cortex": "Autobiographical memory",
                "Temporoparietal Junction": "Social cognition",
                "PD Impact": "Altered network connectivity affects semantic processing"
            }
        }
        
        for system, components in neurobiology.items():
            print(f"\n🔬 {system}:")
            for component, function in components.items():
                if component != "PD Impact":
                    print(f"   {component}: {function}")
                else:
                    print(f"   🚨 {component}: {function}")
        
        # ============================================================================
        # CLINICAL IMPLICATIONS
        # ============================================================================
        
        print(f"\n{'='*50}")
        print(f"CLINICAL IMPLICATIONS")
        print(f"{'='*50}")
        
        implications = [
            "1. Cognitive Assessment: Exploration-exploitation ratio as biomarker",
            "2. Treatment Monitoring: Track cognitive changes with dopaminergic therapy", 
            "3. Rehabilitation: Target exploitation training to improve semantic clustering",
            "4. Early Detection: Exploration bias as early cognitive marker",
            "5. Personalized Medicine: Individual exploration patterns guide treatment"
        ]
        
        for implication in implications:
            print(f"   {implication}")
        
        # ============================================================================
        # DATA-DRIVEN INSIGHTS
        # ============================================================================
        
        print(f"\n{'='*50}")
        print(f"DATA-DRIVEN INSIGHTS FROM OUR ANALYSIS")
        print(f"{'='*50}")
        
        # Analyze high exploration participants
        high_exploration = full_results[full_results['exploration_percentage'] > 
                                      full_results['exploration_percentage'].quantile(0.75)]
        
        print(f"\n📊 HIGH EXPLORATION PARTICIPANTS (Top 25%):")
        print(f"   Count: {len(high_exploration)}")
        print(f"   Mean Exploration: {high_exploration['exploration_percentage'].mean():.1f}%")
        print(f"   Mean Exploitation: {high_exploration['exploitation_percentage'].mean():.1f}%")
        print(f"   Mean Switches: {high_exploration['num_switches'].mean():.1f}")
        print(f"   Mean Novelty: {high_exploration['novelty_score'].mean():.3f}")
        
        # Analyze low exploration participants
        low_exploration = full_results[full_results['exploration_percentage'] < 
                                     full_results['exploration_percentage'].quantile(0.25)]
        
        print(f"\n📊 LOW EXPLORATION PARTICIPANTS (Bottom 25%):")
        print(f"   Count: {len(low_exploration)}")
        print(f"   Mean Exploration: {low_exploration['exploration_percentage'].mean():.1f}%")
        print(f"   Mean Exploitation: {low_exploration['exploitation_percentage'].mean():.1f}%")
        print(f"   Mean Switches: {low_exploration['num_switches'].mean():.1f}")
        print(f"   Mean Novelty: {low_exploration['novelty_score'].mean():.3f}")
        
        # ============================================================================
        # RESEARCH HYPOTHESES
        # ============================================================================
        
        print(f"\n{'='*50}")
        print(f"RESEARCH HYPOTHESES")
        print(f"{'='*50}")
        
        hypotheses = [
            "H1: PD patients show higher exploration due to dopaminergic dysfunction",
            "H2: Exploration-exploitation ratio correlates with disease severity",
            "H3: Dopaminergic medication normalizes exploration patterns",
            "H4: High exploration predicts cognitive decline in PD",
            "H5: Exploration patterns differ by PD subtype (tremor vs. akinetic-rigid)",
            "H6: Exploration bias is present in prodromal PD",
            "H7: Cognitive training can improve exploitation in PD patients"
        ]
        
        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"   {hypothesis}")
        
        # ============================================================================
        # METHODOLOGICAL CONSIDERATIONS
        # ============================================================================
        
        print(f"\n{'='*50}")
        print(f"METHODOLOGICAL CONSIDERATIONS")
        print(f"{'='*50}")
        
        considerations = [
            "• Need control group for direct comparison",
            "• Control for medication effects and disease duration",
            "• Consider motor vs. cognitive subtypes of PD",
            "• Account for age, education, and general cognitive status",
            "• Use longitudinal design to track changes over time",
            "• Combine with neuroimaging for mechanistic insights",
            "• Validate with other cognitive tasks"
        ]
        
        for consideration in considerations:
            print(f"   {consideration}")
        
        return full_results, exploration_stats
        
    except Exception as e:
        print(f"❌ Error in PD exploration analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_exploration_visualization(results_df):
    """Create visualization of exploration patterns"""
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Exploration vs Exploitation distribution
    plt.subplot(2, 3, 1)
    plt.scatter(results_df['exploitation_percentage'], results_df['exploration_percentage'], 
               alpha=0.6, color='blue')
    plt.xlabel('Exploitation Percentage (%)')
    plt.ylabel('Exploration Percentage (%)')
    plt.title('Exploration vs Exploitation Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line
    max_val = max(results_df['exploitation_percentage'].max(), 
                 results_df['exploration_percentage'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal')
    plt.legend()
    
    # Subplot 2: Number of switches distribution
    plt.subplot(2, 3, 2)
    plt.hist(results_df['num_switches'], bins=15, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Number of Phase Switches')
    plt.ylabel('Frequency')
    plt.title('Distribution of Phase Switches')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Novelty score distribution
    plt.subplot(2, 3, 3)
    plt.hist(results_df['novelty_score'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Novelty Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Novelty Scores')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Exploration percentage by participant
    plt.subplot(2, 3, 4)
    sorted_data = results_df.sort_values('exploration_percentage', ascending=False)
    plt.bar(range(len(sorted_data)), sorted_data['exploration_percentage'], 
           color='purple', alpha=0.7)
    plt.xlabel('Participant (sorted by exploration)')
    plt.ylabel('Exploration Percentage (%)')
    plt.title('Exploration Percentage by Participant')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Exploitation vs Novelty
    plt.subplot(2, 3, 5)
    plt.scatter(results_df['exploitation_percentage'], results_df['novelty_score'], 
               alpha=0.6, color='red')
    plt.xlabel('Exploitation Percentage (%)')
    plt.ylabel('Novelty Score')
    plt.title('Exploitation vs Novelty')
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Word coverage vs exploration
    plt.subplot(2, 3, 6)
    plt.scatter(results_df['word_coverage'], results_df['exploration_percentage'], 
               alpha=0.6, color='brown')
    plt.xlabel('Word Coverage (%)')
    plt.ylabel('Exploration Percentage (%)')
    plt.title('Word Coverage vs Exploration')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/pd_exploration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved to: output/pd_exploration_analysis.png")

def main():
    """Main function to run PD exploration analysis"""
    
    print("🧠 PD EXPLORATION ANALYSIS")
    print("Understanding why exploration might be higher in Parkinson's Disease patients")
    print("=" * 70)
    
    # Run analysis
    results, stats = analyze_pd_exploration_patterns()
    
    if results is not None:
        # Create visualization
        create_exploration_visualization(results)
        
        print(f"\n✅ PD exploration analysis completed!")
        print(f"   Key finding: Mean exploration = {stats['mean_exploration_percentage']:.1f}%")
        print(f"   Key finding: Mean exploitation = {stats['mean_exploitation_percentage']:.1f}%")
        print(f"   Key finding: Mean switches = {stats['mean_num_switches']:.1f}")
        
        print(f"\n📋 SUMMARY:")
        print(f"   • PD patients may show higher exploration due to dopaminergic dysfunction")
        print(f"   • Executive function impairment leads to difficulty maintaining exploitation")
        print(f"   • Working memory deficits cause frequent semantic switching")
        print(f"   • Attentional dysfunction results in exploration bias")
        print(f"   • Semantic network disruption increases exploration")
        print(f"   • Compensatory strategies may also play a role")
        
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()
