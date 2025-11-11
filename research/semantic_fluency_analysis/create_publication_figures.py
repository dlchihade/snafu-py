#!/usr/bin/env python3
"""
Create publication-quality figures for embedding model comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Set publication-quality parameters to match reference style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'grid.alpha': 0.0,  # No grid for clean look
    'grid.linewidth': 0.0,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white'
})

# Set color palette to match reference style (professional neuroscience colors)
colors = {
    'primary': '#1f77b4',      # Professional blue
    'secondary': '#ff7f0e',    # Orange for contrast
    'accent': '#2ca02c',       # Green for third element
    'highlight': '#d62728',    # Red for emphasis
    'neutral': '#7f7f7f',      # Gray for neutral elements
    'light_blue': '#aec7e8',   # Light blue for HC-like groups
    'light_red': '#ff9896'     # Light red for PD-like groups
}
sns.set_palette([colors['primary'], colors['secondary'], colors['accent'], colors['highlight']])

def create_coverage_comparison_figure():
    """Create Figure 1: Word Coverage Comparison"""
    
    # Data from the comparison results
    models = ['spaCy\n(en_core_web_md)', 'Gensim\nWord2Vec', 'RoBERTa\n(Transformers)']
    coverage = [100.0, 22.4, 0.0]
    covered_words = [170, 38, 0]
    total_words = [170, 170, 170]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Coverage percentage
    bars1 = ax1.bar(models, coverage, color=colors['primary'], alpha=0.8, edgecolor='none', linewidth=0)
    ax1.set_title('A) Vocabulary Coverage by Model', fontweight='normal', pad=15)
    ax1.set_xlabel('Word Embedding Model', fontweight='normal')
    ax1.set_ylabel('Percentage of Animal Words Covered (%)', fontweight='normal')
    ax1.set_ylim(0, 110)
    
    # Add value labels on bars
    for bar, value in zip(bars1, coverage):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='normal', fontsize=9)
    
    # Plot 2: Absolute numbers
    bars2 = ax2.bar(models, covered_words, color=colors['secondary'], alpha=0.8, edgecolor='none', linewidth=0)
    ax2.set_title('B) Absolute Word Count Coverage', fontweight='normal', pad=15)
    ax2.set_xlabel('Word Embedding Model', fontweight='normal')
    ax2.set_ylabel('Number of Animal Words with Embeddings', fontweight='normal')
    ax2.set_ylim(0, 180)
    
    # Add value labels on bars
    for bar, value, total in zip(bars2, covered_words, total_words):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{value}/{total}', ha='center', va='bottom', fontweight='normal', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/publication_figure_1_coverage.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved: {output_path}")
    
    return fig

def create_performance_comparison_figure():
    """Create Figure 2: Performance Comparison"""
    
    # Data from the comparison results
    models = ['spaCy\n(en_core_web_md)', 'Gensim\nWord2Vec', 'RoBERTa\n(Transformers)']
    load_times = [0.56, 0.04, 23.65]
    retrieval_times = [1.73, 0.00, 0.06]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Load times
    bars1 = ax1.bar(models, load_times, color=colors['primary'], alpha=0.8, edgecolor='none', linewidth=0)
    ax1.set_title('A) Model Initialization Time', fontweight='normal', pad=15)
    ax1.set_xlabel('Word Embedding Model', fontweight='normal')
    ax1.set_ylabel('Time to Load Model into Memory (seconds)', fontweight='normal')
    ax1.set_ylim(0, 25)
    
    # Add value labels on bars
    for bar, value in zip(bars1, load_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.2f}s', ha='center', va='bottom', fontweight='normal', fontsize=9)
    
    # Plot 2: Retrieval times
    bars2 = ax2.bar(models, retrieval_times, color=colors['secondary'], alpha=0.8, edgecolor='none', linewidth=0)
    ax2.set_title('B) Word Vector Retrieval Speed', fontweight='normal', pad=15)
    ax2.set_xlabel('Word Embedding Model', fontweight='normal')
    ax2.set_ylabel('Average Time to Retrieve Word Vector (milliseconds)', fontweight='normal')
    ax2.set_ylim(0, 2.5)
    
    # Add value labels on bars
    for bar, value in zip(bars2, retrieval_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.2f}ms', ha='center', va='bottom', fontweight='normal', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/publication_figure_2_performance.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved: {output_path}")
    
    return fig

def create_similarity_comparison_figure():
    """Create Figure 3: Similarity Quality Comparison"""
    
    # Data from the comparison results
    models = ['spaCy\n(en_core_web_md)', 'Gensim\nWord2Vec']
    mean_similarities = [0.477, 0.043]
    fluency_similarities = [0.592, 0.021]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mean similarity scores
    bars1 = ax1.bar(models, mean_similarities, color=colors['primary'], alpha=0.8, edgecolor='none', linewidth=0)
    ax1.set_title('A) Semantic Similarity Quality', fontweight='normal', pad=15)
    ax1.set_xlabel('Word Embedding Model', fontweight='normal')
    ax1.set_ylabel('Mean Cosine Similarity Between Animal Word Pairs', fontweight='normal')
    ax1.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars1, mean_similarities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='normal', fontsize=9)
    
    # Plot 2: Fluency sequence similarities
    bars2 = ax2.bar(models, fluency_similarities, color=colors['secondary'], alpha=0.8, edgecolor='none', linewidth=0)
    ax2.set_title('B) Fluency Sequence Semantic Coherence', fontweight='normal', pad=15)
    ax2.set_xlabel('Word Embedding Model', fontweight='normal')
    ax2.set_ylabel('Mean Cosine Similarity Between Consecutive Words in Fluency Sequences', fontweight='normal')
    ax2.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars2, fluency_similarities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='normal', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/publication_figure_3_similarity.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 3 saved: {output_path}")
    
    return fig

def create_comprehensive_figure():
    """Create Figure 4: Comprehensive Comparison"""
    
    # Data from the comparison results
    models = ['spaCy\n(en_core_web_md)', 'Gensim\nWord2Vec', 'RoBERTa\n(Transformers)']
    
    # Normalize data for radar plot
    coverage_norm = [100, 22.4, 0]  # Already in percentage
    load_time_norm = [100 - (0.56/25)*100, 100 - (0.04/25)*100, 100 - (23.65/25)*100]  # Inverted (higher is better)
    retrieval_norm = [100 - (1.73/2)*100, 100, 100 - (0.06/2)*100]  # Inverted (higher is better)
    similarity_norm = [47.7, 4.3, 0]  # Scaled similarity scores
    
    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Define angles for each metric
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Create data for each model
    spacy_data = [coverage_norm[0], load_time_norm[0], retrieval_norm[0], similarity_norm[0]]
    spacy_data += spacy_data[:1]
    
    gensim_data = [coverage_norm[1], load_time_norm[1], retrieval_norm[1], similarity_norm[1]]
    gensim_data += gensim_data[:1]
    
    roberta_data = [coverage_norm[2], load_time_norm[2], retrieval_norm[2], similarity_norm[2]]
    roberta_data += roberta_data[:1]
    
    # Plot radar chart
    ax.plot(angles, spacy_data, 'o-', linewidth=2, label='spaCy', color=colors['primary'], markersize=6)
    ax.fill(angles, spacy_data, alpha=0.15, color=colors['primary'])
    
    ax.plot(angles, gensim_data, 'o-', linewidth=2, label='Gensim Word2Vec', color=colors['secondary'], markersize=6)
    ax.fill(angles, gensim_data, alpha=0.15, color=colors['secondary'])
    
    ax.plot(angles, roberta_data, 'o-', linewidth=2, label='RoBERTa', color=colors['accent'], markersize=6)
    ax.fill(angles, roberta_data, alpha=0.15, color=colors['accent'])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Word\nCoverage', 'Load\nSpeed', 'Retrieval\nSpeed', 'Similarity\nQuality'])
    
    # Set y-axis
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Add title
    plt.title('Multi-Dimensional Model Performance Comparison\n(Higher values indicate better performance across all metrics)', 
              fontweight='normal', pad=15)
    
    # Add grid
    ax.grid(True, alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/publication_figure_4_comprehensive.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 4 saved: {output_path}")
    
    return fig

def create_fluency_data_visualization():
    """Create Figure 5: Fluency Data Visualization"""
    
    # Load fluency data
    fluency_data = pd.read_csv('data/fluency_data.csv')
    
    # Get word frequencies
    word_counts = fluency_data['Item'].value_counts().head(20)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Most common words
    bars1 = ax1.barh(range(len(word_counts)), word_counts.values, color=colors['primary'], alpha=0.8, edgecolor='none')
    ax1.set_yticks(range(len(word_counts)))
    ax1.set_yticklabels(word_counts.index)
    ax1.set_xlabel('Number of Participants Who Generated This Word', fontweight='normal')
    ax1.set_title('A) Most Frequently Generated Animal Words', fontweight='normal', pad=15)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, word_counts.values)):
        ax1.text(value + 0.5, i, str(value), va='center', fontweight='normal', fontsize=9)
    
    # Plot 2: Word distribution by participant
    participant_counts = fluency_data.groupby('ID').size()
    
    ax2.hist(participant_counts, bins=15, color=colors['secondary'], alpha=0.8, edgecolor='none', linewidth=0)
    ax2.set_xlabel('Number of Animal Words Generated per Participant', fontweight='normal')
    ax2.set_ylabel('Number of Participants (Frequency)', fontweight='normal')
    ax2.set_title('B) Distribution of Fluency Performance Across Participants', fontweight='normal', pad=15)
    
    # Add statistics
    mean_words = participant_counts.mean()
    std_words = participant_counts.std()
    ax2.axvline(mean_words, color=colors['accent'], linestyle='--', linewidth=1.5, 
                label=f'Mean: {mean_words:.1f} ± {std_words:.1f}')
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/publication_figure_5_fluency_data.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 5 saved: {output_path}")
    
    return fig

def create_summary_table():
    """Create a summary table for publication"""
    
    # Create summary data
    data = {
        'Model': ['spaCy (en_core_web_md)', 'Gensim Word2Vec', 'RoBERTa (Transformers)'],
        'Word Coverage (%)': [100.0, 22.4, 0.0],
        'Words Covered': ['170/170', '38/170', '0/170'],
        'Load Time (s)': [0.56, 0.04, 23.65],
        'Retrieval Time (ms)': [1.73, 0.00, 0.06],
        'Mean Similarity': [0.477, 0.043, 'N/A'],
        'Fluency Similarity': [0.592, 0.021, 'N/A']
    }
    
    df = pd.DataFrame(data)
    
    # Save as CSV for easy import into LaTeX
    output_path = Path('output/embedding_comparison_table.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Summary table saved: {output_path}")
    
    # Also create a LaTeX table
    latex_table = df.to_latex(index=False, float_format='%.2f', escape=False)
    latex_path = Path('output/embedding_comparison_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved: {latex_path}")
    
    return df

def main():
    """Create all publication-quality figures"""
    print("🎨 Creating Publication-Quality Figures")
    print("=" * 50)
    
    # Create all figures
    fig1 = create_coverage_comparison_figure()
    fig2 = create_performance_comparison_figure()
    fig3 = create_similarity_comparison_figure()
    fig4 = create_comprehensive_figure()
    fig5 = create_fluency_data_visualization()
    
    # Create summary table
    summary_table = create_summary_table()
    
    print("\n✅ All publication-quality figures created!")
    print("\n📊 Generated Files:")
    print("   - publication_figure_1_coverage.png")
    print("   - publication_figure_2_performance.png")
    print("   - publication_figure_3_similarity.png")
    print("   - publication_figure_4_comprehensive.png")
    print("   - publication_figure_5_fluency_data.png")
    print("   - embedding_comparison_table.csv")
    print("   - embedding_comparison_table.tex")
    
    print("\n📋 Figure Descriptions:")
    print("   Figure 1: Word coverage comparison")
    print("   Figure 2: Performance metrics")
    print("   Figure 3: Similarity quality comparison")
    print("   Figure 4: Comprehensive radar plot")
    print("   Figure 5: Fluency data visualization")
    
    print("\n🎯 Key Finding: spaCy provides 100% coverage with high-quality similarities")
    
    return summary_table

if __name__ == "__main__":
    main()
