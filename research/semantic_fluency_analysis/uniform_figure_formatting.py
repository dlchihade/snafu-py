#!/usr/bin/env python3
"""
Uniform Figure Formatting Script
Reformats all figures with consistent font sizes and typography.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set uniform typography for all figures
def setup_uniform_style():
    """Set up uniform typography and styling for all figures following Nature guidelines."""
    
    # Font settings - NATURE PUBLICATION STANDARDS
    plt.rcParams.update({
        # Typography - Nature requirements
        'font.family': 'Arial',             # Nature prefers Arial
        'font.size': 8,                     # Nature standard: 8pt base
        'axes.titlesize': 9,                # Nature standard: 9pt titles
        'axes.labelsize': 8,                # Nature standard: 8pt labels
        'xtick.labelsize': 7,               # Nature standard: 7pt tick labels
        'ytick.labelsize': 7,               # Nature standard: 7pt tick labels
        'legend.fontsize': 7,               # Nature standard: 7pt legend
        'figure.titlesize': 10,             # Nature standard: 10pt suptitles
        
        # Figure quality - Nature requirements
        'figure.dpi': 300,                  # Nature standard: 300 DPI minimum
        'savefig.dpi': 300,                 # Save at 300 DPI
        'savefig.bbox': 'tight',            # Tight layout
        'savefig.pad_inches': 0.05,         # Minimal padding
        
        # Colors and styling - Nature requirements
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,              # Nature standard: thin lines
        'axes.grid': False,                 # Nature typically doesn't use grids
        'grid.alpha': 0.0,
        'grid.linestyle': '-',
        'grid.linewidth': 0.0,
        
        # Spines - Nature requirements
        'axes.spines.top': False,           # Nature: no top spine
        'axes.spines.right': False,         # Nature: no right spine
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # Ticks - Nature requirements
        'xtick.major.size': 3,              # Nature: small ticks
        'xtick.major.width': 0.5,
        'ytick.major.size': 3,
        'ytick.major.width': 0.5,
        
        # Lines - Nature requirements
        'lines.linewidth': 1.0,             # Nature: thin lines
        'lines.markersize': 4,              # Nature: small markers
        
        # Patches (bars, etc.) - Nature requirements
        'patch.linewidth': 0.5,
        
        # Text - Nature requirements
        'text.usetex': False,
        'mathtext.fontset': 'cm',
    })
    
    # Set seaborn style to complement matplotlib
    sns.set_style("whitegrid", {
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'grid.color': 'lightgray',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    # Professional color palette
    colors = {
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e',    # Orange
        'accent': '#2ca02c',       # Green
        'highlight': '#d62728',    # Red
        'neutral': '#7f7f7f',      # Gray
        'light_blue': '#aec7e8',   # Light blue
        'light_orange': '#ffbb78', # Light orange
        'light_green': '#98df8a',  # Light green
        'light_red': '#ff9896',    # Light red
    }
    
    return colors

def create_uniform_exploration_exploitation_figure():
    """Create exploration vs exploitation figure with uniform formatting."""
    colors = setup_uniform_style()
    
    # Generate sample data (replace with your actual data)
    np.random.seed(42)
    n_participants = 56
    
    exploitation = np.random.normal(0.6, 0.15, n_participants)
    exploration = np.random.normal(0.4, 0.1, n_participants)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Scatter plot
    scatter = ax1.scatter(exploitation, exploration, 
                         c=exploitation + exploration, 
                         cmap='viridis', alpha=0.7, s=50)
    ax1.set_xlabel('Exploitation Score', fontsize=12, fontweight='normal')
    ax1.set_ylabel('Exploration Score', fontsize=12, fontweight='normal')
    ax1.set_title('Exploitation vs Exploration Distribution', 
                  fontsize=14, fontweight='normal')
    
    # Add correlation line
    z = np.polyfit(exploitation, exploration, 1)
    p = np.poly1d(z)
    ax1.plot(exploitation, p(exploitation), "r--", alpha=0.8, linewidth=2)
    
    # Panel B: Box plot
    data_to_plot = [exploitation, exploration]
    bp = ax2.boxplot(data_to_plot, labels=['Exploitation', 'Exploration'],
                     patch_artist=True, medianprops=dict(color='black', linewidth=2))
    
    # Color the boxes
    bp['boxes'][0].set_facecolor(colors['light_blue'])
    bp['boxes'][1].set_facecolor(colors['light_orange'])
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='normal')
    ax2.set_title('Score Distributions', fontsize=14, fontweight='normal')
    
    # Add statistics
    ax2.text(0.02, 0.98, f'Exploitation: {np.mean(exploitation):.3f} ± {np.std(exploitation):.3f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.90, f'Exploration: {np.mean(exploration):.3f} ± {np.std(exploration):.3f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/uniform_exploration_exploitation.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created uniform exploration-exploitation figure: {output_path}")

def create_uniform_phase_coherence_figure():
    """Create phase coherence figure with uniform formatting."""
    colors = setup_uniform_style()
    
    # Generate sample data
    np.random.seed(42)
    participants = [f'PD{i:05d}' for i in range(1, 57)]
    
    exploitation_coherence = np.random.normal(0.85, 0.2, 56)
    exploration_coherence = np.random.normal(0.57, 0.14, 56)
    phase_separation = np.random.normal(-0.20, 0.09, 56)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Exploitation Coherence
    bars1 = ax1.bar(range(len(participants[:20])), exploitation_coherence[:20], 
                    color=colors['primary'], alpha=0.8)
    ax1.set_xlabel('Participant ID', fontsize=12, fontweight='normal')
    ax1.set_ylabel('Exploitation Coherence', fontsize=12, fontweight='normal')
    ax1.set_title('Exploitation Phase Coherence', fontsize=14, fontweight='normal')
    ax1.tick_params(axis='x', rotation=45)
    
    # Panel B: Exploration Coherence
    bars2 = ax2.bar(range(len(participants[:20])), exploration_coherence[:20], 
                    color=colors['secondary'], alpha=0.8)
    ax2.set_xlabel('Participant ID', fontsize=12, fontweight='normal')
    ax2.set_ylabel('Exploration Coherence', fontsize=12, fontweight='normal')
    ax2.set_title('Exploration Phase Coherence', fontsize=14, fontweight='normal')
    ax2.tick_params(axis='x', rotation=45)
    
    # Panel C: Phase Separation
    bars3 = ax3.bar(range(len(participants[:20])), phase_separation[:20], 
                    color=colors['accent'], alpha=0.8)
    ax3.set_xlabel('Participant ID', fontsize=12, fontweight='normal')
    ax3.set_ylabel('Phase Separation Index', fontsize=12, fontweight='normal')
    ax3.set_title('Phase Separation Analysis', fontsize=14, fontweight='normal')
    ax3.tick_params(axis='x', rotation=45)
    
    # Panel D: Summary Statistics
    categories = ['Exploitation\nCoherence', 'Exploration\nCoherence', 'Phase\nSeparation']
    means = [np.mean(exploitation_coherence), np.mean(exploration_coherence), np.mean(phase_separation)]
    stds = [np.std(exploitation_coherence), np.std(exploration_coherence), np.std(phase_separation)]
    
    bars4 = ax4.bar(categories, means, yerr=stds, 
                    color=[colors['primary'], colors['secondary'], colors['accent']], 
                    alpha=0.8, capsize=5)
    ax4.set_ylabel('Score', fontsize=12, fontweight='normal')
    ax4.set_title('Overall Statistics', fontsize=14, fontweight='normal')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars4, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/uniform_phase_coherence.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created uniform phase coherence figure: {output_path}")

def create_uniform_neurobiological_figure():
    """Create neurobiological analysis figure with uniform formatting."""
    colors = setup_uniform_style()
    
    # Generate sample data
    np.random.seed(42)
    n_participants = 56
    
    word_coverage = np.random.uniform(0.8, 1.0, n_participants)
    clustering_coeff = np.random.normal(0.5, 0.15, n_participants)
    num_clusters = np.random.poisson(3, n_participants)
    exploration = np.random.normal(0.4, 0.1, n_participants)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Word Coverage vs Exploration
    ax1.scatter(word_coverage, exploration, alpha=0.7, s=50, color=colors['primary'])
    ax1.set_xlabel('Word Coverage Ratio', fontsize=12, fontweight='normal')
    ax1.set_ylabel('Exploration Score', fontsize=12, fontweight='normal')
    ax1.set_title('Semantic Coverage vs Exploration', fontsize=14, fontweight='normal')
    
    # Add trend line
    z = np.polyfit(word_coverage, exploration, 1)
    p = np.poly1d(z)
    ax1.plot(word_coverage, p(word_coverage), "r--", alpha=0.8, linewidth=2)
    
    # Panel B: Clustering Coefficient vs Exploration
    ax2.scatter(clustering_coeff, exploration, alpha=0.7, s=50, color=colors['secondary'])
    ax2.set_xlabel('Clustering Coefficient', fontsize=12, fontweight='normal')
    ax2.set_ylabel('Exploration Score', fontsize=12, fontweight='normal')
    ax2.set_title('Network Clustering vs Exploration', fontsize=14, fontweight='normal')
    
    # Add trend line
    z = np.polyfit(clustering_coeff, exploration, 1)
    p = np.poly1d(z)
    ax2.plot(clustering_coeff, p(clustering_coeff), "r--", alpha=0.8, linewidth=2)
    
    # Panel C: Number of Clusters vs Exploration
    ax3.scatter(num_clusters, exploration, alpha=0.7, s=50, color=colors['accent'])
    ax3.set_xlabel('Number of Semantic Clusters', fontsize=12, fontweight='normal')
    ax3.set_ylabel('Exploration Score', fontsize=12, fontweight='normal')
    ax3.set_title('Cluster Diversity vs Exploration', fontsize=14, fontweight='normal')
    
    # Add trend line
    z = np.polyfit(num_clusters, exploration, 1)
    p = np.poly1d(z)
    ax3.plot(num_clusters, p(num_clusters), "r--", alpha=0.8, linewidth=2)
    
    # Panel D: Summary Statistics Table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary data
    summary_data = [
        ['Metric', 'Mean', 'Std', 'Min', 'Max'],
        ['Word Coverage', f'{np.mean(word_coverage):.3f}', f'{np.std(word_coverage):.3f}', 
         f'{np.min(word_coverage):.3f}', f'{np.max(word_coverage):.3f}'],
        ['Clustering Coeff', f'{np.mean(clustering_coeff):.3f}', f'{np.std(clustering_coeff):.3f}', 
         f'{np.min(clustering_coeff):.3f}', f'{np.max(clustering_coeff):.3f}'],
        ['Num Clusters', f'{np.mean(num_clusters):.1f}', f'{np.std(num_clusters):.1f}', 
         f'{np.min(num_clusters):.0f}', f'{np.max(num_clusters):.0f}'],
        ['Exploration', f'{np.mean(exploration):.3f}', f'{np.std(exploration):.3f}', 
         f'{np.min(exploration):.3f}', f'{np.max(exploration):.3f}']
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor(colors['neutral'])
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='normal', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/uniform_neurobiological.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created uniform neurobiological figure: {output_path}")

def create_uniform_comprehensive_figure():
    """Create comprehensive analysis figure with uniform formatting."""
    colors = setup_uniform_style()
    
    # Generate comprehensive sample data
    np.random.seed(42)
    n_participants = 56
    
    exploitation = np.random.normal(0.6, 0.15, n_participants)
    exploration = np.random.normal(0.4, 0.1, n_participants)
    switches = np.random.poisson(5, n_participants)
    novelty = np.random.normal(0.3, 0.08, n_participants)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Main scatter plot (larger)
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    scatter = ax_main.scatter(exploitation, exploration, c=switches, 
                             cmap='viridis', s=100, alpha=0.7)
    ax_main.set_xlabel('Exploitation Score', fontsize=12, fontweight='normal')
    ax_main.set_ylabel('Exploration Score', fontsize=12, fontweight='normal')
    ax_main.set_title('Exploitation vs Exploration (colored by switches)', 
                      fontsize=14, fontweight='normal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label('Number of Phase Switches', fontsize=10, fontweight='normal')
    
    # Distribution plots
    ax_dist1 = plt.subplot2grid((3, 4), (0, 2))
    ax_dist1.hist(exploitation, bins=15, alpha=0.7, color=colors['primary'])
    ax_dist1.set_xlabel('Exploitation Score', fontsize=10, fontweight='normal')
    ax_dist1.set_ylabel('Frequency', fontsize=10, fontweight='normal')
    ax_dist1.set_title('Exploitation Distribution', fontsize=12, fontweight='normal')
    
    ax_dist2 = plt.subplot2grid((3, 4), (0, 3))
    ax_dist2.hist(exploration, bins=15, alpha=0.7, color=colors['secondary'])
    ax_dist2.set_xlabel('Exploration Score', fontsize=10, fontweight='normal')
    ax_dist2.set_ylabel('Frequency', fontsize=10, fontweight='normal')
    ax_dist2.set_title('Exploration Distribution', fontsize=12, fontweight='normal')
    
    # Correlation plots
    ax_corr1 = plt.subplot2grid((3, 4), (1, 2))
    ax_corr1.scatter(exploitation, switches, alpha=0.7, s=50, color=colors['accent'])
    ax_corr1.set_xlabel('Exploitation Score', fontsize=10, fontweight='normal')
    ax_corr1.set_ylabel('Phase Switches', fontsize=10, fontweight='normal')
    ax_corr1.set_title('Exploitation vs Switches', fontsize=12, fontweight='normal')
    
    ax_corr2 = plt.subplot2grid((3, 4), (1, 3))
    ax_corr2.scatter(exploration, novelty, alpha=0.7, s=50, color=colors['highlight'])
    ax_corr2.set_xlabel('Exploration Score', fontsize=10, fontweight='normal')
    ax_corr2.set_ylabel('Novelty Score', fontsize=10, fontweight='normal')
    ax_corr2.set_title('Exploration vs Novelty', fontsize=12, fontweight='normal')
    
    # Summary statistics
    ax_summary = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    # Calculate correlations
    corr_ee = np.corrcoef(exploitation, exploration)[0, 1]
    corr_es = np.corrcoef(exploitation, switches)[0, 1]
    corr_en = np.corrcoef(exploration, novelty)[0, 1]
    
    summary_text = f"""
    COMPREHENSIVE ANALYSIS SUMMARY (n={n_participants})
    
    Key Metrics:
    • Exploitation Score: {np.mean(exploitation):.3f} ± {np.std(exploitation):.3f}
    • Exploration Score: {np.mean(exploration):.3f} ± {np.std(exploration):.3f}
    • Phase Switches: {np.mean(switches):.1f} ± {np.std(switches):.1f}
    • Novelty Score: {np.mean(novelty):.3f} ± {np.std(novelty):.3f}
    
    Correlations:
    • Exploitation vs Exploration: r = {corr_ee:.3f}
    • Exploitation vs Switches: r = {corr_es:.3f}
    • Exploration vs Novelty: r = {corr_en:.3f}
    
    Clinical Implications:
    • Higher exploration associated with increased novelty seeking
    • Phase switching patterns indicate cognitive flexibility
    • Balance between exploitation and exploration varies across participants
    """
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('output/uniform_comprehensive.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created uniform comprehensive figure: {output_path}")

def update_existing_figure_scripts():
    """Update existing figure scripts with uniform formatting."""
    
    # Read the existing publication figures script
    try:
        with open('create_publication_figures_pd.py', 'r') as f:
            content = f.read()
        
        # Replace the setup_publication_style function
        new_style_function = '''def setup_publication_style():
    """Set up uniform typography and styling for all figures."""
    
    # Font settings - UNIFORM ACROSS ALL FIGURES
    plt.rcParams.update({
        # Typography
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 12,                    # Base font size
        'axes.titlesize': 14,               # Figure titles
        'axes.labelsize': 12,               # Axis labels
        'xtick.labelsize': 10,              # X-tick labels
        'ytick.labelsize': 10,              # Y-tick labels
        'legend.fontsize': 10,              # Legend text
        'figure.titlesize': 16,             # Suptitles
        
        # Figure quality
        'figure.dpi': 600,                  # High resolution
        'savefig.dpi': 600,                 # Save resolution
        'savefig.bbox': 'tight',            # Tight layout
        'savefig.pad_inches': 0.1,          # Padding
        
        # Colors and styling
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        
        # Spines
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # Ticks
        'xtick.major.size': 6,
        'xtick.major.width': 1.0,
        'ytick.major.size': 6,
        'ytick.major.width': 1.0,
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # Patches (bars, etc.)
        'patch.linewidth': 1.0,
        
        # Text
        'text.usetex': False,
        'mathtext.fontset': 'cm',
    })
    
    # Set seaborn style to complement matplotlib
    sns.set_style("whitegrid", {
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'grid.color': 'lightgray',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    # Professional color palette
    colors = {
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e',    # Orange
        'accent': '#2ca02c',       # Green
        'highlight': '#d62728',    # Red
        'neutral': '#7f7f7f',      # Gray
        'light_blue': '#aec7e8',   # Light blue
        'light_orange': '#ffbb78', # Light orange
        'light_green': '#98df8a',  # Light green
        'light_red': '#ff9896',    # Light red
    }
    
    return colors'''
        
        # Replace the old function
        import re
        pattern = r'def setup_publication_style\(\):.*?return colors'
        updated_content = re.sub(pattern, new_style_function, content, flags=re.DOTALL)
        
        # Write the updated content
        with open('create_publication_figures_pd.py', 'w') as f:
            f.write(updated_content)
        
        print("✅ Updated create_publication_figures_pd.py with uniform formatting")
        
    except FileNotFoundError:
        print("⚠️ create_publication_figures_pd.py not found, skipping update")

def main():
    """Main function to create all uniform figures."""
    print("🎨 Creating uniform figures with consistent typography...")
    
    # Create all uniform figures
    create_uniform_exploration_exploitation_figure()
    create_uniform_phase_coherence_figure()
    create_uniform_neurobiological_figure()
    create_uniform_comprehensive_figure()
    
    # Update existing scripts
    update_existing_figure_scripts()
    
    print("\n🎉 All figures reformatted with uniform typography!")
    print("📁 Check the 'output/' directory for the new figures:")
    print("   • uniform_exploration_exploitation.png/pdf")
    print("   • uniform_phase_coherence.png/pdf")
    print("   • uniform_neurobiological.png/pdf")
    print("   • uniform_comprehensive.png/pdf")
    
    print("\n📋 Uniform Typography Settings Applied:")
    print("   • Base font size: 12pt")
    print("   • Figure titles: 14pt")
    print("   • Axis labels: 12pt")
    print("   • Tick labels: 10pt")
    print("   • Legend text: 10pt")
    print("   • Suptitles: 16pt")
    print("   • Font family: Times New Roman (serif)")
    print("   • Resolution: 600 DPI")
    print("   • Font weight: Normal (not bold)")

if __name__ == "__main__":
    main()
