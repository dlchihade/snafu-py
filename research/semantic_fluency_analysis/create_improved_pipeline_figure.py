#!/usr/bin/env python3
"""
Create an improved semantic fluency analysis pipeline figure
with correct terminology, phase labeling, and accurate representation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set up publication-quality style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def create_improved_pipeline_figure():
    """Create improved semantic fluency analysis pipeline figure"""
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.5, 2.5, 1.8], hspace=0.3)
    
    # Colors
    colors = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'accent': '#2ca02c',
        'highlight': '#d62728',
        'neutral': '#7f7f7f',
        'light_gray': '#e0e0e0',
        'exploitation': '#2ca02c',  # Green for exploitation (high similarity)
        'exploration': '#ff7f0e',   # Orange for exploration (low similarity)
    }
    
    # ============================================================================
    # TOP SECTION: Analysis Pipeline
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10.5)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Pipeline boxes - adjust for 8 steps
    box_width = 1.1
    box_height = 0.6
    y_pos = 0.2
    x_positions = [0.6, 1.8, 3.0, 4.2, 5.4, 6.6, 7.8, 9.0]
    
    steps = [
        ("1. Data\nCollection", "Patient SVF\nrecording with\ntimestamps"),
        ("2. Preprocessing", "Text normalization\nand formatting"),
        ("3. spaCy Pipeline", "Word vectors\n(300-dim embeddings)"),
        ("4. Similarity\nCalculation", "Cosine similarity\nbetween consecutive\nword pairs"),
        ("5. Phase Detection", "Threshold-based\nphase classification\n(τ = 0.6)"),
        ("6. Intra-Phase\nSimilarity", "Pairwise similarities\nwithin each phase"),
        ("7. Inter-Phase\nSimilarity", "Centroid-based\nsimilarity between\nphases"),
        ("8. Coherence\nRatios", "Intra-phase / Inter-phase\nsimilarity ratios")
    ]
    
    boxes = []
    for i, (title, desc) in enumerate(steps):
        x = x_positions[i]
        
        # Main box
        box = FancyBboxPatch(
            (x - box_width/2, y_pos - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor='white',
            edgecolor=colors['primary'],
            linewidth=1.5,
            zorder=2
        )
        ax1.add_patch(box)
        boxes.append((x, y_pos))
        
        # Title
        ax1.text(x, y_pos + 0.15, title, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=colors['primary'])
        
        # Description
        ax1.text(x, y_pos - 0.1, desc, ha='center', va='top',
                fontsize=7.5, color='black')
        
        # Arrow to next
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + box_width/2, y_pos),
                (x_positions[i+1] - box_width/2, y_pos),
                arrowstyle='->',
                mutation_scale=20,
                color=colors['neutral'],
                linewidth=1.5,
                zorder=1
            )
            ax1.add_patch(arrow)
    
    ax1.text(5, 0.95, 'Semantic Fluency Analysis Pipeline', 
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    # ============================================================================
    # MIDDLE SECTION: Intermediate Data Examples
    # ============================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    ax2.text(5, 0.95, 'Intermediate Data Representations', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Raw responses
    y_start = 0.7
    ax2.text(1.2, y_start, 'Raw Responses:', fontsize=9, fontweight='bold', ha='left')
    raw_text = "dog (2s), cat (4s), lion (12s), tiger (12.5s), mouse (20s), rat (30s)"
    ax2.text(1.2, y_start - 0.12, raw_text, fontsize=8, ha='left', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['light_gray'], alpha=0.3))
    
    # Word vectors (show as 300-dim with ellipsis)
    ax2.text(1.2, y_start - 0.35, 'Word Vectors (300-dim):', fontsize=9, fontweight='bold', ha='left')
    vector_text = "dog: [0.12, -0.45, 0.78, ..., 0.23] (300 dimensions)\ncat: [0.15, -0.42, 0.81, ..., 0.25] (300 dimensions)"
    ax2.text(1.2, y_start - 0.47, vector_text, fontsize=7.5, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['light_gray'], alpha=0.3))
    
    # Similarity scores (consecutive pairs)
    ax2.text(5.5, y_start, 'Consecutive Pair Similarities:', fontsize=9, fontweight='bold', ha='left')
    sim_text = "dog → cat: 0.82 (High)\ncat → lion: 0.38 (Low)\nlion → tiger: 0.91 (High)\ntiger → mouse: 0.31 (Low)\nmouse → rat: 0.85 (High)"
    ax2.text(5.5, y_start - 0.12, sim_text, fontsize=7.5, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['light_gray'], alpha=0.3))
    
    # Semantic clusters (based on phases)
    ax2.text(5.5, y_start - 0.4, 'Detected Phases:', fontsize=9, fontweight='bold', ha='left')
    cluster_text = "Exploitation Phase 1: dog, cat (avg sim: 0.82)\nExploration Phase: cat → lion (sim: 0.38)\nExploitation Phase 2: lion, tiger (avg sim: 0.91)\nExploration Phase: tiger → mouse (sim: 0.31)\nExploitation Phase 3: mouse, rat (avg sim: 0.85)"
    ax2.text(5.5, y_start - 0.52, cluster_text, fontsize=7.5, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['light_gray'], alpha=0.3))
    
    # ============================================================================
    # BOTTOM SECTION: Cosine Similarity Plot and Phase Detection
    # ============================================================================
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Create example data: consecutive word-pair similarities
    word_pairs = ['dog→cat', 'cat→lion', 'lion→tiger', 'tiger→mouse', 'mouse→rat']
    similarities = [0.82, 0.38, 0.91, 0.31, 0.85]  # Cosine similarity values
    positions = np.arange(len(word_pairs))
    
    # Threshold line
    threshold = 0.6
    ax3.axhline(y=threshold, color=colors['neutral'], linestyle='--', 
               linewidth=2, label=f'Threshold (τ = {threshold})', zorder=1)
    
    # Plot similarities
    bars = ax3.bar(positions, similarities, width=0.6, alpha=0.7, zorder=2)
    
    # Color bars based on phase type
    for i, (pair, sim) in enumerate(zip(word_pairs, similarities)):
        if sim > threshold:
            bars[i].set_color(colors['exploitation'])  # Green for exploitation
        else:
            bars[i].set_color(colors['exploration'])    # Orange for exploration
    
    # Add similarity values on bars
    for i, (pair, sim) in enumerate(zip(word_pairs, similarities)):
        ax3.text(i, sim + 0.03, f'{sim:.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    
    # Labels
    ax3.set_xlabel('Consecutive Word Pairs', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Cosine Similarity', fontsize=10, fontweight='bold')
    ax3.set_title('Semantic Similarity Between Consecutive Words', 
                 fontsize=11, fontweight='bold', pad=10)
    
    # X-axis
    ax3.set_xticks(positions)
    ax3.set_xticklabels(word_pairs, rotation=45, ha='right', fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.grid(True, alpha=0.3, axis='y', zorder=0)
    
    # Phase annotations
    phase_y = 1.05
    # Phase 1: Exploitation (dog→cat)
    ax3.annotate('', xy=(0, phase_y), xytext=(0.5, phase_y),
                arrowprops=dict(arrowstyle='<->', color=colors['exploitation'], lw=2))
    ax3.text(0.25, phase_y + 0.02, 'EXPLOIT', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=colors['exploitation'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Phase 2: Exploration (cat→lion)
    ax3.annotate('', xy=(1, phase_y), xytext=(1, phase_y),
                arrowprops=dict(arrowstyle='-', color=colors['exploration'], lw=2))
    ax3.text(1, phase_y + 0.02, 'EXPLORE', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=colors['exploration'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Phase 3: Exploitation (lion→tiger)
    ax3.annotate('', xy=(2, phase_y), xytext=(2.5, phase_y),
                arrowprops=dict(arrowstyle='<->', color=colors['exploitation'], lw=2))
    ax3.text(2.25, phase_y + 0.02, 'EXPLOIT', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=colors['exploitation'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Phase 4: Exploration (tiger→mouse)
    ax3.annotate('', xy=(3, phase_y), xytext=(3, phase_y),
                arrowprops=dict(arrowstyle='-', color=colors['exploration'], lw=2))
    ax3.text(3, phase_y + 0.02, 'EXPLORE', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=colors['exploration'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Phase 5: Exploitation (mouse→rat)
    ax3.annotate('', xy=(4, phase_y), xytext=(4.5, phase_y),
                arrowprops=dict(arrowstyle='<->', color=colors['exploitation'], lw=2))
    ax3.text(4.25, phase_y + 0.02, 'EXPLOIT', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=colors['exploitation'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors['exploitation'], label='Exploitation (similarity > 0.6)'),
        mpatches.Patch(color=colors['exploration'], label='Exploration (similarity ≤ 0.6)'),
        plt.Line2D([0], [0], color=colors['neutral'], linestyle='--', 
                  linewidth=2, label=f'Threshold (τ = {threshold})')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    
    # Phase Detection Summary Box
    summary_text = (
        "Phase Detection:\n"
        "• Exploitation: Similarity > 0.6\n"
        "• Exploration: Similarity ≤ 0.6"
    )
    ax3.text(0.02, 0.98, summary_text, transform=ax3.transAxes,
            fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=colors['neutral'], alpha=0.95, linewidth=1.5))
    
    # ============================================================================
    # FOURTH SECTION: Coherence Ratio Computation
    # ============================================================================
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    ax4.text(5, 0.95, 'Coherence Ratio Computation', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Step 1: Intra-phase similarity
    y_pos = 0.65
    ax4.text(1.5, y_pos + 0.15, 'Step 1: Intra-Phase Similarity', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors['primary'])
    
    # Example: Exploitation phase
    exploit_text = (
        "Exploitation Phase (dog, cat):\n"
        "• Pairwise: dog↔cat = 0.82\n"
        "• μ_exploit_intra = 0.82"
    )
    ax4.text(1.5, y_pos - 0.1, exploit_text, ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['exploitation'], 
                     alpha=0.2, edgecolor=colors['exploitation'], linewidth=1.5))
    
    # Example: Exploration phase
    explore_text = (
        "Exploration Phase (cat→lion):\n"
        "• Transition: cat→lion = 0.38\n"
        "• μ_explore_intra = 0.38"
    )
    ax4.text(1.5, y_pos - 0.45, explore_text, ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['exploration'], 
                     alpha=0.2, edgecolor=colors['exploration'], linewidth=1.5))
    
    # Step 2: Inter-phase similarity
    ax4.text(5, y_pos + 0.15, 'Step 2: Inter-Phase Similarity', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors['primary'])
    
    inter_text = (
        "Phase Centroids:\n"
        "• Centroid_1 = mean(dog, cat)\n"
        "• Centroid_2 = mean(lion, tiger)\n"
        "• Similarity = 0.65\n"
        "• μ_inter = 0.65"
    )
    ax4.text(5, y_pos - 0.25, inter_text, ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['light_gray'], 
                     alpha=0.3, edgecolor=colors['neutral'], linewidth=1.5))
    
    # Step 3: Coherence ratios
    ax4.text(8.5, y_pos + 0.15, 'Step 3: Coherence Ratios', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors['primary'])
    
    ratio_text = (
        "Exploitation Ratio:\n"
        "= 0.82 / 0.65\n"
        "= 1.26\n\n"
        "Exploration Ratio:\n"
        "= 0.38 / 0.65\n"
        "= 0.58"
    )
    ax4.text(8.5, y_pos - 0.25, ratio_text, ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['accent'], 
                     alpha=0.2, edgecolor=colors['accent'], linewidth=1.5))
    
    # Arrows between steps
    arrow1 = FancyArrowPatch(
        (2.8, y_pos - 0.25),
        (4.2, y_pos - 0.25),
        arrowstyle='->',
        mutation_scale=20,
        color=colors['neutral'],
        linewidth=1.5,
        zorder=1
    )
    ax4.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch(
        (5.8, y_pos - 0.25),
        (7.2, y_pos - 0.25),
        arrowstyle='->',
        mutation_scale=20,
        color=colors['neutral'],
        linewidth=1.5,
        zorder=1
    )
    ax4.add_patch(arrow2)
    
    # Interpretation box
    interpret_text = (
        "Interpretation:\n"
        "• Ratio > 1.0: Words within phase more similar than phases to each other\n"
        "• Exploitation = 1.26: Strong clustering\n"
        "• Exploration = 0.58: Less clustering"
    )
    ax4.text(5, 0.15, interpret_text, ha='center', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=colors['accent'], alpha=0.95, linewidth=1.5))
    
    # Save figure
    output_dir = Path('output/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_png = output_dir / 'improved_semantic_fluency_pipeline.png'
    output_pdf = output_dir / 'improved_semantic_fluency_pipeline.pdf'
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    print(f"✅ Saved improved pipeline figure:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    plt.close(fig)


if __name__ == '__main__':
    create_improved_pipeline_figure()

