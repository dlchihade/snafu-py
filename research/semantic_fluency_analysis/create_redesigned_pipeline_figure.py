#!/usr/bin/env python3
"""
Redesigned semantic fluency analysis pipeline figure
- Landscape orientation
- Consistent box sizes
- Color coding: Blue (exploitation), Orange (exploration), Gray (neutral)
- Larger fonts, better spacing
- Results panel with key findings
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

# Set up publication-quality style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def create_redesigned_pipeline_figure():
    """Create redesigned landscape pipeline figure"""
    
    # Landscape figure
    fig = plt.figure(figsize=(16, 10))
    
    # Colors
    colors = {
        'exploitation': '#2E86AB',  # Blue
        'exploration': '#F77F00',   # Orange
        'neutral': '#6C757D',        # Gray
        'light_gray': '#E9ECEF',
        'white': '#FFFFFF',
    }
    
    # Box dimensions (in figure coordinates, will be scaled)
    box_width = 1.2
    box_height = 0.6
    box_spacing = 0.3
    
    # ============================================================================
    # MAIN PIPELINE SECTION (Top)
    # ============================================================================
    
    # Section 1: Data Processing (Left)
    section1_x = 0.5
    section1_y = 8.5
    
    # Section 2: Phase Detection (Middle)
    section2_x = 6.0
    section2_y = 8.5
    
    # Section 3: Coherence Computation (Right)
    section3_x = 11.5
    section3_y = 8.5
    
    # Vertical separator lines
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 16)
    ax_main.set_ylim(0, 10)
    ax_main.axis('off')
    
    # Section separators (dotted vertical lines)
    ax_main.plot([5.5, 5.5], [7, 9.5], '--', color=colors['neutral'], 
                linewidth=2, alpha=0.5, zorder=0)
    ax_main.plot([11, 11], [7, 9.5], '--', color=colors['neutral'], 
                linewidth=2, alpha=0.5, zorder=0)
    
    # Section headers
    ax_main.text(section1_x + 0.6, 9.3, 'SECTION 1: DATA PROCESSING', 
                fontsize=11, fontweight='bold', ha='left', color=colors['neutral'])
    ax_main.text(section2_x + 0.6, 9.3, 'SECTION 2: PHASE DETECTION', 
                fontsize=11, fontweight='bold', ha='left', color=colors['neutral'])
    ax_main.text(section3_x + 0.6, 9.3, 'SECTION 3: COHERENCE COMPUTATION', 
                fontsize=11, fontweight='bold', ha='left', color=colors['neutral'])
    
    # SECTION 1: Data Processing
    steps1 = [
        ("1. Data\nCollection", colors['neutral']),
        ("2. Text\nPreprocessing", colors['neutral']),
        ("3. Word\nEmbeddings\n(300-dim)", colors['neutral']),
        ("4. Consecutive\nSimilarity\nCalculation", colors['neutral']),
    ]
    
    x_pos = section1_x
    for i, (title, color) in enumerate(steps1):
        box = FancyBboxPatch(
            (x_pos, section1_y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.2,
            zorder=2
        )
        ax_main.add_patch(box)
        ax_main.text(x_pos + box_width/2, section1_y, title, 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps1) - 1:
            arrow = FancyArrowPatch(
                (x_pos + box_width, section1_y),
                (x_pos + box_width + box_spacing, section1_y),
                arrowstyle='->',
                mutation_scale=20,
                color=colors['neutral'],
                linewidth=2,
                zorder=1
            )
            ax_main.add_patch(arrow)
        
        x_pos += box_width + box_spacing
    
    # SECTION 2: Phase Detection
    steps2 = [
        ("5. Threshold\nClassification\n(median split)", colors['neutral']),
        ("6. Phase\nAssignment", colors['neutral']),
    ]
    
    x_pos = section2_x
    for i, (title, color) in enumerate(steps2):
        box = FancyBboxPatch(
            (x_pos, section2_y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.2,
            zorder=2
        )
        ax_main.add_patch(box)
        ax_main.text(x_pos + box_width/2, section2_y, title, 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps2) - 1:
            arrow = FancyArrowPatch(
                (x_pos + box_width, section2_y),
                (x_pos + box_width + box_spacing, section2_y),
                arrowstyle='->',
                mutation_scale=20,
                color=colors['neutral'],
                linewidth=2,
                zorder=1
            )
            ax_main.add_patch(arrow)
        
        x_pos += box_width + box_spacing
    
    # SECTION 3: Coherence Computation
    steps3 = [
        ("7. Intra-Phase\nSimilarity\nComputation", colors['exploitation']),
        ("8. Inter-Phase\nSimilarity\nComputation", colors['neutral']),
        ("9. Coherence\nRatios", colors['exploration']),
    ]
    
    x_pos = section3_x
    for i, (title, color) in enumerate(steps3):
        box = FancyBboxPatch(
            (x_pos, section3_y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.3 if color != colors['neutral'] else 0.2,
            zorder=2
        )
        ax_main.add_patch(box)
        ax_main.text(x_pos + box_width/2, section3_y, title, 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps3) - 1:
            arrow = FancyArrowPatch(
                (x_pos + box_width, section3_y),
                (x_pos + box_width + box_spacing, section3_y),
                arrowstyle='->',
                mutation_scale=20,
                color=colors['neutral'],
                linewidth=2,
                zorder=1
            )
            ax_main.add_patch(arrow)
        
        x_pos += box_width + box_spacing
    
    # ============================================================================
    # DETAILED EXAMPLE PANEL (Middle)
    # ============================================================================
    
    example_y = 6.0
    ax_main.text(8, example_y + 1.2, 'DETAILED EXAMPLE: Phase Detection', 
                fontsize=12, fontweight='bold', ha='center', color='black')
    
    # Example data
    example_box = FancyBboxPatch(
        (1, example_y - 0.8),
        14, 1.6,
        boxstyle="round,pad=0.2",
        facecolor=colors['light_gray'],
        edgecolor=colors['neutral'],
        linewidth=2,
        alpha=0.3,
        zorder=1
    )
    ax_main.add_patch(example_box)
    
    # Word sequence
    ax_main.text(2, example_y + 0.5, 'Word Sequence:', fontsize=10, fontweight='bold', ha='left')
    ax_main.text(2, example_y + 0.2, 'dog → cat → lion → tiger → mouse → rat', 
                fontsize=10, ha='left', family='monospace')
    
    # Similarity values with gradient bars
    ax_main.text(2, example_y - 0.1, 'Consecutive Similarities:', fontsize=10, fontweight='bold', ha='left')
    
    similarities = [0.82, 0.38, 0.91, 0.31, 0.85]
    pairs = ['dog→cat', 'cat→lion', 'lion→tiger', 'tiger→mouse', 'mouse→rat']
    threshold = 0.6
    
    bar_x_start = 2
    bar_y = example_y - 0.4
    bar_width = 0.15
    bar_height = 0.3
    
    for i, (pair, sim) in enumerate(zip(pairs, similarities)):
        x_pos_bar = bar_x_start + i * 1.5
        color = colors['exploitation'] if sim > threshold else colors['exploration']
        
        # Gradient bar (simplified as solid for now)
        bar = Rectangle(
            (x_pos_bar, bar_y),
            bar_width, bar_height * sim,
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            alpha=0.7,
            zorder=2
        )
        ax_main.add_patch(bar)
        
        # Label
        ax_main.text(x_pos_bar + bar_width/2, bar_y - 0.15, pair, 
                    fontsize=9, ha='center', rotation=45)
        ax_main.text(x_pos_bar + bar_width/2, bar_y + bar_height * sim + 0.05, f'{sim:.2f}', 
                    fontsize=9, ha='center', fontweight='bold')
    
    # Threshold line
    ax_main.plot([bar_x_start, bar_x_start + 7.5], 
                [bar_y + bar_height * threshold, bar_y + bar_height * threshold],
                '--', color=colors['neutral'], linewidth=2, label='Threshold (median)')
    ax_main.text(bar_x_start + 7.8, bar_y + bar_height * threshold, 'Threshold = 0.6', 
                fontsize=9, va='center', color=colors['neutral'])
    
    # Phase labels (simplified)
    ax_main.text(2, example_y - 0.9, 'Detected Phases:', fontsize=10, fontweight='bold', ha='left')
    phase_text = (
        "• Exploitation: [dog, cat], [lion, tiger], [mouse, rat]\n"
        "• Exploration: [cat→lion], [tiger→mouse]"
    )
    ax_main.text(2, example_y - 1.3, phase_text, fontsize=10, ha='left', va='top')
    
    # ============================================================================
    # RESULTS PANEL (Bottom Left)
    # ============================================================================
    
    results_y = 3.5
    results_box = FancyBboxPatch(
        (0.5, results_y - 2),
        7, 2,
        boxstyle="round,pad=0.3",
        facecolor='white',
        edgecolor=colors['neutral'],
        linewidth=2,
        zorder=2
    )
    ax_main.add_patch(results_box)
    
    ax_main.text(4, results_y + 1.5, 'KEY RESULTS (n = 52 PD patients)', 
                fontsize=12, fontweight='bold', ha='center', color='black')
    
    # Main findings
    results_text = (
        "• Exploitation coherence = 0.622 ± 0.146\n"
        "• Exploration coherence = 0.422 ± 0.083\n"
        "• E-E index = 1.52 ± 0.39 (75% > 1.0)\n"
        "• Paired t-test: t = 8.83, p < 0.001"
    )
    ax_main.text(1, results_y + 0.5, results_text, fontsize=10, ha='left', va='top')
    
    # Bar chart
    chart_x = 4.5
    chart_y = results_y - 0.5
    chart_width = 2
    chart_height = 1.2
    
    # Bar chart axes
    ax_main.plot([chart_x, chart_x + chart_width], [chart_y, chart_y], 
                'k-', linewidth=1.5, zorder=3)
    ax_main.plot([chart_x, chart_x], [chart_y, chart_y + chart_height], 
                'k-', linewidth=1.5, zorder=3)
    
    # Bars
    exploit_val = 0.622
    explore_val = 0.422
    max_val = 0.8
    
    bar1 = Rectangle(
        (chart_x + 0.2, chart_y),
        0.3, (exploit_val / max_val) * chart_height,
        facecolor=colors['exploitation'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7,
        zorder=2
    )
    ax_main.add_patch(bar1)
    ax_main.text(chart_x + 0.35, chart_y + (exploit_val / max_val) * chart_height + 0.05,
                f'{exploit_val:.3f}', fontsize=9, ha='center', fontweight='bold')
    ax_main.text(chart_x + 0.35, chart_y - 0.15, 'Exploit', fontsize=9, ha='center')
    
    bar2 = Rectangle(
        (chart_x + 0.7, chart_y),
        0.3, (explore_val / max_val) * chart_height,
        facecolor=colors['exploration'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7,
        zorder=2
    )
    ax_main.add_patch(bar2)
    ax_main.text(chart_x + 0.85, chart_y + (explore_val / max_val) * chart_height + 0.05,
                f'{explore_val:.3f}', fontsize=9, ha='center', fontweight='bold')
    ax_main.text(chart_x + 0.85, chart_y - 0.15, 'Explore', fontsize=9, ha='center')
    
    ax_main.text(chart_x + chart_width/2, chart_y + chart_height + 0.1,
                'Intra-Phase Coherence', fontsize=10, ha='center', fontweight='bold')
    
    # ============================================================================
    # COHERENCE COMPUTATION SUMMARY (Bottom Right)
    # ============================================================================
    
    detail_y = 3.5
    detail_box = FancyBboxPatch(
        (8.5, detail_y - 2),
        7, 2,
        boxstyle="round,pad=0.3",
        facecolor='white',
        edgecolor=colors['neutral'],
        linewidth=2,
        zorder=2
    )
    ax_main.add_patch(detail_box)
    
    ax_main.text(12, detail_y + 1.5, 'COHERENCE COMPUTATION SUMMARY', 
                fontsize=12, fontweight='bold', ha='center', color='black')
    
    detail_text = (
        "Intra-Phase Similarity:\n"
        "• Pairwise similarities within each phase type\n"
        "• Separate means for exploitation and exploration phases\n\n"
        "Inter-Phase Similarity:\n"
        "• Centroid-based similarity between all phases\n"
        "• Single mean across all phase pairs\n\n"
        "Final Ratios:\n"
        "• Exploitation Coherence = Intra_exploit / Inter\n"
        "• Exploration Coherence = Intra_explore / Inter"
    )
    ax_main.text(9, detail_y + 0.3, detail_text, fontsize=10, ha='left', va='top')
    
    # ============================================================================
    # LEGEND (Bottom)
    # ============================================================================
    
    legend_y = 0.8
    legend_elements = [
        mpatches.Patch(color=colors['exploitation'], label='Exploitation-related'),
        mpatches.Patch(color=colors['exploration'], label='Exploration-related'),
        mpatches.Patch(color=colors['neutral'], label='Processing steps'),
    ]
    ax_main.legend(handles=legend_elements, loc='lower center', 
                  ncol=3, fontsize=10, framealpha=0.9)
    
    # Save figure
    output_dir = Path('output/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_png = output_dir / 'redesigned_semantic_fluency_pipeline.png'
    output_pdf = output_dir / 'redesigned_semantic_fluency_pipeline.pdf'
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.3)
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', pad_inches=0.3)
    
    print(f"✅ Saved redesigned pipeline figure:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    plt.close(fig)


if __name__ == '__main__':
    create_redesigned_pipeline_figure()

