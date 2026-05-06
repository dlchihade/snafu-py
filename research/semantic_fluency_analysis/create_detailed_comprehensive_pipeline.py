#!/usr/bin/env python3
"""
Create a comprehensive, detailed semantic fluency pipeline figure
CLEAN VERSION - No boxes, proper LaTeX formulas, professional layout
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# Set up publication-quality style - clean and professional
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'font.weight': 'normal',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'none',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'text.usetex': False,  # Use matplotlib's mathtext
    'mathtext.fontset': 'cm',
})

def create_detailed_comprehensive_pipeline():
    """Create comprehensive pipeline - CLEAN VERSION with LaTeX formulas"""
    
    # Create larger figure with more space to prevent overlap
    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(6, 1, height_ratios=[0.4, 1.4, 2.2, 2.4, 2.4, 2.2], hspace=0.25)
    
    # Clean neutral color scheme
    text_color = '#000000'
    arrow_color = '#666666'
    
    # ============================================================================
    # SECTION 1: High-Level Pipeline Overview (18 steps)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 22)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_facecolor('white')
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ax1.text(11, 0.9, 'Complete Semantic Fluency Analysis Pipeline', 
            ha='center', va='top', fontsize=14, fontweight='bold', color=text_color)
    
    # Simple step numbers with arrows - no colored boxes, better spacing
    step_nums = [str(i) for i in range(1, 19)]
    box_w = 0.7
    box_h = 0.4
    y_pos = 0.4
    x_start = 0.8
    x_spacing = 1.1
    
    for i, num in enumerate(step_nums):
        x = x_start + i * x_spacing
        # Simple text, no boxes
        ax1.text(x, y_pos, num, ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_color)
        
        if i < len(step_nums) - 1:
            arrow = FancyArrowPatch(
                (x + box_w/2, y_pos),
                (x + x_spacing - box_w/2, y_pos),
                arrowstyle='->',
                mutation_scale=10,
                color=arrow_color,
                linewidth=0.8,
                zorder=1
            )
            ax1.add_patch(arrow)
    
    # ============================================================================
    # SECTION 2: Steps 1-5 Detailed
    # ============================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, 22)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_facecolor('white')
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    ax2.text(11, 0.95, 'Steps 1-5: Data Collection, Preprocessing & Embedding', 
            ha='center', va='top', fontsize=11, fontweight='bold', color=text_color)
    
    # Clean text layout - no boxes, better spacing to prevent overlap
    ax2.text(2, 0.80, 'Step 1-2: Raw Data', fontsize=10, fontweight='bold', ha='left', color=text_color)
    ax2.text(2, 0.45, 'dog (2s), cat (4s), lion (12s),\ntiger (12.5s), mouse (20s), rat (30s)',
            fontsize=8, ha='left', va='top', color=text_color, family='monospace')
    
    ax2.text(8, 0.80, 'Step 3: Preprocessing', fontsize=10, fontweight='bold', ha='left', color=text_color)
    ax2.text(8, 0.45, 'Text normalization\nLowercase conversion\nPunctuation removal\nStop word filtering',
            fontsize=8, ha='left', va='top', color=text_color)
    
    ax2.text(14, 0.80, 'Step 4: Word Embeddings (spaCy)', fontsize=10, fontweight='bold', ha='left', color=text_color)
    ax2.text(14, 0.45, 'dog: [0.12, -0.45, 0.78, ..., 0.23]\ncat: [0.15, -0.42, 0.81, ..., 0.25]\n300-dimensional vectors',
            fontsize=8, ha='left', va='top', color=text_color, family='monospace')
    
    ax2.text(19, 0.80, 'Step 5: Vector Validation', fontsize=10, fontweight='bold', ha='left', color=text_color)
    ax2.text(19, 0.45, 'Filter valid embeddings\nCheck vector norms\nRemove OOV words',
            fontsize=8, ha='left', va='top', color=text_color)
    
    # ============================================================================
    # SECTION 3: Steps 6-8 Detailed
    # ============================================================================
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_xlim(0, 22)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_facecolor('white')
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    ax3.text(11, 0.95, 'Steps 6-8: Similarity Computation & Phase Detection', 
            ha='center', va='top', fontsize=11, fontweight='bold', color=text_color)
    
    ax3.text(3, 0.85, 'Step 6: Consecutive Similarities', fontsize=10, fontweight='bold', ha='left', color=text_color)
    ax3.text(3, 0.25, 'dog→cat: 0.82\ncat→lion: 0.38\nlion→tiger: 0.91\ntiger→mouse: 0.31\nmouse→rat: 0.85',
            fontsize=8, ha='left', va='top', color=text_color, family='monospace')
    ax3.text(3, 0.65, r'Formula: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$',
            fontsize=9, ha='left', va='top', color=text_color)
    
    ax3.text(11, 0.85, 'Step 7: Phase Detection', fontsize=10, fontweight='bold', ha='left', color=text_color)
    ax3.text(11, 0.25, r'Threshold: $\tau = 0.6$\n\nif similarity $> \tau$:\n    → Exploitation phase\nelse:\n    → Exploration phase',
            fontsize=8, ha='left', va='top', color=text_color, family='monospace')
    
    ax3.text(18, 0.85, 'Step 8: Detected Phases', fontsize=10, fontweight='bold', ha='left', color=text_color)
    ax3.text(18, 0.25, 'Exploit: [dog, cat]\nExplore: [cat→lion]\nExploit: [lion, tiger]\nExplore: [tiger→mouse]\nExploit: [mouse, rat]',
            fontsize=8, ha='left', va='top', color=text_color, family='monospace')
    
    # ============================================================================
    # SECTION 4: Steps 9-10 Detailed (Intra-Phase Analysis)
    # ============================================================================
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_xlim(0, 22)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_facecolor('white')
    ax4.grid(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    
    ax4.text(11, 0.95, 'Steps 9-10: Intra-Phase Similarity Computation', 
            ha='center', va='top', fontsize=11, fontweight='bold', color=text_color)
    
    ax4.text(3, 0.88, 'Step 9: Exploitation Phase [dog, cat]', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    matrix_text = (
        "Similarity Matrix:\n"
        "      dog    cat\n"
        "dog  1.00   0.82\n"
        "cat  0.82   1.00\n\n"
        "Upper triangle: 0.82"
    )
    ax4.text(3, 0.25, matrix_text, ha='left', va='top', fontsize=8,
            color=text_color, family='monospace')
    ax4.text(3, 0.65, r'$\mu_{\text{exploit}}^{\text{intra}} = 0.82$',
            fontsize=9, ha='left', va='top', color=text_color)
    
    ax4.text(10, 0.88, 'Step 9: Exploration Phase [cat→lion]', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    explore_text = (
        "Transition similarity:\n"
        "cat→lion = 0.38\n\n"
        "For single transitions:\n"
        "intra = transition similarity"
    )
    ax4.text(10, 0.25, explore_text, ha='left', va='top', fontsize=8,
            color=text_color, family='monospace')
    ax4.text(10, 0.65, r'$\mu_{\text{explore}}^{\text{intra}} = 0.38$',
            fontsize=9, ha='left', va='top', color=text_color)
    
    ax4.text(17, 0.88, 'Step 10: Phase Centroid Calculation', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    centroid_text = (
        "For each phase:\n"
        "Centroid = mean(all word vectors)\n\n"
        "Phase 1 (dog, cat):\n"
        r"$c_1 = \text{mean}([\mathbf{dog}, \mathbf{cat}])$" + "\n"
        "   = [0.135, -0.435, 0.795, ...]\n\n"
        "Phase 2 (lion, tiger):\n"
        r"$c_2 = \text{mean}([\mathbf{lion}, \mathbf{tiger}])$" + "\n"
        "   = [0.180, -0.380, 0.820, ...]\n\n"
        "Aggregation:\n"
        r"$\mu_{\text{exploit}}^{\text{intra}} = \text{mean}$" + "\n"
        r"  (all pairwise similarities" + "\n"
        r"  within exploit phases)"
    )
    ax4.text(17, 0.25, centroid_text, ha='left', va='top', fontsize=8, color=text_color)
    
    # ============================================================================
    # SECTION 5: Steps 11-12 Detailed (Inter-Phase Analysis)
    # ============================================================================
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.set_xlim(0, 22)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_facecolor('white')
    ax5.grid(False)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    
    ax5.text(11, 0.95, 'Steps 11-12: Inter-Phase Similarity Computation', 
            ha='center', va='top', fontsize=11, fontweight='bold', color=text_color)
    
    ax5.text(4, 0.88, 'Step 11: Centroid Normalization', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    norm_text = (
        "Normalize to unit length:\n"
        r"$\hat{c} = \frac{c}{||c||}$" + "\n\n"
        r"$\hat{c}_1 = \frac{c_1}{||c_1||}$" + "\n"
        r"$\hat{c}_2 = \frac{c_2}{||c_2||}$" + "\n"
        r"$\hat{c}_3 = \frac{c_3}{||c_3||}$" + "\n\n"
        "For cosine similarity:\n"
        r"$\cos(\theta) = \hat{c}_i \cdot \hat{c}_j$"
    )
    ax5.text(4, 0.25, norm_text, ha='left', va='top', fontsize=9, color=text_color)
    
    ax5.text(12, 0.88, 'Step 12: Inter-Phase Similarity', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    inter_text = (
        "Compute similarity between\n"
        "all phase centroid pairs:\n\n"
        r"$S_{\text{inter}}(1,2) = \hat{c}_1 \cdot \hat{c}_2 = 0.65$" + "\n"
        r"$S_{\text{inter}}(1,3) = \hat{c}_1 \cdot \hat{c}_3 = 0.58$" + "\n"
        r"$S_{\text{inter}}(2,3) = \hat{c}_2 \cdot \hat{c}_3 = 0.72$"
    )
    ax5.text(12, 0.45, inter_text, ha='left', va='top', fontsize=9, color=text_color)
    ax5.text(12, 0.25, r'$\mu_{\text{inter}} = \text{mean}(\text{all } S_{\text{inter}}) = 0.65$',
            fontsize=9, ha='left', va='top', color=text_color)
    
    ax5.text(19, 0.88, 'Inter-Phase Matrix', fontsize=10, fontweight='bold', ha='left', color=text_color)
    matrix_inter = (
        "    P1   P2   P3\n"
        "P1 1.00  .65  .58\n"
        "P2  .65 1.00  .72\n"
        "P3  .58  .72 1.00\n\n"
        "Upper triangle mean: 0.65"
    )
    ax5.text(19, 0.25, matrix_inter, ha='left', va='top', fontsize=8,
            color=text_color, family='monospace')
    
    # ============================================================================
    # SECTION 6: Steps 13-18 Detailed (Coherence & Final Metrics)
    # ============================================================================
    ax6 = fig.add_subplot(gs[5, 0])
    ax6.set_xlim(0, 22)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_facecolor('white')
    ax6.grid(False)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    
    ax6.text(11, 0.95, 'Steps 13-18: Coherence Ratios & Final Metrics', 
            ha='center', va='top', fontsize=11, fontweight='bold', color=text_color)
    
    # Top row: Steps 13-14 - more spacing
    ax6.text(3, 0.88, 'Step 13: Coherence Ratios', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    ratio_text = (
        "Exploitation Coherence:\n"
        r"$= \frac{\mu_{\text{exploit}}^{\text{intra}}}{\mu_{\text{inter}}} = \frac{0.82}{0.65} = 1.26$" + "\n\n"
        "Exploration Coherence:\n"
        r"$= \frac{\mu_{\text{explore}}^{\text{intra}}}{\mu_{\text{inter}}} = \frac{0.38}{0.65} = 0.58$"
    )
    ax6.text(3, 0.55, ratio_text, ha='left', va='top', fontsize=9, color=text_color)
    
    ax6.text(11, 0.88, 'Step 14: Phase Separation Index (PSI)', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    psi_text = (
        r"$\text{PSI} = \frac{\mu_{\text{exploit}}^{\text{intra}} + \mu_{\text{explore}}^{\text{intra}}}{2} - \mu_{\text{inter}}$" + "\n\n"
        r"$= \frac{0.82 + 0.38}{2} - 0.65 = 0.60 - 0.65 = -0.05$" + "\n\n"
        "(Negative = phases less distinct)"
    )
    ax6.text(11, 0.55, psi_text, ha='left', va='top', fontsize=9, color=text_color)
    
    ax6.text(18, 0.88, 'Steps 15-17: Additional Metrics', 
            fontsize=10, fontweight='bold', ha='left', color=text_color)
    
    additional_text = (
        "Step 15: Basic Phase Metrics\n"
        "• Exploitation %: 60%\n"
        "• Exploration %: 40%\n"
        "• Number of switches: 4\n\n"
        "Step 16: Clustering\n"
        "• Coefficient: 0.75\n"
        "• Num clusters: 3\n\n"
        "Step 17: Word Frequency\n"
        "• Mean rank: 1250\n"
        "• Median rank: 980"
    )
    ax6.text(18, 0.55, additional_text, ha='left', va='top', fontsize=8, color=text_color)
    
    # Bottom: Step 18 - more space from above
    ax6.text(11, 0.35, 'Step 18: Final Results', 
            fontsize=10, fontweight='bold', ha='center', color=text_color)
    ax6.text(11, 0.20, 'Comprehensive metrics exported for statistical analysis', 
            ha='center', va='top', fontsize=9, fontstyle='italic', color=text_color)
    
    # Save figure
    output_dir = Path('output/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_png = output_dir / 'detailed_comprehensive_pipeline.png'
    output_pdf = output_dir / 'detailed_comprehensive_pipeline.pdf'
    output_svg = output_dir / 'detailed_comprehensive_pipeline.svg'
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.02,
                facecolor='white', edgecolor='none')
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', pad_inches=0.02,
                facecolor='white', edgecolor='none')
    fig.savefig(output_svg, format='svg', bbox_inches='tight', pad_inches=0.02,
                facecolor='white', edgecolor='none')
    
    print(f"✅ Saved detailed comprehensive pipeline (CLEAN VERSION):")
    print(f"   PNG: {output_png}")
    print(f"   SVG: {output_svg}")
    print(f"   PDF: {output_pdf}")
    
    plt.close(fig)


if __name__ == '__main__':
    create_detailed_comprehensive_pipeline()
