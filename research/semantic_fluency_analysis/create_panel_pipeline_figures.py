#!/usr/bin/env python3
"""
Create publication-ready panel figures for semantic fluency pipeline
Each panel is designed as a standalone PowerPoint slide with symmetric layout
Optimized for journal submissions with clear aesthetics and minimal white space
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Publication-quality style settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'font.weight': 'normal',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Professional color scheme
colors = {
    'data': '#4a90e2',
    'preprocessing': '#1f77b4',
    'embedding': '#2e86ab',
    'similarity': '#ff7f0e',
    'phase': '#ff9500',
    'intra': '#2ca02c',
    'inter': '#28a745',
    'coherence': '#20c997',
    'metrics': '#17a2b8',
    'output': '#6c757d',
    'light_gray': '#f5f5f5',
    'exploitation': '#2ca02c',
    'exploration': '#ff7f0e',
    'neutral': '#7f7f7f',
    'border': '#cccccc',
}

# Standardized publication-ready constants
PUB_STYLE = {
    # Slide dimensions (standard PowerPoint ratio)
    'figure_width': 10,
    'figure_height': 7.5,
    
    # Title settings
    'panel_title_y': 0.97,
    'panel_title_fontsize': 14,
    'panel_title_color': '#000000',
    
    # Step title settings
    'step_title_fontsize': 11,
    'step_title_color': '#000000',
    'title_to_box_gap': 0.05,
    
    # Content settings
    'content_fontsize': 10,
    'box_padding': 0.12,
    'box_linewidth': 1.2,
    'box_alpha': 0.25,
    
    # Layout settings
    'y_lim_bottom': 0.05,
    'y_lim_top': 1.0,
    'section_spacing': 0.15,  # Vertical spacing between sections
}

def add_content_box(ax, x, y, text, width=3.8, height=None, color='light_gray', 
                    alpha=None, edgecolor=None, fontsize=None, ha='center', va='top', 
                    flat_box=False):
    """Helper function to add a standardized content box"""
    if alpha is None:
        alpha = PUB_STYLE['box_alpha']
    if edgecolor is None:
        edgecolor = colors.get('border', colors['neutral'])
    if fontsize is None:
        fontsize = PUB_STYLE['content_fontsize']
    
    # Calculate height if not provided (estimate based on text lines)
    if height is None:
        num_lines = text.count('\n') + 1
        height = max(0.4, num_lines * 0.08 + 0.15)
    
    if flat_box:
        # Flat box with thin gray border, white background
        box = FancyBboxPatch(
            (x - width/2, y - height), width, height,
            boxstyle="square,pad=0",
            facecolor='white',
            edgecolor='#d0d0d0',
            linewidth=0.5,
            alpha=1.0,
            zorder=1
        )
    else:
    box = FancyBboxPatch(
        (x - width/2, y - height), width, height,
        boxstyle=f"round,pad={PUB_STYLE['box_padding']}",
        facecolor=colors.get(color, colors['light_gray']),
        edgecolor=edgecolor,
        linewidth=PUB_STYLE['box_linewidth'],
        alpha=alpha,
        zorder=1
    )
    ax.add_patch(box)
    # Position text inside box, accounting for box padding
    if flat_box:
        text_y = y - 0.02
        padding = 0.05
    else:
    text_y = y - PUB_STYLE['box_padding'] - 0.02
        padding = PUB_STYLE['box_padding']
    ax.text(x, text_y, text, ha=ha, va=va, fontsize=fontsize,
            color='#000000', zorder=2)
    return box

def create_panel_1_data_preprocessing():
    """Panel 1: Steps 1-5 - Data Collection, Preprocessing & Embedding"""
    fig = plt.figure(figsize=(PUB_STYLE['figure_width'], PUB_STYLE['figure_height']))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, PUB_STYLE['figure_width'])
    ax.set_ylim(PUB_STYLE['y_lim_bottom'], PUB_STYLE['y_lim_top'])
    ax.axis('off')
    
    # Panel title
    ax.text(PUB_STYLE['figure_width']/2, PUB_STYLE['panel_title_y'], 
            'Panel 1: Data Collection, Preprocessing & Embedding', 
            ha='center', va='top', fontsize=PUB_STYLE['panel_title_fontsize'], 
            fontweight='bold', color=PUB_STYLE['panel_title_color'])
    
    # Top row: Steps 1-3
    top_y = 0.82
    step_title_y = top_y
    box_y = step_title_y - PUB_STYLE['title_to_box_gap']
    
    # Step 1-2: Raw Data (left)
    ax.text(2.5, step_title_y, 'Steps 1-2: Raw Data', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=PUB_STYLE['step_title_color'])
    add_content_box(ax, 2.5, box_y, 
                    'dog (2s), cat (4s), lion (12s),\ntiger (12.5s), mouse (20s), rat (30s)',
                    width=4.0, color='data', alpha=0.2, edgecolor=colors['data'])
    
    # Step 3: Preprocessing (right)
    ax.text(7.5, step_title_y, 'Step 3: Preprocessing', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=PUB_STYLE['step_title_color'])
    add_content_box(ax, 7.5, box_y,
                    'Text normalization\nLowercase conversion\nPunctuation removal\nStop word filtering',
                    width=4.0, color='preprocessing', alpha=0.2, edgecolor=colors['preprocessing'])
    
    # Bottom row: Steps 4-5
    bottom_y = 0.45
    step_title_y2 = bottom_y
    box_y2 = step_title_y2 - PUB_STYLE['title_to_box_gap']
    
    # Step 4: Word Embeddings (left)
    ax.text(2.5, step_title_y2, 'Step 4: Word Embeddings (spaCy)', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=PUB_STYLE['step_title_color'])
    add_content_box(ax, 2.5, box_y2,
                    'dog: [0.12, -0.45, 0.78, ..., 0.23]\ncat: [0.15, -0.42, 0.81, ..., 0.25]\n300-dimensional vectors',
                    width=4.0, color='embedding', alpha=0.2, edgecolor=colors['embedding'])
    
    # Step 5: Vector Validation (right)
    ax.text(7.5, step_title_y2, 'Step 5: Vector Validation', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=PUB_STYLE['step_title_color'])
    add_content_box(ax, 7.5, box_y2,
                    'Filter valid embeddings\nCheck vector norms\nRemove OOV words',
                    width=4.0, color='embedding', alpha=0.2, edgecolor=colors['embedding'])
    
    output_dir = Path('output/figures/panels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'panel1_data_preprocessing.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel1_data_preprocessing.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel1_data_preprocessing.svg', format='svg', bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"✅ Saved Panel 1: {output_dir / 'panel1_data_preprocessing.png'}")

def create_panel_2_similarity_phases():
    """Panel 2: Steps 6-8 - Similarity Computation & Phase Detection"""
    fig = plt.figure(figsize=(PUB_STYLE['figure_width'], PUB_STYLE['figure_height']))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, PUB_STYLE['figure_width'])
    ax.set_ylim(PUB_STYLE['y_lim_bottom'], PUB_STYLE['y_lim_top'])
    ax.axis('off')
    
    # Panel title
    ax.text(PUB_STYLE['figure_width']/2, PUB_STYLE['panel_title_y'], 
            'Panel 2: Similarity Computation & Phase Detection', 
            ha='center', va='top', fontsize=PUB_STYLE['panel_title_fontsize'], 
            fontweight='bold', color=PUB_STYLE['panel_title_color'])
    
    # Top row: Steps 6-7
    top_y = 0.78
    step_title_y = top_y
    box_y = step_title_y - PUB_STYLE['title_to_box_gap']
    
    # Step 6: Consecutive Similarities (left)
    ax.text(2.5, step_title_y, 'Step 6: Consecutive Similarities', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=PUB_STYLE['step_title_color'])
    add_content_box(ax, 2.5, box_y,
                    'dog->cat: 0.82\ncat->lion: 0.38\nlion->tiger: 0.91\ntiger->mouse: 0.31\nmouse->rat: 0.85\n\nFormula: cos(θ) = (a·b)/(||a||·||b||)',
                    width=4.0, color='similarity', alpha=0.2, edgecolor=colors['similarity'])
    
    # Step 7: Phase Detection (right)
    ax.text(7.5, step_title_y, 'Step 7: Phase Detection', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=PUB_STYLE['step_title_color'])
    add_content_box(ax, 7.5, box_y,
                    'Threshold: τ = 0.6\n\nif similarity > τ:\n    -> Exploitation phase\nelse:\n    -> Exploration phase',
                    width=4.0, color='phase', alpha=0.2, edgecolor=colors['phase'])
    
    # Bottom: Step 8
    bottom_y = 0.40
    step_title_y2 = bottom_y
    box_y2 = step_title_y2 - PUB_STYLE['title_to_box_gap']
    
    ax.text(PUB_STYLE['figure_width']/2, step_title_y2, 'Step 8: Detected Phases', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=PUB_STYLE['step_title_color'])
    add_content_box(ax, PUB_STYLE['figure_width']/2, box_y2,
                    'Exploit: [dog, cat]\nExplore: [cat->lion]\nExploit: [lion, tiger]\nExplore: [tiger->mouse]\nExploit: [mouse, rat]',
                    width=7.0, color='phase', alpha=0.2, edgecolor=colors['phase'])
    
    output_dir = Path('output/figures/panels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'panel2_similarity_phases.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel2_similarity_phases.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel2_similarity_phases.svg', format='svg', bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"✅ Saved Panel 2: {output_dir / 'panel2_similarity_phases.png'}")

def create_panel_3_intra_phase():
    """Panel 3: Steps 9-10 - Intra-Phase Similarity Computation"""
    fig = plt.figure(figsize=(PUB_STYLE['figure_width'], PUB_STYLE['figure_height']))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, PUB_STYLE['figure_width'])
    ax.set_ylim(PUB_STYLE['y_lim_bottom'], PUB_STYLE['y_lim_top'])
    ax.axis('off')
    
    # Panel title
    ax.text(PUB_STYLE['figure_width']/2, PUB_STYLE['panel_title_y'], 
            'Panel 3: Intra-Phase Similarity Computation', 
            ha='center', va='top', fontsize=PUB_STYLE['panel_title_fontsize'], 
            fontweight='bold', color=PUB_STYLE['panel_title_color'])
    
    # Top row: Step 9 (Exploitation vs Exploration)
    top_y = 0.80
    step_title_y = top_y
    box_y = step_title_y - PUB_STYLE['title_to_box_gap']
    
    # Exploitation phase (left)
    ax.text(2.5, step_title_y, 'Step 9: Exploitation Phase [dog, cat]', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=colors['exploitation'])
    matrix_text = (
        "Similarity Matrix:\n"
        "        dog    cat\n"
        "dog     1.00   0.82\n"
        "cat     0.82   1.00\n\n"
        "Upper triangle: 0.82\n"
        "μ_exploit_intra = 0.82"
    )
    add_content_box(ax, 2.5, box_y, matrix_text,
                    width=4.0, color='exploitation', alpha=0.2, edgecolor=colors['exploitation'])
    
    # Exploration phase (right)
    ax.text(7.5, step_title_y, 'Step 9: Exploration Phase [cat->lion]', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=colors['exploration'])
    explore_text = (
        "Transition similarity:\n"
        "cat->lion = 0.38\n\n"
        "μ_explore_intra = 0.38\n\n"
        "For single transitions:\n"
        "intra = transition similarity"
    )
    add_content_box(ax, 7.5, box_y, explore_text,
                    width=4.0, color='exploration', alpha=0.2, edgecolor=colors['exploration'])
    
    # Bottom: Step 10
    bottom_y = 0.38
    step_title_y2 = bottom_y
    box_y2 = step_title_y2 - PUB_STYLE['title_to_box_gap']
    
    ax.text(PUB_STYLE['figure_width']/2, step_title_y2, 'Step 10: Phase Centroid Calculation', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=colors['intra'])
    centroid_text = (
        "For each phase:\n"
        "Centroid = mean(all word vectors)\n\n"
        "Phase 1 (dog, cat):\n"
        "c₁ = mean([dog_vec, cat_vec])\n"
        "   = [0.135, -0.435, 0.795, ...]\n\n"
        "Phase 2 (lion, tiger):\n"
        "c₂ = mean([lion_vec, tiger_vec])\n"
        "   = [0.180, -0.380, 0.820, ...]"
    )
    add_content_box(ax, PUB_STYLE['figure_width']/2, box_y2, centroid_text,
                    width=7.0, color='intra', alpha=0.2, edgecolor=colors['intra'])
    
    output_dir = Path('output/figures/panels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'panel3_intra_phase.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel3_intra_phase.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel3_intra_phase.svg', format='svg', bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"✅ Saved Panel 3: {output_dir / 'panel3_intra_phase.png'}")

def create_panel_4_inter_phase():
    """Panel 4: Two side-by-side panels - Centroid Normalization and Inter-Phase Similarity"""
    fig = plt.figure(figsize=(PUB_STYLE['figure_width'], PUB_STYLE['figure_height']))
    # Minimal padding - tight layout
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, PUB_STYLE['figure_width'])
    ax.set_ylim(PUB_STYLE['y_lim_bottom'], PUB_STYLE['y_lim_top'])
    ax.set_facecolor('white')
    ax.axis('off')
    
    # Panel title
    ax.text(PUB_STYLE['figure_width']/2, PUB_STYLE['panel_title_y'], 
            'PANEL 4. INTER PHASE SIMILARITY COMPUTATION', 
            ha='center', va='top', fontsize=PUB_STYLE['panel_title_fontsize'], 
            fontweight='bold', color='#000000')
    
    # Shared alignment grid - all elements use these y-positions for perfect alignment
    panel_title_y = 0.82  # Shared baseline for both panel titles A and B
    row1_y = 0.72  # First content row (both panels)
    row2_y = 0.62  # Second content row (Panel A only)
    row3_y = 0.52  # Third content row (Panel A only)
    row4_y = 0.42  # Matrix label row (Panel B)
    row5_y = 0.32  # Matrix content row (Panel B)
    row6_y = 0.22  # Mean similarity row (Panel B)
    
    # Panel positions
    panel_a_x = 2.5
    panel_b_x = 7.5
    
    # Text and equation positions - consistent spacing
    label_x_a = panel_a_x - 0.7
    eq_x_a = panel_a_x + 0.4  # Spacing between label and equation
    label_x_b = panel_b_x - 0.7
    eq_x_b = panel_b_x + 0.4
    
    font_size = 9
    title_fontsize = 11
    
    # Panel titles - aligned on shared baseline
    ax.text(panel_a_x, panel_title_y, 'A. Centroid Normalization', 
            ha='center', va='top', fontsize=title_fontsize, 
            fontweight='bold', color='#000000')
    
    ax.text(panel_b_x, panel_title_y, 'B. Inter Phase Similarity', 
            ha='center', va='top', fontsize=title_fontsize, 
            fontweight='bold', color='#000000')
    
    # Panel A - Row 1: Normalize to unit length
    ax.text(label_x_a, row1_y, 'Normalize to unit length', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    ax.text(eq_x_a, row1_y, r'$\hat{c} = \frac{c}{||c||}$', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    
    # Panel A - Row 2: Apply to all centroids
    ax.text(label_x_a, row2_y, 'Apply to all centroids', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    eq2_text = r'$\hat{c}_1 = \frac{c_1}{||c_1||}$' + '\n' + \
               r'$\hat{c}_2 = \frac{c_2}{||c_2||}$' + '\n' + \
               r'$\hat{c}_3 = \frac{c_3}{||c_3||}$'
    ax.text(eq_x_a, row2_y, eq2_text, 
            ha='left', va='center', fontsize=font_size, color='#000000')
    
    # Panel A - Row 3: Cosine similarity formula (with proper spacing)
    ax.text(label_x_a, row3_y, 'Cosine similarity formula', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    ax.text(eq_x_a, row3_y, r'$\cos(\theta) = \hat{c}_i \cdot \hat{c}_j$', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    
    # Panel B - Row 1: Similarity between phase pairs
    ax.text(label_x_b, row1_y, 'Similarity between phase pairs', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    ax.text(eq_x_b, row1_y, r'$S_{\text{inter}}(i,j) = \hat{c}_i \cdot \hat{c}_j$', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    
    # Panel B - Similarity Matrix (no box, just label and matrix)
    ax.text(label_x_b, row4_y, 'Similarity Matrix', 
            ha='left', va='center', fontsize=font_size, fontweight='bold', 
            color='#000000')
    
    # Matrix content - aligned with left boundary (label_x_b)
    matrix_text = (
        "    P1   P2   P3\n"
        "P1 1.00  .65  .58\n"
        "P2  .65 1.00  .72\n"
        "P3  .58  .72 1.00"
    )
    ax.text(label_x_b, row5_y, matrix_text, 
            ha='left', va='top', fontsize=9, color='#000000', 
            family='monospace')
    
    # Panel B - Mean similarity directly under matrix, left-aligned with matrix
    ax.text(label_x_b, row6_y, 'Mean similarity', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    ax.text(eq_x_b, row6_y, r'$\mu_{\text{inter}} = 0.65$', 
            ha='left', va='center', fontsize=font_size, color='#000000')
    
    output_dir = Path('output/figures/panels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tight bounding box with zero padding - removes all text box padding
    fig.savefig(output_dir / 'panel4_inter_phase.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    fig.savefig(output_dir / 'panel4_inter_phase.pdf', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    fig.savefig(output_dir / 'panel4_inter_phase.svg', format='svg', bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved Panel 4: {output_dir / 'panel4_inter_phase.png'}")

def create_panel_5_coherence_metrics():
    """Panel 5: Steps 13-18 - Coherence Ratios & Final Metrics"""
    fig = plt.figure(figsize=(PUB_STYLE['figure_width'], PUB_STYLE['figure_height']))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, PUB_STYLE['figure_width'])
    ax.set_ylim(PUB_STYLE['y_lim_bottom'], PUB_STYLE['y_lim_top'])
    ax.axis('off')
    
    # Panel title
    ax.text(PUB_STYLE['figure_width']/2, PUB_STYLE['panel_title_y'], 
            'Panel 5: Coherence Ratios & Final Metrics', 
            ha='center', va='top', fontsize=PUB_STYLE['panel_title_fontsize'], 
            fontweight='bold', color=PUB_STYLE['panel_title_color'])
    
    # Top row: Steps 13-14
    top_y = 0.78
    step_title_y = top_y
    box_y = step_title_y - PUB_STYLE['title_to_box_gap']
    
    # Step 13: Coherence Ratios (left)
    ax.text(2.5, step_title_y, 'Step 13: Coherence Ratios', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=colors['coherence'])
    ratio_text = (
        "Exploitation Coherence:\n"
        "= μ_exploit_intra / μ_inter\n"
        "= 0.82 / 0.65 = 1.26\n\n"
        "Exploration Coherence:\n"
        "= μ_explore_intra / μ_inter\n"
        "= 0.38 / 0.65 = 0.58"
    )
    add_content_box(ax, 2.5, box_y, ratio_text,
                    width=4.0, color='coherence', alpha=0.2, edgecolor=colors['coherence'])
    
    # Step 14: Phase Separation Index (right)
    ax.text(7.5, step_title_y, 'Step 14: Phase Separation Index (PSI)', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=colors['coherence'])
    psi_text = (
        "PSI = (μ_exploit_intra + μ_explore_intra)/2\n"
        "      - μ_inter\n\n"
        "= (0.82 + 0.38) / 2 - 0.65\n"
        "= 0.60 - 0.65 = -0.05\n\n"
        "(Negative = phases less distinct)"
    )
    add_content_box(ax, 7.5, box_y, psi_text,
                    width=4.0, color='coherence', alpha=0.2, edgecolor=colors['coherence'])
    
    # Middle row: Steps 15-17
    middle_y = 0.45
    step_title_y2 = middle_y
    box_y2 = step_title_y2 - PUB_STYLE['title_to_box_gap']
    
    # Step 15: Basic Metrics (left)
    ax.text(2.5, step_title_y2, 'Steps 15-17: Additional Metrics', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=colors['metrics'])
    basic_text = (
        "Exploitation %: 60%\n"
        "Exploration %: 40%\n"
        "Number of switches: 4\n"
        "Avg exploit size: 2.0\n"
        "Avg explore size: 1.0"
    )
    add_content_box(ax, 2.5, box_y2, basic_text,
                    width=4.0, color='metrics', alpha=0.2, edgecolor=colors['metrics'])
    
    # Steps 16-17: Clustering & Frequency (right)
    add_text = (
        "Clustering:\n"
        "Coefficient: 0.75\n"
        "Num clusters: 3\n\n"
        "Frequency:\n"
        "Mean rank: 1250\n"
        "Median rank: 980"
    )
    add_content_box(ax, 7.5, box_y2, add_text,
                    width=4.0, color='metrics', alpha=0.2, edgecolor=colors['metrics'])
    
    # Bottom: Step 18
    bottom_y = 0.15
    step_title_y3 = bottom_y
    box_y3 = step_title_y3 - PUB_STYLE['title_to_box_gap']
    
    ax.text(PUB_STYLE['figure_width']/2, step_title_y3, 'Step 18: Final Results', 
            fontsize=PUB_STYLE['step_title_fontsize'], fontweight='bold', 
            ha='center', color=colors['output'])
    final_text = "Comprehensive metrics exported for statistical analysis"
    add_content_box(ax, PUB_STYLE['figure_width']/2, box_y3, final_text,
                    width=7.0, color='output', alpha=0.2, edgecolor=colors['output'])
    
    output_dir = Path('output/figures/panels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / 'panel5_coherence_metrics.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel5_coherence_metrics.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(output_dir / 'panel5_coherence_metrics.svg', format='svg', bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"✅ Saved Panel 5: {output_dir / 'panel5_coherence_metrics.png'}")

def create_all_panels():
    """Create all panel figures"""
    print("Creating publication-ready panel figures...\n")
    create_panel_1_data_preprocessing()
    create_panel_2_similarity_phases()
    create_panel_3_intra_phase()
    create_panel_4_inter_phase()
    create_panel_5_coherence_metrics()
    print("\n✅ All panels created in output/figures/panels/")

if __name__ == '__main__':
    create_all_panels()
