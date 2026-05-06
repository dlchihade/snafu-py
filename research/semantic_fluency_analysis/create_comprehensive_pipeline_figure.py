#!/usr/bin/env python3
"""
Create a comprehensive, publication-ready semantic fluency pipeline schematic
Showing ALL methods from raw fluency data to final metrics
Suitable for Nature-style figures and Visio import
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Set up publication-quality style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'font.weight': 'normal',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def create_comprehensive_pipeline_figure():
    """Create comprehensive pipeline schematic with all processing steps"""
    
    # Create figure with appropriate aspect ratio for comprehensive pipeline
    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 5.5)
    ax.set_facecolor('white')
    ax.axis('off')
    
    # Clean neutral color scheme - no colors, just black text and gray borders
    text_color = '#000000'
    border_color = '#d0d0d0'
    arrow_color = '#666666'
    background_color = '#ffffff'
    
    # Box dimensions
    box_width = 1.4
    box_height = 0.75
    box_pad = 0.15  # Padding between boxes
    arrow_gap = 0.18  # Gap between box edge and arrow start
    
    # Reorganized into 4 balanced rows - all white boxes with black text
    # Row 1: Data Collection & Preprocessing (Steps 1-5)
    row1_y = 4.5
    row1_steps = [
        "1. Data\nCollection",
        "2. Data\nLoading",
        "3. Text\nPreprocessing",
        "4. Word\nEmbedding",
        "5. Vector\nValidation",
    ]
    
    # Row 2: Similarity & Phase Analysis (Steps 6-10)
    row2_y = 3.2
    row2_steps = [
        "6. Consecutive\nSimilarity",
        "7. Phase\nDetection",
        "8. Phase\nClassification",
        "9. Intra-Phase\nSimilarity",
        "10. Phase\nCentroids",
    ]
    
    # Row 3: Inter-Phase & Coherence (Steps 11-14)
    row3_y = 1.9
    row3_steps = [
        "11. Centroid\nNormalization",
        "12. Inter-Phase\nSimilarity",
        "13. Coherence\nRatios",
        "14. Phase\nSeparation Index",
    ]
    
    # Row 4: Final Metrics & Results (Steps 15-18)
    row4_y = 0.6
    row4_steps = [
        "15. Basic\nMetrics",
        "16. Clustering\nMetrics",
        "17. Frequency\nStatistics",
        "18. Final\nResults",
    ]
    
    # Helper function to calculate spacing and draw a row
    def draw_row(steps, y_pos, start_x_offset=0):
        """Draw a row of boxes with arrows - CLEAN VERSION"""
        total_box_width = len(steps) * box_width
        total_gap_width = (len(steps) - 1) * box_pad
        total_width = total_box_width + total_gap_width
        
        # Center the row
        start_x = (22 - total_width) / 2 + start_x_offset + box_width / 2
        
        x_positions = []
        for i, label in enumerate(steps):
            x = start_x + i * (box_width + box_pad)
            x_positions.append(x)
            
            # Draw box - white with thin gray border, no rounded corners
            box = FancyBboxPatch(
                (x - box_width/2, y_pos - box_height/2),
                box_width, box_height,
                boxstyle="square,pad=0",
                facecolor=background_color,
                edgecolor=border_color,
                linewidth=0.5,
                zorder=2
            )
            ax.add_patch(box)
            
            # Add text - black text
            ax.text(x, y_pos, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color,
                    zorder=3)
            
            # Draw arrow to next box (except for last box)
            if i < len(steps) - 1:
                next_x = start_x + (i + 1) * (box_width + box_pad)
                # Arrow starts from right edge of current box, ends at left edge of next box
                arrow_start_x = x + box_width/2 + arrow_gap
                arrow_end_x = next_x - box_width/2 - arrow_gap
                
                # Ensure arrows are perfectly horizontal at box center
                arrow = FancyArrowPatch(
                    (arrow_start_x, y_pos),
                    (arrow_end_x, y_pos),
                    arrowstyle='->',
                    mutation_scale=18,
                    color=arrow_color,
                    linewidth=1.0,
                    zorder=1
                )
                ax.add_patch(arrow)
        
        return x_positions
    
    # Draw all rows
    row1_positions = draw_row(row1_steps, row1_y)
    row2_positions = draw_row(row2_steps, row2_y)
    row3_positions = draw_row(row3_steps, row3_y)
    row4_positions = draw_row(row4_steps, row4_y)
    
    # Draw connecting arrows between rows
    # Ensure arrows connect from center of last box to center of first box
    # Row 1 to Row 2: Step 5 to Step 6
    start_x1 = row1_positions[-1]  # Center of last box in row 1
    start_y1 = row1_y - box_height/2 - 0.08  # Bottom of box with small gap
    end_x1 = row2_positions[0]  # Center of first box in row 2
    end_y1 = row2_y + box_height/2 + 0.08  # Top of box with small gap
    
    arrow1 = FancyArrowPatch(
        (start_x1, start_y1),
        (end_x1, end_y1),
        arrowstyle='->',
        mutation_scale=20,
        color=arrow_color,
        linewidth=1.0,
        zorder=1,
        connectionstyle="arc3,rad=0.3"
    )
    ax.add_patch(arrow1)
    
    # Row 2 to Row 3: Step 10 to Step 11
    start_x2 = row2_positions[-1]  # Center of last box in row 2
    start_y2 = row2_y - box_height/2 - 0.08  # Bottom of box with small gap
    end_x2 = row3_positions[0]  # Center of first box in row 3
    end_y2 = row3_y + box_height/2 + 0.08  # Top of box with small gap
    
    arrow2 = FancyArrowPatch(
        (start_x2, start_y2),
        (end_x2, end_y2),
        arrowstyle='->',
        mutation_scale=20,
        color=arrow_color,
        linewidth=1.0,
        zorder=1,
        connectionstyle="arc3,rad=0.3"
    )
    ax.add_patch(arrow2)
    
    # Row 3 to Row 4: Step 14 to Step 15
    start_x3 = row3_positions[-1]  # Center of last box in row 3
    start_y3 = row3_y - box_height/2 - 0.08  # Bottom of box with small gap
    end_x3 = row4_positions[0]  # Center of first box in row 4
    end_y3 = row4_y + box_height/2 + 0.08  # Top of box with small gap
    
    arrow3 = FancyArrowPatch(
        (start_x3, start_y3),
        (end_x3, end_y3),
        arrowstyle='->',
        mutation_scale=20,
        color=arrow_color,
        linewidth=1.0,
        zorder=1,
        connectionstyle="arc3,rad=0.3"
    )
    ax.add_patch(arrow3)
    
    # Save figure
    output_dir = Path('output/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG (high resolution) - minimal padding
    output_png = output_dir / 'comprehensive_pipeline_schematic.png'
    fig.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    
    # Save as SVG (for Visio import with editable text)
    output_svg = output_dir / 'comprehensive_pipeline_schematic.svg'
    fig.savefig(output_svg, format='svg', bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    
    # Save as PDF (backup)
    output_pdf = output_dir / 'comprehensive_pipeline_schematic.pdf'
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    
    print(f"✅ Saved comprehensive pipeline schematic:")
    print(f"   PNG: {output_png}")
    print(f"   SVG: {output_svg} (editable text, suitable for Visio)")
    print(f"   PDF: {output_pdf}")
    print(f"\n   Pipeline includes 18 steps in 4 balanced rows:")
    print(f"   - Row 1: Data Collection & Preprocessing (Steps 1-5)")
    print(f"   - Row 2: Similarity & Phase Analysis (Steps 6-10)")
    print(f"   - Row 3: Inter-Phase & Coherence (Steps 11-14)")
    print(f"   - Row 4: Final Metrics & Results (Steps 15-18)")
    
    plt.close(fig)


if __name__ == '__main__':
    create_comprehensive_pipeline_figure()

