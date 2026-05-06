#!/usr/bin/env python3

"""

Create a clean, publication-ready semantic fluency pipeline schematic

Suitable for Nature-style figures and Visio import

"""



import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pathlib import Path



# Set up publication-quality style

plt.rcParams.update({

    'font.family': 'Arial',

    'font.size': 11,

    'font.weight': 'normal',

    'figure.dpi': 300,

    'savefig.dpi': 300,

    'savefig.bbox': 'tight',

})



def create_clean_pipeline_schematic():

    """Create clean, minimal pipeline schematic with two rows"""

    

    # Create figure with appropriate aspect ratio

    fig = plt.figure(figsize=(16, 7))

    ax = fig.add_subplot(111)

    ax.set_xlim(0, 16)

    ax.set_ylim(0, 5)

    ax.axis('off')

    

    # Color scheme

    colors = {

        'preprocessing': '#1f77b4',      # Blue

        'similarity': '#ff7f0e',         # Orange

        'output': '#2ca02c',             # Green

        'arrow': '#333333',              # Dark gray for arrows

        'text': '#000000',               # Black text

    }

    

    # Box dimensions

    box_width = 1.6

    box_height = 0.9

    box_pad = 0.15  # Padding between boxes

    arrow_gap = 0.2  # Gap between box edge and arrow start

    

    # Row 1: Steps 1-5 (top row)

    row1_y = 3.5

    row1_steps = [

        ("1. Data\nCollection", colors['preprocessing']),

        ("2. Preprocessing", colors['preprocessing']),

        ("3. Word\nEmbeddings", colors['preprocessing']),

        ("4. Similarity\nComputation", colors['similarity']),

        ("5. Phase\nDetection", colors['similarity']),

    ]

    

    # Row 2: Steps 6-8 (bottom row)

    row2_y = 1.5

    row2_steps = [

        ("6. Intra-Phase\nMetrics", colors['output']),

        ("7. Inter-Phase\nMetrics", colors['output']),

        ("8. Coherence Ratios\n& Final Metrics", colors['output']),

    ]

    

    # Calculate spacing for row 1 (5 boxes)

    # Total width needed: 5 boxes + 4 gaps

    total_box_width_row1 = 5 * box_width

    total_gap_width_row1 = 4 * box_pad

    total_width_row1 = total_box_width_row1 + total_gap_width_row1

    

    # Center row 1

    row1_start_x = (16 - total_width_row1) / 2 + box_width / 2

    

    # Calculate spacing for row 2 (3 boxes, centered)

    total_box_width_row2 = 3 * box_width

    total_gap_width_row2 = 2 * box_pad

    total_width_row2 = total_box_width_row2 + total_gap_width_row2

    

    # Center row 2

    row2_start_x = (16 - total_width_row2) / 2 + box_width / 2

    

    # Draw Row 1 boxes and arrows

    row1_x_positions = []

    for i, (label, color) in enumerate(row1_steps):

        x = row1_start_x + i * (box_width + box_pad)

        row1_x_positions.append(x)

        

        # Draw box

        box = FancyBboxPatch(

            (x - box_width/2, row1_y - box_height/2),

            box_width, box_height,

            boxstyle="round,pad=0.05",

            facecolor=color,

            edgecolor='white',

            linewidth=1.5,

            alpha=0.8,

            zorder=2

        )

        ax.add_patch(box)

        

        # Add text (white for visibility on colored backgrounds)

        ax.text(x, row1_y, label, ha='center', va='center',

                fontsize=10, fontweight='bold', color='white',

                zorder=3)

        

        # Draw arrow to next box (except for last box)

        if i < len(row1_steps) - 1:

            next_x = row1_start_x + (i + 1) * (box_width + box_pad)

            arrow_start_x = x + box_width/2 + arrow_gap

            arrow_end_x = next_x - box_width/2 - arrow_gap

            

            arrow = FancyArrowPatch(

                (arrow_start_x, row1_y),

                (arrow_end_x, row1_y),

                arrowstyle='->',

                mutation_scale=20,

                color=colors['arrow'],

                linewidth=1.5,

                zorder=1

            )

            ax.add_patch(arrow)

    

    # Draw Row 2 boxes and arrows

    row2_x_positions = []

    for i, (label, color) in enumerate(row2_steps):

        x = row2_start_x + i * (box_width + box_pad)

        row2_x_positions.append(x)

        

        # Draw box

        box = FancyBboxPatch(

            (x - box_width/2, row2_y - box_height/2),

            box_width, box_height,

            boxstyle="round,pad=0.05",

            facecolor=color,

            edgecolor='white',

            linewidth=1.5,

            alpha=0.8,

            zorder=2

        )

        ax.add_patch(box)

        

        # Add text

        ax.text(x, row2_y, label, ha='center', va='center',

                fontsize=10, fontweight='bold', color='white',

                zorder=3)

        

        # Draw arrow to next box (except for last box)

        if i < len(row2_steps) - 1:

            next_x = row2_start_x + (i + 1) * (box_width + box_pad)

            arrow_start_x = x + box_width/2 + arrow_gap

            arrow_end_x = next_x - box_width/2 - arrow_gap

            

            arrow = FancyArrowPatch(

                (arrow_start_x, row2_y),

                (arrow_end_x, row2_y),

                arrowstyle='->',

                mutation_scale=20,

                color=colors['arrow'],

                linewidth=1.5,

                zorder=1

            )

            ax.add_patch(arrow)

    

    # Draw connecting arrow from Row 1 to Row 2

    # Connect from Step 5 (last in row 1) to Step 6 (first in row 2)

    step5_x = row1_x_positions[-1]

    step6_x = row2_x_positions[0]

    

    # Calculate safe connection points (from bottom of step 5, to top of step 6)

    start_y = row1_y - box_height/2 - 0.1

    end_y = row2_y + box_height/2 + 0.1

    

    # Create a curved arrow that avoids overlap

    arrow_down = FancyArrowPatch(

        (step5_x, start_y),

        (step6_x, end_y),

        arrowstyle='->',

        mutation_scale=20,

        color=colors['arrow'],

        linewidth=1.5,

        zorder=1,

        connectionstyle="arc3,rad=0.3"  # More curvature to avoid overlap

    )

    ax.add_patch(arrow_down)

    

    # Save figure

    output_dir = Path('output/figures')

    output_dir.mkdir(parents=True, exist_ok=True)

    

    # Save as PNG (high resolution)

    output_png = output_dir / 'clean_pipeline_schematic.png'

    fig.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.3,

                facecolor='white', edgecolor='none')

    

    # Save as SVG (for Visio import with editable text)

    output_svg = output_dir / 'clean_pipeline_schematic.svg'

    fig.savefig(output_svg, format='svg', bbox_inches='tight', pad_inches=0.3,

                facecolor='white', edgecolor='none')

    

    # Save as PDF (backup)

    output_pdf = output_dir / 'clean_pipeline_schematic.pdf'

    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', pad_inches=0.3,

                facecolor='white', edgecolor='none')

    

    print(f"✅ Saved clean pipeline schematic:")

    print(f"   PNG: {output_png}")

    print(f"   SVG: {output_svg} (editable text, suitable for Visio)")

    print(f"   PDF: {output_pdf}")

    

    plt.close(fig)





if __name__ == '__main__':

    create_clean_pipeline_schematic()

