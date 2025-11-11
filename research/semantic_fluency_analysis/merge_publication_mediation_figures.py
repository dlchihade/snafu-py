#!/usr/bin/env python3
"""Merge publication-quality disease stage-adjusted mediation figures"""

from PIL import Image, ImageDraw, ImageFont
import os

def merge_publication_mediation_figures():
    """Merge SVF and EE mediation figures with publication-quality formatting"""
    print("Merging publication-quality disease stage-adjusted mediation figures...")
    
    # Load the individual figures
    svf_path = 'output/mediation_svf_disease_stage_publication.png'
    ee_path = 'output/mediation_ee_disease_stage_publication.png'
    
    if not os.path.exists(svf_path) or not os.path.exists(ee_path):
        print("Error: One or both mediation figures not found")
        return
    
    # Open images
    svf_img = Image.open(svf_path)
    ee_img = Image.open(ee_path)
    
    # Get dimensions
    width, height = svf_img.size
    
    # Create combined image (stacked vertically with proper spacing)
    spacing = 20  # pixels between figures
    combined_height = height * 2 + spacing
    combined_img = Image.new('RGB', (width, combined_height), 'white')
    
    # Paste images with spacing
    combined_img.paste(svf_img, (0, 0))  # SVF on top
    combined_img.paste(ee_img, (0, height + spacing))  # EE on bottom
    
    # Add professional panel labels
    draw = ImageDraw.Draw(combined_img)
    
    # Try to use Arial font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        font_bold = ImageFont.truetype("Arial.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            font_bold = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_bold = ImageFont.load_default()
    
    # Add panel labels with better positioning and styling
    panel_a_pos = (15, 15)
    panel_b_pos = (15, height + spacing + 15)
    
    # Draw panel labels with background
    for pos, label in [(panel_a_pos, "A"), (panel_b_pos, "B")]:
        # Create background rectangle for panel label
        text_bbox = draw.textbbox(pos, label, font=font)
        bg_padding = 5
        bg_rect = [
            text_bbox[0] - bg_padding,
            text_bbox[1] - bg_padding,
            text_bbox[2] + bg_padding,
            text_bbox[3] + bg_padding
        ]
        
        # Draw background
        draw.rectangle(bg_rect, fill='white', outline='#2C2C2C', width=1)
        
        # Draw text
        draw.text(pos, label, fill='#2C2C2C', font=font)
    
    # Save combined figure with high quality
    output_path = 'output/mediation_disease_stage_publication_combined.png'
    combined_img.save(output_path, 'PNG', dpi=(300, 300))
    
    # Also save as PDF
    pdf_path = 'output/mediation_disease_stage_publication_combined.pdf'
    combined_img.save(pdf_path, 'PDF', resolution=300.0)
    
    print(f"Saved publication-quality combined disease stage-adjusted mediation figure:")
    print(f" - {output_path}")
    print(f" - {pdf_path}")
    
    # Print figure specifications
    print(f"\nFigure specifications:")
    print(f" - Dimensions: {width} × {combined_height} pixels")
    print(f" - Resolution: 300 DPI")
    print(f" - Format: PNG and PDF")
    print(f" - Panel spacing: {spacing} pixels")
    print(f" - Font: Arial, 18pt")

if __name__ == '__main__':
    merge_publication_mediation_figures()

