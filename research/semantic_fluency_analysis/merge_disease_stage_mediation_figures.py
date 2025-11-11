#!/usr/bin/env python3
"""Merge disease stage-adjusted mediation figures into a combined figure"""

from PIL import Image
import os

def merge_disease_stage_mediation_figures():
    """Merge SVF and EE mediation figures with disease stage adjustment"""
    print("Merging disease stage-adjusted mediation figures...")
    
    # Load the individual figures
    svf_path = 'output/mediation_svf_disease_stage_nature.png'
    ee_path = 'output/mediation_ee_disease_stage_nature.png'
    
    if not os.path.exists(svf_path) or not os.path.exists(ee_path):
        print("Error: One or both mediation figures not found")
        return
    
    # Open images
    svf_img = Image.open(svf_path)
    ee_img = Image.open(ee_path)
    
    # Get dimensions
    width, height = svf_img.size
    
    # Create combined image (stacked vertically)
    combined_height = height * 2
    combined_img = Image.new('RGB', (width, combined_height), 'white')
    
    # Paste images
    combined_img.paste(svf_img, (0, 0))  # SVF on top
    combined_img.paste(ee_img, (0, height))  # EE on bottom
    
    # Add panel labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined_img)
    
    # Try to use a default font, fall back to basic if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add panel labels
    draw.text((10, 10), "A", fill='black', font=font)
    draw.text((10, height + 10), "B", fill='black', font=font)
    
    # Save combined figure
    output_path = 'output/mediation_disease_stage_combined.png'
    combined_img.save(output_path, 'PNG')
    
    # Also save as PDF
    pdf_path = 'output/mediation_disease_stage_combined.pdf'
    combined_img.save(pdf_path, 'PDF')
    
    print(f"Saved combined disease stage-adjusted mediation figure:")
    print(f" - {output_path}")
    print(f" - {pdf_path}")

if __name__ == '__main__':
    merge_disease_stage_mediation_figures()

