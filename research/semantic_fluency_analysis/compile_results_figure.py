#!/usr/bin/env python3
"""
Compile a composite figure with key results:
- E-E index vs MoCA
- LC vs Alpha
- Behavior performance (E-E index distribution)
"""

from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont
import sys

BACKGROUND = (255, 255, 255)

def trim_image(path: Path, pad: int = 20) -> Image.Image:
    """Trim white margins from an image and add padding."""
    if not path.exists():
        print(f"⚠️ Missing figure: {path}")
        return None
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, BACKGROUND)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    left = max(bbox[0] - pad, 0)
    upper = max(bbox[1] - pad, 0)
    right = min(bbox[2] + pad, img.width)
    lower = min(bbox[3] + pad, img.height)
    return img.crop((left, upper, right, lower))

def create_composite_figure():
    """Create a composite figure with key results."""
    print("=" * 70)
    print("COMPILING COMPOSITE RESULTS FIGURE")
    print("=" * 70)
    
    output_dir = Path('research/semantic_fluency_analysis/output')
    
    # Define figures to include
    figures = [
        {
            'path': output_dir / 'NATURE_REAL_ee_vs_moca.png',
            'label': 'A',
            'title': 'E–E Index vs MoCA Score'
        },
        {
            'path': output_dir / 'figures/intermediate/lc_vs_alpha.png',
            'label': 'B',
            'title': 'LC Neuromelanin vs MEG Alpha Power'
        },
        {
            'path': output_dir / 'NATURE_REAL_behavior_performance.png',
            'label': 'C',
            'title': 'Behavior Performance Metrics'
        }
    ]
    
    # Load and trim images
    images = []
    labels = []
    titles = []
    
    for fig_info in figures:
        img = trim_image(fig_info['path'], pad=25)
        if img is not None:
            images.append(img)
            labels.append(fig_info['label'])
            titles.append(fig_info['title'])
            print(f"✅ Loaded: {fig_info['path'].name}")
        else:
            print(f"❌ Skipped: {fig_info['path'].name} (not found)")
    
    if not images:
        print("❌ No images found to compile!")
        return
    
    print(f"\n📊 Compiling {len(images)} panels...")
    
    # Determine layout: 2x2 if 4 panels, otherwise arrange vertically
    if len(images) == 3:
        # Arrange in a 2x2 grid (top row: 2 panels, bottom row: 1 centered)
        layout = 'grid_2x2'
    else:
        # Stack vertically
        layout = 'vertical'
    
    # Create composite
    gap = 40
    margin = 30
    
    if layout == 'grid_2x2':
        # Top row: first two images side by side
        top_width = images[0].width + images[1].width + gap
        top_height = max(images[0].height, images[1].height)
        
        # Bottom row: third image (centered)
        bottom_width = images[2].width
        bottom_height = images[2].height
        
        # Total dimensions
        total_width = max(top_width, bottom_width) + 2 * margin
        total_height = top_height + bottom_height + gap + 2 * margin
        
        composite = Image.new("RGB", (total_width, total_height), BACKGROUND)
        
        # Top row
        x = margin
        y = margin
        composite.paste(images[0], (x, y))
        x += images[0].width + gap
        composite.paste(images[1], (x, y))
        
        # Bottom row (centered)
        y = margin + top_height + gap
        x = margin + (top_width - bottom_width) // 2
        composite.paste(images[2], (x, y))
        
    else:
        # Stack vertically
        total_width = max(img.width for img in images) + 2 * margin
        total_height = sum(img.height for img in images) + gap * (len(images) - 1) + 2 * margin
        
        composite = Image.new("RGB", (total_width, total_height), BACKGROUND)
        
        y = margin
        for img in images:
            x = margin + (total_width - 2 * margin - img.width) // 2  # Center
            composite.paste(img, (x, y))
            y += img.height + gap
    
    # Add panel labels and titles
    draw = ImageDraw.Draw(composite)
    
    # Try to load Arial font
    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
        font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except:
        try:
            font_label = ImageFont.truetype("Arial Bold.ttf", 24)
            font_title = ImageFont.truetype("Arial.ttf", 18)
        except:
            font_label = ImageFont.load_default()
            font_title = ImageFont.load_default()
    
    # Add labels and titles
    label_positions = []
    if layout == 'grid_2x2':
        # Top left
        label_positions.append((margin + 10, margin + 10))
        # Top right
        label_positions.append((margin + images[0].width + gap + 10, margin + 10))
        # Bottom center
        label_positions.append((margin + (top_width - bottom_width) // 2 + 10, 
                               margin + top_height + gap + 10))
    else:
        y = margin + 10
        for _ in images:
            label_positions.append((margin + 10, y))
            y += images[0].height + gap
    
    for i, (label, title, pos) in enumerate(zip(labels, titles, label_positions)):
        # Draw label with background
        bbox = draw.textbbox(pos, label, font=font_label)
        label_bg = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5)
        draw.rectangle(label_bg, fill='white', outline='black', width=2)
        draw.text(pos, label, fill='black', font=font_label)
    
    # Save outputs
    out_base = output_dir / 'NATURE_REAL_compiled_results'
    png_path = out_base.with_suffix('.png')
    pdf_path = out_base.with_suffix('.pdf')
    
    composite.save(png_path, quality=95)
    # For PDF, we need to use a different approach
    try:
        composite.save(pdf_path, format='PDF', resolution=600.0)
    except:
        # Fallback: save as PNG and convert
        composite.save(pdf_path.with_suffix('.png'), quality=95)
        print(f"   Note: PDF saved as PNG (PIL PDF support may be limited)")
    
    print(f"\n✅ Saved composite figure:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    print(f"   Dimensions: {composite.width} x {composite.height} pixels")
    print(f"   Panels: {len(images)}")

if __name__ == '__main__':
    create_composite_figure()

