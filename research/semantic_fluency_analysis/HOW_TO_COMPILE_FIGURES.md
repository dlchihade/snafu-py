# How to Compile All Figures into PowerPoint

This document explains how the figure compilation process works.

## Overview

The compilation script (`compile_all_figures_complete.py`) does the following:

1. **Finds all figure files** in multiple formats (.png, .pdf, .svg)
2. **Creates a PowerPoint presentation** with one slide per figure format
3. **Displays PNG images** directly in slides
4. **Shows file paths** for PDF and SVG files (since PowerPoint can't display them directly)

## Step-by-Step Process

### 1. Define Base Figure Names

The script maintains a list of base figure names (without file extensions):

```python
base_figures = [
    'NATURE_REAL_figure0_alpha_violin_swarm',
    'NATURE_REAL_figure1_exploration_exploitation',
    'mediation_svf_age_nature',
    # ... etc
]
```

### 2. Find All Formats for Each Figure

For each base name, the script checks for three formats:

```python
for ext in ['.png', '.pdf', '.svg']:
    file_path = base_path / f"{base_name}{ext}"
    if file_path.exists():
        formats_found.append((ext, file_path))
```

This creates a list like:
- `NATURE_REAL_figure4_comprehensive.png` ✅
- `NATURE_REAL_figure4_comprehensive.pdf` ✅
- `NATURE_REAL_figure4_comprehensive.svg` ✅

### 3. Create PowerPoint Slides

For each figure format found:

**PNG files:**
- Load the image using PIL (Python Imaging Library)
- Calculate optimal size to fit the slide
- Add the image directly to the slide
- Display the image

**PDF/SVG files:**
- PowerPoint cannot directly display these formats
- Instead, add a text box explaining the format
- Include the full file path so users can open it externally

### 4. Add Metadata to Each Slide

Each slide includes:
- **Title**: Figure name + format (e.g., "Figure 4 Comprehensive (PNG)")
- **Relative path**: `output/NATURE_REAL_figure4_comprehensive.png`
- **Absolute path**: Full system path for easy access

## Running the Script

```bash
cd research/semantic_fluency_analysis
python3 compile_all_figures_complete.py
```

## Output

The script creates:
- **File**: `output/all_figures_all_formats_complete.pptx`
- **Size**: ~7-8 MB (depends on number of PNG images)
- **Slides**: 1 title slide + 1 slide per figure format found

## Dependencies

Required Python packages:
- `python-pptx`: For creating PowerPoint files
- `PIL` (Pillow): For reading PNG images
- `pathlib`: For file path handling (built-in)

Install with:
```bash
pip install python-pptx pillow
```

## Limitations

1. **PDF files**: Cannot be displayed directly in PowerPoint
   - Solution: File paths are shown, users can open PDFs externally
   - Alternative: Convert PDFs to PNG first (requires `pdf2image`)

2. **SVG files**: Cannot be displayed directly in PowerPoint
   - Solution: File paths are shown, users can open SVGs externally
   - Alternative: Convert SVGs to PNG first (requires `cairosvg`)

3. **Large images**: Very large PNG files may cause warnings
   - The script handles this gracefully
   - Images are automatically resized to fit slides

## Customization

To add more figures, edit the `base_figures` list in `compile_all_figures_complete.py`:

```python
base_figures = [
    # Add your new figure base names here
    'your_new_figure_name',
]
```

The script will automatically find all formats (.png, .pdf, .svg) for each base name.

## Alternative: Auto-Discover All Figures

If you want to automatically find ALL PNG/PDF/SVG files in the output directory:

```python
def find_all_figures_automatically():
    """Find all figure files automatically"""
    output_dir = Path('output')
    all_files = []
    
    # Find all PNG, PDF, SVG files
    for ext in ['.png', '.pdf', '.svg']:
        for file_path in output_dir.rglob(f'*{ext}'):
            # Skip archive directories
            if 'archive' not in str(file_path):
                all_files.append(file_path)
    
    # Group by base name
    figures_dict = {}
    for file_path in all_files:
        base_name = file_path.stem  # Name without extension
        if base_name not in figures_dict:
            figures_dict[base_name] = []
        figures_dict[base_name].append((file_path.suffix, file_path))
    
    return figures_dict
```

This approach automatically discovers all figures without needing a predefined list.

## File Structure

```
research/semantic_fluency_analysis/
├── compile_all_figures_complete.py  # Main compilation script
├── output/
│   ├── all_figures_all_formats_complete.pptx  # Generated PowerPoint
│   ├── NATURE_REAL_figure4_comprehensive.png
│   ├── NATURE_REAL_figure4_comprehensive.pdf
│   ├── NATURE_REAL_figure4_comprehensive.svg
│   └── ... (all other figures)
└── HOW_TO_COMPILE_FIGURES.md  # This file
```

## Troubleshooting

**Problem**: "ModuleNotFoundError: No module named 'pptx'"
- **Solution**: `pip install python-pptx`

**Problem**: "Could not add PNG"
- **Solution**: Check if the PNG file is corrupted or too large
- **Workaround**: The script will still create a slide with the file path

**Problem**: PowerPoint file is too large
- **Solution**: Only include PNG files (modify script to skip PDF/SVG)
- **Alternative**: Compress images before adding to PowerPoint

## Summary

The compilation process:
1. ✅ Finds all figure files in multiple formats
2. ✅ Creates organized PowerPoint slides
3. ✅ Displays PNG images directly
4. ✅ Provides file paths for PDF/SVG
5. ✅ Includes metadata (titles, paths) on each slide

This makes it easy to:
- Review all figures in one place
- Access files in their original formats
- Share figures with collaborators
- Document which formats are available for each figure

