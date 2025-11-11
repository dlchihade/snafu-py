#!/usr/bin/env python3
"""
Compose the four Nature-quality figures into a single-page 2x2 grid (PNG + PDF),
with trimming and consistent margins/gaps.

Inputs (must exist):
 - output/NATURE_REAL_figure1_exploration_exploitation.png
 - output/NATURE_REAL_figure2_phase_coherence.png
 - output/NATURE_REAL_figure3_meg_correlations.png
 - output/NATURE_REAL_figure4_comprehensive.png

Outputs:
 - output/NATURE_REAL_all_figures_2x2.png
 - output/NATURE_REAL_all_figures_2x2.pdf
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageChops


def trim_image_with_padding(img: Image.Image, pad: int = 12) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if not bbox:
        return rgb
    left = max(bbox[0] - pad, 0)
    upper = max(bbox[1] - pad, 0)
    right = min(bbox[2] + pad, rgb.width)
    lower = min(bbox[3] + pad, rgb.height)
    return rgb.crop((left, upper, right, lower))


def load_and_trim(paths: List[Path]) -> List[Image.Image]:
    out = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing figure: {p}")
        out.append(trim_image_with_padding(Image.open(p), pad=12))
    return out


def create_uniform_panels(images: List[Image.Image]) -> Tuple[List[Image.Image], int, int]:
    # Use the maximum width/height across images as panel size; pad smaller ones (no upscaling).
    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)
    panels: List[Image.Image] = []
    for im in images:
        panel = Image.new("RGB", (max_w, max_h), (255, 255, 255))
        x = (max_w - im.width) // 2
        y = (max_h - im.height) // 2
        panel.paste(im, (x, y))
        panels.append(panel)
    return panels, max_w, max_h


def compose_grid_2x2(panels: List[Image.Image], panel_w: int, panel_h: int,
                      margin_px: int = 28, gap_px: int = 20) -> Image.Image:
    cols, rows = 2, 2
    width = margin_px * 2 + cols * panel_w + (cols - 1) * gap_px
    height = margin_px * 2 + rows * panel_h + (rows - 1) * gap_px
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    positions = [
        (margin_px, margin_px),
        (margin_px + panel_w + gap_px, margin_px),
        (margin_px, margin_px + panel_h + gap_px),
        (margin_px + panel_w + gap_px, margin_px + panel_h + gap_px),
    ]

    for im, (x, y) in zip(panels, positions):
        canvas.paste(im, (x, y))
    return canvas


def main():
    root = Path(__file__).parent
    out_dir = root / "output"
    figure_paths = [
        out_dir / "NATURE_REAL_figure1_exploration_exploitation.png",
        out_dir / "NATURE_REAL_figure2_phase_coherence.png",
        out_dir / "NATURE_REAL_figure3_meg_correlations.png",
        out_dir / "NATURE_REAL_figure4_comprehensive.png",
    ]

    images = load_and_trim(figure_paths)
    panels, pw, ph = create_uniform_panels(images)
    grid = compose_grid_2x2(panels, pw, ph, margin_px=28, gap_px=20)

    out_png = out_dir / "NATURE_REAL_all_figures_2x2.png"
    out_pdf = out_dir / "NATURE_REAL_all_figures_2x2.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)
    grid.convert("RGB").save(out_pdf)
    print(f"✅ Wrote {out_png}")
    print(f"✅ Wrote {out_pdf}")


if __name__ == "__main__":
    main()



