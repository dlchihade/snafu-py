#!/usr/bin/env python3
"""
Compose a publication-ready panel by stacking existing figures into a single PNG/PDF.

Usage example:
  python compose_publication_panels.py \
      --panels output/mediation_ee_disease_stage_working.png \
               output/mediation_svf_disease_stage_working.png \
               output/exploit_explore_bar.png \
      --out output/combined_mediation_exploit_panels \
      --layout vertical \
      --gap 36 \
      --pad 18

If --panels is omitted, the default list matches the example above.
"""

from pathlib import Path
from typing import List
import argparse

from PIL import Image, ImageChops


BACKGROUND = (255, 255, 255)


def trim_image(path: Path, pad: int) -> Image.Image:
    """Trim white margins from an image and add padding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing figure: {path}")
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


def stack_vertically(images: List[Image.Image], gap: int) -> Image.Image:
    width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, total_height), BACKGROUND)
    y = 0
    for img in images:
        x = (width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + gap
    return canvas


def stack_horizontally(images: List[Image.Image], gap: int) -> Image.Image:
    height = max(img.height for img in images)
    total_width = sum(img.width for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (total_width, height), BACKGROUND)
    x = 0
    for img in images:
        y = (height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + gap
    return canvas


def stack_grid(images: List[Image.Image], cols: int, gap: int) -> Image.Image:
    if cols <= 0:
        raise ValueError("grid columns must be greater than zero")
    rows = (len(images) + cols - 1) // cols
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    canvas_w = cols * max_w + gap * (cols - 1)
    canvas_h = rows * max_h + gap * (rows - 1)
    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * (max_w + gap) + (max_w - img.width) // 2
        y = r * (max_h + gap) + (max_h - img.height) // 2
        canvas.paste(img, (x, y))
    return canvas


def save_outputs(image: Image.Image, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    image.save(png_path, dpi=(300, 300))
    image.save(pdf_path, dpi=(300, 300))
    print(f"✅ Saved composite figure:\n - {png_path}\n - {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose multiple figure panels.")
    root = Path(__file__).parent
    default_panels = [
        root / "output" / "mediation_ee_disease_stage_working.png",
        root / "output" / "mediation_svf_disease_stage_working.png",
        root / "output" / "exploit_explore_bar.png",
    ]
    parser.add_argument(
        "--panels",
        nargs="+",
        type=Path,
        default=default_panels,
        help="List of figure paths to include (default: mediation+bar panels).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "output" / "combined_mediation_exploit_panels",
        help="Output base path (PNG/PDF will be written).",
    )
    parser.add_argument(
        "--layout",
        choices=["vertical", "horizontal", "grid"],
        default="vertical",
        help="Layout for stacking the panels.",
    )
    parser.add_argument("--gap", type=int, default=32, help="Gap between panels in pixels.")
    parser.add_argument("--pad", type=int, default=16, help="Padding for trimming each panel.")
    parser.add_argument("--grid-cols", type=int, default=2, help="Columns when layout=grid.")
    return parser.parse_args()


def main():
    args = parse_args()
    panels = [trim_image(p, pad=args.pad) for p in args.panels]

    if args.layout == "vertical":
        composite = stack_vertically(panels, gap=args.gap)
    elif args.layout == "horizontal":
        composite = stack_horizontally(panels, gap=args.gap)
    else:
        composite = stack_grid(panels, cols=args.grid_cols, gap=args.gap)

    save_outputs(composite, args.out)


if __name__ == "__main__":
    main()


