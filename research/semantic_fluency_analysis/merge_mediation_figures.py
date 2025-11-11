#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def stack_images_vertically(top_path: Path, bottom_path: Path, out_png: Path, out_pdf: Path,
                            gap_px: int = 20, margin_px: int = 28):
    img_top = Image.open(top_path).convert('RGB')
    img_bottom = Image.open(bottom_path).convert('RGB')

    # Match widths by scaling to the larger width for consistency
    top_w, top_h = img_top.size
    bot_w, bot_h = img_bottom.size
    target_w = max(top_w, bot_w)
    if top_w != target_w:
        scale = target_w / top_w
        img_top = img_top.resize((target_w, int(top_h * scale)), Image.LANCZOS)
        top_w, top_h = img_top.size
    if bot_w != target_w:
        scale = target_w / bot_w
        img_bottom = img_bottom.resize((target_w, int(bot_h * scale)), Image.LANCZOS)
        bot_w, bot_h = img_bottom.size

    # Create canvas with margins for breathing room
    canvas_w = target_w + margin_px * 2
    canvas_h = top_h + bot_h + gap_px + margin_px * 2
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    # Paste images centered with margins
    y_top = margin_px
    x_left = margin_px
    canvas.paste(img_top, (x_left, y_top))
    y_bottom = y_top + top_h + gap_px
    canvas.paste(img_bottom, (x_left, y_bottom))

    # Subtle divider line for visual separation
    draw = ImageDraw.Draw(canvas)
    line_y = y_top + top_h + gap_px // 2
    draw.line([(x_left, line_y), (x_left + target_w, line_y)], fill=(220, 220, 220), width=1)

    # Small panel labels for readability
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((x_left + 2, y_top + 2), 'A', fill=(0, 0, 0), font=font)
    draw.text((x_left + 2, y_bottom + 2), 'B', fill=(0, 0, 0), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, format='PNG', optimize=True)
    canvas.save(out_pdf, format='PDF', resolution=300)


def main():
    base = Path(__file__).resolve().parent
    out_dir = base / 'output'
    top_png = out_dir / 'mediation_svf_age_nature.png'
    bottom_png = out_dir / 'mediation_exploit_age_nature.png'
    if not top_png.exists() or not bottom_png.exists():
        print('Missing input PNGs. Expected:', top_png, bottom_png)
        sys.exit(1)
    out_png = out_dir / 'mediation_combined.png'
    out_pdf = out_dir / 'mediation_combined.pdf'
    stack_images_vertically(top_png, bottom_png, out_png, out_pdf)
    print('Saved combined mediation figure:')
    print(' -', out_png)
    print(' -', out_pdf)


if __name__ == '__main__':
    main()


