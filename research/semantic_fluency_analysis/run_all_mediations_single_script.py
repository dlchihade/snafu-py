#!/usr/bin/env python3
"""
One-stop mediation runner for real data:
 1) Loads metrics from output/NATURE_REAL_metrics.csv and merges SVF counts if available
 2) Runs age-adjusted mediation for EE metric and SVF count
 3) Saves Nature-style mediation figures for both outcomes
 4) Stacks them into a combined, trimmed PDF/PNG

Outputs:
 - output/mediation_exploit_age_nature.(png|pdf)
 - output/mediation_svf_age_nature.(png|pdf)  (if SVF present)
 - output/mediation_age_adjusted_combined.(png|pdf)
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

from mediation_figures_nature import mediation_age_adjusted, plot_mediation_nature


def _load_metrics() -> pd.DataFrame:
    base = Path(__file__).parent
    metrics_path = base / 'output' / 'NATURE_REAL_metrics.csv'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}. Run create_nature_quality_figures_real.py first.")
    df = pd.read_csv(metrics_path)

    # Merge SVF counts when present
    for candidate in [base / 'data' / 'fluency_data.csv',
                      base.parent / 'fluency_data' / 'SVF Data1.csv',
                      base.parent / 'fluency_data' / 'snafu_sample.csv']:
        try:
            if candidate.exists():
                flu = pd.read_csv(candidate)
                if 'ID' in flu.columns:
                    svf = flu.groupby('ID').size().reset_index(name='SVF_count')
                    df = df.merge(svf, on='ID', how='left')
                    break
        except Exception:
            pass
    return df


def _compose_vertical(top_path: Path, bottom_path: Path, out_stem: str) -> None:
    # Load and trim white margins with small padding
    def trim_with_pad(p: Path, pad: int = 12) -> Image.Image:
        img = Image.open(p).convert('RGB')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bbox = ImageChops.difference(img, bg).getbbox()
        if not bbox:
            return img
        left = max(bbox[0] - pad, 0)
        upper = max(bbox[1] - pad, 0)
        right = min(bbox[2] + pad, img.width)
        lower = min(bbox[3] + pad, img.height)
        return img.crop((left, upper, right, lower))

    top = trim_with_pad(top_path)
    bot = trim_with_pad(bottom_path)

    margin, gap = 28, 20
    width = max(top.width, bot.width) + 2 * margin
    height = top.height + bot.height + gap + 2 * margin
    canvas = Image.new('RGB', (width, height), (255, 255, 255))

    x_top = (width - top.width) // 2
    x_bot = (width - bot.width) // 2
    y_top = margin
    y_bot = margin + top.height + gap
    canvas.paste(top, (x_top, y_top))
    canvas.paste(bot, (x_bot, y_bot))

    # Panel labels
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((x_top + 2, y_top + 2), 'A', fill=(0, 0, 0), font=font)
    draw.text((x_bot + 2, y_bot + 2), 'B', fill=(0, 0, 0), font=font)

    out_dir = top_path.parent
    out_png = out_dir / f'{out_stem}.png'
    out_pdf = out_dir / f'{out_stem}.pdf'
    canvas.save(out_png)
    canvas.save(out_pdf)
    print(f"✅ Wrote {out_png}\n✅ Wrote {out_pdf}")


def main():
    base = Path(__file__).parent
    out_dir = base / 'output'
    out_dir.mkdir(exist_ok=True)

    df = _load_metrics()

    # EE metric (exploitation_coherence_ratio)
    res_ee = mediation_age_adjusted(df, 'exploitation_coherence_ratio')
    plot_mediation_nature(
        res_ee,
        'LC → α-power → EE metric (Age-adjusted)',
        out_dir / 'mediation_exploit_age_nature.png',
        outcome_type='coherence',
    )

    # SVF count when available
    svf_out: Optional[Path] = None
    if 'SVF_count' in df.columns and df['SVF_count'].notna().any():
        res_svf = mediation_age_adjusted(df, 'SVF_count')
        svf_out = out_dir / 'mediation_svf_age_nature.png'
        plot_mediation_nature(
            res_svf,
            'LC → α-power → SVF (Age-adjusted)',
            svf_out,
            outcome_type='svf',
        )

    # Compose combined (if both present)
    ee_path = out_dir / 'mediation_exploit_age_nature.png'
    if svf_out and ee_path.exists() and svf_out.exists():
        _compose_vertical(svf_out, ee_path, 'mediation_age_adjusted_combined')

    print('\nDone creating mediation results and combined figure.')


if __name__ == '__main__':
    main()



