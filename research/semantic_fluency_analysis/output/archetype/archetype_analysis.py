#!/usr/bin/env python3
"""
Archetype analysis script
- Consistent plotting style (colorblind-friendly, no yellow)
- Reproducible output paths
- Example: load metrics and produce a horizontal bar chart

Usage:
  python archetype_analysis.py --metrics ../NATURE_REAL_metrics.csv --out-tag demo
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Style ----------

def setup_style() -> None:
	plt.rcParams.update({
		"font.family": "Arial",
		"font.size": 10,
		"axes.titlesize": 12,
		"axes.labelsize": 11,
		"xtick.labelsize": 10,
		"ytick.labelsize": 10,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"savefig.bbox": "tight",
		"savefig.dpi": 600,
	})

def color_palette(n: int) -> list:
	return sns.color_palette("colorblind", n)

# ---------- IO helpers ----------

def resolve_output_dir(tag: str) -> Path:
	base = Path(__file__).resolve().parents[1] / "figures"
	out = base / tag
	out.mkdir(parents=True, exist_ok=True)
	return out

# ---------- Example figure ----------

def example_exploit_explore_bar(metrics_csv: Path, out_dir: Path, title: str) -> None:
	df = pd.read_csv(metrics_csv)
	df = df.dropna(subset=["exploitation_intra_mean", "exploration_intra_mean"])  # required columns
	stats = []
	for label, col in [("Exploitation", "exploitation_intra_mean"), ("Exploration", "exploration_intra_mean")]:
		vals = df[col].astype(float).to_numpy()
		mu = float(np.mean(vals))
		sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
		se = sd / np.sqrt(len(vals)) if len(vals) > 0 else 0.0
		ci95 = 1.96 * se
		stats.append((label, mu, ci95))
	labels = [s[0] for s in stats]
	means = [s[1] for s in stats]
	err = [s[2] for s in stats]
	colors = {"Exploitation": "#D55E00", "Exploration": "#0072B2"}
	fig, ax = plt.subplots(figsize=(4.4, 2.6))
	ax.barh(np.arange(len(labels)), means, xerr=err, color=[colors[l] for l in labels], capsize=4, height=0.6)
	for i, (m, e) in enumerate(zip(means, err)):
		ax.text(m + max(0.01, 0.02 * m), i, f"{m:.3f}", va="center", ha="left", fontsize=10)
	ax.set_yticks(np.arange(len(labels)), labels)
	ax.set_xlabel("Mean cosine similarity (± 95% CI)")
	ax.set_title(title)
	ax.grid(axis="x", alpha=0.25)
	out_png = out_dir / f"{title.lower().replace(' ', '_')}.png"
	out_pdf = out_dir / f"{title.lower().replace(' ', '_')}.pdf"
	fig.savefig(out_png)
	fig.savefig(out_pdf)
	plt.close(fig)

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Archetype analysis script")
	p.add_argument("--metrics", type=Path, default=Path(__file__).resolve().parents[1] / "NATURE_REAL_metrics.csv",
				help="Path to metrics CSV (default: output/NATURE_REAL_metrics.csv)")
	p.add_argument("--out-tag", type=str, default="archetype",
				help="Subfolder name under output/figures for saving results")
	p.add_argument("--title", type=str, default="Exploitation vs Exploration",
				help="Title for the example bar figure")
	return p.parse_args()


def main() -> None:
	setup_style()
	args = parse_args()
	out_dir = resolve_output_dir(args.out_tag)
	example_exploit_explore_bar(args.metrics, out_dir, args.title)
	print("Saved figures to", out_dir)

if __name__ == "__main__":
	main()
