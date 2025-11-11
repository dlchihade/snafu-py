#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t as t_dist
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from create_nature_quality_figures_real import setup_nature_style


def ols(X: np.ndarray, y: np.ndarray):
	n, p = X.shape
	XtX = X.T @ X
	beta = np.linalg.inv(XtX) @ (X.T @ y)
	resid = y - X @ beta
	dof = max(n - p, 1)
	s2 = float(resid.T @ resid) / dof
	cov = s2 * np.linalg.inv(XtX)
	se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
	with np.errstate(divide='ignore', invalid='ignore'):
		t_vals = np.where(se > 0, beta / se, np.nan)
	p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), dof))
	return beta, se, t_vals, p_vals


def mediation_age_adjusted(df: pd.DataFrame, outcome_col: str, B: int = 5000, seed: int = 42):
	df = df[['norm_LC_avg', 'alpha_NET_mean', outcome_col, 'Age']].dropna()
	X = df['norm_LC_avg'].to_numpy(float)
	M = df['alpha_NET_mean'].to_numpy(float)
	Y = df[outcome_col].to_numpy(float)
	C = df['Age'].to_numpy(float)
	# z-score
	Xz = (X - X.mean()) / X.std(ddof=1)
	Mz = (M - M.mean()) / M.std(ddof=1)
	Yz = (Y - Y.mean()) / Y.std(ddof=1)
	Cz = (C - C.mean()) / C.std(ddof=1)
	ones = np.ones_like(Xz)
	# a: M ~ X + Age
	ba, _, ta, pa = ols(np.column_stack([ones, Xz, Cz]), Mz)
	a = float(ba[1]); p_a = float(pa[1])
	# b & c': Y ~ X + M + Age
	bb, _, tb, pb = ols(np.column_stack([ones, Xz, Mz, Cz]), Yz)
	c_prime = float(bb[1]); b = float(bb[2]); p_b = float(pb[2])
	# c: Y ~ X + Age
	bc, _, tc, pc = ols(np.column_stack([ones, Xz, Cz]), Yz)
	c_total = float(bc[1]); p_c = float(pc[1])
	# bootstrap for ab
	rng = np.random.default_rng(seed)
	N = len(df)
	ab = np.empty(B)
	for i in range(B):
		idx = rng.integers(0, N, N)
		Xb, Mb, Yb, Cb = Xz[idx], Mz[idx], Yz[idx], Cz[idx]
		a_b = ols(np.column_stack([np.ones_like(Xb), Xb, Cb]), Mb)[0][1]
		b_b = ols(np.column_stack([np.ones_like(Xb), Xb, Mb, Cb]), Yb)[0][2]
		ab[i] = a_b * b_b
	ci_low, ci_high = np.percentile(ab, [2.5, 97.5])
	return {
		'N': N,
		'a': a, 'p_a': p_a,
		'b': b, 'p_b': p_b,
		"c'": c_prime,
		'c': c_total, 'p_c': p_c,
		'indirect': float(a * b),
		'ci_low': float(ci_low), 'ci_high': float(ci_high),
		'ab_boot': ab,
		'X': Xz, 'M': Mz, 'Y': Yz,
	}

def plot_mediation(result: dict, title: str, out_path: Path):
	# Two-row layout inside a 3.46×2.4 in figure (Nature single column)
	fig = plt.figure(figsize=(3.46, 2.4))
	gs = gridspec.GridSpec(2, 1, height_ratios=[0.68, 0.32], hspace=0.04)
	ax = fig.add_subplot(gs[0])
	ax_tbl = fig.add_subplot(gs[1])
	ax.axis('off')
	ax_tbl.axis('off')

	# Colors and typography per spec
	text_color = '#000000'
	arrow_color = '#606060'
	node_edge = '#000000'
	lw_node = 0.5
	lw_arrow = 0.8

	# Determine labels
	outcome_label = 'Exploitation Coherence Ratio'
	if 'SVF' in title:
		outcome_label = 'SVF Count'

	# Node rectangles (simple, white fill, 0.5pt black border)
	x_pos, m_pos, y_pos = 0.18, 0.50, 0.82
	base_y = 0.58
	w_box, h_box = 0.16, 0.10
	for pos in [x_pos, m_pos, y_pos]:
		ax.add_patch(Rectangle((pos - w_box/2, base_y - h_box/2), w_box, h_box, facecolor='white', edgecolor=node_edge, linewidth=lw_node))

	# Node labels (7pt)
	ax.text(x_pos, base_y, 'X', ha='center', va='center', fontweight='bold', fontsize=7, color=text_color)
	ax.text(m_pos, base_y + 0.00, 'M', ha='center', va='center', fontweight='bold', fontsize=7, color=text_color)
	ax.text(y_pos, base_y, 'Y', ha='center', va='center', fontweight='bold', fontsize=7, color=text_color)
	ax.text(x_pos, base_y - 0.16, 'LC integrity (NM-MRI)', ha='center', va='top', fontsize=7, color=text_color)
	ax.text(m_pos, base_y + 0.16, 'α-power (8-12 Hz)', ha='center', va='top', fontsize=7, color=text_color)
	ax.text(y_pos, base_y - 0.16, outcome_label, ha='center', va='top', fontsize=7, color=text_color)

	# Arrow styles per significance: a, b dashed (p>=0.05); c solid
	ls_a = '--' if result['p_a'] >= 0.05 else '-'
	ls_b = '--' if result['p_b'] >= 0.05 else '-'
	ls_c = '-'

	# Draw arrows
	ax.annotate('', xy=(m_pos - w_box/2, base_y), xytext=(x_pos + w_box/2, base_y),
		arrowprops=dict(arrowstyle='->', lw=lw_arrow, color=arrow_color, linestyle=ls_a))
	ax.annotate('', xy=(y_pos - w_box/2, base_y), xytext=(m_pos + w_box/2, base_y),
		arrowprops=dict(arrowstyle='->', lw=lw_arrow, color=arrow_color, linestyle=ls_b))
	ax.annotate('', xy=(y_pos - w_box/2, base_y - 0.02), xytext=(x_pos + w_box/2, base_y - 0.02),
		arrowprops=dict(arrowstyle='->', lw=lw_arrow, color=arrow_color, linestyle=ls_c))

	# Coefficient labels (plain 6.5pt black at arrow midpoints)
	ax.text((x_pos + m_pos)/2, base_y + 0.04, f"β = {result['a']:.3f}", ha='center', va='bottom', fontsize=6.5, color=text_color)
	ax.text((m_pos + y_pos)/2, base_y + 0.04, f"β = {result['b']:.3f}", ha='center', va='bottom', fontsize=6.5, color=text_color)
	ax.text((x_pos + y_pos)/2, base_y - 0.08, f"β = {result['c']:.3f}", ha='center', va='top', fontsize=6.5, color=text_color)

	# Panel label 'A'
	ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=9, fontweight='bold', ha='left', va='top', color=text_color)

	# Bottom statistics table (6.5pt; black borders; #F0F0F0 header)
	def fmt_p(p):
		return '<0.001' if p < 0.001 else f'{p:.3f}'
	rows = [
		['Path', 'β', 'p'],
		['X→M', f"{result['a']:.3f}", fmt_p(result['p_a'])],
		['M→Y', f"{result['b']:.3f}", fmt_p(result['p_b'])],
		["X→Y", f"{result['c']:.3f}", fmt_p(result['p_c'])],
		['Indirect (a×b)', f"{result['indirect']:.3f}", ''],
		['95% CI', f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}]", ''],
		['N', f"{result['N']}", ''],
	]
	table = ax_tbl.table(cellText=rows[1:], colLabels=rows[0], cellLoc='center', colLoc='center', loc='center')
	table.auto_set_font_size(False)
	table.set_fontsize(6.5)
	table.scale(1.0, 1.2)
	for (i, j), cell in table.get_celld().items():
		cell.set_edgecolor('#000000')
		cell.set_linewidth(0.5)
		if i == 0:
			cell.set_facecolor('#F0F0F0')
			cell.get_text().set_weight('bold')

	# Title and margins
	fig.text(0.5, 0.995, title, ha='center', va='top', fontsize=8, fontweight='bold', color=text_color)
	fig.subplots_adjust(left=0.06, right=0.995, top=0.955, bottom=0.09)
	out_path.parent.mkdir(exist_ok=True)
	fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
	fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.01)
	plt.close(fig)
	# Determine Y label from title (Title Case, consistent with Nature figures)
	outcome_label = 'Exploitation Coherence Ratio'
	if 'SVF' in title:
		outcome_label = 'SVF Count'
	# Grayscale styling per guidance
	arrow_color = '#606060'
	node_edge = '#808080'
	text_color = '#222222'
	lw_node = 0.6
	lw_arrow = 0.8
	# Node positions and subtle rectangles
	x_pos, m_pos, y_pos = 0.18, 0.50, 0.82
	base_y = 0.58
	w_box, h_box = 0.14, 0.09
	ax.add_patch(Rectangle((x_pos - w_box/2, base_y - h_box/2), w_box, h_box, facecolor='white', edgecolor=node_edge, linewidth=lw_node))
	ax.add_patch(Rectangle((m_pos - w_box/2, base_y + 0.12 - h_box/2), w_box, h_box, facecolor='white', edgecolor=node_edge, linewidth=lw_node))
	ax.add_patch(Rectangle((y_pos - w_box/2, base_y - h_box/2), w_box, h_box, facecolor='white', edgecolor=node_edge, linewidth=lw_node))
	# Labels and variable names (formal)
	ax.text(x_pos, base_y, 'X', ha='center', va='center', fontweight='bold', color=text_color)
	ax.text(m_pos, base_y + 0.12, 'M', ha='center', va='center', fontweight='bold', color=text_color)
	ax.text(y_pos, base_y, 'Y', ha='center', va='center', fontweight='bold', color=text_color)
	ax.text(x_pos, base_y - 0.15, 'LC integrity', ha='center', va='top', fontsize=7, color=text_color)
	ax.text(m_pos, base_y + 0.26, 'Alpha power', ha='center', va='top', fontsize=7, color=text_color)
	ax.text(y_pos, base_y - 0.15, outcome_label, ha='center', va='top', fontsize=7, color=text_color)
	# Subtle arrows and coefficient labels
	arrow_kw = dict(arrowstyle='->', lw=lw_arrow, color=arrow_color)
	# a path
	ax.annotate('', xy=(m_pos - w_box/2, base_y + 0.12), xytext=(x_pos + w_box/2, base_y), arrowprops=arrow_kw)
	# b path
	ax.annotate('', xy=(y_pos - w_box/2, base_y), xytext=(m_pos + w_box/2, base_y + 0.12), arrowprops=arrow_kw)
	# c' path
	ax.annotate('', xy=(y_pos - w_box/2, base_y), xytext=(x_pos + w_box/2, base_y), arrowprops=arrow_kw)
	# p-value helper
	def fmt_p(p):
		return '<0.001' if p < 0.001 else f'{p:.3f}'
	# β labels near arrows with white background (no border)
	ax.text((x_pos + m_pos)/2 - 0.03, (base_y + (base_y + 0.12))/2 + 0.03, f"β = {result['a']:.3f}", ha='center', fontsize=7, color=text_color,
		bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.95))
	ax.text((m_pos + y_pos)/2 + 0.03, (base_y + (base_y + 0.12))/2 + 0.03, f"β = {result['b']:.3f}", ha='center', fontsize=7, color=text_color,
		bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.95))
	ax.text((x_pos + y_pos)/2, base_y - 0.06, f"β = {result['c']:.3f}", ha='center', fontsize=7, color=text_color,
		bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.95))
	# Panel label and age note
	ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', ha='left', va='top')
	ax.text(0.01, 0.02, 'Age-adjusted analysis', transform=ax.transAxes, fontsize=6, color='#666666', ha='left', va='bottom')
	# Inset statistics box inside main panel
	inset = (
		f"Indirect effect: a×b = {result['indirect']:.3f}\n"
		f"95% CI: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]\n"
		f"a p={fmt_p(result['p_a'])}, b p={fmt_p(result['p_b'])}, N={result['N']}"
	)
	ax.text(0.98, 0.02, inset, transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5,
		bbox=dict(facecolor='#F5F5F5', edgecolor='#DDDDDD', boxstyle='round,pad=0.3', linewidth=0.5))
	# Title and compact padding
	fig.text(0.5, 0.995, title, ha='center', va='top', fontsize=8, fontweight='bold')
	fig.subplots_adjust(left=0.06, right=0.995, top=0.955, bottom=0.06)
	out_path.parent.mkdir(exist_ok=True)
	fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
	fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.01)
	plt.close(fig)


def main():
    metrics = pd.read_csv('output/NATURE_REAL_metrics.csv')
    # SVF count merge
    flu = pd.read_csv('data/fluency_data.csv')
    svf = flu.groupby('ID').size().reset_index(name='SVF_count')
    metrics = metrics.merge(svf, on='ID', how='left')
    # Outcome 1: exploitation_coherence_ratio
    res1 = mediation_age_adjusted(metrics, 'exploitation_coherence_ratio')
    plot_mediation(res1, 'LC → Alpha → Exploitation Coherence (Age-Adjusted)', Path('output/mediation_exploit_age.png'))
    # Outcome 2: SVF_count
    res2 = mediation_age_adjusted(metrics, 'SVF_count')
    plot_mediation(res2, 'LC → Alpha → SVF Performance (Age-Adjusted)', Path('output/mediation_svf_age.png'))
    print('Saved mediation figures to output/:')
    print(' - mediation_exploit_age.(png|pdf)')
    print(' - mediation_svf_age.(png|pdf)')


if __name__ == '__main__':
	main()
