#!/usr/bin/env python3
"""Create Nature-quality mediation figures with minimal, clean design"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.stats import t as t_dist
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
"""Nature-like style preset (fonts, ticks, spines, savefig) and muted palette."""
nature_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "lines.linewidth": 1.2,
    "lines.solid_capstyle": "butt",
    "patch.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

nature_palette = [
    "#2F4F4F",  # dark slate gray
    "#6C8EBF",  # muted blue
    "#B07AA1",  # muted magenta
    "#79A97A",  # muted green
]

def apply_nature_style():
    mpl.rcParams.update(nature_rc)


# Display label for outcome when using exploitation_coherence_ratio (not the E–E ratio index).
OUTCOME_LABEL_COHERENCE = 'Exploitation coherence'


def _annotate_bar_values(ax_bars, bars, totals):
    """Place value labels in offset points so PDF export does not overlap x-axis tick labels."""
    for bar, val in zip(bars, totals):
        x = bar.get_x() + bar.get_width() / 2.0
        h = bar.get_height()
        off = 10 if val >= 0 else -10
        va = 'bottom' if val >= 0 else 'top'
        ax_bars.annotate(
            f'{val:.3f}',
            xy=(x, h),
            xytext=(0, off),
            textcoords='offset points',
            ha='center', va=va, fontsize=7, color='#000000',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'),
        )


def mediation_age_adjusted(df: pd.DataFrame, outcome_col: str, B: int = 5000, seed: int = 42):
    # Load age data - use the correct age file with only participants who have complete data
    try:
        ages_df = pd.read_csv('participant_ages_correct_46.csv')
        df = df.merge(ages_df, on='ID', how='inner')
        df = df[['norm_LC_avg', 'alpha_NET_mean', outcome_col, 'Age']].dropna()
        # Exclude participants with zero values for exploitation_coherence_ratio (if that's the outcome)
        if outcome_col == 'exploitation_coherence_ratio':
            n_before = len(df)
            df = df[df[outcome_col] > 0].copy()
            n_after = len(df)
            if n_before > n_after:
                print(f'📊 Mediation ({outcome_col}): Excluding {n_before - n_after} participant(s) with zero values')
                print(f'   Sample size: {n_after} (was {n_before})')
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
    except Exception as e:
        print(f"Warning: Could not load age data ({e}), proceeding without age adjustment")
        df = df[['norm_LC_avg', 'alpha_NET_mean', outcome_col]].dropna()
        # Exclude participants with zero values for exploitation_coherence_ratio (if that's the outcome)
        if outcome_col == 'exploitation_coherence_ratio':
            n_before = len(df)
            df = df[df[outcome_col] > 0].copy()
            n_after = len(df)
            if n_before > n_after:
                print(f'📊 Mediation ({outcome_col}): Excluding {n_before - n_after} participant(s) with zero values')
                print(f'   Sample size: {n_after} (was {n_before})')
        X = df['norm_LC_avg'].to_numpy(float)
        M = df['alpha_NET_mean'].to_numpy(float)
        Y = df[outcome_col].to_numpy(float)
        # z-score
        Xz = (X - X.mean()) / X.std(ddof=1)
        Mz = (M - M.mean()) / M.std(ddof=1)
        Yz = (Y - Y.mean()) / Y.std(ddof=1)
        ones = np.ones_like(Xz)
        # a: M ~ X
        ba, _, ta, pa = ols(np.column_stack([ones, Xz]), Mz)
        a = float(ba[1]); p_a = float(pa[1])
        # b & c': Y ~ X + M
        bb, _, tb, pb = ols(np.column_stack([ones, Xz, Mz]), Yz)
        c_prime = float(bb[1]); b = float(bb[2]); p_b = float(pb[2])
        # c: Y ~ X
        bc, _, tc, pc = ols(np.column_stack([ones, Xz]), Yz)
        c_total = float(bc[1]); p_c = float(pc[1])
        # bootstrap for ab
        rng = np.random.default_rng(seed)
        N = len(df)
        ab = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, N, N)
            Xb, Mb, Yb = Xz[idx], Mz[idx], Yz[idx]
            a_b = ols(np.column_stack([np.ones_like(Xb), Xb]), Mb)[0][1]
            b_b = ols(np.column_stack([np.ones_like(Xb), Xb, Mb]), Yb)[0][2]
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
    }


def plot_mediation_nature(result: dict, title: str, out_path: Path, outcome_type: str = 'coherence'):
    """Mediation figure with left path diagram and right effects decomposition bars (Total, Direct, Indirect)."""

    # Apply style and load the same color palette used by Figure 1
    # (setup_nature_style updates rcParams and returns a colors dict)
    colors_fig1 = setup_nature_style()

    fig = plt.figure(figsize=(6.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])
    ax.axis('off')

    # Node centers (x, y); tuples are (px, py) for each vertex — do not swap
    x_c = (0.18, 0.32)
    m_c = (0.50, 0.70)
    y_c = (0.82, 0.32)
    rect_w, rect_h = 0.16, 0.10

    # Publication palette mapping: keep X and M fixed; vary Y by outcome type to differentiate panels
    colors = {
        'x': colors_fig1['accent'],     # X: blue
        'm': colors_fig1['secondary'],  # M: orange
    }
    # EE (coherence) uses green; SVF uses purple for clearer differentiation
    y_color = colors_fig1['highlight'] if outcome_type == 'coherence' else colors_fig1['purple']
    edge = '#000000'
    ax.add_patch(Rectangle((x_c[0] - rect_w/2, x_c[1] - rect_h/2), rect_w, rect_h, facecolor=colors['x'], edgecolor=edge, linewidth=0.5))
    ax.add_patch(Rectangle((m_c[0] - rect_w/2, m_c[1] - rect_h/2), rect_w, rect_h, facecolor=colors['m'], edgecolor=edge, linewidth=0.5))
    ax.add_patch(Rectangle((y_c[0] - rect_w/2, y_c[1] - rect_h/2), rect_w, rect_h, facecolor=y_color, edgecolor=edge, linewidth=0.5))

    gap_below_box = 0.018
    half_h = rect_h / 2
    outcome_label = OUTCOME_LABEL_COHERENCE if outcome_type == 'coherence' else 'SVF Count'
    fs_node = 6 if (outcome_type == 'coherence' and len(outcome_label) > 14) else 7

    def _node_caption(x0, y0, s, fs=7):
        # Box bottom y = y0 - half_h; text top sits gap_below_box under that edge
        ty = y0 - half_h - gap_below_box
        ax.text(x0, ty, s, ha='center', va='top', fontsize=fs, color='#000000')

    _node_caption(x_c[0], x_c[1], 'LC integrity', 7)
    _node_caption(m_c[0], m_c[1], 'α-power', 7)
    _node_caption(y_c[0], y_c[1], outcome_label, fs_node)

    # Arrow styling
    arrow_color = colors_fig1['neutral']
    lw_arrow = 0.8

    def draw_arrow(p0, p1):
        ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='->', lw=lw_arrow, color=arrow_color))

    # Compute arrow endpoints from rectangle edges
    def edge_point(c_from, c_to):
        x0, y0 = c_from
        x1, y1 = c_to
        dx, dy = x1 - x0, y1 - y0
        if dx == dy == 0:
            return c_from
        L = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / L, dy / L
        hx, hy = rect_w/2, rect_h/2
        tx = hx / abs(ux) if abs(ux) > 1e-8 else float('inf')
        ty = hy / abs(uy) if abs(uy) > 1e-8 else float('inf')
        t = min(tx, ty)
        return (x0 + ux * t, y0 + uy * t)

    def label_perp(p0, p1, text):
        mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = (dx**2 + dy**2) ** 0.5
        if L < 1e-8:
            return
        # Bottom horizontal c′: default ⟂ points up into triangle — use downward offset instead
        if abs(dy) < 1e-5 and my < 0.42:
            ux, uy = 0.0, -1.0
        else:
            ux, uy = -dy / L, dx / L
        off = 0.045
        ax.text(mx + off * ux, my + off * uy, text, ha='center', va='center', fontsize=7, color='#000000',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

    # X -> M (a)
    a_start = edge_point(x_c, m_c)
    a_end = edge_point(m_c, x_c)
    draw_arrow(a_start, a_end)
    label_perp(a_start, a_end, f"a: β = {result['a']:.3f}")

    # M -> Y (b)
    b_start = edge_point(m_c, y_c)
    b_end = edge_point(y_c, m_c)
    draw_arrow(b_start, b_end)
    label_perp(b_start, b_end, f"b: β = {result['b']:.3f}")

    # X -> Y (c' direct)
    c_start = edge_point(x_c, y_c)
    c_end = edge_point(y_c, x_c)
    draw_arrow(c_start, c_end)
    cprime = result["c'"]
    label_perp(c_start, c_end, f"c′: β = {cprime:.3f}")

    # Remove extra footnote text for a cleaner look

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    legend_handles = [
        mpatches.Patch(facecolor=colors['x'], edgecolor='black', linewidth=0.5, label='LC integrity'),
        mpatches.Patch(facecolor=colors['m'], edgecolor='black', linewidth=0.5, label='α-power'),
        mpatches.Patch(facecolor=y_color, edgecolor='black', linewidth=0.5, label=outcome_label),
        mlines.Line2D([0], [0], color=arrow_color, lw=lw_arrow, label='Path (a, b, c′)')
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.02, 1.0),
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        borderaxespad=0.3,
        labelspacing=0.5,
    )

    # Right: Effects decomposition bars
    totals = [result['c'], cprime, result['indirect']]
    labels = ['Total (c)', 'Direct (c′)', 'Indirect (a×b)']
    # Bar colors: make Total (c) follow the outcome color to differentiate panels
    bar_colors = [y_color, colors_fig1['secondary'], colors_fig1['accent']]
    # Tighter spacing between bars
    xs = np.array([0.0, 0.85, 1.7])
    bars = ax_bars.bar(xs, totals, width=0.55, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_bars.set_xticks(xs, labels)
    ax_bars.set_ylabel('Effect Size', fontsize=7)
    ax_bars.tick_params(axis='both', labelsize=7)
    ax_bars.tick_params(axis='x', pad=7)
    # Horizontal zero line
    ax_bars.axhline(0, color='#888888', linewidth=0.6)
    if outcome_type == 'svf':
        y_max = np.nanmax(totals) * 1.22
        ax_bars.set_ylim(0.0, y_max)
    else:
        y_abs = max(0.22, np.nanmax(np.abs(totals)) * 1.38)
        ax_bars.set_ylim(-y_abs, y_abs)
    ax_bars.set_xlim(-0.45, 2.05)
    ax_bars.margins(x=0.02)
    _annotate_bar_values(ax_bars, bars, totals)
    ax_bars.set_title(
        f"Indirect: {result['indirect']:.3f} (95% CI: {result['ci_low']:.3f}, {result['ci_high']:.3f})  |  "
        f"N = {result['N']}  |  Age-adjusted",
        fontsize=7, color='#000000', pad=10, loc='center',
    )

    out_path.parent.mkdir(exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    fig.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    plt.close(fig)


def plot_mediation_nature_coherence(result: dict, title: str, out_path: Path, outcome_type: str = 'coherence'):
    """Same layout as plot_mediation_nature; kept for a second output file if needed."""
    colors_fig1 = setup_nature_style()
    fig = plt.figure(figsize=(6.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])
    ax.axis('off')

    x_c = (0.18, 0.32)
    m_c = (0.50, 0.70)
    y_c = (0.82, 0.32)
    rect_w, rect_h = 0.16, 0.10

    colors = {
        'x': colors_fig1['accent'],
        'm': colors_fig1['secondary'],
    }
    y_color = colors_fig1['highlight'] if outcome_type == 'coherence' else colors_fig1['purple']
    edge = '#000000'
    ax.add_patch(Rectangle((x_c[0] - rect_w/2, x_c[1] - rect_h/2), rect_w, rect_h, facecolor=colors['x'], edgecolor=edge, linewidth=0.5))
    ax.add_patch(Rectangle((m_c[0] - rect_w/2, m_c[1] - rect_h/2), rect_w, rect_h, facecolor=colors['m'], edgecolor=edge, linewidth=0.5))
    ax.add_patch(Rectangle((y_c[0] - rect_w/2, y_c[1] - rect_h/2), rect_w, rect_h, facecolor=y_color, edgecolor=edge, linewidth=0.5))

    gap_below_box = 0.018
    half_h = rect_h / 2
    outcome_label = OUTCOME_LABEL_COHERENCE if outcome_type == 'coherence' else 'SVF Count'
    fs_node = 6 if (outcome_type == 'coherence' and len(outcome_label) > 14) else 7

    def _node_caption(x0, y0, s, fs=7):
        ty = y0 - half_h - gap_below_box
        ax.text(x0, ty, s, ha='center', va='top', fontsize=fs, color='#000000')

    _node_caption(x_c[0], x_c[1], 'LC integrity', 7)
    _node_caption(m_c[0], m_c[1], 'α-power', 7)
    _node_caption(y_c[0], y_c[1], outcome_label, fs_node)

    arrow_color = colors_fig1['neutral']
    lw_arrow = 0.8

    def draw_arrow(p0, p1):
        ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='->', lw=lw_arrow, color=arrow_color))

    def edge_point(c_from, c_to):
        x0, y0 = c_from
        x1, y1 = c_to
        dx, dy = x1 - x0, y1 - y0
        if dx == dy == 0:
            return c_from
        L = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / L, dy / L
        hx, hy = rect_w/2, rect_h/2
        tx = hx / abs(ux) if abs(ux) > 1e-8 else float('inf')
        ty = hy / abs(uy) if abs(uy) > 1e-8 else float('inf')
        t = min(tx, ty)
        return (x0 + ux * t, y0 + uy * t)

    def label_perp(p0, p1, text):
        mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = (dx**2 + dy**2) ** 0.5
        if L < 1e-8:
            return
        if abs(dy) < 1e-5 and my < 0.42:
            ux, uy = 0.0, -1.0
        else:
            ux, uy = -dy / L, dx / L
        off = 0.045
        ax.text(mx + off * ux, my + off * uy, text, ha='center', va='center', fontsize=7, color='#000000',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

    a_start = edge_point(x_c, m_c)
    a_end = edge_point(m_c, x_c)
    draw_arrow(a_start, a_end)
    label_perp(a_start, a_end, f"a: β = {result['a']:.3f}")

    b_start = edge_point(m_c, y_c)
    b_end = edge_point(y_c, m_c)
    draw_arrow(b_start, b_end)
    label_perp(b_start, b_end, f"b: β = {result['b']:.3f}")

    c_start = edge_point(x_c, y_c)
    c_end = edge_point(y_c, x_c)
    draw_arrow(c_start, c_end)
    cprime = result["c'"]
    label_perp(c_start, c_end, f"c′: β = {cprime:.3f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    legend_handles = [
        mpatches.Patch(facecolor=colors['x'], edgecolor='black', linewidth=0.5, label='LC integrity'),
        mpatches.Patch(facecolor=colors['m'], edgecolor='black', linewidth=0.5, label='α-power'),
        mpatches.Patch(facecolor=y_color, edgecolor='black', linewidth=0.5, label=outcome_label),
        mlines.Line2D([0], [0], color=arrow_color, lw=lw_arrow, label='Path (a, b, c′)')
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.02, 1.0),
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        borderaxespad=0.3,
        labelspacing=0.5,
    )

    totals = [result['c'], cprime, result['indirect']]
    labels = ['Total (c)', 'Direct (c′)', 'Indirect (a×b)']
    bar_colors = [y_color, colors_fig1['secondary'], colors_fig1['accent']]
    xs = np.array([0.0, 0.85, 1.7])
    bars = ax_bars.bar(xs, totals, width=0.55, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_bars.set_xticks(xs, labels)
    ax_bars.set_ylabel('Effect Size', fontsize=7)
    ax_bars.tick_params(axis='both', labelsize=7)
    ax_bars.tick_params(axis='x', pad=7)
    ax_bars.axhline(0, color='#888888', linewidth=0.6)
    if outcome_type == 'svf':
        y_max = np.nanmax(totals) * 1.22
        ax_bars.set_ylim(0.0, y_max)
    else:
        y_abs = max(0.22, np.nanmax(np.abs(totals)) * 1.38)
        ax_bars.set_ylim(-y_abs, y_abs)
    ax_bars.set_xlim(-0.45, 2.05)
    ax_bars.margins(x=0.02)
    _annotate_bar_values(ax_bars, bars, totals)
    ax_bars.set_title(
        f"Indirect: {result['indirect']:.3f} (95% CI: {result['ci_low']:.3f}, {result['ci_high']:.3f})  |  "
        f"N = {result['N']}  |  Age-adjusted",
        fontsize=7, color='#000000', pad=10, loc='center',
    )

    out_path.parent.mkdir(exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    fig.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    metrics_path = base_dir / 'output' / 'NATURE_REAL_metrics.csv'
    metrics = pd.read_csv(metrics_path)

    # Try to merge SVF counts if available
    svf_merged = False
    for candidate in [base_dir / 'data' / 'fluency_data.csv',
                      base_dir.parent / 'fluency_data' / 'snafu_sample.csv',
                      base_dir.parent / 'fluency_data' / 'SVF Data1.csv']:
        try:
            if candidate.exists():
                flu = pd.read_csv(candidate)
                if 'ID' in flu.columns:
                    svf = flu.groupby('ID').size().reset_index(name='SVF_count')
                    metrics = metrics.merge(svf, on='ID', how='left')
                    svf_merged = True
                    break
        except Exception:
            pass

    # Outcome 1: exploitation_coherence_ratio
    res1 = mediation_age_adjusted(metrics, 'exploitation_coherence_ratio')
    plot_mediation_nature(
        res1,
        f'LC → α-power → {OUTCOME_LABEL_COHERENCE} (Age-adjusted)',
        base_dir / 'output' / 'mediation_exploit_age_nature.png',
        outcome_type='coherence'
    )
    
    # Outcome 1b: duplicate export (same computation and labels)
    plot_mediation_nature_coherence(
        res1,
        f'LC → α-power → {OUTCOME_LABEL_COHERENCE} (Age-adjusted)',
        base_dir / 'output' / 'mediation_exploit_coherence_metric_nature.png',
        outcome_type='coherence'
    )

    # Outcome 2: SVF_count (only if SVF merged)
    if svf_merged and 'SVF_count' in metrics.columns:
        res2 = mediation_age_adjusted(metrics, 'SVF_count')
        plot_mediation_nature(
            res2,
            'LC → α-power → SVF Performance (Age-adjusted)',
            base_dir / 'output' / 'mediation_svf_age_nature.png',
            outcome_type='svf'
        )

    print('\nCreated Nature-quality mediation figures:')
    print(f' - semantic_fluency_analysis/output/mediation_exploit_age_nature.(png|pdf) [{OUTCOME_LABEL_COHERENCE}]')
    print(f' - semantic_fluency_analysis/output/mediation_exploit_coherence_metric_nature.(png|pdf) [{OUTCOME_LABEL_COHERENCE}]')
    if svf_merged:
        print(' - semantic_fluency_analysis/output/mediation_svf_age_nature.(png|pdf)')


if __name__ == '__main__':
    main()
