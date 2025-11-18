#!/usr/bin/env python3
"""Create disease stage-adjusted mediation figures using exact same formatting as working original"""

import numpy as np
import pandas as pd
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

def mediation_disease_stage_adjusted(df: pd.DataFrame, outcome_col: str, B: int = 5000, seed: int = 42):
    """Mediation analysis with disease stage (Hoehn and Yahr score) as covariate"""
    try:
        df = df[['norm_LC_avg', 'alpha_NET_mean', outcome_col, 'hoehn_yahr_score']].dropna()
        
        print(f"Disease stage adjusted mediation analysis: {len(df)} participants")
        
        X = df['norm_LC_avg'].to_numpy(float)
        M = df['alpha_NET_mean'].to_numpy(float)
        Y = df[outcome_col].to_numpy(float)
        C = df['hoehn_yahr_score'].to_numpy(float)  # Disease stage as covariate
        
        # z-score
        Xz = (X - X.mean()) / X.std(ddof=1)
        Mz = (M - M.mean()) / M.std(ddof=1)
        Yz = (Y - Y.mean()) / Y.std(ddof=1)
        Cz = (C - C.mean()) / C.std(ddof=1)
        ones = np.ones_like(Xz)
        
        # a: M ~ X + Disease Stage
        ba, _, ta, pa = ols(np.column_stack([ones, Xz, Cz]), Mz)
        a = float(ba[1]); p_a = float(pa[1])
        
        # b & c': Y ~ X + M + Disease Stage
        bb, _, tb, pb = ols(np.column_stack([ones, Xz, Mz, Cz]), Yz)
        c_prime = float(bb[1]); b = float(bb[2]); p_b = float(pb[2])
        
        # c: Y ~ X + Disease Stage
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
        }
        
    except Exception as e:
        print(f"Error in disease stage adjusted mediation: {e}")
        return None

def plot_mediation_disease_stage(result: dict, title: str, out_path: Path, outcome_type: str = 'coherence'):
    """Mediation figure with left path diagram and right effects decomposition bars - EXACT same as working version"""

    # Apply style and load the same color palette used by Figure 1
    colors_fig1 = setup_nature_style()

    # Figure: one row, two columns (diagram 55%, bars 45%) - EXACT same dimensions
    fig = plt.figure(figsize=(6.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])
    ax.axis('off')

    # Node centers (triangle) and rectangle size - EXACT same as working version
    x_c = (0.18, 0.32)
    m_c = (0.50, 0.70)
    y_c = (0.82, 0.32)
    rect_w, rect_h = 0.16, 0.10

    # Use the same palette as the age-adjusted Nature figures for consistency
    colors = {
        'x': colors_fig1['accent'],      # LC integrity
        'm': colors_fig1['secondary'],   # α-power
    }
    y_color = colors_fig1['highlight'] if outcome_type == 'coherence' else colors_fig1['purple']
    edge = '#000000'
    ax.add_patch(Rectangle((x_c[0] - rect_w/2, x_c[1] - rect_h/2), rect_w, rect_h, facecolor=colors['x'], edgecolor=edge, linewidth=0.5))
    ax.add_patch(Rectangle((m_c[0] - rect_w/2, m_c[1] - rect_h/2), rect_w, rect_h, facecolor=colors['m'], edgecolor=edge, linewidth=0.5))
    ax.add_patch(Rectangle((y_c[0] - rect_w/2, y_c[1] - rect_h/2), rect_w, rect_h, facecolor=y_color, edgecolor=edge, linewidth=0.5))

    # Variable labels below nodes (7pt) - EXACT same as working version
    outcome_label = 'EE metric' if outcome_type == 'coherence' else 'SVF Count'
    ax.text(x_c[0], x_c[1] - (rect_h/2 + 0.05), 'LC integrity', ha='center', va='top', fontsize=7, color='#000000')
    ax.text(m_c[0], m_c[1] - (rect_h/2 + 0.05), 'α-power', ha='center', va='top', fontsize=7, color='#000000')
    ax.text(y_c[0], y_c[1] - (rect_h/2 + 0.05), outcome_label, ha='center', va='top', fontsize=7, color='#000000')

    # Arrow styling - EXACT same as working version
    arrow_color = colors_fig1['neutral']
    lw_arrow = 0.8

    def draw_arrow(p0, p1):
        ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='->', lw=lw_arrow, color=arrow_color))

    # Compute arrow endpoints from rectangle edges - EXACT same as working version
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

    # Perpendicular offset label at arrow midpoint (offset = 0.03 units) - EXACT same as working version
    def label_perp(p0, p1, text):
        mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = (dx**2 + dy**2) ** 0.5
        if L < 1e-8:
            ux, uy = 0.0, 0.0
        else:
            ux, uy = -dy / L, dx / L
        ax.text(mx + 0.03 * ux, my + 0.03 * uy, text, ha='center', va='center', fontsize=7, color='#000000',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

    # X -> M (a) - EXACT same as working version
    a_start = edge_point(x_c, m_c)
    a_end = edge_point(m_c, x_c)
    draw_arrow(a_start, a_end)
    label_perp(a_start, a_end, f"a: β = {result['a']:.3f}")

    # M -> Y (b) - EXACT same as working version
    b_start = edge_point(m_c, y_c)
    b_end = edge_point(y_c, m_c)
    draw_arrow(b_start, b_end)
    label_perp(b_start, b_end, f"b: β = {result['b']:.3f}")

    # X -> Y (c' direct) - EXACT same as working version
    c_start = edge_point(x_c, y_c)
    c_end = edge_point(y_c, x_c)
    draw_arrow(c_start, c_end)
    cprime = result["c'"]
    label_perp(c_start, c_end, f"c′: β = {cprime:.3f}")

    # Axis limits - EXACT same as working version
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # In-figure legend (minimal, matches Fig 1 palette) - EXACT same as working version
    legend_handles = [
        mpatches.Patch(facecolor=colors['x'], edgecolor='black', linewidth=0.5, label='LC integrity'),
        mpatches.Patch(facecolor=colors['m'], edgecolor='black', linewidth=0.5, label='α-power'),
        mpatches.Patch(facecolor=y_color, edgecolor='black', linewidth=0.5, label=outcome_label),
        mlines.Line2D([0], [0], color=arrow_color, lw=lw_arrow, label='Path (a, b, c′)')
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.02, 1.00),
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        borderaxespad=0.3,
        labelspacing=0.5
    )

    # Right: Effects decomposition bars - EXACT same as working version
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
    # Value labels above bars - EXACT same as working version
    for rect, val in zip(bars, totals):
        ax_bars.text(rect.get_x() + rect.get_width()/2.0, rect.get_height() + np.sign(val)*0.02,
                     f"{val:.3f}", ha='center', va='bottom' if val >= 0 else 'top', fontsize=7,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    # Horizontal zero line
    ax_bars.axhline(0, color='#888888', linewidth=0.6)
    # Tight y-limits
    y_abs = max(0.2, np.nanmax(np.abs(totals)) * 1.25)
    ax_bars.set_ylim(-y_abs, y_abs)
    # Tight x-limits and margins
    ax_bars.set_xlim(-0.45, 2.05)
    ax_bars.margins(x=0.02)
    # Clean single-line overlay above bars (only change: "Disease stage-adjusted" instead of "Age-adjusted")
    ax_bars.set_title('', fontsize=8)
    ax_bars.text(0.5, 1.03,
                 f"Indirect: {result['indirect']:.3f} (95% CI: {result['ci_low']:.3f}, {result['ci_high']:.3f})  |  N = {result['N']}  |  Disease stage-adjusted",
                 transform=ax_bars.transAxes, ha='center', va='bottom', fontsize=7, color='#000000')

    # Save - EXACT same as working version
    out_path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main():
    """Generate disease stage-adjusted mediation figures using exact same formatting as working version"""
    print("Creating disease stage-adjusted mediation figures with exact working formatting...")
    
    # Load data with disease severity
    df = pd.read_csv('final_complete_disease_severity_mediation_data.csv')
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Outcome 1: exploitation_coherence_ratio (EE metric)
    res1 = mediation_disease_stage_adjusted(df, 'exploitation_coherence_ratio')
    if res1:
        plot_mediation_disease_stage(
            res1,
            'LC → α-power → Exploitation Coherence (Disease stage-adjusted)',
            output_dir / 'mediation_ee_disease_stage_working.png',
            outcome_type='coherence'
        )
        print(f"EE metric mediation (disease stage adjusted): N = {res1['N']}")
    
    # Outcome 2: SVF_count
    res2 = mediation_disease_stage_adjusted(df, 'SVF_count')
    if res2:
        plot_mediation_disease_stage(
            res2,
            'LC → α-power → SVF Performance (Disease stage-adjusted)',
            output_dir / 'mediation_svf_disease_stage_working.png',
            outcome_type='svf'
        )
        print(f"SVF Count mediation (disease stage adjusted): N = {res2['N']}")
    
    print('\nCreated disease stage-adjusted mediation figures with exact working formatting:')
    print(' - output/mediation_ee_disease_stage_working.(png|pdf)')
    print(' - output/mediation_svf_disease_stage_working.(png|pdf)')

if __name__ == '__main__':
    main()
