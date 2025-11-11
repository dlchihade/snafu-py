#!/usr/bin/env python3
"""Create disease stage-adjusted mediation figures with exact same formatting as original"""

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
            X_boot = Xz[idx]
            M_boot = Mz[idx]
            Y_boot = Yz[idx]
            C_boot = Cz[idx]
            ones_boot = np.ones_like(X_boot)
            
            # a: M ~ X + Disease Stage
            ba_boot, _, _, _ = ols(np.column_stack([ones_boot, X_boot, C_boot]), M_boot)
            a_boot = float(ba_boot[1])
            
            # b: Y ~ X + M + Disease Stage
            bb_boot, _, _, _ = ols(np.column_stack([ones_boot, X_boot, M_boot, C_boot]), Y_boot)
            b_boot = float(bb_boot[2])
            
            ab[i] = a_boot * b_boot
        
        # confidence interval
        ab_sorted = np.sort(ab)
        ci_lower = ab_sorted[int(0.025 * B)]
        ci_upper = ab_sorted[int(0.975 * B)]
        
        result = {
            'a': a, 'p_a': p_a,
            'b': b, 'p_b': p_b,
            'c': c_total, 'p_c': p_c,
            "c'": c_prime, 'p_c_prime': float(pb[1]),
            'indirect': a * b,
            'ci_low': ci_lower,
            'ci_high': ci_upper,
            'N': len(df)
        }
        
        return result
        
    except Exception as e:
        print(f"Error in disease stage adjusted mediation: {e}")
        return None

def create_mediation_figure_exact_format(result: dict, outcome_type: str, out_path: Path):
    """Create mediation figure with exact same formatting as original"""
    colors_fig1 = setup_nature_style()
    
    # Set outcome-specific colors and labels (exact same as original)
    if outcome_type == 'svf':
        outcome_label = 'SVF Count'
        y_color = colors_fig1['purple']  # Same as original
    else:
        outcome_label = 'EE metric'
        y_color = colors_fig1['highlight']  # Same as original
    
    # Create figure with exact same dimensions and layout
    fig = plt.figure(figsize=(3.46, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.3)
    
    # Left: Mediation diagram (exact same as original)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Node coordinates (exact same as original)
    x_c = (0.2, 0.35)
    m_c = (0.5, 0.65)
    y_c = (0.8, 0.35)
    
    # Node colors (exact same as original)
    colors = {
        'x': colors_fig1['primary'],
        'm': colors_fig1['secondary'],
        'y': y_color
    }
    
    # Draw nodes (exact same as original)
    node_coords = [x_c, m_c, y_c]
    node_colors = [colors['x'], colors['m'], colors['y']]
    node_labels = ['LC integrity', 'α-power', outcome_label]
    
    for (x, y), color, label in zip(node_coords, node_colors, node_labels):
        rect = Rectangle((x-0.06, y-0.03), 0.12, 0.06, 
                        facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x, y-0.08, label, ha='center', va='top', fontsize=7)
    
    # Arrow styling (exact same as original)
    arrow_color = '#404040'
    lw_arrow = 0.8
    
    def draw_arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=lw_arrow, color=arrow_color))
    
    def edge_point(from_c, to_c):
        dx, dy = to_c[0] - from_c[0], to_c[1] - from_c[1]
        rect_w, rect_h = 0.12, 0.06
        L = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / L, dy / L
        hx, hy = rect_w/2, rect_h/2
        tx = hx / abs(ux) if abs(ux) > 1e-8 else float('inf')
        ty = hy / abs(uy) if abs(uy) > 1e-8 else float('inf')
        t = min(tx, ty)
        return (from_c[0] + ux * t, from_c[1] + uy * t)
    
    # Perpendicular offset label at arrow midpoint (exact same as original)
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
    
    # Axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # In-figure legend (exact same as original)
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
    
    # Right: Effects decomposition bars (exact same as original)
    ax_bars = fig.add_subplot(gs[1])
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
    # Value labels above bars
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
    # Clean single-line overlay above bars (modified for disease stage)
    ax_bars.set_title('', fontsize=8)
    ax_bars.text(0.5, 1.03,
                 f"Indirect: {result['indirect']:.3f} (95% CI: {result['ci_low']:.3f}, {result['ci_high']:.3f})  |  N = {result['N']}  |  Disease stage-adjusted",
                 transform=ax_bars.transAxes, ha='center', va='bottom', fontsize=7, color='#000000')
    
    # Save with exact same parameters
    out_path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main():
    """Generate disease stage-adjusted mediation figures with exact same formatting"""
    print("Creating disease stage-adjusted mediation figures with exact formatting...")
    
    # Load data with disease severity
    df = pd.read_csv('final_complete_disease_severity_mediation_data.csv')
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # SVF Count mediation
    result_svf = mediation_disease_stage_adjusted(df, 'SVF_count')
    if result_svf:
        create_mediation_figure_exact_format(result_svf, 'svf', 
                                           output_dir / 'mediation_svf_disease_stage_exact.png')
        print(f"SVF Count mediation (disease stage adjusted): N = {result_svf['N']}")
    
    # EE metric mediation
    result_ee = mediation_disease_stage_adjusted(df, 'exploitation_coherence_ratio')
    if result_ee:
        create_mediation_figure_exact_format(result_ee, 'ee', 
                                           output_dir / 'mediation_ee_disease_stage_exact.png')
        print(f"EE metric mediation (disease stage adjusted): N = {result_ee['N']}")
    
    print("\nCreated disease stage-adjusted mediation figures with exact formatting:")
    print(" - output/mediation_svf_disease_stage_exact.(png|pdf)")
    print(" - output/mediation_ee_disease_stage_exact.(png|pdf)")

if __name__ == '__main__':
    main()

