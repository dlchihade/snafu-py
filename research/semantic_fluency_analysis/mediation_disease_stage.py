#!/usr/bin/env python3
"""Create Nature-quality mediation figures with disease stage as covariate"""

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

def mediation_disease_stage_adjusted(df: pd.DataFrame, outcome_col: str, B: int = 5000, seed: int = 42):
    """Mediation analysis with disease stage (Hoehn and Yahr score) as covariate"""
    # Use the disease severity data directly
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
            'ab': a * b,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(df)
        }
        
        return result
        
    except Exception as e:
        print(f"Error in disease stage adjusted mediation: {e}")
        return None

def create_mediation_figure_disease_stage(result: dict, outcome_type: str, save_path: str):
    """Create Nature-quality mediation figure with disease stage adjustment"""
    setup_nature_style()
    colors_fig1 = setup_nature_style()
    
    # Set outcome label
    if outcome_type == 'svf':
        outcome_label = 'SVF Count'
        y_color = colors_fig1['purple']
    else:
        outcome_label = 'EE metric'
        y_color = colors_fig1['highlight']
    
    # Create figure with 3.46 × 3.0 inch size
    fig = plt.figure(figsize=(3.46, 3.0))
    
    # Create grid: 55% for diagram, 45% for bars
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.3)
    
    # Panel A: Mediation diagram
    ax_diagram = fig.add_subplot(gs[0])
    ax_diagram.set_xlim(0, 1)
    ax_diagram.set_ylim(0, 1)
    ax_diagram.set_aspect('equal')
    
    # Node coordinates
    x_coords = [0.2, 0.5, 0.8]
    y_coords = [0.35, 0.65, 0.35]
    
    # Draw nodes
    node_colors = [colors_fig1['primary'], colors_fig1['secondary'], y_color]
    node_labels = ['LC integrity', 'α-power', outcome_label]
    
    for i, (x, y, color, label) in enumerate(zip(x_coords, y_coords, node_colors, node_labels)):
        rect = Rectangle((x-0.06, y-0.03), 0.12, 0.06, 
                        facecolor=color, edgecolor='black', linewidth=0.5)
        ax_diagram.add_patch(rect)
        ax_diagram.text(x, y-0.08, label, ha='center', va='top', fontsize=7)
    
    # Draw arrows and labels
    def edge_point(x1, y1, x2, y2, offset=0.06):
        """Calculate point on edge of rectangle"""
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx/length, dy/length
        return x1 + offset*dx, y1 + offset*dy
    
    def label_perp(x1, y1, x2, y2, text, offset=0.03):
        """Place label perpendicular to arrow"""
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            ux, uy = -dy/length, dx/length  # perpendicular unit vector
            ax_diagram.text(mx + offset * ux, my + offset * uy, text, 
                           ha='center', va='center', fontsize=7, color='#000000',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # X -> M (path a)
    x1, y1 = edge_point(x_coords[0], y_coords[0], x_coords[1], y_coords[1])
    x2, y2 = edge_point(x_coords[1], y_coords[1], x_coords[0], y_coords[0])
    ax_diagram.arrow(x1, y1, x2-x1, y2-y1, head_width=0.02, head_length=0.02, 
                    fc='#404040', ec='#404040', linewidth=0.8)
    label_perp(x1, y1, x2, y2, f"a: β = {result['a']:.3f}")
    
    # M -> Y (path b)
    x1, y1 = edge_point(x_coords[1], y_coords[1], x_coords[2], y_coords[2])
    x2, y2 = edge_point(x_coords[2], y_coords[2], x_coords[1], y_coords[1])
    ax_diagram.arrow(x1, y1, x2-x1, y2-y1, head_width=0.02, head_length=0.02, 
                    fc='#404040', ec='#404040', linewidth=0.8)
    label_perp(x1, y1, x2, y2, f"b: β = {result['b']:.3f}")
    
    # X -> Y (path c')
    x1, y1 = edge_point(x_coords[0], y_coords[0], x_coords[2], y_coords[2])
    x2, y2 = edge_point(x_coords[2], y_coords[2], x_coords[0], y_coords[0])
    ax_diagram.arrow(x1, y1, x2-x1, y2-y1, head_width=0.02, head_length=0.02, 
                    fc='#404040', ec='#404040', linewidth=0.8)
    label_perp(x1, y1, x2, y2, f"c′: β = {result['c']:.3f}")
    
    # Add legend
    handles = [
        mpatches.Patch(color=colors_fig1['primary'], label='LC integrity'),
        mpatches.Patch(color=colors_fig1['secondary'], label='α-power'),
        mpatches.Patch(color=y_color, label=outcome_label),
        mlines.Line2D([], [], color='#404040', linewidth=0.8, label='Path')
    ]
    labels = ['LC integrity', 'α-power', outcome_label, 'Path']
    ax_diagram.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.02, 1.00), 
                     fontsize=7, frameon=False)
    
    ax_diagram.set_xticks([])
    ax_diagram.set_yticks([])
    for spine in ax_diagram.spines.values():
        spine.set_visible(False)
    
    # Panel B: Bar plot
    ax_bars = fig.add_subplot(gs[1])
    
    # Bar data
    bar_labels = ['Total (c)', 'Direct (c′)', 'Indirect (a×b)']
    bar_values = [result['c'], result["c'"], result['ab']]
    bar_colors = [y_color, colors_fig1['primary'], colors_fig1['secondary']]
    
    xs = [0.0, 0.85, 1.7]
    width = 0.55
    
    bars = ax_bars.bar(xs, bar_values, width=width, color=bar_colors, alpha=0.85)
    
    # Add value labels with background boxes
    for rect, val in zip(bars, bar_values):
        ax_bars.text(rect.get_x() + rect.get_width()/2.0, rect.get_height() + np.sign(val)*0.02,
                    f"{val:.3f}", ha='center', va='bottom' if val >= 0 else 'top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add statistical summary
    indirect_text = f"Indirect effect: {result['ab']:.3f} (95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]) | N = {result['n']} | Disease stage-adjusted"
    ax_bars.text(0.5, 0.95, indirect_text, transform=ax_bars.transAxes, 
                ha='center', va='top', fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax_bars.set_xticks(xs)
    ax_bars.set_xticklabels(bar_labels, fontsize=7)
    ax_bars.set_ylabel('Effect Size', fontsize=7)
    ax_bars.set_xlim(-0.5, 2.2)
    ax_bars.axhline(y=0, color='black', linewidth=0.6, alpha=0.6)
    
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax_bars.spines[spine].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate mediation figures with disease stage adjustment"""
    print("Creating disease stage-adjusted mediation figures...")
    
    # Load data with disease severity
    df = pd.read_csv('final_complete_disease_severity_mediation_data.csv')
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # SVF Count mediation
    result_svf = mediation_disease_stage_adjusted(df, 'SVF_count')
    if result_svf:
        create_mediation_figure_disease_stage(result_svf, 'svf', 
                                            output_dir / 'mediation_svf_disease_stage_nature.png')
        create_mediation_figure_disease_stage(result_svf, 'svf', 
                                            output_dir / 'mediation_svf_disease_stage_nature.pdf')
        print(f"SVF Count mediation (disease stage adjusted): N = {result_svf['n']}")
    
    # EE metric mediation
    result_ee = mediation_disease_stage_adjusted(df, 'exploitation_coherence_ratio')
    if result_ee:
        create_mediation_figure_disease_stage(result_ee, 'ee', 
                                            output_dir / 'mediation_ee_disease_stage_nature.png')
        create_mediation_figure_disease_stage(result_ee, 'ee', 
                                            output_dir / 'mediation_ee_disease_stage_nature.pdf')
        print(f"EE metric mediation (disease stage adjusted): N = {result_ee['n']}")
    
    print("\nCreated disease stage-adjusted mediation figures:")
    print(" - output/mediation_svf_disease_stage_nature.(png|pdf)")
    print(" - output/mediation_ee_disease_stage_nature.(png|pdf)")

if __name__ == '__main__':
    main()
