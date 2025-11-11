#!/usr/bin/env python3
"""Create publication-quality mediation figures with disease stage as covariate"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.stats import t as t_dist

def setup_publication_style():
    """Setup publication-quality style parameters"""
    # Set seaborn style
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_context("paper", font_scale=1.0)
    
    # Publication-quality parameters
    plt.rcParams.update({
        # Font settings
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Line and marker settings
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # Tick settings
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        
        # Patch settings
        'patch.linewidth': 0.8,
    })

def ols(X: np.ndarray, y: np.ndarray):
    """Ordinary least squares regression"""
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
            'ab': a * b,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(df)
        }
        
        return result
        
    except Exception as e:
        print(f"Error in disease stage adjusted mediation: {e}")
        return None

def create_mediation_figure_publication(result: dict, outcome_type: str, save_path: str):
    """Create publication-quality mediation figure"""
    setup_publication_style()
    
    # Color palette
    colors = {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',    # Purple
        'accent': '#F18F01',       # Orange
        'text': '#2C2C2C',         # Dark gray
        'light_gray': '#E5E5E5',   # Light gray
        'white': '#FFFFFF'         # White
    }
    
    # Set outcome-specific colors
    if outcome_type == 'svf':
        outcome_label = 'SVF Count'
        outcome_color = colors['accent']
    else:
        outcome_label = 'EE metric'
        outcome_color = colors['secondary']
    
    # Create figure with proper dimensions
    fig, (ax_diagram, ax_bars) = plt.subplots(1, 2, figsize=(7.2, 3.0), 
                                             gridspec_kw={'width_ratios': [0.6, 0.4], 'wspace': 0.3})
    
    # Panel A: Mediation diagram
    ax_diagram.set_xlim(0, 1)
    ax_diagram.set_ylim(0, 1)
    ax_diagram.set_aspect('equal')
    
    # Node coordinates
    x_coords = [0.2, 0.5, 0.8]
    y_coords = [0.35, 0.65, 0.35]
    
    # Draw nodes with better styling
    node_colors = [colors['primary'], colors['secondary'], outcome_color]
    node_labels = ['LC integrity', 'α-power', outcome_label]
    
    for i, (x, y, color, label) in enumerate(zip(x_coords, y_coords, node_colors, node_labels)):
        # Create rounded rectangle
        rect = Rectangle((x-0.06, y-0.03), 0.12, 0.06, 
                        facecolor=color, edgecolor=colors['text'], 
                        linewidth=1.0, alpha=0.9, zorder=3)
        ax_diagram.add_patch(rect)
        
        # Add label with better positioning
        ax_diagram.text(x, y-0.1, label, ha='center', va='top', 
                       fontsize=8, fontweight='normal', color=colors['text'])
    
    # Draw arrows with better styling
    def draw_arrow(ax, x1, y1, x2, y2, offset=0.06, color=colors['text']):
        """Draw arrow with proper styling"""
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx/length, dy/length
        start_x, start_y = x1 + offset*dx, y1 + offset*dy
        end_x, end_y = x2 - offset*dx, y2 - offset*dy
        
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=color, 
                                 connectionstyle='arc3,rad=0', alpha=0.8))
    
    def add_path_label(ax, x1, y1, x2, y2, text, offset=0.04):
        """Add path label with better styling"""
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            ux, uy = -dy/length, dx/length  # perpendicular unit vector
            ax.text(mx + offset * ux, my + offset * uy, text, 
                   ha='center', va='center', fontsize=7, color=colors['text'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['white'], 
                           alpha=0.95, edgecolor=colors['light_gray'], linewidth=0.5),
                   fontweight='normal')
    
    # Draw arrows and labels
    # X -> M (path a)
    draw_arrow(ax_diagram, x_coords[0], y_coords[0], x_coords[1], y_coords[1])
    add_path_label(ax_diagram, x_coords[0], y_coords[0], x_coords[1], y_coords[1], 
                  f"a: β = {result['a']:.3f}")
    
    # M -> Y (path b)
    draw_arrow(ax_diagram, x_coords[1], y_coords[1], x_coords[2], y_coords[2])
    add_path_label(ax_diagram, x_coords[1], y_coords[1], x_coords[2], y_coords[2], 
                  f"b: β = {result['b']:.3f}")
    
    # X -> Y (path c')
    draw_arrow(ax_diagram, x_coords[0], y_coords[0], x_coords[2], y_coords[2])
    add_path_label(ax_diagram, x_coords[0], y_coords[0], x_coords[2], y_coords[2], 
                  f"c′: β = {result['c']:.3f}")
    
    # Add legend with better styling
    handles = [
        mpatches.Patch(color=colors['primary'], label='LC integrity'),
        mpatches.Patch(color=colors['secondary'], label='α-power'),
        mpatches.Patch(color=outcome_color, label=outcome_label),
        mlines.Line2D([], [], color=colors['text'], linewidth=1.5, label='Path')
    ]
    labels = ['LC integrity', 'α-power', outcome_label, 'Path']
    ax_diagram.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                     fontsize=7, frameon=True, fancybox=True, shadow=False,
                     facecolor=colors['white'], edgecolor=colors['light_gray'])
    
    # Clean up diagram
    ax_diagram.set_xticks([])
    ax_diagram.set_yticks([])
    for spine in ax_diagram.spines.values():
        spine.set_visible(False)
    
    # Panel B: Bar plot with better styling
    bar_labels = ['Total (c)', 'Direct (c′)', 'Indirect (a×b)']
    bar_values = [result['c'], result["c'"], result['ab']]
    bar_colors = [outcome_color, colors['primary'], colors['secondary']]
    
    # Create bars with better spacing
    x_pos = np.arange(len(bar_labels))
    bars = ax_bars.bar(x_pos, bar_values, color=bar_colors, alpha=0.8, 
                      edgecolor=colors['text'], linewidth=0.5, width=0.6)
    
    # Add value labels with better styling
    for bar, val in zip(bars, bar_values):
        height = bar.get_height()
        ax_bars.text(bar.get_x() + bar.get_width()/2., height + np.sign(val)*0.01,
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=7, fontweight='normal',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colors['white'], 
                            alpha=0.9, edgecolor=colors['light_gray'], linewidth=0.3))
    
    # Add statistical summary with better styling
    indirect_text = f"Indirect effect: {result['ab']:.3f} (95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}])\nN = {result['n']} | Disease stage-adjusted"
    ax_bars.text(0.5, 0.95, indirect_text, transform=ax_bars.transAxes, 
                ha='center', va='top', fontsize=7, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['white'], 
                         alpha=0.95, edgecolor=colors['light_gray'], linewidth=0.5))
    
    # Style the bar plot
    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(bar_labels, fontsize=7, rotation=0)
    ax_bars.set_ylabel('Effect Size', fontsize=8, fontweight='normal')
    ax_bars.axhline(y=0, color=colors['text'], linewidth=0.8, alpha=0.6, linestyle='-')
    
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax_bars.spines[spine].set_visible(False)
    
    # Set spine colors and widths
    for spine in ['left', 'bottom']:
        ax_bars.spines[spine].set_color(colors['text'])
        ax_bars.spines[spine].set_linewidth(0.8)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Generate publication-quality mediation figures with disease stage adjustment"""
    print("Creating publication-quality disease stage-adjusted mediation figures...")
    
    # Load data with disease severity
    df = pd.read_csv('final_complete_disease_severity_mediation_data.csv')
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # SVF Count mediation
    result_svf = mediation_disease_stage_adjusted(df, 'SVF_count')
    if result_svf:
        create_mediation_figure_publication(result_svf, 'svf', 
                                          output_dir / 'mediation_svf_disease_stage_publication.png')
        create_mediation_figure_publication(result_svf, 'svf', 
                                          output_dir / 'mediation_svf_disease_stage_publication.pdf')
        print(f"SVF Count mediation (disease stage adjusted): N = {result_svf['n']}")
    
    # EE metric mediation
    result_ee = mediation_disease_stage_adjusted(df, 'exploitation_coherence_ratio')
    if result_ee:
        create_mediation_figure_publication(result_ee, 'ee', 
                                          output_dir / 'mediation_ee_disease_stage_publication.png')
        create_mediation_figure_publication(result_ee, 'ee', 
                                          output_dir / 'mediation_ee_disease_stage_publication.pdf')
        print(f"EE metric mediation (disease stage adjusted): N = {result_ee['n']}")
    
    print("\nCreated publication-quality disease stage-adjusted mediation figures:")
    print(" - output/mediation_svf_disease_stage_publication.(png|pdf)")
    print(" - output/mediation_ee_disease_stage_publication.(png|pdf)")

if __name__ == '__main__':
    main()

