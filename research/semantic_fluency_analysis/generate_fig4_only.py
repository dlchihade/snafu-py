#!/usr/bin/env python3
"""Generate Figure 4 only, loading data directly from CSV to avoid spaCy dependency"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, linregress, t as t_dist
from create_nature_quality_figures_real import setup_nature_style

def fig4_comprehensive():
    """Generate Figure 4 with zero values excluded"""
    colors = setup_nature_style()
    
    # Load data directly from CSV
    df = pd.read_csv('final_complete_disease_severity_mediation_data.csv')
    
    # Exclude participants with zero exploitation_intra_mean (no exploitation phases)
    df_filtered = df[df['exploitation_intra_mean'] > 0].copy()
    print(f'📊 Figure 4: Excluding {len(df) - len(df_filtered)} participant(s) with zero exploitation_intra_mean')
    print(f'   Sample size: {len(df_filtered)} (was {len(df)})')
    
    x = df_filtered['exploitation_intra_mean'].values
    y = df_filtered['exploration_intra_mean'].values
    switches = df_filtered['num_switches'].values
    novelty = df_filtered['novelty_score'].values

    # Local helper to add regression line with 95% CI shading
    def add_line_with_ci(ax, x_vals: np.ndarray, y_vals: np.ndarray, color: str):
        mask_xy = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        if mask_xy.sum() < 3:
            return
        x_clean = x_vals[mask_xy]
        y_clean = y_vals[mask_xy]
        # Fit
        z = np.polyfit(x_clean, y_clean, 1)
        xr = np.linspace(np.nanmin(x_clean), np.nanmax(x_clean), 200)
        yr = np.poly1d(z)(xr)
        ax.plot(xr, yr, color=colors['primary'], linewidth=1.6, zorder=2)
        # CI using classic formula
        slope, intercept, r_val, p_val, std_err = linregress(x_clean, y_clean)
        y_pred = slope * x_clean + intercept
        resid = y_clean - y_pred
        n = len(x_clean)
        if n <= 2:
            return
        mse = np.sum(resid**2) / (n - 2)
        se_res = np.sqrt(mse)
        x_mean = np.mean(x_clean)
        ssx = np.sum((x_clean - x_mean)**2)
        se_pred = se_res * np.sqrt(1.0/n + (xr - x_mean)**2 / ssx) if ssx > 0 else np.full_like(xr, np.nan)
        t_crit = t_dist.ppf(0.975, n - 2)
        ci_upper = yr + t_crit * se_pred
        ci_lower = yr - t_crit * se_pred
        ax.fill_between(xr, ci_lower, ci_upper, color='gray', alpha=0.25, zorder=1)
        return r_val

    # Figure with 3 scatter plots in a single row
    fig = plt.figure(figsize=(16.0, 5.5))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.96, top=0.92, bottom=0.15, 
                          wspace=0.35, width_ratios=[1, 1, 1])
    
    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[0, 2])

    # A: Exploitation vs Exploration scatter with trend line
    mask = ~np.isnan(x) & ~np.isnan(y)
    axA.scatter(x[mask], y[mask], s=35, alpha=0.75, color=colors['accent'], 
                edgecolor='white', linewidth=0.5, zorder=3)
    
    if mask.sum() >= 3:
        r, p = pearsonr(x[mask], y[mask])
        add_line_with_ci(axA, x[mask], y[mask], colors['primary'])
        p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
        axA.text(0.98, 0.98, f'r = {r:.3f}\np = {p_str}\nn = {mask.sum()}', transform=axA.transAxes, 
                ha='right', va='top', fontsize=15, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.8))
    
    axA.set_xlabel('Exploitation (cosine similarity)', fontsize=18, fontweight='normal')
    axA.set_ylabel('Exploration (cosine similarity)', fontsize=18, fontweight='normal')
    axA.set_title('A', loc='left', fontsize=20, fontweight='bold', pad=8)
    axA.tick_params(axis='both', which='major', labelsize=15, width=0.8, length=4)
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    axA.spines['left'].set_linewidth(1.2)
    axA.spines['bottom'].set_linewidth(1.2)

    # B: Exploitation vs Switches with trend line
    mask = ~np.isnan(x) & ~np.isnan(switches)
    axC.scatter(x[mask], switches[mask], s=35, alpha=0.75, color=colors['highlight'], 
                edgecolor='white', linewidth=0.5, zorder=3)
    
    if mask.sum() >= 3:
        r, p = pearsonr(x[mask], switches[mask])
        add_line_with_ci(axC, x[mask], switches[mask], colors['primary'])
        p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
        axC.text(0.98, 0.98, f'r = {r:.3f}\np = {p_str}\nn = {mask.sum()}', transform=axC.transAxes, 
                ha='right', va='top', fontsize=15, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.8))
    
    axC.set_xlabel('Exploitation (cosine similarity)', fontsize=18, fontweight='normal')
    axC.set_ylabel('Cluster switches (count)', fontsize=18, fontweight='normal')
    axC.set_title('B', loc='left', fontsize=20, fontweight='bold', pad=8)
    axC.tick_params(axis='both', which='major', labelsize=15, width=0.8, length=4)
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    axC.spines['left'].set_linewidth(1.2)
    axC.spines['bottom'].set_linewidth(1.2)

    # C: Exploration vs Novelty with trend line
    mask = ~np.isnan(y) & ~np.isnan(novelty)
    axD.scatter(y[mask], novelty[mask], s=35, alpha=0.75, color=colors['red'], 
                edgecolor='white', linewidth=0.5, zorder=3)
    
    if mask.sum() >= 3:
        r, p = pearsonr(y[mask], novelty[mask])
        add_line_with_ci(axD, y[mask], novelty[mask], colors['primary'])
        p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
        axD.text(0.98, 0.98, f'r = {r:.3f}\np = {p_str}\nn = {mask.sum()}', transform=axD.transAxes, 
                ha='right', va='top', fontsize=15, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.8))
    
    axD.set_xlabel('Exploration (cosine similarity)', fontsize=18, fontweight='normal')
    axD.set_ylabel('Novelty (a.u.)', fontsize=18, fontweight='normal')
    axD.set_title('C', loc='left', fontsize=20, fontweight='bold', pad=8)
    axD.tick_params(axis='both', which='major', labelsize=15, width=0.8, length=4)
    axD.spines['top'].set_visible(False)
    axD.spines['right'].set_visible(False)
    axD.spines['left'].set_linewidth(1.2)
    axD.spines['bottom'].set_linewidth(1.2)
    
    # Save figure
    output = Path('output/NATURE_REAL_figure4_comprehensive')
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix('.png'), bbox_inches='tight', pad_inches=0.05, facecolor='white', dpi=600)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.05, facecolor='white', dpi=600)
    fig.savefig(output.with_suffix('.svg'), format='svg', bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")

if __name__ == '__main__':
    fig4_comprehensive()
