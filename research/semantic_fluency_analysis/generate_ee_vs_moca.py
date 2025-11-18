#!/usr/bin/env python3
"""
Standalone script to generate E-E index vs MoCA comparison figure.
Uses existing computed metrics data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, linregress, t as t_dist

def setup_nature_style():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 3,
        'xtick.major.width': 0.5,
        'ytick.major.size': 3,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,
        'text.usetex': False,
        'mathtext.fontset': 'cm',
    })

    colors = {
        'primary': '#000000',
        'secondary': '#7B4F9E',
        'accent': '#56B4E9',
        'highlight': '#009E73',
        'neutral': '#999999',
        'red': '#D55E00',
        'purple': '#CC79A7',
    }
    return colors

def try_load_and_merge_demographics(base_df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort merge of available demographics/clinical CSVs on ID."""
    # First, try to get MoCA from comprehensive file specifically
    df = base_df.copy()
    
    # Priority: comprehensive file has more MoCA data
    comp_file = Path('research/semantic_fluency_analysis/final_comprehensive_mediation_data.csv')
    if not comp_file.exists():
        comp_file = Path('final_comprehensive_mediation_data.csv')
    
    if comp_file.exists():
        try:
            comp_df = pd.read_csv(comp_file)
            if 'ID' in comp_df.columns and 'cognitive_measure_2' in comp_df.columns:
                # Merge only the MoCA column
                comp_moca = comp_df[['ID', 'cognitive_measure_2']].dropna()
                df = df.merge(comp_moca, on='ID', how='left', suffixes=('', '_comp'))
                
                # Fill missing values in main with values from comprehensive
                if 'cognitive_measure_2_comp' in df.columns:
                    mask_missing = df['cognitive_measure_2'].isna() & df['cognitive_measure_2_comp'].notna()
                    if mask_missing.sum() > 0:
                        df.loc[mask_missing, 'cognitive_measure_2'] = df.loc[mask_missing, 'cognitive_measure_2_comp']
                        print(f'✅ Filled {mask_missing.sum()} missing MoCA values from comprehensive file')
                    df = df.drop(columns=['cognitive_measure_2_comp'], errors='ignore')
        except Exception as e:
            print(f'⚠️ Could not merge MoCA from comprehensive file: {e}')
    
    return df

def main():
    print('🔬 Generating E-E index vs MoCA comparison figure...')
    colors = setup_nature_style()
    
    # Load existing metrics
    metrics_path = Path('research/semantic_fluency_analysis/final_complete_disease_severity_mediation_data.csv')
    if not metrics_path.exists():
        metrics_path = Path('research/semantic_fluency_analysis/output/NATURE_REAL_metrics.csv')
    
    if not metrics_path.exists():
        print(f'❌ Could not find metrics file. Tried: {metrics_path}')
        return
    
    df = pd.read_csv(metrics_path)
    print(f'📊 Loaded {len(df)} participants from {metrics_path}')
    
    # Merge demographics
    df_demo = try_load_and_merge_demographics(df)
    
    # Calculate E-E index
    explo_ratio = df['exploitation_coherence_ratio'].to_numpy(float)
    expo_ratio = df['exploration_coherence_ratio'].to_numpy(float)
    mask_ee = np.isfinite(explo_ratio) & np.isfinite(expo_ratio) & (expo_ratio > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ee_index = (explo_ratio[mask_ee] / expo_ratio[mask_ee])
    ee_index = ee_index[np.isfinite(ee_index)]
    
    # Get corresponding IDs for matching
    ee_ids = df['ID'].values[mask_ee]
    ee_ids = ee_ids[np.isfinite(ee_index)]
    
    # Find MoCA column
    moca_col = next((c for c in ['MoCA', 'moca_score', 'MoCA_score', 'MoCA_total', 'cognitive_measure_2'] 
                     if c in df_demo.columns), None)
    
    if moca_col is None:
        print("⚠️ MoCA column not found in data")
        print(f"   Available columns: {[c for c in df_demo.columns if 'moca' in c.lower() or 'cognitive' in c.lower()]}")
        return
    
    print(f'✅ Found MoCA column: {moca_col}')
    
    # Extract MoCA scores matching the EE index participants
    moca_scores = []
    ee_index_matched = []
    ee_ids_matched = []
    
    for i, pid in enumerate(ee_ids):
        moca_val = df_demo[df_demo['ID'] == pid][moca_col].values
        if len(moca_val) > 0 and np.isfinite(moca_val[0]):
            moca_scores.append(float(moca_val[0]))
            ee_index_matched.append(ee_index[i])
            ee_ids_matched.append(pid)
    
    moca_scores = np.array(moca_scores)
    ee_index_matched = np.array(ee_index_matched)
    
    if len(moca_scores) < 3:
        print(f"⚠️ Insufficient data for correlation (n={len(moca_scores)})")
        return
    
    print(f'📈 Found {len(moca_scores)} participants with both E-E index and MoCA scores')
    
    # Compute correlation
    r, p = pearsonr(ee_index_matched, moca_scores)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    # Scatter plot
    ax.scatter(ee_index_matched, moca_scores, s=35, alpha=0.75, 
               color=colors['accent'], edgecolor='white', linewidth=0.5, zorder=3)
    
    # Regression line with confidence intervals
    if len(ee_index_matched) >= 3:
        slope, intercept, r_value, p_value, std_err = linregress(ee_index_matched, moca_scores)
        xr = np.linspace(ee_index_matched.min(), ee_index_matched.max(), 100)
        yr = slope * xr + intercept
        
        # Calculate confidence intervals
        n = len(ee_index_matched)
        y_pred = slope * ee_index_matched + intercept
        residuals = moca_scores - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se_residual = np.sqrt(mse)
        x_mean = np.mean(ee_index_matched)
        ssx = np.sum((ee_index_matched - x_mean)**2)
        se_pred = se_residual * np.sqrt(1.0/n + (xr - x_mean)**2 / ssx)
        t_crit = t_dist.ppf(0.975, n - 2)
        ci_upper = yr + t_crit * se_pred
        ci_lower = yr - t_crit * se_pred
        
        # Fill confidence interval
        ax.fill_between(xr, ci_lower, ci_upper, color='gray', alpha=0.25, zorder=0)
        # Plot regression line
        ax.plot(xr, yr, color=colors['primary'], linewidth=1.5, zorder=1)
    
    # Add correlation text
    p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
    ax.text(0.98, 0.98, f'r = {r:.3f}\np = {p_str}\nn = {len(moca_scores)}', 
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=0.8))
    
    # Labels and title
    ax.set_xlabel('E–E index (exploitation/exploration coherence)', fontsize=11, fontweight='normal')
    ax.set_ylabel('MoCA score', fontsize=11, fontweight='normal')
    ax.set_title('E–E Index vs MoCA Score', fontsize=12, fontweight='bold', pad=10)
    
    # Styling
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(False)
    
    # Save
    output_dir = Path('research/semantic_fluency_analysis/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'NATURE_REAL_ee_vs_moca.png'
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")
    print(f"   Correlation: r = {r:.4f}, p = {p:.4f}, n = {len(moca_scores)}")
    print(f"   E-E index range: {ee_index_matched.min():.2f} - {ee_index_matched.max():.2f}")
    print(f"   MoCA range: {moca_scores.min():.1f} - {moca_scores.max():.1f}")

if __name__ == '__main__':
    main()

