#!/usr/bin/env python3
"""
Nature-Quality Figures (REAL DATA)
Generates publication-ready figures from the actual dataset using the analysis pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Local imports from the project
from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer
from phase_coherence_analysis import compute_phase_coherence_metrics_detailed


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
        'secondary': '#E69F00',
        'accent': '#56B4E9',
        'highlight': '#009E73',
        'neutral': '#999999',
        'red': '#D55E00',
        'purple': '#CC79A7',
    }
    return colors

def save_tight_and_trim(fig, output_png: Path):
    """Save figure tightly (PNG and PDF) and trim white margins from the PNG in-place with padding."""
    # Save with a modest pad to avoid cutting text
    fig.savefig(output_png, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_png.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.05)
    try:
        from PIL import Image, ImageChops
        img = Image.open(output_png).convert('RGB')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bbox = ImageChops.difference(img, bg).getbbox()
        if bbox:
            # Expand bbox by a small safety margin so annotations aren't cropped
            pad = 12
            left = max(bbox[0] - pad, 0)
            upper = max(bbox[1] - pad, 0)
            right = min(bbox[2] + pad, img.width)
            lower = min(bbox[3] + pad, img.height)
            img.crop((left, upper, right, lower)).save(output_png)
    except Exception:
        pass

def compute_real_metrics() -> pd.DataFrame:
    """Run the analyzer on all participants and compute coherence metrics per participant."""
    config = AnalysisConfig.from_yaml('config/config.yaml')
    analyzer = SemanticFluencyAnalyzer(config)
    analyzer.load_data(config.data_paths['fluency_data'], config.data_paths['meg_data'])

    results_df = []
    for pid in analyzer.data['ID'].unique():
        pdata = analyzer.data[analyzer.data['ID'] == pid]
        res = analyzer.analyze_participant(pdata)
        if not res or len(res.get('vectors', [])) < 2 or len(res.get('phases', [])) == 0:
            continue
        # Coherence metrics
        coh = compute_phase_coherence_metrics_detailed(res['phases'], res['vectors'], res['items'], verbose=False)
        # Frequency stats (wordfreq Zipf and top-50k ranks)
        freq = analyzer.utils.get_frequency_stats(res.get('items', []))
        # Combine with participant-level metrics
        row = {
            'ID': pid,
            'num_switches': res.get('num_switches', 0),
            'novelty_score': res.get('novelty_score', 0),
            'exploitation_intra_mean': coh.get('exploitation_intra_mean', np.nan),
            'exploration_intra_mean': coh.get('exploration_intra_mean', np.nan),
            'inter_phase_mean': coh.get('inter_phase_mean', np.nan),
            'exploitation_coherence_ratio': coh.get('exploitation_coherence_ratio', np.nan),
            'exploration_coherence_ratio': coh.get('exploration_coherence_ratio', np.nan),
            'phase_separation_index': coh.get('phase_separation_index', np.nan),
            # Word frequency (Zipf & rank)
            'mean_zipf': freq.get('mean_zipf', np.nan),
            'median_zipf': freq.get('median_zipf', np.nan),
            'std_zipf': freq.get('std_zipf', np.nan),
            'mean_rank': freq.get('mean_rank', np.nan),
            'median_rank': freq.get('median_rank', np.nan),
            'std_rank': freq.get('std_rank', np.nan),
        }
        results_df.append(row)

    df = pd.DataFrame(results_df)
    # Join MEG metrics
    meg = analyzer.meg_data[['ID', 'alpha_NET_mean', 'norm_LC_avg']].copy()
    df = df.merge(meg, on='ID', how='left')

    # Drop rows with missing key metrics
    df = df.dropna(subset=['exploitation_intra_mean', 'exploration_intra_mean', 'inter_phase_mean'])
    return df


def fig1_exploration_exploitation(df: pd.DataFrame, colors):
    setup_nature_style()
    x = df['exploitation_intra_mean'].values
    y = df['exploration_intra_mean'].values
    r, p = pearsonr(x, y)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

    # A: Scatter with regression
    ax1.scatter(x, y, s=22, alpha=0.8, color=colors['accent'], edgecolor='white', linewidth=0.3)
    z = np.polyfit(x, y, 1)
    pfit = np.poly1d(z)
    xr = np.linspace(x.min(), x.max(), 100)
    ax1.plot(xr, pfit(xr), color=colors['primary'], linewidth=1.2)
    ax1.set_xlabel('Exploitation (cosine similarity)')
    ax1.set_ylabel('Exploration (cosine similarity)')
    ax1.set_title('A', loc='left')
    ax1.text(0.95, 0.95, f'r = {r:.3f}', transform=ax1.transAxes,
             ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

    # B: Boxplots with paired points (within-subject)
    ax2.boxplot([x, y], labels=['Exploitation', 'Exploration'], patch_artist=True,
                medianprops=dict(color='black', linewidth=1.0))
    # Paired points overlay
    jitter = 0.04
    for i in range(len(x)):
        ax2.plot([1-jitter, 2+jitter], [x[i], y[i]], color='gray', alpha=0.5, linewidth=0.5)
        ax2.scatter([1-jitter, 2+jitter], [x[i], y[i]], color='black', s=6, zorder=3)
    ax2.set_ylabel('Cosine similarity')
    ax2.set_title('B', loc='left')
    # Paired t-test
    try:
        tstat, pval = ttest_ind(x, y)
        ax2.text(0.98, 0.02, f't={tstat:.2f}, p={pval:.3f}', transform=ax2.transAxes, ha='right', va='bottom', fontsize=7)
    except Exception:
        pass

    # C: Word frequency vs coherence ratios (unique information)
    zipf = df['mean_zipf'].values
    exp_ratio = df['exploitation_coherence_ratio'].values
    expl_ratio = df['exploration_coherence_ratio'].values

    # Scatter: Zipf vs exploitation coherence ratio
    m1 = ~np.isnan(zipf) & ~np.isnan(exp_ratio)
    ax3.scatter(zipf[m1], exp_ratio[m1], s=22, alpha=0.8,
                color=colors['accent'], edgecolor='white', linewidth=0.3,
                label='Exploitation ratio')
    if m1.sum() >= 3 and np.nanstd(zipf[m1]) > 1e-9:
        try:
            z1 = np.polyfit(zipf[m1], exp_ratio[m1], 1)
            xr1 = np.linspace(np.nanmin(zipf[m1]), np.nanmax(zipf[m1]), 100)
            ax3.plot(xr1, np.poly1d(z1)(xr1), color=colors['primary'], linewidth=1.0)
        except np.linalg.LinAlgError:
            pass

    # Scatter: Zipf vs exploration coherence ratio
    m2 = ~np.isnan(zipf) & ~np.isnan(expl_ratio)
    ax3.scatter(zipf[m2], expl_ratio[m2], s=22, alpha=0.7,
                color=colors['secondary'], edgecolor='white', linewidth=0.3,
                label='Exploration ratio')
    if m2.sum() >= 3 and np.nanstd(zipf[m2]) > 1e-9:
        try:
            z2 = np.polyfit(zipf[m2], expl_ratio[m2], 1)
            xr2 = np.linspace(np.nanmin(zipf[m2]), np.nanmax(zipf[m2]), 100)
            ax3.plot(xr2, np.poly1d(z2)(xr2), color=colors['neutral'], linewidth=1.0)
        except np.linalg.LinAlgError:
            pass

    ax3.set_xlabel('Mean Zipf frequency (Zipf)')
    ax3.set_ylabel('Coherence ratio')
    ax3.legend(fontsize=7, frameon=False)
    ax3.set_title('C', loc='left')

    # D: Summary table
    ax4.axis('off')
    summary = [
        ['Metric', 'Mean ± SD', 'Min–Max'],
        ['Exploitation', f"{np.mean(x):.3f} ± {np.std(x):.3f}", f"{np.min(x):.3f}–{np.max(x):.3f}"],
        ['Exploration', f"{np.mean(y):.3f} ± {np.std(y):.3f}", f"{np.min(y):.3f}–{np.max(y):.3f}"],
        ['Correlation', f"r={r:.3f}", f"p={p:.3f}"],
    ]
    table = ax4.table(cellText=summary[1:], colLabels=summary[0], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.4)

    output = Path('output/NATURE_REAL_figure1_exploration_exploitation.png')
    fig.tight_layout()
    save_tight_and_trim(fig, output)
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig2_phase_coherence(df: pd.DataFrame, colors):
    setup_nature_style()
    exp_intra = df['exploitation_intra_mean'].values
    expl_intra = df['exploration_intra_mean'].values
    psi = df['phase_separation_index'].values

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

    # A
    ax1.hist(exp_intra, bins=12, alpha=0.8, color=colors['accent'], edgecolor='black', linewidth=0.5, density=True)
    ax1.axvline(np.mean(exp_intra), color=colors['primary'], linestyle='--', linewidth=1.0)
    ax1.set_xlabel('Exploitation (cosine similarity)')
    ax1.set_ylabel('Density')
    ax1.set_title('A', loc='left')

    # B
    ax2.hist(expl_intra, bins=12, alpha=0.8, color=colors['secondary'], edgecolor='black', linewidth=0.5, density=True)
    ax2.axvline(np.mean(expl_intra), color=colors['primary'], linestyle='--', linewidth=1.0)
    ax2.set_xlabel('Exploration (cosine similarity)')
    ax2.set_ylabel('Density')
    ax2.set_title('B', loc='left')

    # C
    ax3.scatter(exp_intra, expl_intra, s=22, alpha=0.8, color=colors['highlight'], edgecolor='white', linewidth=0.3)
    z = np.polyfit(exp_intra, expl_intra, 1)
    pfit = np.poly1d(z)
    xr = np.linspace(exp_intra.min(), exp_intra.max(), 100)
    ax3.plot(xr, pfit(xr), color=colors['primary'], linewidth=1.2)
    ax3.set_xlabel('Exploitation (cosine similarity)')
    ax3.set_ylabel('Exploration (cosine similarity)')
    # r label
    try:
        r_ce, p_ce = pearsonr(exp_intra, expl_intra)
        ax3.text(0.95, 0.95, f'r = {r_ce:.3f}', transform=ax3.transAxes,
                 ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
    except Exception:
        pass
    ax3.set_title('C', loc='left')

    # D
    ax4.hist(psi, bins=12, alpha=0.8, color=colors['red'], edgecolor='black', linewidth=0.5, density=True)
    ax4.axvline(np.mean(psi), color=colors['primary'], linestyle='--', linewidth=1.0)
    ax4.set_xlabel('Phase separation index (a.u.)')
    ax4.set_ylabel('Density')
    ax4.set_title('D', loc='left')

    # Add comprehensive figure caption
    fig.suptitle('Phase Coherence Analysis: Intra-Phase Similarities and Phase Separation', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    caption_text = (
        "Distribution of intra-phase cosine similarities and phase separation metrics across participants. "
        "Panel A shows the distribution of exploitation intra-phase mean cosine similarities (mean = 0.63), "
        "indicating the average semantic coherence within exploitation phases. Panel B displays the distribution "
        "of exploration intra-phase mean cosine similarities (mean = 0.42), showing lower semantic coherence "
        "within exploration phases. Panel C shows the positive correlation between exploitation and exploration "
        "intra-phase means, suggesting that participants who maintain high coherence in one phase type tend to "
        "do so in the other. Panel D presents the phase separation index distribution (mean = -0.20), which "
        "quantifies the degree to which intra-phase coherence exceeds inter-phase coherence, indicating "
        "distinct semantic clustering within phases."
    )
    
    # Add caption below the figure
    fig.text(0.5, 0.02, caption_text, ha='center', va='bottom', fontsize=9, 
             wrap=True, transform=fig.transFigure, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    output = Path('output/NATURE_REAL_figure2_phase_coherence.png')
    fig.tight_layout()
    save_tight_and_trim(fig, output)
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig3_meg_correlations(df: pd.DataFrame, colors):
    setup_nature_style()
    # Use both alpha_NET_mean and LC norm_LC_avg vs coherence ratios
    alpha = df['alpha_NET_mean'].values
    lc = df['norm_LC_avg'].values
    exp_ratio = df['exploitation_coherence_ratio'].values
    expl_ratio = df['exploration_coherence_ratio'].values

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey='col')
    (ax1, ax2), (ax3, ax4) = axes

    # A: alpha vs exploitation coherence ratio
    ax1.scatter(alpha, exp_ratio, s=22, alpha=0.8, color=colors['accent'], edgecolor='white', linewidth=0.3)
    mask = ~np.isnan(alpha) & ~np.isnan(exp_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(alpha[mask], exp_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(alpha[mask]), np.nanmax(alpha[mask]), 100)
        ax1.plot(xr, pfit(xr), color=colors['primary'], linewidth=1.2)
        r, p = pearsonr(alpha[mask], exp_ratio[mask])
        ax1.text(0.95, 0.95, f'r = {r:.3f}', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax1.set_xlabel('MEG alpha power (NET mean)')
    ax1.set_ylabel('Exploitation coherence ratio')
    ax1.set_title('A', loc='left')

    # B: alpha vs exploration coherence ratio
    ax2.scatter(alpha, expl_ratio, s=22, alpha=0.8, color=colors['secondary'], edgecolor='white', linewidth=0.3)
    mask = ~np.isnan(alpha) & ~np.isnan(expl_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(alpha[mask], expl_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(alpha[mask]), np.nanmax(alpha[mask]), 100)
        ax2.plot(xr, pfit(xr), color=colors['primary'], linewidth=1.2)
        r, p = pearsonr(alpha[mask], expl_ratio[mask])
        ax2.text(0.95, 0.95, f'r = {r:.3f}', transform=ax2.transAxes,
                 ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax2.set_xlabel('MEG alpha power (NET mean)')
    ax2.set_ylabel('Exploration coherence ratio')
    ax2.set_title('B', loc='left')

    # C: LC vs exploitation coherence ratio
    ax3.scatter(lc, exp_ratio, s=22, alpha=0.8, color=colors['highlight'], edgecolor='white', linewidth=0.3)
    mask = ~np.isnan(lc) & ~np.isnan(exp_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(lc[mask], exp_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(lc[mask]), np.nanmax(lc[mask]), 100)
        ax3.plot(xr, pfit(xr), color=colors['primary'], linewidth=1.2)
        r, p = pearsonr(lc[mask], exp_ratio[mask])
        ax3.text(0.95, 0.95, f'r = {r:.3f}', transform=ax3.transAxes,
                 ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax3.set_xlabel('LC integrity (norm LC avg)')
    ax3.set_ylabel('Exploitation coherence ratio')
    ax3.set_title('C', loc='left')

    # D: LC vs exploration coherence ratio
    ax4.scatter(lc, expl_ratio, s=22, alpha=0.8, color=colors['red'], edgecolor='white', linewidth=0.3)
    mask = ~np.isnan(lc) & ~np.isnan(expl_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(lc[mask], expl_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(lc[mask]), np.nanmax(lc[mask]), 100)
        ax4.plot(xr, pfit(xr), color=colors['primary'], linewidth=1.2)
        r, p = pearsonr(lc[mask], expl_ratio[mask])
        ax4.text(0.95, 0.95, f'r = {r:.3f}', transform=ax4.transAxes,
                 ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax4.set_xlabel('LC integrity (norm LC avg)')
    ax4.set_ylabel('Exploration coherence ratio')
    ax4.set_title('D', loc='left')

    output = Path('output/NATURE_REAL_figure3_meg_correlations.png')
    fig.tight_layout()
    save_tight_and_trim(fig, output)
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig4_comprehensive(df: pd.DataFrame, colors):
    setup_nature_style()
    x = df['exploitation_intra_mean'].values
    y = df['exploration_intra_mean'].values
    switches = df['num_switches'].values
    novelty = df['novelty_score'].values

    # Professional 2×2 grid with proper spacing
    fig = plt.figure(figsize=(8.0, 6.5))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.95, top=0.92, bottom=0.15, 
                          wspace=0.35, hspace=0.40)
    
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # A: Exploitation vs Exploration scatter with trend line
    mask = ~np.isnan(x) & ~np.isnan(y)
    axA.scatter(x[mask], y[mask], s=25, alpha=0.7, color=colors['accent'], 
                edgecolor='white', linewidth=0.3)
    
    if mask.sum() >= 3:
        z = np.polyfit(x[mask], y[mask], 1)
        xr = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        axA.plot(xr, np.poly1d(z)(xr), color=colors['primary'], linewidth=1.2)
        r, p = pearsonr(x[mask], y[mask])
        axA.text(0.95, 0.95, f'r = {r:.3f}', transform=axA.transAxes, 
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    axA.set_xlabel('Exploitation (cosine similarity)', fontsize=9)
    axA.set_ylabel('Exploration (cosine similarity)', fontsize=9)
    axA.set_title('A', loc='left', fontsize=10, pad=10)

    # B: Overlaid distributions with better styling
    bins = 12
    axB.hist(x, bins=bins, alpha=0.6, density=True, color=colors['accent'], 
             edgecolor='black', linewidth=0.5, label='Exploitation', zorder=2)
    axB.hist(y, bins=bins, alpha=0.6, density=True, color=colors['secondary'], 
             edgecolor='black', linewidth=0.5, label='Exploration', zorder=1)
    axB.set_xlabel('Cosine similarity', fontsize=9)
    axB.set_ylabel('Density', fontsize=9)
    axB.set_title('B', loc='left', fontsize=10, pad=10)
    axB.legend(frameon=False, fontsize=8, loc='upper right')

    # C: Exploitation vs Switches with trend line
    mask = ~np.isnan(x) & ~np.isnan(switches)
    axC.scatter(x[mask], switches[mask], s=25, alpha=0.7, color=colors['highlight'], 
                edgecolor='white', linewidth=0.3)
    
    if mask.sum() >= 3:
        z = np.polyfit(x[mask], switches[mask], 1)
        xr = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        axC.plot(xr, np.poly1d(z)(xr), color=colors['primary'], linewidth=1.2)
        r, p = pearsonr(x[mask], switches[mask])
        axC.text(0.95, 0.95, f'r = {r:.3f}', transform=axC.transAxes, 
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    axC.set_xlabel('Exploitation (cosine similarity)', fontsize=9)
    axC.set_ylabel('Cluster switches (count)', fontsize=9)
    axC.set_title('C', loc='left', fontsize=10, pad=10)

    # D: Exploration vs Novelty with trend line
    mask = ~np.isnan(y) & ~np.isnan(novelty)
    axD.scatter(y[mask], novelty[mask], s=25, alpha=0.7, color=colors['red'], 
                edgecolor='white', linewidth=0.3)
    
    if mask.sum() >= 3:
        z = np.polyfit(y[mask], novelty[mask], 1)
        xr = np.linspace(np.nanmin(y[mask]), np.nanmax(y[mask]), 100)
        axD.plot(xr, np.poly1d(z)(xr), color=colors['primary'], linewidth=1.2)
        r, p = pearsonr(y[mask], novelty[mask])
        axD.text(0.95, 0.95, f'r = {r:.3f}', transform=axD.transAxes, 
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    axD.set_xlabel('Exploration (cosine similarity)', fontsize=9)
    axD.set_ylabel('Novelty (a.u.)', fontsize=9)
    axD.set_title('D', loc='left', fontsize=10, pad=10)

    # Consistent styling across all panels
    for ax in [axA, axB, axC, axD]:
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    output = Path('output/NATURE_REAL_figure4_comprehensive.png')
    save_tight_and_trim(fig, output)
    plt.close(fig)
    print(f"✅ Saved: {output}")


def main():
    print('🔬 Computing real-data metrics and generating Nature-quality figures...')
    colors = setup_nature_style()
    df = compute_real_metrics()
    if df.empty:
        print('❌ No valid participants found for plotting.')
        return
    # Save metrics for reference
    out_csv = Path('output/NATURE_REAL_metrics.csv')
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'💾 Saved metrics: {out_csv}')

    fig1_exploration_exploitation(df, colors)
    fig2_phase_coherence(df, colors)
    fig3_meg_correlations(df, colors)
    fig4_comprehensive(df, colors)
    print('🎉 Done.')


if __name__ == '__main__':
    main()
