#!/usr/bin/env python3
"""
Nature-Quality Figures (REAL DATA)
Generates publication-ready figures from the actual dataset using the analysis pipeline.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ttest_ind, wilcoxon, mannwhitneyu, linregress, t as t_dist
import warnings
warnings.filterwarnings('ignore')

# Optional interactive hover (graceful fallback if not installed)
try:
    import mplcursors  # type: ignore
except Exception:
    mplcursors = None

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
        # Use a colorblind-friendly non-yellow secondary color (purple)
        'secondary': '#7B4F9E',
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


def try_load_and_merge_demographics(base_df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort merge of available demographics/clinical CSVs on ID."""
    candidates = [
        'final_complete_disease_severity_mediation_data.csv',
        'final_comprehensive_mediation_data.csv',
        'final_complete_mediation_data.csv',
        'final_clean_mediation_data.csv',
        'comprehensive_mediation_data.csv',
        'participant_ages_correct_46.csv',
        'participant_ages.csv',
    ]
    df = base_df.copy()
    for fname in candidates:
        p = Path(fname)
        if not p.exists():
            p2 = Path('research/semantic_fluency_analysis') / fname
            p = p2 if p2.exists() else p
        if p.exists():
            try:
                extra = pd.read_csv(p)
                if 'ID' in extra.columns:
                    before = len(df)
                    df = df.merge(extra, on='ID', how='left')
                    after = len(df)
                    print(f'🔗 Merged {fname}: {extra.shape} -> base {before} rows (now {after})')
            except Exception as e:
                print(f'⚠️ Could not merge {fname}: {e}')
    return df


def fig_demographic_sections(df: pd.DataFrame, colors):
    """Generate figures for demographics/clinical sections when columns are available."""
    out_dir = Path('output/figures/intermediate')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sex / Gender composition
    for col in ['Sex', 'Gender', 'sex', 'gender']:
        if col in df.columns:
            counts = df[col].dropna().astype(str).value_counts()
            fig, ax = plt.subplots(figsize=(4.0, 3.2))
            ax.bar(counts.index, counts.values, color=colors['accent'], edgecolor='black', linewidth=0.5)
            ax.set_title('Sex distribution')
            ax.set_ylabel('Participants')
            fig.tight_layout(pad=1.0)
            fig.savefig(out_dir / 'sex_distribution.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
            fig.savefig(out_dir / 'sex_distribution.pdf', dpi=300, bbox_inches='tight', pad_inches=0.4)
            plt.close(fig)
            break

    # 2) Age histogram
    if 'Age' in df.columns:
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        ax.hist(df['Age'].dropna(), bins=12, color=colors['accent'], edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_title('Age distribution')
        ax.set_xlabel('Age (years)'); ax.set_ylabel('Participants')
        mean_age = df['Age'].dropna().mean()
        ax.axvline(mean_age, color=colors['primary'], linestyle='--', linewidth=1.2, label=f'μ={mean_age:.1f}')
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(pad=1.0)
        fig.savefig(out_dir / 'age_distribution.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
        fig.savefig(out_dir / 'age_distribution.pdf', dpi=300, bbox_inches='tight', pad_inches=0.4)
        plt.close(fig)

    # 3) Education level
    for col in ['Education', 'education_level', 'EducationLevel']:
        if col in df.columns:
            counts = df[col].dropna().astype(str).value_counts()
            fig, ax = plt.subplots(figsize=(4.6, 3.2))
            ax.bar(counts.index, counts.values, color=colors['secondary'], edgecolor='black', linewidth=0.5)
            ax.set_title('Education level'); ax.set_ylabel('Participants')
            ax.tick_params(axis='x', rotation=20)
            fig.tight_layout(pad=1.0)
            fig.savefig(out_dir / 'education_level.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
            fig.savefig(out_dir / 'education_level.pdf', dpi=300, bbox_inches='tight', pad_inches=0.4)
            plt.close(fig)
            break

    # 4) Hoehn & Yahr stages
    for col in ['hoehn_yahr_score', 'HoehnYahr', 'H_Y_stage']:
        if col in df.columns:
            counts = df[col].dropna().round().astype(int).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(4.0, 3.2))
            ax.bar(counts.index.astype(str), counts.values, color=colors['highlight'], edgecolor='black', linewidth=0.5)
            ax.set_title('Hoehn & Yahr stages'); ax.set_xlabel('Stage'); ax.set_ylabel('Participants')
            fig.tight_layout(pad=1.0)
            fig.savefig(out_dir / 'hoehn_yahr_stages.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
            fig.savefig(out_dir / 'hoehn_yahr_stages.pdf', dpi=300, bbox_inches='tight', pad_inches=0.4)
            plt.close(fig)
            break

    # 5) MoCA scores
    for col in ['MoCA', 'moca_score', 'MoCA_score', 'MoCA_total']:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(4.0, 3.2))
            ax.hist(df[col].dropna(), bins=12, color=colors['neutral'], edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.set_title('MoCA scores'); ax.set_xlabel('MoCA'); ax.set_ylabel('Participants')
            mean_val = df[col].dropna().mean()
            ax.axvline(mean_val, color=colors['primary'], linestyle='--', linewidth=1.2, label=f'μ={mean_val:.1f}')
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout(pad=1.0)
            fig.savefig(out_dir / 'moca_scores.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
            fig.savefig(out_dir / 'moca_scores.pdf', dpi=300, bbox_inches='tight', pad_inches=0.4)
            plt.close(fig)
            break

    # 6) UPDRS Part III motor scores
    for col in ['UPDRS_III', 'UPDRS_part_III', 'UPDRS_part3', 'UPDRS3', 'UPDRS_PartIII']:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(4.0, 3.2))
            ax.hist(df[col].dropna(), bins=12, color=colors['red'], edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.set_title('UPDRS Part III'); ax.set_xlabel('Score'); ax.set_ylabel('Participants')
            mean_val = df[col].dropna().mean()
            ax.axvline(mean_val, color=colors['primary'], linestyle='--', linewidth=1.2, label=f'μ={mean_val:.1f}')
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout(pad=1.0)
            fig.savefig(out_dir / 'updrs_part3.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
            fig.savefig(out_dir / 'updrs_part3.pdf', dpi=300, bbox_inches='tight', pad_inches=0.4)
            plt.close(fig)
            break

    # 7) SVF performance (animals)
    for col in ['SVF_count', 'svf_count', 'SVF', 'animals_total']:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(4.0, 3.2))
            ax.hist(df[col].dropna(), bins=12, color=colors['accent'], edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.set_title('Semantic Verbal Fluency'); ax.set_xlabel('Words'); ax.set_ylabel('Participants')
            mean_val = df[col].dropna().mean()
            ax.axvline(mean_val, color=colors['primary'], linestyle='--', linewidth=1.2, label=f'μ={mean_val:.1f}')
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout(pad=1.0)
            fig.savefig(out_dir / 'svf_performance.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
            fig.savefig(out_dir / 'svf_performance.pdf', dpi=300, bbox_inches='tight', pad_inches=0.4)
            plt.close(fig)
            break

    print(f'📊 Saved demographics/clinical figures to {out_dir}')


def fig0_alpha_violin_swarm(df: pd.DataFrame, colors):
    """Pre-analysis descriptive plot: violin + swarm for MEG alpha power and LC neuromelanin."""
    setup_nature_style()
    # Hourhal-style tweaks
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
    alpha_series = df[['ID', 'alpha_NET_mean']].dropna()
    lc_series = df[['ID', 'norm_LC_avg']].dropna()
    alpha = alpha_series['alpha_NET_mean'].values
    lc = lc_series['norm_LC_avg'].values

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    ax1, ax2 = axes

    # Violin + swarm for alpha power
    parts = ax1.violinplot([alpha], showmeans=True, showextrema=False, widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor(colors['accent'])
        pc.set_alpha(0.35)
        pc.set_edgecolor('gray')
        pc.set_linewidth(0.8)
    parts['cmeans'].set_color(colors['primary'])
    parts['cmeans'].set_linewidth(1.6)
    # Swarm: jittered scatter
    rng = np.random.default_rng(42)
    jitter = (rng.random(len(alpha)) - 0.5) * 0.25
    sc1 = ax1.scatter(1 + jitter, alpha, s=28, color=colors['accent'], edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
    ax1.set_xticks([1])
    ax1.set_xticklabels(['MEG alpha_NET_mean'])
    ax1.set_ylabel('Alpha power (NET mean)')
    ax1.set_title('Alpha power distribution', loc='left')
    ax1.grid(False)
    for sp in ax1.spines.values():
        sp.set_linewidth(1.0)

    # Violin + swarm for LC neuromelanin integrity
    parts2 = ax2.violinplot([lc], showmeans=True, showextrema=False, widths=0.7)
    for pc in parts2['bodies']:
        pc.set_facecolor(colors['highlight'])
        pc.set_alpha(0.35)
        pc.set_edgecolor('gray')
        pc.set_linewidth(0.8)
    parts2['cmeans'].set_color(colors['primary'])
    parts2['cmeans'].set_linewidth(1.6)
    jitter2 = (rng.random(len(lc)) - 0.5) * 0.25
    sc2 = ax2.scatter(1 + jitter2, lc, s=28, color=colors['highlight'], edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['neuromelanin scores - LC'])
    ax2.set_ylabel('LC norm_LC_avg')
    ax2.set_title('LC neuromelanin distribution', loc='left')
    ax2.grid(False)
    for sp in ax2.spines.values():
        sp.set_linewidth(1.0)

    # Optional hover labels with participant IDs
    if mplcursors is not None:
        try:
            cursor1 = mplcursors.cursor(sc1, hover=True)
            @cursor1.connect("add")
            def on_add1(sel):
                idx = sel.target.index if hasattr(sel.target, 'index') else int(sel.index)
                pid = alpha_series['ID'].iloc[idx]
                sel.annotation.set(text=f'ID: {pid}\nalpha={alpha[idx]:.6f}')
            cursor2 = mplcursors.cursor(sc2, hover=True)
            @cursor2.connect("add")
            def on_add2(sel):
                idx = sel.target.index if hasattr(sel.target, 'index') else int(sel.index)
                pid = lc_series['ID'].iloc[idx]
                sel.annotation.set(text=f'ID: {pid}\nLC={lc[idx]:.4f}')
        except Exception:
            pass

    # Add simple descriptive stats on the right plot
    try:
        txt = f"Alpha: n={len(alpha)}, μ={np.mean(alpha):.4f}, σ={np.std(alpha, ddof=1):.4f}\nLC: n={len(lc)}, μ={np.mean(lc):.3f}, σ={np.std(lc, ddof=1):.3f}"
        # bottom-right of upper panel
        ax2.text(0.98, -0.22, txt, transform=ax2.transAxes, ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray'))
    except Exception:
        pass

    output = Path('output/NATURE_REAL_figure0_alpha_violin_swarm.png')
    fig.tight_layout(pad=1.6)
    fig.savefig(output, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig0b_coherence_violin_swarm(df: pd.DataFrame, colors):
    """Descriptive violin + swarm for coherence ratios (exploitation & exploration)."""
    setup_nature_style()
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
    exp_ratio_s = df[['ID', 'exploitation_coherence_ratio']].dropna()
    expl_ratio_s = df[['ID', 'exploration_coherence_ratio']].dropna()
    exp_ratio = exp_ratio_s['exploitation_coherence_ratio'].values
    expl_ratio = expl_ratio_s['exploration_coherence_ratio'].values

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    ax1, ax2 = axes

    # Exploitation ratio
    parts = ax1.violinplot([exp_ratio], showmeans=True, showextrema=False, widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor(colors['accent']); pc.set_alpha(0.35); pc.set_edgecolor('gray'); pc.set_linewidth(0.8)
    parts['cmeans'].set_color(colors['primary']); parts['cmeans'].set_linewidth(1.6)
    rng = np.random.default_rng(7)
    jitter = (rng.random(len(exp_ratio)) - 0.5) * 0.25
    sc1 = ax1.scatter(1 + jitter, exp_ratio, s=28, color=colors['accent'], edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
    ax1.set_xticks([1]); ax1.set_xticklabels(['Exploitation coherence ratio'])
    ax1.set_ylabel('Coherence ratio'); ax1.set_title('Exploitation', loc='left')
    ax1.grid(False); [sp.set_linewidth(1.0) for sp in ax1.spines.values()]

    # Exploration ratio
    parts2 = ax2.violinplot([expl_ratio], showmeans=True, showextrema=False, widths=0.7)
    for pc in parts2['bodies']:
        pc.set_facecolor(colors['secondary']); pc.set_alpha(0.35); pc.set_edgecolor('gray'); pc.set_linewidth(0.8)
    parts2['cmeans'].set_color(colors['primary']); parts2['cmeans'].set_linewidth(1.6)
    jitter2 = (rng.random(len(expl_ratio)) - 0.5) * 0.25
    sc2 = ax2.scatter(1 + jitter2, expl_ratio, s=28, color=colors['secondary'], edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
    ax2.set_xticks([1]); ax2.set_xticklabels(['Exploration coherence ratio'])
    ax2.set_ylabel('Coherence ratio'); ax2.set_title('Exploration', loc='left')
    ax2.grid(False); [sp.set_linewidth(1.0) for sp in ax2.spines.values()]

    # Optional hover labels
    if mplcursors is not None:
        try:
            c1 = mplcursors.cursor(sc1, hover=True)
            @c1.connect("add")
            def on_add1(sel):
                idx = int(sel.index); pid = exp_ratio_s['ID'].iloc[idx]
                sel.annotation.set(text=f'ID: {pid}\nratio={exp_ratio[idx]:.3f}')
            c2 = mplcursors.cursor(sc2, hover=True)
            @c2.connect("add")
            def on_add2(sel):
                idx = int(sel.index); pid = expl_ratio_s['ID'].iloc[idx]
                sel.annotation.set(text=f'ID: {pid}\nratio={expl_ratio[idx]:.3f}')
        except Exception:
            pass

    # Descriptive summary
    txt = (f"Exploitation: n={len(exp_ratio)}, μ={np.mean(exp_ratio):.3f}, σ={np.std(exp_ratio, ddof=1):.3f}\n"
           f"Exploration: n={len(expl_ratio)}, μ={np.mean(expl_ratio):.3f}, σ={np.std(expl_ratio, ddof=1):.3f}")
    ax2.text(0.98, -0.22, txt, transform=ax2.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray'))

    output = Path('output/NATURE_REAL_figure0b_coherence_violin_swarm.png')
    fig.tight_layout(pad=1.6)
    fig.savefig(output, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig0c_participant_table(df: pd.DataFrame):
    """Static table of per-participant values used in figures."""
    cols = ['ID', 'alpha_NET_mean', 'norm_LC_avg', 'exploitation_coherence_ratio', 'exploration_coherence_ratio']
    table_df = df[cols].copy().sort_values('ID')

    # Render table in matplotlib
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.set_title('Participant values used in figures', loc='left')
    cell_text = table_df.round(6).values.tolist()
    table = ax.table(cellText=cell_text, colLabels=table_df.columns.tolist(), cellLoc='center', loc='upper left')
    table.auto_set_font_size(False); table.set_fontsize(7)
    table.scale(1.0, 1.2)

    out = Path('output/NATURE_REAL_participant_values_table.png')
    fig.tight_layout(pad=1.2)
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {out}")


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
    exp_mean = np.mean(exp_intra)
    ax1.hist(exp_intra, bins=12, alpha=0.8, color=colors['accent'], edgecolor='black', linewidth=0.5, density=True)
    ax1.axvline(exp_mean, color=colors['primary'], linestyle='--', linewidth=1.0)
    ax1.set_xlabel('Exploitation (cosine similarity)')
    ax1.set_ylabel('Density')
    ax1.set_title('A', loc='left')
    # Add mean label
    ax1.text(0.95, 0.95, f'μ = {exp_mean:.2f}', transform=ax1.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

    # B
    expl_mean = np.mean(expl_intra)
    ax2.hist(expl_intra, bins=12, alpha=0.8, color=colors['secondary'], edgecolor='black', linewidth=0.5, density=True)
    ax2.axvline(expl_mean, color=colors['primary'], linestyle='--', linewidth=1.0)
    ax2.set_xlabel('Exploration (cosine similarity)')
    ax2.set_ylabel('Density')
    ax2.set_title('B', loc='left')
    # Add mean label
    ax2.text(0.95, 0.95, f'μ = {expl_mean:.2f}', transform=ax2.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

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
    psi_mean = np.mean(psi)
    ax4.hist(psi, bins=12, alpha=0.8, color=colors['red'], edgecolor='black', linewidth=0.5, density=True)
    ax4.axvline(psi_mean, color=colors['primary'], linestyle='--', linewidth=1.0)
    ax4.set_xlabel('Phase separation index (a.u.)')
    ax4.set_ylabel('Density')
    ax4.set_title('D', loc='left')
    # Add mean label
    ax4.text(0.95, 0.95, f'μ = {psi_mean:.2f}', transform=ax4.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

    # Add comprehensive figure caption
    fig.suptitle('Phase Coherence Analysis: Intra-Phase Similarities and Phase Separation', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    caption_text = (
        f"Distribution of intra-phase cosine similarities and phase separation metrics across participants. "
        f"Panel A shows the distribution of exploitation intra-phase mean cosine similarities (mean = {exp_mean:.2f}), "
        f"indicating the average semantic coherence within exploitation phases. Panel B displays the distribution "
        f"of exploration intra-phase mean cosine similarities (mean = {expl_mean:.2f}), showing lower semantic coherence "
        f"within exploration phases. Panel C shows the positive correlation between exploitation and exploration "
        f"intra-phase means, suggesting that participants who maintain high coherence in one phase type tend to "
        f"do so in the other. Panel D presents the phase separation index distribution (mean = {psi_mean:.2f}), which "
        f"quantifies the degree to which intra-phase coherence exceeds inter-phase coherence, indicating "
        f"distinct semantic clustering within phases."
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
    # Use consistent font sizes - match the setup_nature_style but larger
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Use both alpha_NET_mean and LC norm_LC_avg vs coherence ratios
    alpha = df['alpha_NET_mean'].values
    lc = df['norm_LC_avg'].values
    exp_ratio = df['exploitation_coherence_ratio'].values
    expl_ratio = df['exploration_coherence_ratio'].values

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey='col')
    (ax1, ax2), (ax3, ax4) = axes

    # Helper function to add confidence interval shading around regression line
    def add_confidence_interval(ax, x_data, y_data, x_line, y_line):
        """Add shaded confidence interval around regression line."""
        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        if mask.sum() < 3:
            return
        
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        # Fit regression to calculate residuals and standard errors
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        
        # Calculate predicted values
        y_pred = slope * x_clean + intercept
        
        # Calculate residuals and standard error of residuals
        residuals = y_clean - y_pred
        n = len(x_clean)
        mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
        se_residual = np.sqrt(mse)
        
        # Calculate standard error for prediction
        x_mean = np.mean(x_clean)
        ssx = np.sum((x_clean - x_mean)**2)
        
        # For each x in x_line, calculate confidence interval
        se_pred = se_residual * np.sqrt(1.0/n + (x_line - x_mean)**2 / ssx)
        
        # 95% confidence interval (t-distribution with n-2 degrees of freedom)
        t_crit = t_dist.ppf(0.975, n - 2)
        ci_upper = y_line + t_crit * se_pred
        ci_lower = y_line - t_crit * se_pred
        
        # Fill between upper and lower confidence bounds
        ax.fill_between(x_line, ci_lower, ci_upper, color='gray', alpha=0.25, zorder=0)
    
    # Helper function to compute Wilcoxon test and draw cut-off line
    def compute_wilcoxon_stats(x_data, y_data):
        """Compute Wilcoxon test statistics."""
        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        if mask.sum() < 3:
            return np.nan, np.nan
        
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        # Split data at median of x
        x_median = np.median(x_clean)
        below_median = y_clean[x_clean <= x_median]
        above_median = y_clean[x_clean > x_median]
        
        if len(below_median) > 0 and len(above_median) > 0:
            # Mann-Whitney U test (Wilcoxon rank-sum) for two independent groups
            try:
                stat, p_wilcoxon = mannwhitneyu(below_median, above_median, alternative='two-sided')
                return stat, p_wilcoxon
            except:
                return np.nan, np.nan
        return np.nan, np.nan
    
    # Helper function to add Wilcoxon cut-off line
    def add_wilcoxon_cutoff_line(ax, y_data):
        """Add horizontal cut-off line at median."""
        mask = ~np.isnan(y_data)
        if mask.sum() < 3:
            return
        # Use mean line to match reference styling
        y_mean = np.mean(y_data[mask])
        ax.axhline(y=y_mean, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)
    
    # Helper function to add combined statistics legend
    def add_stats_legend(ax, r, p_pearson, rho, p_spearman, w_stat, p_wilcoxon, x_pos=0.98, y_pos=0.95):
        """Add a single legend box with all three statistics."""
        # Build legend text
        lines = [f'r={r:.3f}, p={p_pearson:.3f}']
        lines.append(f'ρ={rho:.3f}, p={p_spearman:.3f}')
        if not np.isnan(w_stat) and not np.isnan(p_wilcoxon):
            lines.append(f'W={w_stat:.1f}, p={p_wilcoxon:.3f}')
        
        legend_text = '\n'.join(lines)
        
        # Create single legend box - use consistent font family (Arial) and smaller size to avoid distraction
        ax.text(x_pos, y_pos, legend_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.8))

    # A: alpha vs exploitation coherence ratio
    ax1.scatter(alpha, exp_ratio, s=28, alpha=0.9, color=colors['accent'], edgecolor='white', linewidth=0.4, zorder=3)
    mask = ~np.isnan(alpha) & ~np.isnan(exp_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(alpha[mask], exp_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(alpha[mask]), np.nanmax(alpha[mask]), 100)
        yr = pfit(xr)
        # Add confidence interval shading first (behind line)
        add_confidence_interval(ax1, alpha, exp_ratio, xr, yr)
        ax1.plot(xr, yr, color=colors['primary'], linewidth=1.6, zorder=2)
        r, p_pearson = pearsonr(alpha[mask], exp_ratio[mask])
        rho, p_spearman = spearmanr(alpha[mask], exp_ratio[mask])
        w_stat, p_wilcoxon = compute_wilcoxon_stats(alpha, exp_ratio)
        # Add combined statistics legend in one box
        add_stats_legend(ax1, r, p_pearson, rho, p_spearman, w_stat, p_wilcoxon)
        add_wilcoxon_cutoff_line(ax1, exp_ratio)
    ax1.set_xlabel('MEG alpha_NET_mean')
    ax1.set_ylabel('Exploitation Coherence Ratio')
    ax1.set_title('A', loc='left', fontweight='bold')

    # B: alpha vs exploration coherence ratio
    ax2.scatter(alpha, expl_ratio, s=28, alpha=0.9, color=colors['secondary'], edgecolor='white', linewidth=0.4, zorder=3)
    mask = ~np.isnan(alpha) & ~np.isnan(expl_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(alpha[mask], expl_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(alpha[mask]), np.nanmax(alpha[mask]), 100)
        yr = pfit(xr)
        # Add confidence interval shading first (behind line)
        add_confidence_interval(ax2, alpha, expl_ratio, xr, yr)
        ax2.plot(xr, yr, color=colors['primary'], linewidth=1.6, zorder=2)
        r, p_pearson = pearsonr(alpha[mask], expl_ratio[mask])
        rho, p_spearman = spearmanr(alpha[mask], expl_ratio[mask])
        w_stat, p_wilcoxon = compute_wilcoxon_stats(alpha, expl_ratio)
        # Add combined statistics legend in one box
        add_stats_legend(ax2, r, p_pearson, rho, p_spearman, w_stat, p_wilcoxon)
        add_wilcoxon_cutoff_line(ax2, expl_ratio)
    ax2.set_xlabel('MEG alpha_NET_mean')
    ax2.set_ylabel('Exploration Coherence Ratio')
    ax2.set_title('B', loc='left', fontweight='bold')

    # C: LC vs exploitation coherence ratio
    ax3.scatter(lc, exp_ratio, s=28, alpha=0.9, color=colors['highlight'], edgecolor='white', linewidth=0.4, zorder=3)
    mask = ~np.isnan(lc) & ~np.isnan(exp_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(lc[mask], exp_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(lc[mask]), np.nanmax(lc[mask]), 100)
        yr = pfit(xr)
        # Add confidence interval shading first (behind line)
        add_confidence_interval(ax3, lc, exp_ratio, xr, yr)
        ax3.plot(xr, yr, color=colors['primary'], linewidth=1.6, zorder=2)
        r, p_pearson = pearsonr(lc[mask], exp_ratio[mask])
        rho, p_spearman = spearmanr(lc[mask], exp_ratio[mask])
        w_stat, p_wilcoxon = compute_wilcoxon_stats(lc, exp_ratio)
        # Add combined statistics legend in one box
        add_stats_legend(ax3, r, p_pearson, rho, p_spearman, w_stat, p_wilcoxon)
        add_wilcoxon_cutoff_line(ax3, exp_ratio)
    ax3.set_xlabel('neuromelanin scores - LC')
    ax3.set_ylabel('Exploitation Coherence Ratio')
    ax3.set_title('C', loc='left', fontweight='bold')

    # D: LC vs exploration coherence ratio
    ax4.scatter(lc, expl_ratio, s=28, alpha=0.9, color=colors['red'], edgecolor='white', linewidth=0.4, zorder=3)
    mask = ~np.isnan(lc) & ~np.isnan(expl_ratio)
    if mask.sum() >= 3:
        z = np.polyfit(lc[mask], expl_ratio[mask], 1)
        pfit = np.poly1d(z)
        xr = np.linspace(np.nanmin(lc[mask]), np.nanmax(lc[mask]), 100)
        yr = pfit(xr)
        # Add confidence interval shading first (behind line)
        add_confidence_interval(ax4, lc, expl_ratio, xr, yr)
        ax4.plot(xr, yr, color=colors['primary'], linewidth=1.6, zorder=2)
        r, p_pearson = pearsonr(lc[mask], expl_ratio[mask])
        rho, p_spearman = spearmanr(lc[mask], expl_ratio[mask])
        w_stat, p_wilcoxon = compute_wilcoxon_stats(lc, expl_ratio)
        # Add combined statistics legend in one box
        add_stats_legend(ax4, r, p_pearson, rho, p_spearman, w_stat, p_wilcoxon)
        add_wilcoxon_cutoff_line(ax4, expl_ratio)
    ax4.set_xlabel('neuromelanin scores - LC')
    ax4.set_ylabel('Exploration Coherence Ratio')
    ax4.set_title('D', loc='left', fontweight='bold')

    # Add publication-style figure legend/caption
    # REMOVED: figure caption and legend per user request
    # legend_text = (
    #     "Figure 3. Neurobiological correlates of coherence ratios. A–B: Relationship between MEG alpha power "
    #     "(NET mean) and the exploitation (A) and exploration (B) coherence ratios. C–D: Relationship between "
    #     "locus coeruleus (LC) neuromelanin integrity (norm LC avg) and the exploitation (C) and exploration (D) "
    #     "coherence ratios. Points denote participants; the solid black line is the least-squares fit and the "
    #     "shaded band is the 95% CI. Insets report Pearson r and Spearman ρ (and Wilcoxon rank-sum p) for each panel."
    # )
    # # Append panel sample sizes (n)
    # try:
    #     n_a = int((~np.isnan(alpha) & ~np.isnan(exp_ratio)).sum())
    #     n_b = int((~np.isnan(alpha) & ~np.isnan(expl_ratio)).sum())
    #     n_c = int((~np.isnan(lc) & ~np.isnan(exp_ratio)).sum())
    #     n_d = int((~np.isnan(lc) & ~np.isnan(expl_ratio)).sum())
    #     legend_text += f" Panel Ns: A n={n_a}, B n={n_b}, C n={n_c}, D n={n_d}."
    # except Exception:
    #     pass
    # fig.text(0.5, 0.01, legend_text, ha='center', va='bottom', fontsize=9,
    #          bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='gray'))

    output = Path('output/NATURE_REAL_figure3_meg_correlations.png')
    fig.tight_layout(pad=2.0)  # Increased padding for white space around figure
    # Save with more padding for edits and higher DPI
    fig.savefig(output, bbox_inches='tight', pad_inches=0.3, facecolor='white', dpi=600)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.3, facecolor='white', dpi=600)
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig4_comprehensive(df: pd.DataFrame, colors):
    setup_nature_style()
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

    # Figure with 3 scatter plots in a single row (conceptual diagram removed)
    fig = plt.figure(figsize=(16.0, 5.5))
    # Create gridspec: 3 panels evenly distributed with equal widths and balanced spacing
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.96, top=0.92, bottom=0.15, 
                          wspace=0.35, width_ratios=[1, 1, 1])
    
    # Data plots: A, C, D (removed Panel B - cosine similarity distributions)
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

    # B: Exploitation vs Switches with trend line (renumbered from C to B)
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

    # D: Exploration vs Novelty with trend line
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
    
    # Consistent styling - grid removed, spines already set above
    for ax in [axA, axC, axD]:
        ax.grid(False, alpha=0.3)

    output = Path('output/NATURE_REAL_figure4_comprehensive.png')
    # Save with minimal padding to reduce white space
    fig.savefig(output, dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), format='svg', bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")
    print(f"✅ Saved: {output.with_suffix('.pdf')}")
    print(f"✅ Saved: {output.with_suffix('.svg')}")


def fig4_schematic_only():
    """Create schematic diagram only as SVG"""
    setup_nature_style()
    
    # Create figure for schematic only
    fig = plt.figure(figsize=(5.0, 7.0))
    ax_concept = fig.add_subplot(111)
    ax_concept.axis('off')
    
    # Replicate SVG layout exactly with absolute positioning
    words = ["Dog", "Cat", "Rabbit", "Hamster", "Bird", "Horse", "Cow"]
    
    TEXT_X = 40 / 420
    TEXT_FONTSIZE = 16
    
    word_y_svg = [55, 100, 145, 190, 275, 320, 365]
    word_y_norm = [1 - (y / 520) for y in word_y_svg]
    
    for word, y_norm in zip(words, word_y_norm):
        ax_concept.text(TEXT_X, y_norm, word, fontsize=TEXT_FONTSIZE, 
                       fontfamily='Arial', ha='left', va='center',
                       transform=ax_concept.transAxes)
    
    BRACKET_X = 180 / 420
    BRACKET_CAP_LEFT = 165 / 420
    BRACKET_LINE_WIDTH = 2
    
    bracket1_top_svg = 35
    bracket1_bottom_svg = 210
    bracket1_top_norm = 1 - (bracket1_top_svg / 520)
    bracket1_bottom_norm = 1 - (bracket1_bottom_svg / 520)
    bracket1_mid_norm = (bracket1_top_norm + bracket1_bottom_norm) / 2
    
    ax_concept.plot([BRACKET_X, BRACKET_X], [bracket1_top_norm, bracket1_bottom_norm], 
                   color='black', linewidth=BRACKET_LINE_WIDTH, 
                   transform=ax_concept.transAxes)
    ax_concept.plot([BRACKET_X, BRACKET_CAP_LEFT], 
                   [bracket1_top_norm, bracket1_top_norm], 
                   color='black', linewidth=BRACKET_LINE_WIDTH, 
                   transform=ax_concept.transAxes)
    ax_concept.plot([BRACKET_X, BRACKET_CAP_LEFT], 
                   [bracket1_bottom_norm, bracket1_bottom_norm], 
                   color='black', linewidth=BRACKET_LINE_WIDTH, 
                   transform=ax_concept.transAxes)
    
    exploit1_x = 200 / 420
    exploit1_y_norm = bracket1_mid_norm
    ax_concept.text(exploit1_x, exploit1_y_norm, 'Exploitation', 
                   fontsize=16, fontfamily='Arial', ha='left', va='center',
                   rotation=-90, transform=ax_concept.transAxes)
    
    pets_x = 245 / 420
    pets_y_norm = bracket1_mid_norm
    ax_concept.text(pets_x, pets_y_norm, 'Pets', 
                   fontsize=16, fontfamily='Arial', ha='left', va='center',
                   transform=ax_concept.transAxes)
    
    arrow_y_svg = 232.5
    arrow_y_norm = 1 - (arrow_y_svg / 520)
    arrow_x_start = 245 / 420
    arrow_x_end = 200 / 420
    
    arrowhead_size = 0.025
    arrowhead_tip_x = arrow_x_end - arrowhead_size * 2
    arrowhead_base_x = arrow_x_end
    arrowhead_y_top = arrow_y_norm + arrowhead_size
    arrowhead_y_bottom = arrow_y_norm - arrowhead_size
    
    line_end_x = arrow_x_end + arrowhead_size * 0.5
    ax_concept.plot([arrow_x_start, line_end_x], [arrow_y_norm, arrow_y_norm],
                   color='#3C66FF', linewidth=8,
                   transform=ax_concept.transAxes, zorder=1)
    
    arrowhead = Polygon([(arrowhead_tip_x, arrow_y_norm),
                        (arrowhead_base_x, arrowhead_y_top),
                        (arrowhead_base_x, arrowhead_y_bottom)],
                       closed=True, facecolor='#3C66FF', edgecolor='#3C66FF',
                       transform=ax_concept.transAxes, zorder=2)
    ax_concept.add_patch(arrowhead)
    
    patch_x = 265 / 420
    patch_y_norm = arrow_y_norm
    ax_concept.text(patch_x, patch_y_norm, 'Patch Switch', 
                   fontsize=16, fontfamily='Arial', ha='left', va='center',
                   transform=ax_concept.transAxes)
    
    bracket2_top_svg = 255
    bracket2_bottom_svg = 395
    bracket2_top_norm = 1 - (bracket2_top_svg / 520)
    bracket2_bottom_norm = 1 - (bracket2_bottom_svg / 520)
    bracket2_mid_norm = (bracket2_top_norm + bracket2_bottom_norm) / 2
    
    ax_concept.plot([BRACKET_X, BRACKET_X], [bracket2_top_norm, bracket2_bottom_norm], 
                   color='black', linewidth=BRACKET_LINE_WIDTH, 
                   transform=ax_concept.transAxes)
    ax_concept.plot([BRACKET_X, BRACKET_CAP_LEFT], 
                   [bracket2_top_norm, bracket2_top_norm], 
                   color='black', linewidth=BRACKET_LINE_WIDTH, 
                   transform=ax_concept.transAxes)
    ax_concept.plot([BRACKET_X, BRACKET_CAP_LEFT], 
                   [bracket2_bottom_norm, bracket2_bottom_norm], 
                   color='black', linewidth=BRACKET_LINE_WIDTH, 
                   transform=ax_concept.transAxes)
    
    exploit2_x = 200 / 420
    exploit2_y_norm = bracket2_mid_norm
    ax_concept.text(exploit2_x, exploit2_y_norm, 'Exploitation', 
                   fontsize=16, fontfamily='Arial', ha='left', va='center',
                   rotation=-90, transform=ax_concept.transAxes)
    
    farm_x = 245 / 420
    farm_y_norm = bracket2_mid_norm
    ax_concept.text(farm_x, farm_y_norm, 'Farm Animals', 
                   fontsize=16, fontfamily='Arial', ha='left', va='center',
                   transform=ax_concept.transAxes)
    
    output = Path('output/figures/schematic_diagram.svg')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format='svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig4_panels_only(df: pd.DataFrame, colors):
    """Create four data panels only as SVG"""
    setup_nature_style()
    x = df['exploitation_intra_mean'].values
    y = df['exploration_intra_mean'].values
    switches = df['num_switches'].values
    novelty = df['novelty_score'].values
    
    def add_line_with_ci(ax, x_vals: np.ndarray, y_vals: np.ndarray, color: str):
        mask_xy = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        if mask_xy.sum() < 3:
            return
        x_clean = x_vals[mask_xy]
        y_clean = y_vals[mask_xy]
        z = np.polyfit(x_clean, y_clean, 1)
        xr = np.linspace(np.nanmin(x_clean), np.nanmax(x_clean), 200)
        yr = np.poly1d(z)(xr)
        ax.plot(xr, yr, color=colors['primary'], linewidth=1.6, zorder=2)
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
    
    fig = plt.figure(figsize=(9.0, 7.0))
    gs = fig.add_gridspec(2, 2, left=0.12, right=0.96, top=0.96, bottom=0.14, 
                          wspace=0.32, hspace=0.38)
    
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    
    # A: Exploitation vs Exploration
    mask = ~np.isnan(x) & ~np.isnan(y)
    axA.scatter(x[mask], y[mask], s=35, alpha=0.75, color=colors['accent'], 
                edgecolor='white', linewidth=0.5, zorder=3)
    if mask.sum() >= 3:
        r, p = pearsonr(x[mask], y[mask])
        add_line_with_ci(axA, x[mask], y[mask], colors['primary'])
        p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
        axA.text(0.98, 0.98, f'r = {r:.3f}\np = {p_str}', transform=axA.transAxes, 
                ha='right', va='top', fontsize=11, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.8))
    axA.set_xlabel('Exploitation (cosine similarity)', fontsize=14, fontweight='normal')
    axA.set_ylabel('Exploration (cosine similarity)', fontsize=14, fontweight='normal')
    axA.set_title('A', loc='left', fontsize=14, fontweight='bold', pad=8)
    axA.tick_params(axis='both', which='major', labelsize=11, width=0.8, length=4)
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    axA.spines['left'].set_linewidth(1.2)
    axA.spines['bottom'].set_linewidth(1.2)
    
    # B: Overlaid distributions
    bins = 14
    axB.hist(x, bins=bins, alpha=0.65, density=True, color=colors['accent'], 
             edgecolor='white', linewidth=0.8, label='Exploitation', zorder=2)
    axB.hist(y, bins=bins, alpha=0.65, density=True, color=colors['secondary'], 
             edgecolor='white', linewidth=0.8, label='Exploration', zorder=1)
    axB.set_xlabel('Cosine similarity', fontsize=14, fontweight='normal')
    axB.set_ylabel('Normalized frequency', fontsize=14, fontweight='normal')
    axB.set_title('B', loc='left', fontsize=14, fontweight='bold', pad=8)
    axB.legend(frameon=True, fontsize=11, loc='upper right', 
              framealpha=0.95, edgecolor='gray', fancybox=False)
    axB.tick_params(axis='both', which='major', labelsize=11, width=0.8, length=4)
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    axB.spines['left'].set_linewidth(1.2)
    axB.spines['bottom'].set_linewidth(1.2)
    
    # C: Exploitation vs Switches
    mask = ~np.isnan(x) & ~np.isnan(switches)
    axC.scatter(x[mask], switches[mask], s=35, alpha=0.75, color=colors['highlight'], 
                edgecolor='white', linewidth=0.5, zorder=3)
    if mask.sum() >= 3:
        r, p = pearsonr(x[mask], switches[mask])
        add_line_with_ci(axC, x[mask], switches[mask], colors['primary'])
        p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
        axC.text(0.98, 0.98, f'r = {r:.3f}\np = {p_str}', transform=axC.transAxes, 
                ha='right', va='top', fontsize=11, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.8))
    axC.set_xlabel('Exploitation (cosine similarity)', fontsize=14, fontweight='normal')
    axC.set_ylabel('Cluster switches (count)', fontsize=14, fontweight='normal')
    axC.set_title('C', loc='left', fontsize=14, fontweight='bold', pad=8)
    axC.tick_params(axis='both', which='major', labelsize=11, width=0.8, length=4)
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    axC.spines['left'].set_linewidth(1.2)
    axC.spines['bottom'].set_linewidth(1.2)
    
    # D: Exploration vs Novelty
    mask = ~np.isnan(y) & ~np.isnan(novelty)
    axD.scatter(y[mask], novelty[mask], s=35, alpha=0.75, color=colors['red'], 
                edgecolor='white', linewidth=0.5, zorder=3)
    if mask.sum() >= 3:
        r, p = pearsonr(y[mask], novelty[mask])
        add_line_with_ci(axD, y[mask], novelty[mask], colors['primary'])
        p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.2e}'
        axD.text(0.98, 0.98, f'r = {r:.3f}\np = {p_str}', transform=axD.transAxes, 
                ha='right', va='top', fontsize=11, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.8))
    axD.set_xlabel('Exploration (cosine similarity)', fontsize=14, fontweight='normal')
    axD.set_ylabel('Novelty (a.u.)', fontsize=14, fontweight='normal')
    axD.set_title('D', loc='left', fontsize=14, fontweight='bold', pad=8)
    axD.tick_params(axis='both', which='major', labelsize=11, width=0.8, length=4)
    axD.spines['top'].set_visible(False)
    axD.spines['right'].set_visible(False)
    axD.spines['left'].set_linewidth(1.2)
    axD.spines['bottom'].set_linewidth(1.2)
    
    for ax in [axA, axB, axC, axD]:
        ax.grid(False, alpha=0.3)

    output = Path('output/figures/NATURE_REAL_figure4_panels.svg')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format='svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig_behavior_performance(df: pd.DataFrame, colors):
    """Create a figure that cross-checks paragraph metrics from the current dataset."""
    setup_nature_style()
    # Merge in demographics/clinical if available
    df_demo = try_load_and_merge_demographics(df)
    # Core metrics
    expl_intra = df['exploration_intra_mean'].to_numpy(float)
    explo_ratio = df['exploitation_coherence_ratio'].to_numpy(float)
    expo_ratio = df['exploration_coherence_ratio'].to_numpy(float)
    explo_intra = df['exploitation_intra_mean'].to_numpy(float)
    # E–E proxy index: exploitation / exploration coherence ratio (dimensionless > 1 means more exploitation)
    mask_ee = np.isfinite(explo_ratio) & np.isfinite(expo_ratio) & (expo_ratio > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ee_index = (explo_ratio[mask_ee] / expo_ratio[mask_ee])
    ee_index = ee_index[np.isfinite(ee_index)]
    # Proportion exploitation-dominant: EE index > 1
    if len(ee_index) > 0:
        prop_exploit = float((ee_index > 1.0).sum()) / float(len(ee_index))
    else:
        prop_exploit = np.nan
    # Optional correlations (best-effort)
    svf = df_demo['SVF_count'].to_numpy(float) if 'SVF_count' in df_demo.columns else np.array([])
    moca_col = next((c for c in ['MoCA', 'moca_score', 'MoCA_score', 'MoCA_total'] if c in df_demo.columns), None)
    updrs_col = next((c for c in ['UPDRS_III', 'UPDRS_part_III', 'UPDRS_part3', 'UPDRS3', 'UPDRS_PartIII'] if c in df_demo.columns), None)
    # Compute correlations
    def pearson_safe(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3:
            r, p = pearsonr(x[mask], y[mask])
            return r, p, int(mask.sum())
        return np.nan, np.nan, 0
    r_moca, p_moca, n_moca = (np.nan, np.nan, 0)
    if moca_col is not None and 'SVF_count' in df_demo.columns:
        r_moca, p_moca, n_moca = pearson_safe(df_demo['SVF_count'].to_numpy(float), df_demo[moca_col].to_numpy(float))
    r_updrs, p_updrs, n_updrs = (np.nan, np.nan, 0)
    if updrs_col is not None:
        r_updrs, p_updrs, n_updrs = pearson_safe(ee_index, df_demo[updrs_col].to_numpy(float)[:len(ee_index)])

    # Build figure
    fig = plt.figure(figsize=(8.4, 6.2))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.95, top=0.92, bottom=0.15, wspace=0.35, hspace=0.40)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # A: Distribution of exploration intra-phase mean (cosine similarity)
    vals = expl_intra[np.isfinite(expl_intra)]
    axA.hist(vals, bins=12, color=colors['accent'], edgecolor='black', linewidth=0.5, alpha=0.85, density=False)
    muA, sdA = float(np.mean(vals)), float(np.std(vals, ddof=1))
    axA.axvline(muA, color=colors['primary'], linestyle='--', linewidth=1.2)
    axA.set_xlabel('Exploration intra-phase mean (cosine)')
    axA.set_ylabel('Participants')
    axA.set_title('A', loc='left')
    axA.text(0.98, 0.95, f'μ={muA:.2f}, σ={sdA:.2f}', transform=axA.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    # B: Distribution of E–E index (proxy: exploitation / exploration coherence ratios)
    axB.hist(ee_index, bins=12, color=colors['secondary'], edgecolor='black', linewidth=0.5, alpha=0.85)
    muB = float(np.mean(ee_index)) if len(ee_index) else np.nan
    sdB = float(np.std(ee_index, ddof=1)) if len(ee_index) > 1 else np.nan
    rngB = (float(np.nanmin(ee_index)) if len(ee_index) else np.nan,
            float(np.nanmax(ee_index)) if len(ee_index) else np.nan)
    axB.axvline(muB, color=colors['primary'], linestyle='--', linewidth=1.2)
    axB.set_xlabel('E–E index proxy (exploitation/exploration coherence)')
    axB.set_ylabel('Participants')
    axB.set_title('B', loc='left')
    axB.text(0.98, 0.95, f'μ={muB:.2f}, σ={sdB:.2f}\nrange={rngB[0]:.2f}–{rngB[1]:.2f}',
             transform=axB.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    # C: Proportion exploitation-dominant (EE > 1)
    axC.axis('off')
    if np.isfinite(prop_exploit):
        axC.text(0.0, 0.8, 'Exploitation-dominant proportion', fontsize=10, fontweight='bold', transform=axC.transAxes)
        axC.text(0.0, 0.6, f'{prop_exploit*100:.1f}% (EE>1)', fontsize=10, transform=axC.transAxes)
        axC.text(0.0, 0.45, f'n={len(ee_index)}', fontsize=9, transform=axC.transAxes)
    else:
        axC.text(0.0, 0.8, 'Exploitation-dominant proportion', fontsize=10, fontweight='bold', transform=axC.transAxes)
        axC.text(0.0, 0.6, 'EE index not available', fontsize=10, transform=axC.transAxes)

    # D: Correlations (best-effort annotations)
    axD.axis('off')
    lines = []
    lines.append(f'Exploration intra mean: μ={muA:.2f}, σ={sdA:.2f}')
    lines.append(f'E–E index (proxy): μ={muB:.2f}, σ={sdB:.2f}, range={rngB[0]:.2f}–{rngB[1]:.2f}')
    if n_moca > 0 and np.isfinite(r_moca):
        lines.append(f'SVF vs MoCA: r={r_moca:.2f}, p={p_moca:.02f}, n={n_moca}')
    else:
        lines.append('SVF vs MoCA: data unavailable')
    if n_updrs > 0 and np.isfinite(r_updrs):
        lines.append(f'E–E vs UPDRS-III: r={r_updrs:.2f}, p={p_updrs:.02f}, n={n_updrs}')
    else:
        lines.append('E–E vs UPDRS-III: data unavailable')
    axD.text(0.0, 1.0, '\n'.join(lines), ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='gray'))
    # Save
    output = Path('output/NATURE_REAL_behavior_performance.png')
    fig.tight_layout(pad=1.6)
    fig.savefig(output, dpi=600, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")


def fig_ee_vs_moca(df: pd.DataFrame, colors):
    """Create scatter plot comparing E-E index with MoCA scores."""
    setup_nature_style()
    # Merge in demographics/clinical if available
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
        return
    
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
    ax.grid(False, alpha=0.3)
    
    # Save
    output = Path('output/NATURE_REAL_ee_vs_moca.png')
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {output}")
    print(f"   Correlation: r = {r:.4f}, p = {p:.4f}, n = {len(moca_scores)}")


def main():
    print('🔬 Computing real-data metrics and generating Nature-quality figures...')
    colors = setup_nature_style()
    df = compute_real_metrics()
    # Best-effort demographics merge (for downstream figures)
    df_demo = try_load_and_merge_demographics(df)
    if df.empty:
        print('❌ No valid participants found for plotting.')
        return
    # Save metrics for reference
    out_csv = Path('output/NATURE_REAL_metrics.csv')
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'💾 Saved metrics: {out_csv}')

    # New pre-analysis descriptive figure (violin + swarm)
    fig0_alpha_violin_swarm(df, colors)
    fig0b_coherence_violin_swarm(df, colors)
    fig0c_participant_table(df)
    # Demographic/clinical sections (as available)
    fig_demographic_sections(df_demo, colors)
    fig1_exploration_exploitation(df, colors)
    fig2_phase_coherence(df, colors)
    fig3_meg_correlations(df, colors)
    fig4_comprehensive(df, colors)
    fig_behavior_performance(df, colors)
    fig_ee_vs_moca(df, colors)

    # Combine descriptive plots into a single PDF
    try:
        from PIL import Image
        combined = Path('output/coherence_results.pdf')
        pages = [
            Path('output/NATURE_REAL_figure0_alpha_violin_swarm.png'),
            Path('output/NATURE_REAL_figure0b_coherence_violin_swarm.png'),
            Path('output/NATURE_REAL_participant_values_table.png'),
        ]
        imgs = []
        for p in pages:
            if p.exists():
                imgs.append(Image.open(p).convert('RGB'))
        if imgs:
            first, rest = imgs[0], imgs[1:]
            first.save(combined, save_all=True, append_images=rest)
            print(f'📄 Combined descriptive PDF saved: {combined}')
    except Exception as e:
        print(f'⚠️ Could not combine descriptive PDF: {e}')
    print('🎉 Done.')


if __name__ == '__main__':
    main()
