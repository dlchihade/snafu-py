#!/usr/bin/env python3
"""
Create a corrected, publication-ready 4-panel (A/B/C/D) schematic of the
semantic-fluency pipeline.

Every definition and number in this figure is a faithful read-out of the ACTUAL
analysis code (CODE IS GROUND TRUTH). Nothing is hardcoded: the example sequence
is pushed through the real pipeline at runtime by importing:
  - src.analyzer.SemanticFluencyAnalyzer  (spaCy vectors, consecutive similarity,
    _identify_phases / _should_transition phase detection)
  - phase_coherence_analysis.compute_intra_phase_metrics / compute_inter_phase_metrics
    / compute_phase_coherence_metrics_detailed (intra/inter/ratios/PSI)

No analysis code is modified; the real logic is imported read-only.
"""

import os
# Harmless: fall back to a writable matplotlib cache dir if the default is not writable.
if not os.environ.get('MPLCONFIGDIR'):
    _default_mpl = os.path.expanduser('~/.matplotlib')
    if not os.access(os.path.dirname(_default_mpl), os.W_OK):
        os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
# Guard against duplicate-OpenMP native aborts (numpy/sklearn/spaCy vs matplotlib).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import io
import sys
import contextlib
from pathlib import Path

import numpy as np

# Make the real analysis modules importable (run from project root).
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# NOTE: import the real analysis stack (numpy/spaCy/sklearn) BEFORE matplotlib.
# The reverse order triggers a native OpenMP conflict (Abort trap 6) on macOS.
from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer
from phase_coherence_analysis import (
    compute_intra_phase_metrics,
    compute_inter_phase_metrics,
    compute_phase_coherence_metrics_detailed,
)
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ----------------------------------------------------------------------------
# Publication-quality style settings
# ----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'font.weight': 'normal',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Professional color scheme
colors = {
    'data': '#4a90e2',
    'preprocessing': '#1f77b4',
    'embedding': '#2e86ab',
    'similarity': '#ff7f0e',
    'phase': '#ff9500',
    'intra': '#2ca02c',
    'inter': '#28a745',
    'coherence': '#20c997',
    'metrics': '#17a2b8',
    'output': '#6c757d',
    'light_gray': '#f5f5f5',
    'exploitation': '#2ca02c',
    'exploration': '#ff7f0e',
    'inter_phase': '#2e6fb0',
    'neutral': '#7f7f7f',
    'border': '#cccccc',
    'threshold': '#d62728',
}

# Standardized publication-ready constants
PUB_STYLE = {
    'figure_width': 10,
    'figure_height': 7.5,
    'panel_title_y': 0.97,
    'panel_title_fontsize': 14,
    'panel_title_color': '#000000',
    'step_title_fontsize': 11,
    'step_title_color': '#000000',
    'title_to_box_gap': 0.05,
    'content_fontsize': 10,
    'box_padding': 0.12,
    'box_linewidth': 1.2,
    'box_alpha': 0.25,
    'y_lim_bottom': 0.05,
    'y_lim_top': 1.0,
    'section_spacing': 0.15,
}


def add_content_box(ax, x, y, text, width=3.8, height=None, color='light_gray',
                    alpha=None, edgecolor=None, fontsize=None, ha='center', va='top',
                    flat_box=False):
    """Helper function to add a standardized content box (kept for reuse)."""
    if alpha is None:
        alpha = PUB_STYLE['box_alpha']
    if edgecolor is None:
        edgecolor = colors.get('border', colors['neutral'])
    if fontsize is None:
        fontsize = PUB_STYLE['content_fontsize']

    # Calculate height if not provided (estimate based on text lines)
    if height is None:
        num_lines = text.count('\n') + 1
        height = max(0.4, num_lines * 0.08 + 0.15)

    if flat_box:
        # Flat box with thin gray border, white background
        box = FancyBboxPatch(
            (x - width / 2, y - height), width, height,
            boxstyle="square,pad=0",
            facecolor='white',
            edgecolor='#d0d0d0',
            linewidth=0.5,
            alpha=1.0,
            zorder=1
        )
    else:
        box = FancyBboxPatch(
            (x - width / 2, y - height), width, height,
            boxstyle=f"round,pad={PUB_STYLE['box_padding']}",
            facecolor=colors.get(color, colors['light_gray']),
            edgecolor=edgecolor,
            linewidth=PUB_STYLE['box_linewidth'],
            alpha=alpha,
            zorder=1
        )
    ax.add_patch(box)
    # Position text inside box, accounting for box padding
    if flat_box:
        text_y = y - 0.02
        padding = 0.05
    else:
        text_y = y - PUB_STYLE['box_padding'] - 0.02
        padding = PUB_STYLE['box_padding']
    ax.text(x, text_y, text, ha=ha, va=va, fontsize=fontsize,
            color='#000000', zorder=2)
    return box


# ----------------------------------------------------------------------------
# STEP 1 - Run the REAL pipeline on the example sequence and collect numbers.
# ----------------------------------------------------------------------------
def compute_real_pipeline_numbers(example_words):
    """Push `example_words` through the real analysis code and return every number
    the figure needs. No values are invented here."""
    config = AnalysisConfig.from_yaml('config/config.yaml')
    analyzer = SemanticFluencyAnalyzer(config)

    # spaCy 300-d vectors (lowercased) + consecutive cosine similarities,
    # exactly like the real pipeline.
    vectors, valid_words, _ = analyzer.spacy_optimizer.get_vectors_batch(example_words)
    similarities = analyzer.utils.calculate_similarities_vectorized(vectors)

    # Real phase detection (_identify_phases / _should_transition).
    phases = analyzer._identify_phases(
        similarities, valid_words, vectors, config.similarity_threshold
    )

    # Intra / inter / ratios / PSI via the real coherence-analysis functions.
    # (These functions print progress unconditionally; silence to keep our stdout clean.)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        intra = compute_intra_phase_metrics(phases, vectors, verbose=False)
        inter = compute_inter_phase_metrics(phases, vectors, verbose=False)
        full = compute_phase_coherence_metrics_detailed(
            phases, vectors, valid_words, verbose=False
        )

    # Per-phase pairwise cosine matrices + upper-triangle values (as the code does).
    for p in phases:
        pv = np.array(p['vectors'])
        m = cosine_similarity(pv)
        p['sim_matrix'] = m
        p['upper_tri'] = m[np.triu_indices_from(m, k=1)]

    # Normalized centroids in phase (sequence) order -> inter-phase matrix.
    centroids = []
    for p in phases:
        c = np.mean(p['vectors'], axis=0)
        n = np.linalg.norm(c)
        centroids.append(c / n if n > 0 else c)
    centroids = np.array(centroids)
    inter_matrix = centroids @ centroids.T  # normalized => dot == cosine

    threshold = config.similarity_threshold
    min_phase_length = config.min_phase_length

    return {
        'config': config,
        'words': valid_words,
        'similarities': np.asarray(similarities, dtype=float),
        'phases': phases,
        'inter_matrix': inter_matrix,
        'mu_exploit_intra': float(intra['exploitation_intra_mean']),
        'mu_explore_intra': float(intra['exploration_intra_mean']),
        'mu_inter': float(inter['inter_phase_mean']),
        'rho_exploit': float(full['exploitation_coherence_ratio']),
        'rho_explore': float(full['exploration_coherence_ratio']),
        'psi': float(full['phase_separation_index']),
        'threshold': float(threshold),
        'min_phase_length': int(min_phase_length),
        'n_inter_pairs': len(inter['inter_phase_similarities']),
    }


# ----------------------------------------------------------------------------
# Small drawing helper for annotated similarity heatmaps.
# ----------------------------------------------------------------------------
def _draw_heatmap(ax, matrix, labels, cmap, title, vmin=0.0, vmax=1.0,
                  title_color='#000000', label_fs=8, cell_fs=8):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=label_fs)
    ax.set_yticklabels(labels, fontsize=label_fs)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            txt_color = 'white' if val >= (vmin + 0.62 * (vmax - vmin)) else '#111111'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=cell_fs, color=txt_color)
    if title:
        ax.set_title(title, fontsize=9, color=title_color, fontweight='bold', pad=6)
    return im


# ----------------------------------------------------------------------------
# STEP 2 - Build the A/B/C/D composite figure from the real numbers.
# ----------------------------------------------------------------------------
def build_composite_figure(R):
    words = R['words']
    sims = R['similarities']
    phases = R['phases']
    tau = R['threshold']

    exploitation_color = colors['exploitation']
    exploration_color = colors['exploration']

    def phase_color(ptype):
        return exploitation_color if ptype == 'Exploitation' else exploration_color

    # Pivot (boundary) word indices where one phase's end == next phase's start.
    pivots = set()
    for a, b in zip(phases[:-1], phases[1:]):
        if a['end'] == b['start']:
            pivots.add(a['end'])

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        'Figure 1.  Semantic-fluency pipeline: similarity, phase detection, and coherence metrics',
        fontsize=16, fontweight='bold', y=0.985,
    )

    outer = fig.add_gridspec(
        2, 2, hspace=0.34, wspace=0.20,
        left=0.055, right=0.975, top=0.905, bottom=0.055,
    )

    # Panel letters + titles (figure coordinates).
    def panel_header(x, y, letter, title):
        fig.text(x, y, letter, fontsize=18, fontweight='bold', color='#000000',
                 ha='left', va='center')
        fig.text(x + 0.022, y, title, fontsize=13, fontweight='bold',
                 color='#000000', ha='left', va='center')

    panel_header(0.055, 0.923, 'A', 'Data processing and phase detection')
    panel_header(0.545, 0.923, 'B', 'Intra-phase similarity')
    panel_header(0.055, 0.470, 'C', 'Inter-phase similarity')
    panel_header(0.545, 0.470, 'D', 'Coherence ratios and Phase Separation Index')

    # ========================================================================
    # PANEL A: consecutive similarity plot + rule + phase-classification strip
    # ========================================================================
    gsA = outer[0, 0].subgridspec(3, 1, height_ratios=[2.1, 0.55, 1.25], hspace=0.55)
    axA_plot = fig.add_subplot(gsA[0])
    axA_rule = fig.add_subplot(gsA[1]); axA_rule.axis('off')
    axA_strip = fig.add_subplot(gsA[2]); axA_strip.axis('off')

    # --- consecutive cosine similarity line/marker plot ---
    x = np.arange(len(sims))
    axA_plot.plot(x, sims, '-', color='#555555', linewidth=1.4, zorder=1)
    for xi, s in zip(x, sims):
        c = exploitation_color if s > tau else exploration_color
        axA_plot.plot(xi, s, 'o', color=c, markersize=9,
                      markeredgecolor='white', markeredgewidth=1.0, zorder=3)
    axA_plot.axhline(tau, ls='--', color=colors['threshold'], linewidth=1.4, zorder=2)
    axA_plot.text(len(sims) - 1, tau + 0.03, f'$\\tau = {tau:g}$',
                  color=colors['threshold'], fontsize=10, ha='right', va='bottom')
    trans_labels = [f'{words[i]}\u2192{words[i + 1]}' for i in range(len(sims))]
    axA_plot.set_xticks(x)
    axA_plot.set_xticklabels(trans_labels, rotation=45, ha='right', fontsize=7.5)
    axA_plot.set_ylim(0, 1.32)
    axA_plot.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axA_plot.set_ylabel('consecutive\ncosine similarity', fontsize=9)
    axA_plot.grid(axis='y', ls=':', color='#dddddd', linewidth=0.7)
    for spine in ['top', 'right']:
        axA_plot.spines[spine].set_visible(False)
    # small legend for marker meaning
    axA_plot.plot([], [], 'o', color=exploitation_color, label=r'$\cos>\tau$ (Exploitation)')
    axA_plot.plot([], [], 'o', color=exploration_color, label=r'$\cos\leq\tau$ (Exploration)')
    axA_plot.legend(loc='lower left', fontsize=7.0, frameon=True, framealpha=0.9,
                    handletextpad=0.3, borderpad=0.3)

    # --- cosine formula + classification rule ---
    axA_rule.set_xlim(0, 1); axA_rule.set_ylim(0, 1)
    axA_rule.text(
        0.5, 0.78,
        r'$\cos(\theta)=\dfrac{v_i \cdot v_{i+1}}{\|v_i\|\,\|v_{i+1}\|}$',
        ha='center', va='center', fontsize=11)
    axA_rule.text(
        0.5, 0.18,
        r'Exploitation if $\cos(w_i,w_{i+1}) > \tau$   |   '
        r'Exploration if $\cos(w_i,w_{i+1}) \leq \tau$   '
        r'(ties $\rightarrow$ Exploration),  $\tau=%.1f$' % tau,
        ha='center', va='center', fontsize=8.6)

    # --- phase classification output strip (contiguous spans with overlap) ---
    axA_strip.set_xlim(-0.6, len(words) - 0.4)
    axA_strip.set_ylim(0, 1)
    axA_strip.text((len(words) - 1) / 2.0, 0.96, 'Phase classification output',
                   ha='center', va='top', fontsize=9, fontweight='bold')

    # word tape
    for i, w in enumerate(words):
        is_pivot = i in pivots
        axA_strip.text(
            i, 0.66, w, ha='center', va='center', fontsize=7.6,
            fontweight='bold' if is_pivot else 'normal',
            color='#000000',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='#fff3cd' if is_pivot else '#f2f2f2',
                      edgecolor='#b8860b' if is_pivot else '#cccccc',
                      linewidth=1.1 if is_pivot else 0.6))

    # phase bars, exploitation lane vs exploration lane so overlap is visible on x
    lane_y = {'Exploitation': 0.40, 'Exploration': 0.20}
    bar_h = 0.11
    for p in phases:
        c = phase_color(p['type'])
        x0 = p['start'] - 0.42
        w_bar = (p['end'] - p['start']) + 0.84
        y0 = lane_y[p['type']]
        axA_strip.add_patch(FancyBboxPatch(
            (x0, y0), w_bar, bar_h, boxstyle='round,pad=0.02',
            facecolor=c, edgecolor=c, alpha=0.35, linewidth=1.2, zorder=1))
        axA_strip.text((p['start'] + p['end']) / 2.0, y0 + bar_h / 2.0,
                       f"{p['type']} [{p['end'] - p['start'] + 1}]",
                       ha='center', va='center', fontsize=7.3, color='#333333',
                       fontweight='bold', zorder=2)
    axA_strip.text(
        (len(words) - 1) / 2.0, 0.02,
        f"boundary/pivot words (highlighted) are shared between adjacent phases; "
        f"each emitted phase has \u22653 words (min_phase_length gate = {R['min_phase_length']})",
        ha='center', va='bottom', fontsize=6.8, color='#555555', style='italic')

    # ========================================================================
    # PANEL B: intra-phase similarity (two example heatmaps + pooled means)
    # ========================================================================
    gsB = outer[0, 1].subgridspec(2, 2, height_ratios=[3.0, 0.85],
                                  wspace=0.55, hspace=0.75)
    axB_exploit = fig.add_subplot(gsB[0, 0])
    axB_explore = fig.add_subplot(gsB[0, 1])
    axB_txt = fig.add_subplot(gsB[1, :]); axB_txt.axis('off')

    ex_phase = next(p for p in phases if p['type'] == 'Exploitation')
    er_phase = next(p for p in phases if p['type'] == 'Exploration')

    _draw_heatmap(axB_exploit, ex_phase['sim_matrix'], ex_phase['items'],
                  cmap='Greens', title='Within exploitation',
                  title_color=exploitation_color)
    _draw_heatmap(axB_explore, er_phase['sim_matrix'], er_phase['items'],
                  cmap='Oranges', title='Within exploration',
                  title_color=exploration_color)

    ex_ut = float(np.mean(ex_phase['upper_tri']))
    er_ut = float(np.mean(er_phase['upper_tri']))
    axB_txt.set_xlim(0, 1); axB_txt.set_ylim(0, 1)
    axB_txt.text(
        0.5, 0.86,
        'Both = pairwise cosine similarity, upper triangle (k=1) of the phase matrix',
        ha='center', va='center', fontsize=8.4, style='italic', color='#444444')
    axB_txt.text(
        0.25, 0.42,
        f"example upper-tri mean\n= {ex_ut:.3f}",
        ha='center', va='center', fontsize=8.0, color=exploitation_color)
    axB_txt.text(
        0.75, 0.42,
        f"example upper-tri mean\n= {er_ut:.3f}",
        ha='center', va='center', fontsize=8.0, color=exploration_color)
    axB_txt.text(
        0.5, 0.02,
        r'pooled  $\mu_{\mathrm{exploit\,intra}} = %.3f$        '
        r'$\mu_{\mathrm{explore\,intra}} = %.3f$'
        % (R['mu_exploit_intra'], R['mu_explore_intra']),
        ha='center', va='bottom', fontsize=9.6, fontweight='bold')

    # ========================================================================
    # PANEL C: inter-phase similarity (centroid matrix + definitions)
    # ========================================================================
    gsC = outer[1, 0].subgridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30)
    axC_heat = fig.add_subplot(gsC[0])
    axC_txt = fig.add_subplot(gsC[1]); axC_txt.axis('off')

    phase_labels = []
    for k, p in enumerate(phases, start=1):
        tag = 'Expt' if p['type'] == 'Exploitation' else 'Expl'
        phase_labels.append(f"$\\Phi_{{{k}}}$\n{tag}")
    _draw_heatmap(axC_heat, R['inter_matrix'], phase_labels,
                  cmap='Blues', title='Normalized-centroid similarity',
                  title_color=colors['inter_phase'], label_fs=7.5)

    axC_txt.set_xlim(0, 1); axC_txt.set_ylim(0, 1)
    axC_txt.text(0.02, 0.90, 'Phase centroid', fontsize=9.5, fontweight='bold',
                 ha='left', va='center')
    axC_txt.text(0.05, 0.76, r'$c_p = \dfrac{1}{n_p}\sum_{i} v_i$',
                 fontsize=11, ha='left', va='center')
    axC_txt.text(0.02, 0.60, 'Normalize to unit length', fontsize=9.5,
                 fontweight='bold', ha='left', va='center')
    axC_txt.text(0.05, 0.47, r'$\hat{c}_p = \dfrac{c_p}{\|c_p\|}$',
                 fontsize=11, ha='left', va='center')
    axC_txt.text(0.02, 0.31, 'Inter-phase similarity', fontsize=9.5,
                 fontweight='bold', ha='left', va='center')
    axC_txt.text(0.05, 0.19,
                 r'$S_{\mathrm{inter}}(i,j) = \hat{c}_i \cdot \hat{c}_j,\ \ i \neq j$',
                 fontsize=10.5, ha='left', va='center')
    axC_txt.text(0.02, 0.055,
                 f"over ALL ordered pairs ({R['n_inter_pairs']} here; same-type + cross-type pooled)",
                 fontsize=7.6, ha='left', va='center', style='italic', color='#555555')
    axC_txt.text(0.02, -0.02,
                 r'$\mu_{\mathrm{inter}} = %.3f$' % R['mu_inter'],
                 fontsize=11, fontweight='bold', ha='left', va='center')

    # ========================================================================
    # PANEL D: coherence ratios (bars) + PSI arithmetic (signed) + E-E index
    # ========================================================================
    gsD = outer[1, 1].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.40)
    axD_bar = fig.add_subplot(gsD[0])
    axD_txt = fig.add_subplot(gsD[1]); axD_txt.axis('off')

    rho_exploit = R['rho_exploit']
    rho_explore = R['rho_explore']
    bar_vals = [rho_exploit, rho_explore]
    bar_cols = [exploitation_color, exploration_color]
    bars = axD_bar.bar([0, 1], bar_vals, color=bar_cols, alpha=0.85, width=0.62,
                       edgecolor='white')
    axD_bar.axhline(1.0, ls='--', color='#333333', linewidth=1.2)
    axD_bar.text(1.48, 1.02, 'within = between', fontsize=6.8, color='#333333',
                 ha='right', va='bottom')
    axD_bar.set_xticks([0, 1])
    axD_bar.set_xticklabels([r'$\rho_{\mathrm{exploit}}$', r'$\rho_{\mathrm{explore}}$'],
                            fontsize=10)
    axD_bar.set_ylim(0, max(1.25, max(bar_vals) * 1.2))
    axD_bar.set_ylabel('coherence ratio', fontsize=9)
    axD_bar.set_title('Coherence ratios', fontsize=9.5, pad=4)
    for spine in ['top', 'right']:
        axD_bar.spines[spine].set_visible(False)
    for b, v in zip(bars, bar_vals):
        axD_bar.text(b.get_x() + b.get_width() / 2.0, v + 0.03, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    psi = R['psi']
    mu_ex = R['mu_exploit_intra']
    mu_er = R['mu_explore_intra']
    mu_in = R['mu_inter']
    avg_intra = (mu_ex + mu_er) / 2.0
    ee_index = mu_ex / mu_er if mu_er else float('nan')
    if psi > 0:
        psi_interp = ('PSI > 0: within-phase coherence exceeds between-phase\n'
                      '(phases well separated)')
        psi_color = '#1a7f37'
    else:
        psi_interp = ('PSI < 0: within-phase coherence below between-phase\n'
                      '(phases poorly separated)')
        psi_color = colors['threshold']

    axD_txt.set_xlim(0, 1); axD_txt.set_ylim(0, 1)
    axD_txt.text(0.0, 0.96,
                 r'$\rho_{\mathrm{exploit}} = \dfrac{\mu_{\mathrm{exploit\,intra}}}{\mu_{\mathrm{inter}}}'
                 r' = \dfrac{%.3f}{%.3f} = %.3f$' % (mu_ex, mu_in, rho_exploit),
                 fontsize=9.2, ha='left', va='top')
    axD_txt.text(0.0, 0.80,
                 r'$\rho_{\mathrm{explore}} = \dfrac{\mu_{\mathrm{explore\,intra}}}{\mu_{\mathrm{inter}}}'
                 r' = \dfrac{%.3f}{%.3f} = %.3f$' % (mu_er, mu_in, rho_explore),
                 fontsize=9.2, ha='left', va='top')
    axD_txt.text(0.0, 0.635,
                 r'$\rho>1\Rightarrow$ within > between;  $\rho<1\Rightarrow$ within < between',
                 fontsize=7.8, ha='left', va='top', color='#555555', style='italic')

    axD_txt.text(0.0, 0.52,
                 r'$\mathrm{PSI} = \dfrac{\mu_{\mathrm{exploit\,intra}}+\mu_{\mathrm{explore\,intra}}}{2}'
                 r' - \mu_{\mathrm{inter}}$',
                 fontsize=9.6, ha='left', va='top')
    axD_txt.text(0.0, 0.35,
                 r'$= \dfrac{%.3f + %.3f}{2} - %.3f = %.3f - %.3f = %.3f$'
                 % (mu_ex, mu_er, mu_in, avg_intra, mu_in, psi),
                 fontsize=9.2, ha='left', va='top')
    axD_txt.text(0.0, 0.205, psi_interp, fontsize=8.2, ha='left', va='top',
                 color=psi_color, fontweight='bold')
    axD_txt.text(0.0, 0.045,
                 r'E$-$E index (downstream) $= \dfrac{\rho_{\mathrm{exploit}}}{\rho_{\mathrm{explore}}}'
                 r' = \dfrac{\mu_{\mathrm{exploit\,intra}}}{\mu_{\mathrm{explore\,intra}}} = %.3f$'
                 % ee_index,
                 fontsize=8.6, ha='left', va='top', color='#333333')

    return fig


def create_corrected_pipeline_figure():
    """Compute the real numbers and render the corrected A/B/C/D composite."""
    # Example sequence: extended animal list chosen so the REAL pipeline yields
    # >=1 exploitation and >=1 exploration phase, each with >=3 words, giving
    # non-degenerate numbers for every panel. (The base list dog,cat,lion,tiger,
    # mouse,rat collapses to a near-degenerate PSI ~ 0, so it is extended.)
    example_words = ['dog', 'cat', 'lion', 'tiger', 'wolf',
                     'shark', 'whale', 'dolphin', 'mouse', 'rat']

    R = compute_real_pipeline_numbers(example_words)

    # Report the real numbers to stdout for verification.
    print("=" * 70)
    print("REAL PIPELINE READ-OUT (ground truth from analysis code)")
    print("=" * 70)
    print(f"Example sequence : {R['words']}")
    print(f"Consecutive sims : {[round(float(s), 4) for s in R['similarities']]}")
    print(f"threshold tau    : {R['threshold']}   min_phase_length: {R['min_phase_length']}")
    print("Phases:")
    for k, p in enumerate(R['phases'], start=1):
        print(f"  Phase {k}: {p['type']:12s} words[{p['start']}:{p['end'] + 1}] = {p['items']}")
    print(f"mu_exploit_intra : {R['mu_exploit_intra']:.4f}")
    print(f"mu_explore_intra : {R['mu_explore_intra']:.4f}")
    print(f"mu_inter         : {R['mu_inter']:.4f}  ({R['n_inter_pairs']} ordered pairs)")
    print(f"rho_exploit      : {R['rho_exploit']:.4f}")
    print(f"rho_explore      : {R['rho_explore']:.4f}")
    print(f"PSI (subtraction): {R['psi']:.4f}")
    ee = R['mu_exploit_intra'] / R['mu_explore_intra'] if R['mu_explore_intra'] else float('nan')
    print(f"E-E index        : {ee:.4f}")
    print("=" * 70)

    fig = build_composite_figure(R)

    output_dir = Path('output/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / 'pipeline_figure_corrected'
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n[OK] Saved corrected 4-panel figure:")
    for ext in ('png', 'pdf', 'svg'):
        print(f"     {stem.with_suffix('.' + ext).resolve()}")
    return R


if __name__ == '__main__':
    create_corrected_pipeline_figure()
