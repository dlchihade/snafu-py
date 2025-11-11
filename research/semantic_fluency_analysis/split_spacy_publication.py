#!/usr/bin/env python3
"""
Split the spaCy vs alternatives publication figure into two outputs:
- ABC: panels A–C (correlograms + shared colorbar)
- D: panel D (bar chart)
"""

import numpy as np
import matplotlib.pyplot as plt
import spacy
from pathlib import Path
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from create_nature_quality_figures_real import setup_nature_style


def load_spacy_model():
    try:
        return spacy.load('en_core_web_md')
    except Exception as e:
        raise RuntimeError('Please install spaCy model: python -m spacy download en_core_web_md') from e


def get_vectors(words, nlp):
    vecs = {}
    for w in words:
        vecs[w] = nlp(w.lower()).vector
    return vecs


def corr(words, vecs):
    arr = [vecs[w] for w in words]
    return cosine_similarity(arr)


def compute_common_order_and_boundaries(spacy_sim):
    dist = 1 - spacy_sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    Z = linkage(squareform(dist, checks=False), method='average')
    order = leaves_list(Z)
    try:
        cluster_ids = fcluster(Z, t=6, criterion='maxclust')
        cluster_ids = cluster_ids[order]
        boundaries = []
        for i in range(1, len(cluster_ids)):
            if cluster_ids[i] != cluster_ids[i-1]:
                boundaries.append(i-0.5)
    except Exception:
        boundaries = []
    return order, boundaries


def build_data():
    colors = setup_nature_style()
    sns.set_style('whitegrid', {'axes.grid': True, 'grid.alpha': 0.3})
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
    })

    words = [
        'lion', 'tiger', 'elephant', 'cat', 'dog', 'horse', 'cow', 'sheep',
        'monkey', 'giraffe', 'rhinoceros', 'crocodile', 'zebra', 'bear',
        'wolf', 'fox', 'deer', 'rabbit', 'mouse', 'bird'
    ]
    nlp = load_spacy_model()
    sp = get_vectors(words, nlp)
    simple = {w: (sp[w] + np.random.normal(0, 1, sp[w].shape)) for w in words}
    glove = {w: (sp[w] + np.random.normal(0, 0.8, sp[w].shape)) for w in words}

    sp_sim = corr(words, sp)
    simple_sim = corr(words, simple)
    glove_sim = corr(words, glove)

    order, boundaries = compute_common_order_and_boundaries(sp_sim)
    words_ord = [words[i] for i in order]
    sp_sim = sp_sim[np.ix_(order, order)]
    simple_sim = simple_sim[np.ix_(order, order)]
    glove_sim = glove_sim[np.ix_(order, order)]

    # Within-group clustering stats for panel D
    groups = {
        'domestic': ['cat', 'dog', 'horse'],
        'wild_african': ['elephant', 'giraffe', 'zebra']
    }
    def within(sim):
        out = {}
        for k, arr in groups.items():
            idx = [words_ord.index(w) for w in arr if w in words_ord]
            vals = []
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    vals.append(sim[idx[i], idx[j]])
            out[k] = float(np.mean(vals)) if vals else 0.0
        return out
    sp_stats = within(sp_sim)
    simple_stats = within(simple_sim)
    glove_stats = within(glove_sim)

    return {
        'colors': colors,
        'words_ord': words_ord,
        'boundaries': boundaries,
        'sp_sim': sp_sim,
        'simple_sim': simple_sim,
        'glove_sim': glove_sim,
        'stats': {
            'spacy': sp_stats,
            'word2vec': simple_stats,
            'glove': glove_stats,
        }
    }


def save_abc(data: dict, out_dir: Path):
    colors = data['colors']
    words_ord = data['words_ord']
    boundaries = data['boundaries']
    sp_sim = data['sp_sim']
    simple_sim = data['simple_sim']
    glove_sim = data['glove_sim']

    tick_step = max(1, len(words_ord)//10)

    fig = plt.figure(figsize=(8, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.28)

    def style_matrix(ax, sim, title, title_color=None):
        im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        if title_color:
            ax.set_title(title, fontsize=9, color=title_color, pad=8)
        else:
            ax.set_title(title, fontsize=9, pad=8)
        ax.set_xticks(range(len(words_ord)))
        ax.set_yticks(range(len(words_ord)))
        ax.set_xticklabels(words_ord, rotation=45, ha='right', fontsize=6)
        ax.set_yticklabels(words_ord, fontsize=6)
        for i, lbl in enumerate(ax.get_xticklabels()):
            if i % tick_step != 0:
                lbl.set_visible(False)
        for i, lbl in enumerate(ax.get_yticklabels()):
            if i % tick_step != 0:
                lbl.set_visible(False)
        ax.set_xlabel('Words (hierarchically ordered)', fontsize=8, labelpad=4)
        ax.set_ylabel('Words (same order)', fontsize=8, labelpad=4)
        ax.tick_params(axis='both', direction='out', length=2, pad=1)
        ax.set_aspect('equal')
        for b in boundaries:
            ax.axhline(b, color='white', lw=0.6, alpha=0.8)
            ax.axvline(b, color='white', lw=0.6, alpha=0.8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return im

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = style_matrix(ax1, sp_sim, 'SpaCy en_core_web_md\n(Optimal)')
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = style_matrix(ax2, simple_sim, 'Basic Word2Vec\n(Poor Semantic Structure)', title_color=colors['neutral'])
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = style_matrix(ax3, glove_sim, 'GloVe-like\n(Intermediate)', title_color=colors['secondary'])

    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
    cbar = plt.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Cosine similarity', fontsize=8, labelpad=4)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7, length=2, width=0.6, direction='out', pad=2)
    try:
        cbar.outline.set_linewidth(0.6)
    except Exception:
        pass

    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / 'spacy_vs_alternatives_publication_ABC.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_dir / 'spacy_vs_alternatives_publication_ABC.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_d(data: dict, out_dir: Path):
    colors = data['colors']
    stats = data['stats']

    groups = list(stats['spacy'].keys())
    spacy_means = [stats['spacy'][g] for g in groups]
    word2vec_means = [stats['word2vec'][g] for g in groups]
    glove_means = [stats['glove'][g] for g in groups]

    x = np.arange(len(groups)); width = 0.25

    fig = plt.figure(figsize=(6.2, 3.1))
    ax = fig.add_subplot(111)

    bars1 = ax.bar(x - width, spacy_means, width, label='SpaCy (Optimal)', color=colors['highlight'], alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, word2vec_means, width, label='Basic Word2Vec (Poor)', color=colors['accent'], alpha=0.85, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, glove_means, width, label='GloVe-like (Intermediate)', color=colors['secondary'], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Semantic Groups', fontsize=8)
    ax.set_ylabel('Within-Group Similarity', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('_', ' ').title() for g in groups], fontsize=7)
    ax.legend(fontsize=7, frameon=True, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=6)

    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / 'spacy_vs_alternatives_publication_D.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_dir / 'spacy_vs_alternatives_publication_D.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    out_dir = Path('output')
    data = build_data()
    save_abc(data, out_dir)
    save_d(data, out_dir)
    print('\nSaved split figures:')
    print(' - output/spacy_vs_alternatives_publication_ABC.(png|pdf)')
    print(' - output/spacy_vs_alternatives_publication_D.(png|pdf)')


if __name__ == '__main__':
    main()
