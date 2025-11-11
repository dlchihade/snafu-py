#!/usr/bin/env python3
"""
Render a single correlogram (spaCy en_core_web_md) with themed styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import spacy
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from create_nature_quality_figures_real import setup_nature_style


def load_spacy_model():
    try:
        return spacy.load('en_core_web_md')
    except Exception as e:
        raise RuntimeError('Please install spaCy model: python -m spacy download en_core_web_md') from e


def main():
    colors = setup_nature_style()
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
    vecs = [nlp(w.lower()).vector for w in words]
    sim = cosine_similarity(vecs)

    # Reorder by clustering and compute boundaries
    dist = 1 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    Z = linkage(squareform(dist, checks=False), method='average')
    order = leaves_list(Z)
    words_ord = [words[i] for i in order]
    sim_ord = sim[np.ix_(order, order)]

    try:
        cluster_ids = fcluster(Z, t=6, criterion='maxclust')
        cluster_ids = cluster_ids[order]
        boundaries = []
        for i in range(1, len(cluster_ids)):
            if cluster_ids[i] != cluster_ids[i-1]:
                boundaries.append(i-0.5)
    except Exception:
        boundaries = []

    tick_step = max(1, len(words_ord)//10)

    fig = plt.figure(figsize=(4.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(sim_ord, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(words_ord)))
    ax.set_yticks(range(len(words_ord)))
    ax.set_xticklabels(words_ord, rotation=45, ha='right')
    ax.set_yticklabels(words_ord)
    for i, lbl in enumerate(ax.get_xticklabels()):
        if i % tick_step != 0:
            lbl.set_visible(False)
    for i, lbl in enumerate(ax.get_yticklabels()):
        if i % tick_step != 0:
            lbl.set_visible(False)
    ax.set_xlabel('Words (hierarchically ordered)')
    ax.set_ylabel('Words (same order)')
    ax.tick_params(axis='both', direction='out', length=2, pad=1)
    ax.set_aspect('equal')
    for b in boundaries:
        ax.axhline(b, color='white', lw=0.6, alpha=0.8)
        ax.axvline(b, color='white', lw=0.6, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cax = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Cosine similarity', fontsize=8, labelpad=4)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7, length=2, width=0.6, direction='out', pad=2)
    try:
        cbar.outline.set_linewidth(0.6)
    except Exception:
        pass

    out = Path('output')
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'spacy_matrix_only.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out / 'spacy_matrix_only.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print('Saved single correlogram:')
    print(' - output/spacy_matrix_only.(png|pdf)')


if __name__ == '__main__':
    main()

