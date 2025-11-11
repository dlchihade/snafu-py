#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_steps(participant_id: str):
    config = AnalysisConfig.from_yaml('config/config.yaml')
    analyzer = SemanticFluencyAnalyzer(config)
    analyzer.load_data(config.data_paths['fluency_data'], config.data_paths['meg_data'])

    pdata = analyzer.data[analyzer.data['ID'] == participant_id]
    if pdata.empty:
        print(f"❌ Participant {participant_id} not found")
        return

    items = pdata['Item'].tolist()
    print_header(f"Participant {participant_id} - Raw Items (order preserved)")
    print(items)

    # Get vectors and valid words
    vectors, valid_words, valid_indices = analyzer.spacy_optimizer.get_vectors_batch(
        items,
        min_similarity=analyzer.config.min_similarity,
        max_similarity=analyzer.config.max_similarity,
    )

    print_header("Word Coverage and Valid Indices")
    print(f"Total items: {len(items)}")
    print(f"Valid vectors: {len(vectors)}")
    mapping = [(i, items[i], v_idx in valid_indices) for i, v_idx in enumerate(range(len(items)))]
    print("Index | Item | HasVector")
    for i, item in enumerate(items):
        has_vec = (i in valid_indices)
        print(f"{i:3d} | {item:<20s} | {str(has_vec)}")

    # Compute consecutive cosine similarities on valid vectors only
    print_header("Consecutive Cosine Similarities (on valid words only)")
    # Build aligned sequence of valid words and vectors
    valid_pairs = [(idx, items[idx], vectors[j]) for j, idx in enumerate(valid_indices)]
    cosines = []
    for k in range(len(valid_pairs) - 1):
        idx1, w1, v1 = valid_pairs[k]
        idx2, w2, v2 = valid_pairs[k + 1]
        sim = float(cosine_similarity([v1], [v2])[0][0])
        cosines.append((idx1, w1, idx2, w2, sim))
    print("i -> i+1 | word_i -> word_{i+1} | cosine")
    for idx1, w1, idx2, w2, sim in cosines:
        print(f"{idx1:3d}->{idx2:3d} | {w1:<15s}->{w2:<15s} | {sim: .4f}")

    # Phase assignment step-by-step (use analyzer logic)
    print_header("Phase Assignment (threshold-based)")
    threshold = analyzer.config.similarity_threshold
    min_len = analyzer.config.min_phase_length
    print(f"Threshold τ = {threshold}, min_phase_length = {min_len}")

    similarities = []
    for k in range(len(vectors) - 1):
        similarities.append(float(cosine_similarity([vectors[k]], [vectors[k + 1]])[0][0]))
    similarities = np.array(similarities)

    if len(similarities) == 0:
        print("Insufficient similarities to form phases.")
        return

    current_phase = "Exploitation" if similarities[0] > threshold else "Exploration"
    phase_start = 0
    phases = []
    print(f"Start at index 0 as {current_phase} (cos={similarities[0]:.4f})")
    for i in range(1, len(similarities)):
        should_switch = (current_phase == "Exploitation" and similarities[i] <= threshold) or \
                        (current_phase == "Exploration" and similarities[i] > threshold)
        print(f"i={i}, cos={similarities[i]:.4f}, phase={current_phase}, switch={should_switch}")
        if should_switch and (i - phase_start) >= min_len:
            # close phase
            phase_items = valid_words[phase_start:i+1] if i+1 <= len(valid_words) else valid_words[phase_start:]
            phases.append({
                'type': current_phase,
                'start': phase_start,
                'end': i,
                'items': phase_items,
                'vectors': vectors[phase_start:i+1],
                'similarities': similarities[phase_start:i]
            })
            print(f"  → CLOSE {current_phase} [{phase_start}:{i}] with {len(phase_items)} items")
            current_phase = "Exploration" if current_phase == "Exploitation" else "Exploitation"
            phase_start = i
    # final phase
    if (len(similarities) - phase_start) >= min_len:
        phase_items = valid_words[phase_start:len(valid_words)]
        phases.append({
            'type': current_phase,
            'start': phase_start,
            'end': len(similarities),
            'items': phase_items,
            'vectors': vectors[phase_start:len(valid_words)],
            'similarities': similarities[phase_start:len(similarities)]
        })
        print(f"  → CLOSE {current_phase} [{phase_start}:{len(similarities)}] with {len(phase_items)} items")

    print_header("Final Phases")
    for p_idx, p in enumerate(phases, 1):
        print(f"{p_idx:2d}. {p['type']} [{p['start']}:{p['end']}] (size={len(p['items'])})")
        print(f"   Items: {p['items']}")
        if len(p['similarities']) > 0:
            print(f"   Avg consecutive cosine: {np.mean(p['similarities']):.4f}")

    # Intra-phase pairwise cosine similarities
    print_header("Intra-Phase Pairwise Similarities (cosine)")
    exp_pairs, expl_pairs = [], []
    for p in phases:
        if len(p['vectors']) >= 2:
            mat = cosine_similarity(np.array(p['vectors']))
            tri = mat[np.triu_indices_from(mat, k=1)]
            if p['type'] == 'Exploitation':
                exp_pairs.extend(tri.tolist())
            else:
                expl_pairs.extend(tri.tolist())
            print(f"{p['type']} phase size={len(p['vectors'])}, pairs={len(tri)}; mean={tri.mean():.4f}")
    if exp_pairs:
        print(f"Exploitation intra mean={np.mean(exp_pairs):.4f}, var={np.var(exp_pairs):.4f}")
    if expl_pairs:
        print(f"Exploration intra mean={np.mean(expl_pairs):.4f}, var={np.var(expl_pairs):.4f}")

    # Phase centroids and inter-phase similarities
    print_header("Inter-Phase Similarities (centroid–centroid)")
    centroids = []
    for p in phases:
        if len(p['vectors']) > 0:
            c = np.mean(np.array(p['vectors']), axis=0)
            n = np.linalg.norm(c)
            c = c / n if n > 0 else c
            centroids.append((p['type'], c))
    inter = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            sim = float(np.dot(centroids[i][1], centroids[j][1]))
            inter.append(sim)
            print(f"{centroids[i][0]} ↔ {centroids[j][0]}: {sim:.4f}")
    if inter:
        inter_mean = float(np.mean(inter))
        print(f"Inter-phase mean={inter_mean:.4f}, var={np.var(inter):.4f}")
    else:
        inter_mean = 0.0

    # Coherence ratios
    exp_mean = float(np.mean(exp_pairs)) if exp_pairs else 0.0
    expl_mean = float(np.mean(expl_pairs)) if expl_pairs else 0.0
    exp_ratio = (exp_mean / inter_mean) if inter_mean > 0 else 0.0
    expl_ratio = (expl_mean / inter_mean) if inter_mean > 0 else 0.0
    psi = ((exp_mean + expl_mean) / 2 - inter_mean) if (exp_pairs and expl_pairs) else 0.0

    print_header("Coherence Ratios and PSI")
    print(f"Exploitation coherence ratio = {exp_ratio:.4f}  (intra_exploit / inter)")
    print(f"Exploration coherence ratio = {expl_ratio:.4f}  (intra_explore / inter)")
    print(f"Phase Separation Index (PSI) = {psi:.4f}  (avg(intras) - inter)")


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else 'PD00020'
    show_steps(pid)
