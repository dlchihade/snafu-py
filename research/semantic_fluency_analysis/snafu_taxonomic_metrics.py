#!/usr/bin/env python3
"""Compute per-participant SVF metrics with snafu using the Troyer **taxonomic** scheme.

Inputs
- research/semantic_fluency_analysis/data/fluency_data.csv  (columns: ID, Item)
- data/schemes/animals_troyer_scheme.csv  (canonical taxonomic scheme; Troyer et al., 1997)

Outputs
- research/semantic_fluency_analysis/output/snafu_taxonomic_metrics.csv
- console summary

Per-participant columns
- num_items, num_repetitions, num_intrusions_taxonomic,
  num_categories_unique,
  num_switches_static, num_switches_fluid,
  switch_rate_static, switch_rate_fluid,
  avg_cluster_size_static, avg_cluster_size_fluid

The Troyer taxonomic scheme is csv "category,item"; switches and cluster sizes
follow snafu.clusterSwitch / snafu.findClusters with clustertype='static' (strict
shared-category runs) and 'fluid' (any-shared-category runs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import snafu  # noqa: E402  (import after sys.path tweak)


def _clean_item(s: str) -> str:
    return str(s).strip().lower().replace(' ', '').replace("'", '').replace('-', '')


def _scheme_items(scheme_path: Path) -> set[str]:
    items: set[str] = set()
    with open(scheme_path, 'rt', encoding='utf-8-sig') as fh:
        for line in fh:
            line = line.rstrip()
            if not line or line.startswith('#'):
                continue
            _, item = line.split(',', 1)
            items.add(_clean_item(item))
    return items


def _per_participant(items_raw: list[str], scheme_path: str, scheme_items: set[str]) -> dict:
    items = [_clean_item(x) for x in items_raw if str(x).strip()]
    n = len(items)

    repetitions = sum(items.count(t) - 1 for t in set(items)) if items else 0
    intrusions = sum(1 for t in items if t not in scheme_items)

    labels = snafu.labelClusters(items, scheme_path)
    n_used = len(labels)

    unique_categories = set()
    for lab in labels:
        unique_categories.update(c for c in lab.split(';') if c)

    cs_static = snafu.findClusters(items, scheme_path, clustertype='static')
    cs_fluid = snafu.findClusters(items, scheme_path, clustertype='fluid')

    def _summary(sizes: list[int]) -> tuple[int, float, float]:
        if not sizes:
            return 0, 0.0, 0.0
        n_clusters = len(sizes)
        n_switches = max(n_clusters - 1, 0)
        avg_size = float(sum(sizes)) / n_clusters
        return n_switches, avg_size, n_clusters

    sw_s, avg_s, _nc_s = _summary(cs_static)
    sw_f, avg_f, _nc_f = _summary(cs_fluid)

    rate_s = (sw_s / n_used) if n_used else 0.0
    rate_f = (sw_f / n_used) if n_used else 0.0

    return {
        'num_items': n,
        'num_items_with_category': n_used,
        'num_repetitions': repetitions,
        'num_intrusions_taxonomic': intrusions,
        'num_categories_unique': len(unique_categories),
        'num_switches_static': sw_s,
        'num_switches_fluid': sw_f,
        'switch_rate_static': rate_s,
        'switch_rate_fluid': rate_f,
        'avg_cluster_size_static': avg_s,
        'avg_cluster_size_fluid': avg_f,
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    data_path = base / 'data' / 'fluency_data.csv'
    out_dir = base / 'output'
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'snafu_taxonomic_metrics.csv'

    scheme_path = REPO_ROOT / 'data' / 'schemes' / 'animals_troyer_scheme.csv'
    if not scheme_path.exists():
        raise FileNotFoundError(f"Troyer scheme not found at {scheme_path}")

    df = pd.read_csv(data_path)
    if not {'ID', 'Item'}.issubset(df.columns):
        raise ValueError(f"Expected columns ID, Item in {data_path}, got {list(df.columns)}")

    scheme_items = _scheme_items(scheme_path)

    rows = []
    for pid, sub in df.groupby('ID', sort=False):
        items_raw = sub['Item'].tolist()
        metrics = _per_participant(items_raw, str(scheme_path), scheme_items)
        rows.append({'ID': pid, **metrics})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} participants)")
    print(out.describe(include='all').T)


if __name__ == '__main__':
    main()
