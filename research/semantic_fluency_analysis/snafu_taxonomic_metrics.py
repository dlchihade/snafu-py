#!/usr/bin/env python3
"""Compute per-participant SVF metrics with snafu under two animal schemes.

Inputs (CLI: --input PATH; default = research/.../data/fluency_data.csv)
- A long CSV with columns ID, Item (one row per response), OR
- A wide CSV / Excel where the first column is the subject ID and the remaining
  columns are response slots (NaN / empty cells are skipped). The Excel file
  /Users/diettachihade/Downloads/animal_fluency_snafu_03_05.xlsx is the wide
  format used here.

- data/schemes/animals_troyer_scheme.csv  (Troyer et al., 1997 taxonomic scheme)
- data/schemes/animals_snafu_scheme.csv   (extended Troyer + Hills + UW snafu scheme;
                                          this is what the Colab notebook uses, see
                                          examples/brm_demo.py)

Outputs (CLI: --output PATH; default = output/snafu_taxonomic_metrics_<input_stem>.csv)
- A per-participant CSV at the chosen path
- console summary

Per-participant columns (Troyer scheme, unprefixed for backward compatibility)
- num_items, num_items_with_category, num_repetitions, num_intrusions_taxonomic,
  num_categories_unique,
  num_switches_static, num_switches_fluid,
  switch_rate_static, switch_rate_fluid,
  avg_cluster_size_static, avg_cluster_size_fluid

Per-participant columns (snafu / Colab scheme, prefixed `snafu_`)
- snafu_num_items_with_category, snafu_num_intrusions,
  snafu_num_categories_unique,
  snafu_num_switches_static, snafu_num_switches_fluid,
  snafu_switch_rate_static, snafu_switch_rate_fluid,
  snafu_avg_cluster_size_static, snafu_avg_cluster_size_fluid

The Colab equivalent of "cluster switches" / "switch rate" is
`snafu_num_switches_static` / `snafu_switch_rate_static` (matches `clustertype='static'`
in `snafu.clusterSwitch`, per examples/brm_demo.py).

Schemes are CSV "category,item"; switches and cluster sizes follow
snafu.findClusters / snafu.clusterSwitch with clustertype='static' (strict
shared-category runs) and 'fluid' (any-shared-category runs).
"""

from __future__ import annotations

import argparse
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


def _summary(sizes: list[int]) -> tuple[int, float]:
    if not sizes:
        return 0, 0.0
    n_clusters = len(sizes)
    n_switches = max(n_clusters - 1, 0)
    avg_size = float(sum(sizes)) / n_clusters
    return n_switches, avg_size


def _scheme_metrics(items: list[str], scheme_path: str, scheme_items: set[str]) -> dict:
    intrusions = sum(1 for t in items if t not in scheme_items)
    labels = snafu.labelClusters(items, scheme_path)
    n_used = len(labels)

    unique_categories: set[str] = set()
    for lab in labels:
        unique_categories.update(c for c in lab.split(';') if c)

    cs_static = snafu.findClusters(items, scheme_path, clustertype='static')
    cs_fluid = snafu.findClusters(items, scheme_path, clustertype='fluid')
    sw_s, avg_s = _summary(cs_static)
    sw_f, avg_f = _summary(cs_fluid)

    rate_s = (sw_s / n_used) if n_used else 0.0
    rate_f = (sw_f / n_used) if n_used else 0.0

    return {
        'num_items_with_category': n_used,
        'num_intrusions': intrusions,
        'num_categories_unique': len(unique_categories),
        'num_switches_static': sw_s,
        'num_switches_fluid': sw_f,
        'switch_rate_static': rate_s,
        'switch_rate_fluid': rate_f,
        'avg_cluster_size_static': avg_s,
        'avg_cluster_size_fluid': avg_f,
    }


def _per_participant(
    items_raw: list[str],
    troyer_path: str,
    troyer_items: set[str],
    snafu_path: str,
    snafu_items: set[str],
) -> dict:
    items = [_clean_item(x) for x in items_raw if str(x).strip()]
    n = len(items)
    repetitions = sum(items.count(t) - 1 for t in set(items)) if items else 0

    troyer = _scheme_metrics(items, troyer_path, troyer_items)
    snafu_m = _scheme_metrics(items, snafu_path, snafu_items)

    out = {
        'num_items': n,
        'num_items_with_category': troyer['num_items_with_category'],
        'num_repetitions': repetitions,
        'num_intrusions_taxonomic': troyer['num_intrusions'],
        'num_categories_unique': troyer['num_categories_unique'],
        'num_switches_static': troyer['num_switches_static'],
        'num_switches_fluid': troyer['num_switches_fluid'],
        'switch_rate_static': troyer['switch_rate_static'],
        'switch_rate_fluid': troyer['switch_rate_fluid'],
        'avg_cluster_size_static': troyer['avg_cluster_size_static'],
        'avg_cluster_size_fluid': troyer['avg_cluster_size_fluid'],
    }
    out.update({f'snafu_{k}': v for k, v in snafu_m.items()})
    return out


def _load_long_format(path: Path) -> pd.DataFrame:
    """Load fluency data and return a long DataFrame with columns ID, Item.

    Accepts:
      - .xlsx (first sheet) or .xls
      - .csv (utf-8 then latin-1 fallback for legacy Mac files)
    Auto-detects whether the file is already long (ID, Item) or wide (ID + many
    response columns / NaN spacers).
    """
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1')

    cols = list(df.columns)
    if len(cols) == 0:
        raise ValueError(f"No columns in {path}")

    if {'ID', 'Item'}.issubset(cols) and len(cols) == 2:
        long_df = df[['ID', 'Item']].copy()
    else:
        id_col = cols[0]
        value_cols = cols[1:]
        long_df = df.melt(id_vars=[id_col], value_vars=value_cols,
                          var_name='_slot', value_name='_item_value')
        long_df = long_df.rename(columns={id_col: 'ID', '_item_value': 'Item'})[['ID', 'Item']]

    long_df['Item'] = long_df['Item'].astype('string').str.strip()
    long_df = long_df[long_df['Item'].notna() & (long_df['Item'] != '')]
    long_df['ID'] = long_df['ID'].astype('string').str.strip()
    long_df = long_df[long_df['ID'].notna() & (long_df['ID'] != '')]
    return long_df.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=base / 'data' / 'fluency_data.csv',
                        help='Long CSV (ID,Item) or wide CSV/XLSX of subjects x responses')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output CSV path (default: output/snafu_taxonomic_metrics_<input_stem>.csv)')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = Path(__file__).resolve().parent
    out_dir = base / 'output'
    out_dir.mkdir(exist_ok=True)

    if args.output is not None:
        out_csv = args.output
    else:
        out_csv = out_dir / f'snafu_taxonomic_metrics_{args.input.stem}.csv'

    troyer_path = REPO_ROOT / 'data' / 'schemes' / 'animals_troyer_scheme.csv'
    snafu_path = REPO_ROOT / 'data' / 'schemes' / 'animals_snafu_scheme.csv'
    for p, label in [(troyer_path, 'Troyer'), (snafu_path, 'snafu')]:
        if not p.exists():
            raise FileNotFoundError(f"{label} scheme not found at {p}")

    df = _load_long_format(args.input)
    print(f"Loaded {len(df)} responses across {df['ID'].nunique()} participants from {args.input}")

    troyer_items = _scheme_items(troyer_path)
    snafu_items = _scheme_items(snafu_path)

    rows = []
    for pid, sub in df.groupby('ID', sort=False):
        items_raw = sub['Item'].tolist()
        metrics = _per_participant(
            items_raw,
            str(troyer_path), troyer_items,
            str(snafu_path), snafu_items,
        )
        rows.append({'ID': pid, **metrics})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} participants)")
    print(out.describe(include='all').T)


if __name__ == '__main__':
    main()
