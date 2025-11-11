#!/usr/bin/env python3
"""
Load output/NATURE_REAL_metrics.csv and print aggregated means/variances for
key inter-phase metrics to verify figure panels.
"""
from pathlib import Path
import pandas as pd


def summarize(series: pd.Series, name: str) -> None:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        print(f"{name}: no data")
        return
    print(f"{name} (N = {len(s)}):")
    print(f"  mean = {s.mean():.6f}")
    print(f"  variance = {s.var(ddof=0):.6f}  # population variance")
    print(f"  std = {s.std(ddof=0):.6f}")


def main():
    metrics_path = Path(__file__).parent / 'output' / 'NATURE_REAL_metrics.csv'
    if not metrics_path.exists():
        print(f"Missing metrics file: {metrics_path}")
        return
    df = pd.read_csv(metrics_path)

    summarize(df.get('exploitation_intra_mean'), 'Exploitation intra-phase mean (cosine)')
    summarize(df.get('exploration_intra_mean'), 'Exploration intra-phase mean (cosine)')
    summarize(df.get('inter_phase_mean'), 'Inter-phase mean (cosine)')
    summarize(df.get('phase_separation_index'), 'Phase separation index (a.u.)')


if __name__ == '__main__':
    main()



