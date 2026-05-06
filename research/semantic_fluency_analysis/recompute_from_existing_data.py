#!/usr/bin/env python3
"""
Recompute inter-phase mean statistics from existing phase_coherence_analysis data
for the specified participants.
"""

import csv
from pathlib import Path

# Participant list from the provided data
PARTICIPANT_LIST = [
    'PD00020', 'PD00048', 'PD00146', 'PD00215', 'PD00267', 'PD00457', 'PD00458',
    'PD00471', 'PD00472', 'PD00576', 'PD00583', 'PD00647', 'PD00653', 'PD00666',
    'PD00786', 'PD00849', 'PD00869', 'PD00955', 'PD00959', 'PD00999', 'PD01003',
    'PD01126', 'PD01133', 'PD01145', 'PD01146', 'PD01156', 'PD01160', 'PD01161',
    'PD01201', 'PD01223', 'PD01225', 'PD01237', 'PD01247', 'PD01270', 'PD01284',
    'PD01290', 'PD01306', 'PD01312', 'PD01319', 'PD01369', 'PD01377', 'PD01485',
    'PD01559', 'PD01623', 'PD01660', 'PD01667', 'PD01715'
]

def compute_stats(values):
    """Compute statistics for a list of values."""
    if not values:
        return None
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    mean = sum(sorted_vals) / n
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    
    # Standard deviation
    variance = sum((x - mean) ** 2 for x in sorted_vals) / (n - 1) if n > 1 else 0
    std = variance ** 0.5
    
    return {
        'count': n,
        'mean': mean,
        'median': median,
        'std': std,
        'min': min(sorted_vals),
        'max': max(sorted_vals),
        'q1': q1,
        'q3': q3
    }

def recompute_from_csv():
    """Recompute statistics from existing CSV file."""
    
    csv_path = Path('output/phase_coherence_analysis_all_participants.csv')
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    print("="*70)
    print("RECOMPUTING INTER-PHASE MEANS FROM EXISTING DATA")
    print("="*70)
    print(f"Participants to process: {len(PARTICIPANT_LIST)}")
    print(f"Data file: {csv_path}")
    
    participant_set = set(PARTICIPANT_LIST)
    
    exploitation_inter_vals = []
    exploration_inter_vals = []
    found_participants = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            participant_id = row.get('ID', '').strip()
            
            if participant_id in participant_set:
                found_participants.append(participant_id)
                
                # Extract inter-phase mean values
                try:
                    exp_val = row.get('inter_phase_mean_exploitation', '').strip()
                    expl_val = row.get('inter_phase_mean_exploration', '').strip()
                    
                    if exp_val and exp_val != '':
                        exp_float = float(exp_val)
                        if exp_float > 0:
                            exploitation_inter_vals.append(exp_float)
                    
                    if expl_val and expl_val != '':
                        expl_float = float(expl_val)
                        if expl_float > 0:
                            exploration_inter_vals.append(expl_float)
                            
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Warning: Could not parse values for {participant_id}: {e}")
                    continue
    
    print(f"\n✅ Found data for {len(found_participants)}/{len(PARTICIPANT_LIST)} participants")
    
    missing = set(PARTICIPANT_LIST) - set(found_participants)
    if missing:
        print(f"⚠️  Missing participants: {sorted(missing)}")
    
    # Compute statistics
    print(f"\n{'='*70}")
    print("📊 RECOMPUTED STATISTICS")
    print(f"{'='*70}")
    
    exp_stats = compute_stats(exploitation_inter_vals)
    expl_stats = compute_stats(exploration_inter_vals)
    
    if exp_stats:
        print(f"\n🔴 Exploitation Inter-Phase Mean:")
        print(f"   Count: {exp_stats['count']}")
        print(f"   Mean:  {exp_stats['mean']:.4f}")
        print(f"   Median: {exp_stats['median']:.4f}")
        print(f"   Std:   {exp_stats['std']:.4f}")
        print(f"   Min:   {exp_stats['min']:.4f}")
        print(f"   Max:   {exp_stats['max']:.4f}")
        print(f"   Q1:    {exp_stats['q1']:.4f}")
        print(f"   Q3:    {exp_stats['q3']:.4f}")
    
    if expl_stats:
        print(f"\n🟢 Exploration Inter-Phase Mean:")
        print(f"   Count: {expl_stats['count']}")
        print(f"   Mean:  {expl_stats['mean']:.4f}")
        print(f"   Median: {expl_stats['median']:.4f}")
        print(f"   Std:   {expl_stats['std']:.4f}")
        print(f"   Min:   {expl_stats['min']:.4f}")
        print(f"   Max:   {expl_stats['max']:.4f}")
        print(f"   Q1:    {expl_stats['q1']:.4f}")
        print(f"   Q3:    {expl_stats['q3']:.4f}")
    
    # Compare with figure values
    print(f"\n{'='*70}")
    print("📊 COMPARISON WITH FIGURE VALUES")
    print(f"{'='*70}")
    
    if exp_stats and expl_stats:
        print(f"\nFigure shows (from boxplot medians):")
        print(f"  Exploitation: ~0.79")
        print(f"  Exploration: ~0.33")
        print(f"\nRecomputed values (medians):")
        print(f"  Exploitation: {exp_stats['median']:.4f}")
        print(f"  Exploration: {expl_stats['median']:.4f}")
        print(f"\nDifference:")
        print(f"  Exploitation: {abs(exp_stats['median'] - 0.79):.4f}")
        print(f"  Exploration: {abs(expl_stats['median'] - 0.33):.4f}")
    
    print(f"\n{'='*70}")
    print("✅ Recomputation complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    recompute_from_csv()

