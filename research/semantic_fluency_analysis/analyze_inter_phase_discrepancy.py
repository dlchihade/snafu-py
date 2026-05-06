#!/usr/bin/env python3
"""
Analyze the discrepancy between figure values and computed values.
Check if columns might be swapped or if there's a different calculation method.
"""

import csv
import ast

def analyze_discrepancy():
    """Analyze the inter-phase mean values to understand the discrepancy."""
    
    csv_path = 'output/phase_coherence_analysis_all_participants.csv'
    
    print("="*70)
    print("ANALYZING INTER-PHASE MEAN DISCREPANCY")
    print("="*70)
    
    # Participant list
    participant_list = [
        'PD00020', 'PD00048', 'PD00146', 'PD00215', 'PD00267', 'PD00457', 'PD00458',
        'PD00471', 'PD00472', 'PD00576', 'PD00583', 'PD00647', 'PD00653', 'PD00666',
        'PD00786', 'PD00849', 'PD00869', 'PD00955', 'PD00959', 'PD00999', 'PD01003',
        'PD01126', 'PD01133', 'PD01145', 'PD01146', 'PD01156', 'PD01160', 'PD01161',
        'PD01201', 'PD01223', 'PD01225', 'PD01237', 'PD01247', 'PD01270', 'PD01284',
        'PD01290', 'PD01306', 'PD01312', 'PD01319', 'PD01369', 'PD01377', 'PD01485',
        'PD01559', 'PD01623', 'PD01660', 'PD01667', 'PD01715'
    ]
    
    participant_set = set(participant_list)
    
    exp_means = []
    expl_means = []
    exp_means_swapped = []  # Test if columns are swapped
    expl_means_swapped = []
    
    detailed_analysis = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get('ID', '').strip()
            if pid in participant_set:
                try:
                    exp_val = float(row.get('inter_phase_mean_exploitation', 0))
                    expl_val = float(row.get('inter_phase_mean_exploration', 0))
                    
                    if exp_val > 0 and expl_val > 0:
                        exp_means.append(exp_val)
                        expl_means.append(expl_val)
                        
                        # Test swapped columns
                        exp_means_swapped.append(expl_val)
                        expl_means_swapped.append(exp_val)
                        
                        detailed_analysis.append({
                            'pid': pid,
                            'exp': exp_val,
                            'expl': expl_val,
                            'diff': abs(exp_val - expl_val)
                        })
                except:
                    continue
    
    print(f"\nLoaded {len(exp_means)} participants with valid data")
    
    # Current values
    print(f"\n{'='*70}")
    print("CURRENT VALUES (as stored in CSV):")
    print(f"{'='*70}")
    print(f"Exploitation inter-phase mean:")
    print(f"  Mean: {sum(exp_means)/len(exp_means):.4f}")
    print(f"  Median: {sorted(exp_means)[len(exp_means)//2]:.4f}")
    print(f"  Range: [{min(exp_means):.4f}, {max(exp_means):.4f}]")
    
    print(f"\nExploration inter-phase mean:")
    print(f"  Mean: {sum(expl_means)/len(expl_means):.4f}")
    print(f"  Median: {sorted(expl_means)[len(expl_means)//2]:.4f}")
    print(f"  Range: [{min(expl_means):.4f}, {max(expl_means):.4f}]")
    
    # Test swapped columns
    print(f"\n{'='*70}")
    print("TEST: IF COLUMNS ARE SWAPPED:")
    print(f"{'='*70}")
    print(f"Exploitation (using exploration column):")
    print(f"  Mean: {sum(exp_means_swapped)/len(exp_means_swapped):.4f}")
    print(f"  Median: {sorted(exp_means_swapped)[len(exp_means_swapped)//2]:.4f}")
    
    print(f"\nExploration (using exploitation column):")
    print(f"  Mean: {sum(expl_means_swapped)/len(expl_means_swapped):.4f}")
    print(f"  Median: {sorted(expl_means_swapped)[len(expl_means_swapped)//2]:.4f}")
    
    # Compare with figure values
    print(f"\n{'='*70}")
    print("COMPARISON WITH FIGURE VALUES:")
    print(f"{'='*70}")
    print(f"Figure shows: Exploitation ≈ 0.79, Exploration ≈ 0.33")
    
    exp_median = sorted(exp_means)[len(exp_means)//2]
    expl_median = sorted(expl_means)[len(expl_means)//2]
    
    print(f"\nCurrent values:")
    print(f"  Exploitation: {exp_median:.4f} (diff from 0.79: {abs(exp_median-0.79):.4f})")
    print(f"  Exploration: {expl_median:.4f} (diff from 0.33: {abs(expl_median-0.33):.4f})")
    
    exp_swapped_median = sorted(exp_means_swapped)[len(exp_means_swapped)//2]
    expl_swapped_median = sorted(expl_means_swapped)[len(expl_means_swapped)//2]
    
    print(f"\nIf columns swapped:")
    print(f"  Exploitation: {exp_swapped_median:.4f} (diff from 0.79: {abs(exp_swapped_median-0.79):.4f})")
    print(f"  Exploration: {expl_swapped_median:.4f} (diff from 0.33: {abs(expl_swapped_median-0.33):.4f})")
    
    # Check if maybe it's 1 - value (inverse)?
    print(f"\n{'='*70}")
    print("TEST: IF VALUES ARE INVERTED (1 - value):")
    print(f"{'='*70}")
    exp_inverted = [1 - x for x in exp_means]
    expl_inverted = [1 - x for x in expl_means]
    
    exp_inv_median = sorted(exp_inverted)[len(exp_inverted)//2]
    expl_inv_median = sorted(expl_inverted)[len(expl_inverted)//2]
    
    print(f"  Exploitation (1 - value): {exp_inv_median:.4f} (diff from 0.79: {abs(exp_inv_median-0.79):.4f})")
    print(f"  Exploration (1 - value): {expl_inv_median:.4f} (diff from 0.33: {abs(expl_inv_median-0.33):.4f})")
    
    # Show participants with extreme values
    print(f"\n{'='*70}")
    print("PARTICIPANTS WITH EXTREME VALUES:")
    print(f"{'='*70}")
    detailed_analysis.sort(key=lambda x: x['diff'], reverse=True)
    
    print(f"\nTop 5 largest differences between exp and expl:")
    for item in detailed_analysis[:5]:
        print(f"  {item['pid']}: exp={item['exp']:.4f}, expl={item['expl']:.4f}, diff={item['diff']:.4f}")
    
    print(f"\nTop 5 smallest differences:")
    for item in detailed_analysis[-5:]:
        print(f"  {item['pid']}: exp={item['exp']:.4f}, expl={item['expl']:.4f}, diff={item['diff']:.4f}")

if __name__ == "__main__":
    analyze_discrepancy()






