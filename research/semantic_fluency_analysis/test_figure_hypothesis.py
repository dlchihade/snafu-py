#!/usr/bin/env python3
"""
Test hypothesis: Figure might be using:
- Exploitation: swapped (using exploration column values)
- Exploration: inverted exploitation values (1 - exploitation)
"""

import csv

def test_hypothesis():
    """Test if figure uses swapped/inverted values."""
    
    csv_path = 'output/phase_coherence_analysis_all_participants.csv'
    
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
    
    # Hypothesis: 
    # - Exploitation in figure = Exploration column from CSV (swapped)
    # - Exploration in figure = 1 - Exploitation column from CSV (inverted)
    
    exp_figure_values = []  # Using exploration column
    expl_figure_values = []  # Using 1 - exploitation column
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get('ID', '').strip()
            if pid in participant_set:
                try:
                    exp_val = float(row.get('inter_phase_mean_exploitation', 0))
                    expl_val = float(row.get('inter_phase_mean_exploration', 0))
                    
                    if exp_val > 0 and expl_val > 0:
                        # Hypothesis: Figure uses swapped/inverted
                        exp_figure_values.append(expl_val)  # Swapped: use exploration
                        expl_figure_values.append(1 - exp_val)  # Inverted: 1 - exploitation
                except:
                    continue
    
    print("="*70)
    print("TESTING HYPOTHESIS:")
    print("Figure might use:")
    print("  - Exploitation: exploration column (swapped)")
    print("  - Exploration: 1 - exploitation column (inverted)")
    print("="*70)
    
    if exp_figure_values and expl_figure_values:
        exp_sorted = sorted(exp_figure_values)
        expl_sorted = sorted(expl_figure_values)
        
        exp_median = exp_sorted[len(exp_sorted)//2]
        expl_median = expl_sorted[len(expl_sorted)//2]
        
        print(f"\nHypothesis values:")
        print(f"  Exploitation (using exploration column):")
        print(f"    Median: {exp_median:.4f}")
        print(f"    Mean: {sum(exp_figure_values)/len(exp_figure_values):.4f}")
        print(f"    Range: [{min(exp_figure_values):.4f}, {max(exp_figure_values):.4f}]")
        
        print(f"\n  Exploration (using 1 - exploitation column):")
        print(f"    Median: {expl_median:.4f}")
        print(f"    Mean: {sum(expl_figure_values)/len(expl_figure_values):.4f}")
        print(f"    Range: [{min(expl_figure_values):.4f}, {max(expl_figure_values):.4f}]")
        
        print(f"\n{'='*70}")
        print("COMPARISON WITH FIGURE:")
        print(f"{'='*70}")
        print(f"Figure shows: Exploitation ≈ 0.79, Exploration ≈ 0.33")
        print(f"\nHypothesis values:")
        print(f"  Exploitation: {exp_median:.4f} (diff from 0.79: {abs(exp_median-0.79):.4f})")
        print(f"  Exploration: {expl_median:.4f} (diff from 0.33: {abs(expl_median-0.33):.4f})")
        
        if abs(exp_median - 0.79) < 0.1 and abs(expl_median - 0.33) < 0.1:
            print(f"\n✅ HYPOTHESIS MATCHES! This is likely how the figure was generated.")
        else:
            print(f"\n❌ Hypothesis doesn't match. Continuing investigation...")
            
            # Test other combinations
            print(f"\n{'='*70}")
            print("TESTING OTHER COMBINATIONS:")
            print(f"{'='*70}")
            
            # Test: Exploitation = exploration, Exploration = 1 - exploration
            exp_test2 = [expl_val for expl_val in expl_figure_values if expl_val > 0]
            expl_test2 = [1 - expl_val for expl_val in expl_figure_values if expl_val > 0]
            if exp_test2 and expl_test2:
                exp2_median = sorted(exp_test2)[len(exp_test2)//2]
                expl2_median = sorted(expl_test2)[len(expl_test2)//2]
                print(f"\nTest 2: Exp=exploration, Expl=1-exploration")
                print(f"  Exploitation: {exp2_median:.4f} (diff from 0.79: {abs(exp2_median-0.79):.4f})")
                print(f"  Exploration: {expl2_median:.4f} (diff from 0.33: {abs(expl2_median-0.33):.4f})")

if __name__ == "__main__":
    test_hypothesis()






