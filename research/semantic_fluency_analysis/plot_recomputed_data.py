#!/usr/bin/env python3
"""
Plot the recomputed inter-phase means data in the same format as the original figure.
"""

import csv
import math
from pathlib import Path

# Participant list
PARTICIPANT_LIST = [
    'PD00020', 'PD00048', 'PD00146', 'PD00215', 'PD00267', 'PD00457', 'PD00458',
    'PD00471', 'PD00472', 'PD00576', 'PD00583', 'PD00647', 'PD00653', 'PD00666',
    'PD00786', 'PD00849', 'PD00869', 'PD00955', 'PD00959', 'PD00999', 'PD01003',
    'PD01126', 'PD01133', 'PD01145', 'PD01146', 'PD01156', 'PD01160', 'PD01161',
    'PD01201', 'PD01223', 'PD01225', 'PD01237', 'PD01247', 'PD01270', 'PD01284',
    'PD01290', 'PD01306', 'PD01312', 'PD01319', 'PD01369', 'PD01377', 'PD01485',
    'PD01559', 'PD01623', 'PD01660', 'PD01667', 'PD01715'
]

def load_data():
    """Load data from CSV file."""
    csv_path = Path('output/phase_coherence_analysis_all_participants.csv')
    
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    participant_set = set(PARTICIPANT_LIST)
    
    data = {
        'exploitation_intra_mean': [],
        'exploration_intra_mean': [],
        'exploitation_intra_variance': [],
        'exploration_intra_variance': [],
        'inter_phase_mean_exploitation': [],
        'inter_phase_mean_exploration': [],
        'inter_phase_variance_exploitation': [],
        'inter_phase_variance_exploration': []
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            participant_id = row.get('ID', '').strip()
            
            if participant_id in participant_set:
                try:
                    # Extract all values
                    for key in data.keys():
                        val = row.get(key, '').strip()
                        if val and val != '':
                            float_val = float(val)
                            if float_val > 0 or key.endswith('_variance'):  # Allow variance to be 0
                                data[key].append(float_val)
                except (ValueError, KeyError):
                    continue
    
    return data

def compute_stats(data1, data2):
    """Compute t-test and effect size statistics."""
    n1, n2 = len(data1), len(data2)
    
    if n1 < 2 or n2 < 2:
        return None
    
    # Mean
    mean1 = sum(data1) / n1
    mean2 = sum(data2) / n2
    
    # Variance
    var1 = sum((x - mean1) ** 2 for x in data1) / (n1 - 1) if n1 > 1 else 0
    var2 = sum((x - mean2) ** 2 for x in data2) / (n2 - 1) if n2 > 1 else 0
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Standard error
    se1 = math.sqrt(var1 / n1) if n1 > 0 else 0
    se2 = math.sqrt(var2 / n2) if n2 > 0 else 0
    se_diff = math.sqrt(se1**2 + se2**2)
    
    # t-statistic
    if se_diff > 0:
        t_stat = (mean1 - mean2) / se_diff
    else:
        t_stat = 0
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    # Cohen's d
    if pooled_std > 0:
        cohen_d = (mean1 - mean2) / pooled_std
    else:
        cohen_d = 0
    
    # Eta-squared (approximation)
    eta_sq = (t_stat ** 2) / (t_stat ** 2 + df) if df > 0 else 0
    
    # p-value (approximation using t-distribution)
    # For large df, t ~ normal, so we can approximate
    if df > 30:
        # Use normal approximation
        p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    else:
        # Rough approximation for smaller samples
        p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    
    # Ensure p-value is reasonable
    p_val = max(1e-100, min(1.0, p_val))
    
    return {
        't': t_stat,
        'p': p_val,
        'd': cohen_d,
        'eta_sq': eta_sq
    }

def create_boxplot_svg(data1, data2, title, panel_label, output_lines):
    """Create SVG boxplot code."""
    
    if not data1 or not data2:
        return
    
    # Compute statistics
    stats = compute_stats(data1, data2)
    
    # Sort data for quartiles
    sorted1 = sorted(data1)
    sorted2 = sorted(data2)
    
    n1, n2 = len(sorted1), len(sorted2)
    
    def get_quartiles(data):
        n = len(data)
        q1_idx = n // 4
        median_idx = n // 2
        q3_idx = 3 * n // 4
        return {
            'min': min(data),
            'q1': data[q1_idx],
            'median': data[median_idx] if n % 2 == 1 else (data[median_idx-1] + data[median_idx]) / 2,
            'q3': data[q3_idx],
            'max': max(data)
        }
    
    q1 = get_quartiles(sorted1)
    q2 = get_quartiles(sorted2)
    
    # Boxplot positions
    x1, x2 = 1, 2
    box_width = 0.6
    
    # Y-axis range
    all_vals = data1 + data2
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min
    y_padding = y_range * 0.1
    y_plot_min = max(0, y_min - y_padding)
    y_plot_max = y_max + y_padding
    
    # Scale to SVG coordinates (assuming 400x300 plot area)
    plot_width = 400
    plot_height = 300
    margin = 50
    
    def scale_y(val):
        return margin + (plot_height - 2*margin) * (1 - (val - y_plot_min) / (y_plot_max - y_plot_min))
    
    def scale_x(val):
        return margin + (plot_width - 2*margin) * (val - 0.5) / 2.5
    
    # Draw boxplots
    # Exploitation (tan/orange)
    x1_scaled = scale_x(x1)
    box1_bottom = scale_y(q1['q1'])
    box1_top = scale_y(q1['q3'])
    median1_y = scale_y(q1['median'])
    
    # Exploration (light green)
    x2_scaled = scale_x(x2)
    box2_bottom = scale_y(q2['q1'])
    box2_top = scale_y(q2['q3'])
    median2_y = scale_y(q2['median'])
    
    output_lines.append(f'    <!-- {panel_label}. {title} -->')
    output_lines.append(f'    <!-- Exploitation: median={q1["median"]:.4f}, Q1={q1["q1"]:.4f}, Q3={q1["q3"]:.4f} -->')
    output_lines.append(f'    <!-- Exploration: median={q2["median"]:.4f}, Q1={q2["q1"]:.4f}, Q3={q2["q3"]:.4f} -->')
    if stats:
        p_str = f'{stats["p"]:.1e}' if stats["p"] < 0.001 else f'{stats["p"]:.3f}'
        output_lines.append(f'    <!-- Stats: t={stats["t"]:.2f}, p={p_str}, d={stats["d"]:.2f}, η²={stats["eta_sq"]:.2f} -->')
    output_lines.append('')

def main():
    """Generate the figure."""
    print("Loading data...")
    data = load_data()
    
    print(f"Loaded data:")
    print(f"  Exploitation intra-mean: {len(data['exploitation_intra_mean'])} values")
    print(f"  Exploration intra-mean: {len(data['exploration_intra_mean'])} values")
    print(f"  Exploitation inter-mean: {len(data['inter_phase_mean_exploitation'])} values")
    print(f"  Exploration inter-mean: {len(data['inter_phase_mean_exploration'])} values")
    
    # Create output file with statistics
    output_file = Path('output/recomputed_figure_stats.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RECOMPUTED FIGURE STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        # Panel A: Intra-Phase Mean
        f.write("Panel A: Intra-Phase Mean\n")
        f.write("-"*70 + "\n")
        exp_intra = data['exploitation_intra_mean']
        expl_intra = data['exploration_intra_mean']
        stats = compute_stats(exp_intra, expl_intra)
        if stats:
            p_str = f'{stats["p"]:.1e}' if stats["p"] < 0.001 else f'{stats["p"]:.3f}'
            f.write(f"t = {stats['t']:.2f}, p = {p_str}, d = {stats['d']:.2f}, η² = {stats['eta_sq']:.2f}\n")
        f.write(f"Exploitation: n={len(exp_intra)}, median={sorted(exp_intra)[len(exp_intra)//2]:.4f}\n")
        f.write(f"Exploration: n={len(expl_intra)}, median={sorted(expl_intra)[len(expl_intra)//2]:.4f}\n\n")
        
        # Panel B: Intra-Phase Variance
        f.write("Panel B: Intra-Phase Variance\n")
        f.write("-"*70 + "\n")
        exp_intra_var = data['exploitation_intra_variance']
        expl_intra_var = data['exploration_intra_variance']
        stats = compute_stats(exp_intra_var, expl_intra_var)
        if stats:
            p_str = f'{stats["p"]:.1e}' if stats["p"] < 0.001 else f'{stats["p"]:.3f}'
            f.write(f"t = {stats['t']:.2f}, p = {p_str}, d = {stats['d']:.2f}, η² = {stats['eta_sq']:.2f}\n")
        f.write(f"Exploitation: n={len(exp_intra_var)}, median={sorted(exp_intra_var)[len(exp_intra_var)//2]:.4f}\n")
        f.write(f"Exploration: n={len(expl_intra_var)}, median={sorted(expl_intra_var)[len(expl_intra_var)//2]:.4f}\n\n")
        
        # Panel C: Inter-Phase Mean
        f.write("Panel C: Inter-Phase Mean\n")
        f.write("-"*70 + "\n")
        exp_inter = data['inter_phase_mean_exploitation']
        expl_inter = data['inter_phase_mean_exploration']
        stats = compute_stats(exp_inter, expl_inter)
        if stats:
            p_str = f'{stats["p"]:.1e}' if stats["p"] < 0.001 else f'{stats["p"]:.3f}'
            f.write(f"t = {stats['t']:.2f}, p = {p_str}, d = {stats['d']:.2f}, η² = {stats['eta_sq']:.2f}\n")
        f.write(f"Exploitation: n={len(exp_inter)}, median={sorted(exp_inter)[len(exp_inter)//2]:.4f}\n")
        f.write(f"Exploration: n={len(expl_inter)}, median={sorted(expl_inter)[len(expl_inter)//2]:.4f}\n\n")
        
        # Panel D: Inter-Phase Variance
        f.write("Panel D: Inter-Phase Variance\n")
        f.write("-"*70 + "\n")
        exp_inter_var = data['inter_phase_variance_exploitation']
        expl_inter_var = data['inter_phase_variance_exploration']
        stats = compute_stats(exp_inter_var, expl_inter_var)
        if stats:
            p_str = f'{stats["p"]:.1e}' if stats["p"] < 0.001 else f'{stats["p"]:.3f}'
            f.write(f"t = {stats['t']:.2f}, p = {p_str}, d = {stats['d']:.2f}, η² = {stats['eta_sq']:.2f}\n")
        f.write(f"Exploitation: n={len(exp_inter_var)}, median={sorted(exp_inter_var)[len(exp_inter_var)//2]:.4f}\n")
        f.write(f"Exploration: n={len(expl_inter_var)}, median={sorted(expl_inter_var)[len(expl_inter_var)//2]:.4f}\n\n")
    
    print(f"\n✅ Statistics saved to: {output_file}")
    print("\nTo create the actual figure, run:")
    print("  python create_exploit_explore_metrics_figure.py")
    print("\nOr use the recomputed data by updating the script to use:")
    print("  output/recomputed_inter_phase_means.csv")

if __name__ == "__main__":
    main()






