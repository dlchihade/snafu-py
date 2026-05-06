#!/usr/bin/env python3
"""Create publication-quality mediation figures with disease stage as covariate"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

from mediation_figures_nature import plot_mediation_figure_shared

# Display label for outcome when using exploitation_coherence_ratio (same computation as before).
OUTCOME_LABEL_COHERENCE = 'Exploitation coherence'


def _disease_result_to_shared(result: dict) -> dict:
    """Map mediation_disease_stage_adjusted keys to plot_mediation_figure_shared keys."""
    return {
        'a': result['a'],
        'b': result['b'],
        'c': result['c'],
        "c'": result["c'"],
        'indirect': result['ab'],
        'ci_low': result['ci_lower'],
        'ci_high': result['ci_upper'],
        'N': result['n'],
    }


def ols(X: np.ndarray, y: np.ndarray):
    """Ordinary least squares regression"""
    n, p = X.shape
    XtX = X.T @ X
    beta = np.linalg.inv(XtX) @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - p, 1)
    s2 = float(resid.T @ resid) / dof
    cov = s2 * np.linalg.inv(XtX)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vals = np.where(se > 0, beta / se, np.nan)
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), dof))
    return beta, se, t_vals, p_vals

def mediation_disease_stage_adjusted(df: pd.DataFrame, outcome_col: str, B: int = 5000, seed: int = 42):
    """Mediation analysis with disease stage (Hoehn and Yahr score) as covariate"""
    try:
        df = df[['norm_LC_avg', 'alpha_NET_mean', outcome_col, 'hoehn_yahr_score']].dropna()
        
        print(f"Disease stage adjusted mediation analysis: {len(df)} participants")
        
        X = df['norm_LC_avg'].to_numpy(float)
        M = df['alpha_NET_mean'].to_numpy(float)
        Y = df[outcome_col].to_numpy(float)
        C = df['hoehn_yahr_score'].to_numpy(float)  # Disease stage as covariate
        
        # z-score
        Xz = (X - X.mean()) / X.std(ddof=1)
        Mz = (M - M.mean()) / M.std(ddof=1)
        Yz = (Y - Y.mean()) / Y.std(ddof=1)
        Cz = (C - C.mean()) / C.std(ddof=1)
        ones = np.ones_like(Xz)
        
        # a: M ~ X + Disease Stage
        ba, _, ta, pa = ols(np.column_stack([ones, Xz, Cz]), Mz)
        a = float(ba[1]); p_a = float(pa[1])
        
        # b & c': Y ~ X + M + Disease Stage
        bb, _, tb, pb = ols(np.column_stack([ones, Xz, Mz, Cz]), Yz)
        c_prime = float(bb[1]); b = float(bb[2]); p_b = float(pb[2])
        
        # c: Y ~ X + Disease Stage
        bc, _, tc, pc = ols(np.column_stack([ones, Xz, Cz]), Yz)
        c_total = float(bc[1]); p_c = float(pc[1])
        
        # bootstrap for ab
        rng = np.random.default_rng(seed)
        N = len(df)
        ab = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, N, N)
            X_boot = Xz[idx]
            M_boot = Mz[idx]
            Y_boot = Yz[idx]
            C_boot = Cz[idx]
            ones_boot = np.ones_like(X_boot)
            
            # a: M ~ X + Disease Stage
            ba_boot, _, _, _ = ols(np.column_stack([ones_boot, X_boot, C_boot]), M_boot)
            a_boot = float(ba_boot[1])
            
            # b: Y ~ X + M + Disease Stage
            bb_boot, _, _, _ = ols(np.column_stack([ones_boot, X_boot, M_boot, C_boot]), Y_boot)
            b_boot = float(bb_boot[2])
            
            ab[i] = a_boot * b_boot
        
        # confidence interval
        ab_sorted = np.sort(ab)
        ci_lower = ab_sorted[int(0.025 * B)]
        ci_upper = ab_sorted[int(0.975 * B)]
        
        result = {
            'a': a, 'p_a': p_a,
            'b': b, 'p_b': p_b,
            'c': c_total, 'p_c': p_c,
            "c'": c_prime, 'p_c_prime': float(pb[1]),
            'ab': a * b,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(df)
        }
        
        return result
        
    except Exception as e:
        print(f"Error in disease stage adjusted mediation: {e}")
        return None

def create_mediation_figure_publication(result: dict, outcome_type: str, save_path: str):
    """Same layout, colors, and margins as age-adjusted figures (mediation_figures_nature)."""
    ot = 'coherence' if outcome_type == 'ee' else 'svf'
    plot_mediation_figure_shared(
        _disease_result_to_shared(result),
        Path(save_path),
        outcome_type=ot,
        subtitle='Disease stage-adjusted',
        color_scheme='disease_stage',
    )

def main():
    """Generate publication-quality mediation figures with disease stage adjustment"""
    print("Creating publication-quality disease stage-adjusted mediation figures...")
    
    # Load data with disease severity
    df = pd.read_csv('final_complete_disease_severity_mediation_data.csv')
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # SVF Count mediation
    result_svf = mediation_disease_stage_adjusted(df, 'SVF_count')
    if result_svf:
        create_mediation_figure_publication(
            result_svf, 'svf', output_dir / 'mediation_svf_disease_stage_publication.png',
        )
        print(f"SVF Count mediation (disease stage adjusted): N = {result_svf['n']}")
    
    # Exploitation coherence (exploitation_coherence_ratio) mediation
    result_ee = mediation_disease_stage_adjusted(df, 'exploitation_coherence_ratio')
    if result_ee:
        create_mediation_figure_publication(
            result_ee, 'ee', output_dir / 'mediation_ee_disease_stage_publication.png',
        )
        print(f"{OUTCOME_LABEL_COHERENCE} mediation (disease stage adjusted): N = {result_ee['n']}")
    
    print("\nCreated publication-quality disease stage-adjusted mediation figures:")
    print(" - output/mediation_svf_disease_stage_publication.(png|pdf)")
    print(" - output/mediation_ee_disease_stage_publication.(png|pdf)")

if __name__ == '__main__':
    main()

