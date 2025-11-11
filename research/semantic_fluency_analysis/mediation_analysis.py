#!/usr/bin/env python3
"""
Mediation analysis on real-data metrics
Predictor: norm_LC_avg (LC)
Mediator: alpha_NET_mean (alpha PSD)
Outcome: exploitation_coherence_ratio
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist


def ols(X: np.ndarray, y: np.ndarray):
    """Ordinary least squares with standard errors and p-values.
    X should include a column of ones for intercept.
    Returns: coef, se, t, p, s2
    """
    n, p = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - p, 1)
    rss = float(resid.T @ resid)
    s2 = rss / dof
    cov = s2 * XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vals = np.where(se > 0, beta / se, np.nan)
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), dof))
    return beta, se, t_vals, p_vals, s2


def mediation(df: pd.DataFrame, B: int = 5000, random_state: int = 42):
    df = df[['norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', 'Age']].dropna()
    if len(df) < 10:
        raise ValueError('Not enough data for mediation analysis after dropna.')

    X = df['norm_LC_avg'].to_numpy().astype(float)
    M = df['alpha_NET_mean'].to_numpy().astype(float)
    Y = df['exploitation_coherence_ratio'].to_numpy().astype(float)
    C = df['Age'].to_numpy().astype(float)

    # Standardize for interpretability (optional, does not change inference on indirect effect sign)
    Xz = (X - X.mean()) / X.std(ddof=1)
    Mz = (M - M.mean()) / M.std(ddof=1)
    Yz = (Y - Y.mean()) / Y.std(ddof=1)
    Cz = (C - C.mean()) / C.std(ddof=1)

    ones = np.ones_like(Xz)

    # a path: M ~ X + C (Age)
    beta_a, se_a, t_a, p_a, _ = ols(np.column_stack([ones, Xz, Cz]), Mz)
    a = float(beta_a[1])

    # b & c' paths: Y ~ X + M + C (Age)
    beta_b, se_b, t_b, p_b, _ = ols(np.column_stack([ones, Xz, Mz, Cz]), Yz)
    c_prime = float(beta_b[1])  # direct effect of X
    b = float(beta_b[2])        # effect of M

    # c path: Y ~ X + C (Age)
    beta_c, se_c, t_c, p_c, _ = ols(np.column_stack([ones, Xz, Cz]), Yz)
    c_total = float(beta_c[1])

    indirect = a * b
    prop_med = (indirect / c_total) if c_total != 0 else np.nan

    # Bootstrap CI for indirect effect
    rng = np.random.default_rng(random_state)
    ab_boot = np.empty(B, dtype=float)
    n = len(df)
    for i in range(B):
        idx = rng.integers(0, n, n)
        Xb, Mb, Yb, Cb = Xz[idx], Mz[idx], Yz[idx], Cz[idx]
        # a with covariate
        a_b = ols(np.column_stack([np.ones_like(Xb), Xb, Cb]), Mb)[0][1]
        # b with covariate
        b_b = ols(np.column_stack([np.ones_like(Xb), Xb, Mb, Cb]), Yb)[0][2]
        ab_boot[i] = a_b * b_b
    ci_low, ci_high = np.percentile(ab_boot, [2.5, 97.5])

    results = {
        'N': n,
        'a (M~X)': a,
        'b (Y~M|X)': b,
        "c' (Y~X|M)": c_prime,
        'c (Y~X)': c_total,
        'indirect (a*b)': indirect,
        'ab_ci_low_95': ci_low,
        'ab_ci_high_95': ci_high,
        'prop_mediated': prop_med,
        'p_a': float(p_a[1]),
        'p_b': float(p_b[2]),
        'p_c_total': float(p_c[1]),
    }
    return results


def main():
    metrics_path = Path('output/NATURE_REAL_metrics.csv')
    if not metrics_path.exists():
        raise FileNotFoundError(f'Metrics not found: {metrics_path}. Run create_nature_quality_figures_real.py first.')
    df = pd.read_csv(metrics_path)

    res = mediation(df, B=5000)

    out_txt = Path('output/mediation_nmlc_alpha_exploitation_adj_age.txt')
    cprime = res["c' (Y~X|M)"]
    lines = [
        'Mediation (Age-adjusted): X=norm_LC_avg (predictor), M=alpha_NET_mean (mediator), Y=exploitation_coherence_ratio (outcome); covariate: Age\n',
        f"N = {res['N']}\n",
        f"a (M~X) = {res['a (M~X)']:.4f}  (p={res['p_a']:.4f})\n",
        f"b (Y~M|X) = {res['b (Y~M|X)']:.4f}  (p={res['p_b']:.4f})\n",
        f"c' (direct Y~X|M) = {cprime:.4f}\n",
        f"c (total Y~X) = {res['c (Y~X)']:.4f}  (p={res['p_c_total']:.4f})\n",
        f"indirect (a*b) = {res['indirect (a*b)']:.4f}  95% CI [{res['ab_ci_low_95']:.4f}, {res['ab_ci_high_95']:.4f}]\n",
        f"proportion mediated = {res['prop_mediated']:.4f}\n",
    ]
    out_txt.write_text(''.join(lines))
    print(''.join(lines))
    print(f"Saved: {out_txt}")


if __name__ == '__main__':
    main()
