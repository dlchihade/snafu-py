#!/usr/bin/env python3
"""
One-stop script to:
 1) Load final comprehensive mediation dataset (with covariates)
 2) Run mediation analyses (SVF count and EE metric) with Age and optional disease stage
 3) Generate publication-style Figures 1–4 using real data
 4) Compose combined outputs (multi-page PDF and 2x2 grid)

Inputs expected (first found among these):
 - final_clean_mediation_data.csv
 - final_comprehensive_mediation_data.csv
 - final_complete_mediation_data.csv

Outputs:
 - output/NATURE_REAL_figure[1-4]_*.{png,pdf}
 - output/NATURE_REAL_all_figures.pdf
 - output/NATURE_REAL_all_figures_2x2.{png,pdf}
"""

from pathlib import Path
import pandas as pd

# Local imports (existing utilities)
from create_nature_quality_figures_real import (
    compute_real_metrics,
    fig1_exploration_exploitation,
    fig2_phase_coherence,
    fig3_meg_correlations,
    fig4_comprehensive,
    setup_nature_style,
)
import statsmodels.api as sm
import compose_all_figures as compose_pages
import compose_all_figures_grid as compose_grid


def load_final_dataframe() -> pd.DataFrame:
    root = Path(__file__).parent
    candidates = [
        root / 'final_clean_mediation_data.csv',
        root / 'final_comprehensive_mediation_data.csv',
        root / 'final_complete_mediation_data.csv',
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("No final mediation dataset found in semantic_fluency_analysis/.")


def _fit_ols(y: pd.Series, X: pd.DataFrame):
    X_ = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_, missing="drop").fit()


def mediation_flexible(df: pd.DataFrame, x_col: str, m_col: str, y_col: str, covariates):
    use = df[[x_col, m_col, y_col] + covariates].dropna().copy()
    n = len(use)
    if n < 10:
        raise ValueError("Not enough complete rows for mediation.")

    X = use[[x_col] + covariates]
    M = use[m_col]
    Y = use[y_col]

    a_mod = _fit_ols(M, X)
    a = a_mod.params[x_col]

    bc_mod = _fit_ols(Y, use[[x_col, m_col] + covariates])
    b = bc_mod.params[m_col]
    cprime = bc_mod.params[x_col]

    c_mod = _fit_ols(Y, X)
    c_total = c_mod.params[x_col]

    indirect = a * b
    prop_med = (indirect / c_total) if c_total != 0 else float('nan')

    # Bootstrap ind. effect CI
    n_boot = 5000
    rng = __import__('numpy').random.default_rng(42)
    idx = __import__('numpy').arange(n)
    ab = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        d = use.iloc[s]
        try:
            a_i = _fit_ols(d[m_col], d[[x_col] + covariates]).params[x_col]
            b_i = _fit_ols(d[y_col], d[[x_col, m_col] + covariates]).params[m_col]
            ab.append(a_i * b_i)
        except Exception:
            continue
    import numpy as np
    ab = np.array(ab)
    lo = np.quantile(ab, 0.025) if ab.size else float('nan')
    hi = np.quantile(ab, 0.975) if ab.size else float('nan')

    return {
        'n': n,
        'a': a, 'b': b, 'c': c_total, "c'": cprime,
        'indirect_ab': indirect, 'direct_cprime': cprime, 'total_c': c_total,
        'prop_mediated': prop_med, 'indirect_ci': (lo, hi),
    }


def run_mediations(df: pd.DataFrame):
    cols_needed = ['norm_LC_avg', 'alpha_NET_mean', 'SVF_count',
                   'exploitation_coherence_ratio', 'Age']
    for c in cols_needed:
        if c not in df.columns:
            raise KeyError(f"Missing column in dataset: {c}")

    def wrap(x_col, m_col, y_col, covars, label):
        res = mediation_flexible(df, x_col=x_col, m_col=m_col, y_col=y_col, covariates=covars)
        lo, hi = res['indirect_ci']
        cprime_val = res["c'"]
        print(
            f"\nMediation {label} (N={res['n']})\n"
            f" a: {res['a']:.4f}\n b: {res['b']:.4f}\n c: {res['c']:.4f}\n c': {cprime_val:.4f}\n"
            f" Indirect a×b: {res['indirect_ab']:.4f}  CI[{lo:.4f}, {hi:.4f}]  Proportion: {res['prop_mediated']:.4f}"
        )
        return res

    cov_age = ['Age']
    cov_age_disease = ['Age'] + ([c for c in ['hoehn_yahr_score'] if c in df.columns])

    svf_age = wrap('norm_LC_avg', 'alpha_NET_mean', 'SVF_count', cov_age, 'SVF (Age)')
    ee_age = wrap('norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', cov_age, 'EE metric (Age)')

    # Optional disease-stage adjusted
    if len(cov_age_disease) > 1:
        svf_dis = wrap('norm_LC_avg', 'alpha_NET_mean', 'SVF_count', cov_age_disease, 'SVF (Age + disease)')
        ee_dis = wrap('norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', cov_age_disease, 'EE metric (Age + disease)')
        return {'svf_age': svf_age, 'ee_age': ee_age, 'svf_dis': svf_dis, 'ee_dis': ee_dis}
    else:
        return {'svf_age': svf_age, 'ee_age': ee_age}


def main():
    print('📥 Loading final mediation dataframe...')
    final_df = load_final_dataframe()
    print(f'✅ Loaded {len(final_df)} rows; columns: {list(final_df.columns)[:10]}...')

    print('\n🧠 Running mediation analyses (Age and optional disease stage)...')
    _ = run_mediations(final_df)

    print('\n📊 Computing metrics and generating Figures 1–4...')
    colors = setup_nature_style()
    df_metrics = compute_real_metrics()
    if df_metrics.empty:
        raise RuntimeError('No valid participants for figure generation.')
    fig1_exploration_exploitation(df_metrics, colors)
    fig2_phase_coherence(df_metrics, colors)
    fig3_meg_correlations(df_metrics, colors)
    fig4_comprehensive(df_metrics, colors)

    print('\n🗂️  Composing combined outputs...')
    compose_pages.main()
    compose_grid.main()
    print('🎉 All done.')


if __name__ == '__main__':
    main()


