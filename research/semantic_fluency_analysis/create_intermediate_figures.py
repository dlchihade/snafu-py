#!/usr/bin/env python3
"""
Generate intermediary figures that showcase raw metrics and participant-level context
prior to the mediation analyses. The outputs include:

1. Violin + swarm distribution of alpha power (alpha_NET_mean) with optional highlighted IDs.
2. Box + swarm plots for SVF counts and exploitation coherence ratio.
3. Scatter examples of LC integrity vs alpha power, and SVF count vs EE tradeoff.
4. Demographic summary (age histogram + Hoehn & Yahr distribution + disease duration).

All figures are saved under output/figures/intermediate/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, linregress, t as t_dist

from create_nature_quality_figures_real import setup_nature_style, try_load_and_merge_demographics

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "final_complete_disease_severity_mediation_data.csv"
OUT_DIR = ROOT / "output" / "figures" / "intermediate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IDs to highlight in swarm/violin plots (if present in the dataset)
HIGHLIGHT_IDS = {
    "PD00020": "Study patient",
    "HC00001": "Healthy control",  # will be ignored if not found
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "alpha_NET_mean",
        "norm_LC_avg",
        "SVF_count",
        "exploitation_coherence_ratio",
        "exploration_coherence_ratio",
        "Age",
        "disease_duration_symptoms",
        "disease_duration_diagnosis",
        "hoehn_yahr_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def base_style():
    colors = setup_nature_style()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    sns.set_style("whitegrid", {"axes.grid": False})
    return colors


def highlight_points(ax, df: pd.DataFrame, x_pos: float, y_col: str, color: str):
    for pid, label in HIGHLIGHT_IDS.items():
        row = df.loc[df["ID"] == pid]
        if row.empty or pd.isna(row[y_col]).all():
            continue
        ax.scatter(
            [x_pos],
            row[y_col],
            s=70,
            edgecolor="black",
            facecolor=color,
            zorder=5,
            label=f"{label} ({pid})",
        )


def violin_alpha(df: pd.DataFrame, colors: Dict[str, str]):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    sns.violinplot(
        data=df,
        y="alpha_NET_mean",
        x=["Alpha power"] * len(df),
        inner=None,
        color=colors["secondary"],
        cut=0,
        ax=ax,
    )
    sns.swarmplot(
        data=df,
        y="alpha_NET_mean",
        x=["Alpha power"] * len(df),
        color=colors["accent"],
        size=3.5,
        ax=ax,
    )
    highlight_points(ax, df, 0, "alpha_NET_mean", colors["highlight"])
    ax.set_ylabel("Alpha power (NET mean)")
    ax.set_xlabel("")
    ax.set_title("Alpha power distribution across participants")
    ax.set_ylim(df["alpha_NET_mean"].min() - 0.01, df["alpha_NET_mean"].max() + 0.01)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_power_violin.png", dpi=300)
    fig.savefig(OUT_DIR / "alpha_power_violin.pdf", dpi=300)
    plt.close(fig)


def box_swarm_metrics(df: pd.DataFrame, columns: Sequence[Tuple[str, str]], colors: Dict[str, str]):
    records = []
    for label, col in columns:
        if col not in df.columns:
            continue
        sub = df[["ID", col]].dropna()
        records.append(pd.DataFrame({"Metric": label, "Value": sub[col], "ID": sub["ID"]}))
    if not records:
        return
    plot_df = pd.concat(records, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    palette = [colors["accent"], colors["secondary"]]
    sns.boxplot(
        data=plot_df,
        x="Metric",
        y="Value",
        palette=palette[: len(columns)],
        ax=ax,
        showfliers=False,
    )
    sns.swarmplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Metric",
        dodge=False,
        palette=palette[: len(columns)],
        size=3.2,
        ax=ax,
    )
    for idx, (label, col) in enumerate(columns):
        highlight_points(ax, df, idx, col, colors["highlight"])
    ax.set_title("Behavioral summary: SVF and EE tradeoff")
    ax.set_xlabel("")
    ax.legend([], [], frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "svf_ee_boxswarm.png", dpi=300)
    fig.savefig(OUT_DIR / "svf_ee_boxswarm.pdf", dpi=300)
    plt.close(fig)


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, colors: Dict[str, str], out_name: str, title: str):
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        return
    r, p = pearsonr(data[x_col], data[y_col])
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    ax.scatter(
        data[x_col],
        data[y_col],
        s=26,
        alpha=0.9,
        color=colors["accent"],
        edgecolor="white",
        linewidth=0.4,
    )
    # regression line with confidence intervals
    x_clean = data[x_col].values
    y_clean = data[y_col].values
    
    # Fit regression using linregress for confidence interval calculation
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    
    # Generate x values for regression line
    xr = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    yr = slope * xr + intercept
    
    # Calculate confidence intervals
    n = len(x_clean)
    if n >= 3:
        # Calculate predicted values for original data
        y_pred = slope * x_clean + intercept
        
        # Calculate residuals and standard error of residuals
        residuals = y_clean - y_pred
        mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
        se_residual = np.sqrt(mse)
        
        # Calculate standard error for prediction
        x_mean = np.mean(x_clean)
        ssx = np.sum((x_clean - x_mean)**2)
        
        # For each x in xr, calculate confidence interval
        se_pred = se_residual * np.sqrt(1.0/n + (xr - x_mean)**2 / ssx)
        
        # 95% confidence interval (t-distribution with n-2 degrees of freedom)
        t_crit = t_dist.ppf(0.975, n - 2)
        ci_upper = yr + t_crit * se_pred
        ci_lower = yr - t_crit * se_pred
        
        # Fill between upper and lower confidence bounds (behind the line)
        ax.fill_between(xr, ci_lower, ci_upper, color='gray', alpha=0.25, zorder=0)
    
    # Plot regression line on top
    ax.plot(xr, yr, color=colors["secondary"], linewidth=1.2, zorder=1)
    
    # Better axis labels for common columns
    label_map = {
        "norm_LC_avg": "LC neuromelanin integrity",
        "alpha_NET_mean": "MEG alpha power (NET mean)",
        "SVF_count": "SVF count",
        "exploitation_coherence_ratio": "Exploitation coherence ratio",
        "exploration_coherence_ratio": "Exploration coherence ratio",
    }
    xlabel = label_map.get(x_col, x_col.replace("_", " ").title())
    ylabel = label_map.get(y_col, y_col.replace("_", " ").title())
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.98,
        0.02,
        f"r = {r:.2f}\np = {p:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=300)
    fig.savefig(OUT_DIR / out_name.replace(".png", ".pdf"), dpi=300)
    plt.close(fig)


def demographics_figure(df: pd.DataFrame, colors: Dict[str, str]):
    # Create a 2x3 grid for more comprehensive demographics
    # Increased figure size and adjusted spacing for better balance
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    axes = axes.flatten()

    # Panel 1: Age histogram
    age_data = df["Age"].dropna()
    if len(age_data) > 0:
        sns.histplot(age_data, bins=12, color=colors["accent"], ax=axes[0], edgecolor="white", linewidth=0.5)
        mean_age = age_data.mean()
        axes[0].axvline(mean_age, color=colors["primary"], linestyle="--", linewidth=1.5, label=f"μ={mean_age:.1f}")
        axes[0].set_title("Age distribution", fontsize=13, fontweight="bold", pad=8)
        axes[0].set_xlabel("Age (years)", fontsize=11, fontweight="normal")
        axes[0].set_ylabel("Participants", fontsize=11, fontweight="normal")
        axes[0].legend(frameon=False, fontsize=10)
        axes[0].tick_params(labelsize=10, width=0.8, length=4)
    else:
        axes[0].axis("off")

    # Panel 2: Hoehn & Yahr stages
    if "hoehn_yahr_score" in df.columns:
        hy_data = df["hoehn_yahr_score"].dropna()
        if len(hy_data) > 0:
            counts = hy_data.round().astype(int).value_counts().sort_index()
            axes[1].bar(counts.index.astype(str), counts.values, color=colors["secondary"], 
                       edgecolor="white", linewidth=0.8)
            axes[1].set_title("Hoehn & Yahr stages", fontsize=13, fontweight="bold", pad=8)
            axes[1].set_xlabel("Stage", fontsize=11, fontweight="normal")
            axes[1].set_ylabel("Participants", fontsize=11, fontweight="normal")
            axes[1].tick_params(labelsize=10, width=0.8, length=4)
        else:
            axes[1].axis("off")
    else:
        axes[1].axis("off")

    # Panel 3: Disease duration scatter
    if {"disease_duration_symptoms", "disease_duration_diagnosis"} <= set(df.columns):
        mask = df["disease_duration_symptoms"].notna() & df["disease_duration_diagnosis"].notna()
        if mask.sum() > 0:
            axes[2].scatter(
                df.loc[mask, "disease_duration_symptoms"],
                df.loc[mask, "disease_duration_diagnosis"],
                s=35,
                color=colors["highlight"],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.75,
            )
            max_val = max(df.loc[mask, "disease_duration_symptoms"].max(), 
                         df.loc[mask, "disease_duration_diagnosis"].max())
            axes[2].plot([0, max_val], [0, max_val], "--", color="gray", linewidth=1.0, alpha=0.6)
            axes[2].set_xlabel("Duration since symptoms (years)", fontsize=11, fontweight="normal")
            axes[2].set_ylabel("Duration since diagnosis (years)", fontsize=11, fontweight="normal")
            axes[2].set_title("Disease duration profile", fontsize=13, fontweight="bold", pad=8)
            axes[2].tick_params(labelsize=10, width=0.8, length=4)
        else:
            axes[2].axis("off")
    else:
        axes[2].axis("off")

    # Panel 4: Sex/Gender distribution
    sex_col = None
    for col in ["Sex", "Gender", "sex", "gender"]:
        if col in df.columns:
            sex_col = col
            break
    if sex_col:
        sex_data = df[sex_col].dropna().astype(str)
        if len(sex_data) > 0:
            counts = sex_data.value_counts()
            axes[3].bar(counts.index, counts.values, color=colors["accent"], 
                       edgecolor="white", linewidth=0.8)
            axes[3].set_title("Sex distribution", fontsize=13, fontweight="bold", pad=8)
            axes[3].set_ylabel("Participants", fontsize=11, fontweight="normal")
            axes[3].tick_params(labelsize=10, width=0.8, length=4)
        else:
            axes[3].axis("off")
    else:
        axes[3].axis("off")

    # Panel 5: MoCA scores (cognitive_measure_2)
    moca_col = None
    for col in ["cognitive_measure_2", "MoCA", "moca_score", "MoCA_score", "MoCA_total"]:
        if col in df.columns:
            moca_col = col
            break
    if moca_col:
        moca_data = pd.to_numeric(df[moca_col], errors="coerce").dropna()
        if len(moca_data) > 0:
            sns.histplot(moca_data, bins=12, color=colors["neutral"], ax=axes[4], 
                        edgecolor="white", linewidth=0.5)
            mean_moca = moca_data.mean()
            axes[4].axvline(mean_moca, color=colors["primary"], linestyle="--", 
                          linewidth=1.5, label=f"μ={mean_moca:.1f}")
            axes[4].set_title("MoCA scores", fontsize=13, fontweight="bold", pad=8)
            axes[4].set_xlabel("MoCA score", fontsize=11, fontweight="normal")
            axes[4].set_ylabel("Participants", fontsize=11, fontweight="normal")
            axes[4].legend(frameon=False, fontsize=10)
            axes[4].tick_params(labelsize=10, width=0.8, length=4)
        else:
            axes[4].axis("off")
    else:
        axes[4].axis("off")

    # Panel 6: SVF count distribution
    if "SVF_count" in df.columns:
        svf_data = pd.to_numeric(df["SVF_count"], errors="coerce").dropna()
        if len(svf_data) > 0:
            sns.histplot(svf_data, bins=12, color=colors["accent"], ax=axes[5], 
                        edgecolor="white", linewidth=0.5)
            mean_svf = svf_data.mean()
            axes[5].axvline(mean_svf, color=colors["primary"], linestyle="--", 
                          linewidth=1.5, label=f"μ={mean_svf:.1f}")
            axes[5].set_title("Semantic Verbal Fluency", fontsize=13, fontweight="bold", pad=8)
            axes[5].set_xlabel("Word count", fontsize=11, fontweight="normal")
            axes[5].set_ylabel("Participants", fontsize=11, fontweight="normal")
            axes[5].legend(frameon=False, fontsize=10)
            axes[5].tick_params(labelsize=10, width=0.8, length=4)
        else:
            axes[5].axis("off")
    else:
        axes[5].axis("off")

    # Remove figure title and rebalance spacing
    fig.tight_layout(rect=[0, 0, 1, 1], pad=2.5)
    fig.subplots_adjust(wspace=0.35, hspace=0.4)
    fig.savefig(OUT_DIR / "participant_demographics.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(OUT_DIR / "participant_demographics.pdf", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    df = load_data(DATA_PATH)
    colors = base_style()

    violin_alpha(df, colors)
    box_swarm_metrics(
        df,
        [
            ("SVF Count", "SVF_count"),
            ("EE tradeoff", "exploitation_coherence_ratio"),
        ],
        colors,
    )
    scatter_plot(
        df,
        "norm_LC_avg",
        "alpha_NET_mean",
        colors,
        out_name="lc_vs_alpha.png",
        title="LC integrity vs α-power",
    )
    scatter_plot(
        df,
        "SVF_count",
        "exploitation_coherence_ratio",
        colors,
        out_name="svf_vs_ee.png",
        title="SVF performance vs EE metric",
    )
    # Use merged demographics data to match other figures
    df_demo = try_load_and_merge_demographics(df)
    demographics_figure(df_demo, colors)
    print(f"Saved intermediary figures to {OUT_DIR}")


if __name__ == "__main__":
    main()


