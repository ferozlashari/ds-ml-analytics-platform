"""
Exploratory Data Analysis: summary statistics and visualizations.
Plots are saved as PNG files rather than shown interactively, so this runs
headlessly in any environment (CI, servers, this sandbox).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend - no display required
import matplotlib.pyplot as plt
import pandas as pd


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose()


def churn_rate_by_category(df: pd.DataFrame, category_col: str, target_col: str = "churn") -> pd.Series:
    return df.groupby(category_col)[target_col].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)


def plot_churn_distribution(df: pd.DataFrame, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    df["churn"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_churn_by_contract(df: pd.DataFrame, out_path: str) -> str:
    rates = churn_rate_by_category(df, "contract_type")
    fig, ax = plt.subplots(figsize=(6, 4))
    rates.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_numeric_correlations(df: pd.DataFrame, out_path: str) -> str:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_tenure_distribution(df: pd.DataFrame, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    df[df["churn"] == "Yes"]["tenure_months"].plot(kind="hist", bins=20, alpha=0.6, label="Churned", ax=ax)
    df[df["churn"] == "No"]["tenure_months"].plot(kind="hist", bins=20, alpha=0.6, label="Retained", ax=ax)
    ax.set_title("Tenure Distribution by Churn Status")
    ax.set_xlabel("Tenure (months)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def run_eda(df: pd.DataFrame, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "churn_distribution": plot_churn_distribution(df, os.path.join(output_dir, "churn_distribution.png")),
        "churn_by_contract": plot_churn_by_contract(df, os.path.join(output_dir, "churn_by_contract.png")),
        "correlation_matrix": plot_numeric_correlations(df, os.path.join(output_dir, "correlation_matrix.png")),
        "tenure_distribution": plot_tenure_distribution(df, os.path.join(output_dir, "tenure_distribution.png")),
    }
    return paths
