"""
Data cleaning: handles missing values, duplicates, type coercion,
inconsistent categorical labels, and outlier capping.

Each function is pure (takes a DataFrame, returns a new DataFrame) so the
whole cleaning pipeline is independently testable step by step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"] + ["customer_id"])
    # Duplicate customer_id with identical data is a true duplicate row; drop by customer_id too.
    df = df.drop_duplicates(subset="customer_id", keep="first")
    after = len(df)
    df.attrs["duplicates_removed"] = before - after
    return df.reset_index(drop=True)


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent casing/whitespace in categorical columns, e.g. 'ONE YEAR' -> 'One year'."""
    df = df.copy()
    if "contract_type" in df.columns:
        df["contract_type"] = df["contract_type"].str.strip().str.lower().map(
            {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}
        )
    if "payment_method" in df.columns:
        df["payment_method"] = df["payment_method"].str.strip()
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fix wrong dtypes: total_charges arrives as a string (with some blanks)."""
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric columns with median, categorical columns with mode."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    missing_report = {}
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            missing_report[col] = int(n_missing)

    for col in categorical_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)
            missing_report[col] = int(n_missing)

    df.attrs["missing_values_imputed"] = missing_report
    return df


def cap_outliers(df: pd.DataFrame, columns: list[str], n_std: float = 3.0) -> pd.DataFrame:
    """Cap numeric outliers at mean +/- n_std standard deviations (winsorization)."""
    df = df.copy()
    outliers_capped = {}
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        lower, upper = mean - n_std * std, mean + n_std * std
        n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower, upper)
        if n_capped > 0:
            outliers_capped[col] = n_capped
    df.attrs["outliers_capped"] = outliers_capped
    return df


def clean_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline in order, returning the cleaned df and a report dict."""
    report = {"rows_before": len(df)}

    df = drop_duplicates(df)
    report["duplicates_removed"] = df.attrs.get("duplicates_removed", 0)

    df = normalize_categories(df)
    df = coerce_types(df)

    df = handle_missing_values(df)
    report["missing_values_imputed"] = df.attrs.get("missing_values_imputed", {})

    df = cap_outliers(df, columns=["monthly_charges", "total_charges", "age"])
    report["outliers_capped"] = df.attrs.get("outliers_capped", {})

    report["rows_after"] = len(df)
    return df, report
