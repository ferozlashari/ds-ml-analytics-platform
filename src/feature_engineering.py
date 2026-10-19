"""
Feature engineering: derives new predictive features and encodes categoricals.
"""

from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average monthly spend so far - a stronger churn signal than total_charges alone.
    df["avg_monthly_spend"] = (df["total_charges"] / df["tenure_months"].replace(0, 1)).round(2)

    # Tenure buckets - captures non-linear relationship between tenure and churn.
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
    )

    # High-support-contact flag - frequent support calls often precede churn.
    df["high_support_contact"] = (df["support_calls"] >= 3).astype(int)

    # Is on the most churn-prone contract type.
    df["is_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)

    return df


def encode_features(df: pd.DataFrame, target_col: str = "churn") -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categoricals and return (X, y) ready for modeling."""
    df = df.copy()
    y = (df[target_col] == "Yes").astype(int)

    drop_cols = [target_col, "customer_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y
