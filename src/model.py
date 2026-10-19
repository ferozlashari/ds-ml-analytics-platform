"""
Model training and evaluation: trains a baseline and a tuned classifier,
compares them, and reports standard classification metrics + feature importance.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def train_baseline(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=8, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def feature_importance(model: RandomForestClassifier, feature_names: list[str], top_n: int = 10) -> pd.Series:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


def plot_confusion_matrix(model, X_test, y_test, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["No churn", "Churn"], ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix - Random Forest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_roc_curve(model, X_test, y_test, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curve - Random Forest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_feature_importance(importances: pd.Series, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Top Feature Importances - Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def run_modeling(X: pd.DataFrame, y: pd.Series, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    baseline = train_baseline(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test)

    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)

    plot_confusion_matrix(rf, X_test, y_test, os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc_curve(rf, X_test, y_test, os.path.join(output_dir, "roc_curve.png"))
    importances = feature_importance(rf, X.columns.tolist())
    plot_feature_importance(importances, os.path.join(output_dir, "feature_importance.png"))

    results = {
        "baseline_logistic_regression": baseline_metrics,
        "random_forest": rf_metrics,
        "best_model": "random_forest" if rf_metrics["f1_score"] >= baseline_metrics["f1_score"] else "logistic_regression",
        "top_features": importances.to_dict(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
