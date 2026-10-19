"""
End-to-end pipeline: load -> clean -> EDA -> feature engineer -> model -> evaluate.

Run with:  python3 run_pipeline.py
Outputs land in outputs/ (plots/*.png, metrics.json, cleaning_report.json).
"""

from __future__ import annotations

import json
import os

import pandas as pd

from src.cleaning import clean_pipeline
from src.eda import run_eda
from src.feature_engineering import encode_features, engineer_features
from src.model import run_modeling

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw_customer_churn.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main(raw_data_path: str = RAW_DATA_PATH, output_dir: str = OUTPUT_DIR) -> dict:
    print("1/5 Loading raw data...")
    df = pd.read_csv(raw_data_path)
    print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")

    print("2/5 Cleaning data...")
    clean_df, cleaning_report = clean_pipeline(df)
    with open(os.path.join(output_dir, "cleaning_report.json"), "w") as f:
        json.dump(cleaning_report, f, indent=2)
    print(f"    {cleaning_report}")

    print("3/5 Running EDA...")
    eda_paths = run_eda(clean_df, os.path.join(output_dir, "plots"))
    print(f"    Saved {len(eda_paths)} EDA plots to {output_dir}/plots/")

    print("4/5 Engineering features...")
    featured_df = engineer_features(clean_df)
    X, y = encode_features(featured_df)
    print(f"    {X.shape[1]} features ready, churn rate: {y.mean():.1%}")

    print("5/5 Training and evaluating models...")
    results = run_modeling(X, y, os.path.join(output_dir, "plots"))
    print(f"    Best model: {results['best_model']}")
    print(f"    Random Forest metrics: {results['random_forest']}")

    return results


if __name__ == "__main__":
    main()
