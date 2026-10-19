"""
Test suite covering cleaning.py, feature_engineering.py, and model.py.
Run with: python3 tests/test_all.py
Everything here runs for real (no mocking needed - all deps are installed).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.cleaning import clean_pipeline, cap_outliers, coerce_types
from src.feature_engineering import engineer_features, encode_features
from src.model import split_and_scale, train_baseline, train_random_forest, evaluate_model

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_customer_churn.csv")

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


raw_df = pd.read_csv(DATA_PATH)

print("== cleaning ==")
clean_df, report = clean_pipeline(raw_df)
check("removes duplicate customer_ids", clean_df["customer_id"].duplicated().sum() == 0)
check("no missing values remain", clean_df.isna().sum().sum() == 0)
check("normalizes inconsistent contract_type casing", set(clean_df["contract_type"].unique()) == {"Month-to-month", "One year", "Two year"})
check("total_charges coerced to numeric", pd.api.types.is_numeric_dtype(clean_df["total_charges"]))
check("cleaning report tracks removed duplicates", report["duplicates_removed"] == 30)

print("== outlier capping ==")
test_df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
capped = cap_outliers(test_df, columns=["x"], n_std=2.0)
check("caps extreme outlier", capped["x"].max() < 1000)
check("preserves normal values", capped["x"].iloc[0] == 1)

print("== type coercion ==")
messy = pd.DataFrame({"total_charges": ["10.5", "", "20.0", "abc"]})
coerced = coerce_types(messy)
check("parses valid numeric strings", coerced["total_charges"].iloc[0] == 10.5)
check("blank string becomes NaN", pd.isna(coerced["total_charges"].iloc[1]))
check("invalid string becomes NaN (not a crash)", pd.isna(coerced["total_charges"].iloc[3]))

print("== feature engineering ==")
featured = engineer_features(clean_df)
check("adds avg_monthly_spend", "avg_monthly_spend" in featured.columns)
check("adds tenure_bucket", "tenure_bucket" in featured.columns)
check("adds high_support_contact flag", set(featured["high_support_contact"].unique()) <= {0, 1})
check("avg_monthly_spend has no inf/nan (tenure=0 handled)", np.isfinite(featured["avg_monthly_spend"]).all())

X, y = encode_features(featured)
check("target is binary 0/1", set(y.unique()) <= {0, 1})
check("customer_id excluded from features", "customer_id" not in X.columns)
check("categoricals one-hot encoded (no object dtype columns)", not any(X.dtypes == "object"))

print("== modeling ==")
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
check("train/test split proportions correct", abs(len(X_test) / len(X) - 0.2) < 0.01)

baseline = train_baseline(X_train, y_train)
baseline_metrics = evaluate_model(baseline, X_test, y_test)
check("baseline model produces valid accuracy", 0 <= baseline_metrics["accuracy"] <= 1)
check("baseline beats random guessing on ROC-AUC", baseline_metrics["roc_auc"] > 0.55)

rf = train_random_forest(X_train, y_train, n_estimators=50)
rf_metrics = evaluate_model(rf, X_test, y_test)
check("random forest produces valid accuracy", 0 <= rf_metrics["accuracy"] <= 1)
check("random forest beats random guessing on ROC-AUC", rf_metrics["roc_auc"] > 0.55)
check("random forest learned real signal (tenure_months should be a top feature)",
      "tenure_months" in pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(5).index)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
