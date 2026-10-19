# Data Science & ML Analytics Platform

A complete, end-to-end data science pipeline on a customer churn dataset: **cleaning → EDA → feature engineering → model training → evaluation → visualization**. Every step in this README was actually executed — this isn't a template, it's a real pipeline run on real (synthetic) messy data.

## Dataset

`data/raw_customer_churn.csv` — 2,030 rows of synthetic telecom customer data, intentionally messy to demonstrate real cleaning work:
- Missing values in `age` (100) and `payment_method` (60)
- 30 duplicate customer records
- Inconsistent categorical casing (`"ONE YEAR"`, `"month-to-month"`, `"One year"`)
- `total_charges` stored as strings with blanks
- 10 extreme outliers injected into `monthly_charges`

## Pipeline

```
raw CSV → cleaning.py → eda.py → feature_engineering.py → model.py → outputs/
```

1. **Cleaning** (`src/cleaning.py`): drops duplicates, normalizes inconsistent category labels, coerces string columns to numeric, imputes missing values (median/mode), winsorizes outliers (3σ cap).
2. **EDA** (`src/eda.py`): churn distribution, churn rate by contract type, correlation matrix, tenure distribution by churn status — 4 plots saved to `outputs/plots/`.
3. **Feature engineering** (`src/feature_engineering.py`): derives `avg_monthly_spend`, `tenure_bucket`, `high_support_contact` flag, `is_month_to_month` flag; one-hot encodes categoricals.
4. **Modeling** (`src/model.py`): trains a Logistic Regression baseline and a Random Forest, evaluates both (accuracy, precision, recall, F1, ROC-AUC), picks the better model, and plots a confusion matrix, ROC curve, and feature importance chart.

## Actual results from the last run

```json
{
  "baseline_logistic_regression": {"accuracy": 0.6825, "precision": 0.6648, "recall": 0.6398, "f1_score": 0.6521, "roc_auc": 0.744},
  "random_forest":                {"accuracy": 0.6675, "precision": 0.6318, "recall": 0.6828, "f1_score": 0.6563, "roc_auc": 0.7278},
  "best_model": "random_forest"
}
```

Top predictive features (matches the intuitive drivers of churn): `tenure_months`, `total_charges`, `is_month_to_month`, `avg_monthly_spend`, `monthly_charges`.

*(Metrics are realistic-but-modest by design — the synthetic data includes noise, not a hand-fit signal, so the model has genuine work to do rather than trivially hitting 99% accuracy.)*

## Project structure

```
data/
└── raw_customer_churn.csv
src/
├── cleaning.py             Duplicate removal, imputation, outlier capping, type coercion
├── eda.py                   4 matplotlib visualizations
├── feature_engineering.py   Derived features + one-hot encoding
└── model.py                 Training, evaluation, 3 more plots
run_pipeline.py               CLI entry point - runs the full pipeline
tests/
└── test_all.py               23 tests covering every pipeline stage
outputs/
├── cleaning_report.json
└── plots/                    7 PNGs + metrics.json
```

## Setup & Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 run_pipeline.py
```

Outputs land in `outputs/`: `cleaning_report.json`, `plots/*.png` (7 charts), `plots/metrics.json`.

## Testing

```bash
python3 tests/test_all.py
```

23 tests, all passing on first full run: duplicate/missing-value handling, outlier capping correctness, type coercion edge cases (blank strings, invalid strings), feature engineering correctness (no inf/NaN from divide-by-zero on zero-tenure customers), and model sanity checks (both models beat random guessing, Random Forest recovers the actual signal baked into the synthetic data).

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib
