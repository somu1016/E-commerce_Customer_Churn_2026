# Project Report: E-Commerce Churn Analysis

## Executive Summary

This project profiles 50,000 e-commerce churn records and compares four classification models. The analysis includes exploratory charts, guarded feature engineering, class-imbalance handling, model evaluation, feature importance, and optional SHAP explanations.

The source data contains 48 columns and 20,653 records labelled as churned, for an observed record-level churn rate of 41.3%. Because `customer_id` repeats, these are records rather than necessarily 50,000 unique customers.

## Audit Findings and Fixes

The original project was audited across the Python script, CSV metadata, requirements, README, report, and generated-output directory. The following issues were corrected:

- Replaced global top-level execution with a callable `main()` pipeline and command-line options.
- Removed the process-wide working-directory change; all paths now use `pathlib` relative to the script.
- Removed interactive plotting from the default run and close every figure after saving.
- Added validation for missing files, required columns, and unexpected target values.
- Handled missing categories and numeric values without dropping rows.
- Guarded all ratio features against division by zero.
- Removed identifiers and target-derived or post-churn fields from model inputs, including risk and probability fields that can leak the target.
- Applied SMOTE only to the training partition.
- Added reproducible output files: `model_metrics.csv` and `summary.json`.
- Limited SHAP to a sample of the test set and made it optional with `--no-shap`.
- Removed unused dependencies and documented the actual repository contents.
- Added `.gitignore` rules for caches, environments, local settings, logs, and generated analysis outputs.

## Methodology

1. Load and validate the CSV.
2. Encode `is_churned` as a binary target.
3. Generate eight exploratory visualisations.
4. Remove identifiers, derived labels, post-churn information, and target-derived scores from modelling.
5. Fill missing categories with `Unknown`, impute numeric values with training-compatible medians, one-hot encode categorical features, and create five ratio features.
6. Split the data with a fixed random seed and stratification.
7. Train Logistic Regression, Random Forest, Histogram Gradient Boosting, and XGBoost models. Logistic Regression uses standardised features; tree models use the unscaled features.
8. Compare AUC-ROC, accuracy, precision, recall, and F1 on the untouched test partition.
9. Explain the best tree model with feature importance and optional SHAP plots.

## Reproducibility

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python churn_analysis.py
```

Use `python churn_analysis.py --no-shap` for a faster run or `python churn_analysis.py --show` when interactive chart display is desired. The latest run writes all generated files under `outputs/`; the exact scores are recorded in `outputs/model_metrics.csv` rather than hard-coded in this report.

## Interpretation and Limitations

- Model scores show predictive association, not causal impact.
- The dataset appears synthetic or internally generated; provenance, licensing, and privacy should be confirmed before public release.
- Repeated IDs may create correlated records. A production evaluation should consider a group-aware split by `customer_id` if the prediction unit is an individual customer.
- The model excludes useful-looking fields such as `churn_risk_score` and `churn_probability` because they are likely derived from the churn outcome or from another model. Including them would inflate offline performance.
- Threshold selection should be based on retention capacity and intervention cost rather than defaulting automatically to 0.5.
- The analysis does not constitute a deployed prediction service or a causal retention experiment.

## Outputs

The script creates eight EDA plots, ROC and model-comparison plots, best-model diagnostic plots, optional SHAP plots, `model_metrics.csv`, and `summary.json`. These generated files are intentionally ignored by Git; publish only selected artifacts that are appropriate for the repository.
