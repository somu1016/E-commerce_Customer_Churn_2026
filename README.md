# E-Commerce Customer Churn Analysis

An end-to-end Python analysis and machine-learning project for exploring churn records and comparing churn classifiers. It was prepared for an IBM Data Analytics with AI internship project.

## Repository Contents

```text
.
├── SomeshwariChoudhary_Customer_churn_analysis.py
├── PROJECT_REPORT.docx
├── README.md
├── Requirements.txt
└── outputs/                 # generated locally; ignored by Git
```

The analysis expects a local `E-commerce_Customer_Churn_2026.csv` file containing 50,000 records and 48 source columns. It is a record-level dataset, not a guaranteed one-row-per-customer table: `customer_id` is repeated for 18,356 records. The target is `is_churned` (`Yes` or `No`). The CSV is ignored by Git because it contains customer-level fields; publish or redistribute it only after confirming privacy and licensing rights.

## Requirements

- Python 3.10 or newer
- pip
- Approximately 15 MB for the dataset plus package and output storage

Python 3.14 was used for the latest local validation. Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

## Run

From the repository root:

```bash
python SomeshwariChoudhary_Customer_churn_analysis.py
```

Place the dataset at the repository root before running. A clone without the local CSV will fail fast with a clear `Dataset not found` message.

The command is non-interactive and writes charts, `model_metrics.csv`, and `summary.json` to `outputs/`. Useful options:

```bash
python SomeshwariChoudhary_Customer_churn_analysis.py --no-shap  # skip optional SHAP charts
python SomeshwariChoudhary_Customer_churn_analysis.py --show     # display charts while running
```

The script resolves the dataset and output directory relative to its own file, so it can also be started from another working directory. It does not modify the source CSV.

## Pipeline

1. Validate and load the CSV, including target values and required columns.
2. Produce eight EDA charts for churn, segment, reasons, satisfaction, risk, prevention, correlations, and subscription type.
3. Remove identifiers, post-churn fields, and target-derived fields from model features to reduce leakage.
4. Fill missing categorical values as `Unknown`, impute numeric values with medians, one-hot encode categories, and add guarded ratio features.
5. Split data with stratification, scale the logistic-regression inputs, and apply SMOTE only to training data.
6. Compare Logistic Regression, Random Forest, Histogram Gradient Boosting, and XGBoost using AUC-ROC, accuracy, precision, recall, and F1 for the churn class.
7. Save an executive dashboard, best-model confusion-matrix, precision-recall, feature-importance, and optional SHAP charts.

## Generated Outputs

| File | Description |
|---|---|
| `00_executive_dashboard.png` | KPI snapshot of churn, satisfaction, and risk |
| `01_churn_distribution.png` to `08_subscription_analysis.png` | EDA charts |
| `09_roc_curves.png` | ROC curves for all models |
| `10_model_comparison.png` | Model metric comparison |
| `11_confusion_matrix.png` | Best model confusion matrix |
| `12_precision_recall.png` | Best model precision-recall curve |
| `13_feature_importance.png` | Best tree-model feature importance |
| `14_shap_bar.png`, `15_shap_beeswarm.png` | Optional SHAP explanations |
| `model_metrics.csv` | Reproducible evaluation metrics |
| `summary.json` | Run summary and selected best model |

Generated artifacts are excluded by `.gitignore`; keep them locally or publish selected images deliberately.

## Data and Modelling Notes

- The dataset contains 20,653 churned records, an observed churn rate of 41.3%.
- Missing values occur in `subscription_type`, `prevention_method`, and `social_media_engagement` and are handled without dropping rows.
- Fields that encode or occur after churn, such as `churn_reason`, `churn_date`, `churn_probability`, `churn_risk_score`, `predicted_churn`, and revenue-impact fields, are excluded from modelling.
- Results are predictive associations, not causal evidence. SMOTE changes the training distribution; all reported metrics are calculated on the untouched stratified test set.
- The data file appears synthetic or internally generated. Review privacy, licensing, and redistribution rights before publishing it to a public GitHub repository.

See [PROJECT_REPORT.docx](PROJECT_REPORT.docx) for the methodology, audit notes, and interpretation guidance.

## License

No license has been selected. Do not treat this repository as licensed for reuse until a license file and ownership terms are added.
