"""End-to-end e-commerce customer churn analysis.

Run with ``python churn_analysis.py``. Charts and model metrics are written to
``outputs/`` relative to this file, so the command works from any directory.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "E-commerce_Customer_Churn_2026.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"


def save_figure(filename: str, show: bool = False) -> None:
    """Save the current figure and release it to avoid memory growth."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    required = {"is_churned", "customer_segment", "satisfaction_score"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data["Churned"] = data["is_churned"].map({"Yes": 1, "No": 0})
    if data["Churned"].isna().any():
        invalid = sorted(data.loc[data["Churned"].isna(), "is_churned"].dropna().unique())
        raise ValueError(f"Unexpected is_churned values: {invalid}")
    return data


def prepare_model_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create numeric features while excluding fields unavailable before churn."""
    target = data["Churned"].astype(int)
    drop_cols = {
        "Churned", "churn_id", "customer_id", "is_churned", "churn_date",
        "days_since_churn", "churn_flag", "churn_reason", "churn_category",
        "churn_probability", "churn_risk_score", "is_high_risk_customer",
        "predicted_churn", "churn_rate_indicator", "churn_revenue_impact_usd",
        "recovery_revenue_usd", "customer_health_score", "customer_health_category",
        "retention_probability", "customer_retention_cost_usd",
        "satisfaction_level", "churn_risk_category", "customer_value_category",
        "engagement_level",
    }
    features = data.drop(columns=[column for column in drop_cols if column in data])
    categorical = features.select_dtypes(include=["object"]).columns
    features = features.copy()
    features[categorical] = features[categorical].fillna("Unknown").astype(str)
    numeric = features.select_dtypes(exclude=["object"]).columns
    features[numeric] = features[numeric].apply(pd.to_numeric, errors="coerce")
    features[numeric] = features[numeric].fillna(features[numeric].median()).fillna(0)
    features = pd.get_dummies(features, columns=list(categorical), dtype=float)

    features["email_open_rate"] = np.divide(
        features["marketing_emails_opened"], features["marketing_emails_sent"],
        out=np.zeros(len(features), dtype=float), where=features["marketing_emails_sent"] > 0,
    )
    features["email_click_rate"] = np.divide(
        features["marketing_emails_clicked"], features["marketing_emails_opened"],
        out=np.zeros(len(features), dtype=float), where=features["marketing_emails_opened"] > 0,
    )
    features["spend_per_purchase"] = np.divide(
        features["total_spend_usd"], features["num_purchases"],
        out=np.zeros(len(features), dtype=float), where=features["num_purchases"] > 0,
    )
    features["ticket_per_purchase"] = np.divide(
        features["support_tickets"], features["num_purchases"],
        out=np.zeros(len(features), dtype=float), where=features["num_purchases"] > 0,
    )
    features["return_rate"] = np.divide(
        features["returns_count"], features["num_purchases"],
        out=np.zeros(len(features), dtype=float), where=features["num_purchases"] > 0,
    )
    return features.astype(float), target


def run_eda(data: pd.DataFrame, show: bool) -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    churn_counts = data["is_churned"].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["Not Churned", "Churned"], churn_counts.values, color=["#3b82d4", "#e74c3c"])
    axes[0].set_ylabel("Count")
    axes[0].set_title("Churn Distribution")
    for index, value in enumerate(churn_counts.values):
        axes[0].text(index, value, f"{value:,}\n({value / len(data) * 100:.1f}%)", ha="center", va="bottom")
    axes[1].pie(churn_counts.values, labels=["Not Churned", "Churned"], autopct="%1.1f%%",
                colors=["#3b82d4", "#e74c3c"], startangle=90)
    axes[1].set_title("Churn Proportion")
    fig.suptitle("Overall Churn Distribution")
    save_figure("01_churn_distribution.png", show)

    grouped = data.groupby("customer_segment", dropna=False)["Churned"].agg(["sum", "count"])
    grouped["churn_rate"] = grouped["sum"] / grouped["count"] * 100
    grouped = grouped.sort_values("churn_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(grouped.index.astype(str), grouped["churn_rate"], color="#e74c3c")
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Customer Segment")
    save_figure("02_churn_by_segment.png", show)

    reasons = data.loc[data["Churned"].eq(1), "churn_reason"].fillna("Unknown").value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    reasons.plot(kind="barh", ax=ax, color="#e74c3c")
    ax.set_title("Top 10 Churn Reasons")
    ax.set_xlabel("Number of Customers")
    save_figure("03_churn_reasons.png", show)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, name in [(1, "#e74c3c", "Churned"), (0, "#3b82d4", "Retained")]:
        values = data.loc[data["Churned"].eq(label), "satisfaction_score"]
        ax.hist(values, bins=5, alpha=0.6, color=color, label=f"{name} (mean={values.mean():.2f})")
    ax.set_title("Satisfaction Score Distribution")
    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Count")
    ax.legend()
    save_figure("04_satisfaction_distribution.png", show)

    fig, ax = plt.subplots(figsize=(8, 5))
    data.boxplot(column="churn_risk_score", by="is_churned", ax=ax)
    ax.set_title("Churn Risk Score by Churn Status")
    ax.set_xlabel("Churned")
    ax.set_ylabel("Churn Risk Score")
    fig.suptitle("")
    save_figure("05_risk_score_boxplot.png", show)

    prevention = data.groupby(data["prevention_method"].fillna("Unknown"))["Churned"].agg(["sum", "count"])
    prevention["retention_rate"] = (prevention["count"] - prevention["sum"]) / prevention["count"] * 100
    prevention = prevention.sort_values("retention_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(prevention.index.astype(str), prevention["retention_rate"], color="#3b82d4")
    ax.set_ylabel("Retention Rate (%)")
    ax.set_title("Prevention Method Retention Rate")
    ax.tick_params(axis="x", rotation=30)
    save_figure("06_prevention_effectiveness.png", show)

    numeric_cols = [
        "satisfaction_score", "loyalty_score", "support_tickets", "returns_count",
        "days_since_last_purchase", "total_spend_usd", "num_purchases",
        "customer_lifetime_value_usd", "customer_age_months", "feedback_score", "Churned",
    ]
    corr = data[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    save_figure("07_correlation_heatmap.png", show)

    subscription = data.groupby(data["subscription_type"].fillna("None"))["Churned"].agg(["sum", "count"])
    subscription["churn_rate"] = subscription["sum"] / subscription["count"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(subscription.index.astype(str), subscription["count"], color="#3b82d4")
    axes[1].bar(subscription.index.astype(str), subscription["churn_rate"], color="#e74c3c")
    axes[0].set_title("Customer Count by Subscription Type")
    axes[1].set_title("Churn Rate by Subscription Type")
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Churn Rate (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    save_figure("08_subscription_analysis.png", show)


def train_and_evaluate(features: pd.DataFrame, target: pd.Series, show: bool) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)
    x_train_scaled_sm, y_train_scaled_sm = smote.fit_resample(x_train_scaled, y_train)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False),
        "Histogram Gradient Boosting": (HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_STATE), False),
        "XGBoost": (XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_STATE,
                                  n_jobs=2, tree_method="hist"), False),
    }
    results = {}
    probabilities = {}
    predictions = {}
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (model, scaled) in models.items():
        if scaled:
            model.fit(x_train_scaled_sm, y_train_scaled_sm)
            probability = model.predict_proba(x_test_scaled)[:, 1]
        else:
            model.fit(x_train_sm, y_train_sm)
            probability = model.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, probability)
        results[name] = {
            "AUC-ROC": round(auc, 4), "Accuracy": round(report["accuracy"], 4),
            "Precision (Churn)": round(report["1"]["precision"], 4),
            "Recall (Churn)": round(report["1"]["recall"], 4),
            "F1 (Churn)": round(report["1"]["f1-score"], 4), "model": model,
        }
        probabilities[name] = probability
        predictions[name] = prediction
        fpr, tpr, _ = roc_curve(y_test, probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        print(f"{name}: AUC={auc:.4f}, F1={report['1']['f1-score']:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    save_figure("09_roc_curves.png", show)

    metrics = pd.DataFrame(results).T.drop(columns="model")
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv")
    metrics_to_plot = ["AUC-ROC", "Accuracy", "F1 (Churn)"]
    metrics[metrics_to_plot].plot(kind="bar", figsize=(11, 6), ylim=(0, 1.1), rot=15)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    save_figure("10_model_comparison.png", show)

    best_name = metrics["AUC-ROC"].idxmax()
    best_model = results[best_name]["model"]
    best_prediction = predictions[best_name]
    best_probability = probabilities[best_name]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, best_prediction), display_labels=["Retained", "Churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    save_figure("11_confusion_matrix.png", show)
    precision, recall, _ = precision_recall_curve(y_test, best_probability)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#3b82d4")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {best_name}")
    save_figure("12_precision_recall.png", show)
    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=features.columns).nlargest(20).sort_values()
        importance.plot(kind="barh", figsize=(10, 7), color="#3b82d4")
        plt.xlabel("Importance")
        plt.title(f"Top Feature Importances - {best_name}")
        save_figure("13_feature_importance.png", show)
    return results, metrics, x_test


def run_shap(results: dict, x_test: pd.DataFrame, best_name: str, show: bool) -> None:
    try:
        import shap

        sample = x_test.iloc[: min(300, len(x_test))]
        values = shap.TreeExplainer(results[best_name]["model"]).shap_values(sample)
        if isinstance(values, list):
            values = values[1]
        shap.summary_plot(values, sample, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {best_name}")
        save_figure("14_shap_bar.png", show)
        shap.summary_plot(values, sample, show=False)
        plt.title(f"SHAP Beeswarm Plot - {best_name}")
        save_figure("15_shap_beeswarm.png", show)
    except Exception as error:
        print(f"SHAP skipped: {error}")


def main(show: bool = False, skip_shap: bool = False) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data):,} records and {data.shape[1] - 1} source columns")
    print(f"Churned: {data['Churned'].sum():,} ({data['Churned'].mean() * 100:.1f}%)")
    print("Creating exploratory charts...")
    run_eda(data, show)
    features, target = prepare_model_data(data)
    print(f"Training on {features.shape[1]} leakage-filtered features...")
    results, metrics, x_test = train_and_evaluate(features, target, show)
    best_name = metrics["AUC-ROC"].idxmax()
    if not skip_shap:
        run_shap(results, x_test, best_name, show)
    summary = {
        "records": int(len(data)),
        "source_columns": int(data.shape[1] - 1),
        "churned_records": int(target.sum()),
        "churn_rate": round(float(target.mean()), 4),
        "best_model": best_name,
        "best_auc_roc": float(metrics.loc[best_name, "AUC-ROC"]),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best model: {best_name} (AUC-ROC={summary['best_auc_roc']:.4f})")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Display charts while running")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP plots")
    arguments = parser.parse_args()
    main(show=arguments.show, skip_shap=arguments.no_shap)
