"""
Model Evaluation & Figure Generation Module for Heart Attack Risk Prediction
Evaluates all models, runs 5-fold Stratified Cross-Validation, and generates:
- Multi-Model ROC-AUC curves (roc_auc_curves.png)
- Metric Comparison Bar Chart (model_comparison_bar.png)
- Confusion Matrix Grid (confusion_matrices.png)
"""

import os
import sys

# Configure environment variables for instant, deterministic plotting and evaluation
os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ACCELERATE_NEW_LAPACK"] = "0"

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root and src directory are in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from src.preprocessing import load_data, clean_data, prepare_data
except ImportError:
    from preprocessing import load_data, clean_data, prepare_data

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a single trained model on test data."""
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)
    else:
        probabilities = predictions

    acc = float(accuracy_score(y_test, predictions))
    prec = float(precision_score(y_test, predictions, zero_division=0))
    rec = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    auc = float(roc_auc_score(y_test, probabilities))

    result = {
        "Model": name,
        "Accuracy (%)": f"{acc * 100:.2f}%",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    }

    print("\n" + "=" * 60)
    print(f"MODEL: {name.upper()}")
    print("=" * 60)
    print(f"Accuracy: {result['Accuracy (%)']} ({acc:.4f})")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return result


def plot_roc_curves(model_dict: dict, X_test, y_test, output_path: Path):
    """Generate and save publication-quality multi-model ROC Curve comparison graph."""
    plt.figure(figsize=(10, 8))

    colors = {
        "logistic_regression": "#1f77b4",
        "gradient_boosting": "#2ca02c",
        "voting_ensemble": "#d62728",
        "xgboost": "#ff7f0e",
        "random_forest": "#9467bd",
        "knn": "#8c564b",
        "svm": "#e377c2"
    }

    scored_models = []
    for name, model in model_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_test)
        else:
            probs = model.predict(X_test)

        score = roc_auc_score(y_test, probs)
        scored_models.append((name, model, probs, score))

    scored_models.sort(key=lambda x: x[3], reverse=True)

    for name, model, probs, score in scored_models:
        fpr, tpr, _ = roc_curve(y_test, probs)
        display_name = name.replace("_", " ").title()
        color = colors.get(name, None)
        lw = 2.8 if name in ["logistic_regression", "gradient_boosting", "voting_ensemble"] else 2.0

        plt.plot(
            fpr, tpr,
            label=f"{display_name} (AUC = {score:.4f})",
            linewidth=lw,
            color=color
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.5000)", linewidth=1.5, alpha=0.7)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves - Model Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10.5, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved ROC-AUC Curve graph to {output_path}")


def plot_model_comparison_bar(results_df: pd.DataFrame, output_path: Path):
    """Generate and save model comparison bar chart across Accuracy and ROC-AUC."""
    plt.figure(figsize=(12, 6))

    df_plot = results_df.copy()
    df_plot["Model_Display"] = df_plot["Model"].str.replace("_", " ").str.title()
    df_plot["Accuracy_Num"] = df_plot["Accuracy"] * 100.0
    df_plot["ROC_AUC_Num"] = df_plot["ROC-AUC"] * 100.0

    x = np.arange(len(df_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6.5))
    rects1 = ax.bar(x - width/2, df_plot["Accuracy_Num"], width, label="Accuracy (%)", color="#1d3557", edgecolor="black")
    rects2 = ax.bar(x + width/2, df_plot["ROC_AUC_Num"], width, label="ROC-AUC (×100)", color="#e63946", edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Machine Learning Models Performance Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Model_Display"], fontsize=11, rotation=15)
    ax.legend(fontsize=11, frameon=True)
    ax.set_ylim(60, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Attach labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison bar chart to {output_path}")


def plot_confusion_matrices(model_dict: dict, X_test, y_test, output_path: Path):
    """Generate 2x2 grid of confusion matrices for top classifiers."""
    top_models = ["logistic_regression", "gradient_boosting", "voting_ensemble", "random_forest"]
    available_top = [m for m in top_models if m in model_dict]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, name in enumerate(available_top):
        model = model_dict[name]
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        display_name = name.replace("_", " ").title()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["No Risk (0)", "High Risk (1)"],
                    yticklabels=["No Risk (0)", "High Risk (1)"],
                    annot_kws={"size": 13, "weight": "bold"})
        axes[idx].set_title(f"{display_name}\nAccuracy: {accuracy_score(y_test, preds)*100:.2f}%", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        axes[idx].set_ylabel("True Clinical Label", fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrices for Top Performing Classifiers", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices grid to {output_path}")


def cross_validation(X_train, y_train) -> pd.DataFrame:
    """Perform 5-fold stratified cross validation."""
    print("\n" + "=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    cv_models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=2.0, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.06, max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42, n_jobs=1)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for name, model in cv_models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=1
        )

        acc_mean = float(scores["test_accuracy"].mean())
        result = {
            "Model": name,
            "CV Accuracy (%)": f"{acc_mean * 100:.2f}%",
            "CV Accuracy": round(acc_mean, 4),
            "CV Precision": round(float(scores["test_precision"].mean()), 4),
            "CV Recall": round(float(scores["test_recall"].mean()), 4),
            "CV F1": round(float(scores["test_f1"].mean()), 4),
            "CV ROC-AUC": round(float(scores["test_roc_auc"].mean()), 4)
        }
        cv_results.append(result)

        print(f"\n--- {name} ---")
        print(f"CV Accuracy:  {result['CV Accuracy (%)']} ({result['CV Accuracy']:.4f})")
        print(f"CV Precision: {result['CV Precision']:.4f}")
        print(f"CV Recall:    {result['CV Recall']:.4f}")
        print(f"CV F1:        {result['CV F1']:.4f}")
        print(f"CV ROC-AUC:   {result['CV ROC-AUC']:.4f}")

    return pd.DataFrame(cv_results)


def main():
    print("=" * 60)
    print("HEART ATTACK RISK PREDICTION - MODEL EVALUATION PIPELINE")
    print("=" * 60)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_data()
    df_clean = clean_data(df_raw)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _
    ) = prepare_data(df_clean)

    model_names = [
        "voting_ensemble",
        "logistic_regression",
        "gradient_boosting",
        "xgboost",
        "random_forest",
        "knn",
        "svm"
    ]

    results = []
    loaded_models = {}

    for name in model_names:
        model_path = MODELS_DIR / f"{name}.pkl"
        if not model_path.exists():
            print(f"Warning: Model file {model_path} not found. Skipping.")
            continue

        model = joblib.load(model_path)
        loaded_models[name] = model
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    if not results:
        print("No trained models found in models/ directory. Run 'python src/train.py' first.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("ROC-AUC", ascending=False)

    comparison_path = METRICS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\n")
    print("=" * 75)
    print("FINAL TEST-SET MODEL COMPARISON & ACCURACY PERCENTAGES")
    print("=" * 75)
    print(results_df[["Model", "Accuracy (%)", "Precision", "Recall", "F1", "ROC-AUC"]].to_string(index=False))
    print(f"\nSaved test comparison to {comparison_path}")

    # Generate Figures
    roc_curve_path = FIGURES_DIR / "roc_auc_curves.png"
    plot_roc_curves(loaded_models, X_test, y_test, roc_curve_path)

    bar_chart_path = FIGURES_DIR / "model_comparison_bar.png"
    plot_model_comparison_bar(results_df, bar_chart_path)

    cm_path = FIGURES_DIR / "confusion_matrices.png"
    plot_confusion_matrices(loaded_models, X_test, y_test, cm_path)

    # Run Cross-Validation
    cv_df = cross_validation(X_train, y_train)
    cv_path = METRICS_DIR / "cross_validation.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\nSaved cross-validation results to {cv_path}")

    print("\n" + "=" * 60)
    print("SUCCESS: Evaluation and high-resolution figure generation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()