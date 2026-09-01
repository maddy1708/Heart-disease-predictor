"""
Model Explainability & Feature Importance Module for Heart Attack Risk Prediction
Generates feature importance rankings and SHAP summary visualizations.
Exports figures to results/figures/ directory.
"""

import os
import sys

# Configure environment variables for instant, deterministic plotting
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
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure project root and src directory are in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.preprocessing import load_data, clean_data, prepare_data
except ImportError:
    from preprocessing import load_data, clean_data, prepare_data

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def clean_feature_names(feature_names: np.ndarray) -> np.ndarray:
    """Format scikit-learn feature names into human-readable labels."""
    cleaned = []
    for name in feature_names:
        name = str(name)
        if name.startswith("num__"):
            name = name[5:]
        elif name.startswith("cat__"):
            name = name[5:].replace("_", ": ")
        cleaned.append(name)
    return np.array(cleaned)


def generate_feature_importance_plot(model, model_name: str, feature_names: np.ndarray):
    """Generate and save the top 15 feature importances horizontal bar plot."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names))

    indices = np.argsort(importances)[-15:]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(range(len(indices)), importances[indices], color="#1d3557", edgecolor="black", alpha=0.85)
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=10)
    plt.xlabel("Feature Importance Score", fontsize=11, fontweight="bold")
    plt.title(f"Top 15 Most Influential Features ({model_name.replace('_', ' ').title()})", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fi_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to {fi_path}")


def generate_shap_plot(model, X_test, y_test, feature_names: np.ndarray):
    """Generate SHAP summary plot or permutation importance plot."""
    shap_plotted = False
    try:
        import shap
        print("Computing SHAP values on test sample...")
        sample_size = min(50, len(X_test))
        X_sample = X_test[:sample_size]

        explainer = shap.TreeExplainer(model, check_additivity=False)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(shap_values, "values"):
            sv = shap_values.values
            if len(sv.shape) == 3:
                sv = sv[:, :, 1]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv,
            X_sample,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP Summary Plot - Feature Impact on Heart Attack Risk", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()

        shap_path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot to {shap_path}")
        shap_plotted = True
    except Exception as e:
        print(f"SHAP note ({e}), generating Permutation Importance figure...")

    if not shap_plotted:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test[:200], y_test[:200], n_repeats=3, random_state=42, n_jobs=1)
        indices = np.argsort(perm.importances_mean)[-15:]

        plt.figure(figsize=(10, 7))
        plt.barh(range(len(indices)), perm.importances_mean[indices], color="#e63946", edgecolor="black", alpha=0.85)
        plt.yticks(range(len(indices)), feature_names[indices], fontsize=10)
        plt.xlabel("Mean Permutation Importance (Drop in Accuracy)", fontsize=11, fontweight="bold")
        plt.title("Permutation Feature Importance on Test Set", fontsize=13, fontweight="bold", pad=15)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()

        fallback_path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(fallback_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved importance figure to {fallback_path}")


def main():
    print("=" * 60)
    print("HEART ATTACK RISK PREDICTION - MODEL EXPLAINABILITY")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_data()
    df_clean = clean_data(df_raw)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(df_clean)

    # Choose model
    model = None
    model_name = None
    for candidate in ["gradient_boosting", "random_forest", "xgboost", "logistic_regression"]:
        cand_path = MODELS_DIR / f"{candidate}.pkl"
        if cand_path.exists():
            model = joblib.load(cand_path)
            model_name = candidate
            break

    if model is None:
        print("No trained models found in models/. Please run 'python src/train.py' first.")
        return

    print(f"Using '{model_name}' for feature importance and interpretability...")

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = clean_feature_names(raw_feature_names)

    # 1. Feature Importance Plot
    generate_feature_importance_plot(model, model_name, feature_names)

    # 2. SHAP / Permutation Plot
    generate_shap_plot(model, X_test, y_test, feature_names)

    print("\n" + "=" * 60)
    print("SUCCESS: Explainability figures generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()