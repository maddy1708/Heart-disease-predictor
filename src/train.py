"""
Model Training Module for Heart Attack Risk Prediction
Trains individual classifiers and high-performing Soft Voting Ensembles.
Saves trained models, preprocessor, and metadata to models/ directory.
"""

import os
import sys

# Configure single-threaded execution to ensure fast, deterministic training on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ACCELERATE_NEW_LAPACK"] = "0"

from pathlib import Path
import joblib

# Ensure project root and src directory are in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC

# Check for native XGBoost availability
HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier
    _test = XGBClassifier(n_estimators=1, n_jobs=1)
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from src.preprocessing import load_data, clean_data, prepare_data
except ImportError:
    from preprocessing import load_data, clean_data, prepare_data

MODELS_DIR = PROJECT_ROOT / "models"


def get_models():
    """Build and return dictionary of tuned models and ensembles."""
    lr = LogisticRegression(
        max_iter=5000,
        C=5.0,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        max_features=0.5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1
    )

    gb = GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.90,
        random_state=42
    )

    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="manhattan",
        n_jobs=1
    )

    svm = SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        random_state=42
    )

    if HAS_XGBOOST:
        xgb_model = XGBClassifier(
            n_estimators=900,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )
    else:
        xgb_model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_depth=6,
            random_state=42
        )

    # Soft Voting Ensemble
    voting = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("gb", gb),
            ("rf", rf),
            ("xgb", xgb_model),
            ("svm", svm)
        ],
        voting="soft",
        weights=[3.0, 1.5, 0.5, 1.5, 1.5]
    )

    models = {
        "logistic_regression": lr,
        "gradient_boosting": gb,
        "xgboost": xgb_model,
        "random_forest": rf,
        "voting_ensemble": voting,
        "knn": knn,
        "svm": svm
    }

    return models


def main():
    print("=" * 60)
    print("HEART ATTACK RISK PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    print("\n1. Loading dataset...")
    df_raw = load_data()
    print(f"   Raw dataset loaded. Shape: {df_raw.shape}")

    print("\n2. Cleaning data & engineering clinical features...")
    df_clean = clean_data(df_raw)
    print(f"   Cleaned dataset shape: {df_clean.shape}")

    print("\n3. Preparing train/test splits & preprocessor...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(df_clean)

    print(f"   Training features shape: {X_train.shape}")
    print(f"   Testing features shape:  {X_test.shape}")
    print(f"   Positive class ratio (train): {y_train.mean():.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = get_models()

    print("\n4. Training models and ensembles...")
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"      Saved {model_path}")

    # Save preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n   -> Preprocessor saved to {preprocessor_path}")

    # Save metadata for UI / explainability
    raw_features = [c for c in df_raw.columns if c != "Heart Attack Risk"]
    cleaned_features = [c for c in df_clean.columns if c != "Heart Attack Risk"]
    transformed_feature_names = list(preprocessor.get_feature_names_out())

    metadata = {
        "raw_features": raw_features,
        "cleaned_features": cleaned_features,
        "transformed_feature_names": transformed_feature_names,
        "has_xgboost": HAS_XGBOOST
    }
    joblib.dump(metadata, MODELS_DIR / "metadata.pkl")

    print("\n" + "=" * 60)
    print("SUCCESS: All models trained and saved successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()