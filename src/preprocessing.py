"""
Data Preprocessing & Advanced Feature Engineering Module for Heart Attack Risk Prediction.
Extracts clinical cardiovascular indicators, scales numerical features, encodes categories,
and prepares stratified train/test datasets.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "heart_attack_prediction.csv"
TARGET = "Heart Attack Risk"


def load_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load dataset from CSV or TSV with flexible delimiter detection.
    """
    if filepath is None:
        filepath = DEFAULT_DATA_PATH
    else:
        filepath = Path(filepath)

    if not filepath.exists() or filepath.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath, sep=r"\t|,", engine="python")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer high-impact clinical features:
    - Remove duplicate records and patient identifier
    - Parse Blood Pressure into Systolic and Diastolic BP
    - Engineer hemodynamic & metabolic indices:
        * Pulse Pressure: Arterial stiffness proxy (Systolic - Diastolic)
        * Mean Arterial Pressure (MAP): Organ perfusion pressure
        * Lipid Index: (Cholesterol * Triglycerides) / 10000
        * Triglyceride-to-Cholesterol Ratio
        * Sedentary-to-Active Lifestyle Ratio
        * Metabolic Risk Composite
        * Cardio Composite Risk Index
    - Drop uninformative geolocation columns
    """
    df = df.copy()

    # Deduplicate
    df = df.drop_duplicates()

    # Drop patient identifier
    if "Patient ID" in df.columns:
        df = df.drop(columns=["Patient ID"])

    # Parse Blood Pressure
    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
        df["Systolic BP"] = pd.to_numeric(bp[0], errors="coerce").fillna(135.0)
        df["Diastolic BP"] = pd.to_numeric(bp[1], errors="coerce").fillna(85.0)
        df = df.drop(columns=["Blood Pressure"])
    elif "Systolic BP" not in df.columns:
        df["Systolic BP"] = 135.0
        df["Diastolic BP"] = 85.0

    # Ensure critical numeric columns exist and are typed
    for col in ["Cholesterol", "Triglycerides", "Heart Rate", "Age", "BMI", "Exercise Hours Per Week", "Sedentary Hours Per Day"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # High-Impact Clinical Feature Engineering
    df["Pulse Pressure"] = df["Systolic BP"] - df["Diastolic BP"]
    df["MAP"] = (2.0 * df["Diastolic BP"] + df["Systolic BP"]) / 3.0
    df["Lipid_Index"] = (df["Cholesterol"] * df["Triglycerides"]) / 10000.0
    df["Trig_Chol_Ratio"] = df["Triglycerides"] / (df["Cholesterol"] + 1.0)
    df["Sedentary_Active_Ratio"] = df["Sedentary Hours Per Day"] / (df["Exercise Hours Per Week"] / 7.0 + 0.1)

    # Risk composites
    diabetes = df["Diabetes"] if "Diabetes" in df.columns else 0
    obesity = df["Obesity"] if "Obesity" in df.columns else 0
    prev_heart = df["Previous Heart Problems"] if "Previous Heart Problems" in df.columns else 0
    family_hist = df["Family History"] if "Family History" in df.columns else 0
    smoking = df["Smoking"] if "Smoking" in df.columns else 0

    df["Metabolic_Risk"] = (
        diabetes.astype(float)
        + obesity.astype(float)
        + (df["BMI"] >= 30.0).astype(float)
        + (df["Systolic BP"] >= 130.0).astype(float)
    )

    df["Cardio_Composite"] = (
        prev_heart.astype(float) * 2.5
        + diabetes.astype(float) * 2.0
        + family_hist.astype(float) * 1.5
        + smoking.astype(float) * 1.5
        + (df["Systolic BP"] >= 140.0).astype(float) * 1.8
        + (df["Cholesterol"] >= 240.0).astype(float) * 1.2
    )

    # Drop geographic identifiers
    geographic_columns = ["Country", "Continent", "Hemisphere"]
    cols_to_drop = [col for col in geographic_columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """
    Prepare dataset for machine learning:
    1. Separate features (X) and target (y)
    2. Build preprocessor (StandardScaler for numerical, OneHotEncoder for categorical)
    3. Stratified Train-Test split (80/20)
    4. Fit preprocessor on training data and transform train/test
    5. Return (X_train_transformed, X_test_transformed, y_train, y_test, preprocessor)
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataframe columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    categorical_features = ["Sex", "Diet"]
    categorical_features = [c for c in categorical_features if c in X.columns]
    numerical_features = [c for c in X.columns if c not in categorical_features]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return (
        X_train_transformed,
        X_test_transformed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    df = load_data()
    print("Raw dataset shape:", df.shape)
    df_clean = clean_data(df)
    print("Cleaned dataset shape:", df_clean.shape)
    X_tr, X_te, y_tr, y_te, prep = prepare_data(df_clean)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"Positive class prevalence (train): {y_tr.mean()*100:.2f}%")
    print("Preprocessing completed successfully.")