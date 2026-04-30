"""
Data Preprocessing Pipeline for Water Quality Assessment System.

This module handles:
- Loading and cleaning the groundwater quality dataset
- Feature selection and engineering
- Missing value handling (median imputation)
- Duplicate removal
- Feature scaling (StandardScaler)
- Train-test split (80/20, stratified)
- Saving the preprocessing pipeline with joblib
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib


# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────

# Feature columns used for model training
FEATURE_COLUMNS = [
    "pH", "E.C", "TDS", "CO3", "HCO3", "Cl", "F",
    "NO3 ", "SO4", "Na", "K", "Ca", "Mg", "T.H", "SAR"
]

# Friendly names for display in the UI and explanations
FEATURE_DISPLAY_NAMES = {
    "pH": "pH Level",
    "E.C": "Electrical Conductivity (µS/cm)",
    "TDS": "Total Dissolved Solids (mg/L)",
    "CO3": "Carbonate (mg/L)",
    "HCO3": "Bicarbonate (mg/L)",
    "Cl": "Chloride (mg/L)",
    "F": "Fluoride (mg/L)",
    "NO3 ": "Nitrate (mg/L)",
    "SO4": "Sulphate (mg/L)",
    "Na": "Sodium (mg/L)",
    "K": "Potassium (mg/L)",
    "Ca": "Calcium (mg/L)",
    "Mg": "Magnesium (mg/L)",
    "T.H": "Total Hardness (mg/L)",
    "SAR": "Sodium Adsorption Ratio",
}

# Target column and class mapping
TARGET_COLUMN = "Classification.1"
CLASS_MAPPING = {
    "P.S.": 1,   # Presumably Safe  → Suitable
    "MR": 1,     # Marginally Safe  → Suitable (borderline acceptable)
    "U.S.": 0,   # Unsuitable       → Not Suitable
}

LABEL_NAMES = {0: "Not Suitable", 1: "Suitable"}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_data_path_1 = os.path.join(BASE_DIR, "data", "ground_water_quality.csv")
_data_path_2 = os.path.join(BASE_DIR, "ground_water_quality.csv")
DATA_PATH = _data_path_1 if os.path.exists(_data_path_1) else _data_path_2
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def load_and_clean_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the groundwater quality CSV and perform initial cleaning.

    Steps:
        1. Read CSV
        2. Select feature + target columns
        3. Drop rows where target or all features are NA
        4. Convert features to numeric (coerce errors)
        5. Remove duplicate rows

    Returns:
        Cleaned pandas DataFrame with numeric features and mapped target.
    """
    if data_path is None:
        data_path = DATA_PATH

    df = pd.read_csv(data_path)

    # Keep only the columns we need
    cols_to_keep = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[cols_to_keep].copy()

    # Drop rows where the target is missing or not in our mapping
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN].isin(CLASS_MAPPING.keys())]

    # Convert feature columns to numeric (some might be strings like 'NA')
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where ALL features are NaN (completely empty rows)
    df = df.dropna(subset=FEATURE_COLUMNS, how="all")

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  ℹ  Removed {n_removed} duplicate rows.")

    # Map target to binary labels
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(CLASS_MAPPING)

    df = df.reset_index(drop=True)
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline for preprocessing:
        1. Median imputation for remaining missing values
        2. Standard scaling
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline


def preprocess_data(data_path: str = None):
    """
    Full preprocessing workflow:
        1. Load & clean data
        2. Split into train/test (80/20, stratified)
        3. Fit preprocessing pipeline on training data
        4. Transform train & test sets
        5. Save pipeline and processed data

    Returns:
        dict with X_train, X_test, y_train, y_test, pipeline, feature_names
    """
    print("=" * 60)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # --- Load ---
    print("\n▸ Loading dataset...")
    df = load_and_clean_data(data_path)
    print(f"  Samples after cleaning: {len(df)}")

    # --- Features and Target ---
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {LABEL_NAMES[cls]}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # --- Train/Test Split ---
    print("\n▸ Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")

    # --- Build & Fit Pipeline ---
    print("\n▸ Fitting preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    print("  ✓ Imputation (median) applied")
    print("  ✓ Standard scaling applied")

    # --- Save ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    pipeline_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"\n▸ Pipeline saved → {pipeline_path}")

    # Save processed data for reuse
    data_dict = {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "feature_names": FEATURE_COLUMNS,
    }
    data_path_out = os.path.join(MODELS_DIR, "processed_data.joblib")
    joblib.dump(data_dict, data_path_out)
    print(f"  Processed data saved → {data_path_out}")

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE ✓")
    print("=" * 60)

    return data_dict


# ──────────────────────────────────────────────
#  RUN AS SCRIPT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    preprocess_data()
