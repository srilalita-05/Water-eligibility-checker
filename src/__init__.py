"""
AI-Based Freshwater Quality Assessment System for Sustainable Agriculture.

This package provides a modular ML pipeline for classifying groundwater
quality (Suitable / Not Suitable) based on 15 chemical parameters.

Modules:
    preprocessing : Data cleaning, feature scaling, and pipeline management
    train_model   : Model training with hyperparameter tuning (RF, GBM, XGBoost)
    evaluate      : Performance metrics, confusion matrix, SHAP analysis
    predict       : Real-time prediction with interpretable explanations
"""

__version__ = "1.0.0"
__author__ = "Water Quality Assessment Team"

# ── Public API ──
from src.preprocessing import (
    preprocess_data,
    load_and_clean_data,
    build_preprocessing_pipeline,
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    LABEL_NAMES,
    CLASS_MAPPING,
)

from src.train_model import train_models

from src.evaluate import evaluate_model

from src.predict import predict, load_artifacts

__all__ = [
    # Preprocessing
    "preprocess_data",
    "load_and_clean_data",
    "build_preprocessing_pipeline",
    # Training
    "train_models",
    # Evaluation
    "evaluate_model",
    # Prediction
    "predict",
    "load_artifacts",
    # Constants
    "FEATURE_COLUMNS",
    "FEATURE_DISPLAY_NAMES",
    "LABEL_NAMES",
    "CLASS_MAPPING",
]
