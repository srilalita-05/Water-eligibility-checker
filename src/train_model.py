"""
Model Training Module for Water Quality Assessment System.

Trains Random Forest, Gradient Boosting, and XGBoost classifiers
with hyperparameter tuning via RandomizedSearchCV (5-fold CV).
Selects best model by ROC-AUC and saves with joblib.
"""

import os
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def get_model_configs():
    """Return list of (name, estimator, param_distributions) tuples."""
    configs = [
        ("Random Forest", RandomForestClassifier(random_state=42), {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", None],
        }),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42), {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
        }),
    ]
    try:
        from xgboost import XGBClassifier
        configs.append(("XGBoost", XGBClassifier(
            random_state=42, eval_metric="logloss", use_label_encoder=False
        ), {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "scale_pos_weight": [1, 3, 5],
        }))
    except ImportError:
        print("  XGBoost not installed. Skipping.")
    return configs


def train_models():
    """Train all models, compare, and save the best one."""
    print("=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    data = joblib.load(os.path.join(MODELS_DIR, "processed_data.joblib"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    results = {}
    best_score, best_model, best_name = -1, None, None

    for name, est, params in get_model_configs():
        print(f"\n  Training: {name}...")
        search = RandomizedSearchCV(
            est, params, n_iter=30, scoring=scorer, cv=cv,
            random_state=42, n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        results[name] = {
            "cv_score": search.best_score_,
            "test_auc": test_auc,
            "model": model,
            "best_params": search.best_params_,
        }
        print(f"  CV AUC: {search.best_score_:.4f} | Test AUC: {test_auc:.4f}")

        if test_auc > best_score:
            best_score, best_model, best_name = test_auc, model, name

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: Best = {best_name} (AUC={best_score:.4f})")
    print(f"{'=' * 60}")

    # Save
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    meta = {"best_model_name": best_name, "test_auc": best_score,
            "best_params": results[best_name]["best_params"]}
    joblib.dump(meta, os.path.join(MODELS_DIR, "model_metadata.joblib"))
    print(f"  Model saved to models/best_model.joblib")
    print("  TRAINING COMPLETE ✓")
    return best_model, results


if __name__ == "__main__":
    train_models()
