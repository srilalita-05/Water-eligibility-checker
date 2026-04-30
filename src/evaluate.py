"""
Model Evaluation Module for Water Quality Assessment System.

Generates: Accuracy, Precision, Recall, F1, ROC-AUC,
Confusion Matrix, Feature Importance, SHAP plots,
and human-readable explanations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import shap
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FEATURE_DISPLAY = {
    "pH": "pH Level", "E.C": "Electrical Conductivity",
    "TDS": "Total Dissolved Solids", "CO3": "Carbonate",
    "HCO3": "Bicarbonate", "Cl": "Chloride", "F": "Fluoride",
    "NO3 ": "Nitrate", "SO4": "Sulphate", "Na": "Sodium",
    "K": "Potassium", "Ca": "Calcium", "Mg": "Magnesium",
    "T.H": "Total Hardness", "SAR": "Sodium Adsorption Ratio",
}

LABEL_NAMES = {0: "Not Suitable", 1: "Suitable"}


def evaluate_model():
    """Run full evaluation pipeline and save all outputs."""
    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Load model and data
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    data = joblib.load(os.path.join(MODELS_DIR, "processed_data.joblib"))
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    print("\n  Performance Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Suitable', 'Suitable'])}")

    # --- Confusion Matrix ---
    print("  Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Suitable", "Suitable"],
                yticklabels=["Not Suitable", "Suitable"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # --- Correlation Heatmap ---
    print("  Generating correlation heatmap...")
    X_raw = data.get("X_train_raw", data["X_train"])
    df_corr = pd.DataFrame(X_raw, columns=feature_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df_corr.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # --- Feature Importance ---
    print("  Generating feature importance plot...")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        display_names = [FEATURE_DISPLAY.get(feature_names[i], feature_names[i]) for i in indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), importances[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title("Feature Importance (Best Model)", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    # --- SHAP Analysis ---
    print("  Generating SHAP explanations (this may take a moment)...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Class 1 (Suitable)
        else:
            shap_vals = shap_values

        # SHAP Summary Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test,
                         feature_names=[FEATURE_DISPLAY.get(f, f) for f in feature_names],
                         show=False)
        plt.title("SHAP Feature Impact Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

        # Individual prediction explanation (first test sample)
        idx = 0
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
                data=X_test[idx],
                feature_names=[FEATURE_DISPLAY.get(f, f) for f in feature_names],
            ),
            show=False
        )
        plt.title("Individual Prediction Explanation (Sample #1)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "shap_individual.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

        # Human-readable explanation for the sample
        print("\n  --- Human-Readable Explanation (Sample #1) ---")
        pred_label = LABEL_NAMES[int(y_pred[idx])]
        conf = y_proba[idx] if y_pred[idx] == 1 else 1 - y_proba[idx]
        top_indices = np.argsort(np.abs(shap_vals[idx]))[::-1][:3]
        factors = []
        for i in top_indices:
            fname = FEATURE_DISPLAY.get(feature_names[i], feature_names[i])
            direction = "high" if shap_vals[idx][i] > 0 else "low"
            factors.append(f"{direction} {fname}")
        explanation = (
            f"  Prediction: {pred_label} (Confidence: {conf:.1%})\n"
            f"  Key factors: {', '.join(factors)} contributed to this classification."
        )
        print(explanation)

        # Save SHAP values
        joblib.dump({"shap_values": shap_vals, "explainer": explainer},
                    os.path.join(MODELS_DIR, "shap_data.joblib"))

    except Exception as e:
        print(f"  Warning: SHAP analysis failed: {e}")

    print(f"\n  All outputs saved to: {OUTPUTS_DIR}/")
    print("=" * 60)
    print("  EVALUATION COMPLETE ✓")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    evaluate_model()
