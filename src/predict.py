"""
Prediction Module for Water Quality Assessment System.

Loads the saved model and preprocessing pipeline, accepts new input
data, and returns prediction with confidence and SHAP explanation.
"""

import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMNS = [
    "pH", "E.C", "TDS", "CO3", "HCO3", "Cl", "F",
    "NO3 ", "SO4", "Na", "K", "Ca", "Mg", "T.H", "SAR"
]

FEATURE_DISPLAY = {
    "pH": "pH Level", "E.C": "Electrical Conductivity",
    "TDS": "Total Dissolved Solids", "CO3": "Carbonate",
    "HCO3": "Bicarbonate", "Cl": "Chloride", "F": "Fluoride",
    "NO3 ": "Nitrate", "SO4": "Sulphate", "Na": "Sodium",
    "K": "Potassium", "Ca": "Calcium", "Mg": "Magnesium",
    "T.H": "Total Hardness", "SAR": "Sodium Adsorption Ratio",
}

LABEL_NAMES = {0: "Not Suitable for Agriculture", 1: "Suitable for Agriculture"}


def load_artifacts():
    """Load model, pipeline, and SHAP explainer."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    pipeline = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))
    shap_data = None
    shap_path = os.path.join(MODELS_DIR, "shap_data.joblib")
    if os.path.exists(shap_path):
        shap_data = joblib.load(shap_path)
    return model, pipeline, shap_data


def predict(input_data: dict) -> dict:
    """
    Make a prediction for a single water sample.

    Args:
        input_data: dict with keys matching FEATURE_COLUMNS.
            Example: {"pH": 8.1, "E.C": 1200, "TDS": 768, ...}

    Returns:
        dict with keys: prediction, label, confidence, explanation, top_factors
    """
    model, pipeline, shap_data = load_artifacts()

    # Build feature array in correct order
    values = []
    for col in FEATURE_COLUMNS:
        val = input_data.get(col, input_data.get(col.strip(), np.nan))
        values.append(float(val) if val is not None else np.nan)

    X_raw = np.array(values).reshape(1, -1)
    X_processed = pipeline.transform(X_raw)

    # Prediction
    pred = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0]
    confidence = proba[pred]
    label = LABEL_NAMES[int(pred)]

    # SHAP explanation
    explanation = ""
    top_factors = []
    if shap_data is not None:
        try:
            import shap
            explainer = shap_data["explainer"]
            shap_vals = explainer.shap_values(X_processed)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]  # Class 1
            else:
                sv = shap_vals[0]

            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            for i in top_idx:
                fname = FEATURE_DISPLAY.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i])
                direction = "High" if sv[i] > 0 else "Low"
                impact = "positive" if (sv[i] > 0 and pred == 1) or (sv[i] < 0 and pred == 0) else "negative"
                top_factors.append({
                    "feature": fname,
                    "value": values[i],
                    "direction": direction,
                    "impact": impact,
                    "shap_value": float(sv[i]),
                })

            factor_strs = [f"{f['direction'].lower()} {f['feature']}" for f in top_factors[:3]]
            explanation = (
                f"The water is classified as '{label}' with {confidence:.1%} confidence. "
                f"Key contributing factors: {', '.join(factor_strs)}."
            )
        except Exception:
            explanation = f"Classified as '{label}' with {confidence:.1%} confidence."
    else:
        explanation = f"Classified as '{label}' with {confidence:.1%} confidence."

    return {
        "prediction": int(pred),
        "label": label,
        "confidence": float(confidence),
        "explanation": explanation,
        "top_factors": top_factors,
        "probabilities": {"suitable": float(proba[1]), "not_suitable": float(proba[0])},
    }


# Example usage
if __name__ == "__main__":
    sample = {
        "pH": 8.04, "E.C": 1065, "TDS": 682, "CO3": 0,
        "HCO3": 230, "Cl": 150, "F": 0.72, "NO3 ": 56.4,
        "SO4": 69, "Na": 85, "K": 3, "Ca": 64, "Mg": 53,
        "T.H": 380, "SAR": 1.90,
    }
    result = predict(sample)
    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    print(f"  Label:      {result['label']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Explanation: {result['explanation']}")
    if result["top_factors"]:
        print("\n  Top Contributing Factors:")
        for f in result["top_factors"]:
            print(f"    • {f['feature']}: {f['value']:.2f} ({f['direction']}, SHAP={f['shap_value']:.4f})")
