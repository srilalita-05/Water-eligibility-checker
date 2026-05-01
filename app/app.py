"""
Flask Backend — AI-Based Water Quality Eligibility Checker
REST API + Template rendering for the premium UI.
"""

import os, sys, io, base64, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime

# ── Path Setup ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.predict import predict, FEATURE_COLUMNS, FEATURE_DISPLAY, load_artifacts

app = Flask(__name__)
app.secret_key = "aquacheck-water-quality-2024"

# ═══════════════════════════════════
#  FEATURE METADATA
# ═══════════════════════════════════

SAFE_DEFAULTS = {
    "pH": 7.8, "E.C": 800.0, "TDS": 500.0, "CO3": 0.0, "HCO3": 200.0,
    "Cl": 80.0, "F": 0.5, "NO3 ": 20.0, "SO4": 50.0, "Na": 60.0,
    "K": 3.0, "Ca": 50.0, "Mg": 30.0, "T.H": 250.0, "SAR": 1.5
}

UNSAFE_DEFAULTS = {
    "pH": 9.1, "E.C": 3790.0, "TDS": 2426.0, "CO3": 50.0, "HCO3": 1000.0,
    "Cl": 420.0, "F": 2.3, "NO3 ": 1.5, "SO4": 109.0, "Na": 800.0,
    "K": 32.0, "Ca": 16.0, "Mg": 19.0, "T.H": 120.0, "SAR": 31.8
}

RANGES = {
    "pH": [0, 14], "E.C": [0, 10000], "TDS": [0, 6000], "CO3": [0, 200],
    "HCO3": [0, 1500], "Cl": [0, 2000], "F": [0, 10], "NO3 ": [0, 500],
    "SO4": [0, 2000], "Na": [0, 1500], "K": [0, 300], "Ca": [0, 500],
    "Mg": [0, 500], "T.H": [0, 2000], "SAR": [0, 40]
}

UNITS = {
    "pH": "", "E.C": "µS/cm", "TDS": "mg/L", "CO3": "mg/L", "HCO3": "mg/L",
    "Cl": "mg/L", "F": "mg/L", "NO3 ": "mg/L", "SO4": "mg/L", "Na": "mg/L",
    "K": "mg/L", "Ca": "mg/L", "Mg": "mg/L", "T.H": "mg/L", "SAR": ""
}

ICONS = {
    "pH": "⚗️", "E.C": "⚡", "TDS": "🧂", "CO3": "🔬", "HCO3": "🧫",
    "Cl": "🟢", "F": "🔵", "NO3 ": "🟡", "SO4": "🟠", "Na": "🔴",
    "K": "🟣", "Ca": "⚪", "Mg": "🟤", "T.H": "💎", "SAR": "📐"
}

STEPS = {
    "pH": 0.01, "E.C": 1, "TDS": 1, "CO3": 1, "HCO3": 1,
    "Cl": 1, "F": 0.01, "NO3 ": 0.1, "SO4": 1, "Na": 1,
    "K": 0.1, "Ca": 1, "Mg": 1, "T.H": 1, "SAR": 0.01
}

TOOLTIPS = {
    "pH": "Acidity/alkalinity measure. 7 is neutral. Ideal for agriculture: 6.5–8.5.",
    "E.C": "Electrical Conductivity — indicates dissolved salt concentration.",
    "TDS": "Total Dissolved Solids — total mineral content in water.",
    "CO3": "Carbonate ion concentration. Affects soil alkalinity.",
    "HCO3": "Bicarbonate — major component of water alkalinity.",
    "Cl": "Chloride — high levels can harm salt-sensitive crops.",
    "F": "Fluoride — excessive levels are toxic to plants and humans.",
    "NO3 ": "Nitrate — essential nutrient, but excess causes algal blooms.",
    "SO4": "Sulphate — high levels can cause soil crusting.",
    "Na": "Sodium — excess causes soil structure deterioration.",
    "K": "Potassium — essential nutrient for plant growth.",
    "Ca": "Calcium — helps maintain soil structure.",
    "Mg": "Magnesium — essential for chlorophyll production.",
    "T.H": "Total Hardness — combined calcium and magnesium.",
    "SAR": "Sodium Adsorption Ratio — key indicator of sodium hazard."
}


# ═══════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze")
def analyze():
    preset = request.args.get("preset", "safe")
    defaults = UNSAFE_DEFAULTS if preset == "unsafe" else SAFE_DEFAULTS
    return render_template("analyze.html",
                           features=FEATURE_COLUMNS,
                           display=FEATURE_DISPLAY,
                           units=UNITS, icons=ICONS,
                           ranges=RANGES, steps=STEPS,
                           tooltips=TOOLTIPS,
                           defaults=defaults)


@app.route("/dashboard")
def dashboard():
    plot_files = [
        ("confusion_matrix.png", "Confusion Matrix", "Correct vs incorrect predictions per class."),
        ("correlation_heatmap.png", "Correlation Heatmap", "Relationships between water parameters."),
        ("feature_importance.png", "Feature Importance", "Which parameters influence predictions most."),
        ("shap_summary.png", "SHAP Summary", "Global feature impact across all test samples."),
        ("shap_individual.png", "SHAP Individual", "How each feature pushed a single prediction."),
    ]
    plots = []
    for fname, title, desc in plot_files:
        exists = os.path.exists(os.path.join(BASE_DIR, "outputs", fname))
        plots.append({"file": fname, "title": title, "desc": desc, "exists": exists})

    meta = None
    try:
        import joblib
        meta = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.joblib"))
    except Exception:
        pass
    return render_template("dashboard.html", plots=plots, meta=meta)


@app.route("/about")
def about():
    return render_template("about.html")


# ═══════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    input_values = {}
    for col in FEATURE_COLUMNS:
        val = data.get(col, data.get(col.strip(), 0))
        try:
            input_values[col] = float(val)
        except (ValueError, TypeError):
            input_values[col] = 0.0

    try:
        result = predict(input_values)
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run python run_all.py first."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Generate SHAP waterfall chart as base64
    shap_url = None
    try:
        import shap
        model, pipeline, shap_data = load_artifacts()
        if shap_data and pipeline:
            X_raw = np.array([input_values.get(c, 0) for c in FEATURE_COLUMNS]).reshape(1, -1)
            X_proc = pipeline.transform(X_raw)
            explainer = shap_data["explainer"]
            sv = explainer.shap_values(X_proc)
            if isinstance(sv, list):
                sv = sv[1]
            exp = shap.Explanation(
                values=sv[0],
                base_values=(explainer.expected_value if not isinstance(explainer.expected_value, list)
                             else explainer.expected_value[1]),
                data=X_proc[0],
                feature_names=[FEATURE_DISPLAY.get(f, f) for f in FEATURE_COLUMNS],
            )
            fig, _ = plt.subplots(figsize=(9, 5))
            shap.waterfall_plot(exp, show=False)
            plt.tight_layout()
            img = io.BytesIO()
            fig.savefig(img, format="png", bbox_inches="tight", dpi=120)
            img.seek(0)
            shap_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)
    except Exception as e:
        print(f"SHAP chart error: {e}")

    result["shap_plot"] = shap_url
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["input_values"] = {k: round(v, 3) for k, v in input_values.items()}
    return jsonify(result)


@app.route("/api/model-info")
def api_model_info():
    try:
        import joblib
        meta = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.joblib"))
        return jsonify({"status": "ready", "model": meta["best_model_name"],
                        "auc": round(meta["test_auc"], 4)})
    except Exception:
        return jsonify({"status": "not_trained", "model": None, "auc": None})


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(os.path.join(BASE_DIR, "outputs"), filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
