import os
import sys
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.predict import predict, FEATURE_COLUMNS, FEATURE_DISPLAY, load_artifacts

app = Flask(__name__)

# Constants and Defaults
SAFE_DEFAULTS = {"pH": 7.8, "E.C": 800.0, "TDS": 500.0, "CO3": 0.0, "HCO3": 200.0, "Cl": 80.0, "F": 0.5,
                 "NO3 ": 20.0, "SO4": 50.0, "Na": 60.0, "K": 3.0, "Ca": 50.0, "Mg": 30.0, "T.H": 250.0, "SAR": 1.5}
UNSAFE_DEFAULTS = {"pH": 9.1, "E.C": 3790.0, "TDS": 2426.0, "CO3": 50.0, "HCO3": 1000.0, "Cl": 420.0, "F": 2.3,
                   "NO3 ": 1.5, "SO4": 109.0, "Na": 800.0, "K": 32.0, "Ca": 16.0, "Mg": 19.0, "T.H": 120.0, "SAR": 31.8}
RANGES = {"pH": (0.0, 14.0), "E.C": (0.0, 10000.0), "TDS": (0.0, 6000.0), "CO3": (0.0, 200.0),
          "HCO3": (0.0, 1500.0), "Cl": (0.0, 2000.0), "F": (0.0, 10.0), "NO3 ": (0.0, 500.0),
          "SO4": (0.0, 2000.0), "Na": (0.0, 1500.0), "K": (0.0, 300.0), "Ca": (0.0, 500.0),
          "Mg": (0.0, 500.0), "T.H": (0.0, 2000.0), "SAR": (0.0, 40.0)}
UNITS = {"pH": "", "E.C": "µS/cm", "TDS": "mg/L", "CO3": "mg/L", "HCO3": "mg/L", "Cl": "mg/L",
         "F": "mg/L", "NO3 ": "mg/L", "SO4": "mg/L", "Na": "mg/L", "K": "mg/L", "Ca": "mg/L",
         "Mg": "mg/L", "T.H": "mg/L", "SAR": ""}
ICONS = {"pH": "⚗️", "E.C": "⚡", "TDS": "🧂", "CO3": "🔬", "HCO3": "🧫", "Cl": "🟢", "F": "🔵",
         "NO3 ": "🟡", "SO4": "🟠", "Na": "🔴", "K": "🟣", "Ca": "⚪", "Mg": "🟤", "T.H": "💎", "SAR": "📐"}

def get_model_metadata():
    try:
        import joblib
        meta = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.joblib"))
        return meta
    except Exception:
        return None

@app.context_processor
def inject_global_vars():
    meta = get_model_metadata()
    return dict(meta=meta)

@app.route("/", methods=["GET"])
@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_values = {}
        for col in FEATURE_COLUMNS:
            val = request.form.get(col)
            try:
                input_values[col] = float(val) if val else 0.0
            except ValueError:
                input_values[col] = 0.0

        try:
            result = predict(input_values)
            is_safe = result["prediction"] == 1
            
            # Generate SHAP waterfall chart
            shap_plot_url = None
            try:
                import shap
                model, pipeline, shap_data_loaded = load_artifacts()
                if shap_data_loaded and pipeline:
                    X_raw = np.array([input_values.get(c, 0) for c in FEATURE_COLUMNS]).reshape(1, -1)
                    X_proc = pipeline.transform(X_raw)
                    explainer = shap_data_loaded["explainer"]
                    sv = explainer.shap_values(X_proc)
                    if isinstance(sv, list): sv = sv[1]
                    exp = shap.Explanation(
                        values=sv[0],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                        data=X_proc[0],
                        feature_names=[FEATURE_DISPLAY.get(f, f) for f in FEATURE_COLUMNS],
                    )
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.waterfall_plot(exp, show=False)
                    plt.tight_layout()
                    
                    # Save to BytesIO
                    img = io.BytesIO()
                    fig.savefig(img, format='png', bbox_inches='tight')
                    img.seek(0)
                    shap_plot_url = base64.b64encode(img.getvalue()).decode()
                    plt.close(fig)
            except Exception as e:
                print(f"SHAP error: {e}")

            return render_template("index.html", 
                                 features=FEATURE_COLUMNS, 
                                 display=FEATURE_DISPLAY, 
                                 units=UNITS, 
                                 icons=ICONS, 
                                 defaults=input_values,
                                 ranges=RANGES,
                                 result=result,
                                 is_safe=is_safe,
                                 shap_plot_url=shap_plot_url)
        except Exception as e:
            return render_template("index.html", 
                                 features=FEATURE_COLUMNS, 
                                 display=FEATURE_DISPLAY, 
                                 units=UNITS, 
                                 icons=ICONS, 
                                 defaults=SAFE_DEFAULTS,
                                 ranges=RANGES,
                                 error=str(e))

    # GET request
    preset = request.args.get("preset", "safe")
    defaults = UNSAFE_DEFAULTS if preset == "unsafe" else SAFE_DEFAULTS

    return render_template("index.html", 
                         features=FEATURE_COLUMNS, 
                         display=FEATURE_DISPLAY, 
                         units=UNITS, 
                         icons=ICONS, 
                         defaults=defaults,
                         ranges=RANGES,
                         preset=preset)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/outputs/<filename>")
def serve_output(filename):
    from flask import send_from_directory
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    return send_from_directory(outputs_dir, filename)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
