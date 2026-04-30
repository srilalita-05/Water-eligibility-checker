"""
Master runner script - Executes the entire ML pipeline:
1. Copies dataset to data/
2. Runs preprocessing
3. Trains models
4. Evaluates + generates plots
5. Tests prediction module
"""

import os
import sys
import shutil

# Ensure we're in the right directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

print("\n" + "=" * 60)
print("  MASTER PIPELINE RUNNER")
print("=" * 60)

# Step 0: Copy dataset
print("\n[0/5] Copying dataset to data/ folder...")
src_csv = os.path.join(BASE_DIR, "ground_water_quality.csv")
dst_csv = os.path.join(BASE_DIR, "data", "ground_water_quality.csv")
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
if os.path.exists(src_csv):
    shutil.copy2(src_csv, dst_csv)
    print(f"  Copied to {dst_csv}")
else:
    print(f"  CSV already in data/ or not found at root.")

# Step 1: Preprocessing
print("\n" + "=" * 60)
print("[1/5] PREPROCESSING")
print("=" * 60)
from src.preprocessing import preprocess_data
data = preprocess_data()
print("  ✓ Preprocessing done!")

# Step 2: Training
print("\n" + "=" * 60)
print("[2/5] MODEL TRAINING")
print("=" * 60)
from src.train_model import train_models
best_model, results = train_models()
print("  ✓ Training done!")

# Step 3: Evaluation
print("\n" + "=" * 60)
print("[3/5] MODEL EVALUATION")
print("=" * 60)
from src.evaluate import evaluate_model
metrics = evaluate_model()
print("  ✓ Evaluation done!")

# Step 4: Prediction test
print("\n" + "=" * 60)
print("[4/5] PREDICTION TEST")
print("=" * 60)
from src.predict import predict

# Test with a known sample
sample = {
    "pH": 8.04, "E.C": 1065, "TDS": 682, "CO3": 0,
    "HCO3": 230, "Cl": 150, "F": 0.72, "NO3 ": 56.4,
    "SO4": 69, "Na": 85, "K": 3, "Ca": 64, "Mg": 53,
    "T.H": 380, "SAR": 1.90,
}
result = predict(sample)
print(f"  Sample prediction: {result['label']}")
print(f"  Confidence: {result['confidence']:.1%}")
print(f"  Explanation: {result['explanation']}")
print("  ✓ Prediction module works!")

# Test with an unsuitable sample
sample2 = {
    "pH": 9.13, "E.C": 3790, "TDS": 2426, "CO3": 50,
    "HCO3": 1000, "Cl": 420, "F": 2.30, "NO3 ": 1.45,
    "SO4": 109, "Na": 800, "K": 32, "Ca": 16, "Mg": 19,
    "T.H": 120, "SAR": 31.76,
}
result2 = predict(sample2)
print(f"\n  Unsuitable sample: {result2['label']}")
print(f"  Confidence: {result2['confidence']:.1%}")
print(f"  Explanation: {result2['explanation']}")

# Step 5: Summary
print("\n" + "=" * 60)
print("[5/5] VERIFICATION SUMMARY")
print("=" * 60)

# Check all outputs exist
checks = [
    ("models/preprocessing_pipeline.joblib", "Preprocessing pipeline"),
    ("models/best_model.joblib", "Best model"),
    ("models/processed_data.joblib", "Processed data"),
    ("models/model_metadata.joblib", "Model metadata"),
    ("outputs/confusion_matrix.png", "Confusion matrix plot"),
    ("outputs/correlation_heatmap.png", "Correlation heatmap"),
    ("outputs/feature_importance.png", "Feature importance plot"),
    ("outputs/shap_summary.png", "SHAP summary plot"),
    ("outputs/shap_individual.png", "SHAP individual plot"),
]

all_ok = True
for path, name in checks:
    full = os.path.join(BASE_DIR, path)
    exists = os.path.exists(full)
    status = "✓" if exists else "✗"
    if not exists:
        all_ok = False
    print(f"  {status} {name}: {path}")

print(f"\n  Metrics: {metrics}")
print("\n" + "=" * 60)
if all_ok:
    print("  ALL CHECKS PASSED ✓")
else:
    print("  SOME CHECKS FAILED — see above")
print("=" * 60)
print("\n  To launch the web app, run:")
print("    streamlit run app/streamlit_app.py")
print()
