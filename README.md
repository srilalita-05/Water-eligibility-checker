

```markdown
# 💧 AI-Based Freshwater Quality Assessment System for Sustainable Agriculture

A modular, interpretable Machine Learning system that classifies groundwater suitability for agricultural irrigation using chemical parameters.

---

## 📌 Problem Statement

Farmers in rural and semi-urban areas rely on borewell, canal, or groundwater without analyzing its chemical composition. Poor-quality water leads to:

- Soil degradation and reduced crop yield
- Nutrient imbalance and long-term fertility loss

Traditional lab testing is slow, expensive, and inaccessible. This system provides a fast, data-driven alternative for preliminary water quality assessment.

---

## 🎯 Objectives

1. Preprocess and clean freshwater chemical data
2. Build and compare ensemble classification models
3. Apply interpretability techniques (SHAP)
4. Deploy a simple web UI for real-time prediction

---

## 📁 Project Structure

```
water_quality_ml/
├── data/
│   └── ground_water_quality.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── outputs/
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🔬 Features Analyzed

| Parameter | Description |
|-----------|-------------|
| pH | Acidity / Alkalinity |
| E.C | Electrical Conductivity (µS/cm) |
| TDS | Total Dissolved Solids (mg/L) |
| CO3 | Carbonate (mg/L) |
| HCO3 | Bicarbonate (mg/L) |
| Cl | Chloride (mg/L) |
| F | Fluoride (mg/L) |
| NO3 | Nitrate (mg/L) |
| SO4 | Sulphate (mg/L) |
| Na | Sodium (mg/L) |
| K | Potassium (mg/L) |
| Ca | Calcium (mg/L) |
| Mg | Magnesium (mg/L) |
| T.H | Total Hardness (mg/L) |
| SAR | Sodium Adsorption Ratio |

---

## 🧠 Methodology (CRISP-DM)

1. **Business Understanding** — Define the agricultural problem and impact
2. **Data Understanding** — Analyze dataset structure and distributions
3. **Data Preparation** — Clean, impute missing values, scale features, stratified split (80/20)
4. **Model Development** — Train ensemble classifiers
5. **Evaluation** — Compare models using multiple metrics
6. **Interpretation** — Explain predictions using SHAP

---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of decision trees with bagging |
| Gradient Boosting | Sequential boosting with gradient descent |
| XGBoost | Optimized gradient boosting (optional) |

All models are tuned using **RandomizedSearchCV** with **5-fold stratified cross-validation**.

---

## 📊 Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **Precision** — Correctness of positive predictions
- **Recall** — Coverage of actual positives
- **F1-Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the ROC curve

---

## 🔍 Interpretability

- **Feature Importance** — Built-in model feature rankings
- **SHAP Summary Plot** — Global feature impact across all predictions
- **SHAP Waterfall** — Per-prediction explanation

**Example output:**
```
Prediction: Not Suitable for Agriculture (Confidence: 87.3%)
Key factors: High Sodium Adsorption Ratio, High TDS, Low Calcium
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| ML Libraries | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Interpretability | SHAP |
| Visualization | matplotlib, seaborn |
| UI | Streamlit |
| Environment | Jupyter Notebook, VS Code |

---

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/your-username/water-quality-ml.git
cd water-quality-ml
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run preprocessing**
```bash
python src/preprocessing.py
```

**4. Train models**
```bash
python src/train_model.py
```

**5. Evaluate**
```bash
python src/evaluate.py
```

**6. Launch web app**
```bash
streamlit run app/streamlit_app.py
```

---

## 📌 Expected Outcomes

- Model accuracy: ~85–90%
- Clear feature importance and SHAP insights
- Comparison report across ensemble models
- Functional Streamlit UI prototype

---

## 🌍 Real-World Impact

- Helps farmers avoid harmful irrigation water
- Assists agricultural officers in field screening
- Reduces dependency on costly lab testing
- Promotes sustainable soil and water management

---

## 🔮 Future Enhancements

- IoT-based real-time water monitoring
- Crop-specific irrigation recommendations
- Mobile app for rural users
- Integration with government agricultural data systems

---

## 👨‍💻 Contributors

- Lalita
- Srimuktha
- Ganesh

---

## 📜 License

This project is for academic and research purposes only.

---

## 🙌 Acknowledgments

We acknowledge the use of open datasets and Python ML libraries that made this project possible.
