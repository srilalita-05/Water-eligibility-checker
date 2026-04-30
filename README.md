<<<<<<< HEAD
🌱 AI-Based Freshwater Quality Assessment System for Sustainable Agriculture 

📌 Project Overview

Agriculture depends heavily on water quality, yet in many rural and semi-urban areas, farmers lack access to fast and affordable water testing methods. This project proposes a Machine Learning-based Decision Support System that classifies freshwater quality using chemical parameters and provides interpretable insights.

The goal is not just prediction, but actionable understanding—helping farmers and agricultural stakeholders make informed irrigation decisions.


🎯 Problem Statement

Farmers often rely on borewell, canal, or groundwater without analyzing its chemical composition. Poor-quality water can cause:

Soil degradation

Reduced crop yield

Nutrient imbalance

Long-term fertility loss


Traditional lab testing is:

Time-consuming

Expensive

Not easily accessible


Need: A fast, data-driven system for preliminary water quality assessment.


---
Proposed Solution

We develop a supervised machine learning system that:

Accepts freshwater chemical parameters as input

Classifies water into quality categories (e.g., Suitable / Not Suitable)

Identifies key factors affecting quality

Provides interpretable explanations


🎯 Objectives

1. Perform data preprocessing and cleaning


2. Build classification models for water quality prediction


3. Compare ensemble learning models


4. Apply interpretability techniques


5. Develop a simple prototype UI (optional)



📊 Dataset Description

The dataset includes the following features:

pH

Hardness

Total Dissolved Solids (TDS)

Chloramines

Sulfates

Conductivity

Organic Carbon

Trihalomethanes

Turbidity


These parameters influence irrigation suitability and soil health.


🧠 Methodology

We follow the CRISP-DM Framework:

1. Business Understanding
Define agricultural problem and impact


2. Data Understanding
Analyze dataset structure and distributions


3. Data Preparation
Clean, preprocess, and transform data


4. Model Development
Train classification models


5. Evaluation
Compare performance using metrics


6. Interpretation
Explain model predictions



⚙️ Technical Implementation

🧹 Data Preprocessing

Handling missing values (mean/median imputation)

Removing duplicates

Feature scaling (if required)

Stratified train-test split (80–20)


🤖 Machine Learning Models

Random Forest Classifier

Gradient Boosting Classifier

(Optional) XGBoost


Why Ensemble Models?

Handle nonlinear relationships

Robust to noisy data

High performance on structured datasets

Provide feature importance



---

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC-AUC


👉 Multiple metrics ensure balanced evaluation, especially for imbalanced datasets.


---

🔍 Interpretability

We use:

Feature Importance

SHAP (based on Shapley Value)


Example Insight:

> High TDS and abnormal pH significantly contribute to poor water quality classification.




---

🛠️ Tech Stack

Programming Language

Python


Libraries

NumPy

Pandas

Scikit-learn

XGBoost (optional)

SHAP

Matplotlib


Tools

Jupyter Notebook

VS Code


Optional UI

Streamlit



---

🏗️ System Architecture

Input (Water Chemical Parameters)
        ↓
Data Preprocessing
        ↓
Trained ML Model
        ↓
Prediction Output
        ↓
Interpretability (SHAP / Feature Importance)


---

📁 Project Structure

water-quality-ml/
│
├── data/
│   └── water_quality.csv
│
├── notebooks/
│   └── eda_and_model.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── trained_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── report.pdf


---

🚀 How to Run the Project

1. Clone Repository

git clone https://github.com/your-username/water-quality-ml.git
cd water-quality-ml

2. Install Dependencies

pip install -r requirements.txt

3. Run Notebook

jupyter notebook

4. Run Streamlit App (Optional)

streamlit run app/streamlit_app.py


---

📌 Expected Outcomes

Model accuracy: ~85–90%

Clear feature importance insights

Comparison of ensemble models

Optional UI prototype



---

🌍 Real-World Impact

Helps farmers avoid harmful irrigation water

Assists agricultural officers in screening

Promotes sustainable soil management

Reduces dependency on costly lab testing



---

🔮 Future Enhancements

IoT-based real-time water monitoring

Crop-specific recommendations

Mobile application for rural users

Integration with government agricultural systems



---

🎤 Project Explanation (Short Script)

> “This project develops a machine learning-based decision support system to classify freshwater quality for agricultural use. It analyzes chemical parameters such as pH and dissolved solids, predicts suitability, and provides interpretable insights using SHAP. The goal is to support farmers in making informed irrigation decisions and promote sustainable agriculture.”




---

📚 Key Learnings

Practical application of machine learning in agriculture

Importance of data preprocessing

Model evaluation beyond accuracy

Explainable AI techniques

End-to-end ML project development



---

👨‍💻 Contributors
lalita 
Srimuktha
ganesh



---

📜 License

This project is for academic and research purposes.


---

🙌 Acknowledgment

We acknowledge the use of open datasets and Python ML libraries that made this project possible.

=======
# 💧 AI-Based Freshwater Quality Assessment System for Sustainable Agriculture

A production-ready, modular, and interpretable Machine Learning system that classifies groundwater quality for agricultural use based on chemical parameters.

## 🎯 Project Overview

This system uses supervised classification to analyze freshwater chemical parameters and predict whether water is **suitable** or **not suitable** for agricultural irrigation. It provides SHAP-based interpretable explanations for every prediction, making it ideal for academic presentations and real-world deployment.

## 📁 Project Structure

```
water_elegibility_checker/
├── data/
│   └── ground_water_quality.csv     # Raw dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data cleaning & pipeline
│   ├── train_model.py                # Model training & tuning
│   ├── evaluate.py                   # Evaluation & visualizations
│   └── predict.py                    # Prediction module
├── models/                           # Saved models & pipelines
├── outputs/                          # Generated plots
├── app/
│   └── streamlit_app.py              # Web UI
├── requirements.txt
└── README.md
```

## 🔬 Features Analyzed

| Parameter | Description |
|-----------|-------------|
| pH | Acidity/Alkalinity of water |
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

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing

```bash
python src/preprocessing.py
```

### 3. Train Models

```bash
python src/train_model.py
```

### 4. Evaluate

```bash
python src/evaluate.py
```

### 5. Make Predictions

```bash
python src/predict.py
```

### 6. Launch Web App

```bash
streamlit run app/streamlit_app.py
```

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| **Random Forest** | Ensemble of decision trees with bagging |
| **Gradient Boosting** | Sequential boosting with gradient descent |
| **XGBoost** | Optimized gradient boosting (optional) |

All models are tuned using **RandomizedSearchCV** with **5-fold stratified cross-validation**.

## 📊 Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **Precision** — Correctness of positive predictions
- **Recall** — Coverage of actual positives
- **F1-Score** — Harmonic mean of precision & recall
- **ROC-AUC** — Area under the ROC curve

## 🧠 Interpretability

- **Feature Importance** — Built-in model feature rankings
- **SHAP Summary Plot** — Global feature impact visualization
- **SHAP Waterfall** — Individual prediction explanations
- **Human-readable explanations** — Natural language output

### Example Output
```
Prediction: Not Suitable for Agriculture (Confidence: 87.3%)
Key factors: high Sodium Adsorption Ratio, high Total Dissolved Solids,
low Calcium contributed to this classification.
```

## 🌐 Web Application

The Streamlit app provides:
- Input fields for all 15 water parameters
- Real-time prediction with confidence score
- SHAP-based visual explanation
- Top contributing factors with direction and impact

## 📦 Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- shap, joblib, streamlit
- xgboost (optional)

## 👩‍💻 Author

Built as an academic ML project demonstrating end-to-end machine learning pipeline development with emphasis on interpretability and production readiness.

## 📄 License

This project is for educational purposes.
>>>>>>> f1108bf (Remove tracked binaries and data files)
