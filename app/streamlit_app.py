"""
Streamlit Web Application — AI-Based Freshwater Quality Assessment
Premium UI with multi-page navigation, visualizations, and SHAP explanations.
"""

import os, sys, numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.predict import predict, FEATURE_COLUMNS, FEATURE_DISPLAY, load_artifacts

# ── Page Config ──
st.set_page_config(page_title="AquaCheck AI", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

# ── Premium CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1628 0%, #0f2847 50%, #0a1628 100%); }
[data-testid="stSidebar"] * { color: #c8daf0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }

/* Hero */
.hero { text-align:center; padding:2rem 1rem 1.5rem; 
    background: linear-gradient(135deg, #0a1628, #1a3a5c, #0d4a6b);
    border-radius:20px; margin-bottom:1.5rem; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
    background: radial-gradient(circle at 30% 40%, rgba(56,189,248,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(14,165,233,0.06) 0%, transparent 50%);
    animation: shimmer 8s ease-in-out infinite alternate; }
@keyframes shimmer { 0%{transform:translate(0,0)} 100%{transform:translate(-5%,3%)} }
.hero h1 { color:#fff; font-size:2.6rem; font-weight:800; margin:0; position:relative;
    background: linear-gradient(135deg, #e0f2fe, #38bdf8, #7dd3fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#94a3b8; font-size:1.05rem; margin-top:0.3rem; position:relative; }
.hero .badge { display:inline-block; padding:0.3rem 0.9rem; background:rgba(56,189,248,0.15);
    border:1px solid rgba(56,189,248,0.3); border-radius:20px; color:#38bdf8;
    font-size:0.78rem; font-weight:600; margin-top:0.8rem; position:relative; letter-spacing:0.5px; }

/* Section cards */
.section-card { background: linear-gradient(135deg, #f8fafc, #f1f5f9); border:1px solid #e2e8f0;
    border-radius:16px; padding:1.5rem; margin-bottom:1rem; }
.section-title { font-size:1.15rem; font-weight:700; color:#1e293b; margin-bottom:0.8rem; }

/* Result cards */
.result-box { padding:2rem; border-radius:16px; text-align:center; margin:1rem 0;
    position:relative; overflow:hidden; }
.result-suitable { background: linear-gradient(135deg, #ecfdf5, #d1fae5, #a7f3d0);
    border:2px solid #34d399; box-shadow:0 8px 32px rgba(52,211,153,0.2); }
.result-unsuitable { background: linear-gradient(135deg, #fef2f2, #fecaca, #fca5a5);
    border:2px solid #f87171; box-shadow:0 8px 32px rgba(248,113,113,0.2); }
.result-icon { font-size:3.5rem; margin-bottom:0.3rem; }
.result-label { font-size:1.9rem; font-weight:800; }
.result-conf { font-size:1.1rem; color:#64748b; margin-top:0.2rem; font-weight:500; }

/* Factor pills */
.factor-pill { display:flex; align-items:center; gap:0.7rem; padding:0.7rem 1rem;
    background:#fff; border:1px solid #e2e8f0; border-radius:12px; margin:0.4rem 0;
    transition: transform 0.2s, box-shadow 0.2s; }
.factor-pill:hover { transform:translateX(4px); box-shadow:0 4px 12px rgba(0,0,0,0.06); }
.factor-pill .icon { font-size:1.3rem; }
.factor-pill .name { font-weight:600; color:#1e293b; flex:1; }
.factor-pill .val { font-weight:500; color:#64748b; font-size:0.9rem; }
.factor-pill .shap-pos { color:#059669; font-weight:700; font-size:0.85rem; }
.factor-pill .shap-neg { color:#dc2626; font-weight:700; font-size:0.85rem; }

/* Metric cards */
.metric-row { display:flex; gap:1rem; margin:1rem 0; flex-wrap:wrap; }
.metric-card { flex:1; min-width:140px; text-align:center; padding:1rem; background:#fff;
    border-radius:12px; border:1px solid #e2e8f0; }
.metric-card .val { font-size:1.6rem; font-weight:800; }
.metric-card .lbl { font-size:0.8rem; color:#64748b; margin-top:0.2rem; font-weight:500; }
.mc-green .val { color:#059669; }
.mc-red .val { color:#dc2626; }
.mc-blue .val { color:#2563eb; }

/* Predict button */
.stButton > button { width:100%; padding:0.85rem; font-size:1.05rem; font-weight:700;
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    letter-spacing:0.3px; transition: all 0.3s !important; box-shadow:0 4px 16px rgba(37,99,235,0.3); }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(37,99,235,0.4) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:0.5rem; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:0.5rem 1.2rem; font-weight:600; }

/* Footer */
.footer { text-align:center; color:#94a3b8; font-size:0.8rem; padding:2rem 0 1rem;
    border-top:1px solid #e2e8f0; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
    <h1>💧 AquaCheck AI</h1>
    <p>Intelligent Freshwater Quality Assessment for Sustainable Agriculture</p>
    <div class="badge">🤖 POWERED BY MACHINE LEARNING & SHAP EXPLAINABILITY</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 💧 AquaCheck AI")
    st.markdown("---")
    st.markdown("### 📋 Navigation")
    page = st.radio("", ["🧪 Predict", "📊 Dashboard", "📖 About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    try:
        import joblib
        meta = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.joblib"))
        st.markdown(f"**Algorithm:** {meta['best_model_name']}")
        st.markdown(f"**ROC-AUC:** `{meta['test_auc']:.4f}`")
        st.markdown(f"**Status:** 🟢 Ready")
    except Exception:
        st.warning("Model not trained yet")
    st.markdown("---")
    st.markdown("### 🎯 Quick Presets")
    preset = st.selectbox("Load sample data", ["Custom", "Safe Water Sample", "Unsafe Water Sample"], label_visibility="collapsed")

# ── Preset values ──
SAFE_DEFAULTS = {"pH":7.8,"E.C":800.0,"TDS":500.0,"CO3":0.0,"HCO3":200.0,"Cl":80.0,"F":0.5,
    "NO3 ":20.0,"SO4":50.0,"Na":60.0,"K":3.0,"Ca":50.0,"Mg":30.0,"T.H":250.0,"SAR":1.5}
UNSAFE_DEFAULTS = {"pH":9.1,"E.C":3790.0,"TDS":2426.0,"CO3":50.0,"HCO3":1000.0,"Cl":420.0,"F":2.3,
    "NO3 ":1.5,"SO4":109.0,"Na":800.0,"K":32.0,"Ca":16.0,"Mg":19.0,"T.H":120.0,"SAR":31.8}
RANGES = {"pH":(0.0,14.0),"E.C":(0.0,10000.0),"TDS":(0.0,6000.0),"CO3":(0.0,200.0),
    "HCO3":(0.0,1500.0),"Cl":(0.0,2000.0),"F":(0.0,10.0),"NO3 ":(0.0,500.0),
    "SO4":(0.0,2000.0),"Na":(0.0,1500.0),"K":(0.0,300.0),"Ca":(0.0,500.0),
    "Mg":(0.0,500.0),"T.H":(0.0,2000.0),"SAR":(0.0,40.0)}
UNITS = {"pH":"","E.C":"µS/cm","TDS":"mg/L","CO3":"mg/L","HCO3":"mg/L","Cl":"mg/L",
    "F":"mg/L","NO3 ":"mg/L","SO4":"mg/L","Na":"mg/L","K":"mg/L","Ca":"mg/L",
    "Mg":"mg/L","T.H":"mg/L","SAR":""}

defaults = SAFE_DEFAULTS if preset == "Safe Water Sample" else (UNSAFE_DEFAULTS if preset == "Unsafe Water Sample" else SAFE_DEFAULTS)

# ════════════════════════════════════════════════
#  PAGE: PREDICT
# ════════════════════════════════════════════════
if page == "🧪 Predict":
    st.markdown('<div class="section-card"><div class="section-title">🧪 Enter Water Chemical Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    input_values = {}
    icons = {"pH":"⚗️","E.C":"⚡","TDS":"🧂","CO3":"🔬","HCO3":"🧫","Cl":"🟢","F":"🔵",
             "NO3 ":"🟡","SO4":"🟠","Na":"🔴","K":"🟣","Ca":"⚪","Mg":"🟤","T.H":"💎","SAR":"📐"}
    
    for i, col_name in enumerate(FEATURE_COLUMNS):
        disp = FEATURE_DISPLAY.get(col_name, col_name)
        unit = UNITS.get(col_name, "")
        label = f"{icons.get(col_name,'')} {disp}" + (f" ({unit})" if unit else "")
        default = defaults.get(col_name, 0.0)
        mn, mx = RANGES.get(col_name, (0.0, 10000.0))
        with [col1, col2, col3][i % 3]:
            input_values[col_name] = st.number_input(label, min_value=mn, max_value=mx, value=default, step=0.1, key=col_name)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")

    if st.button("🔍  Analyze Water Quality", use_container_width=True):
        with st.spinner("🔬 Running AI analysis..."):
            try:
                result = predict(input_values)
                is_safe = result["prediction"] == 1
                css = "result-suitable" if is_safe else "result-unsuitable"
                icon = "✅" if is_safe else "❌"
                color = "#059669" if is_safe else "#dc2626"

                st.markdown(f"""
                <div class="result-box {css}">
                    <div class="result-icon">{icon}</div>
                    <div class="result-label" style="color:{color}">{result['label']}</div>
                    <div class="result-conf">Confidence: {result['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics row
                suit_pct = result['probabilities']['suitable'] * 100
                unsuit_pct = result['probabilities']['not_suitable'] * 100
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card mc-green"><div class="val">{suit_pct:.1f}%</div><div class="lbl">Suitable Probability</div></div>
                    <div class="metric-card mc-red"><div class="val">{unsuit_pct:.1f}%</div><div class="lbl">Not Suitable Probability</div></div>
                    <div class="metric-card mc-blue"><div class="val">{result['confidence']:.1%}</div><div class="lbl">Model Confidence</div></div>
                </div>
                """, unsafe_allow_html=True)

                # Explanation
                st.markdown("")
                st.info(f"📝 **AI Explanation:** {result['explanation']}")

                # Two-column layout: Factors + SHAP chart
                col_f, col_s = st.columns([1, 1])
                
                with col_f:
                    st.markdown("#### 🔬 Top Contributing Factors")
                    if result["top_factors"]:
                        for f in result["top_factors"]:
                            ic = "🔺" if f["shap_value"] > 0 else "🔻"
                            shap_cls = "shap-pos" if f["shap_value"] > 0 else "shap-neg"
                            st.markdown(f"""
                            <div class="factor-pill">
                                <div class="icon">{ic}</div>
                                <div class="name">{f['feature']}</div>
                                <div class="val">{f['value']:.2f}</div>
                                <div class="{shap_cls}">SHAP: {f['shap_value']:+.3f}</div>
                            </div>""", unsafe_allow_html=True)

                with col_s:
                    st.markdown("#### 📊 SHAP Waterfall Chart")
                    try:
                        import shap, matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        model, pipeline, shap_data_loaded = load_artifacts()
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
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.caption(f"Waterfall chart unavailable: {e}")

            except FileNotFoundError:
                st.error("⚠️ Model not found! Run `python run_all.py` first.")
            except Exception as e:
                st.error(f"Error: {e}")

# ════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("### 📊 Model Performance & Visualizations")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Confusion Matrix", "🔥 Correlation Heatmap", "🏆 Feature Importance", "🧠 SHAP Summary", "🎯 SHAP Individual"])

    plot_map = {
        "📈 Confusion Matrix": "confusion_matrix.png",
        "🔥 Correlation Heatmap": "correlation_heatmap.png",
        "🏆 Feature Importance": "feature_importance.png",
        "🧠 SHAP Summary": "shap_summary.png",
        "🎯 SHAP Individual": "shap_individual.png",
    }
    
    tabs = [tab1, tab2, tab3, tab4, tab5]
    descs = [
        "Shows how many predictions were correct vs incorrect for each class.",
        "Reveals relationships between different water parameters.",
        "Ranks which chemical parameters have the most influence on predictions.",
        "Global SHAP values showing each feature's impact across all test samples.",
        "SHAP waterfall for a single test sample showing how each feature pushed the prediction.",
    ]
    
    for tab, (name, fname), desc in zip(tabs, plot_map.items(), descs):
        with tab:
            path = os.path.join(BASE_DIR, "outputs", fname)
            st.markdown(f"**{name.split(' ',1)[1]}**")
            st.caption(desc)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Plot not found. Run `python src/evaluate.py` first.")
    
    # Model metrics
    st.markdown("---")
    st.markdown("### 📋 Model Metrics")
    try:
        import joblib
        meta = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.joblib"))
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Model", meta["best_model_name"])
        c2.metric("📈 Test ROC-AUC", f"{meta['test_auc']:.4f}")
        c3.metric("✅ Status", "Production Ready")
    except Exception:
        st.info("Train the model to see metrics.")

# ════════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════════
elif page == "📖 About":
    st.markdown("### 📖 About This System")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="section-card">
        <div class="section-title">🎯 Purpose</div>
        This AI system analyzes <b>15 chemical parameters</b> of groundwater to determine 
        if it is <b>suitable for agricultural irrigation</b>. Built with interpretable ML, 
        every prediction comes with a clear explanation of <i>why</i> the water was classified 
        the way it was.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
        <div class="section-title">🔬 Parameters Analyzed</div>
        <table style="width:100%; font-size:0.9rem;">
        <tr><td>⚗️ pH Level</td><td>⚡ Electrical Conductivity</td></tr>
        <tr><td>🧂 Total Dissolved Solids</td><td>🔬 Carbonate / Bicarbonate</td></tr>
        <tr><td>🟢 Chloride</td><td>🔵 Fluoride</td></tr>
        <tr><td>🟡 Nitrate</td><td>🟠 Sulphate</td></tr>
        <tr><td>🔴 Sodium</td><td>🟣 Potassium</td></tr>
        <tr><td>⚪ Calcium</td><td>🟤 Magnesium</td></tr>
        <tr><td>💎 Total Hardness</td><td>📐 Sodium Adsorption Ratio</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="section-card">
        <div class="section-title">🤖 ML Pipeline</div>
        <ol>
        <li><b>Preprocessing:</b> Median imputation + StandardScaler</li>
        <li><b>Models:</b> Random Forest, Gradient Boosting, XGBoost</li>
        <li><b>Tuning:</b> RandomizedSearchCV (30 iter, 5-fold CV)</li>
        <li><b>Selection:</b> Best model by ROC-AUC</li>
        <li><b>Explainability:</b> SHAP TreeExplainer</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
        <div class="section-title">📊 Dataset</div>
        <ul>
        <li><b>Source:</b> Telangana Groundwater Quality Data</li>
        <li><b>Samples:</b> ~1,000 water samples across 30+ districts</li>
        <li><b>Season:</b> Post-monsoon 2021</li>
        <li><b>Target:</b> P.S. (Safe) / MR (Marginal) / U.S. (Unsuitable)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card" style="text-align:center;">
    <div class="section-title">🚀 How to Use</div>
    <p>1️⃣ Go to <b>🧪 Predict</b> → 2️⃣ Enter water parameters → 3️⃣ Click <b>Analyze</b> → 4️⃣ Get results with AI explanation</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown('<div class="footer">💧 AquaCheck AI — Freshwater Quality Assessment System | Built with Scikit-learn, SHAP & Streamlit</div>', unsafe_allow_html=True)
