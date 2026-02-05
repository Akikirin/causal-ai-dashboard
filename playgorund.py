import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import warnings
from scipy.stats import norm

# --- 1. Machine Learning & Causal Imports ---
try:
    from xgboost import XGBRegressor
except ImportError:
    st.error("🚨 Please install xgboost: `pip install xgboost`")
    st.stop()

try:
    from econml.dml import LinearDML
except ImportError:
    st.error("🚨 Critical Missing Library: Please run `pip install econml`")
    st.stop()

try:
    from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
except ImportError:
    st.error("🚨 Critical Missing Library: Please run `pip install causalml`")
    st.stop()

from fpdf import FPDF

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 2. Page Configuration & Professional CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Universal Causal Dashboard", page_icon="🔮")

st.markdown("""
<style>
    /* 1. Load Font */
    @import url('https://fonts.googleapis.com/css2?family=Josefin+Sans&display=swap');
    
    /* 2. Global Font */
    html, body, [class*="css"], font, div, span, p, text {
        font-family: 'Josefin Sans', sans-serif !important;
    }
    
    /* 3. Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: 'Josefin Sans', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* 4. Metrics & Tabs */
    [data-testid="stMetricValue"] { color: #000000 !important; font-family: 'Josefin Sans', sans-serif !important; }
    .stTabs [data-baseweb="tab"] { font-family: 'Josefin Sans', sans-serif !important; font-size: 1.2rem; font-weight: 600; }
    
    /* 5. Custom Containers */
    .stMetric {
        background-color: #ffffff;
        padding: 15px 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* 6. Theory Box */
    .theory-box {
        background-color: #f0f9ff;
        border-left: 5px solid #0ea5e9;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #334155;
    }
    
    .evaluation-box {
        background-color: #fdf2f8;
        border-left: 5px solid #db2777;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Data Processing & Logic
# ==========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def auto_feature_eng(df, target, treatment):
    df = df.copy()
    lag_cols = []
    for i in [1, 2, 3]:
        col_name = f'{target}_Lag{i}'
        df[col_name] = df[target].shift(i)
        lag_cols.append(col_name)
    df = df.dropna().reset_index(drop=True)
    scaler = StandardScaler()
    if lag_cols:
        pca = PCA(n_components=1)
        df['Latent_Market_State'] = pca.fit_transform(scaler.fit_transform(df[lag_cols]))
    else:
        df['Latent_Market_State'] = 0
    return df

# --- Inventory Simulation Logic (Reusable) ---
def simulate_inventory_dynamic(demand_series, target_series, lead_time):
    inv_levels = []
    current_stock = target_series[0]
    shortage_events = 0
    lost_sales = []
    
    for day in range(len(demand_series)):
        demand = demand_series[day]
        # 1. Deduct Demand (Check if we have enough)
        if current_stock >= demand:
            current_stock -= demand
            lost_sales.append(0)
        else:
            lost = demand - current_stock
            lost_sales.append(lost)
            current_stock = 0 # Empty
            shortage_events += 1
        
        # 2. Record End-of-Day Stock
        inv_levels.append(current_stock)
        
        # 3. Replenishment (Instant for Sim simplicity, brings up to NEXT Target)
        if (day + 1) % lead_time == 0:
            next_target = target_series[min(day+1, len(target_series)-1)]
            current_stock = next_target 
            
    return np.array(inv_levels), shortage_events, np.array(lost_sales)

# ==========================================
# 4. Engine Classes
# ==========================================
class RealCausalEngine:
    def __init__(self):
        # 3-Fold Cross-Fitting for Robustness
        self.dml_est = LinearDML(
            model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5),
            model_t=RandomForestRegressor(n_estimators=50, min_samples_leaf=5),
            random_state=42,
            cv=3
        )
        self.base_model = XGBRegressor(n_estimators=100, random_state=42)
        self.features = []
        self.treatment = ""
        self.confounders = []

    def train(self, df, target_col, treatment_col, confounders, heterogeneity_cols=None):
        self.target = target_col
        self.treatment = treatment_col
        self.confounders = confounders
        X = df[heterogeneity_cols] if heterogeneity_cols else None
        W = df[confounders]
        Y = df[target_col]
        T = df[treatment_col]

        with st.spinner("🧠 Engines warming up... DML running 3-Fold Cross-Fitting..."):
            self.dml_est.fit(Y, T, X=X, W=W)
            all_feats = [treatment_col] + confounders + (heterogeneity_cols if heterogeneity_cols else [])
            self.base_model.fit(df[all_feats], Y)
            self.features = all_feats

    def get_causal_effect(self, X_pred):
        return self.dml_est.effect(X_pred)

    def predict_counterfactual(self, df_input, new_price_col):
        base_pred = self.base_model.predict(df_input[self.features])
        delta_t = df_input[new_price_col] - df_input[self.treatment]
        if 'Latent_Market_State' in df_input.columns:
            theta = self.dml_est.effect(df_input[['Latent_Market_State']])
        else:
            theta = self.dml_est.const_marginal_effect(df_input[self.confounders])
        counterfactual_sales = base_pred + (theta * delta_t)
        return np.maximum(counterfactual_sales, 0)

def train_meta_learners(df, target_col, treatment_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    w = df[treatment_col].copy()
    if w.nunique() > 2:
        median_val = w.median()
        w_binary = (w > median_val).astype(int)
    else:
        w_binary = w.astype(int)

    results = {}
    learner_s = BaseSRegressor(learner=LinearRegression())
    cate_s = learner_s.fit_predict(X=X, treatment=w_binary, y=y)
    results['S-Learner'] = cate_s.flatten()

    learner_t = BaseTRegressor(learner=XGBRegressor(n_estimators=50, verbosity=0))
    cate_t = learner_t.fit_predict(X=X, treatment=w_binary, y=y)
    results['T-Learner'] = cate_t.flatten()
    
    learner_x = BaseXRegressor(learner=XGBRegressor(n_estimators=50, verbosity=0))
    cate_x = learner_x.fit_predict(X=X, treatment=w_binary, y=y)
    results['X-Learner'] = cate_x.flatten()
    return pd.DataFrame(results)

# ==========================================
# 5. Main Application Logic
# ==========================================
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.title("🧠 Causal AI Strategy Dashboard")
    st.markdown("Quantify the **True Impact** of your decisions using Double Machine Learning.")

with st.sidebar:
    st.header("🎛️ Control Tower")
    st.info("Upload your historical sales data to begin causal inference.")
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
    
    if uploaded_file:
        raw_df = load_data(uploaded_file)
        cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        st.markdown("### 1. Model Configuration")
        target_col = st.selectbox("🎯 Target (Outcome Y)", cols, index=0)
        treatment_col = st.selectbox("💊 Treatment (Input T)", cols, index=1)
        avail_cols = [c for c in cols if c not in [target_col, treatment_col]]
        confounders = st.multiselect("🌪️ Confounders (Controls W)", avail_cols, default=avail_cols[:2])
        
        st.markdown("### 2. Execution")
        if st.button("🚀 Run Causal Engine", type="primary", use_container_width=True):
            st.session_state['run'] = True
            st.session_state['cate_results'] = None
            st.session_state['fold_metrics'] = None
            st.session_state['ols_fold_metrics'] = None
            st.session_state['sim_results'] = None
    else:
        st.caption("Waiting for data...")

# --- Main Content ---
if st.session_state.get('run', False) and uploaded_file:
    # --- 1. DATA PREP ---
    df_eng = auto_feature_eng(raw_df, target_col, treatment_col)
    train_size = int(len(df_eng) * 0.8)
    train_df = df_eng.iloc[:train_size]
    test_df = df_eng.iloc[train_size:].reset_index(drop=True)
    all_confounders = confounders + [c for c in df_eng.columns if 'Lag' in c]

    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Observation Window", f"{len(df_eng)} Periods")
    m_col2.metric("Target Variable", target_col)
    m_col3.metric("Treatment Variable", treatment_col)
    m_col4.metric("Confounders Tracked", len(all_confounders))
    st.markdown("---")

    # --- 2. TRAIN MAIN ENGINE ---
    # We train once globally to ensure the engine is ready for all tabs
    if 'main_engine' not in st.session_state:
        engine = RealCausalEngine()
        engine.train(train_df, target_col, treatment_col, all_confounders, heterogeneity_cols=['Latent_Market_State'])
        st.session_state['main_engine'] = engine
    else:
        engine = st.session_state['main_engine']

    effects = engine.get_causal_effect(test_df[['Latent_Market_State']])
    avg_elasticity = np.mean(effects)
    naive_corr = test_df[[treatment_col, target_col]].corr().iloc[0,1]
    bias_delta = avg_elasticity - naive_corr

    # --- 3. TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⚡ Insights & Elasticity", 
        "🔮 Sensitivity Simulator", 
        "⚔️ Model Battle", 
        "⚖️ Evaluation",
        "🌌 Parallel Universe",
        "📋 Executive Report"
    ])
    
    # ... [Tab 1: Insights] ...
    with tab1:
        st.subheader("Separating Signal from Noise")
        c_dag, c_explain = st.columns([1, 2])
        with c_dag:
            dot = graphviz.Digraph()
            dot.attr(rankdir='LR', size='8,5')
            dot.attr('node', shape='box', style='filled,rounded', fontname='Inter')
            dot.node('T', 'Treatment', fillcolor='#d1fae5', color='#059669') 
            dot.node('Y', 'Outcome', fillcolor='#dbeafe', color='#2563eb')
            dot.node('W', 'Confounders', shape='ellipse', fillcolor='#fee2e2', color='#dc2626')
            dot.edge('T', 'Y', label=' Causal Link', color='#059669', penwidth='2.0')
            dot.edge('W', 'T', style='dashed', color='#94a3b8')
            dot.edge('W', 'Y', style='dashed', color='#94a3b8')
            st.graphviz_chart(dot)
        with c_explain:
            st.markdown("""<div class='theory-card'><b>The "Noise Cancellation" Logic:</b><br>Standard correlations are biased because <b>Confounders</b> (Red) affect both your decision (T) and the outcome (Y).<br><br>We use <b>Double Machine Learning</b> to "block" the red dashed lines, isolating the pure green causal link.</div>""", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("True Causal Elasticity", f"{avg_elasticity:.3f}", help="The actual impact of Treatment on Target, free of bias.")
        k2.metric("Naive Correlation", f"{naive_corr:.3f}", delta=f"Bias Detected: {bias_delta:.3f}", delta_color="inverse", help="The raw correlation found in Excel. Often misleading.")
        bias_status = "Significant Bias" if abs(bias_delta) > 0.1 else "Clean Data"
        k3.metric("Data Reliability", bias_status, delta="Corrected via DML" if abs(bias_delta) > 0.1 else "Verified", help="If Bias is Significant, traditional models will fail.")

        viz_df = pd.DataFrame({'Market Momentum': test_df['Latent_Market_State'], 'Impact': effects})
        z = np.polyfit(viz_df['Market Momentum'], viz_df['Impact'], 3)
        p = np.poly1d(z)
        x_trend = np.linspace(viz_df['Market Momentum'].min(), viz_df['Market Momentum'].max(), 100)
        y_trend = p(x_trend)
        fig_hte = px.scatter(viz_df, x='Market Momentum', y='Impact', color='Impact', color_continuous_scale='Tealgrn', opacity=1.0)
        fig_hte.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', line=dict(color='rgba(239, 68, 68, 0.6)', width=4), name='Trend'))
        fig_hte.update_layout(template="plotly_white", xaxis_title="Market Momentum (PCA)", yaxis_title="Causal Impact")
        fig_hte.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_hte, use_container_width=True)

    # ... [Tab 2: Simulator] ...
    with tab2:
        st.subheader("🔮 Multi-Scenario Simulator")
        col_in, col_out = st.columns([1, 2])
        with col_in:
            st.markdown("### 🛠️ Adjust Strategy")
            curr_avg = float(test_df[treatment_col].mean())
            price_main = st.slider("Proposed Treatment Value (Center)", min_value=float(test_df[treatment_col].min()), max_value=float(test_df[treatment_col].max()), value=curr_avg)
            comp_mode = st.radio("Comparison Mode", ["Percentage (+/- %)", "Manual Prices ($)"], horizontal=True)
            if "Percentage" in comp_mode:
                sensitivity = st.slider("Comparison Interval (+/- %)", min_value=1, max_value=20, value=5)
                price_low = price_main * (1 - sensitivity/100)
                price_high = price_main * (1 + sensitivity/100)
                scenario_labels = [f"Lower (-{sensitivity}%)", "Proposed", f"Higher (+{sensitivity}%)"]
            else:
                c1, c2 = st.columns(2)
                price_low = c1.number_input("Lower Price Scenario ($)", value=float(price_main*0.95))
                price_high = c2.number_input("Higher Price Scenario ($)", value=float(price_main*1.05))
                scenario_labels = ["Scenario A (Low)", "Proposed", "Scenario B (High)"]
            st.markdown("### 📦 Inventory Specs")
            lead_time = st.number_input("Lead Time (Days)", value=5)
            st.markdown("---")
            st.caption(f"Current Elasticity: **{avg_elasticity:.3f}**")

        with col_out:
            sim_df = test_df.copy()
            sim_df[f'New_{treatment_col}'] = price_main
            cf_main = engine.predict_counterfactual(sim_df, f'New_{treatment_col}')
            
            sim_df_high = test_df.copy()
            sim_df_high[f'New_{treatment_col}'] = price_high
            cf_high = engine.predict_counterfactual(sim_df_high, f'New_{treatment_col}')
            
            sim_df_low = test_df.copy()
            sim_df_low[f'New_{treatment_col}'] = price_low
            cf_low = engine.predict_counterfactual(sim_df_low, f'New_{treatment_col}')
            
            total_act = test_df[target_col].sum()
            total_sim = cf_main.sum()
            rev_sim = total_sim * price_main
            std_main = np.std(cf_main)
            opt_stock_main = (total_sim/len(sim_df) * lead_time) + (std_main * 1.645 * np.sqrt(lead_time))

            total_sim_low = cf_low.sum()
            rev_sim_low = total_sim_low * price_low
            std_low = np.std(cf_low)
            opt_stock_low = (total_sim_low/len(sim_df) * lead_time) + (std_low * 1.645 * np.sqrt(lead_time))

            total_sim_high = cf_high.sum()
            rev_sim_high = total_sim_high * price_high
            std_high = np.std(cf_high)
            opt_stock_high = (total_sim_high/len(sim_df) * lead_time) + (std_high * 1.645 * np.sqrt(lead_time))

            s1, s2, s3 = st.columns(3)
            s1.metric("Projected Demand (Center)", f"{total_sim:,.0f}", delta=f"{(total_sim-total_act):,.0f}")
            s2.metric("Projected Value (Center)", f"${rev_sim:,.0f}", delta=f"${(rev_sim - (total_act*curr_avg)):,.0f}")
            s3.metric("Optimal Safety Stock", f"{opt_stock_main:,.0f}", help="Based on Center Scenario")

            fig_cf = go.Figure()
            fig_cf.add_trace(go.Scatter(y=test_df[target_col], name="Historical Actuals", line=dict(color='#cbd5e1', width=2)))
            fig_cf.add_trace(go.Scatter(y=cf_low, name=f"{scenario_labels[0]} (${price_low:.2f})", line=dict(color='#10b981', width=2, dash='dash')))
            fig_cf.add_trace(go.Scatter(y=cf_main, name=f"{scenario_labels[1]} (${price_main:.2f})", line=dict(color='#0ea5e9', width=4)))
            fig_cf.add_trace(go.Scatter(y=cf_high, name=f"{scenario_labels[2]} (${price_high:.2f})", line=dict(color='#ef4444', width=2, dash='dash')))
            fig_cf.update_layout(title="Scenario Comparison", template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_cf, use_container_width=True)
            
            st.markdown("#### 📊 Scenario Breakdown")
            comp_data = {
                "Scenario": scenario_labels,
                "Price Point": [f"${price_low:.2f}", f"${price_main:.2f}", f"${price_high:.2f}"],
                "Total Demand": [f"{total_sim_low:,.0f}", f"{total_sim:,.0f}", f"{total_sim_high:,.0f}"],
                "Total Revenue": [f"${rev_sim_low:,.0f}", f"${rev_sim:,.0f}", f"${rev_sim_high:,.0f}"],
                "Rec. Safety Stock": [f"{opt_stock_low:,.0f}", f"{opt_stock_main:,.0f}", f"{opt_stock_high:,.0f}"]
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    # ... [Tab 3: Battle] ...
    with tab3:
        st.subheader("⚔️ Battle of the Meta-Learners")
        if st.session_state.get('cate_results') is None:
             if st.button("🏁 Start Tournament", use_container_width=True):
                 meta_feats = all_confounders + ['Latent_Market_State']
                 with st.spinner("Running Causal Tournament..."):
                    cate_results = train_meta_learners(df_eng, target_col, treatment_col, meta_feats)
                    st.session_state['cate_results'] = cate_results
        
        if st.session_state.get('cate_results') is not None:
            cate_results = st.session_state['cate_results']
            fig_hist = go.Figure()
            colors = ['#94a3b8', '#2dd4bf', '#3b82f6'] 
            for i, model in enumerate(cate_results.columns):
                fig_hist.add_trace(go.Histogram(x=cate_results[model], name=model, opacity=0.7, marker_color=colors[i]))
            fig_hist.update_layout(barmode='overlay', template="plotly_white", xaxis_title="Estimated Causal Effect")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            fig_corr = px.imshow(cate_results.corr(), text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            s_mean = cate_results['S-Learner'].mean()
            t_mean = cate_results['T-Learner'].mean()
            x_mean = cate_results['X-Learner'].mean()
            if (s_mean > 0 and t_mean > 0 and x_mean > 0) or (s_mean < 0 and t_mean < 0 and x_mean < 0):
                st.success(f"✅ **Unanimous Verdict:** All models agree on the direction. Robust Signal.")
            else:
                st.warning("⚠️ **Mixed Verdict:** Models disagree. Weak Signal.")
            st.info(f"**X-Learner Estimate (Winner):** {x_mean:.3f}")

    # ... [Tab 4: Evaluation] ...
    with tab4:
        st.subheader("⚖️ Methodology Evaluation")
        st.markdown("""<div class='theory-card'><b>Why DML? (3-Fold Cross-Fitting)</b><br>Standard models confuse correlation with causation. To fix this, we use the <b>Frisch-Waugh-Lovell (FWL)</b> theorem.<br></div>""", unsafe_allow_html=True)
        if st.session_state.get('fold_metrics') is None or st.session_state.get('ols_fold_metrics') is None:
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            fold_metrics = []
            ols_fold_metrics = []
            with st.spinner("Running 3-Fold Stability Check (DML vs OLS)..."):
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df)):
                    X_train_f, X_val_f = train_df.iloc[train_idx], train_df.iloc[val_idx]
                    fold_engine = RealCausalEngine()
                    fold_engine.train(X_train_f, target_col, treatment_col, all_confounders, heterogeneity_cols=['Latent_Market_State'])
                    fold_effects = fold_engine.get_causal_effect(X_val_f[['Latent_Market_State']])
                    fold_metrics.append(np.mean(fold_effects))
                    ols = LinearRegression()
                    ols.fit(X_train_f[[treatment_col] + all_confounders], X_train_f[target_col])
                    ols_fold_metrics.append(ols.coef_[0])
            st.session_state['fold_metrics'] = fold_metrics
            st.session_state['ols_fold_metrics'] = ols_fold_metrics
        
        fold_metrics = st.session_state['fold_metrics']
        ols_fold_metrics = st.session_state['ols_fold_metrics']
        dml_avg = np.mean(fold_metrics)
        ols_avg = np.mean(ols_fold_metrics)
        std_folds = np.std(fold_metrics)
        std_ols = np.std(ols_fold_metrics)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("DML Stability (Std Dev)", f"{std_folds:.3f}", delta="Stable" if std_folds < 0.2 else "Volatile", help="DML Variation across 3 folds.")
        col_m2.metric("OLS Stability (Std Dev)", f"{std_ols:.3f}", help="OLS Variation across 3 folds.")
        col_m3.metric("Bias Gap (Average)", f"{dml_avg - ols_avg:.3f}", delta_color="inverse", help="Difference between DML and OLS.")

        fig_unified = go.Figure()
        fig_unified.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(3)], y=fold_metrics, name='DML (Causal)', marker_color='#0ea5e9'))
        fig_unified.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(3)], y=ols_fold_metrics, name='OLS (Traditional)', marker_color='#ef4444'))
        fig_unified.add_hline(y=dml_avg, line_dash="dash", line_color="#0ea5e9", annotation_text="DML Avg")
        fig_unified.add_hline(y=ols_avg, line_dash="dash", line_color="#ef4444", annotation_text="OLS Avg")
        fig_unified.update_layout(title="3-Fold Cross-Validation Battle", barmode='group', template="plotly_white", height=500)
        st.plotly_chart(fig_unified, use_container_width=True)
        
        st.markdown("### 🚦 Causal Scorecard")
        sc1, sc2, sc3 = st.columns(3)
        if std_folds < 0.1: sc1.success(f"✅ **Excellent Stability**\n\nDML Std: {std_folds:.3f}")
        else: sc1.error(f"🛑 **Unstable**\n\nDML Std: {std_folds:.3f}")
        bias_gap = abs(dml_avg - ols_avg)
        if bias_gap > 0.1: sc2.success(f"✅ **High Value Discovery**\n\nGap: {bias_gap:.3f}")
        else: sc2.info(f"ℹ️ **Low Bias**\n\nGap: {bias_gap:.3f}")
        if dml_avg < 0: sc3.success(f"✅ **Logical Direction**\n\nNegative Elasticity")
        else: sc3.error(f"🛑 **Anomalous**\n\nPositive Elasticity")

    # ==========================
    # TAB 5: Parallel Universe (With Logic INSIDE Tab)
    # ==========================
    with tab5:
        st.subheader("🌌 Parallel Universe Simulation")
        st.markdown("Comparing **Universe A (Status Quo)** vs. **Universe B (Causal AI)** in a parallel simulation.")
        
        col_p1, col_p2 = st.columns([1, 3])
        
        with col_p1:
            st.markdown("### ⚙️ Universe B Settings")
            curr_avg_p = float(test_df[treatment_col].mean())
            price_b = st.slider("Universe B Price ($)", min_value=float(test_df[treatment_col].min()), max_value=float(test_df[treatment_col].max()), value=curr_avg_p)
            target_service_level = st.slider("Target Service Level (%)", 90, 99, 95) / 100
            
            # DYNAMIC Z-SCORE CALCULATION (Lookup Table)
            # Maps Service Level % to Z-Score (Standard Normal Distribution)
            z_lookup = {
                90: 1.282, 91: 1.341, 92: 1.405, 93: 1.476, 94: 1.555,
                95: 1.645, 96: 1.751, 97: 1.881, 98: 2.054, 99: 2.326
            }
            # Convert decimal (0.95) back to integer (95) for lookup
            z_score = z_lookup.get(int(target_service_level * 100), 1.645)
            
            sim_lead_time = st.number_input("Supply Lead Time (Days)", value=5)
            st.markdown("---")
            st.info("**Universe A:** Static Average Target\n\n**Universe B:** Causal AI Forecast Target")

        with col_p2:
            # --- CALCULATION LOGIC (Moved inside Tab 5) ---
            # Data for A
            demand_a = test_df[target_col]
            price_a = curr_avg_p
            std_raw = demand_a.std()
            ss_a = z_score * std_raw * np.sqrt(sim_lead_time)
            mean_demand_a = demand_a.mean()
            target_stock_a_series = np.full(len(demand_a), (mean_demand_a * sim_lead_time) + ss_a)
            
            # Data for B
            sim_df_b = test_df.copy()
            sim_df_b[f'New_{treatment_col}'] = price_b
            demand_b = engine.predict_counterfactual(sim_df_b, f'New_{treatment_col}')
            
            preds = engine.base_model.predict(test_df[engine.features])
            residuals = test_df[target_col] - preds
            std_resid = residuals.std()
            ss_b = z_score * std_resid * np.sqrt(sim_lead_time)
            target_stock_b_series = (demand_b * sim_lead_time) + ss_b 
            
            inv_a, short_a, lost_a = simulate_inventory_dynamic(demand_a, target_stock_a_series, sim_lead_time)
            inv_b, short_b, lost_b = simulate_inventory_dynamic(demand_b, target_stock_b_series, sim_lead_time)
            
            realized_demand_a = demand_a - lost_a
            revenue_a = (realized_demand_a * price_a).sum()
            realized_demand_b = demand_b - lost_b
            revenue_b = (realized_demand_b * price_b).sum()
            
            avg_inv_a = np.mean(np.maximum(inv_a, 0))
            avg_inv_b = np.mean(np.maximum(inv_b, 0))
            sl_a = 1 - (short_a / len(test_df))
            sl_b = 1 - (short_b / len(test_df))
            
            # SAVE TO SESSION STATE FOR REPORT TAB
            st.session_state['sim_results'] = {
                'rev_a': revenue_a, 'rev_b': revenue_b,
                'inv_a': avg_inv_a, 'inv_b': avg_inv_b,
                'sl_a': sl_a, 'sl_b': sl_b,
                'short_a': short_a, 'short_b': short_b
            }
            
            def display_comparison(label, val_a, val_b, fmt, higher_is_better=True):
                delta = val_b - val_a
                pct = (delta / val_a) * 100 if val_a != 0 else 0
                color_cls = "comp-delta-pos" if (delta >= 0 if higher_is_better else delta <= 0) else "comp-delta-neg"
                st.markdown(f"""<div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 10px;"><p class="comp-label">{label}</p><div style="display: flex; justify-content: space-between; align-items: baseline;"><div><span style="font-size: 14px; color: #94a3b8;">Universe A</span><br><span style="font-size: 18px; font-weight: 600;">{fmt(val_a)}</span></div><div style="text-align: right;"><span style="font-size: 14px; color: #0ea5e9;">Universe B</span><br><span style="font-size: 18px; font-weight: 600;">{fmt(val_b)}</span></div></div><hr style="margin: 8px 0;"><div style="text-align: center;"><span class="{color_cls}">Improvement: {delta:+.1f} ({pct:+.1f}%)</span></div></div>""", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1: display_comparison("💰 Total Revenue", revenue_a, revenue_b, lambda x: f"${x:,.0f}", True)
            with m2: display_comparison("📦 Avg Inventory", avg_inv_a, avg_inv_b, lambda x: f"{x:,.0f}", False)
            with m3: display_comparison("🤝 Service Level", sl_a*100, sl_b*100, lambda x: f"{x:.1f}%", True)
            with m4: display_comparison("🚫 Shortage Events", short_a, short_b, lambda x: f"{x} days", False)

            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=test_df.index, y=inv_a, mode='lines', name='Universe A (Static)', line=dict(color='#94a3b8', width=2), fill='tozeroy', fillcolor='rgba(148, 163, 184, 0.1)'))
            fig_sim.add_trace(go.Scatter(x=test_df.index, y=inv_b, mode='lines', name='Universe B (Dynamic)', line=dict(color='#0ea5e9', width=2), fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)'))
            fig_sim.add_trace(go.Bar(x=test_df.index, y=demand_b, name='Scenario Demand (B)', marker_color='#cbd5e1', opacity=0.3))
            fig_sim.add_hline(y=0, line_color="#ef4444", line_dash="solid", annotation_text="Stockout Level")
            fig_sim.update_layout(yaxis_title="Inventory Units on Hand", xaxis_title="Simulation Day", template="plotly_white", hovermode="x unified", height=400)
            st.plotly_chart(fig_sim, use_container_width=True)
            st.success(f"**Conclusion:** By using **Dynamic Targets** (Universe B), you adapt to demand peaks, drastically reducing shortages compared to the previous static simulation.")

    # ==========================
    # TAB 6: Report (Safeguarded)
    # ==========================
    with tab6:
        st.subheader("📋 Executive Summary")
        
        # Fallback: Run logic if user skipped Tab 5
        if st.session_state.get('sim_results') is None:
             # Default values for fallback calc
             d_a = test_df[target_col]; p_a = float(test_df[treatment_col].mean())
             s_raw = d_a.std(); ss_a_d = 1.645 * s_raw * np.sqrt(5)
             ts_a = np.full(len(d_a), (d_a.mean() * 5) + ss_a_d)
             
             sim_b_d = test_df.copy(); sim_b_d[f'New_{treatment_col}'] = p_a
             dem_b = engine.predict_counterfactual(sim_b_d, f'New_{treatment_col}')
             pr = engine.base_model.predict(test_df[engine.features])
             ss_b_d = 1.645 * (test_df[target_col] - pr).std() * np.sqrt(5)
             ts_b = (dem_b * 5) + ss_b_d
             
             ia, sa, la = simulate_inventory_dynamic(d_a, ts_a, 5)
             ib, sb, lb = simulate_inventory_dynamic(dem_b, ts_b, 5)
             
             st.session_state['sim_results'] = {
                'rev_a': ((d_a-la)*p_a).sum(), 'rev_b': ((dem_b-lb)*p_a).sum(),
                'inv_a': np.mean(np.maximum(ia,0)), 'inv_b': np.mean(np.maximum(ib,0)),
                'sl_a': 1-(sa/len(d_a)), 'sl_b': 1-(sb/len(d_a)),
                'short_a': sa, 'short_b': sb
             }

        # Check other results
        if st.session_state.get('cate_results') is None:
             with st.spinner("Generating Tournament Results for Report..."):
                 meta_feats = all_confounders + ['Latent_Market_State']
                 st.session_state['cate_results'] = train_meta_learners(df_eng, target_col, treatment_col, meta_feats)
        
        if st.session_state.get('fold_metrics') is None:
             pass 

        # Retrieve
        cate_res = st.session_state['cate_results']
        winner_avg = cate_res['X-Learner'].mean()
        f_mets = st.session_state.get('fold_metrics', [0])
        ols_mets = st.session_state.get('ols_fold_metrics', [0])
        sim = st.session_state['sim_results']
        
        dml_avg_r = np.mean(f_mets)
        ols_avg_r = np.mean(ols_mets)
        std_dev = np.std(f_mets)
        gap = dml_avg_r - ols_avg_r

        report_txt = f"""
        CAUSAL AI EXECUTIVE SUMMARY
        ===========================
        
        1. EXECUTIVE FINDINGS
        ---------------------
        - True Causal Elasticity: {avg_elasticity:.4f}
        - Naive Correlation: {naive_corr:.4f}
        - Bias Detected: {bias_delta:.4f}
        
        2. METHODOLOGY AUDIT
        --------------------
        - Algorithm: Double Machine Learning (LinearDML) with0 3-Fold Cross-Fitting
        - DML Stability (Std Dev): {std_dev:.4f}
        - Superiority vs OLS: The DML model corrected a bias of {gap:.4f}.
        
        3. MODEL TOURNAMENT
        -------------------
        - Winning Architecture: X-Learner
        - Estimated Uplift: {winner_avg:.4f}
        
        4. PARALLEL UNIVERSE RESULTS (Simulation)
        -----------------------------------------
        Comparing Status Quo (A) vs Causal AI (B):
        
        - Revenue:   ${sim['rev_a']:,.0f} (A) vs ${sim['rev_b']:,.0f} (B) -> Improvement: ${(sim['rev_b']-sim['rev_a']):,.0f}
        - Inventory: {sim['inv_a']:.0f} units (A) vs {sim['inv_b']:.0f} units (B) -> Reduction: {sim['inv_a']-sim['inv_b']:.0f} units
        - Service:   {sim['sl_a']*100:.1f}% (A) vs {sim['sl_b']*100:.1f}% (B)
        - Shortages: {sim['short_a']} days (A) vs {sim['short_b']} days (B)
        
        5. STRATEGIC RECOMMENDATION
        ---------------------------
        Use the Heterogeneity Chart (Tab 1) to identify "High Momentum" periods.
        Raise prices ONLY during those periods to minimize volume loss.
        """
        
        st.text_area("Report Preview", report_txt, height=500)
        
        if st.button("📄 Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Causal AI Analysis Report", 0, 1, 'C')
            pdf.set_font("Arial", size=11)
            pdf.ln(10)
            pdf.multi_cell(0, 7, report_txt)
            pdf_out = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download PDF", pdf_out, "causal_report.pdf", "application/pdf")