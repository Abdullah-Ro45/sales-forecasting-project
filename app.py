import subprocess
import sys

# Force-install missing libraries directly
try:
    import plotly.express as px
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "scikit-learn", "pandas", "numpy"])
    import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# --- The rest of your app.py code stays exactly the same from here down ---
st.set_page_config(page_title="Market AI Predictor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Set page configuration to wide mode
st.set_page_config(page_title="Sales Forecast Pro", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
# This injects custom styles to make the app look like a modern UI dashboard
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #1E1E2F;
        border: 1px solid #33334d;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        border-left: 5px solid #00E6FF;
    }
    /* Header Gradient Text */
    h1 {
        background: -webkit-linear-gradient(#00E6FF, #0073e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Custom Glow Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00E6FF 0%, #0073e6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0, 230, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_sales_data.csv', parse_dates=['Date'])
    results = pd.read_csv('actual_vs_predicted.csv', parse_dates=['Date'])
    return df, results

@st.cache_resource
def load_model():
    with open('sales_model.pkl', 'rb') as f:
        return pickle.load(f)

df, results_df = load_data()
model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=100)
    st.title("⚙️ Engine Specs")
    st.info("Predicting future revenue using a Machine Learning Random Forest algorithm.")
    st.markdown("### 🧹 Data Pipeline Log")
    st.markdown("- ✅ Datetime parsed\n- ✅ Null values purged\n- ✅ Duplicates dropped\n- ✅ Temporal features engineered")

# --- MAIN HEADER ---
st.title("🚀 AI-Powered Sales Forecasting")
st.markdown("Analyze historical trends and predict future revenue with machine learning.")
st.write("---")

# Create Tabs
tab1, tab2 = st.tabs(["📊 Data Intelligence", "🔮 AI Forecasting Engine"])

# --- TAB 1: EDA ---
with tab1:
    # KPI Metrics Row
    total_sales = df['Sales'].sum()
    avg_daily = df['Sales'].mean()
    monthly_totals = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
    high_idx = monthly_totals['Sales'].idxmax()
    low_idx = monthly_totals['Sales'].idxmin()
    best_month = f"{int(monthly_totals.loc[high_idx, 'Month'])}/{int(monthly_totals.loc[high_idx, 'Year'])}"
    worst_month = f"{int(monthly_totals.loc[low_idx, 'Month'])}/{int(monthly_totals.loc[low_idx, 'Year'])}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Volume", f"{total_sales:,.0f}")
    col2.metric("Daily Average", f"{avg_daily:,.0f}")
    col3.metric("🔥 Peak Month", best_month)
    col4.metric("🧊 Lowest Month", worst_month)
    
    st.write("---")

    # Interactive Plotly Line Chart (Curved Lines)
    st.subheader("📈 Historical Sales Trajectory")
    fig1 = px.line(df, x='Date', y='Sales', template='plotly_dark', line_shape='spline')
    fig1.update_traces(line_color='#00E6FF', line_width=2)
    fig1.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig1, use_container_width=True)
    
    # Interactive Plotly Bar Chart (Gradient Color)
    st.subheader("📅 Seasonal Pattern Analysis")
    monthly_sales = df.groupby('Month')['Sales'].mean().reset_index()
    fig2 = px.bar(monthly_sales, x='Month', y='Sales', template='plotly_dark', color='Sales', color_continuous_scale='Blues')
    fig2.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 2: FORECASTING ---
with tab2:
    st.subheader("🎯 Model Performance: Actual vs Predicted")
    
    # Error metrics
    mae = (results_df['Actual'] - results_df['Predicted']).abs().mean()
    rmse = np.sqrt(((results_df['Actual'] - results_df['Predicted']) ** 2).mean())
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    m_col2.metric("Root Mean Square Error (RMSE)", f"{rmse:.2f}")
    
    # Plotly Actual vs Predicted Chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Actual'], mode='lines', name='Actual', line=dict(color='#00E6FF', width=2)))
    fig3.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Predicted'], mode='lines', name='Predicted', line=dict(color='#FF4B4B', width=2, dash='dot')))
    fig3.update_layout(template='plotly_dark', height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    st.write("---")
    
    st.subheader("🔮 Run Neural Prediction")
    st.markdown("Select a future date to simulate the model's forecasting output.")
    
    selected_date = st.date_input("Target Date", value=date(2024, 1, 1))

    if st.button("🚀 Initialize Forecast Sequence", use_container_width=True):
        pred_year = selected_date.year
        pred_month = selected_date.month
        pred_day = selected_date.day
        pred_dow = selected_date.weekday()
        pred_weekend = 1 if pred_dow >= 5 else 0
        
        input_features = pd.DataFrame([[pred_year, pred_month, pred_day, pred_dow, pred_weekend]], 
                                      columns=['Year', 'Month', 'Day', 'DayOfWeek', 'Is_Weekend'])
        
        prediction = model.predict(input_features)[0]
        st.success(f"### 🎯 Predicted Sales Volume for {selected_date.strftime('%B %d, %Y')}: **{prediction:.2f} units**")
        st.balloons()
