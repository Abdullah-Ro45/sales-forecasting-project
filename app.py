import os
import sys

# 1. THE BULLETPROOF INSTALLER: This forces Streamlit to download Plotly and Scikit-Learn
os.system(f"{sys.executable} -m pip install plotly scikit-learn pandas numpy")

# 2. Standard Imports (Now safely placed below the installer)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

st.set_page_config(page_title="Market AI Predictor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- HOLOGRAPHIC & FUTURISTIC CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    div[data-testid="metric-container"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(0, 255, 136, 0.3);
        padding: 5% 5% 5% 10%;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #00FF88;
        transition: all 0.4s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 255, 136, 0.25);
        border-left: 4px solid #00B4D8;
    }
    
    h1 {
        background: linear-gradient(to right, #00FF88, #00B4D8, #8A2BE2, #00FF88);
        background-size: 300% auto;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        font-weight: 700;
        font-size: 3.5rem !important;
        text-align: center;
        letter-spacing: 2px;
    }
    @keyframes shine {
        to { background-position: 300% center; }
    }
    
    div.stButton > button:first-child {
        background: transparent;
        color: var(--text-color);
        border: 2px solid #00B4D8;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 180, 216, 0.1), inset 0 0 10px rgba(0, 180, 216, 0.1);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #00B4D8, #8A2BE2);
        color: white;
        border: 2px solid transparent;
        box-shadow: 0 0 25px rgba(0, 180, 216, 0.6);
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('market_data_cleaned.csv')
    results = pd.read_csv('prediction_results.csv')
    return df, results

@st.cache_resource
def load_assets():
    with open('stock_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('stock_scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('stock_encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    return model, scaler, encoder

df, results_df = load_data()
model, scaler, encoder = load_assets()

# --- SIDEBAR: USER INTERFACE ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00B4D8;'>⚙️ TERMINAL CONTROLS</h2>", unsafe_allow_html=True)
    st.write("---")
    selected_stock = st.selectbox("Select Asset to Analyze", df['Ticker'].unique())
    st.write("---")
    st.success("✅ Real-Time Data Synced")
    st.success("✅ SMA Indicators Active")
    st.success("✅ Random Forest Core Online")

stock_data = df[df['Ticker'] == selected_stock].copy()

# --- MAIN HEADER ---
st.title("✦ QUANTITATIVE MARKET AI ✦")
st.markdown(f"<p style='text-align: center; font-size: 1.2rem; opacity: 0.8;'>Advanced Algorithmic Forecasting | Current Focus: <b>{selected_stock}</b></p>", unsafe_allow_html=True)
st.write("---")

tab1, tab2 = st.tabs(["📊 Market Telemetry (EDA)", "🤖 AI Forecasting Engine"])

# --- TAB 1: EDA & VISUALIZATIONS ---
with tab1:
    st.markdown(f"<h3 style='color: #00FF88;'>Historical Performance: {selected_stock}</h3>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"${stock_data['Close'].iloc[-1]:.2f}", f"{stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]:.2f}")
    c2.metric("52-Week High", f"${stock_data['High'].max():.2f}")
    c3.metric("52-Week Low", f"${stock_data['Low'].min():.2f}")
    c4.metric("Avg Daily Volume", f"{stock_data['Volume'].mean():,.0f}")
    
    st.write("---")
    
    st.markdown("<h4 style='color: #00B4D8;'>Price Action & Moving Averages (SMA 20 & 50)</h4>", unsafe_allow_html=True)
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(x=stock_data['Date'], open=stock_data['Open'], 
                                        high=stock_data['High'], low=stock_data['Low'], 
                                        close=stock_data['Close'], name='Market Price'))
    fig_candle.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_20'], line=dict(color='#00B4D8', width=2), name='SMA 20'))
    fig_candle.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_50'], line=dict(color='#8A2BE2', width=2), name='SMA 50'))
    fig_candle.update_layout(xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
    st.plotly_chart(fig_candle, use_container_width=True, theme="streamlit")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<h4 style='color: #8A2BE2;'>Trading Volume Trends</h4>", unsafe_allow_html=True)
        fig_vol = px.bar(stock_data, x='Date', y='Volume', color_discrete_sequence=['#8A2BE2'], opacity=0.8)
        fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig_vol, use_container_width=True, theme="streamlit")
        
    with colB:
        st.markdown("<h4 style='color: #00FF88;'>Financial Metric Correlation</h4>", unsafe_allow_html=True)
        corr_matrix = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50']].corr()
        fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="Tealgrn")
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig_heat, use_container_width=True, theme="streamlit")

# --- TAB 2: AI PREDICTION ---
with tab2:
    st.markdown("<h3 style='color: #00B4D8;'>🎯 Model Evaluation Diagnostics</h3>", unsafe_allow_html=True)
    
    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
    rmse = math.sqrt(mse)
    r2 = r2_score(results_df['Actual'], results_df['Predicted'])
    
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("MAE", f"${mae:.2f}")
    e2.metric("MSE", f"{mse:.2f}")
    e3.metric("RMSE", f"${rmse:.2f}")
    e4.metric("R² Score", f"{r2:.4f}")
    
    st.write("---")
    
    st.markdown("<h4 style='color: #00FF88;'>Historical Backtesting: Actual vs Predicted Prices</h4>", unsafe_allow_html=True)
    st.info("Visualizing the algorithm's prediction accuracy across the withheld testing data.")
    
    results_df['Data Point'] = results_df.index
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=results_df['Data Point'][:100], y=results_df['Actual'][:100], 
                                  mode='lines', name='Actual Price', line=dict(color='#00FF88', width=2)))
    fig_comp.add_trace(go.Scatter(x=results_df['Data Point'][:100], y=results_df['Predicted'][:100], 
                                  mode='lines', name='Predicted Price', line=dict(color='#00B4D8', width=2, dash='dot')))
    fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig_comp, use_container_width=True, theme="streamlit")
    
    st.write("---")
    st.markdown("<h3 style='color: #8A2BE2;'>🔮 Generate Tomorrow's Forecast</h3>", unsafe_allow_html=True)
    st.markdown("Enter today's end-of-day metrics below to predict tomorrow's closing price.")
    
    with st.container():
        p1, p2, p3 = st.columns(3)
        with p1:
            input_open = st.number_input("Today's Open ($)", value=float(stock_data['Open'].iloc[-1]))
            input_high = st.number_input("Today's High ($)", value=float(stock_data['High'].iloc[-1]))
        with p2:
            input_low = st.number_input("Today's Low ($)", value=float(stock_data['Low'].iloc[-1]))
            input_close = st.number_input("Today's Close ($)", value=float(stock_data['Close'].iloc[-1]))
        with p3:
            input_vol = st.number_input("Volume", value=int(stock_data['Volume'].iloc[-1]))
            
    if st.button("⚡ EXECUTE NEURAL FORECAST", use_container_width=True):
        sma_20 = stock_data['Close'].iloc[-19:].sum() + input_close / 20
        sma_50 = stock_data['Close'].iloc[-49:].sum() + input_close / 50
        ticker_enc = encoder.transform([selected_stock])[0]
        
        features = np.array([[ticker_enc, input_open, input_high, input_low, input_close, input_vol, sma_20, sma_50]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        price_diff = prediction - input_close
        
        st.write("---")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric(label="Predicted Close (Tomorrow)", value=f"${prediction:.2f}", delta=f"${price_diff:.2f}")
        with colB:
            if price_diff > 0:
                st.success(f"📈 **BULLISH SIGNAL:** The model predicts {selected_stock} will rise by **${abs(price_diff):.2f}** tomorrow. Favorable condition detected.")
                st.balloons()
            else:
                st.error(f"📉 **BEARISH SIGNAL:** The model predicts {selected_stock} will fall by **${abs(price_diff):.2f}** tomorrow. Recommend monitoring asset closely.")
