import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="RF Pairs Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# 2. Styling & Title
st.title("🌲 Random Forest: Pairs Trading Strategy")
st.markdown("""
This dashboard identifies **Mean Reversion** opportunities using a Rolling Random Forest model. 
When the Actual Spread moves significantly away from the 'Fair Value' prediction, a signal is generated.
""")

# 3. Load Data Function
@st.cache_data
def load_data():
    # Make sure this filename matches exactly what you uploaded to GitHub
    df = pd.read_csv('rf_final_results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Fill any NaNs in rolling_std to avoid plotting issues
    df['rolling_std'] = df['rolling_std'].fillna(method='bfill')
    return df

# 4. Main Application Logic
try:
    df = load_data()
    pairs = df['pair'].unique()

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Strategy Settings")
    selected_pair = st.sidebar.selectbox("Select Trading Pair", pairs)
    
    # Allow user to adjust the Sensitivity (Standard Deviations)
    stdev_selector = st.sidebar.slider("Signal Sensitivity (Std Dev)", 1.0, 3.0, 2.0, 0.5)

    # Filter Data for the selected pair
    pair_df = df[df['pair'] == selected_pair].sort_values('date').copy()
    
    # Recalculate bands based on sidebar slider
    pair_df['upper'] = pair_df['prediction'] + (stdev_selector * pair_df['rolling_std'])
    pair_df['lower'] = pair_df['prediction'] - (stdev_selector * pair_df['rolling_std'])

    # --- TOP METRICS ---
    mae = np.mean(np.abs(pair_df['target'] - pair_df['prediction']))
    active_signals = pair_df['signal'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Pair", selected_pair)
    col2.metric("Model MAE", f"{mae:.4f}")
    
    if active_signals == 1:
        col3.success("Current Signal: BUY (Long)")
    elif active_signals == -1:
        col3.error("Current Signal: SELL (Short)")
    else:
        col3.info("Current Signal: Neutral")

    # --- THE TRADING CHART ---
    st.subheader(f"Strategy Visualizer: {selected_pair}")
    
    fig = go.Figure()

    # Upper and Lower Bands (The "Cloud")
    fig.add_trace(go.Scatter(
        x=pair_df['date'], y=pair_df['upper'],
        line=dict(width=0),
        showlegend=False,
        name='Upper Band'
    ))
    fig.add_trace(go.Scatter(
        x=pair_df['date'], y=pair_df['lower'],
        fill='tonexty', # This creates the shaded area
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(width=0),
        name='Confidence Zone',
        showlegend=True
    ))

    # Actual Spread (The Target)
    fig.add_trace(go.Scatter(
        x=pair_df['date'], y=pair_df['target'],
        name="Actual Spread",
        line=dict(color='#1f77b4', width=2)
    ))

    # Predicted Fair Value
    fig.add_trace(go.Scatter(
        x=pair_df['date'], y=pair_df['prediction'],
        name="RF Fair Value",
        line=dict(color='white', width=1, dash='dot')
    ))

    # Add Trading Markers (Triangles)
    buys = pair_df[pair_df['signal'] == 1]
    sells = pair_df[pair_df['signal'] == -1]

    fig.add_trace(go.Scatter(
        x=buys['date'], y=buys['target'],
        mode='markers',
        name='BUY Trigger',
        marker=dict(color='green', size=12, symbol='triangle-up')
    ))

    fig.add_trace(go.Scatter(
        x=sells['date'], y=sells['target'],
        mode='markers',
        name='SELL Trigger',
        marker=dict(color='red', size=12, symbol='triangle-down')
    ))

    fig.update_layout(
        template="plotly_dark", # Looks more professional for finance
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Log Spread Value",
        height=600,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("Recent Activity Log")
    # Show last 10 days, formatted for readability
    display_df = pair_df[['date', 'target', 'prediction', 'signal']].tail(10).copy()
    display_df['signal'] = display_df['signal'].replace({1: '🟩 BUY', -1: '🟥 SELL', 0: '⚪ Neutral'})
    st.dataframe(display_df, use_container_width=True)

except Exception as e:
    st.error(f"Waiting for data... Please ensure 'rf_final_results.csv' is uploaded to GitHub. Error: {e}")
