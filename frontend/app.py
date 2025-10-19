import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from backtesting import backtesting_tab
from prediction import prediction_tab

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Crypto Dashboard", layout="wide", page_icon="📊")
st.title("📊 Crypto Analytics Dashboard")

# Tabs
tabs = st.tabs(["Analytics", "Backtesting", "Forecasting"])

with tabs[0]:
    st.header("Coin Analytics")

    # Coin selection
    coin = st.selectbox("Select Coin", ["BTCUSDT", "ETHUSDT"], index=0)

    # Candle size selection
    timeframe_options = ["1h", "12h", "1d", "1w"]  # removed 1m
    candle_size = st.radio("Candle Size", timeframe_options, horizontal=True, index=0)

    st.markdown("---")

    # 1️⃣ Fetch chart data (aggregated)
    df_chart = pd.DataFrame()
    try:
        res_chart = requests.get(
            f"{BACKEND_URL}/analytics?coin={coin}&candle_size={candle_size}", timeout=30
        )
        data_chart = res_chart.json().get("data", [])
        if data_chart:
            df_chart = pd.DataFrame(data_chart)
            df_chart["open_time"] = pd.to_datetime(df_chart["open_time"])
    except Exception as e:
        st.error(f"Failed to fetch chart data: {e}")

    # 2️⃣ Fetch 1h data for percentage change calculation
    df_1h = pd.DataFrame()
    try:
        res_1h = requests.get(
            f"{BACKEND_URL}/analytics?coin={coin}&candle_size=1h", timeout=30
        )
        data_1h = res_1h.json().get("data", [])
        if data_1h:
            df_1h = pd.DataFrame(data_1h)
            df_1h["open_time"] = pd.to_datetime(df_1h["open_time"])
    except Exception as e:
        st.error(f"Failed to fetch 1h data for change calculation: {e}")

    # Summary metrics (percentage changes)
    if not df_1h.empty:
        def calc_change(df, hours):
            if len(df) < hours:
                return None
            start_price = df["close"].iloc[-hours]
            return (df["close"].iloc[-1] - start_price) / start_price * 100

        daily_change = calc_change(df_1h, 24)
        weekly_change = calc_change(df_1h, 24 * 7)

        def color_change(val):
            if val is None:
                return "N/A"
            color = "green" if val >= 0 else "red"
            return f'<div style="display:inline-block;padding:3px 8px;border-radius:5px;background-color:{color};color:white;text-align:center">{val:.2f}%</div>'

        col1, col2 = st.columns(2)
        col1.markdown("**1D Change**", unsafe_allow_html=True)
        col1.markdown(color_change(daily_change), unsafe_allow_html=True)
        col2.markdown("**1W Change**", unsafe_allow_html=True)
        col2.markdown(color_change(weekly_change), unsafe_allow_html=True)

    # Candlestick chart
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart["open_time"],
            open=df_chart["open"],
            high=df_chart["high"],
            low=df_chart["low"],
            close=df_chart["close"],
            name="Candles"
        )])

        # Overlay technical indicators if they exist
        indicator_cols = [
            ("sma_20", "SMA 20"),
            ("ema_20", "EMA 20"),
            ("bb_upper", "BB Upper"),
            ("bb_middle", "BB Middle"),
            ("bb_lower", "BB Lower")
        ]
        for col, name in indicator_cols:
            if col in df_chart.columns:
                dash = "dot" if "bb" in col else "solid"
                fig.add_trace(go.Scatter(
                    x=df_chart["open_time"], y=df_chart[col],
                    mode="lines", name=name, line=dict(width=1.5, dash=dash)
                ))

        # Focus initial view on recent candles
        recent_candles = 100
        if len(df_chart) > recent_candles:
            x_range = [df_chart["open_time"].iloc[-recent_candles], df_chart["open_time"].iloc[-1]]
        else:
            x_range = [df_chart["open_time"].iloc[0], df_chart["open_time"].iloc[-1]]

        fig.update_layout(
            title=f"{coin} - Candlestick & Indicators",
            xaxis_rangeslider_visible=False,
            xaxis=dict(range=x_range),
            template="plotly_dark",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=20, r=20, t=50, b=20),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Raw Analytics Data"):
            st.dataframe(df_chart.tail(20))
    else:
        st.warning("No analytics data available for this selection.")



# -------------------------------
# 2️⃣ Backtesting Tab
# -------------------------------
with tabs[1]:
    backtesting_tab()
# -------------------------------
# 3️⃣ Forecasting Tab
# -------------------------------
with tabs[2]:
    prediction_tab()
