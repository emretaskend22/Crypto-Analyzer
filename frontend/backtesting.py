import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import datetime
import requests

BACKEND_URL = "http://127.0.0.1:8000"

def backtesting_tab():
    st.header("📉 Strategy Backtesting")

    # --- Selection controls ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        coin = st.selectbox("Select Coin", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=0)
    with col2:
        strategy = st.selectbox(
            "Select Strategy",
            ["Buy & Hold", "SMA Crossover (20/50)", "RSI Strategy", "EMA Crossover",
             "MACD Strategy", "Bollinger Bands"],
            index=1
        )
    with col3:
        timeframe = st.selectbox("Candle Size", ["1h", "12h", "1d"], index=0)

    col4, col5 = st.columns([1, 1])
    with col4:
        start_date = st.date_input(
            "Start Date",
            datetime.date(2023, 1, 1),
            min_value=datetime.date(2023, 1, 1),
            max_value=datetime.date.today()
        )
    with col5:
        end_date = st.date_input(
            "End Date",
            datetime.date.today(),
            min_value=datetime.date(2023, 1, 1),
            max_value=datetime.date.today()
        )

    st.markdown("---")
    run_backtest = st.button("🚀 Run Backtest", use_container_width=True)

    if run_backtest:
        st.subheader(f"Results for **{coin}** with **{strategy}** ({timeframe} candles)")

        # --- Fetch backtest from backend ---
        try:
            res = requests.post(
                f"{BACKEND_URL}/backtest",
                params={
                    "coin": coin,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "strategy": strategy,
                    "initial_balance": 10000
                },
                timeout=60
            )
            data = res.json()

            # --- Debug prints (minimal) ---
            st.text(f"HTTP Status Code: {res.status_code}")
            st.text(f"Response Keys: {list(data.keys())}")

            if "error" in data.get("data", {}):
                st.error(f"Backtest failed: {data['data']['error']}")
                return

            ohlcv = data.get("data", {}).get("ohlcv", [])
            if not ohlcv:
                st.warning("No OHLCV data returned from backend.")
                return

            df = pd.DataFrame(ohlcv)
            df["open_time"] = pd.to_datetime(df["open_time"])

        except Exception as e:
            st.error(f"Failed to fetch backtest data: {e}")
            return

        # --- Performance Metrics ---
        st.markdown("### 📊 Performance Metrics")
        equity = pd.Series([row.get("equity", 10000) for row in ohlcv])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Total Return", f"{data.get('data', {}).get('total_return', 0):.2f}%")
        col2.metric("📊 Win Rate", f"{data.get('data', {}).get('win_rate', 0):.2f}%")
        col3.metric("📉 Max Drawdown", f"{data.get('data', {}).get('max_drawdown', 0):.2f}%")
        col4.metric("⚖️ Sharpe Ratio", f"{data.get('data', {}).get('sharpe_ratio', 0):.2f}")

        st.markdown("---")

        # --- Candlestick Chart ---
        st.subheader("📈 Trade Chart")
        fig_candles = go.Figure(data=[go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles"
        )])

        # Add Buy/Sell markers if available
        if "signal" in df.columns:
            buys = df[df["signal"] == "BUY"]
            sells = df[df["signal"] == "SELL"]

            fig_candles.add_trace(go.Scatter(
                x=buys["open_time"], y=buys["close"],
                mode="markers", marker_symbol="triangle-up", marker_color="green", marker_size=10,
                name="Buy Signal"
            ))
            fig_candles.add_trace(go.Scatter(
                x=sells["open_time"], y=sells["close"],
                mode="markers", marker_symbol="triangle-down", marker_color="red", marker_size=10,
                name="Sell Signal"
            ))

        fig_candles.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_candles, use_container_width=True)

        # --- Equity Curve ---
        st.subheader("📊 Portfolio Value Over Time")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df["open_time"], y=equity,
            mode="lines", line=dict(color="cyan", width=2),
            name="Equity"
        ))
        fig_equity.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)"
        )
        st.plotly_chart(fig_equity, use_container_width=True)
