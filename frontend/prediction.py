import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

BACKEND_URL = "http://127.0.0.1:8000"

def prediction_tab():
    st.header("🤖 Price Prediction")

    # --- Coin selection ---
    coin = st.selectbox(
        "Select Coin",
        ["BTCUSDT", "ETHUSDT"],
        index=0,
        key="prediction_coin"
    )

    st.markdown("---")

    run_prediction_btn = st.button(
        "🚀 Predict Next 6 Hours",
        key="run_prediction_btn"
    )

    if run_prediction_btn:
        try:
            res = requests.get(
                f"{BACKEND_URL}/predict",
                params={"coin": coin},
                timeout=30
            )
            data = res.json().get("data", {})

            if "error" in data:
                st.error(f"Prediction failed: {data['error']}")
                return

            last_prices = data.get("last_prices", [])
            last_timestamps = data.get("last_timestamps", [])
            predicted_prices = data.get("predicted_prices", [])
            predicted_timestamps = data.get("predicted_timestamps", [])

            if not last_prices or not predicted_prices:
                st.warning("No price data returned.")
                return

            # --- Prepare DataFrames for plotting ---
            last_df = pd.DataFrame({
                "time": pd.to_datetime(last_timestamps),
                "close": last_prices
            })

            pred_df = pd.DataFrame({
                "time": pd.to_datetime(predicted_timestamps),
                "predicted_close": predicted_prices
            })

            # --- Plot ---
            fig = go.Figure()

            # Last 48h real prices
            fig.add_trace(go.Scatter(
                x=last_df["time"],
                y=last_df["close"],
                mode="lines",
                name="Real Price",
                line=dict(color="cyan", width=2)
            ))

            # Predicted next 6h as dots
            fig.add_trace(go.Scatter(
                x=pred_df["time"],
                y=pred_df["predicted_close"],
                mode="markers+lines",
                name="Predicted Price",
                marker=dict(color="magenta", size=10, symbol="circle")
            ))

            fig.update_layout(
                template="plotly_dark",
                title=f"{coin} - Last 48h & Next 6h Prediction",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to fetch prediction: {e}")
