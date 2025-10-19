# backend/ml/prediction_manager.py

import os
import pandas as pd
from ml.model_manager import get_model
from ml.preprocessor import LSTMPreprocessor, fetch_and_preprocess

HORIZONS = [1, 2, 3, 4, 5, 6]  # next 6 hours
N_PAST = 24  # last 24 hours used for prediction
MODEL_DIR = "models"

def predict_next_hours(coin: str):
    print("=" * 80)
    print(f"[DEBUG] ⏳ Starting prediction for {coin}")

    try:
        # 1️⃣ Fetch data
        total_hours = max(48 + N_PAST, N_PAST)
        print(f"[DEBUG] Fetching {total_hours}h of data for {coin}")
        df = fetch_and_preprocess(coin, n_hours=total_hours)
        print(f"[DEBUG] ✅ Data fetched successfully. Shape: {df.shape}")
        print(df.tail(5))

        if df.empty or "close" not in df.columns:
            print(f"[ERROR] ❌ No OHLCV data or 'close' column missing for {coin}")
            return {"error": f"No OHLCV data available for {coin}"}

        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)

        # 2️⃣ Last 48h prices + timestamps
        last_48h_df = df.tail(48)
        last_48h = last_48h_df["close"].tolist()
        last_48h_times = last_48h_df["open_time"].tolist()
        print(f"[DEBUG] Last 48h closes ({len(last_48h)}): {last_48h[-5:]} ...")
        print(f"[DEBUG] Last 48h timestamps: {last_48h_times[-5:]} ...")

        # 3️⃣ Load scaler
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{coin}_1h.save")
        print(f"[DEBUG] Looking for scaler at {scaler_path}")
        if not os.path.exists(scaler_path):
            print(f"[ERROR] ❌ Scaler file not found at: {scaler_path}")
            return {"error": f"Scaler file not found for {coin}"}

        preprocessor = LSTMPreprocessor(scaler_path=scaler_path, n_past=N_PAST)
        print(f"[DEBUG] ✅ Scaler loaded successfully")

        # 4️⃣ Prepare sequence
        try:
            seq = preprocessor.prepare_sequence(df)
            print(f"[DEBUG] ✅ LSTM input sequence prepared. Shape: {seq.shape}")
        except Exception as e:
            print(f"[ERROR] ❌ Failed during sequence preparation: {e}")
            return {"error": f"Failed during sequence preparation: {e}"}

        if seq is None or len(seq) == 0:
            print(f"[ERROR] ❌ Sequence is empty after preprocessing")
            return {"error": "Sequence preparation returned empty array"}

        # 5️⃣ Load model
        model_path = os.path.join(MODEL_DIR, f"lstm_{coin}_1_6h.h5")
        print(f"[DEBUG] Loading model from {model_path}")
        model = get_model(model_path)
        if model is None:
            print(f"[ERROR] ❌ Model not found or failed to load for {coin}")
            return {"error": f"No LSTM model loaded for {coin}"}

        print(f"[DEBUG] ✅ Model loaded successfully")

        # 6️⃣ Predict
        try:
            pred_scaled = model.predict(seq)
            print(f"[DEBUG] ✅ Model prediction done. pred_scaled shape: {pred_scaled.shape}")
            print(f"[DEBUG] First 3 scaled preds: {pred_scaled[0] if len(pred_scaled) else 'empty'}")
        except Exception as e:
            print(f"[ERROR] ❌ Model prediction failed: {e}")
            return {"error": f"Model prediction failed: {e}"}

        # 7️⃣ Inverse transform predictions
        try:
            pred_prices = preprocessor.inverse_transform(pred_scaled)
            print(f"[DEBUG] ✅ Predictions inverse-transformed. Predicted prices: {pred_prices}")
        except Exception as e:
            print(f"[ERROR] ❌ Inverse transform failed: {e}")
            return {"error": f"Inverse transform failed: {e}"}

        # 8️⃣ Prediction timestamps aligned with last_48h
        try:
            last_time = last_48h_times[-1]  # last timestamp in historical data
            pred_timestamps = pd.date_range(
                last_time + pd.Timedelta(hours=1),
                periods=len(HORIZONS),
                freq="H"
            ).tolist()
            print(f"[DEBUG] ✅ Prediction timestamps generated. Last time: {last_time}")
            print(f"[DEBUG] Pred timestamps: {pred_timestamps}")
        except Exception as e:
            print(f"[ERROR] ❌ Timestamp generation failed: {e}")
            return {"error": f"Timestamp generation failed: {e}"}

        print(f"[DEBUG] 🎯 Prediction pipeline completed successfully for {coin}")
        print("=" * 80)

        return {
            "last_prices": last_48h,
            "last_timestamps": last_48h_times,
            "predicted_prices": pred_prices,
            "predicted_timestamps": pred_timestamps
        }

    except Exception as e:
        print(f"[FATAL] ❌ Unexpected crash in predict_next_hours: {e}")
        return {"error": f"Unexpected error: {e}"}
