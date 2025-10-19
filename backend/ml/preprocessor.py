import pandas as pd
import numpy as np
import joblib
from backend.db.data_fetcher import fetch_ohlcv_from_db


class LSTMPreprocessor:
    def __init__(self, scaler_path: str, n_past: int = 24):
        print(f"[DEBUG] Loading scaler from: {scaler_path}")
        data = joblib.load(scaler_path)

        # Handle both dict (new format) and legacy scaler
        if isinstance(data, dict):
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            print(f"[DEBUG] Loaded scaler with stored feature names ({len(self.feature_names)} features).")
        else:
            self.scaler = data
            self.feature_names = getattr(self.scaler, "feature_names_in_", None)
            print(f"[DEBUG] Loaded legacy scaler with {len(self.feature_names or [])} features.")

        self.n_past = n_past

    # -------------------------------
    # Feature engineering
    # -------------------------------
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("open_time").reset_index(drop=True)
        print(f"[DEBUG] Creating features from df with {len(df)} rows.")

        # Base sanity check
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"[ERROR] Missing base column: {col}")

        # Lag features
        lag_cols = [c for c in self.feature_names if c.startswith("close_lag_")]
        for lag_col in lag_cols:
            lag = int(lag_col.split("_")[-1])
            df[lag_col] = df["close"].shift(lag)

        # MAs
        if "sma_24" in self.feature_names:
            df["sma_24"] = df["close"].rolling(window=24).mean()
        if "ema_24" in self.feature_names:
            df["ema_24"] = df["close"].ewm(span=24, adjust=False).mean()

        df = df.ffill().bfill()
        return df

    # -------------------------------
    # Prepare sequence for model
    # -------------------------------
    def prepare_sequence(self, df: pd.DataFrame) -> np.ndarray:
        print(f"[DEBUG] Preparing sequence for df with {len(df)} rows.")
        df_full = self._create_features(df)

        # Ensure all expected features exist
        missing_cols = [f for f in self.feature_names if f not in df_full.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] Missing expected features: {missing_cols}")

        df_model = df_full[self.feature_names]
        print(f"[DEBUG] df_model shape: {df_model.shape}")

        if len(df_model) < self.n_past:
            raise ValueError(f"[ERROR] Not enough rows to form sequence: {len(df_model)} < {self.n_past}")

        seq = df_model.values[-self.n_past:]
        seq_scaled = self.scaler.transform(seq)
        print(f"[DEBUG] Sequence scaled: shape {seq_scaled.shape}")

        seq_scaled = np.expand_dims(seq_scaled, axis=0)
        print(f"[DEBUG] ✅ LSTM input sequence prepared. Shape: {seq_scaled.shape}")
        return seq_scaled

    # -------------------------------
    # Inverse transform for predictions
    # -------------------------------
    def inverse_transform(self, scaled_values: np.ndarray) -> list:
        """
        Correctly inverse-transform scaled predictions using the original scaler.
        """
        print(f"[DEBUG] Starting inverse_transform for scaled_values shape: {scaled_values.shape}")

        if scaled_values.ndim == 2:
            scaled_values = scaled_values.flatten()

        original_values = []
        for p in scaled_values:
            # Build a dummy row of zeros, insert the prediction into the 'close' column
            dummy_row = np.zeros((1, len(self.feature_names)))
            dummy_row[0, 0] = p  # assuming 'close' is the first column
            try:
                inv = self.scaler.inverse_transform(dummy_row)[0, 0]
                original_values.append(float(inv))
            except Exception as e:
                print(f"[ERROR] Inverse transform failed for value {p}: {e}")
                raise e

        print(f"[DEBUG] ✅ Inverse transformed preds: {original_values}")
        return original_values


# -------------------------------
# Helper functions
# -------------------------------
def fetch_and_preprocess(coin: str, n_hours: int = 2000) -> pd.DataFrame:
    print(f"[DEBUG] Fetching {n_hours}h of data for {coin}")
    df = fetch_ohlcv_from_db(coin, n_hours)
    print(f"[DEBUG] Fetched {len(df)} rows for {coin}")
    return df


def prepare_lstm_input(coin: str, scaler_path: str, n_past: int = 24):
    df = fetch_and_preprocess(coin, n_hours=2000)
    print(f"[DEBUG] ✅ Data fetched successfully. Shape: {df.shape}")
    preprocessor = LSTMPreprocessor(scaler_path=scaler_path, n_past=n_past)
    seq = preprocessor.prepare_sequence(df)
    return seq, df
