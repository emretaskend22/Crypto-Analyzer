import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from backend.db.data_fetcher import fetch_ohlcv_from_db

# -------------------------------
# CONFIG
# -------------------------------
COINS = ["BTCUSDT", "ETHUSDT"]
HORIZONS = [1, 2, 3, 4, 5, 6]  # Predict next 6 hours
N_PAST = 24                     # Lookback window (24 hours)
TIMEFRAME = 5000                # Number of candles to load
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def create_features(df: pd.DataFrame, n_past: int) -> pd.DataFrame:
    """Add lag, SMA, EMA features."""
    for lag in range(1, n_past + 1):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    df["sma_24"] = df["close"].rolling(window=24).mean()
    df["ema_24"] = df["close"].ewm(span=24, adjust=False).mean()
    df.dropna(inplace=True)
    return df


def create_sequences(data: np.ndarray, n_past: int, horizons: list[int]):
    """Generate supervised learning sequences for LSTM."""
    X, Y = [], []
    for i in range(n_past, len(data) - max(horizons)):
        X.append(data[i - n_past:i])
        Y.append([data[i + h - 1, 0] for h in horizons])
    return np.array(X), np.array(Y)


for coin in COINS:
    print(f"\n🚀 Training model for {coin}...")

    # 1️⃣ Load OHLCV data
    df = fetch_ohlcv_from_db(coin, timeframe=TIMEFRAME)
    if df.empty or len(df) < 200:
        print(f"⚠️ Not enough data for {coin}. Skipping...")
        continue

    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("open_time").reset_index(drop=True)

    # 2️⃣ Feature Engineering
    df = create_features(df, N_PAST)

    # 3️⃣ Scaling
    feature_cols = [c for c in df.columns if c != "open_time"]
    features = df[feature_cols]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    scaler_path = os.path.join(MODEL_DIR, f"scaler_{coin}_1h.save")
    joblib.dump({"scaler": scaler, "feature_names": feature_cols}, scaler_path)
    print(f"[✅] Saved scaler for {coin} → {scaler_path}")

    # 4️⃣ Prepare sequences
    X, Y = create_sequences(scaled_data, N_PAST, HORIZONS)

    # 5️⃣ Train/Validation Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    # 6️⃣ Define Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(N_PAST, X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(len(HORIZONS))
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    model_path = os.path.join(MODEL_DIR, f"lstm_{coin}_1_6h.h5")

    # 7️⃣ Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # 8️⃣ Train Model
    print(f"[🧠] Training on {len(X_train)} samples...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # 9️⃣ Save Model
    model.save(model_path)
    print(f"[✅] Saved model for {coin} → {model_path}")

    # 🔟 Quick Test Prediction
    last_seq = scaled_data[-N_PAST:]
    last_seq = np.expand_dims(last_seq, axis=0)
    pred_scaled = model.predict(last_seq)

    # Inverse transform using stored feature schema
    preds = []
    for p in pred_scaled[0]:
        dummy_row = np.zeros((1, scaled_data.shape[1]))
        dummy_row[0, 0] = p
        inv = scaler.inverse_transform(dummy_row)[0, 0]
        preds.append(float(inv))

    print(f"📈 {coin} Predicted next {len(HORIZONS)}h prices: {preds}")
