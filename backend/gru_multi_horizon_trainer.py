# backend/trainers/gru_multi_horizon_trainer.py

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml.preprocessor import preprocess, create_sequences_multi

# ---- load data
df = pd.read_csv("data/BTCUSDT_1h.csv", parse_dates=["Open time"])
df["Open time"] = pd.to_datetime(df["Open time"])
df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
df = preprocess(df)  # your feature engineering (lags, SMA/EMA, etc.)

model_path = "models/gru_multi_1_3_6_12.h5"

# ---- build sequences
HORIZONS = (1, 3, 6, 12)
N_PAST = 24

X, Y, scaler, _ = create_sequences_multi(df, horizons=HORIZONS, n_past=N_PAST)
n_train = int(0.8 * len(X))
X_train, X_val = X[:n_train], X[n_train:]
Y_train, Y_val = Y[:n_train], Y[n_train:]

# ---- model
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(len(HORIZONS))  # multi-output: 4 horizons
])
model.compile(optimizer="adam", loss="mean_squared_error")


# ---- callbacks
cbs = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1)
]

# ---- train
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=40,
    batch_size=64,
    callbacks=cbs,
    verbose=1
)

# final save (just in case)
model.save(model_path)
print(f"[OK] Saved multi-horizon GRU model to {model_path}")
