# backend/trainers/tft_multi_horizon_trainer.py

import os
import numpy as np
import pandas as pd
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization,
    TimeDistributed, MultiHeadAttention, Embedding, Concatenate, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from ml.preprocessor import add_tft_features

# -----------------------------
# Config
# -----------------------------
DATA_CSV = "data/BTCUSDT_1h.csv"
MODEL_PATH = "models/tft_multi_1_3_6_12.keras"
HORIZONS = [1, 3, 6, 12]
N_PAST = 168
BATCH_SIZE = 64
EPOCHS = 50
D_MODEL = 64
D_FF = 128
NUM_HEADS = 4
DROPOUT = 0.2

# categorical vocab sizes
HOUR_VOCAB = 24
DOW_VOCAB = 7
MONTH_VOCAB = 12
WEEKEND_VOCAB = 2
COIN_VOCAB = 4  # placeholder

# embedding dims
EMB_TIME = 8
EMB_COIN = 8

# -----------------------------
# Custom layer
# -----------------------------
class SqueezeLastDim(Layer):
    """Removes last dimension, replaces Lambda layer."""
    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)

# -----------------------------
# Data prep
# -----------------------------
def load_and_engineer() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["Open time"])
    df = df.set_index("Open time").sort_index()
    df = add_tft_features(df)
    return df

def make_future_timeframe(idx: pd.DatetimeIndex, base_pos: int, hours_ahead: List[int]) -> pd.DataFrame:
    base_ts = idx[base_pos]
    future_ts = [base_ts + pd.Timedelta(hours=h) for h in hours_ahead]
    fdf = pd.DataFrame(index=future_ts)
    fdf["hour"] = [ts.hour for ts in future_ts]
    fdf["dayofweek"] = [ts.dayofweek for ts in future_ts]
    fdf["month"] = [ts.month for ts in future_ts]
    fdf["is_weekend"] = [(d >= 5) * 1 for d in fdf["dayofweek"]]
    return fdf

def build_sequences_for_tft(
    df: pd.DataFrame,
    horizons: List[int],
    n_past: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    df = df.copy()
    continuous_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "return", "volatility", "ema_12", "ema_24", "ema_72", "rsi"
    ]
    for c in continuous_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    scaler = MinMaxScaler()
    cont_scaled = scaler.fit_transform(df[continuous_cols].values)

    max_h = max(horizons)
    idx = df.index
    n_total = len(df)
    start = n_past
    end = n_total - max_h

    enc_cont_list = []
    dec_hour_list, dec_dow_list, dec_mon_list, dec_wend_list = [], [], [], []
    y_list = []

    for i in range(start, end):
        enc_cont = cont_scaled[i - n_past:i, :]
        fdf = make_future_timeframe(idx, i, horizons)
        dec_hour = fdf["hour"].astype(int).values
        dec_dow = fdf["dayofweek"].astype(int).values
        dec_mon = fdf["month"].astype(int).values
        dec_wend = fdf["is_weekend"].astype(int).values
        close_idx = continuous_cols.index("Close")
        future_close_scaled = [cont_scaled[i + h, close_idx] for h in horizons]

        enc_cont_list.append(enc_cont)
        dec_hour_list.append(dec_hour)
        dec_dow_list.append(dec_dow)
        dec_mon_list.append(dec_mon)
        dec_wend_list.append(dec_wend)
        y_list.append(future_close_scaled)

    enc_cont = np.array(enc_cont_list, dtype=np.float32)
    dec_hour = np.array(dec_hour_list, dtype=np.int32)
    dec_dow = np.array(dec_dow_list, dtype=np.int32)
    dec_mon = np.array(dec_mon_list, dtype=np.int32)
    dec_wend = np.array(dec_wend_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.float32)

    return enc_cont, dec_hour, dec_dow, dec_mon, dec_wend, y, scaler

# -----------------------------
# Model
# -----------------------------
def tft_like_model(
    n_past: int,
    len_h: int,
    d_cont: int,
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    d_ff: int = D_FF,
    dropout: float = DROPOUT,
) -> Model:
    enc_cont_in = Input(shape=(n_past, d_cont), name="enc_cont")
    dec_hour_in = Input(shape=(len_h,), dtype="int32", name="dec_hour")
    dec_dow_in = Input(shape=(len_h,), dtype="int32", name="dec_dow")
    dec_mon_in = Input(shape=(len_h,), dtype="int32", name="dec_month")
    dec_wend_in = Input(shape=(len_h,), dtype="int32", name="dec_weekend")

    x_enc = LSTM(d_model, return_sequences=True, name="enc_lstm_1")(enc_cont_in)
    x_enc = Dropout(dropout)(x_enc)
    x_enc = LSTM(d_model, return_sequences=True, name="enc_lstm_2")(x_enc)
    x_enc = Dropout(dropout)(x_enc)

    e_hour = Embedding(HOUR_VOCAB, EMB_TIME, name="emb_hour")(dec_hour_in)
    e_dow = Embedding(DOW_VOCAB, EMB_TIME, name="emb_dow")(dec_dow_in)
    e_mon = Embedding(MONTH_VOCAB, EMB_TIME, name="emb_month")(dec_mon_in)
    e_wend = Embedding(WEEKEND_VOCAB, EMB_TIME, name="emb_weekend")(dec_wend_in)

    dec_time = Concatenate(name="dec_time_concat")([e_hour, e_dow, e_mon, e_wend])
    dec_q = TimeDistributed(Dense(d_model), name="dec_time_proj")(dec_time)

    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name="mha")(
        query=dec_q, value=x_enc, key=x_enc
    )
    x = LayerNormalization(epsilon=1e-6)(dec_q + attn_out)

    ffn = TimeDistributed(Dense(d_ff, activation="relu"), name="ffn_1")(x)
    ffn = Dropout(dropout)(ffn)
    ffn = TimeDistributed(Dense(d_model, activation="relu"), name="ffn_2")(ffn)

    x = LayerNormalization(epsilon=1e-6)(x + ffn)

    out = TimeDistributed(Dense(1), name="horizon_out")(x)
    out = SqueezeLastDim(name="squeeze_last_dim")(out)

    model = Model(
        inputs=[enc_cont_in, dec_hour_in, dec_dow_in, dec_mon_in, dec_wend_in],
        outputs=out,
        name="tft_like_multi_horizon",
    )
    model.compile(optimizer="adam", loss="mse")
    return model

# -----------------------------
# Train
# -----------------------------
def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("[INFO] Loading and engineering data...")
    df = load_and_engineer()
    print("[INFO] Building sequences...")
    enc_cont, dec_hour, dec_dow, dec_mon, dec_wend, y, scaler = build_sequences_for_tft(df, HORIZONS, N_PAST)
    print(f"[INFO] Shapes -> enc_cont: {enc_cont.shape}, dec_hour: {dec_hour.shape}, y: {y.shape}")

    n = len(enc_cont)
    n_train = int(0.8 * n)

    X_train = [enc_cont[:n_train], dec_hour[:n_train], dec_dow[:n_train], dec_mon[:n_train], dec_wend[:n_train]]
    y_train = y[:n_train]
    X_val = [enc_cont[n_train:], dec_hour[n_train:], dec_dow[n_train:], dec_mon[n_train:], dec_wend[n_train:]]
    y_val = y[n_train:]

    model = tft_like_model(
        n_past=N_PAST,
        len_h=len(HORIZONS),
        d_cont=enc_cont.shape[-1],
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    )
    model.summary(print_fn=lambda s: print("[MODEL] " + s))

    cbs = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    print("[INFO] Training...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=1)

    model.save(MODEL_PATH)
    print(f"[OK] Saved TFT-like multi-horizon model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
