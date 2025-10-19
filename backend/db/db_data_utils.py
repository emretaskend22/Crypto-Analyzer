# backend/ml/preprocessor.py
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame, sma_period: int = 20, ema_period: int = 20) -> pd.DataFrame:
    """
    Add common technical indicators to OHLCV DataFrame:
    - SMA, EMA, RSI, MACD, Bollinger Bands
    Uses lowercase underscore column names for consistency.
    Keeps all rows; initial NaNs are kept as None for JSON.
    """
    df = df.copy()

    # Standardize column names first
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "open_time" not in df.columns or "close" not in df.columns:
        raise ValueError("DataFrame must contain 'open_time' and 'close' columns")

    df.set_index("open_time", inplace=True)

    # Moving averages
    df[f"sma_{sma_period}"] = df["close"].rolling(sma_period).mean()
    df[f"ema_{ema_period}"] = df["close"].ewm(span=ema_period, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(sma_period).mean()
    df["bb_std"] = df["close"].rolling(sma_period).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df.drop(columns=["bb_std"], inplace=True)

    df.reset_index(inplace=True)

    # Convert NaN / inf to None for safe JSON output
    df = df.replace([np.inf, -np.inf], None).where(pd.notnull(df), None)

    return df


def filter_new_rows(df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows from df that already exist in existing_df based on 'coin' and 'open_time'
    """
    if df.empty:
        return df
    if existing_df.empty:
        return df
    merged = df.merge(
        existing_df[["coin", "open_time"]],
        on=["coin", "open_time"],
        how="left",
        indicator=True
    )
    return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])


import pandas as pd


CANDLE_HOURS = {
    "1h": 1,
    "12h": 12,
    "1d": 24,
    "1w": 24*7,
    "1m": 24*30
}

def aggregate_candles(df: pd.DataFrame, candle_size: str) -> pd.DataFrame:
    """
    Aggregate hourly OHLCV into larger candle_size using time-based resampling.
    """
    if candle_size == "1h":
        return df  # already hourly

    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df.set_index("open_time", inplace=True)

    # Map candle size to pandas frequency string
    freq_map = {"12h": "12H", "1d": "1D", "1w": "W"}
    freq = freq_map[candle_size]

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }

    df_agg = df.resample(freq).agg(agg_dict).dropna().reset_index()
    return df_agg


