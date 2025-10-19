import pandas as pd
import pandas_ta as ta

def compute_indicators(df):
    # SMA & EMA
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["EMA_20"] = ta.ema(df["Close"], length=20)

    # RSI
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # MACD
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df["BB_lower"] = bbands["BBL_20_2.0"]
    df["BB_middle"] = bbands["BBM_20_2.0"]
    df["BB_upper"] = bbands["BBU_20_2.0"]

    return df.dropna()



