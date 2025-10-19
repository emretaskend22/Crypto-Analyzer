import pandas as pd

def sma_crossover_strategy(df: pd.DataFrame, short_window: int = 20, long_window: int = 50, initial_balance: float = 10000):
    """
    Simple Moving Average Crossover Strategy:
    Buy when short SMA crosses above long SMA, sell when it crosses below.
    """
    df = df.copy()
    df["SMA_short"] = df["close"].rolling(window=short_window, min_periods=1).mean()
    df["SMA_long"] = df["close"].rolling(window=long_window, min_periods=1).mean()

    # Generate signals
    df["signal"] = 0
    df.loc[df["SMA_short"] > df["SMA_long"], "signal"] = 1   # Buy
    df.loc[df["SMA_short"] < df["SMA_long"], "signal"] = -1  # Sell
    df["signal"] = df["signal"].map({1: "BUY", -1: "SELL", 0: None})

    # Calculate returns
    df["returns"] = df["close"].pct_change().fillna(0)
    position = df["signal"].shift(1).fillna(0).replace({"BUY":1,"SELL":-1,None:0})
    df["equity"] = initial_balance * (1 + position * df["returns"]).cumprod()

    return df


def buy_and_hold_strategy(df: pd.DataFrame, initial_balance: float = 10000) -> pd.DataFrame:
    """
    Buy & Hold: buy at the start and hold until the end.
    """
    df = df.copy()
    df["signal"] = None
    df["returns"] = df["close"].pct_change().fillna(0)

    # Buy on the first candle
    df.at[0, "signal"] = "BUY"

    # Equity curve
    df["equity"] = initial_balance * (1 + df["returns"]).cumprod()

    return df


def rsi_strategy(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70,
                 initial_balance: float = 10000) -> pd.DataFrame:
    """
    RSI Strategy: Buy when RSI < oversold, Sell when RSI > overbought
    """
    df = df.copy()

    # Calculate returns
    df["returns"] = df["close"].pct_change().fillna(0)

    # Calculate RSI if not already present
    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

    # Signals
    df["signal"] = None
    df.loc[df["rsi_14"] < oversold, "signal"] = "BUY"
    df.loc[df["rsi_14"] > overbought, "signal"] = "SELL"

    # Equity calculation
    df["equity"] = initial_balance
    position = 0  # 0 means no position, 1 means holding
    equity = initial_balance
    for i in range(1, len(df)):
        if df.at[i, "signal"] == "BUY" and position == 0:
            position = 1
        elif df.at[i, "signal"] == "SELL" and position == 1:
            position = 0
        # Update equity based on position
        df.at[i, "equity"] = df.at[i - 1, "equity"] * (1 + df.at[i, "returns"] * position)

    return df

def ema_crossover_strategy(df, short_window=12, long_window=26, initial_balance=10000):
    """
    EMA Crossover: Buy when short EMA crosses above long EMA, Sell when short EMA crosses below long EMA.
    """
    df["EMA_short"] = df["close"].ewm(span=short_window, adjust=False).mean()
    df["EMA_long"] = df["close"].ewm(span=long_window, adjust=False).mean()
    df["signal"] = None
    position = 0
    equity = initial_balance
    df["equity"] = equity
    df["returns"] = 0.0

    for i in range(1, len(df)):
        if df["EMA_short"].iloc[i] > df["EMA_long"].iloc[i] and position == 0:
            df.at[i, "signal"] = "BUY"
            position = 1
        elif df["EMA_short"].iloc[i] < df["EMA_long"].iloc[i] and position == 1:
            df.at[i, "signal"] = "SELL"
            position = 0
        # Update equity
        df.at[i, "equity"] = equity if position == 0 else equity * df["close"].iloc[i] / df["close"].iloc[i-1]
        df.at[i, "returns"] = df["equity"].pct_change().iloc[i]

    return df

def macd_strategy(df, short_window=12, long_window=26, signal_window=9, initial_balance=10000):
    """
    MACD Strategy: Buy when MACD crosses above signal line, Sell when it crosses below.
    """
    df["EMA_short"] = df["close"].ewm(span=short_window, adjust=False).mean()
    df["EMA_long"] = df["close"].ewm(span=long_window, adjust=False).mean()
    df["MACD"] = df["EMA_short"] - df["EMA_long"]
    df["MACD_signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

    df["signal"] = None
    position = 0
    equity = initial_balance
    df["equity"] = equity
    df["returns"] = 0.0

    for i in range(1, len(df)):
        if df["MACD"].iloc[i] > df["MACD_signal"].iloc[i] and position == 0:
            df.at[i, "signal"] = "BUY"
            position = 1
        elif df["MACD"].iloc[i] < df["MACD_signal"].iloc[i] and position == 1:
            df.at[i, "signal"] = "SELL"
            position = 0
        # Update equity
        df.at[i, "equity"] = equity if position == 0 else equity * df["close"].iloc[i] / df["close"].iloc[i-1]
        df.at[i, "returns"] = df["equity"].pct_change().iloc[i]

    return df

def bollinger_bands_strategy(df, window=20, num_std=2, initial_balance=10000):
    """
    Bollinger Bands: Buy when price touches lower band, sell when it touches upper band.
    """
    df["SMA"] = df["close"].rolling(window=window).mean()
    df["std"] = df["close"].rolling(window=window).std()
    df["bb_upper"] = df["SMA"] + num_std * df["std"]
    df["bb_lower"] = df["SMA"] - num_std * df["std"]

    df["signal"] = None
    position = 0
    equity = initial_balance
    df["equity"] = equity
    df["returns"] = 0.0

    for i in range(window, len(df)):
        if df["close"].iloc[i] < df["bb_lower"].iloc[i] and position == 0:
            df.at[i, "signal"] = "BUY"
            position = 1
        elif df["close"].iloc[i] > df["bb_upper"].iloc[i] and position == 1:
            df.at[i, "signal"] = "SELL"
            position = 0
        df.at[i, "equity"] = equity if position == 0 else equity * df["close"].iloc[i] / df["close"].iloc[i-1]
        df.at[i, "returns"] = df["equity"].pct_change().iloc[i]

    return df

