import pandas as pd


from db.db_data_utils import add_indicators, aggregate_candles
from db.data_fetcher import fetch_ohlcv_from_db, fetch_ohlcv_range_from_db
from ml.backtesting_strategies import sma_crossover_strategy, buy_and_hold_strategy, rsi_strategy, ema_crossover_strategy, macd_strategy, bollinger_bands_strategy
import numpy as np
from ml.prediction_manager import predict_next_hours

CANDLE_HOURS = {
    "1h": 1,
    "12h": 12,
    "1d": 24,
    "1w": 24*7,
    "1m": 24*30
}

LOOKBACK_HOURS = {
    "1h": 200,           # 200 hourly rows
    "12h": 12*200,       # 200 12h candles → 2400 hours
    "1d": 24*90,         # 90 daily candles → 2160 hours
    "1w": 24*7*52,       # 52 weekly candles → 8736 hours

}
def get_coin_analytics(coin_symbol: str = "BTCUSDT", candle_size: str = "1h"):
    """
    Fetch coin analytics data and aggregate hourly candles into candle_size.
    """
    try:
        hours_to_fetch = LOOKBACK_HOURS[candle_size]
        df = fetch_ohlcv_from_db(coin=coin_symbol, timeframe=hours_to_fetch)

        if df.empty:
            return {"error": f"No data available for {coin_symbol}"}
        if candle_size != "1h":
            df = aggregate_candles(df, candle_size)

        # Add indicators if missing
        if not all(col in df.columns for col in ["sma_20", "ema_20", "rsi_14", "macd"]):
            df = add_indicators(df)

        df = df.dropna(subset=["sma_20", "ema_20", "rsi_14", "macd"], how='any')
        return df.to_dict(orient="records")

    except Exception as e:
        print(f"[ERROR] get_coin_analytics: {e}")
        return {"error": str(e)}


def clean_for_json(df):
    """
    Replace NaN and infinite values so that the DataFrame can be safely serialized to JSON.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


STRATEGY_MAP = {
    "Buy & Hold": buy_and_hold_strategy,
    "SMA Crossover (20/50)": sma_crossover_strategy,
    "RSI Strategy": rsi_strategy,
    "EMA Crossover": ema_crossover_strategy,
    "MACD Strategy": macd_strategy,
    "Bollinger Bands": bollinger_bands_strategy
}


def run_backtest(coin: str, start_date: str, end_date: str, strategy: str, initial_balance: float = 10000):
    # 1️⃣ Fetch OHLCV data
    df = fetch_ohlcv_range_from_db(coin, start_date, end_date)
    if df.empty:
        return {"error": f"No data available for {coin} between {start_date} and {end_date}"}

    # 2️⃣ Apply selected strategy
    strategy_func = STRATEGY_MAP.get(strategy)
    if strategy_func is None:
        return {"error": f"Strategy '{strategy}' is not implemented yet."}

    df_result = strategy_func(df, initial_balance=initial_balance)

    # 3️⃣ Calculate returns for Sharpe ratio (if not present)
    if "returns" not in df_result.columns:
        df_result["returns"] = df_result["equity"].pct_change().fillna(0)

    # 4️⃣ KPI calculations
    total_return = (df_result["equity"].iloc[-1] / initial_balance - 1) * 100

    # Win rate: percentage of profitable BUY signals
    buy_signals_idx = df_result.index[df_result["signal"] == "BUY"].tolist()
    win_count = 0
    for idx in buy_signals_idx:
        if idx + 1 < len(df_result) and df_result["close"].iloc[idx + 1] > df_result["close"].iloc[idx]:
            win_count += 1
    win_rate = (win_count / max(len(buy_signals_idx), 1)) * 100

    # Max drawdown
    cumulative_max = df_result["equity"].cummax()
    max_drawdown = ((cumulative_max - df_result["equity"]).max() / cumulative_max.max()) * 100

    # Sharpe ratio (annualized)
    sharpe_ratio = df_result["returns"].mean() / df_result["returns"].std() * np.sqrt(252)

    # 5️⃣ Clean numeric data for JSON
    df_result = clean_for_json(df_result)

    # 6️⃣ Return structured JSON
    return {
        "ohlcv": df_result.to_dict(orient="records"),
        "total_return": total_return,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio
    }



def run_prediction(coin: str):
    result = predict_next_hours(coin)
    if "error" in result:
        return {"error": result["error"]}
    return result

