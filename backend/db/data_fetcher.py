# backend/db/data_fetcher.py
from backend.db.db_connection import engine
from sqlalchemy import text
import requests
import pandas as pd

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_ohlcv(
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        startTime: int = None
) -> pd.DataFrame:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data from Binance.

    Args:
        symbol (str): Trading pair symbol, e.g., "BTCUSDT"
        interval (str): Kline interval, e.g., "1h", "15m"
        limit (int): Number of past candles to fetch (max 1000 per request)
        startTime (int, optional): Fetch candles starting from this timestamp (ms)

    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    if startTime is not None:
        params["startTime"] = startTime

    resp = requests.get(BINANCE_BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()  # Return empty DataFrame if no data

    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])

    df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["coin"] = symbol.upper()

    return df


def fetch_ohlcv_from_db(coin: str = "BTCUSDT", timeframe: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV data from DB for a coin based on timeframe.
    Column names will always be lowercase with underscores.
    """
    hours = timeframe
    query = text("""
        SELECT *
        FROM crypto_ohlcv
        WHERE coin = :coin
        ORDER BY open_time DESC
        LIMIT :hours
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"coin": coin.upper(), "hours": hours})

    df = df.sort_values("open_time").reset_index(drop=True)

    return df


from sqlalchemy import text
import pandas as pd


def fetch_ohlcv_range_from_db(
        coin: str = "BTCUSDT",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01"
) -> pd.DataFrame:
    """
    Fetch OHLCV data from DB for a coin within a specific date range.
    Dates must be in 'YYYY-MM-DD' format (string).

    Args:
        coin (str): Symbol (e.g., "BTCUSDT")
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: OHLCV data with lowercase columns
    """
    query = text("""
                 SELECT *
                 FROM crypto_ohlcv
                 WHERE coin = :coin
                   AND open_time >= :start_date
                   AND open_time <= :end_date
                 ORDER BY open_time ASC
                 """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "coin": coin.upper(),
                "start_date": start_date,
                "end_date": end_date
            }
        )

    df = df.sort_values("open_time").reset_index(drop=True)
    return df


from sqlalchemy import inspect
def get_table_columns(table_name: str):
    """
    Fetch column names of a given table from the DB.

    Args:
        table_name (str): Table name, e.g., 'crypto_ohlcv'

    Returns:
        List[str]: Column names
    """
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return columns


if __name__ == "__main__":
    table_name = "crypto_ohlcv"
    cols = get_table_columns(table_name)
    print(f"Columns in '{table_name}': {cols}")



