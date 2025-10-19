# backfill.py
import pandas as pd
from sqlalchemy import text
from db_connection import engine
from data_fetcher import fetch_binance_ohlcv
from db_data_utils import add_indicators

COINS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"
BATCH_SIZE = 500  # Binance max per request


def backfill_coin(coin: str, start_time: str = "2023-01-01"):
    print(f"Starting backfill for {coin} from {start_time}...")
    start_dt = pd.to_datetime(start_time)

    while True:
        # Convert to milliseconds for Binance API
        start_ms = int(start_dt.timestamp() * 1000)

        # Fetch OHLCV data from Binance
        df = fetch_binance_ohlcv(symbol=coin, interval=INTERVAL, limit=BATCH_SIZE, startTime=start_ms)
        if df.empty:
            print(f"No more data to fetch for {coin}.")
            break

        # Add indicators
        df = add_indicators(df)

        # Clean numeric issues
        df = df.replace([float('inf'), float('-inf')], None).where(pd.notnull(df), None)

        # Ensure lowercase columns for DB consistency
        df.columns = [c.lower() for c in df.columns]
        df["coin"] = coin

        # Skip duplicates already in DB
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT open_time FROM crypto_ohlcv WHERE coin = :coin"),
                {"coin": coin}
            )
            existing_times = set(row[0] for row in result.fetchall())
        df = df[~df["open_time"].isin(existing_times)]

        if df.empty:
            print(f"No new rows to insert for {coin}. Moving to next batch...")
            start_dt += pd.Timedelta(hours=BATCH_SIZE)
            continue

        # Insert batch into DB
        df.to_sql("crypto_ohlcv", engine, if_exists="append", index=False, method="multi")
        last_fetched = df["open_time"].max()
        print(f"Inserted {len(df)} rows up to {last_fetched} for {coin}...")

        # Prepare for next batch
        if len(df) < BATCH_SIZE:
            print(f"Backfill complete for {coin}.")
            break

        start_dt = pd.to_datetime(last_fetched) + pd.Timedelta(hours=1)


def clear_table():
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE crypto_ohlcv RESTART IDENTITY;"))
    print("✅ Table cleared successfully.")


if __name__ == "__main__":
    # Optionally clear table before backfill
    # clear_table()
    for coin in COINS:
        backfill_coin(coin)
