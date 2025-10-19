# etl_crypto.py
import pandas as pd
from sqlalchemy import create_engine
from data_fetcher import fetch_binance_ohlcv
from db_data_utils import add_indicators, filter_new_rows
from db_connection import engine



# -----------------------------
# Coins to fetch
# -----------------------------
COINS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"
LIMIT = 500  # last 500 hours

# -----------------------------
# ETL process
# -----------------------------
def etl_crypto():
    for coin in COINS:
        print(f"Fetching data for {coin}...")
        df = fetch_binance_ohlcv(symbol=coin, interval=INTERVAL, limit=LIMIT)
        df = add_indicators(df)

        # Check existing data in DB to avoid duplicates
        try:
            existing_df = pd.read_sql(f"SELECT coin, open_time FROM crypto_ohlcv WHERE coin='{coin}'", engine)
        except Exception:
            existing_df = pd.DataFrame(columns=["coin", "Open time"])

        df_new = filter_new_rows(df, existing_df)
        if df_new.empty:
            print(f"No new rows for {coin}. Skipping insert.")
            continue

        # Insert into PostgreSQL
        df_new.to_sql("crypto_ohlcv", engine, if_exists="append", index=False, method="multi")
        print(f"Inserted {len(df_new)} rows for {coin} into crypto_ohlcv.")

if __name__ == "__main__":
    etl_crypto()
    print("ETL completed successfully!")
